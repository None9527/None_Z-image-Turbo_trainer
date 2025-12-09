# -*- coding: utf-8 -*-
"""
LongCat-Image Adapter - LongCat-Image 模型适配器

支持 LongCat-Image（基于 FLUX 架构）的加速 LoRA 训练。
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging

from ..base import ModelAdapter, ModelConfig
from ..registry import register_adapter

logger = logging.getLogger(__name__)


@register_adapter("longcat")
class LongCatAdapter(ModelAdapter):
    """
    LongCat-Image 模型适配器
    
    特点:
    - 64 通道 latent (16 * 4 packing)
    - 基于 FLUX 的双流 Transformer
    - 3D 位置编码 (modality, h, w)
    - logit-normal 时间步采样
    - dynamic shift
    """
    
    # LongCat-Image 源码路径（可通过环境变量覆盖）
    LONGCAT_SOURCE_PATH = os.environ.get("LONGCAT_SOURCE_PATH", "D:/AI/LongCat-Image")
    
    def __init__(
        self,
        turbo_steps: int = 10,
        use_dynamic_shifting: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
    ):
        """
        Args:
            turbo_steps: 目标加速步数
            use_dynamic_shifting: 是否使用动态 shift
            base_shift: 基础 shift 值
            max_shift: 最大 shift 值
            base_seq_len: 基础序列长度
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.turbo_steps = turbo_steps
        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_seq_len = base_seq_len
        self.max_seq_len = max_seq_len
        
        # 确保 LongCat 源码在路径中
        self._ensure_longcat_in_path()
        
        # 预计算锚点
        self._compute_anchors()
    
    def _ensure_longcat_in_path(self):
        """确保 LongCat-Image 源码在 Python 路径中"""
        longcat_path = Path(self.LONGCAT_SOURCE_PATH)
        if longcat_path.exists() and str(longcat_path) not in sys.path:
            sys.path.insert(0, str(longcat_path))
            logger.debug(f"Added LongCat-Image to path: {longcat_path}")
    
    @property
    def name(self) -> str:
        return "longcat"
    
    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            name="LongCat-Image",
            in_channels=64,  # 16 * 4 (packing)
            vae_scale_factor=8,
            uses_shift_factor=True,
            uses_scaling_factor=True,
            prediction_type="v_prediction",
            max_text_len=512,
            supports_cfg=True,
        )
    
    def _compute_anchors(self):
        """计算加速训练锚点"""
        # 线性分布的 sigma 锚点
        sigmas = torch.linspace(1.0, 0.0, self.turbo_steps + 1)[:-1]
        self.anchor_sigmas = sigmas
        logger.debug(f"LongCat anchors: {self.anchor_sigmas.tolist()}")
    
    def _compute_mu(self, image_seq_len: int) -> float:
        """计算动态 shift 值"""
        m = (self.max_shift - self.base_shift) / (self.max_seq_len - self.base_seq_len)
        b = self.base_shift - m * self.base_seq_len
        mu = image_seq_len * m + b
        return mu
    
    # ==================== 模型加载 ====================
    
    def load_transformer(
        self,
        model_path: str,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> nn.Module:
        """加载 LongCat-Image Transformer"""
        self._ensure_longcat_in_path()
        
        try:
            from longcat_image.models import LongCatImageTransformer2DModel
        except ImportError:
            raise ImportError(
                f"Cannot import LongCatImageTransformer2DModel. "
                f"Ensure LongCat-Image is at: {self.LONGCAT_SOURCE_PATH}"
            )
        
        if dtype is None:
            dtype = torch.bfloat16
        
        logger.info(f"Loading LongCat-Image transformer from {model_path}")
        
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            ignore_mismatched_sizes=False,
        )
        
        # 确保模型转换到正确的精度
        transformer = transformer.to(device=device, dtype=dtype)
        self._transformer = transformer
        
        logger.info(f"LongCat-Image transformer loaded on {device} with dtype {dtype}")
        return transformer
    
    def load_vae(
        self,
        vae_path: str,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        """加载 VAE"""
        from diffusers import AutoencoderKL
        
        if dtype is None:
            dtype = torch.float32
        
        logger.info(f"Loading VAE from {vae_path}")
        
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=dtype,
        )
        
        vae = vae.to(device)
        vae.eval()
        
        return vae
    
    # ==================== LoRA 配置 ====================
    
    def get_lora_target_modules(self) -> List[str]:
        """LongCat-Image LoRA 目标层（基于 FLUX 架构）"""
        return [
            # 双流注意力层
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            # 交叉注意力层
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            # FFN 层
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    
    # ==================== Latent 处理 ====================
    
    def pack_latents(
        self,
        latents: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        打包 latents (FLUX 风格 2x2 packing)
        
        (B, C, H, W) -> (B, H/2 * W/2, C*4)
        """
        batch_size, num_channels, height, width = latents.shape
        
        pack_info = {
            "batch_size": batch_size,
            "num_channels": num_channels,
            "height": height,
            "width": width,
            "vae_scale_factor": 16,  # LongCat 使用 16
        }
        
        # 标准 FLUX packing
        latents = latents.view(
            batch_size, num_channels, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels * 4
        )
        
        return latents, pack_info
    
    def unpack_latents(
        self,
        latents: torch.Tensor,
        pack_info: Dict[str, Any],
    ) -> torch.Tensor:
        """解包 latents"""
        batch_size = pack_info["batch_size"]
        height = pack_info["height"]
        width = pack_info["width"]
        
        batch_size_out, num_patches, channels = latents.shape
        
        # 计算实际尺寸
        h = height
        w = width
        
        latents = latents.view(batch_size_out, h // 2, w // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size_out, channels // 4, h, w)
        
        return latents
    
    # ==================== 时间步处理 ====================
    
    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        image_seq_len: Optional[int] = None,
        use_anchor: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样时间步（logit-normal + dynamic shift）
        
        Args:
            batch_size: 批次大小
            device: 设备
            dtype: 数据类型
            image_seq_len: 图像序列长度（用于动态 shift）
            use_anchor: 是否使用锚点采样
            
        Returns:
            (timesteps, sigmas)
        """
        if use_anchor:
            # 从锚点采样
            anchor_indices = torch.randint(
                0, len(self.anchor_sigmas), (batch_size,), device=device
            )
            sigmas = self.anchor_sigmas.to(device, dtype=dtype)[anchor_indices]
        else:
            # logit-normal 采样
            sigmas = torch.sigmoid(torch.randn(batch_size, device=device, dtype=dtype))
        
        # 应用动态 shift
        if self.use_dynamic_shifting and image_seq_len is not None:
            mu = self._compute_mu(image_seq_len)
            sigmas = mu * sigmas / (1 + (mu - 1) * sigmas)
        
        # LongCat 使用 t/1000 格式
        timesteps = sigmas  # 模型内部会 * 1000
        
        return timesteps, sigmas
    
    # ==================== 位置编码 ====================
    
    def prepare_position_ids(
        self,
        latent_height: int,
        latent_width: int,
        text_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """
        准备 LongCat 的 3D 位置编码
        
        Returns:
            {
                "img_ids": 图像位置 ID (H/2 * W/2, 3)
                "txt_ids": 文本位置 ID (text_len, 3)
            }
        """
        # 文本位置 ID
        txt_ids = torch.zeros(text_len, 3, device=device)
        txt_ids[..., 0] = 0  # modality = 0 for text
        txt_ids[..., 1] = torch.arange(text_len, device=device)
        txt_ids[..., 2] = torch.arange(text_len, device=device)
        
        # 图像位置 ID (packed 后的尺寸)
        img_h = latent_height // 2
        img_w = latent_width // 2
        
        img_ids = torch.zeros(img_h, img_w, 3, device=device)
        img_ids[..., 0] = 1  # modality = 1 for image
        img_ids[..., 1] = (
            torch.arange(img_h, device=device)[:, None] + text_len
        ).expand(img_h, img_w)
        img_ids[..., 2] = (
            torch.arange(img_w, device=device)[None, :] + text_len
        ).expand(img_h, img_w)
        
        img_ids = img_ids.reshape(img_h * img_w, 3)
        
        return {
            "img_ids": img_ids.to(dtype=torch.float64),
            "txt_ids": txt_ids.to(dtype=torch.float64),
        }
    
    # ==================== 模型前向 ====================
    
    def forward(
        self,
        transformer: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: Dict[str, torch.Tensor],
        guidance: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """LongCat-Image 前向传播"""
        img_ids = position_ids.get("img_ids")
        txt_ids = position_ids.get("txt_ids")
        
        output = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,  # 模型内部会 * 1000
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]
        
        return output
    
    # ==================== 工具方法 ====================
    
    def get_gradient_checkpointing_segments(self) -> Optional[List[str]]:
        """LongCat 支持分段梯度检查点"""
        return ["transformer_blocks", "single_transformer_blocks"]

