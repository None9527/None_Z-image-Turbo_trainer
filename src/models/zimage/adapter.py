# -*- coding: utf-8 -*-
"""
Z-Image Adapter - Z-Image (Turbo) 模型适配器

支持 Z-Image Turbo 10 步加速模型的训练。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from ..base import ModelAdapter, ModelConfig
from ..registry import register_adapter

logger = logging.getLogger(__name__)


@register_adapter("zimage")
class ZImageAdapter(ModelAdapter):
    """
    Z-Image (Turbo) 模型适配器
    
    特点:
    - 16 通道 latent
    - 自定义 2x2 packing（支持奇数尺寸）
    - 使用 sigma ∈ [0, 1] 时间步
    - 10 步 Turbo 采样
    """
    
    def __init__(
        self,
        turbo_steps: int = 10,
        shift: float = 3.0,
        jitter_scale: float = 0.02,
    ):
        """
        Args:
            turbo_steps: Turbo 步数（锚点数量）
            shift: 时间步 shift 参数
            jitter_scale: 锚点抖动幅度
        """
        super().__init__()
        self.turbo_steps = turbo_steps
        self.shift = shift
        self.jitter_scale = jitter_scale
        
        # 预计算锚点
        self._compute_anchors()
    
    @property
    def name(self) -> str:
        return "zimage"
    
    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            name="Z-Image Turbo",
            in_channels=16,
            vae_scale_factor=8,
            uses_shift_factor=True,
            uses_scaling_factor=True,
            prediction_type="v_prediction",
            max_text_len=512,
            supports_cfg=True,
        )
    
    def _compute_anchors(self):
        """计算 Turbo 锚点时间步"""
        # 均匀分布的 sigma 锚点
        sigmas = torch.linspace(1.0, 0.0, self.turbo_steps + 1)[:-1]
        
        # 应用 shift
        if self.shift > 0:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        
        self.anchor_sigmas = sigmas
        self.anchor_timesteps = (sigmas * 1000).long()
        
        logger.debug(f"Z-Image anchors: {self.anchor_sigmas.tolist()}")
    
    # ==================== 模型加载 ====================
    
    def load_transformer(
        self,
        model_path: str,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> nn.Module:
        """加载 Z-Image Transformer"""
        try:
            from diffusers import ZImageTransformer2DModel
        except ImportError:
            raise ImportError("diffusers>=0.32.0 is required for ZImageTransformer2DModel")
        
        if dtype is None:
            dtype = torch.bfloat16
        
        logger.info(f"Loading Z-Image transformer from {model_path}")
        
        if model_path.endswith(".safetensors"):
            transformer = ZImageTransformer2DModel.from_single_file(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
            )
        else:
            transformer = ZImageTransformer2DModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
            )
        
        transformer = transformer.to(device)
        self._transformer = transformer
        
        logger.info(f"Z-Image transformer loaded on {device}")
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
        
        if vae_path.endswith(".safetensors"):
            vae = AutoencoderKL.from_single_file(
                vae_path,
                torch_dtype=dtype,
                local_files_only=True,
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                torch_dtype=dtype,
                local_files_only=True,
            )
        
        vae = vae.to(device)
        vae.eval()
        
        return vae
    
    # ==================== LoRA 配置 ====================
    
    def get_lora_target_modules(self) -> List[str]:
        """Z-Image LoRA 目标层"""
        return [
            "to_q",
            "to_k", 
            "to_v",
            "to_out.0",
        ]
    
    # ==================== Latent 处理 ====================
    
    def pack_latents(
        self,
        latents: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        打包 latents (2x2 patch packing，支持奇数尺寸)
        
        (B, C, H, W) -> (B, H/2 * W/2, C*4)
        """
        batch_size, num_channels, height, width = latents.shape
        
        pack_info = {
            "orig_height": height,
            "orig_width": width,
            "batch_size": batch_size,
            "num_channels": num_channels,
        }
        
        # 填充到偶数尺寸
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            latents = torch.nn.functional.pad(
                latents, (0, pad_w, 0, pad_h), mode='replicate'
            )
            height = height + pad_h
            width = width + pad_w
        
        pack_info["padded_height"] = height
        pack_info["padded_width"] = width
        
        # Reshape 和 permute
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        
        return latents, pack_info
    
    def unpack_latents(
        self,
        latents: torch.Tensor,
        pack_info: Dict[str, Any],
    ) -> torch.Tensor:
        """解包 latents"""
        batch_size = pack_info["batch_size"]
        num_channels = pack_info["num_channels"]
        orig_height = pack_info["orig_height"]
        orig_width = pack_info["orig_width"]
        padded_height = pack_info["padded_height"]
        padded_width = pack_info["padded_width"]
        
        channels = latents.shape[-1]
        
        # 反向操作
        latents = latents.view(
            batch_size, padded_height // 2, padded_width // 2, 
            channels // 4, 2, 2
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, padded_height, padded_width)
        
        # 裁剪回原始尺寸
        if padded_height != orig_height or padded_width != orig_width:
            latents = latents[:, :, :orig_height, :orig_width]
        
        return latents
    
    # ==================== 时间步处理 ====================
    
    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        use_anchor: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样 Turbo 锚点时间步
        
        Args:
            batch_size: 批次大小
            device: 设备
            dtype: 数据类型
            use_anchor: 是否使用锚点采样
            
        Returns:
            (timesteps, sigmas)
        """
        if use_anchor:
            # 从锚点均匀采样
            anchor_indices = torch.randint(
                0, len(self.anchor_sigmas), (batch_size,), device=device
            )
            sigmas = self.anchor_sigmas.to(device)[anchor_indices]
            
            # 添加抖动
            if self.jitter_scale > 0:
                jitter = torch.randn_like(sigmas) * self.jitter_scale
                sigmas = (sigmas + jitter).clamp(0.0, 1.0)
        else:
            # 连续采样
            sigmas = torch.rand(batch_size, device=device, dtype=dtype)
            if self.shift > 0:
                sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        
        sigmas = sigmas.to(dtype)
        timesteps = sigmas  # Z-Image 直接使用 sigma 作为 timestep
        
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
        Z-Image 不需要显式的位置 ID（使用内部 RoPE）
        """
        return {}
    
    # ==================== 模型前向 ====================
    
    def forward(
        self,
        transformer: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Z-Image 前向传播"""
        output = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            return_dict=False,
        )[0]
        
        return output

