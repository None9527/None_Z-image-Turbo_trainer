# -*- coding: utf-8 -*-
"""
Base Model Adapter - 模型适配器抽象基类

定义所有模型适配器必须实现的接口，使训练脚本可以复用核心逻辑。

架构说明:
- 每个模型（Z-Image、LongCat、FLUX 等）在 models/ 下有独立子目录
- 子目录包含 adapter.py（适配器）、pipeline.py（推理）、utils.py（工具）
- 训练脚本通过 registry 获取适配器，实现训练逻辑复用
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置数据类"""
    name: str                          # 模型名称
    in_channels: int                   # latent 通道数
    vae_scale_factor: int              # VAE 压缩因子
    uses_shift_factor: bool = True     # 是否使用 shift factor
    uses_scaling_factor: bool = True   # 是否使用 scaling factor
    prediction_type: str = "v_prediction"  # 预测类型
    max_text_len: int = 512            # 最大文本长度
    supports_cfg: bool = True          # 是否支持 CFG


@dataclass
class LatentInfo:
    """Latent 信息数据类"""
    latents: torch.Tensor              # latent tensor
    height: int                        # 原始 latent 高度
    width: int                         # 原始 latent 宽度
    packed: bool = False               # 是否已打包


class ModelAdapter(ABC):
    """
    模型适配器抽象基类
    
    每个模型（Z-Image、LongCat、FLUX 等）需要实现此接口。
    训练脚本通过此接口与模型交互，实现训练逻辑复用。
    
    子类必须实现:
    - name: 适配器名称
    - config: 模型配置
    - load_transformer(): 加载 Transformer
    - load_vae(): 加载 VAE
    - get_lora_target_modules(): LoRA 目标模块
    - pack_latents() / unpack_latents(): Latent 打包/解包
    - sample_timesteps(): 时间步采样
    - prepare_position_ids(): 位置编码
    - forward(): 模型前向
    """
    
    def __init__(self):
        self._transformer = None
        self._config: Optional[ModelConfig] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """适配器名称"""
        pass
    
    @property
    @abstractmethod
    def config(self) -> ModelConfig:
        """模型配置"""
        pass
    
    # ==================== 模型加载 ====================
    
    @abstractmethod
    def load_transformer(
        self,
        model_path: str,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> nn.Module:
        """
        加载 Transformer 模型
        
        Args:
            model_path: 模型路径（目录或 safetensors 文件）
            device: 目标设备
            dtype: 数据类型
            **kwargs: 模型特定参数
            
        Returns:
            加载的 Transformer 模型
        """
        pass
    
    @abstractmethod
    def load_vae(
        self,
        vae_path: str,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        """
        加载 VAE 模型
        
        Args:
            vae_path: VAE 路径
            device: 目标设备
            dtype: 数据类型
            
        Returns:
            加载的 VAE 模型
        """
        pass
    
    # ==================== LoRA 配置 ====================
    
    @abstractmethod
    def get_lora_target_modules(self) -> List[str]:
        """
        获取 LoRA 目标模块列表
        
        不同模型的注意力层命名不同，需要各自实现。
        
        Returns:
            LoRA 目标模块名称列表
        """
        pass
    
    def get_lora_config(self, rank: int = 8, alpha: float = 4.0) -> Dict[str, Any]:
        """
        获取 LoRA 配置
        
        Args:
            rank: LoRA rank
            alpha: LoRA alpha
            
        Returns:
            LoRA 配置字典（可用于 peft.LoraConfig）
        """
        return {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": self.get_lora_target_modules(),
            "lora_dropout": 0.0,
            "bias": "none",
        }
    
    # ==================== Latent 处理 ====================
    
    @abstractmethod
    def pack_latents(
        self,
        latents: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        打包 latents 为模型输入格式
        
        不同模型有不同的 packing 方式（2x2 patch、直接使用等）。
        
        Args:
            latents: 原始 latents (B, C, H, W)
            **kwargs: 模型特定参数
            
        Returns:
            Tuple of:
            - packed_latents: 打包后的 latents
            - pack_info: 用于 unpack 的信息字典
        """
        pass
    
    @abstractmethod
    def unpack_latents(
        self,
        latents: torch.Tensor,
        pack_info: Dict[str, Any],
    ) -> torch.Tensor:
        """
        解包 latents 回原始格式
        
        Args:
            latents: 打包的 latents
            pack_info: pack_latents 返回的信息字典
            
        Returns:
            解包后的 latents (B, C, H, W)
        """
        pass
    
    def scale_latents(self, latents: torch.Tensor, vae) -> torch.Tensor:
        """
        缩放 latents（应用 VAE 的 scaling/shift factor）
        
        Args:
            latents: 原始 VAE 输出
            vae: VAE 模型
            
        Returns:
            缩放后的 latents
        """
        if self.config.uses_shift_factor and hasattr(vae.config, 'shift_factor'):
            latents = latents - vae.config.shift_factor
        if self.config.uses_scaling_factor and hasattr(vae.config, 'scaling_factor'):
            latents = latents * vae.config.scaling_factor
        return latents
    
    def unscale_latents(self, latents: torch.Tensor, vae) -> torch.Tensor:
        """
        反缩放 latents（用于解码前）
        
        Args:
            latents: 缩放后的 latents
            vae: VAE 模型
            
        Returns:
            原始格式的 latents
        """
        if self.config.uses_scaling_factor and hasattr(vae.config, 'scaling_factor'):
            latents = latents / vae.config.scaling_factor
        if self.config.uses_shift_factor and hasattr(vae.config, 'shift_factor'):
            latents = latents + vae.config.shift_factor
        return latents
    
    # ==================== 时间步处理 ====================
    
    @abstractmethod
    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样时间步
        
        不同模型使用不同的时间步分布（均匀、logit-normal 等）。
        
        Args:
            batch_size: 批次大小
            device: 设备
            dtype: 数据类型
            **kwargs: 模型特定参数（如 turbo_steps、shift 等）
            
        Returns:
            Tuple of:
            - timesteps: 时间步 tensor (用于模型输入)
            - sigmas: sigma 值 (用于噪声混合)
        """
        pass
    
    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """
        添加噪声（线性插值，Rectified Flow 方式）
        
        x_t = (1 - sigma) * x_0 + sigma * noise
        
        Args:
            latents: 干净的 latents (x_0)
            noise: 噪声 (x_1)
            sigmas: sigma 值
            
        Returns:
            加噪后的 latents (x_t)
        """
        # 确保 sigmas 形状正确
        while sigmas.dim() < latents.dim():
            sigmas = sigmas.unsqueeze(-1)
        
        return (1 - sigmas) * latents + sigmas * noise
    
    def compute_velocity_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算速度预测目标
        
        v = noise - latents (Rectified Flow)
        
        Args:
            latents: 干净的 latents (x_0)
            noise: 噪声 (x_1)
            
        Returns:
            目标速度
        """
        return noise - latents
    
    # ==================== 位置编码 ====================
    
    @abstractmethod
    def prepare_position_ids(
        self,
        latent_height: int,
        latent_width: int,
        text_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """
        准备位置编码
        
        不同模型使用不同的位置编码方案（RoPE、绝对位置等）。
        
        Args:
            latent_height: latent 高度
            latent_width: latent 宽度
            text_len: 文本长度
            device: 设备
            dtype: 数据类型
            
        Returns:
            位置编码字典（键名因模型而异）
        """
        pass
    
    # ==================== 模型前向 ====================
    
    @abstractmethod
    def forward(
        self,
        transformer: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        执行模型前向传播
        
        封装不同模型的前向接口差异。
        
        Args:
            transformer: Transformer 模型
            hidden_states: 输入 hidden states（打包后的 latents）
            encoder_hidden_states: 文本编码
            timesteps: 时间步
            position_ids: 位置编码
            **kwargs: 模型特定参数
            
        Returns:
            模型预测输出
        """
        pass
    
    # ==================== 工具方法 ====================
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        计算基础 MSE 损失
        
        Args:
            pred: 模型预测
            target: 目标
            reduction: 归约方式
            
        Returns:
            损失值
        """
        loss = (pred.float() - target.float()) ** 2
        
        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss
        elif reduction == "batch":
            return loss.reshape(loss.shape[0], -1).mean(dim=1)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def get_required_components(self) -> List[str]:
        """
        获取模型必需的组件列表
        
        Returns:
            组件名称列表
        """
        return ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
    
    def validate_model_path(
        self,
        model_path,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        验证模型路径完整性(默认实现)
        
        子类可覆盖以实现自定义检测。
        
        Args:
            model_path: 模型根目录(Path或str)
            strict: 是否严格检查
            
        Returns:
            检测结果字典
        """
        from pathlib import Path
        model_path = Path(model_path)
        
        result = {
            "valid": True,
            "components": {},
            "missing": []
        }
        
        for component in self.get_required_components():
            comp_path = model_path / component
            comp_info = {
                "exists": comp_path.exists(),
                "has_config": (comp_path / "config.json").exists() if comp_path.exists() else False
            }
            
            result["components"][component] = comp_info
            
            if not comp_info["exists"]:
                result["missing"].append(component)
                result["valid"] = False
        
        return result
    
    def get_gradient_checkpointing_segments(self) -> Optional[List[str]]:
        """
        获取梯度检查点分段
        
        Returns:
            分段名称列表，None 表示使用默认
        """
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
