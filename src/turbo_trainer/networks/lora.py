# -*- coding: utf-8 -*-
"""
Standalone LoRA implementation for Z-Image.

No external dependencies on musubi-tuner.
"""

import logging
import math
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

# Target modules for Z-Image LoRA
ZIMAGE_TARGET_MODULES = [
    "Linear",  # All linear layers
]

# ============================================================
# Z-Image LoRA Target Configuration
# ============================================================
# 推荐方案：只训练 Attention 层（稳健，不易过拟合）
# 进阶方案：加上 feed_forward（风格更强烈）
# ============================================================

ZIMAGE_TARGET_NAMES = [
    # 核心 Attention 层 (必选)
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    #、 Feed Forward 层 (可选，风格不够强时取消注释)
    "feed_forward",
]

# 排除模式 - 这些层绝对不能训练
EXCLUDE_PATTERNS = [
    r".*embedder.*",       # Embedding 层 - 会破坏输入空间
    r".*pad_token.*",      # Padding token
    r".*norm.*",           # 所有归一化层 - 易发散
    r".*adaLN.*",          # AdaLN 调制层 - 归一化相关
    r".*refiner.*",        # Refiner 模块 - 条件预处理
    r".*final_layer.*",    # 输出层 - 会破坏像素分布
]


class LoRAModule(nn.Module):
    """Single LoRA module for a linear layer."""
    
    def __init__(
        self,
        lora_name: str,
        original_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        
        self.lora_name = lora_name
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        
        # Get dimensions
        if isinstance(original_module, nn.Linear):
            in_dim = original_module.in_features
            out_dim = original_module.out_features
        else:
            raise ValueError(f"Unsupported module type: {type(original_module)}")
        
        # Create LoRA layers
        self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
        self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
        # Store original forward (不作为子模块，避免参数被重复计算)
        # 使用 object.__setattr__ 绕过 nn.Module 的属性追踪
        object.__setattr__(self, '_original_forward', original_module.forward)
        
        # Scale factor
        self.scale = alpha / lora_dim
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Original output (使用 _original_forward 避免参数追踪)
        original_output = self._original_forward(x, *args, **kwargs)
        
        # LoRA output
        lora_output = self.lora_up(self.dropout(self.lora_down(x)))
        
        # Combined
        return original_output + lora_output * self.multiplier * self.scale
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        return [self.lora_down.weight, self.lora_up.weight]


class LoRANetwork(nn.Module):
    """LoRA network for Z-Image transformer."""
    
    def __init__(
        self,
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        target_names: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        
        self.target_names = target_names or ZIMAGE_TARGET_NAMES
        self.exclude_patterns = exclude_patterns or EXCLUDE_PATTERNS
        
        # Create LoRA modules
        self.lora_modules: Dict[str, LoRAModule] = {}
        self._create_modules(unet)
        
        # Register as ModuleDict for proper parameter tracking
        self.unet_loras = nn.ModuleDict(self.lora_modules)
        
        logger.info(f"Created {len(self.lora_modules)} LoRA modules")
    
    def _should_target(self, name: str) -> bool:
        """Check if module should be targeted for LoRA."""
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if re.match(pattern, name):
                return False
        
        # Check target names
        for target in self.target_names:
            if target in name:
                return True
        
        return False
    
    def _create_modules(self, model: nn.Module):
        """Create LoRA modules for target layers."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self._should_target(name):
                lora_name = name.replace(".", "_")
                
                lora_module = LoRAModule(
                    lora_name=lora_name,
                    original_module=module,
                    multiplier=self.multiplier,
                    lora_dim=self.lora_dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                
                self.lora_modules[lora_name] = lora_module
                logger.debug(f"Created LoRA for: {name}")
    
    def apply_to(self, model: nn.Module):
        """Apply LoRA modules to model by replacing forward methods."""
        for name, module in model.named_modules():
            lora_name = name.replace(".", "_")
            if lora_name in self.lora_modules:
                lora_module = self.lora_modules[lora_name]
                # Replace forward method
                module.forward = lora_module.forward
    
    def prepare_optimizer_params(
        self,
        unet_lr: float = 1e-4,
        **kwargs,
    ) -> Tuple[List[Dict], List[str]]:
        """Prepare parameters for optimizer."""
        params = []
        for lora_module in self.lora_modules.values():
            params.extend([
                {"params": lora_module.lora_down.weight, "lr": unet_lr},
                {"params": lora_module.lora_up.weight, "lr": unet_lr},
            ])
        
        descriptions = [f"LoRA (dim={self.lora_dim}, alpha={self.alpha})"]
        return params, descriptions
    
    def _convert_name_to_key(self, name: str) -> str:
        """Convert internal module name to proper key format.
        
        Example: layers_0_attention_to_q -> layers.0.attention.to_q
        """
        parts = name.split('_')
        result = []
        i = 0
        while i < len(parts):
            part = parts[i]
            # Numbers stay as-is
            if part.isdigit():
                result.append(part)
            # to_q, to_k, to_v, to_out 保持为整体
            elif part == 'to' and i+1 < len(parts) and parts[i+1] in ['q', 'k', 'v', 'out']:
                result.append(f'to_{parts[i+1]}')
                i += 1
            # Special multi-word tokens
            elif part == 'adaLN' and i+1 < len(parts) and parts[i+1] == 'modulation':
                result.append('adaLN_modulation')
                i += 1
            elif part == 'feed' and i+1 < len(parts) and parts[i+1] == 'forward':
                result.append('feed_forward')
                i += 1
            elif part == 'noise' and i+1 < len(parts) and parts[i+1] == 'refiner':
                result.append('noise_refiner')
                i += 1
            elif part == 'context' and i+1 < len(parts) and parts[i+1] == 'refiner':
                result.append('context_refiner')
                i += 1
            else:
                result.append(part)
            i += 1
        return '.'.join(result)
    
    def get_state_dict(
        self,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, torch.Tensor]:
        """Get LoRA weights as state dict.
        
        Uses ComfyUI-compatible format: diffusion_model.layers.0.xxx
        
        Returns:
            State dict with LoRA weights
        """
        state_dict = {}
        
        for name, lora_module in self.lora_modules.items():
            base_key = self._convert_name_to_key(name)
            key = f"diffusion_model.{base_key}"
            
            state_dict[f"{key}.lora_down.weight"] = lora_module.lora_down.weight.to(dtype)
            state_dict[f"{key}.lora_up.weight"] = lora_module.lora_up.weight.to(dtype)
            state_dict[f"{key}.alpha"] = torch.tensor(float(self.alpha))
        
        return state_dict
    
    def save_weights(
        self,
        path: str,
        dtype: torch.dtype = torch.bfloat16,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Save LoRA weights to safetensors file.
        
        Uses ComfyUI-compatible format: diffusion_model.layers.0.xxx
        """
        state_dict = self.get_state_dict(dtype)
        
        # Add metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "lora_dim": str(self.lora_dim),
            "alpha": str(self.alpha),
            "num_modules": str(len(self.lora_modules)),
        })
        
        save_file(state_dict, path, metadata=metadata)
        logger.info(f"Saved LoRA weights to {path}")
    
    def load_weights(self, path: str):
        """Load LoRA weights from safetensors file.
        
        Supports multiple formats:
        1. diffusion_model.layers.0.xxx (ComfyUI v2 format)
        2. layers.0.xxx (simple format)
        3. lora_unet_layers_0_xxx (legacy Kohya format)
        """
        from safetensors.torch import load_file
        
        state_dict = load_file(path)
        loaded = 0
        
        for name, lora_module in self.lora_modules.items():
            base_key = self._convert_name_to_key(name)
            
            # Try formats in order of preference
            formats = [
                f"diffusion_model.{base_key}",  # ComfyUI v2
                base_key,                        # Simple
                f"lora_unet_{name}",            # Legacy Kohya
            ]
            
            down_key = None
            up_key = None
            for prefix in formats:
                test_down = f"{prefix}.lora_down.weight"
                if test_down in state_dict:
                    down_key = test_down
                    up_key = f"{prefix}.lora_up.weight"
                    break
            
            if down_key and down_key in state_dict:
                lora_module.lora_down.weight.data.copy_(state_dict[down_key])
                loaded += 1
            if up_key and up_key in state_dict:
                lora_module.lora_up.weight.data.copy_(state_dict[up_key])
        
        logger.info(f"Loaded {loaded} LoRA modules from {path}")


def create_network(
    multiplier: float = 1.0,
    network_dim: int = 4,
    network_alpha: float = 1.0,
    vae: Optional[nn.Module] = None,
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> LoRANetwork:
    """
    Factory function to create LoRA network.
    
    Args:
        multiplier: LoRA output multiplier
        network_dim: LoRA rank
        network_alpha: LoRA alpha for scaling
        vae: VAE model (not used)
        text_encoders: Text encoders (not used)
        unet: Target model for LoRA
        neuron_dropout: Dropout rate
        
    Returns:
        LoRANetwork instance
    """
    if unet is None:
        raise ValueError("unet (Z-Image transformer) is required")
    
    return LoRANetwork(
        unet=unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        **kwargs,
    )
