# -*- coding: utf-8 -*-
"""
Loss Functions for Z-Image Trainer

可用损失函数：
- StandardLoss: 标准训练损失 (Charbonnier + Cosine) ★推荐★
- MSELoss: 标准 L2/MSE 损失
- CharbonnierLoss: 平滑 L1 损失 (Robust L1)
- L2CosineLoss: L2 + Cosine 组合损失
- L1CosineLoss: Charbonnier + Cosine 组合损失
- FrequencyAwareLoss: 频域分离的混合损失（高频 L1 + 低频 Cosine）
- AdaptiveFrequencyLoss: 自适应权重的频域损失
- StyleStructureLoss: 结构锁风格迁移损失（SSIM + Lab 空间统计量）
- LatentStyleStructureLoss: Latent 空间的风格结构损失（省显存）
"""

from .standard_loss import StandardLoss, compute_standard_loss
from .mse_loss import (
    MSELoss,
    CharbonnierLoss,
    L2CosineLoss,
    L1CosineLoss,
    compute_mse_loss,
    compute_charbonnier_loss,
    compute_cosine_loss,
)
from .frequency_aware_loss import FrequencyAwareLoss, AdaptiveFrequencyLoss
from .style_structure_loss import StyleStructureLoss, LatentStyleStructureLoss
from .dpo_loss import DPOLoss, DPOLossWithSNR

__all__ = [
    # 标准损失 (推荐)
    "StandardLoss",
    "compute_standard_loss",
    # 基础损失
    "MSELoss",
    "CharbonnierLoss",
    "L2CosineLoss",
    "L1CosineLoss",
    "compute_mse_loss",
    "compute_charbonnier_loss",
    "compute_cosine_loss",
    # 频域损失
    "FrequencyAwareLoss",
    "AdaptiveFrequencyLoss",
    # 风格损失
    "StyleStructureLoss",
    "LatentStyleStructureLoss",
    # DPO 损失
    "DPOLoss",
    "DPOLossWithSNR",
]

