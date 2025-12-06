# -*- coding: utf-8 -*-
"""
Loss Functions for Z-Image Trainer

可用损失函数：
- FrequencyAwareLoss: 频域分离的混合损失（高频 L1 + 低频 Cosine）
- AdaptiveFrequencyLoss: 自适应权重的频域损失
"""

from .frequency_aware_loss import FrequencyAwareLoss, AdaptiveFrequencyLoss

__all__ = [
    "FrequencyAwareLoss",
    "AdaptiveFrequencyLoss",
]

