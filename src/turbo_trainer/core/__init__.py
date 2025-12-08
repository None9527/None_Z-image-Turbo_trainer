"""
核心训练模块 - 模型无关的训练逻辑

包含:
- trainer: 基础训练器
- flow_matching: Rectified Flow 核心算法
- losses: 通用损失函数
"""

from .flow_matching import (
    sample_timesteps,
    get_noisy_model_input_and_timesteps,
    get_z_t,
    flow_matching_loss,
    compute_loss_weighting,
)

from .trainer import ACRFTrainer

__all__ = [
    "sample_timesteps",
    "get_noisy_model_input_and_timesteps", 
    "get_z_t",
    "flow_matching_loss",
    "compute_loss_weighting",
    "ACRFTrainer",
]

