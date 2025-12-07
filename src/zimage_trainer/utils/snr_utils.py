# -*- coding: utf-8 -*-
"""
Min-SNR 加权工具

针对 Turbo/LCM 类 10 步加速模型的定制 Min-SNR 策略：
- 标准 Min-SNR 在高噪区权重过低，导致模型无法学习构图
- Floored Min-SNR 增加保底权重，确保每一步都参与训练

公式：
    标准: weight = min(SNR, gamma) / SNR
    改进: weight = max(min(SNR, gamma) / SNR, floor)
"""

import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_snr(
    timesteps: torch.Tensor,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """
    计算 SNR (Signal-to-Noise Ratio)
    
    对于 Rectified Flow / Z-Image:
        sigma = timestep / 1000
        SNR = ((1 - sigma) / sigma)^2
    
    Args:
        timesteps: 时间步 (B,)
        num_train_timesteps: 总训练时间步数
        
    Returns:
        snr: SNR 值 (B,)
    """
    sigmas = timesteps.float() / num_train_timesteps
    # 避免除零
    sigmas = sigmas.clamp(min=0.001, max=0.999)
    snr = ((1 - sigmas) / sigmas) ** 2
    return snr


def compute_snr_weights(
    timesteps: torch.Tensor,
    num_train_timesteps: int = 1000,
    snr_gamma: float = 5.0,
    snr_floor: float = 0.1,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """
    计算 Floored Min-SNR 权重
    
    针对 10 步 Turbo 模型的定制策略：
    - gamma: 截断 SNR 上限，防止低噪区过拟合细节
    - floor: 保底权重，确保高噪区（构图阶段）参与训练
    
    Args:
        timesteps: 时间步 (B,)
        num_train_timesteps: 总训练时间步数
        snr_gamma: SNR 截断值 (推荐 5.0)
        snr_floor: 保底权重 (推荐 0.1，10步模型关键参数)
        prediction_type: 预测类型 ("v_prediction" 或 "epsilon")
        
    Returns:
        weights: 加权系数 (B, 1, 1, 1) 可直接与 loss 相乘
    """
    if snr_gamma <= 0:
        # 禁用 SNR 加权
        return torch.ones(timesteps.shape[0], 1, 1, 1, device=timesteps.device, dtype=torch.float32)
    
    snr = compute_snr(timesteps, num_train_timesteps)
    
    # Min-SNR-γ: 截断 SNR
    clipped_snr = torch.clamp(snr, max=snr_gamma)
    
    if prediction_type == "v_prediction":
        # v-prediction: weight = min(SNR, γ) / (SNR + 1)
        # 这个公式来自 Min-SNR 论文对 v-prediction 的推导
        weights = clipped_snr / (snr + 1)
    else:
        # epsilon-prediction: weight = min(SNR, γ) / SNR
        weights = clipped_snr / snr.clamp(min=0.001)
    
    # Floored Min-SNR: 增加保底权重（关键改进！）
    # 这确保高噪区（锚点 0-2，构图阶段）仍然参与训练
    if snr_floor > 0:
        weights = torch.maximum(weights, torch.tensor(snr_floor, device=weights.device))
    
    # 扩展维度以匹配 (B, C, H, W) 的 loss
    weights = weights.view(-1, 1, 1, 1)
    
    return weights


def print_anchor_snr_weights(
    turbo_steps: int = 10,
    shift: float = 3.0,
    snr_gamma: float = 5.0,
    snr_floor: float = 0.1,
):
    """
    打印锚点的 SNR 和权重分布（用于调试和理解）
    
    Args:
        turbo_steps: Turbo 步数
        shift: Z-Image shift 参数
        snr_gamma: SNR gamma
        snr_floor: 保底权重
    """
    from diffusers import FlowMatchEulerDiscreteScheduler
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=shift,
    )
    scheduler.set_timesteps(num_inference_steps=turbo_steps, device="cpu")
    
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas[:-1]  # 排除最后的 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"锚点 SNR 权重分布 (steps={turbo_steps}, shift={shift})")
    logger.info(f"gamma={snr_gamma}, floor={snr_floor}")
    logger.info(f"{'='*60}")
    logger.info(f"{'锚点':^6} | {'Timestep':^10} | {'Sigma':^8} | {'SNR':^10} | {'标准权重':^10} | {'改进权重':^10}")
    logger.info("-" * 70)
    
    for i, (t, s) in enumerate(zip(timesteps, sigmas)):
        snr = ((1 - s) / s) ** 2
        
        # 标准 Min-SNR
        std_weight = min(snr.item(), snr_gamma) / (snr.item() + 1)
        
        # Floored Min-SNR
        floored_weight = max(std_weight, snr_floor)
        
        logger.info(f"{i:^6} | {t.item():^10.1f} | {s.item():^8.4f} | {snr.item():^10.4f} | {std_weight:^10.4f} | {floored_weight:^10.4f}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_anchor_snr_weights(turbo_steps=10, shift=3.0, snr_gamma=5.0, snr_floor=0.1)

