# -*- coding: utf-8 -*-
"""
Training utilities for Z-Image LoRA training.

Similar to musubi-tuner's hv_train_network.py, provides:
- Timestep sampling strategies (uniform, sigmoid, shift)
- Noisy model input generation
- Flow matching utilities
- Loss weighting schemes
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Timestep Sampling
# ============================================================================

def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute density for non-uniform timestep sampling.
    
    From SD3 paper: https://arxiv.org/abs/2403.03206v1
    """
    if device is None:
        device = torch.device("cpu")
    
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device)
        u = torch.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(batch_size, device=device)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        # uniform
        u = torch.rand(batch_size, device=device)
    
    return u


def sample_timesteps(
    batch_size: int,
    sampling_method: str = "shift",
    shift: float = 2.2,
    sigmoid_scale: float = 1.0,
    min_timestep: float = 0.0,
    max_timestep: float = 1000.0,
    device: torch.device = None,
    latent_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Sample timesteps for training.
    
    Args:
        batch_size: Number of samples
        sampling_method: One of "uniform", "sigmoid", "shift", "flux_shift", "logsnr"
        shift: Shift factor for shift-based sampling
        sigmoid_scale: Scale for sigmoid-based sampling
        min_timestep: Minimum timestep (0-1000)
        max_timestep: Maximum timestep (0-1000)
        device: Target device
        latent_shape: Latent shape for dynamic shift calculation
        
    Returns:
        Timesteps tensor in [0, 1] range
    """
    if device is None:
        device = torch.device("cpu")
    
    t_min = min_timestep / 1000.0
    t_max = max_timestep / 1000.0
    
    if sampling_method == "uniform":
        # Simple uniform sampling
        t = torch.rand(batch_size, device=device)
        
    elif sampling_method == "sigmoid":
        # Sigmoid of random normal
        t = torch.sigmoid(sigmoid_scale * torch.randn(batch_size, device=device))
        
    elif sampling_method == "shift":
        # Shifted sigmoid (recommended for Z-Image)
        logits_norm = torch.randn(batch_size, device=device)
        logits_norm = logits_norm * sigmoid_scale
        t = logits_norm.sigmoid()
        t = (t * shift) / (1 + (shift - 1) * t)
        
    elif sampling_method == "flux_shift":
        # Dynamic shift based on image size (FLUX style)
        if latent_shape is not None:
            h, w = latent_shape[-2:]
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            shift = math.exp(mu)
        logits_norm = torch.randn(batch_size, device=device)
        logits_norm = logits_norm * sigmoid_scale
        t = logits_norm.sigmoid()
        t = (t * shift) / (1 + (shift - 1) * t)
        
    elif sampling_method == "logsnr":
        # Log-SNR sampling (https://arxiv.org/abs/2411.14793v3)
        logsnr = torch.randn(batch_size, device=device) * 1.0  # std=1.0
        t = torch.sigmoid(-logsnr / 2)
        
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Apply min/max constraints
    t = t * (t_max - t_min) + t_min
    t = t.clamp(t_min, t_max)
    
    return t


def get_lin_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15,
):
    """Create linear interpolation function."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


# ============================================================================
# Noisy Model Input Generation
# ============================================================================

def get_noisy_model_input_and_timesteps(
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create noisy model input using flow matching interpolation.
    
    Z-Image Flow Matching:
        z_t = (1 - t) * z_0 + t * noise
        
    Args:
        latents: Clean latents z_0 (B, C, H, W) or (B, C, F, H, W)
        noise: Random noise (same shape as latents)
        timesteps: Timesteps in [0, 1] range (B,)
        
    Returns:
        noisy_model_input: z_t
        timesteps_normalized: Z-Image format (1000-t)/1000
    """
    # Expand timesteps for broadcasting
    if latents.dim() == 4:
        t = timesteps.view(-1, 1, 1, 1)
    else:
        t = timesteps.view(-1, 1, 1, 1, 1)
    
    # Flow matching interpolation: z_t = (1-t)*z_0 + t*noise
    noisy_model_input = (1.0 - t) * latents + t * noise
    
    # Convert to Z-Image timestep format: (1000-t)/1000
    timesteps_normalized = (1.0 - timesteps)
    
    return noisy_model_input, timesteps_normalized


def get_z_t(
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Create noisy latent z_t.
    
    Args:
        latents: Clean latents z_0
        noise: Random noise
        timesteps: Timesteps in [0, 1] range
        
    Returns:
        z_t: Noisy latent
    """
    if latents.dim() == 4:
        t = timesteps.view(-1, 1, 1, 1)
    else:
        t = timesteps.view(-1, 1, 1, 1, 1)
    
    return (1.0 - t) * latents + t * noise


# ============================================================================
# Loss Weighting
# ============================================================================

def compute_loss_weighting(
    weighting_scheme: str,
    timesteps: torch.Tensor,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Compute loss weighting based on timesteps.
    
    Args:
        weighting_scheme: One of "none", "sigma_sqrt", "cosmap"
        timesteps: Timesteps in [0, 1] range
        device: Target device
        dtype: Target dtype
        
    Returns:
        Loss weights (B,)
    """
    if device is None:
        device = timesteps.device
    if dtype is None:
        dtype = timesteps.dtype
    
    if weighting_scheme == "none":
        return torch.ones_like(timesteps)
    
    elif weighting_scheme == "sigma_sqrt":
        # sqrt(sigma) weighting
        return torch.sqrt(timesteps)
    
    elif weighting_scheme == "cosmap":
        # Cosine mapping
        return 1.0 - torch.cos(timesteps * math.pi / 2)
    
    else:
        return torch.ones_like(timesteps)


# ============================================================================
# Flow Matching Utilities
# ============================================================================

def flow_matching_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute flow matching loss.
    
    Standard flow matching uses:
        target = velocity = noise - z_0
        
    Z-Image uses:
        target = -z_t (with Norm_opt scaling)
    """
    return F.mse_loss(model_pred, target, reduction=reduction)


def compute_norm_opt_scale(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    min_scale: float = 5.0,
    max_scale: float = 100.0,
) -> float:
    """
    Compute optimal scaling factor k using Norm_opt.
    
    k = E[model_pred²] / E[model_pred · target]
    
    This minimizes the normalized loss:
        L = MSE(model_pred, target × k) / k²
    """
    with torch.no_grad():
        mp_flat = model_pred.flatten().float()
        tgt_flat = target.flatten().float()
        
        mp_sq = (mp_flat ** 2).mean()
        mp_tgt = (mp_flat * tgt_flat).mean()
        
        if mp_tgt.abs() > 1e-8:
            k = (mp_sq / mp_tgt).item()
        else:
            k = 50.0
        
        k = max(min_scale, min(max_scale, k))
    
    return k


def compute_target_with_schedule(
    latents: torch.Tensor,
    z_t: torch.Tensor,
    timesteps: torch.Tensor,
    schedule_mode: str = "content",
) -> Tuple[torch.Tensor, float]:
    """
    Compute target using time-dependent schedule.
    
    Args:
        latents: Clean latents z_0
        z_t: Noisy latents
        timesteps: Timesteps in Z-Image format (1000-t)/1000
        schedule_mode: "content" or "quality"
        
    Returns:
        target: Training target
        alpha_t: Schedule factor
    """
    if schedule_mode == "quality":
        # Quality mode: pure proxy target
        target = -z_t
        alpha_t = 1.0
    else:
        # Content mode: time-dependent blending
        alpha_t = timesteps.mean().item()
        T_content = -latents  # -z_0
        T_proxy = -z_t        # -z_t
        target = alpha_t * T_content + (1.0 - alpha_t) * T_proxy
    
    return target, alpha_t


# ============================================================================
# Gradient Utilities
# ============================================================================

def clip_gradients(
    parameters,
    max_norm: float = 1.0,
    clip_type: str = "norm",
) -> float:
    """
    Clip gradients.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum gradient norm/value
        clip_type: "norm" or "value"
        
    Returns:
        Total gradient norm before clipping
    """
    if max_norm <= 0:
        return 0.0
    
    if clip_type == "norm":
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm).item()
    else:
        torch.nn.utils.clip_grad_value_(parameters, max_norm)
        return 0.0


# ============================================================================
# Memory Utilities
# ============================================================================

def clean_memory(device: torch.device = None):
    """Clean up GPU memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)


def get_memory_usage() -> dict:
    """Get current memory usage."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
    }


# ============================================================================
# Logging Utilities
# ============================================================================

def log_training_info(
    epoch: int,
    step: int,
    loss_dict: dict,
    lr: float,
    alpha_t: float = None,
):
    """Log training information."""
    loss_str = ", ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
    alpha_str = f", α(t)={alpha_t:.2f}" if alpha_t is not None else ""
    logger.info(f"[Epoch {epoch}][Step {step}] {loss_str}{alpha_str}, lr={lr:.2e}")

