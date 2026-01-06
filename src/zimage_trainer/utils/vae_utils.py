# -*- coding: utf-8 -*-
"""
VAE utilities for Z-Image.

Mirrors the functionality from musubi-tuner/zimage/zimage_utils.py.
"""

import logging
from typing import Optional, Union

# 延迟导入 torch（避免多卡模式下的 CUDA 初始化冲突）
# 实际导入在各函数内部进行

logger = logging.getLogger(__name__)

# VAE configuration constants
VAE_SCALE_FACTOR = 8
SCALING_FACTOR = 0.3611
SHIFT_FACTOR = 0.1159


def load_vae(
    vae_path: str,
    device = "cuda",
    dtype = None,
):
    """
    Load Z-Image VAE.
    
    Supports both directory (HuggingFace format) and single file (.safetensors).
    
    Args:
        vae_path: Path to VAE checkpoint or directory
        device: Target device
        dtype: Model dtype
        
    Returns:
        Loaded VAE model
    """
    import torch
    from diffusers import AutoencoderKL
    import os
    
    if dtype is None:
        dtype = torch.bfloat16
    
    logger.info(f"Loading VAE from {vae_path}")
    
    if os.path.isdir(vae_path):
        # Load from directory (HuggingFace format with config.json)
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
    elif vae_path.endswith(".safetensors"):
        # Load from single file
        vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported VAE path format: {vae_path}")
    
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    logger.info(f"VAE loaded: scaling={getattr(vae.config, 'scaling_factor', 'N/A')}, "
                f"shift={getattr(vae.config, 'shift_factor', 'N/A')}")
    return vae


def decode_latents_to_pixels(
    vae,
    latents: torch.Tensor,
    is_scaled_latents: bool = True,
) -> torch.Tensor:
    """
    Decode latents to pixel space using VAE.
    
    Handles proper scaling and shifting for Z-Image VAE,
    identical to musubi-tuner implementation.
    
    Args:
        vae: AutoencoderKL model from diffusers
        latents: Latent tensor (B, C, H, W) or (B, C, F, H, W) or (B, F, C, H, W)
        is_scaled_latents: If True, latents are in Pipeline format (already scaled),
                          need to apply inverse transform before decode.
                          If False, latents are in Prototype format (raw VAE output),
                          no transform needed.
        
    Returns:
        Decoded pixels in [0, 1] range
        
    Example:
        >>> latents = torch.randn(1, 16, 128, 128)
        >>> pixels = decode_latents_to_pixels(vae, latents, is_scaled_latents=True)
        >>> pixels.shape
        torch.Size([1, 3, 1024, 1024])
    """
    import torch
    
    # Handle 5D latents by squeezing to 4D
    if latents.dim() == 5:
        # Could be (B, C, F, H, W) or (B, F, C, H, W)
        if latents.shape[1] == 1:
            # (B, F=1, C, H, W) -> (B, C, H, W)
            latents = latents.squeeze(1)
        elif latents.shape[2] == 1:
            # (B, C, F=1, H, W) -> (B, C, H, W)
            latents = latents.squeeze(2)
        else:
            # Take first frame
            latents = latents[:, :, 0, :, :]
    
    # Move to VAE dtype
    latents = latents.to(vae.dtype)
    
    # Apply inverse scaling only for Pipeline (scaled) latents
    # Pipeline latents: z' = (z - shift) * scale
    # To decode: z = z' / scale + shift
    if is_scaled_latents:
        if hasattr(vae.config, 'scaling_factor'):
            latents = latents / vae.config.scaling_factor
        if hasattr(vae.config, 'shift_factor'):
            latents = latents + vae.config.shift_factor
    # For Prototype latents: no transform needed
    
    # Decode
    with torch.no_grad():
        pixels = vae.decode(latents, return_dict=False)[0]
    
    # Convert from [-1, 1] to [0, 1]
    pixels = (pixels / 2 + 0.5).clamp(0, 1)
    
    return pixels


def encode_pixels_to_latents(
    vae,
    pixels: torch.Tensor,
    use_scaled: bool = True,
) -> torch.Tensor:
    """
    Encode pixels to latent space.
    
    Args:
        vae: VAE model
        pixels: Pixel tensor (B, 3, H, W) in [0, 1]
        use_scaled: Whether to return Pipeline format (scaled)
        
    Returns:
        Latent tensor (B, C, H//8, W//8)
    """
    import torch
    
    # Normalize to [-1, 1]
    pixels = pixels * 2.0 - 1.0
    pixels = pixels.to(vae.device, dtype=vae.dtype)
    
    with torch.no_grad():
        latent = vae.encode(pixels).latent_dist.sample()
    
    # Apply scaling for Pipeline format
    if use_scaled:
        scaling_factor = getattr(vae.config, 'scaling_factor', SCALING_FACTOR)
        shift_factor = getattr(vae.config, 'shift_factor', SHIFT_FACTOR)
        latent = (latent - shift_factor) * scaling_factor
    
    return latent


def convert_latents_prototype_to_pipeline(
    latents: torch.Tensor,
    scaling_factor: float = SCALING_FACTOR,
    shift_factor: float = SHIFT_FACTOR,
) -> torch.Tensor:
    """
    Convert Prototype latents to Pipeline format.
    
    Prototype (raw VAE output, std≈4.8) -> Pipeline (scaled, std≈0.6)
    
    Args:
        latents: Prototype format latents
        scaling_factor: VAE scaling factor
        shift_factor: VAE shift factor
        
    Returns:
        Pipeline format latents
    """
    return (latents - shift_factor) * scaling_factor


def convert_latents_pipeline_to_prototype(
    latents: torch.Tensor,
    scaling_factor: float = SCALING_FACTOR,
    shift_factor: float = SHIFT_FACTOR,
) -> torch.Tensor:
    """
    Convert Pipeline latents to Prototype format.
    
    Pipeline (scaled, std≈0.6) -> Prototype (raw VAE output, std≈4.8)
    
    Args:
        latents: Pipeline format latents
        scaling_factor: VAE scaling factor
        shift_factor: VAE shift factor
        
    Returns:
        Prototype format latents
    """
    return latents / scaling_factor + shift_factor
