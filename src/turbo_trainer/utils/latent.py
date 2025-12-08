# -*- coding: utf-8 -*-
"""
Latent manipulation utilities for Z-Image.

Identical to musubi-tuner implementation for compatibility.
"""

from typing import Tuple

import torch


def pack_latents(
    latents: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pack latents for Z-Image transformer.
    
    Z-Image uses standard 4D latents (B, C, H, W), packs into (B, H/2 * W/2, C*4).
    This is the "patchify" operation: 2x2 patches -> sequence.
    
    If height or width is odd, pads to even size.
    
    Args:
        latents: Latent tensor (B, C, H, W)
        
    Returns:
        Tuple of:
        - Packed latents (B, Seq, C*4) where Seq = ceil(H/2) * ceil(W/2)
        - Original (height, width) for unpack
        
    Example:
        >>> latents = torch.randn(1, 16, 128, 128)  # B=1, C=16, H=128, W=128
        >>> packed, orig_size = pack_latents(latents)
        >>> packed.shape
        torch.Size([1, 4096, 64])  # B=1, Seq=128/2*128/2=4096, C*4=16*4=64
    """
    batch_size, num_channels, height, width = latents.shape
    orig_size = (height, width)
    
    # Pad to even dimensions if needed
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        latents = torch.nn.functional.pad(latents, (0, pad_w, 0, pad_h), mode='replicate')
        height = height + pad_h
        width = width + pad_w
    
    # Reshape to extract 2x2 patches
    # (B, C, H, W) -> (B, C, H//2, 2, W//2, 2)
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    
    # Rearrange to group patches
    # (B, C, H//2, 2, W//2, 2) -> (B, H//2, W//2, C, 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    
    # Flatten patches into sequence
    # (B, H//2, W//2, C, 2, 2) -> (B, H//2 * W//2, C*4)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    
    return latents, orig_size


def unpack_latents(
    latents: torch.Tensor, 
    orig_height: int, 
    orig_width: int, 
) -> torch.Tensor:
    """
    Unpack latents from Z-Image transformer output.
    
    Converts from sequence format (B, Seq, C*4) back to spatial format (B, C, H, W).
    Handles padding that was added during pack_latents.
    
    Args:
        latents: Packed latents (B, Seq, C*4)
        orig_height: Original latent height before padding
        orig_width: Original latent width before padding
        
    Returns:
        Unpacked latents (B, C, H, W) in latent space, cropped to original size
        
    Example:
        >>> packed = torch.randn(1, 4096, 64)  # B=1, Seq=4096, C*4=64
        >>> unpacked = unpack_latents(packed, orig_height=128, orig_width=128)
        >>> unpacked.shape
        torch.Size([1, 16, 128, 128])  # B=1, C=16, H=128, W=128
    """
    batch_size, num_patches, channels = latents.shape
    
    # Calculate padded dimensions (what was used during packing)
    pad_h = orig_height % 2
    pad_w = orig_width % 2
    padded_height = orig_height + pad_h
    padded_width = orig_width + pad_w
    
    # Verify dimensions match
    expected_patches = (padded_height // 2) * (padded_width // 2)
    if num_patches != expected_patches:
        raise ValueError(
            f"Dimension mismatch: packed latents have {num_patches} patches, "
            f"but orig_height={orig_height}, orig_width={orig_width} (padded to {padded_height}x{padded_width}) "
            f"implies {expected_patches} patches"
        )
    
    # Reverse the packing process using padded dimensions
    # (B, Seq, C*4) -> (B, H//2, W//2, C, 2, 2)
    latents = latents.view(batch_size, padded_height // 2, padded_width // 2, channels // 4, 2, 2)
    
    # Rearrange dimensions
    # (B, H//2, W//2, C, 2, 2) -> (B, C, H//2, 2, W//2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    
    # Reshape to spatial format
    # (B, C, H//2, 2, W//2, 2) -> (B, C, H, W)
    latents = latents.reshape(batch_size, channels // 4, padded_height, padded_width)
    
    # Crop back to original size if padding was applied
    if pad_h or pad_w:
        latents = latents[:, :, :orig_height, :orig_width]
    
    return latents
