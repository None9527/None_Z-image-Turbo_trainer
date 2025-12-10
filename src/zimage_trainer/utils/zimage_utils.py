# Copyright 2025 Z-Image Team and musubi-tuner. All rights reserved.
# Utility functions for Z-Image pipeline.

import json
import logging
from typing import List, Optional, Union

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Z-Image Constants
VAE_SCALE_FACTOR = 8  # Z-Image VAE uses 8x compression (verified: 1024/128 = 8)


def pack_latents(latents: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
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


def _convert_zimage_vae_keys(state_dict: dict) -> dict:
    """
    Convert Z-Image VAE state dict keys from original format to diffusers format.
    
    Z-Image VAE uses a format similar to original SD VAE but without quant_conv layers.
    
    Key mappings:
    - encoder.down.X.block.Y -> encoder.down_blocks.X.resnets.Y
    - decoder.up.X.block.Y -> decoder.up_blocks.X.resnets.Y  
    - encoder.mid.block_X -> encoder.mid_block.resnets.X-1
    - decoder.mid.block_X -> decoder.mid_block.resnets.X-1
    - *.attn_1.* -> *.attentions.0.*
    - .k. -> .to_k.
    - .q. -> .to_q.
    - .v. -> .to_v.
    - .proj_out. -> .to_out.0.
    - .norm. -> .group_norm.
    - .nin_shortcut. -> .conv_shortcut.
    """
    import re
    
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # encoder.down.X.block.Y -> encoder.down_blocks.X.resnets.Y
        new_key = re.sub(r'encoder\.down\.(\d+)\.block\.(\d+)', r'encoder.down_blocks.\1.resnets.\2', new_key)
        
        # decoder.up.X.block.Y -> decoder.up_blocks.X.resnets.Y
        new_key = re.sub(r'decoder\.up\.(\d+)\.block\.(\d+)', r'decoder.up_blocks.\1.resnets.\2', new_key)
        
        # encoder.down.X.downsample -> encoder.down_blocks.X.downsamplers.0
        new_key = re.sub(r'encoder\.down\.(\d+)\.downsample', r'encoder.down_blocks.\1.downsamplers.0', new_key)
        
        # decoder.up.X.upsample -> decoder.up_blocks.X.upsamplers.0
        new_key = re.sub(r'decoder\.up\.(\d+)\.upsample', r'decoder.up_blocks.\1.upsamplers.0', new_key)
        
        # encoder.mid.block_1 -> encoder.mid_block.resnets.0
        new_key = re.sub(r'encoder\.mid\.block_1', r'encoder.mid_block.resnets.0', new_key)
        new_key = re.sub(r'encoder\.mid\.block_2', r'encoder.mid_block.resnets.1', new_key)
        
        # decoder.mid.block_1 -> decoder.mid_block.resnets.0
        new_key = re.sub(r'decoder\.mid\.block_1', r'decoder.mid_block.resnets.0', new_key)
        new_key = re.sub(r'decoder\.mid\.block_2', r'decoder.mid_block.resnets.1', new_key)
        
        # encoder.mid.attn_1 -> encoder.mid_block.attentions.0
        new_key = re.sub(r'encoder\.mid\.attn_1', r'encoder.mid_block.attentions.0', new_key)
        
        # decoder.mid.attn_1 -> decoder.mid_block.attentions.0
        new_key = re.sub(r'decoder\.mid\.attn_1', r'decoder.mid_block.attentions.0', new_key)
        
        # Attention layer key conversions
        new_key = re.sub(r'\.k\.', r'.to_k.', new_key)
        new_key = re.sub(r'\.q\.', r'.to_q.', new_key)
        new_key = re.sub(r'\.v\.', r'.to_v.', new_key)
        new_key = re.sub(r'\.proj_out\.', r'.to_out.0.', new_key)
        new_key = re.sub(r'\.norm\.', r'.group_norm.', new_key)
        
        # ResNet block conversions
        new_key = re.sub(r'\.nin_shortcut\.', r'.conv_shortcut.', new_key)
        new_key = re.sub(r'\.norm1\.', r'.norm1.', new_key)  # already correct
        new_key = re.sub(r'\.norm2\.', r'.norm2.', new_key)  # already correct
        
        # encoder.norm_out -> encoder.conv_norm_out
        new_key = re.sub(r'encoder\.norm_out', r'encoder.conv_norm_out', new_key)
        
        # decoder.norm_out -> decoder.conv_norm_out
        new_key = re.sub(r'decoder\.norm_out', r'decoder.conv_norm_out', new_key)
        
        new_state_dict[new_key] = value
    
    return new_state_dict


def load_vae(
    vae_path: str, 
    device: Union[str, torch.device] = "cpu", 
    disable_mmap: bool = False,
    dtype: torch.dtype = torch.float32,
):
    """
    Load VAE model from path using diffusers.
    
    Args:
        vae_path: Path to VAE checkpoint (directory or .safetensors file)
        device: Device to load model on
        disable_mmap: Whether to disable memory mapping
        dtype: Data type for model weights
        
    Returns:
        Loaded VAE model
    """
    import os
    try:
        from diffusers import AutoencoderKL
    except ImportError:
        raise ImportError("diffusers is required for loading VAE")
    
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
    
    logger.info(f"VAE loaded successfully on {device}")
    return vae


def load_transformer(
    transformer_path: str,
    device: Union[str, torch.device] = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    attn_mode: str = "torch",
    split_attn: bool = False,
    num_layers: Optional[int] = None,
    model_size: str = "turbo",
    fp8_scaled: bool = False,
    lora_weights_list: Optional[dict] = None,
    lora_multipliers: Optional[list] = None,
    disable_numpy_memmap: bool = False,
):
    """
    Load Z-Image transformer model from local safetensors checkpoint or diffusers format.
    
    Uses official diffusers implementation.
    
    Args:
        transformer_path: Path to transformer checkpoint (safetensors file or directory)
        device: Device to load model on
        torch_dtype: Data type for model weights
        attn_mode: Attention mode (ignored for now, handled by diffusers)
        split_attn: Whether to split attention (ignored)
        num_layers: Override number of layers (ignored)
        model_size: Model size preset (ignored)
        fp8_scaled: Whether to use FP8 scaling (ignored for now)
        lora_weights_list: LoRA weights to merge (not supported yet)
        lora_multipliers: LoRA weight multipliers (not supported yet)
        disable_numpy_memmap: Whether to disable numpy mmap
        
    Returns:
        Loaded transformer model
    """
    try:
        # Use local modified version for 12GB training (Frozen + Checkpointing fix)
        from zimage_trainer.models.transformer_z_image import ZImageTransformer2DModel
        logger.info("  [LOCAL] 使用本地修复版 ZImageTransformer2DModel (use_reentrant=False)")
    except ImportError as e:
        logger.error(f"Failed to load local Z-Image model: {e}")
        from diffusers import ZImageTransformer2DModel
        logger.warning("  [FALLBACK] 使用 diffusers 版本 ZImageTransformer2DModel")

    logger.info(f"Loading Z-Image transformer from {transformer_path}")
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    
    # Check if it's a single file or directory
    if transformer_path.endswith(".safetensors"):
        transformer = ZImageTransformer2DModel.from_single_file(
            transformer_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
    else:
        transformer = ZImageTransformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
    
    transformer = transformer.to(device)
    transformer.eval()
    
    logger.info(f"Z-Image transformer loaded successfully on {device}")
    return transformer


def load_text_encoder_and_tokenizer(model_path: str, device: Union[str, torch.device] = "cpu"):
    """
    Load text encoder and tokenizer.
    
    Args:
        model_path: Path to model
        device: Device to load on
        
    Returns:
        Tuple of (text_encoder, tokenizer)
        
    Example:
        ```python
        from musubi_tuner.zimage import load_text_encoder_and_tokenizer
        
        text_encoder, tokenizer = load_text_encoder_and_tokenizer(
            "Qwen/Qwen2.5-7B-Instruct",
            device="cuda"
        )
        ```
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("transformers is required. Install it with: pip install transformers")
    
    logger.info(f"Loading text encoder from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text_encoder = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    text_encoder.eval()
    
    logger.info("Text encoder and tokenizer loaded successfully")
    return text_encoder, tokenizer


class FlowMatchEulerScheduler:
    """
    Local Flow Match Euler Discrete Scheduler implementation.
    
    Implements the sigma schedule for flow matching:
    sigma(t) = t, with t in [0, 1]
    
    This is a simplified local implementation that does not depend on diffusers.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_dynamic_shifting: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        
        self.timesteps = None
        self.sigmas = None
        self._step_index = None
        
        # Config-like interface for compatibility
        self.config = type('Config', (), {
            'shift': shift,
            'use_dynamic_shifting': use_dynamic_shifting,
            'base_shift': base_shift,
            'max_shift': max_shift,
            'base_image_seq_len': base_image_seq_len,
            'max_image_seq_len': max_image_seq_len,
        })()
    
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """Set timesteps for inference."""
        if sigmas is not None:
            self.sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
        else:
            # Linear schedule from 1.0 to 0.0
            self.sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        
        # Apply shift if specified
        if mu is not None:
            self.sigmas = self._apply_shift(self.sigmas, mu)
        
        self.timesteps = self.sigmas[:-1] * self.num_train_timesteps
        self._step_index = 0
    
    def _apply_shift(self, sigmas: torch.Tensor, mu: float) -> torch.Tensor:
        """Apply time shift to sigmas."""
        return mu * sigmas / (1 + (mu - 1) * sigmas)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = False,
    ):
        """Perform one step of the Euler method."""
        if self._step_index is None:
            self._step_index = 0
        
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        
        # Euler step: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v_pred
        # For flow matching: v = (x_1 - x_0), so x_0 = x_t - sigma * v
        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output
        
        self._step_index += 1
        
        if return_dict:
            return {"prev_sample": prev_sample}
        return (prev_sample,)
    
    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Scale noise for flow matching (linear interpolation)."""
        if isinstance(timestep, torch.Tensor):
            t = timestep.view(-1, 1, 1, 1).to(sample.device, sample.dtype)
        else:
            t = torch.tensor([timestep], device=sample.device, dtype=sample.dtype).view(-1, 1, 1, 1)
        
        # Normalize t to [0, 1] if needed
        if t.max() > 1:
            t = t / self.num_train_timesteps
        
        # Linear interpolation: x_t = (1-t) * x_0 + t * noise
        return (1 - t) * sample + t * noise


def load_scheduler(scheduler_type: str = "flow_match_euler", use_diffusers: bool = False):
    """
    Load scheduler for Z-Image.
    
    Args:
        scheduler_type: Type of scheduler to load
        use_diffusers: Whether to use diffusers scheduler (default: False for local)
        
    Returns:
        Scheduler instance
        
    Example:
        ```python
        from musubi_tuner.zimage import load_scheduler
        
        # Use local scheduler (default)
        scheduler = load_scheduler("flow_match_euler")
        
        # Use diffusers scheduler
        scheduler = load_scheduler("flow_match_euler", use_diffusers=True)
        ```
    """
    if scheduler_type == "flow_match_euler":
        if use_diffusers:
            try:
                from diffusers import FlowMatchEulerDiscreteScheduler
            except ImportError:
                raise ImportError("diffusers is required for use_diffusers=True")
            scheduler = FlowMatchEulerDiscreteScheduler()
        else:
            scheduler = FlowMatchEulerScheduler()
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    logger.info(f"Loaded scheduler: {scheduler_type} (local={not use_diffusers})")
    return scheduler


def decode_latents_to_pixels(vae, latents: torch.Tensor, is_scaled_latents: bool = True) -> torch.Tensor:
    """
    Decode latents to pixel space using VAE.
    
    This function handles the proper scaling and shifting for Z-Image VAE,
    following the official diffusers implementation.
    
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
    # Handle 5D latents by squeezing to 4D
    if latents.dim() == 5:
        # Could be (B, C, F, H, W) or (B, F, C, H, W)
        # Check which dimension is 1 (frame dimension)
        if latents.shape[1] == 1:
            # (B, F=1, C, H, W) -> (B, C, H, W)
            latents = latents.squeeze(1)
        elif latents.shape[2] == 1:
            # (B, C, F=1, H, W) -> (B, C, H, W)
            latents = latents.squeeze(2)
        else:
            # Neither is 1, take first frame
            latents = latents[:, :, 0, :, :]  # (B, C, H, W)
    
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
    # For Prototype latents: no transform needed, they are already in raw VAE space
    
    # Decode
    with torch.no_grad():
        pixels = vae.decode(latents, return_dict=False)[0]
    
    # Convert from [-1, 1] to [0, 1]
    pixels = (pixels / 2 + 0.5).clamp(0, 1)
    
    return pixels


def create_pipeline_from_components(
    vae,
    text_encoder,
    tokenizer,
    transformer,
    scheduler,
):
    """
    Create Z-Image pipeline from individual components.
    
    Args:
        vae: VAE model
        text_encoder: Text encoder
        tokenizer: Tokenizer
        transformer: Transformer model
        scheduler: Scheduler
        
    Returns:
        ZImagePipeline instance
    """
    try:
        from diffusers import ZImagePipeline
    except ImportError:
        raise ImportError("diffusers>=0.32.0 is required for ZImagePipeline")
    
    pipeline = ZImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    
    logger.info("Created Z-Image pipeline from components")
    return pipeline
