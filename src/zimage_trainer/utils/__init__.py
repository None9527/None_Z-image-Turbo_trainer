# -*- coding: utf-8 -*-
"""Utility functions for Z-Image training."""

from .vae_utils import load_vae, decode_latents_to_pixels, encode_pixels_to_latents
from .latent_utils import pack_latents, unpack_latents
from .training_utils import (
    get_optimizer,
    get_scheduler,
    save_checkpoint,
    load_checkpoint,
    # Gradient clipping (optional, disabled by default for Norm_opt)
    GradientClipper,
    clip_grad_norm,
    should_use_gradient_clipping,
    # Module offloading
    ModuleOffloader,
    SequentialOffloader,
    # Memory utilities
    get_gpu_memory_info,
    print_gpu_memory,
    clear_memory,
)
from .xformers_utils import (
    is_xformers_available,
    get_xformers_version,
    check_xformers_availability,
    xformers_memory_efficient_attention,
    XFormersAttention,
    enable_xformers_for_model,
    apply_xformers_to_transformer,
    get_optimal_attention_backend,
    benchmark_attention_backends,
)
from .model_hooks import (
    BlockSwapperHook,
    apply_block_swapper,
    apply_attention_optimization,
    enable_gradient_checkpointing,
    apply_all_optimizations,
)

__all__ = [
    # VAE
    "load_vae",
    "decode_latents_to_pixels",
    "encode_pixels_to_latents",
    # Latent
    "pack_latents",
    "unpack_latents",
    # Optimizer & Scheduler
    "get_optimizer",
    "get_scheduler",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    # Gradient clipping (optional for Norm_opt)
    "GradientClipper",
    "clip_grad_norm",
    "should_use_gradient_clipping",
    # Module offloading
    "ModuleOffloader",
    "SequentialOffloader",
    # Memory
    "get_gpu_memory_info",
    "print_gpu_memory",
    "clear_memory",
    # xformers
    "is_xformers_available",
    "get_xformers_version",
    "check_xformers_availability",
    "xformers_memory_efficient_attention",
    "XFormersAttention",
    "enable_xformers_for_model",
    "apply_xformers_to_transformer",
    "get_optimal_attention_backend",
    "benchmark_attention_backends",
    # model hooks (diffusers 优化挂载)
    "BlockSwapperHook",
    "apply_block_swapper",
    "apply_attention_optimization",
    "enable_gradient_checkpointing",
    "apply_all_optimizations",
]
