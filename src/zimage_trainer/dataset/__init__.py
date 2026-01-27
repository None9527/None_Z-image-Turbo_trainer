# -*- coding: utf-8 -*-
"""Dataset utilities for Z-Image training."""

from .config_utils import DatasetConfig, load_dataset_config
from .dpo_dataset import DPOLatentDataset, create_dpo_dataloader

__all__ = [
    "DatasetConfig",
    "load_dataset_config",
    "DPOLatentDataset",
    "create_dpo_dataloader",
]


def create_dataloader(dataset_config: DatasetConfig, **kwargs):
    """Create dataloader from config (placeholder)."""
    # This is a simplified placeholder
    # Full implementation would load cached latents and text embeddings
    raise NotImplementedError("Full dataloader implementation needed")

