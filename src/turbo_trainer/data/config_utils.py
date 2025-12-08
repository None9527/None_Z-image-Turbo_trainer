# -*- coding: utf-8 -*-
"""
Dataset configuration utilities for Z-Image training.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

try:
    import toml
except ImportError:
    import tomli as toml

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    image_directory: str
    cache_directory: Optional[str] = None
    num_repeats: int = 1
    batch_size: int = 1
    resolution: Tuple[int, int] = (1024, 1024)
    enable_bucket: bool = True
    bucket_no_upscale: bool = True
    caption_extension: str = ".txt"
    

def load_dataset_config(config_path: str) -> List[DatasetConfig]:
    """
    Load dataset configuration from TOML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        List of DatasetConfig objects
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)
    
    datasets = []
    for ds_config in config.get("datasets", []):
        resolution = ds_config.get("resolution", [1024, 1024])
        if isinstance(resolution, list):
            resolution = tuple(resolution)
        
        datasets.append(DatasetConfig(
            image_directory=ds_config["image_directory"],
            cache_directory=ds_config.get("cache_directory"),
            num_repeats=ds_config.get("num_repeats", 1),
            batch_size=ds_config.get("batch_size", 1),
            resolution=resolution,
            enable_bucket=ds_config.get("enable_bucket", True),
            bucket_no_upscale=ds_config.get("bucket_no_upscale", True),
            caption_extension=ds_config.get("caption_extension", ".txt"),
        ))
    
    return datasets


class ImageDataset(Dataset):
    """Simple image dataset for Z-Image training."""
    
    def __init__(
        self,
        config: DatasetConfig,
        use_cache: bool = True,
    ):
        self.config = config
        self.use_cache = use_cache
        
        # Find all images
        self.image_paths = []
        extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        
        for ext in extensions:
            for f in os.listdir(config.image_directory):
                if f.lower().endswith(ext):
                    self.image_paths.append(os.path.join(config.image_directory, f))
        
        self.image_paths.sort()
        
        # Apply num_repeats
        self.indices = list(range(len(self.image_paths))) * config.num_repeats
        
        logger.info(f"Found {len(self.image_paths)} images, {len(self.indices)} samples with repeats")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_idx = self.indices[idx]
        image_path = self.image_paths[image_idx]
        
        # Try to load from cache
        if self.use_cache and self.config.cache_directory:
            cache_path = self._get_cache_path(image_path)
            if os.path.exists(cache_path):
                return self._load_from_cache(cache_path, image_path)
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = self._preprocess_image(image)
        
        # Load caption
        caption = self._load_caption(image_path)
        
        return {
            "image": image,
            "caption": caption,
            "image_path": image_path,
        }
    
    def _get_cache_path(self, image_path: str) -> str:
        """Get cache file path for an image."""
        basename = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.config.cache_directory, f"{basename}_zi.safetensors")
    
    def _load_from_cache(self, cache_path: str, image_path: str) -> Dict[str, Any]:
        """Load cached latent and text embeddings."""
        from safetensors.torch import load_file
        
        cache = load_file(cache_path)
        caption = self._load_caption(image_path)
        
        return {
            "latent": cache.get("latent"),
            "caption": caption,
            "image_path": image_path,
        }
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for VAE encoding."""
        # Resize to target resolution
        image = image.resize(self.config.resolution, Image.LANCZOS)
        
        # Convert to tensor
        image = np.array(image).astype(np.float32)
        image = image / 127.5 - 1.0  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        return image
    
    def _load_caption(self, image_path: str) -> str:
        """Load caption for an image."""
        caption_path = os.path.splitext(image_path)[0] + self.config.caption_extension
        
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        
        return ""


def create_dataloader(
    config: DatasetConfig,
    use_cache: bool = True,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader from dataset config.
    
    Args:
        config: Dataset configuration
        use_cache: Use cached latents if available
        num_workers: Number of data loading workers
        shuffle: Shuffle data
        
    Returns:
        DataLoader instance
    """
    dataset = ImageDataset(config, use_cache=use_cache)
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

