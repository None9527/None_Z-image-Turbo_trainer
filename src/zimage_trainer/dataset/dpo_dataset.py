# -*- coding: utf-8 -*-
"""
DPO Dataset - Dataset for Direct Preference Optimization training.

Loads pre-cached latent pairs (preferred/rejected) for DPO training.
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file

try:
    import toml
except ImportError:
    try:
        import tomli as toml
    except ImportError:
        toml = None

logger = logging.getLogger(__name__)


class DPOLatentDataset(Dataset):
    """
    Dataset for DPO training with pre-cached preferred/rejected latent pairs.
    
    Expected cache format:
      - {name}_preferred_{WxH}_zi.safetensors  (优选图像 latent)
      - {name}_rejected_{WxH}_zi.safetensors   (劣选图像 latent)
      - {name}_te.safetensors                  (共享文本嵌入)
    
    Alternative format (single file per pair):
      - {name}_{WxH}_zi_dpo.safetensors
        Contains: latent_preferred, latent_rejected, vl_embed_*
    
    Args:
        datasets: List of dataset configs with keys:
            - cache_dir: Path to cached latents directory
            - resolution_limit: Optional max resolution filter
        shuffle: Whether to shuffle within dataset
        max_sequence_length: Max text embedding sequence length
        cache_arch: Cache architecture ('zi' for Z-Image)
    """
    
    def __init__(
        self,
        datasets: List[Dict],
        shuffle: bool = True,
        max_sequence_length: int = 512,
        cache_arch: str = "zi",
    ):
        self.max_sequence_length = max_sequence_length
        self.cache_arch = cache_arch
        
        self.cache_files: List[Tuple[Path, Path, Path]] = []  # (preferred, rejected, text_embed)
        self.resolutions: List[Tuple[int, int]] = []
        
        for ds_cfg in datasets:
            cache_dir = Path(ds_cfg.get("cache_dir", ds_cfg.get("path", "")))
            resolution_limit = ds_cfg.get("resolution_limit", None)
            self._load_dataset(cache_dir, resolution_limit)
        
        if not self.cache_files:
            raise ValueError("No valid DPO cache files found")
        
        logger.info(f"DPO dataset: {len(self.cache_files)} preference pairs")
        
        if shuffle:
            import random
            indices = list(range(len(self.cache_files)))
            random.shuffle(indices)
            self.cache_files = [self.cache_files[i] for i in indices]
            self.resolutions = [self.resolutions[i] for i in indices]
    
    def _load_dataset(self, cache_dir: Path, resolution_limit: Optional[int] = None):
        """Load DPO cache files from directory."""
        if not cache_dir.exists():
            logger.warning(f"DPO cache directory not found: {cache_dir}")
            return
        
        # Pattern 1: Separate preferred/rejected files
        preferred_files = list(cache_dir.glob(f"*_preferred_*_{self.cache_arch}.safetensors"))
        
        for pref_path in preferred_files:
            # Derive rejected and text embedding paths
            name_parts = pref_path.stem.replace("_preferred_", "_SPLIT_").split("_SPLIT_")
            if len(name_parts) != 2:
                continue
            
            base_name, resolution_arch = name_parts
            rej_path = cache_dir / f"{base_name}_rejected_{resolution_arch}.safetensors"
            te_path = cache_dir / f"{base_name}_te.safetensors"
            
            if not rej_path.exists():
                logger.warning(f"Missing rejected file for: {pref_path.name}")
                continue
            
            if not te_path.exists():
                logger.warning(f"Missing text embedding for: {pref_path.name}")
                continue
            
            # Parse resolution
            res = self._parse_resolution(resolution_arch)
            if res is None:
                continue
            
            if resolution_limit and max(res) > resolution_limit:
                continue
            
            self.cache_files.append((pref_path, rej_path, te_path))
            self.resolutions.append(res)
        
        # Pattern 2: Combined DPO files
        combined_files = list(cache_dir.glob(f"*_{self.cache_arch}_dpo.safetensors"))
        
        for dpo_path in combined_files:
            # Resolution is embedded in filename
            res = self._parse_resolution(dpo_path.stem)
            if res is None:
                continue
            
            if resolution_limit and max(res) > resolution_limit:
                continue
            
            # Use dpo_path for all three (will be handled in __getitem__)
            self.cache_files.append((dpo_path, dpo_path, dpo_path))
            self.resolutions.append(res)
    
    def _parse_resolution(self, name: str) -> Optional[Tuple[int, int]]:
        """Parse resolution from filename (e.g., 1024x1024_zi)."""
        import re
        match = re.search(r'(\d+)x(\d+)', name)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None
    
    def __len__(self) -> int:
        return len(self.cache_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pref_path, rej_path, te_path = self.cache_files[idx]
        
        # Check if combined format (all paths are the same)
        if pref_path == rej_path == te_path:
            # Combined DPO file format
            data = load_file(str(pref_path))
            preferred_latents = data["latent_preferred"]
            rejected_latents = data["latent_rejected"]
            
            # Load VL embeddings
            vl_embed = []
            for i in range(10):  # Up to 10 VL blocks
                key = f"vl_embed_{i}"
                if key in data:
                    vl_embed.append(data[key])
                else:
                    break
        else:
            # Separate files format
            pref_data = load_file(str(pref_path))
            rej_data = load_file(str(rej_path))
            te_data = load_file(str(te_path))
            
            preferred_latents = pref_data.get("latent", pref_data.get("latents"))
            rejected_latents = rej_data.get("latent", rej_data.get("latents"))
            
            # Load VL embeddings from text encoder cache
            vl_embed = []
            for i in range(10):
                key = f"vl_embed_{i}"
                if key in te_data:
                    vl_embed.append(te_data[key])
                else:
                    break
        
        return {
            "preferred_latents": preferred_latents,
            "rejected_latents": rejected_latents,
            "vl_embed": vl_embed,
        }


def create_dpo_dataloader(args) -> DataLoader:
    """
    Create DPO dataloader from training args.
    
    Args:
        args: Training arguments with dataset_config path
    
    Returns:
        DataLoader for DPO training
    """
    if toml is None:
        raise ImportError("toml or tomli required for config parsing")
    
    config_path = getattr(args, "dataset_config", getattr(args, "config", None))
    if not config_path or not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")
    
    config = toml.load(config_path)
    
    # Get DPO datasets config
    dpo_cfg = config.get("dpo", config.get("dataset", {}))
    datasets = dpo_cfg.get("datasets", [])
    
    # Support single dataset shorthand
    if not datasets and "cache_dir" in dpo_cfg:
        datasets = [{"cache_dir": dpo_cfg["cache_dir"]}]
    
    if not datasets:
        raise ValueError("No DPO datasets configured in TOML")
    
    # Create dataset
    dataset = DPOLatentDataset(
        datasets=datasets,
        shuffle=True,
        max_sequence_length=dpo_cfg.get("max_sequence_length", 512),
        cache_arch=dpo_cfg.get("cache_arch", "zi"),
    )
    
    # Create dataloader
    batch_size = config.get("training", {}).get("batch_size", 1)
    
    def collate_fn(batch):
        """Collate DPO batch samples."""
        preferred = torch.stack([b["preferred_latents"] for b in batch])
        rejected = torch.stack([b["rejected_latents"] for b in batch])
        
        # Stack VL embeddings (per block)
        num_blocks = len(batch[0]["vl_embed"])
        vl_embed = []
        for i in range(num_blocks):
            block_embeds = torch.stack([b["vl_embed"][i] for b in batch])
            vl_embed.append(block_embeds)
        
        return {
            "preferred_latents": preferred,
            "rejected_latents": rejected,
            "vl_embed": vl_embed,
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Already shuffled in dataset
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    return dataloader
