# -*- coding: utf-8 -*-
"""
Dataset and DataLoader for Z-Image training.

Standalone implementation - no musubi-tuner dependency.
"""

import os
import glob
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


class ZImageLatentDataset(Dataset):
    """
    Dataset for loading pre-cached latents and text embeddings.
    Supports multiple datasets and per-dataset resolution filtering.
    """
    
    def __init__(
        self,
        datasets: List[Dict],
        shuffle: bool = True,
        max_sequence_length: int = 512,
        cache_arch: str = "zi",  # 'zi' (Z-Image) or 'lc' (LongCat)
    ):
        super().__init__()
        
        self.datasets = datasets
        self.shuffle = shuffle
        self.max_sequence_length = max_sequence_length
        self.cache_arch = cache_arch
        
        # Determine suffixes based on architecture
        self.latent_pattern = f"*_{cache_arch}.safetensors"
        self.te_suffix = f"_{cache_arch}_te.safetensors"
        
        self.cache_files = []
        self.resolutions = []
        
        for ds_config in datasets:
            cache_dir = Path(ds_config['cache_directory'])
            repeats = ds_config.get('num_repeats', 1)
            resolution_limit = ds_config.get('resolution_limit', None)
            
            logger.info(f"Loading dataset from: {cache_dir} (repeats={repeats}, limit={resolution_limit})")
            
            files, res_list = self._load_dataset(cache_dir, resolution_limit)
            
            # Apply repeats
            if repeats > 1:
                files = files * repeats
                res_list = res_list * repeats
            
            self.cache_files.extend(files)
            self.resolutions.extend(res_list)
            
        if len(self.cache_files) == 0:
            raise ValueError("No valid cache files found in any dataset")
            
        logger.info(f"Total samples: {len(self.cache_files)} (max_seq_len={max_sequence_length})")
    
    def _load_dataset(self, cache_dir: Path, resolution_limit: Optional[int]) -> Tuple[List[Tuple[Path, Path]], List[Tuple[int, int]]]:
        """Load files from a single directory and filter by resolution"""
        files = []
        resolutions = []
        
        # Find all latent files
        latent_files = list(cache_dir.glob(self.latent_pattern))
        
        for latent_path in latent_files:
            # Parse resolution
            res = self._parse_resolution(latent_path.stem)
            
            # Filter by resolution limit
            if resolution_limit:
                h, w = res
                if max(h, w) > resolution_limit:
                    continue
            
            # Find text encoder cache
            te_path = self._find_te_path(latent_path, cache_dir)
            
            if te_path and te_path.exists():
                files.append((latent_path, te_path))
                resolutions.append(res)
            
        return files, resolutions

    def _parse_resolution(self, name: str) -> Tuple[int, int]:
        """Parse resolution from filename (e.g., image_1024x1024_zi)"""
        parts = name.split('_')
        res = (1024, 1024) # Default
        for part in parts:
            if 'x' in part and part.replace('x', '').isdigit():
                try:
                    w, h = map(int, part.split('x'))
                    res = (h, w) # (H, W)
                    break
                except:
                    pass
        return res

    def _find_te_path(self, latent_path: Path, cache_dir: Path) -> Optional[Path]:
        """Construct text encoder cache path"""
        name = latent_path.stem
        parts = name.rsplit('_', 2)
        if len(parts) >= 3:
            base_name = parts[0]
        else:
            base_name = name.rsplit('_', 1)[0]
        
        return cache_dir / f"{base_name}{self.te_suffix}"
    
    def __len__(self) -> int:
        return len(self.cache_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        latent_path, te_path = self.cache_files[idx]
        
        # Load latent
        latent_data = load_file(str(latent_path))
        latent_key = next((k for k in latent_data.keys() if k.startswith('latents_')), None)
        if latent_key is None:
            raise ValueError(f"No latent key found in {latent_path}")
        latents = latent_data[latent_key]
        
        # 确保latent尺寸能被patch_size=2整除（为Transformer准备）
        C, H, W = latents.shape
        patch_size = 2
        
        # 计算需要填充的尺寸
        H_padded = ((H + patch_size - 1) // patch_size) * patch_size
        W_padded = ((W + patch_size - 1) // patch_size) * patch_size
        
        if H != H_padded or W != W_padded:
            # 填充latent到合适的尺寸 (left, right, top, bottom)
            latents = torch.nn.functional.pad(
                latents, 
                (0, W_padded - W, 0, H_padded - H),  # (left, right, top, bottom)
                mode='reflect'
            )
        
        # Load text encoder output
        te_data = load_file(str(te_path))
        vl_embed_key = next((k for k in te_data.keys() if 'vl_embed' in k), None)
        if vl_embed_key is None:
            raise ValueError(f"No vl_embed key found in {te_path}")
        vl_embed = te_data[vl_embed_key]
        
        # 截断/填充到 max_sequence_length
        seq_len = vl_embed.shape[0]
        if seq_len > self.max_sequence_length:
            vl_embed = vl_embed[:self.max_sequence_length]
        elif seq_len < self.max_sequence_length:
            pad_len = self.max_sequence_length - seq_len
            vl_embed = torch.nn.functional.pad(vl_embed, (0, 0, 0, pad_len), mode='constant', value=0)
        
        return {
            'latents': latents,
            'vl_embed': vl_embed,
        }


class BucketBatchSampler(torch.utils.data.Sampler):
    """
    支持分桶的 Batch Sampler。
    将具有相同分辨率的样本组合在一起。
    """
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 按分辨率分组索引
        self.buckets = {} # (h, w) -> [indices]
        for idx, res in enumerate(dataset.resolutions):
            if res not in self.buckets:
                self.buckets[res] = []
            self.buckets[res].append(idx)
            
    def __iter__(self):
        batches = []
        for res, indices in self.buckets.items():
            if self.shuffle:
                # 打乱桶内索引
                indices = torch.tensor(indices)[torch.randperm(len(indices))].tolist()
            
            # 生成 batch
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        if self.shuffle:
            # 打乱 batch 顺序
            import random
            random.shuffle(batches)
            
        for batch in batches:
            yield batch

    def __len__(self):
        count = 0
        for indices in self.buckets.values():
            if self.drop_last:
                count += len(indices) // self.batch_size
            else:
                count += (len(indices) + self.batch_size - 1) // self.batch_size
        return count


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义 collate 函数。支持不同分辨率的 latent（自动 padding）。
    """
    # 检查是否所有 latents 具有相同形状
    shapes = [item['latents'].shape for item in batch]
    all_same = all(s == shapes[0] for s in shapes)
    
    if all_same:
        # 所有形状相同，直接 stack
        latents = torch.stack([item['latents'] for item in batch])
    else:
        # 形状不同，需要 padding 到最大尺寸
        max_h = max(s[1] for s in shapes)
        max_w = max(s[2] for s in shapes)
        
        # 确保尺寸能被 patch_size=2 整除
        patch_size = 2
        max_h = ((max_h + patch_size - 1) // patch_size) * patch_size
        max_w = ((max_w + patch_size - 1) // patch_size) * patch_size
        
        padded_latents = []
        for item in batch:
            lat = item['latents']
            c, h, w = lat.shape
            if h < max_h or w < max_w:
                # Pad to max size (right and bottom padding)
                lat = torch.nn.functional.pad(
                    lat,
                    (0, max_w - w, 0, max_h - h),
                    mode='constant',
                    value=0
                )
            padded_latents.append(lat)
        
        latents = torch.stack(padded_latents)
        logger.debug(f"Padded latents from {shapes} to {latents.shape}")
    
    vl_embeds = [item['vl_embed'] for item in batch]  # 保持 list 形式
    
    return {
        'latents': latents,
        'vl_embed': vl_embeds,
    }


def create_dataloader(args) -> DataLoader:
    """
    从配置创建 DataLoader。
    
    Args:
        args: 训练参数，包含dataset_config和其他相关配置
        
    Returns:
        DataLoader: 数据加载器
    """
    # 读取 dataset 配置
    if hasattr(args, 'dataset_config') and args.dataset_config:
        config = _read_dataset_config(args.dataset_config)
    else:
        config = {}
    
    # 获取参数
    datasets = config.get('datasets', [])
    
    # 兼容旧配置 (如果 config 中没有 datasets，尝试从 args 或旧 config 读取)
    if not datasets:
        cache_dir = config.get('cache_directory', getattr(args, 'cache_directory', None))
        if cache_dir:
            datasets = [{
                'cache_directory': cache_dir,
                'num_repeats': config.get('num_repeats', getattr(args, 'num_repeats', 1)),
                'resolution_limit': config.get('resolution_limit', None) # 兼容旧的 global limit
            }]
    
    if not datasets:
        raise ValueError("No datasets configured. Please check dataset_config.toml or arguments.")
    
    batch_size = config.get('batch_size', getattr(args, 'batch_size', 4))
    num_workers = config.get('num_workers', getattr(args, 'num_workers', 4))
    max_sequence_length = config.get('max_sequence_length', getattr(args, 'max_sequence_length', 512))
    
    # 分桶设置：--disable_bucket 优先级最高
    if getattr(args, 'disable_bucket', False):
        enable_bucket = False
    else:
        enable_bucket = config.get('enable_bucket', getattr(args, 'enable_bucket', True))
    
    # 读取 Architecture 标识 (区分 zi/lc 缓存)
    cache_arch = config.get('cache_arch', getattr(args, 'cache_arch', 'zi'))
    
    # 创建 dataset
    dataset = ZImageLatentDataset(
        datasets=datasets,
        max_sequence_length=max_sequence_length,
        cache_arch=cache_arch,
    )
    
    if enable_bucket:
        logger.info("🌊 启用分桶 (BucketBatchSampler)")
        batch_sampler = BucketBatchSampler(
            dataset, 
            batch_size=batch_size,
            drop_last=True,
            shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
    
    logger.info("📦 DataLoader 创建完成")
    return dataloader


def _read_dataset_config(config_path: str) -> dict:
    """
    读取 dataset 配置文件，支持多种格式：
    
    1. 合并格式 (新): [dataset] + [[dataset.sources]] 在主配置中
    2. 独立格式 (旧): [general] + [[datasets]] 在单独文件中
    3. 旧格式: [dataset] 块
    """
    if toml is None:
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    # 1. 合并格式 (新): [dataset] + [[dataset.sources]] 
    #    主配置文件中的 dataset 块
    if 'dataset' in config:
        dataset_config = config['dataset'].copy()
        # 将 sources 重命名为 datasets (兼容 create_dataloader)
        if 'sources' in dataset_config:
            dataset_config['datasets'] = dataset_config.pop('sources')
        return dataset_config
    
    # 2. 独立格式: [general] + [[datasets]]
    if 'datasets' in config:
        # 如果有 [general] 块，合并到顶层
        if 'general' in config:
            config.update(config['general'])
        return config
    
    # 3. 根级别配置 (兼容旧版)
    return config