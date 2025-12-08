# -*- coding: utf-8 -*-
"""
通用数据集类

提供 CachedLatentDataset 作为别名，实际实现在 dataloader.py 中。
后续可以在此扩展更多通用数据集类。
"""

from .dataloader import ZImageLatentDataset as CachedLatentDataset

__all__ = ["CachedLatentDataset"]

