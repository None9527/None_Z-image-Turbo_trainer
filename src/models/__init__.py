# -*- coding: utf-8 -*-
"""
Models Package - 多模型适配器

支持的模型:
- Z-Image (Turbo): 阿里 Z-Image Turbo 10 步模型
- LongCat-Image: 美团 LongCat-Image 模型（基于 FLUX）

使用方法:
    from models import get_adapter, list_adapters, auto_detect_adapter
    
    # 获取适配器
    adapter = get_adapter("zimage")
    
    # 列出所有适配器
    print(list_adapters())  # ['zimage', 'longcat']
    
    # 自动检测
    adapter_name = auto_detect_adapter("/path/to/model")

扩展新模型:
    1. 在 models/ 下创建新目录，如 models/flux/
    2. 实现 adapter.py，继承 ModelAdapter
    3. 使用 @register_adapter("flux") 装饰器注册
    4. 在 __init__.py 中导出
"""

from .base import ModelAdapter, ModelConfig, LatentInfo
from .registry import (
    register_adapter,
    get_adapter,
    list_adapters,
    auto_detect_adapter,
)

# 导入子模块以触发注册
from . import zimage
from . import longcat

# 导出具体适配器（可选，方便直接使用）
from .zimage import ZImageAdapter
from .longcat import LongCatAdapter

__all__ = [
    # 基类
    "ModelAdapter",
    "ModelConfig",
    "LatentInfo",
    # 注册表
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "auto_detect_adapter",
    # 具体适配器
    "ZImageAdapter",
    "LongCatAdapter",
]
