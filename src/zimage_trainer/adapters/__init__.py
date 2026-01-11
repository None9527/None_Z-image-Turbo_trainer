# -*- coding: utf-8 -*-
"""
适配器模块 - 向后兼容重定向

注意: 此模块已移动到 src/models/
这里只保留向后兼容的重定向导入。

新代码请使用:
    from models import get_adapter, list_adapters, ModelAdapter
"""

import warnings

# 向后兼容：从新位置导入
from models import (
    ModelAdapter,
    ModelConfig,
    get_adapter,
    list_adapters,
    auto_detect_adapter,
    register_adapter,
    ZImageAdapter,
)

# 发出弃用警告
warnings.warn(
    "zimage_trainer.adapters 已弃用，请改用 from models import get_adapter, ...",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "ModelAdapter",
    "ModelConfig",
    "get_adapter",
    "list_adapters",
    "auto_detect_adapter",
    "register_adapter",
    "ZImageAdapter",
]
