"""
Turbo Trainer - 多模型加速训练框架

支持模型:
- Z-Image Turbo
- LongCat-Image
- (更多模型待添加)

架构:
- turbo_trainer.core: 核心训练逻辑（模型无关）
- turbo_trainer.data: 数据处理
- turbo_trainer.networks: 通用网络组件
- turbo_trainer.utils: 通用工具函数
- models.*: 模型特定实现
"""

__version__ = "0.2.0"

from . import core
from . import data
from . import networks
from . import utils

