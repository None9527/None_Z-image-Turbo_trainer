"""
ComfyUI Z-Image Layer Merge Plugin
用于 ZImageTransformer2DModel 的分层融合

支持功能：
- 按层范围选择性合并两个模型
- 支持主 Transformer 层 (0-29) 和 Refiner 层
- 线性插值融合
"""

from .zimage_layer_merge import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
