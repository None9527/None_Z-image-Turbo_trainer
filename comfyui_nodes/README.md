# Z-Image Layer Merge ComfyUI 插件

用于 ZImageTransformer2DModel 的分层融合 ComfyUI 节点集合。

## 功能特性

- **分层融合**: 支持按层范围选择性合并两个模型
- **模块级控制**: 可单独控制 attention、FFN、norm 等模块的融合比例
- **渐变模式**: 支持线性渐变和钟形曲线融合
- **多种插值**: linear / slerp

## 模型结构

基于 ZImageTransformer2DModel config:

| 组件 | 数量 | 说明 |
|------|------|------|
| `layers` | 30 | 主 Transformer 层 |
| `noise_refiner` | 2 | 噪声精炼层 |
| `context_refiner` | 2 | 上下文精炼层 |
| dim | 3840 | 隐藏维度 |
| n_heads | 30 | 注意力头数 |

## 节点说明

### 1. Z-Image Layer Merge
基础分层融合，支持：
- 主层范围 (`main_start` ~ `main_end`)
- Refiner 层开关
- 渐变模式 (none/linear_in/linear_out/bell)

### 2. Z-Image Layer Merge (Per-Layer)
30个独立 ratio 控制，精细调整每一层。

### 3. Z-Image Layer Merge (By Block)
按模块类型设置 ratio：
- `attention_qkv`: Q/K/V 投影
- `attention_out`: 输出投影
- `ffn`: 前馈网络
- `adaln`: 自适应层归一化

### 4. Z-Image Model Save
保存融合结果为 safetensors 或 diffusers 格式。

## 安装

```bash
# 将 comfyui_nodes 文件夹复制到 ComfyUI/custom_nodes/
cp -r comfyui_nodes ComfyUI/custom_nodes/zimage_layer_merge
```

## 使用示例

```
Model A (基础) ─┬─> [Z-Image Layer Merge] ─> [Z-Image Model Save]
Model B (风格) ─┘
```

典型场景：
- 层 0-9 用 A，层 10-29 用 B：保留结构，融合风格
- attention 用 B，FFN 用 A：融合注意力模式
