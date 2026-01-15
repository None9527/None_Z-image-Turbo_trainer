#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA Merge Script - 将 LoRA 权重融合进 Transformer 并保存为单个模型文件

Usage:
    python scripts/merge_lora_to_transformer.py \
        --transformer_path models/ZImage-Turbo/transformer \
        --lora_path output/lora/my_lora.safetensors \
        --output_path output/merged/my_merged_transformer.safetensors \
        --lora_scale 1.0

Features:
    - 支持多个 LoRA 同时融合 (使用多个 --lora_path)
    - 支持自定义 LoRA 强度 (--lora_scale)
    - 输出单个 .safetensors 文件
    - 支持 BF16/FP16/FP32 输出格式
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_lora_keys(lora_state: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    解析 LoRA 权重，按照目标层分组
    
    支持的格式：
    - diffusion_model.layers.0.attention.to_q.lora_down.weight
    - layers.0.attention.to_q.lora_down.weight
    
    Returns:
        Dict[layer_name, {"lora_down": Tensor, "lora_up": Tensor, "alpha": float}]
    """
    grouped = {}
    
    for key, value in lora_state.items():
        # 跳过 alpha (后面单独处理)
        if key.endswith(".alpha"):
            continue
            
        # 解析 key: xxx.lora_down.weight 或 xxx.lora_up.weight
        if ".lora_down.weight" in key:
            base_key = key.replace(".lora_down.weight", "")
            # 移除 diffusion_model. 前缀 (如果有)
            if base_key.startswith("diffusion_model."):
                base_key = base_key[len("diffusion_model."):]
            
            if base_key not in grouped:
                grouped[base_key] = {}
            grouped[base_key]["lora_down"] = value
            
        elif ".lora_up.weight" in key:
            base_key = key.replace(".lora_up.weight", "")
            if base_key.startswith("diffusion_model."):
                base_key = base_key[len("diffusion_model."):]
            
            if base_key not in grouped:
                grouped[base_key] = {}
            grouped[base_key]["lora_up"] = value
    
    # 提取 alpha 值
    for key, value in lora_state.items():
        if key.endswith(".alpha"):
            base_key = key.replace(".alpha", "")
            if base_key.startswith("diffusion_model."):
                base_key = base_key[len("diffusion_model."):]
            if base_key in grouped:
                grouped[base_key]["alpha"] = value.item() if value.numel() == 1 else float(value)
    
    return grouped


def merge_lora_to_state_dict(
    base_state: Dict[str, torch.Tensor],
    lora_state: Dict[str, torch.Tensor],
    lora_scale: float = 1.0,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    将 LoRA 权重融合到基础模型的 state_dict 中
    
    LoRA 融合公式: W' = W + scale * (alpha/rank) * (lora_up @ lora_down)
    
    Args:
        base_state: 基础模型的 state_dict
        lora_state: LoRA 权重 (safetensors 格式)
        lora_scale: LoRA 强度缩放因子 (默认 1.0)
        device: 计算设备
        
    Returns:
        融合后的 state_dict
    """
    # 解析 LoRA 权重
    lora_grouped = parse_lora_keys(lora_state)
    logger.info(f"解析到 {len(lora_grouped)} 个 LoRA 模块")
    
    # 复制基础权重
    merged_state = {k: v.clone() for k, v in base_state.items()}
    
    merged_count = 0
    skipped_count = 0
    
    for layer_name, lora_weights in tqdm(lora_grouped.items(), desc="融合 LoRA"):
        # 构建目标权重的 key
        target_key = f"{layer_name}.weight"
        
        if target_key not in merged_state:
            logger.debug(f"跳过 {layer_name}: 目标 key 不存在")
            skipped_count += 1
            continue
        
        if "lora_down" not in lora_weights or "lora_up" not in lora_weights:
            logger.warning(f"跳过 {layer_name}: 缺少 lora_down 或 lora_up")
            skipped_count += 1
            continue
        
        # 获取权重
        lora_down = lora_weights["lora_down"].to(device)
        lora_up = lora_weights["lora_up"].to(device)
        base_weight = merged_state[target_key].to(device)
        
        # 计算 scale: alpha / rank
        rank = lora_down.shape[0]
        alpha = lora_weights.get("alpha", rank)  # 默认 alpha = rank
        scale = (alpha / rank) * lora_scale
        
        # 融合: W' = W + scale * (lora_up @ lora_down)
        # lora_down: [rank, in_features]
        # lora_up: [out_features, rank]
        # delta: [out_features, in_features]
        delta = lora_up @ lora_down
        merged_state[target_key] = base_weight + scale * delta.to(base_weight.dtype)
        
        merged_count += 1
    
    logger.info(f"成功融合 {merged_count} 层, 跳过 {skipped_count} 层")
    return merged_state


def load_transformer_state_dict(transformer_path: str) -> Dict[str, torch.Tensor]:
    """
    加载 Transformer 模型的 state_dict
    
    支持两种格式：
    1. 目录格式 (diffusers): transformer_path/diffusion_pytorch_model.safetensors
    2. 单文件格式: xxx.safetensors
    """
    path = Path(transformer_path)
    
    if path.is_dir():
        # diffusers 格式目录
        model_file = path / "diffusion_pytorch_model.safetensors"
        if not model_file.exists():
            # 尝试其他可能的文件名
            model_file = path / "model.safetensors"
        if not model_file.exists():
            # 尝试 .bin 格式
            model_file = path / "diffusion_pytorch_model.bin"
            if model_file.exists():
                return torch.load(model_file, map_location="cpu")
        if not model_file.exists():
            raise FileNotFoundError(f"找不到模型文件: {path}")
        return load_file(str(model_file))
    else:
        # 单文件格式
        if path.suffix == ".safetensors":
            return load_file(str(path))
        elif path.suffix in [".bin", ".pt", ".pth"]:
            return torch.load(str(path), map_location="cpu")
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA Merge Script - 将 LoRA 融合进 Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python merge_lora_to_transformer.py \\
      --transformer_path models/ZImage-Turbo/transformer \\
      --lora_path output/lora/style_lora.safetensors \\
      --output_path output/merged/merged_model.safetensors

  # 调整 LoRA 强度
  python merge_lora_to_transformer.py \\
      --transformer_path models/ZImage-Turbo/transformer \\
      --lora_path output/lora/style_lora.safetensors \\
      --output_path output/merged/merged_model.safetensors \\
      --lora_scale 0.8

  # 融合多个 LoRA
  python merge_lora_to_transformer.py \\
      --transformer_path models/ZImage-Turbo/transformer \\
      --lora_path output/lora/lora1.safetensors \\
      --lora_path output/lora/lora2.safetensors \\
      --lora_scale 1.0 0.5 \\
      --output_path output/merged/merged_model.safetensors
        """
    )
    
    parser.add_argument(
        "--transformer_path", "-t",
        type=str, required=True,
        help="基础 Transformer 模型路径 (目录或 .safetensors 文件)"
    )
    parser.add_argument(
        "--lora_path", "-l",
        type=str, action="append", required=True,
        help="LoRA 权重文件路径 (可多次指定以融合多个 LoRA)"
    )
    parser.add_argument(
        "--lora_scale", "-s",
        type=float, nargs="+", default=[1.0],
        help="LoRA 强度 (默认 1.0，可为每个 LoRA 指定不同强度)"
    )
    parser.add_argument(
        "--output_path", "-o",
        type=str, required=True,
        help="输出文件路径 (.safetensors)"
    )
    parser.add_argument(
        "--dtype",
        type=str, choices=["bf16", "fp16", "fp32"], default="bf16",
        help="输出精度 (默认 bf16)"
    )
    parser.add_argument(
        "--device",
        type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备 (默认自动选择)"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not Path(args.transformer_path).exists():
        logger.error(f"Transformer 路径不存在: {args.transformer_path}")
        sys.exit(1)
    
    for lora_path in args.lora_path:
        if not Path(lora_path).exists():
            logger.error(f"LoRA 文件不存在: {lora_path}")
            sys.exit(1)
    
    # 扩展 lora_scale 列表
    scales = args.lora_scale
    if len(scales) < len(args.lora_path):
        scales = scales + [scales[-1]] * (len(args.lora_path) - len(scales))
    
    # 确定输出精度
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    output_dtype = dtype_map[args.dtype]
    
    logger.info("=" * 60)
    logger.info("LoRA Merge Script")
    logger.info("=" * 60)
    logger.info(f"Transformer: {args.transformer_path}")
    for i, (lora_path, scale) in enumerate(zip(args.lora_path, scales)):
        logger.info(f"LoRA {i+1}: {lora_path} (scale={scale})")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Dtype: {args.dtype}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)
    
    # 1. 加载基础模型
    logger.info("\n[1/3] 加载 Transformer 模型...")
    base_state = load_transformer_state_dict(args.transformer_path)
    logger.info(f"  ✓ 加载了 {len(base_state)} 个参数")
    
    # 计算原始模型大小
    original_size = sum(v.numel() * v.element_size() for v in base_state.values()) / 1024 / 1024
    logger.info(f"  ✓ 原始大小: {original_size:.1f} MB")
    
    # 2. 依次融合 LoRA
    logger.info("\n[2/3] 融合 LoRA 权重...")
    merged_state = base_state
    
    for i, (lora_path, scale) in enumerate(zip(args.lora_path, scales)):
        logger.info(f"\n融合 LoRA {i+1}/{len(args.lora_path)}: {Path(lora_path).name}")
        lora_state = load_file(lora_path)
        logger.info(f"  LoRA 包含 {len(lora_state)} 个 tensor, scale={scale}")
        
        merged_state = merge_lora_to_state_dict(
            merged_state, lora_state,
            lora_scale=scale,
            device=args.device
        )
    
    # 3. 转换精度并保存
    logger.info(f"\n[3/3] 保存融合后的模型 ({args.dtype})...")
    
    # 转换精度
    for key in merged_state:
        merged_state[key] = merged_state[key].to(output_dtype).cpu()
    
    # 确保输出目录存在
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存
    save_file(merged_state, str(output_path))
    
    # 计算输出大小
    output_size = output_path.stat().st_size / 1024 / 1024
    logger.info(f"  ✓ 保存完成: {output_path}")
    logger.info(f"  ✓ 文件大小: {output_size:.1f} MB")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ 融合完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
