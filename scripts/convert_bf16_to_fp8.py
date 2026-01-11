"""
BF16 → FP8 Model Converter for Z-Image Transformer

将 BF16 精度的 DiT 模型转换为 FP8 (E4M3/E5M2) 格式，
用于加速推理或 FP8 训练。

FP8 格式说明:
- E4M3: 4位指数 + 3位尾数, 范围 ±448, 更高精度
- E5M2: 5位指数 + 2位尾数, 范围 ±57344, 更大范围

Usage:
    python scripts/convert_bf16_to_fp8.py \
        --input_dir "path/to/bf16_model" \
        --output_dir "path/to/fp8_model" \
        --fp8_format e4m3 \
        --scale_method dynamic
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# FP8 dtype mapping (PyTorch 2.1+)
FP8_DTYPES = {
    "e4m3": torch.float8_e4m3fn,   # Higher precision, smaller range
    "e5m2": torch.float8_e5m2,     # Larger range, lower precision
}


def get_fp8_scale(tensor: torch.Tensor, fp8_format: str = "e4m3") -> float:
    """
    计算 FP8 量化的缩放因子
    
    动态缩放: scale = max(|tensor|) / fp8_max
    """
    # FP8 最大表示值
    fp8_max_values = {
        "e4m3": 448.0,      # torch.finfo(torch.float8_e4m3fn).max
        "e5m2": 57344.0,    # torch.finfo(torch.float8_e5m2).max
    }
    
    fp8_max = fp8_max_values.get(fp8_format, 448.0)
    tensor_max = tensor.abs().max().item()
    
    if tensor_max == 0:
        return 1.0
    
    # 留 5% 余量防止溢出
    scale = tensor_max / (fp8_max * 0.95)
    return max(scale, 1e-12)


def convert_tensor_to_fp8(
    tensor: torch.Tensor,
    fp8_format: str = "e4m3",
    scale_method: str = "dynamic",
    fixed_scale: Optional[float] = None
) -> tuple[torch.Tensor, float]:
    """
    将单个 tensor 转换为 FP8 格式
    
    Args:
        tensor: 输入 tensor (BF16/FP16/FP32)
        fp8_format: "e4m3" 或 "e5m2"
        scale_method: "dynamic" (逐 tensor) 或 "fixed"
        fixed_scale: 固定缩放因子 (仅 scale_method="fixed" 时使用)
    
    Returns:
        (fp8_tensor, scale): FP8 tensor 和对应的缩放因子
    """
    if tensor.numel() == 0:
        return tensor, 1.0
    
    # 计算缩放因子
    if scale_method == "dynamic":
        scale = get_fp8_scale(tensor, fp8_format)
    else:
        scale = fixed_scale or 1.0
    
    # 缩放 + 转换
    fp8_dtype = FP8_DTYPES[fp8_format]
    scaled_tensor = tensor.float() / scale
    
    # Clamp 到 FP8 范围
    fp8_max = 448.0 if fp8_format == "e4m3" else 57344.0
    scaled_tensor = scaled_tensor.clamp(-fp8_max, fp8_max)
    
    # 转换为 FP8
    fp8_tensor = scaled_tensor.to(fp8_dtype)
    
    return fp8_tensor, scale


def convert_model_to_fp8(
    input_dir: str,
    output_dir: str,
    fp8_format: str = "e4m3",
    scale_method: str = "dynamic",
    skip_patterns: list[str] = None,
    convert_safetensors_only: bool = True
) -> Dict[str, Any]:
    """
    将整个模型目录从 BF16 转换为 FP8
    
    Args:
        input_dir: 输入模型目录
        output_dir: 输出目录
        fp8_format: FP8 格式
        scale_method: 缩放方法
        skip_patterns: 跳过转换的权重名模式 (保持原精度)
        convert_safetensors_only: 仅转换 .safetensors 文件
    
    Returns:
        转换统计信息
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if skip_patterns is None:
        # 默认跳过: 位置编码、LayerNorm、偏置项
        skip_patterns = [
            "pos_embed",
            "norm",
            "bias",
            "time_embed",
            "label_emb",
        ]
    
    stats = {
        "total_tensors": 0,
        "converted_tensors": 0,
        "skipped_tensors": 0,
        "original_size_mb": 0,
        "fp8_size_mb": 0,
        "scales": {},
    }
    
    # 查找所有权重文件
    if convert_safetensors_only:
        weight_files = list(input_path.glob("*.safetensors"))
    else:
        weight_files = list(input_path.glob("*.safetensors")) + list(input_path.glob("*.bin"))
    
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {input_dir}")
    
    logger.info(f"Found {len(weight_files)} weight files")
    
    for weight_file in weight_files:
        logger.info(f"Converting: {weight_file.name}")
        
        # 加载权重
        if weight_file.suffix == ".safetensors":
            state_dict = load_file(str(weight_file))
        else:
            state_dict = torch.load(str(weight_file), map_location="cpu")
        
        converted_dict = {}
        file_scales = {}
        
        for name, tensor in tqdm(state_dict.items(), desc="  Tensors"):
            stats["total_tensors"] += 1
            original_size = tensor.numel() * tensor.element_size()
            stats["original_size_mb"] += original_size / (1024 * 1024)
            
            # 检查是否跳过
            should_skip = any(pattern in name.lower() for pattern in skip_patterns)
            
            # 只转换浮点类型且维度 >= 2 的权重 (跳过 1D 向量如 bias)
            is_convertible = (
                tensor.is_floating_point() and 
                tensor.dim() >= 2 and
                not should_skip
            )
            
            if is_convertible:
                fp8_tensor, scale = convert_tensor_to_fp8(
                    tensor, 
                    fp8_format=fp8_format, 
                    scale_method=scale_method
                )
                converted_dict[name] = fp8_tensor
                file_scales[name] = scale
                stats["converted_tensors"] += 1
                stats["fp8_size_mb"] += fp8_tensor.numel() * 1 / (1024 * 1024)  # FP8 = 1 byte
            else:
                # 保持原精度
                converted_dict[name] = tensor
                stats["skipped_tensors"] += 1
                stats["fp8_size_mb"] += original_size / (1024 * 1024)
        
        # 保存转换后的权重
        output_file = output_path / weight_file.name
        save_file(converted_dict, str(output_file))
        logger.info(f"  Saved: {output_file}")
        
        # 保存缩放因子 (用于推理时反量化)
        scale_file = output_path / f"{weight_file.stem}_scales.json"
        with open(scale_file, "w") as f:
            json.dump(file_scales, f, indent=2)
        logger.info(f"  Scales: {scale_file}")
        
        stats["scales"][weight_file.name] = file_scales
    
    # 复制配置文件
    for config_file in input_path.glob("*.json"):
        if "scale" not in config_file.name:
            import shutil
            shutil.copy(config_file, output_path / config_file.name)
            logger.info(f"Copied: {config_file.name}")
    
    # 保存转换元数据
    meta = {
        "fp8_format": fp8_format,
        "scale_method": scale_method,
        "skip_patterns": skip_patterns,
        "pytorch_version": torch.__version__,
        "stats": {
            "total_tensors": stats["total_tensors"],
            "converted_tensors": stats["converted_tensors"],
            "skipped_tensors": stats["skipped_tensors"],
            "compression_ratio": stats["original_size_mb"] / max(stats["fp8_size_mb"], 1),
        }
    }
    
    with open(output_path / "fp8_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return stats


def verify_fp8_support():
    """验证 PyTorch FP8 支持"""
    if not hasattr(torch, "float8_e4m3fn"):
        logger.error("PyTorch FP8 not supported. Requires PyTorch 2.1+")
        logger.info(f"Current PyTorch version: {torch.__version__}")
        sys.exit(1)
    
    # 检查 CUDA 支持
    if torch.cuda.is_available():
        # FP8 需要 SM89+ (Ada Lovelace / Hopper)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cc = props.major * 10 + props.minor
            if cc < 89:
                logger.warning(
                    f"GPU {i} ({props.name}) compute capability {props.major}.{props.minor} "
                    f"< 8.9, FP8 acceleration may not be available. "
                    f"Conversion will still work for CPU inference."
                )


def main():
    parser = argparse.ArgumentParser(description="Convert BF16 model to FP8")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input model directory (BF16)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for FP8 model")
    parser.add_argument("--fp8_format", type=str, default="e4m3",
                        choices=["e4m3", "e5m2"],
                        help="FP8 format: e4m3 (high precision) or e5m2 (large range)")
    parser.add_argument("--scale_method", type=str, default="dynamic",
                        choices=["dynamic", "fixed"],
                        help="Scale method: dynamic (per-tensor) or fixed")
    parser.add_argument("--skip_patterns", type=str, nargs="*",
                        default=["pos_embed", "norm", "bias", "time_embed"],
                        help="Weight name patterns to skip (keep original precision)")
    
    args = parser.parse_args()
    
    # 验证 FP8 支持
    verify_fp8_support()
    
    logger.info("=" * 60)
    logger.info("BF16 → FP8 Converter")
    logger.info("=" * 60)
    logger.info(f"Input:  {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Format: FP8-{args.fp8_format.upper()}")
    logger.info(f"Scale:  {args.scale_method}")
    logger.info(f"Skip:   {args.skip_patterns}")
    logger.info("=" * 60)
    
    # 执行转换
    stats = convert_model_to_fp8(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fp8_format=args.fp8_format,
        scale_method=args.scale_method,
        skip_patterns=args.skip_patterns,
    )
    
    # 打印统计
    logger.info("\n" + "=" * 60)
    logger.info("Conversion Complete!")
    logger.info("=" * 60)
    logger.info(f"Total tensors:     {stats['total_tensors']}")
    logger.info(f"Converted (FP8):   {stats['converted_tensors']}")
    logger.info(f"Skipped (orig):    {stats['skipped_tensors']}")
    logger.info(f"Original size:     {stats['original_size_mb']:.1f} MB")
    logger.info(f"FP8 size:          {stats['fp8_size_mb']:.1f} MB")
    compression = stats['original_size_mb'] / max(stats['fp8_size_mb'], 1)
    logger.info(f"Compression ratio: {compression:.2f}x")


if __name__ == "__main__":
    main()
