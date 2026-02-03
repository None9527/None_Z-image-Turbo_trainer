#!/usr/bin/env python3
"""
合并分片 safetensors 模型为单个 bf16 文件

用法:
    python merge_sharded_model.py /path/to/model_dir -o merged.safetensors
    python merge_sharded_model.py /path/to/model_dir --dtype fp16
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def merge_sharded_model(
    model_dir: str,
    output_path: str,
    dtype: str = "bf16",
    copy_config: bool = True,
) -> None:
    """
    合并分片模型为单个文件
    
    Args:
        model_dir: 包含分片文件和 model.safetensors.index.json 的目录
        output_path: 输出文件路径
        dtype: 输出数据类型 (bf16/fp16/fp32)
        copy_config: 是否复制 config.json
    """
    model_dir = Path(model_dir)
    
    # 读取 index 文件
    index_file = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index_file.exists():
        index_file = model_dir / "model.safetensors.index.json"
    
    if not index_file.exists():
        raise FileNotFoundError(f"找不到 index 文件: {index_file}")
    
    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    
    weight_map = index_data["weight_map"]
    total_size = index_data["metadata"]["total_size"]
    
    # 获取所有分片文件
    shard_files = sorted(set(weight_map.values()))
    print(f"发现 {len(shard_files)} 个分片文件:")
    for sf in shard_files:
        print(f"  - {sf}")
    
    # 目标数据类型
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map.get(dtype, torch.bfloat16)
    print(f"\n目标数据类型: {dtype} ({target_dtype})")
    print(f"预计原始大小: {total_size / 1e9:.2f} GB")
    
    # 加载并合并
    merged_state_dict: Dict[str, torch.Tensor] = {}
    
    for shard_name in tqdm(shard_files, desc="加载分片"):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"分片文件不存在: {shard_path}")
        
        shard_data = load_file(str(shard_path))
        
        for key, tensor in shard_data.items():
            # 转换数据类型
            if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
                tensor = tensor.to(target_dtype)
            merged_state_dict[key] = tensor
    
    print(f"\n合并完成: {len(merged_state_dict)} 个张量")
    
    # 验证所有权重都已加载
    missing_keys = set(weight_map.keys()) - set(merged_state_dict.keys())
    if missing_keys:
        print(f"⚠️ 警告: 缺少 {len(missing_keys)} 个权重:")
        for k in list(missing_keys)[:5]:
            print(f"  - {k}")
        if len(missing_keys) > 5:
            print(f"  ... 及其他 {len(missing_keys) - 5} 个")
    
    # 计算输出大小
    output_size = sum(t.numel() * t.element_size() for t in merged_state_dict.values())
    print(f"输出大小: {output_size / 1e9:.2f} GB")
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存到: {output_path}")
    save_file(merged_state_dict, str(output_path))
    
    # 复制 config.json
    if copy_config:
        config_src = model_dir / "config.json"
        if config_src.exists():
            config_dst = output_path.parent / "config.json"
            import shutil
            shutil.copy(config_src, config_dst)
            print(f"已复制 config.json -> {config_dst}")
    
    print("\n✅ 完成!")


def main():
    parser = argparse.ArgumentParser(
        description="合并分片 safetensors 模型为单个文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合并为 bf16 (默认)
  python merge_sharded_model.py ./z_image_transformer -o merged_bf16.safetensors
  
  # 合并为 fp16
  python merge_sharded_model.py ./z_image_transformer -o merged_fp16.safetensors --dtype fp16
  
  # 保存到特定目录
  python merge_sharded_model.py ./z_image_transformer -o ./output/model.safetensors
        """
    )
    
    parser.add_argument("model_dir", help="包含分片模型的目录")
    parser.add_argument("-o", "--output", required=True, help="输出文件路径")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="输出数据类型 (默认: bf16)")
    parser.add_argument("--no-config", action="store_true", help="不复制 config.json")
    
    args = parser.parse_args()
    
    merge_sharded_model(
        model_dir=args.model_dir,
        output_path=args.output,
        dtype=args.dtype,
        copy_config=not args.no_config,
    )


if __name__ == "__main__":
    main()
