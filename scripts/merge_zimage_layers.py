#!/usr/bin/env python3
"""
Z-Image 模型分层融合独立脚本
无需 ComfyUI 环境，直接命令行运行

用法:
    python merge_zimage_layers.py model_a.safetensors model_b.safetensors -o merged.safetensors
    python merge_zimage_layers.py model_a model_b --layers 0-14 --ratio 0.7 -o merged
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file


class ZImageLayerMerger:
    """Z-Image 模型分层融合器"""
    
    # 模型结构常量
    N_LAYERS = 30
    N_REFINER_LAYERS = 2
    
    LAYER_GROUPS = {
        "main": "layers",
        "noise_refiner": "noise_refiner", 
        "context_refiner": "context_refiner",
        "embedder": ["all_x_embedder", "cap_embedder", "t_embedder", "x_pad_token", "cap_pad_token"],
        "final": "all_final_layer",
    }
    
    BLOCK_TYPES = {
        "qkv": ["to_q", "to_k", "to_v"],
        "out": ["to_out"],
        "attn_norm": ["attention_norm", "norm_q", "norm_k"],
        "ffn": ["feed_forward"],
        "ffn_norm": ["ffn_norm"],
        "adaln": ["adaLN_modulation"],
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def load_model(self, path: str) -> Dict[str, torch.Tensor]:
        """加载模型 (支持 safetensors 和 diffusers 目录)"""
        path = Path(path)
        
        if path.is_dir():
            # diffusers 格式
            model_file = path / "diffusion_pytorch_model.safetensors"
            if not model_file.exists():
                model_file = path / "model.safetensors"
            if not model_file.exists():
                raise FileNotFoundError(f"找不到模型文件: {path}")
            return load_file(str(model_file))
        else:
            return load_file(str(path))
    
    def _lerp(self, a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
        """线性插值"""
        return a + t * (b - a)
    
    def _slerp(self, a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
        """球面线性插值"""
        a_f = a.flatten().float()
        b_f = b.flatten().float()
        
        a_n = a_f / (a_f.norm() + 1e-8)
        b_n = b_f / (b_f.norm() + 1e-8)
        
        dot = torch.clamp(torch.dot(a_n, b_n), -1.0, 1.0)
        omega = torch.acos(dot)
        
        if omega.abs() < 1e-8:
            return self._lerp(a, b, t)
        
        sin_omega = torch.sin(omega)
        result = (torch.sin((1-t) * omega) / sin_omega) * a_f + \
                 (torch.sin(t * omega) / sin_omega) * b_f
        return result.view_as(a).to(a.dtype)
    
    def _get_layer_keys(self, sd: Dict, prefix: str, layer_idx: int) -> List[str]:
        """获取指定层的所有键"""
        pattern = f"{prefix}.{layer_idx}."
        return [k for k in sd.keys() if k.startswith(pattern)]
    
    def _matches_block(self, key: str, patterns: List[str]) -> bool:
        """检查键是否匹配模式"""
        return any(p in key for p in patterns)
    
    def merge(
        self,
        sd_a: Dict[str, torch.Tensor],
        sd_b: Dict[str, torch.Tensor],
        layer_range: Tuple[int, int] = (0, 29),
        ratio: float = 0.5,
        mode: str = "linear",
        gradient: str = "none",
        block_ratios: Optional[Dict[str, float]] = None,
        merge_refiner: bool = True,
        refiner_ratio: float = 0.5,
        merge_embedder: bool = False,
        embedder_ratio: float = 0.5,
        merge_final: bool = False,
        final_ratio: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        执行分层融合
        
        Args:
            sd_a: 基础模型 state_dict
            sd_b: 融合模型 state_dict
            layer_range: 主层范围 (start, end)
            ratio: 融合比例 (0=全A, 1=全B)
            mode: 插值模式 (linear/slerp)
            gradient: 渐变模式 (none/linear_in/linear_out/bell)
            block_ratios: 按模块类型的 ratio 覆盖
            merge_refiner: 是否融合 refiner 层
            refiner_ratio: refiner 融合比例
            merge_embedder: 是否融合嵌入层
            embedder_ratio: 嵌入层融合比例
            merge_final: 是否融合输出层
            final_ratio: 输出层融合比例
        
        Returns:
            融合后的 state_dict
        """
        # 从 A 复制所有权重
        merged = {k: v.clone() for k, v in sd_a.items()}
        
        merge_fn = self._slerp if mode == "slerp" else self._lerp
        start, end = layer_range
        stats = {"merged": 0, "skipped": 0}
        
        def compute_gradient_ratio(idx: int) -> float:
            if gradient == "none" or start == end:
                return ratio
            progress = (idx - start) / max(end - start, 1)
            if gradient == "linear_in":
                return ratio * progress
            elif gradient == "linear_out":
                return ratio * (1 - progress)
            elif gradient == "bell":
                import math
                return ratio * math.sin(progress * math.pi)
            return ratio
        
        # 1. 融合主 Transformer 层
        for layer_idx in range(start, end + 1):
            layer_ratio = compute_gradient_ratio(layer_idx)
            if layer_ratio <= 0:
                continue
            
            layer_keys = self._get_layer_keys(sd_a, "layers", layer_idx)
            for key in layer_keys:
                if key not in sd_b:
                    continue
                
                # 检查是否有 block 级别的覆盖
                actual_ratio = layer_ratio
                if block_ratios:
                    for block_type, patterns in self.BLOCK_TYPES.items():
                        if self._matches_block(key, patterns) and block_type in block_ratios:
                            actual_ratio = block_ratios[block_type]
                            break
                
                if actual_ratio > 0:
                    merged[key] = merge_fn(sd_a[key], sd_b[key], actual_ratio)
                    stats["merged"] += 1
        
        # 2. 融合 Refiner 层
        if merge_refiner and refiner_ratio > 0:
            for prefix in ["noise_refiner", "context_refiner"]:
                for layer_idx in range(self.N_REFINER_LAYERS):
                    for key in self._get_layer_keys(sd_a, prefix, layer_idx):
                        if key in sd_b:
                            merged[key] = merge_fn(sd_a[key], sd_b[key], refiner_ratio)
                            stats["merged"] += 1
        
        # 3. 融合嵌入层
        if merge_embedder and embedder_ratio > 0:
            for prefix in self.LAYER_GROUPS["embedder"]:
                for key in sd_a.keys():
                    if key.startswith(prefix) and key in sd_b:
                        merged[key] = merge_fn(sd_a[key], sd_b[key], embedder_ratio)
                        stats["merged"] += 1
        
        # 4. 融合输出层
        if merge_final and final_ratio > 0:
            for key in sd_a.keys():
                if key.startswith("all_final_layer") and key in sd_b:
                    merged[key] = merge_fn(sd_a[key], sd_b[key], final_ratio)
                    stats["merged"] += 1
        
        self.log(f"融合完成: {stats['merged']} tensors")
        return merged
    
    def save(self, sd: Dict[str, torch.Tensor], path: str, dtype: Optional[str] = None):
        """保存模型"""
        if dtype:
            dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            target_dtype = dtype_map.get(dtype)
            if target_dtype:
                sd = {k: v.to(target_dtype) for k, v in sd.items()}
        
        save_file(sd, path)
        self.log(f"已保存: {path}")


def parse_layer_range(s: str) -> Tuple[int, int]:
    """解析层范围字符串 (e.g., '0-14', '15')"""
    if "-" in s:
        parts = s.split("-")
        return int(parts[0]), int(parts[1])
    else:
        idx = int(s)
        return idx, idx


def parse_block_ratios(args: List[str]) -> Dict[str, float]:
    """解析 block ratio 参数 (e.g., 'qkv=0.7', 'ffn=0.3')"""
    result = {}
    for arg in args:
        if "=" in arg:
            block, ratio = arg.split("=")
            result[block] = float(ratio)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image 模型分层融合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础融合 (全部层 50%%)
  python merge_zimage_layers.py model_a.safetensors model_b.safetensors -o merged.safetensors
  
  # 只融合前半部分层
  python merge_zimage_layers.py model_a model_b --layers 0-14 --ratio 0.7 -o merged.safetensors
  
  # 渐变融合 (前面层用 A，后面层渐变到 B)
  python merge_zimage_layers.py model_a model_b --gradient linear_in --ratio 1.0 -o merged.safetensors
  
  # 按模块融合 (只融合 attention，不融合 FFN)
  python merge_zimage_layers.py model_a model_b --block qkv=0.7 out=0.7 ffn=0 -o merged.safetensors
        """
    )
    
    parser.add_argument("model_a", help="基础模型 (safetensors 或 diffusers 目录)")
    parser.add_argument("model_b", help="融合模型")
    parser.add_argument("-o", "--output", required=True, help="输出路径")
    parser.add_argument("--layers", default="0-29", help="层范围 (默认: 0-29)")
    parser.add_argument("--ratio", type=float, default=0.5, help="融合比例 (默认: 0.5)")
    parser.add_argument("--mode", choices=["linear", "slerp"], default="linear", help="插值模式")
    parser.add_argument("--gradient", choices=["none", "linear_in", "linear_out", "bell"], 
                        default="none", help="渐变模式")
    parser.add_argument("--block", nargs="*", default=[], help="按模块 ratio (e.g., qkv=0.7 ffn=0.3)")
    parser.add_argument("--no-refiner", action="store_true", help="不融合 refiner 层")
    parser.add_argument("--refiner-ratio", type=float, default=None, help="refiner 融合比例")
    parser.add_argument("--merge-embedder", action="store_true", help="融合嵌入层")
    parser.add_argument("--embedder-ratio", type=float, default=0.5, help="嵌入层融合比例")
    parser.add_argument("--merge-final", action="store_true", help="融合输出层")
    parser.add_argument("--final-ratio", type=float, default=0.5, help="输出层融合比例")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], help="输出数据类型")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    
    args = parser.parse_args()
    
    merger = ZImageLayerMerger(verbose=not args.quiet)
    
    # 加载模型
    print(f"加载 Model A: {args.model_a}")
    sd_a = merger.load_model(args.model_a)
    print(f"加载 Model B: {args.model_b}")
    sd_b = merger.load_model(args.model_b)
    
    # 解析参数
    layer_range = parse_layer_range(args.layers)
    block_ratios = parse_block_ratios(args.block) if args.block else None
    refiner_ratio = args.refiner_ratio if args.refiner_ratio is not None else args.ratio
    
    print(f"\n配置:")
    print(f"  层范围: {layer_range[0]}-{layer_range[1]}")
    print(f"  融合比例: {args.ratio}")
    print(f"  插值模式: {args.mode}")
    print(f"  渐变模式: {args.gradient}")
    if block_ratios:
        print(f"  模块覆盖: {block_ratios}")
    print(f"  Refiner: {'禁用' if args.no_refiner else f'ratio={refiner_ratio}'}")
    
    # 执行融合
    merged = merger.merge(
        sd_a, sd_b,
        layer_range=layer_range,
        ratio=args.ratio,
        mode=args.mode,
        gradient=args.gradient,
        block_ratios=block_ratios,
        merge_refiner=not args.no_refiner,
        refiner_ratio=refiner_ratio,
        merge_embedder=args.merge_embedder,
        embedder_ratio=args.embedder_ratio,
        merge_final=args.merge_final,
        final_ratio=args.final_ratio,
    )
    
    # 保存
    merger.save(merged, args.output, args.dtype)
    print(f"\n✅ 完成!")


if __name__ == "__main__":
    main()
