"""
Z-Image Layer Merge Node for ComfyUI
实现 ZImageTransformer2DModel 的分层融合

模型结构 (基于 config.json):
- n_layers: 30 (主 Transformer)
- n_refiner_layers: 2 (noise_refiner + context_refiner)
- dim: 3840, n_heads: 30

每层包含：
- attention (to_q, to_k, to_v, to_out)
- attention_norm1, attention_norm2
- feed_forward (w1, w2, w3)
- ffn_norm1, ffn_norm2
- adaLN_modulation
"""

import torch
import copy
from typing import Dict, List, Tuple, Optional


class ZImageLayerMerge:
    """
    Z-Image 模型分层融合节点
    支持按层范围选择性地从 model_B 合并到 model_A
    """
    
    LAYER_GROUPS = {
        "main_transformer": {
            "prefix": "layers",
            "count": 30,
            "description": "主 Transformer 层 (0-29)"
        },
        "noise_refiner": {
            "prefix": "noise_refiner", 
            "count": 2,
            "description": "噪声精炼层 (0-1)"
        },
        "context_refiner": {
            "prefix": "context_refiner",
            "count": 2,
            "description": "上下文精炼层 (0-1)"
        },
        "embedder": {
            "prefixes": ["all_x_embedder", "cap_embedder", "t_embedder"],
            "description": "嵌入层"
        },
        "final": {
            "prefix": "all_final_layer",
            "description": "输出层"
        },
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "merge_mode": (["linear", "weighted_sum", "slerp"],),
            },
            "optional": {
                # 主 Transformer 层控制
                "main_start": ("INT", {"default": 0, "min": 0, "max": 29, "step": 1}),
                "main_end": ("INT", {"default": 29, "min": 0, "max": 29, "step": 1}),
                "main_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Refiner 层控制
                "merge_noise_refiner": ("BOOLEAN", {"default": True}),
                "noise_refiner_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "merge_context_refiner": ("BOOLEAN", {"default": True}),
                "context_refiner_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # 嵌入层和输出层控制
                "merge_embedders": ("BOOLEAN", {"default": False}),
                "embedder_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "merge_final": ("BOOLEAN", {"default": False}),
                "final_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # 渐变模式
                "gradient_mode": (["none", "linear_in", "linear_out", "bell"],),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge_layers"
    CATEGORY = "Z-Image/Merge"
    
    def _lerp(self, a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
        """线性插值: (1-t)*a + t*b"""
        return a + t * (b - a)
    
    def _slerp(self, a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
        """球面线性插值 (用于归一化权重)"""
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        
        a_norm = a_flat / (a_flat.norm() + eps)
        b_norm = b_flat / (b_flat.norm() + eps)
        
        dot = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
        omega = torch.acos(dot)
        
        if omega.abs() < eps:
            return self._lerp(a, b, t)
        
        sin_omega = torch.sin(omega)
        result = (torch.sin((1 - t) * omega) / sin_omega) * a_flat + \
                 (torch.sin(t * omega) / sin_omega) * b_flat
        
        return result.view_as(a).to(a.dtype)
    
    def _merge_tensor(self, a: torch.Tensor, b: torch.Tensor, ratio: float, mode: str) -> torch.Tensor:
        """根据模式合并两个张量"""
        if mode == "linear" or mode == "weighted_sum":
            return self._lerp(a, b, ratio)
        elif mode == "slerp":
            return self._slerp(a, b, ratio)
        else:
            return self._lerp(a, b, ratio)
    
    def _compute_gradient_ratio(self, layer_idx: int, start: int, end: int, 
                                  base_ratio: float, mode: str) -> float:
        """计算渐变模式下的实际 ratio"""
        if mode == "none" or start == end:
            return base_ratio
        
        # 归一化位置 [0, 1]
        progress = (layer_idx - start) / max(end - start, 1)
        
        if mode == "linear_in":
            # 从 0 渐变到 base_ratio
            return base_ratio * progress
        elif mode == "linear_out":
            # 从 base_ratio 渐变到 0
            return base_ratio * (1 - progress)
        elif mode == "bell":
            # 钟形曲线: 中间最大
            import math
            return base_ratio * math.sin(progress * math.pi)
        
        return base_ratio
    
    def _get_layer_keys(self, state_dict: Dict, prefix: str, layer_idx: int) -> List[str]:
        """获取指定层的所有键"""
        pattern = f"{prefix}.{layer_idx}."
        return [k for k in state_dict.keys() if k.startswith(pattern)]
    
    def merge_layers(self, model_a, model_b, merge_mode: str,
                     main_start: int = 0, main_end: int = 29, main_ratio: float = 0.5,
                     merge_noise_refiner: bool = True, noise_refiner_ratio: float = 0.5,
                     merge_context_refiner: bool = True, context_refiner_ratio: float = 0.5,
                     merge_embedders: bool = False, embedder_ratio: float = 0.5,
                     merge_final: bool = False, final_ratio: float = 0.5,
                     gradient_mode: str = "none"):
        """执行分层融合"""
        
        # Clone model_a
        merged = model_a.clone()
        
        # 获取 state_dict
        sd_a = model_a.model.state_dict()
        sd_b = model_b.model.state_dict()
        
        merged_sd = {}
        merge_log = []
        
        # 1. 合并主 Transformer 层
        for layer_idx in range(main_start, main_end + 1):
            ratio = self._compute_gradient_ratio(layer_idx, main_start, main_end, 
                                                   main_ratio, gradient_mode)
            if ratio > 0:
                layer_keys = self._get_layer_keys(sd_a, "layers", layer_idx)
                for key in layer_keys:
                    if key in sd_b:
                        merged_sd[key] = self._merge_tensor(sd_a[key], sd_b[key], ratio, merge_mode)
                merge_log.append(f"layers.{layer_idx}: ratio={ratio:.3f}")
        
        # 2. 合并 Refiner 层
        if merge_noise_refiner:
            for layer_idx in range(2):
                layer_keys = self._get_layer_keys(sd_a, "noise_refiner", layer_idx)
                for key in layer_keys:
                    if key in sd_b:
                        merged_sd[key] = self._merge_tensor(sd_a[key], sd_b[key], 
                                                            noise_refiner_ratio, merge_mode)
            merge_log.append(f"noise_refiner: ratio={noise_refiner_ratio:.3f}")
        
        if merge_context_refiner:
            for layer_idx in range(2):
                layer_keys = self._get_layer_keys(sd_a, "context_refiner", layer_idx)
                for key in layer_keys:
                    if key in sd_b:
                        merged_sd[key] = self._merge_tensor(sd_a[key], sd_b[key], 
                                                            context_refiner_ratio, merge_mode)
            merge_log.append(f"context_refiner: ratio={context_refiner_ratio:.3f}")
        
        # 3. 合并嵌入层
        if merge_embedders:
            for prefix in ["all_x_embedder", "cap_embedder", "t_embedder"]:
                embedder_keys = [k for k in sd_a.keys() if k.startswith(prefix)]
                for key in embedder_keys:
                    if key in sd_b:
                        merged_sd[key] = self._merge_tensor(sd_a[key], sd_b[key], 
                                                            embedder_ratio, merge_mode)
            merge_log.append(f"embedders: ratio={embedder_ratio:.3f}")
        
        # 4. 合并输出层
        if merge_final:
            final_keys = [k for k in sd_a.keys() if k.startswith("all_final_layer")]
            for key in final_keys:
                if key in sd_b:
                    merged_sd[key] = self._merge_tensor(sd_a[key], sd_b[key], 
                                                        final_ratio, merge_mode)
            merge_log.append(f"final_layer: ratio={final_ratio:.3f}")
        
        # 应用合并的权重
        if merged_sd:
            merged.model.load_state_dict(merged_sd, strict=False)
            print(f"[ZImage Layer Merge] 合并完成:\n" + "\n".join(f"  - {log}" for log in merge_log))
        
        return (merged,)


class ZImageLayerMergeAdvanced:
    """
    高级分层融合节点
    支持逐层精细控制 ratio
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "merge_mode": (["linear", "slerp"],),
            },
            "optional": {}
        }
        
        # 为每一层添加独立的 ratio 控制
        for i in range(30):
            inputs["optional"][f"layer_{i:02d}"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
        
        return inputs
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge_layers"
    CATEGORY = "Z-Image/Merge"
    
    def merge_layers(self, model_a, model_b, merge_mode: str, **layer_ratios):
        """执行逐层融合"""
        merged = model_a.clone()
        sd_a = model_a.model.state_dict()
        sd_b = model_b.model.state_dict()
        
        merged_sd = {}
        
        for i in range(30):
            ratio = layer_ratios.get(f"layer_{i:02d}", 0.5)
            if ratio > 0:
                pattern = f"layers.{i}."
                layer_keys = [k for k in sd_a.keys() if k.startswith(pattern)]
                for key in layer_keys:
                    if key in sd_b:
                        a, b = sd_a[key], sd_b[key]
                        merged_sd[key] = a + ratio * (b - a)
        
        if merged_sd:
            merged.model.load_state_dict(merged_sd, strict=False)
        
        return (merged,)


class ZImageLayerMergeByBlock:
    """
    按模块类型融合节点
    可以分别控制 attention、FFN、norm 等模块的融合比例
    """
    
    BLOCK_TYPES = {
        "attention_qkv": ["to_q", "to_k", "to_v"],
        "attention_out": ["to_out"],
        "attention_norm": ["attention_norm1", "attention_norm2", "norm_q", "norm_k"],
        "ffn": ["feed_forward"],
        "ffn_norm": ["ffn_norm1", "ffn_norm2"],
        "adaln": ["adaLN_modulation"],
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "layer_start": ("INT", {"default": 0, "min": 0, "max": 29}),
                "layer_end": ("INT", {"default": 29, "min": 0, "max": 29}),
            },
            "optional": {
                "attention_qkv_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attention_out_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attention_norm_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ffn_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ffn_norm_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adaln_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge_by_block"
    CATEGORY = "Z-Image/Merge"
    
    def _matches_block(self, key: str, patterns: List[str]) -> bool:
        """检查键是否匹配任一模式"""
        return any(p in key for p in patterns)
    
    def merge_by_block(self, model_a, model_b, layer_start: int, layer_end: int,
                       attention_qkv_ratio: float = 0.5,
                       attention_out_ratio: float = 0.5,
                       attention_norm_ratio: float = 0.0,
                       ffn_ratio: float = 0.5,
                       ffn_norm_ratio: float = 0.0,
                       adaln_ratio: float = 0.5):
        """按模块类型融合"""
        merged = model_a.clone()
        sd_a = model_a.model.state_dict()
        sd_b = model_b.model.state_dict()
        
        block_ratios = {
            "attention_qkv": attention_qkv_ratio,
            "attention_out": attention_out_ratio,
            "attention_norm": attention_norm_ratio,
            "ffn": ffn_ratio,
            "ffn_norm": ffn_norm_ratio,
            "adaln": adaln_ratio,
        }
        
        merged_sd = {}
        stats = {k: 0 for k in block_ratios}
        
        for layer_idx in range(layer_start, layer_end + 1):
            pattern = f"layers.{layer_idx}."
            layer_keys = [k for k in sd_a.keys() if k.startswith(pattern)]
            
            for key in layer_keys:
                if key not in sd_b:
                    continue
                
                # 确定属于哪个 block 类型
                for block_type, patterns in self.BLOCK_TYPES.items():
                    if self._matches_block(key, patterns):
                        ratio = block_ratios[block_type]
                        if ratio > 0:
                            merged_sd[key] = sd_a[key] + ratio * (sd_b[key] - sd_a[key])
                            stats[block_type] += 1
                        break
        
        if merged_sd:
            merged.model.load_state_dict(merged_sd, strict=False)
            print(f"[ZImage Layer Merge By Block] 统计:")
            for block_type, count in stats.items():
                if count > 0:
                    print(f"  - {block_type}: {count} tensors, ratio={block_ratios[block_type]:.3f}")
        
        return (merged,)


class ZImageModelSave:
    """
    保存融合后的模型
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename": ("STRING", {"default": "merged_model"}),
                "save_format": (["safetensors", "diffusers"],),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_model"
    CATEGORY = "Z-Image/Merge"
    
    def save_model(self, model, filename: str, save_format: str):
        """保存模型"""
        import os
        from safetensors.torch import save_file
        
        output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        if save_format == "safetensors":
            output_path = os.path.join(output_dir, f"{filename}.safetensors")
            save_file(model.model.state_dict(), output_path)
            print(f"[ZImage] 模型已保存: {output_path}")
        elif save_format == "diffusers":
            output_path = os.path.join(output_dir, filename)
            model.model.save_pretrained(output_path)
            print(f"[ZImage] 模型已保存 (diffusers): {output_path}")
        
        return {}


# 节点注册
NODE_CLASS_MAPPINGS = {
    "ZImageLayerMerge": ZImageLayerMerge,
    "ZImageLayerMergeAdvanced": ZImageLayerMergeAdvanced,
    "ZImageLayerMergeByBlock": ZImageLayerMergeByBlock,
    "ZImageModelSave": ZImageModelSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLayerMerge": "Z-Image Layer Merge",
    "ZImageLayerMergeAdvanced": "Z-Image Layer Merge (Per-Layer)",
    "ZImageLayerMergeByBlock": "Z-Image Layer Merge (By Block)",
    "ZImageModelSave": "Z-Image Model Save",
}
