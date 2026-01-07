# -*- coding: utf-8 -*-
"""
Image Generation Core Module

统一的图像生成核心逻辑，抽取自 generation.py
"""

from typing import Optional, List, Callable, Any, Dict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import io
import base64

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from PIL import Image


@dataclass
class GenerationParams:
    """生成参数"""
    prompt: str
    model_type: str = "zimage"
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 9
    guidance_scale: float = 1.0
    seed: int = -1
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    comparison_mode: bool = False


@dataclass
class GeneratedImage:
    """生成结果"""
    image: Image.Image
    seed: int
    timestamp: str
    filename: str
    filepath: Path
    metadata: Dict[str, Any]
    base64: Optional[str] = None


class LoRAManager:
    """LoRA 统一管理器"""
    
    def __init__(self):
        self._current_loras: Dict[str, Optional[str]] = {}
    
    def get_current(self, model_type: str) -> Optional[str]:
        """获取当前加载的 LoRA 路径"""
        return self._current_loras.get(model_type)
    
    def load(self, pipe: Any, lora_path: str, model_type: str) -> bool:
        """加载 LoRA"""
        current = self.get_current(model_type)
        
        if current == lora_path:
            return True  # 已加载
        
        # 先卸载旧的
        if current:
            self.unload(pipe, model_type)
        
        # 加载新的
        try:
            pipe.load_lora_weights(lora_path)
            self._current_loras[model_type] = lora_path
            print(f"[LoRA] Loaded: {lora_path}")
            return True
        except Exception as e:
            print(f"[LoRA] Failed to load {lora_path}: {e}")
            return False
    
    def unload(self, pipe: Any, model_type: str) -> bool:
        """卸载 LoRA"""
        current = self.get_current(model_type)
        if not current:
            return True
        
        try:
            pipe.unload_lora_weights()
            self._current_loras[model_type] = None
            print(f"[LoRA] Unloaded: {current}")
            return True
        except Exception as e:
            print(f"[LoRA] Failed to unload: {e}")
            return False
    
    def set_scale(self, pipe: Any, scale: float):
        """设置 LoRA 权重"""
        pipe.cross_attention_kwargs = {"scale": scale}
    
    def clear_scale(self, pipe: Any):
        """清除 LoRA 权重设置"""
        pipe.cross_attention_kwargs = None


class ImageGenerator:
    """图像生成器核心类"""
    
    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir
        self.lora_manager = LoRAManager()
        self._pipelines: Dict[str, Any] = {}
    
    def get_pipeline(self, model_type: str) -> Optional[Any]:
        """获取已缓存的 pipeline"""
        return self._pipelines.get(model_type)
    
    def set_pipeline(self, model_type: str, pipe: Any):
        """缓存 pipeline"""
        self._pipelines[model_type] = pipe
    
    def create_generator(self, seed: int = -1) -> tuple:
        """创建随机数生成器，返回 (generator, actual_seed)"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device)
        
        if seed != -1:
            generator.manual_seed(seed)
            return generator, seed
        else:
            actual_seed = generator.seed()
            return generator, actual_seed
    
    def _run_pipeline(
        self, 
        pipe: Any, 
        params: GenerationParams, 
        generator: Any,
        progress_callback: Optional[Callable] = None
    ) -> Image.Image:
        """执行 pipeline 推理"""
        model_type = params.model_type.lower()
        
        # 根据模型类型调用不同的生成接口
        kwargs = {
            "prompt": params.prompt,
            "num_inference_steps": params.steps,
            "guidance_scale": params.guidance_scale,
            "width": params.width,
            "height": params.height,
            "generator": generator,
        }
        
        if model_type == "zimage":
            kwargs["negative_prompt"] = params.negative_prompt
        
        if progress_callback:
            kwargs["callback_on_step_end"] = progress_callback
        
        return pipe(**kwargs).images[0]
    
    def _save_result(
        self, 
        image: Image.Image, 
        params: GenerationParams, 
        seed: int,
        include_base64: bool = True
    ) -> GeneratedImage:
        """保存生成结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        generated_dir = self.outputs_dir / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{timestamp}.png"
        filepath = generated_dir / filename
        image.save(filepath)
        
        # 保存元数据
        metadata = {
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "model_type": params.model_type,
            "width": params.width,
            "height": params.height,
            "steps": params.steps,
            "guidance_scale": params.guidance_scale,
            "seed": seed,
            "lora_path": params.lora_path,
            "lora_scale": params.lora_scale,
            "comparison_mode": params.comparison_mode,
            "timestamp": timestamp,
        }
        
        with open(generated_dir / f"{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Base64 编码
        img_base64 = None
        if include_base64:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return GeneratedImage(
            image=image,
            seed=seed,
            timestamp=timestamp,
            filename=filename,
            filepath=filepath,
            metadata=metadata,
            base64=img_base64
        )
    
    def generate(
        self,
        pipe: Any,
        params: GenerationParams,
        progress_callback: Optional[Callable] = None,
    ) -> GeneratedImage:
        """执行单次生成"""
        model_type = params.model_type.lower()
        
        # 处理 LoRA
        if params.lora_path:
            self.lora_manager.load(pipe, params.lora_path, model_type)
            self.lora_manager.set_scale(pipe, params.lora_scale)
        else:
            self.lora_manager.unload(pipe, model_type)
        
        # 创建 generator
        generator, actual_seed = self.create_generator(params.seed)
        
        # 生成
        image = self._run_pipeline(pipe, params, generator, progress_callback)
        
        # 清理 LoRA scale
        self.lora_manager.clear_scale(pipe)
        
        # 保存并返回
        return self._save_result(image, params, actual_seed)
    
    def generate_comparison(
        self,
        pipe: Any,
        params: GenerationParams,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """生成对比图（无 LoRA vs 有 LoRA）
        
        返回:
            {
                "images": [{"image": base64, "lora_path": None}, {"image": base64, "lora_path": "xxx", "lora_scale": 1.0}],
                "composite": GeneratedImage,  # 拼接后的图片（用于历史记录）
                "seed": int
            }
        """
        if not params.lora_path:
            # 没有 LoRA，只生成一张图
            result = self.generate(pipe, params, progress_callback)
            return {
                "images": [{"image": result.base64, "lora_path": None, "lora_scale": None}],
                "composite": result,
                "seed": result.seed
            }
        
        model_type = params.model_type.lower()
        
        # 使用相同的 seed
        _, base_seed = self.create_generator(params.seed)
        
        # 1. 先生成无 LoRA 的原图
        self.lora_manager.unload(pipe, model_type)
        generator_no_lora, _ = self.create_generator(base_seed)
        image_no_lora = self._run_pipeline(pipe, params, generator_no_lora, progress_callback)
        
        # 2. 生成有 LoRA 的图
        self.lora_manager.load(pipe, params.lora_path, model_type)
        self.lora_manager.set_scale(pipe, params.lora_scale)
        generator_with_lora, _ = self.create_generator(base_seed)
        image_with_lora = self._run_pipeline(pipe, params, generator_with_lora, progress_callback)
        self.lora_manager.clear_scale(pipe)
        
        # 3. 将两张图编码为 base64（用于前端展示）
        def image_to_base64(img: Image.Image) -> str:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        
        images_data = [
            {"image": image_to_base64(image_no_lora), "lora_path": None, "lora_scale": None},
            {"image": image_to_base64(image_with_lora), "lora_path": params.lora_path, "lora_scale": params.lora_scale}
        ]
        
        # 4. 拼接两张图（左右并排）用于历史记录
        composite_width = image_no_lora.width + image_with_lora.width + 10  # 10px 间隔
        composite_height = max(image_no_lora.height, image_with_lora.height)
        composite = Image.new('RGB', (composite_width, composite_height), (30, 30, 30))  # 深灰背景
        composite.paste(image_no_lora, (0, 0))
        composite.paste(image_with_lora, (image_no_lora.width + 10, 0))
        
        # 5. 保存拼接图到历史记录（只保存一条）
        params_composite = GenerationParams(
            prompt=params.prompt,
            model_type=params.model_type,
            negative_prompt=params.negative_prompt,
            width=composite_width,
            height=composite_height,
            steps=params.steps,
            guidance_scale=params.guidance_scale,
            seed=base_seed,
            lora_path=params.lora_path,
            lora_scale=params.lora_scale,
            comparison_mode=True,
        )
        composite_result = self._save_result(composite, params_composite, base_seed)
        
        return {
            "images": images_data,
            "composite": composite_result,
            "seed": base_seed
        }


# 全局实例（便于使用）
_generator_instance: Optional[ImageGenerator] = None


def get_generator(outputs_dir: Path) -> ImageGenerator:
    """获取全局生成器实例"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ImageGenerator(outputs_dir)
    return _generator_instance
