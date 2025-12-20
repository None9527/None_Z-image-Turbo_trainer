# -*- coding: utf-8 -*-
"""
Z-Image Finetune Inference Gradio UI

支持加载 Full Finetune 训练的 3.7G+ 模型权重，
并可实时调节权重混合比例（基础模型 vs 微调模型）。

Usage:
    python scripts/gradio_finetune_inference.py
"""

import os
import sys
import copy
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import gradio as gr
from PIL import Image
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL

# --- 全局配置 ---
DEFAULT_CONFIG = {
    "base_model": str(PROJECT_ROOT / "models" / "Z-Image-Turbo"),
    "vae_path": str(PROJECT_ROOT / "models" / "vae"),
    "text_encoder_path": str(PROJECT_ROOT / "models" / "qwen2_5_vl_3b"),
    "finetune_weights": "",  # 用户选择的 3.7G 模型
}

# 全局状态
_pipeline = None
_base_state_dict = None  # 基础模型权重缓存
_finetune_state_dict = None  # 微调权重缓存
_current_blend_ratio = 0.0


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype():
    device = get_device()
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_pipeline(base_model_path: str, vae_path: str, text_encoder_path: str):
    """加载基础 Pipeline"""
    global _pipeline, _base_state_dict
    
    device = get_device()
    dtype = get_dtype()
    
    print(f"[INFO] Loading pipeline on {device} with dtype {dtype}")
    print(f"[INFO] Base model: {base_model_path}")
    
    # 尝试使用 diffusers 原生加载
    try:
        from diffusers import ZImagePipeline
        
        pipe = ZImagePipeline.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
    except Exception as e:
        print(f"[WARN] diffusers native load failed: {e}")
        print("[INFO] Attempting manual component loading...")
        
        # 手动加载组件
        from zimage_trainer.utils.zimage_utils import (
            load_transformer,
            load_text_encoder_and_tokenizer,
            load_scheduler,
        )
        from zimage_trainer.utils.vae_utils import load_vae
        from zimage_trainer.z_image.pipeline_z_image import ZImagePipeline
        
        transformer = load_transformer(base_model_path, device=device, torch_dtype=dtype)
        vae = load_vae(vae_path, device=device, dtype=dtype)
        text_encoder, tokenizer = load_text_encoder_and_tokenizer(text_encoder_path, device=device)
        scheduler = load_scheduler("flow_match_euler", use_diffusers=True)
        
        pipe = ZImagePipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
    
    # 缓存基础模型权重
    _base_state_dict = {}
    for name, param in pipe.transformer.named_parameters():
        _base_state_dict[name] = param.data.clone().cpu()
    
    print(f"[INFO] Cached {len(_base_state_dict)} base model tensors")
    
    pipe.to(device)
    _pipeline = pipe
    
    return pipe


def load_finetune_weights(finetune_path: str):
    """加载微调权重"""
    global _finetune_state_dict
    
    if not finetune_path or not Path(finetune_path).exists():
        return None, "微调权重文件不存在"
    
    print(f"[INFO] Loading finetune weights: {finetune_path}")
    _finetune_state_dict = load_file(finetune_path)
    
    size_mb = sum(t.numel() * t.element_size() for t in _finetune_state_dict.values()) / 1024 / 1024
    print(f"[INFO] Loaded {len(_finetune_state_dict)} finetune tensors ({size_mb:.1f} MB)")
    
    return _finetune_state_dict, f"成功加载 {len(_finetune_state_dict)} 个张量 ({size_mb:.1f} MB)"


def blend_weights(blend_ratio: float):
    """混合基础权重和微调权重
    
    blend_ratio: 0.0 = 纯基础模型, 1.0 = 纯微调模型
    """
    global _pipeline, _base_state_dict, _finetune_state_dict, _current_blend_ratio
    
    if _pipeline is None:
        return "请先加载基础模型"
    
    if _finetune_state_dict is None:
        return "请先加载微调权重"
    
    _current_blend_ratio = blend_ratio
    device = get_device()
    dtype = get_dtype()
    
    print(f"[INFO] Blending weights: base={1-blend_ratio:.2f}, finetune={blend_ratio:.2f}")
    
    blended_count = 0
    with torch.no_grad():
        for name, param in _pipeline.transformer.named_parameters():
            if name in _finetune_state_dict and name in _base_state_dict:
                base_weight = _base_state_dict[name].to(device=device, dtype=dtype)
                finetune_weight = _finetune_state_dict[name].to(device=device, dtype=dtype)
                
                # 线性插值混合
                blended = (1 - blend_ratio) * base_weight + blend_ratio * finetune_weight
                param.data.copy_(blended)
                blended_count += 1
    
    return f"已混合 {blended_count} 个张量 (基础:{1-blend_ratio:.0%} + 微调:{blend_ratio:.0%})"


def generate_image(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    blend_ratio: float,
    shift: float = 3.0,
):
    """生成图像"""
    global _pipeline, _current_blend_ratio
    
    if _pipeline is None:
        return None, "请先加载模型"
    
    # 检查是否需要重新混合权重
    if _finetune_state_dict is not None and abs(blend_ratio - _current_blend_ratio) > 0.001:
        blend_weights(blend_ratio)
    
    # 设置随机种子
    device = get_device()
    if seed == -1:
        generator = torch.Generator(device=device)
        actual_seed = generator.seed()
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        actual_seed = seed
    
    print(f"[INFO] Generating: {width}x{height}, steps={steps}, cfg={guidance_scale}, seed={actual_seed}, shift={shift}")
    print(f"[INFO] Prompt: {prompt[:100]}...")
    
    # 应用 shift 参数到 scheduler
    if shift > 0:
        _pipeline.scheduler.config["base_shift"] = shift
        _pipeline.scheduler.config["max_shift"] = shift
    
    try:
        image = _pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        return image, f"生成成功！Seed: {actual_seed}"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"生成失败: {str(e)}"


def scan_finetune_models(directory: str):
    """扫描目录下的微调模型文件"""
    models = []
    
    if not directory:
        directory = str(PROJECT_ROOT / "output")
    
    path = Path(directory)
    if not path.exists():
        return models
    
    for f in path.rglob("*.safetensors"):
        size_mb = f.stat().st_size / 1024 / 1024
        # 只显示大于 100MB 的文件（排除 LoRA）
        if size_mb > 100:
            models.append({
                "path": str(f),
                "name": f.name,
                "size": f"{size_mb:.1f} MB"
            })
    
    return sorted(models, key=lambda x: x["name"])


# ============================================================
# Gradio UI
# ============================================================

def create_ui():
    with gr.Blocks(title="Z-Image Finetune Inference") as demo:
        gr.Markdown("# 🎨 Z-Image Finetune 推理工具")
        gr.Markdown("支持加载 Full Finetune 训练的模型权重，可实时调节基础模型与微调模型的混合比例。")
        
        with gr.Row():
            # 左侧：设置面板
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 模型设置")
                
                base_model_input = gr.Textbox(
                    label="基础模型路径",
                    value=DEFAULT_CONFIG["base_model"],
                    placeholder="Z-Image-Turbo 完整模型目录",
                )
                
                vae_input = gr.Textbox(
                    label="VAE 路径",
                    value=DEFAULT_CONFIG["vae_path"],
                    placeholder="可选，留空使用模型内置 VAE",
                )
                
                text_encoder_input = gr.Textbox(
                    label="Text Encoder 路径",
                    value=DEFAULT_CONFIG["text_encoder_path"],
                    placeholder="可选，留空使用模型内置 TE",
                )
                
                load_base_btn = gr.Button("📥 加载基础模型", variant="primary")
                base_status = gr.Textbox(label="状态", interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("### 🔧 微调权重")
                
                finetune_dir = gr.Textbox(
                    label="微调模型目录",
                    value=str(PROJECT_ROOT / "output"),
                    placeholder="扫描该目录下的 .safetensors 文件",
                )
                
                scan_btn = gr.Button("🔍 扫描模型")
                
                finetune_dropdown = gr.Dropdown(
                    label="选择微调模型",
                    choices=[],
                    interactive=True,
                )
                
                load_finetune_btn = gr.Button("📂 加载微调权重")
                finetune_status = gr.Textbox(label="微调状态", interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("### ⚖️ 权重混合")
                
                blend_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.05,
                    label="混合比例 (0=基础, 1=微调)",
                    info="实时调节基础模型与微调模型的混合程度",
                )
                
                blend_btn = gr.Button("🔀 应用混合")
                blend_status = gr.Textbox(label="混合状态", interactive=False)
            
            # 右侧：生成面板
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ 图像生成")
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="描述你想生成的图像...",
                    lines=3,
                )
                
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="描述你不想要的元素...",
                    lines=2,
                )
                
                with gr.Row():
                    width_input = gr.Slider(256, 2048, 1024, step=64, label="宽度")
                    height_input = gr.Slider(256, 2048, 1024, step=64, label="高度")
                
                with gr.Row():
                    steps_input = gr.Slider(4, 50, 9, step=1, label="步数")
                    cfg_input = gr.Slider(0.0, 15.0, 1.0, step=0.1, label="CFG Scale")
                    seed_input = gr.Number(label="Seed (-1=随机)", value=-1)
                
                with gr.Row():
                    shift_input = gr.Slider(0.0, 10.0, 3.0, step=0.1, label="Shift", info="Turbo 模型通常使用 3.0")
                
                generate_btn = gr.Button("🎨 生成图像", variant="primary", size="lg")
                
                output_image = gr.Image(label="生成结果", type="pil")
                gen_status = gr.Textbox(label="生成状态", interactive=False)
        
        # 事件绑定
        def on_load_base(base_path, vae_path, te_path):
            try:
                load_pipeline(base_path, vae_path, te_path)
                return "✅ 基础模型加载成功"
            except Exception as e:
                return f"❌ 加载失败: {str(e)}"
        
        def on_scan(directory):
            models = scan_finetune_models(directory)
            if not models:
                return gr.update(choices=[], value=None)
            
            choices = [(f"{m['name']} ({m['size']})", m["path"]) for m in models]
            return gr.update(choices=choices, value=choices[0][1] if choices else None)
        
        def on_load_finetune(finetune_path):
            if not finetune_path:
                return "请先选择微调模型"
            _, msg = load_finetune_weights(finetune_path)
            return msg
        
        def on_blend(ratio):
            return blend_weights(ratio)
        
        def on_generate(prompt, neg_prompt, width, height, steps, cfg, seed, blend, shift):
            img, msg = generate_image(prompt, neg_prompt, int(width), int(height), int(steps), cfg, int(seed), blend, shift)
            return img, msg
        
        load_base_btn.click(
            on_load_base,
            inputs=[base_model_input, vae_input, text_encoder_input],
            outputs=[base_status],
        )
        
        scan_btn.click(
            on_scan,
            inputs=[finetune_dir],
            outputs=[finetune_dropdown],
        )
        
        load_finetune_btn.click(
            on_load_finetune,
            inputs=[finetune_dropdown],
            outputs=[finetune_status],
        )
        
        blend_btn.click(
            on_blend,
            inputs=[blend_slider],
            outputs=[blend_status],
        )
        
        generate_btn.click(
            on_generate,
            inputs=[
                prompt_input,
                negative_prompt_input,
                width_input,
                height_input,
                steps_input,
                cfg_input,
                seed_input,
                blend_slider,
                shift_input,
            ],
            outputs=[output_image, gen_status],
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=3199,
        share=False,
        inbrowser=False,
    )
