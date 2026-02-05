#!/usr/bin/env python3
"""
批量推理对比脚本 - LoRA Comparison Grid
直接在下方配置区修改参数，然后运行脚本
"""

import os
import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# ↓↓↓ 在这里配置 ↓↓↓
# ============================================================================

# 底模型路径 (Z-Image 模型目录)
BASE_MODEL = "/datasets/studio/huggingface/models/Z-Image"

# LoRA 列表 - 添加你想对比的 LoRA
# 格式: ("显示名称", "LoRA路径", 强度)
# 可以添加任意多个，用逗号分隔
LORAS = [
    ("Style A", "/path/to/style_a.safetensors", 1.0),
    ("Style B", "/path/to/style_b.safetensors", 1.0),
    # ("Style C", "/path/to/style_c.safetensors", 0.8),
    # 在这里继续添加更多 LoRA...
]

# 生成提示词
PROMPT = "a beautiful girl in the forest, masterpiece, best quality"
NEGATIVE_PROMPT = ""

# 图片设置
WIDTH = 1024
HEIGHT = 1024
STEPS = 10
CFG = 3.5
SEED = 42  # 固定种子便于对比，-1 为随机

# 是否包含 Base（无 LoRA）的对比图
INCLUDE_BASE = True

# 输出文件
OUTPUT = "comparison.png"

# 精度 (fp16 / bf16 / fp32)
DTYPE = "bf16"

# 标签设置
FONT_SIZE = 24
LABEL_HEIGHT = 40

# ============================================================================
# ↑↑↑ 配置结束 ↑↑↑
# ============================================================================


def get_dtype(dtype_str: str):
    if dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float32


def load_pipeline(model_path: str, dtype: torch.dtype):
    """加载 Z-Image Pipeline"""
    from diffusers import ZImagePipeline
    
    print(f"[Load] 加载底模型: {model_path}")
    pipe = ZImagePipeline.from_pretrained(
        model_path, 
        torch_dtype=dtype,
        local_files_only=True
    )
    
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    
    return pipe


def generate_image(pipe, prompt, negative_prompt, width, height, steps, cfg, seed, lora_path=None, lora_scale=1.0):
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)
    
    if lora_path:
        lora_name = Path(lora_path).stem
        print(f"[Generate] 加载 LoRA: {lora_name}")
        try:
            pipe.load_lora_weights(lora_path, adapter_name=lora_name)
            pipe.set_adapters([lora_name], adapter_weights=[lora_scale])
        except Exception as e:
            print(f"[WARN] 加载 LoRA 失败: {e}")
    else:
        print("[Generate] 生成 Base (无 LoRA)")
        try:
            pipe.unload_lora_weights()
        except:
            pass
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )
    
    if lora_path:
        try:
            pipe.unload_lora_weights()
        except:
            pass
    
    return result.images[0]


def add_label(image, label, font_size=24, label_height=40):
    width, height = image.size
    new_image = Image.new("RGB", (width, height + label_height), color=(32, 32, 32))
    new_image.paste(image, (0, label_height))
    
    draw = ImageDraw.Draw(new_image)
    
    # 尝试加载字体
    font = None
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf", 
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/freefont/FreeSans.ttf"]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except:
                pass
    if font is None:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), label, font=font)
    x = (width - (bbox[2] - bbox[0])) // 2
    y = (label_height - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    
    return new_image


def create_grid(images, labels):
    labeled = [add_label(img, lbl, FONT_SIZE, LABEL_HEIGHT) for img, lbl in zip(images, labels)]
    total_width = sum(img.width for img in labeled)
    max_height = max(img.height for img in labeled)
    
    grid = Image.new("RGB", (total_width, max_height), color=(0, 0, 0))
    x = 0
    for img in labeled:
        grid.paste(img, (x, 0))
        x += img.width
    
    return grid


def main():
    print("=" * 60)
    print("  LoRA Comparison Grid Generator (Z-Image)")
    print("=" * 60)
    print(f"  Base Model:  {BASE_MODEL}")
    print(f"  LoRAs:       {len(LORAS)} 个")
    print(f"  Prompt:      {PROMPT[:50]}...")
    print(f"  Size:        {WIDTH}x{HEIGHT}")
    print(f"  Steps:       {STEPS}, CFG: {CFG}, Seed: {SEED}")
    print("=" * 60)
    
    dtype = get_dtype(DTYPE)
    pipe = load_pipeline(BASE_MODEL, dtype)
    
    images = []
    labels = []
    
    # Base 图
    if INCLUDE_BASE:
        print(f"\n[1/{len(LORAS) + 1}] 生成 Base...")
        img = generate_image(pipe, PROMPT, NEGATIVE_PROMPT, WIDTH, HEIGHT, STEPS, CFG, SEED)
        images.append(img)
        labels.append("Base (No LoRA)")
    
    # 各 LoRA 图
    for i, (name, path, scale) in enumerate(LORAS):
        idx = i + (2 if INCLUDE_BASE else 1)
        total = len(LORAS) + (1 if INCLUDE_BASE else 0)
        print(f"\n[{idx}/{total}] 生成: {name}")
        
        if not os.path.exists(path):
            print(f"  [WARN] 文件不存在: {path}")
            continue
        
        try:
            img = generate_image(pipe, PROMPT, NEGATIVE_PROMPT, WIDTH, HEIGHT, STEPS, CFG, SEED, path, scale)
            images.append(img)
            labels.append(name)
        except Exception as e:
            print(f"  [ERROR] 失败: {e}")
    
    if not images:
        print("\n[ERROR] 没有生成任何图片!")
        return
    
    print("\n[拼接] 创建对比图...")
    grid = create_grid(images, labels)
    
    output_path = Path(OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path), quality=95)
    
    print(f"\n[完成] 保存到: {output_path}")
    print(f"       尺寸: {grid.width}x{grid.height}")
    
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
