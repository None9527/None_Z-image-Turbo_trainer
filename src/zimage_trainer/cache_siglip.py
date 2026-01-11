# -*- coding: utf-8 -*-
"""
Z-Image SigLIP Cache Script

将条件图像编码为 SigLIP 特征并缓存到磁盘（用于 Omni 模式）。

Usage:
    python -m zimage_trainer.cache_siglip \
        --siglip /path/to/siglip \
        --input_dir /path/to/condition_images \
        --output_dir /path/to/cache
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ARCHITECTURE = "zi"


def find_images(input_dir: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp')) -> List[Path]:
    """查找目录中的所有图片 (递归)"""
    input_path = Path(input_dir)
    images = set()
    for ext in extensions:
        images.update(input_path.rglob(f'*{ext}'))
        images.update(input_path.rglob(f'*{ext.upper()}'))
    return sorted(list(images))


def load_siglip(model_path: str, device, dtype):
    """加载 SigLIP 视觉编码器"""
    from transformers import AutoModel, AutoProcessor
    
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, processor


def process_image(
    image_path: Path,
    model,
    processor,
    output_dir: Path,
    device,
    dtype=None,
    input_root: Path = None,
) -> None:
    """处理单张图片，编码为 SigLIP 特征"""
    import torch
    from safetensors.torch import save_file
    
    if dtype is None:
        dtype = torch.bfloat16
    
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 使用 processor 预处理
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    
    # 编码
    with torch.no_grad():
        outputs = model.vision_model(pixel_values=pixel_values)
        # 取 last_hidden_state 或 pooler_output
        siglip_feats = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    
    # 保存
    siglip_feats = siglip_feats.cpu()
    
    # 计算输出路径 (保持目录结构)
    if input_root:
        try:
            rel_path = image_path.relative_to(input_root)
            target_dir = output_dir / rel_path.parent
        except ValueError:
            target_dir = output_dir
    else:
        target_dir = output_dir
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名格式: {name}_{arch}_siglip.safetensors
    name = image_path.stem
    output_file = target_dir / f"{name}_{ARCHITECTURE}_siglip.safetensors"
    
    # 保存为 safetensors
    sd = {"siglip_feats": siglip_feats.squeeze(0)}
    save_file(sd, str(output_file))


def main():
    parser = argparse.ArgumentParser(description="Cache SigLIP features for Z-Image Omni training")
    parser.add_argument("--siglip", type=str, required=True, help="SigLIP model path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    
    args = parser.parse_args()
    
    import torch
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找图片
    images = find_images(args.input_dir)
    total = len(images)
    print(f"Found {total} images", flush=True)
    
    if total == 0:
        print("No images to process", flush=True)
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"Loading SigLIP: {args.siglip}", flush=True)
    model, processor = load_siglip(args.siglip, device=device, dtype=dtype)
    print("SigLIP loaded successfully", flush=True)
    
    processed = 0
    skipped = 0
    
    for i, image_path in enumerate(images, 1):
        name = image_path.stem
        existing = list(output_dir.glob(f"{name}_{ARCHITECTURE}_siglip.safetensors"))
        if args.skip_existing and existing:
            skipped += 1
            if i % 10 == 0 or i == total:
                print(f"Progress: {i}/{total}", flush=True)
            continue
        
        try:
            process_image(image_path, model, processor, output_dir, device, dtype, input_root=Path(args.input_dir))
            processed += 1
            if i % 10 == 0 or i == total:
                print(f"Progress: {i}/{total}", flush=True)
        except Exception as e:
            print(f"Error: {image_path}: {e}", flush=True)
    
    print(f"SigLIP caching completed! Processed: {processed}, Skipped: {skipped}", flush=True)
    
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("SigLIP unloaded, GPU memory released", flush=True)


if __name__ == "__main__":
    main()
