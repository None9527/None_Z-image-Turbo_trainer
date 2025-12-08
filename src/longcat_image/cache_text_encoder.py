# -*- coding: utf-8 -*-
"""
LongCat-Image Text Encoder Cache Script (Standalone)

使用 Qwen2.5-VL 模型将纯文本编码并缓存到磁盘。

Usage:
    python -m longcat_image.cache_text_encoder \
        --text_encoder /path/to/qwen2_5_vl \
        --input_dir /path/to/images \
        --output_dir /path/to/cache
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from safetensors.torch import save_file
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# LongCat architecture identifier
ARCHITECTURE = "lc"


def load_text_encoder(model_path: str, device: torch.device, dtype: torch.dtype):
    """加载 Qwen2.5-VL 文本+视觉编码器"""
    logger.info(f"Loading text encoder: {model_path}")
    
    model_path_obj = Path(model_path)
    
    # 检查 tokenizer 位置（可能在 text_encoder/tokenizer 或与 text_encoder 同级）
    if (model_path_obj / "tokenizer").exists():
        processor_path = str(model_path_obj / "tokenizer")
    elif (model_path_obj.parent / "tokenizer").exists():
        # tokenizer 与 text_encoder 同级
        processor_path = str(model_path_obj.parent / "tokenizer")
    else:
        processor_path = model_path
    
    logger.info(f"Using processor from: {processor_path}")
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        processor_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    # 加载模型
    if model_path.endswith('.safetensors'):
        # 单文件加载
        from safetensors.torch import load_file
        model_dir = model_path_obj.parent
        state_dict = load_file(model_path)
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            state_dict=state_dict,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True
        )
    else:
        # 目录加载
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True
        )
    
    model.to(device)
    model.eval()
    
    return processor, model


def get_caption(image_path: Path) -> Optional[str]:
    """获取图片对应的文本描述"""
    txt_paths = [
        image_path.with_suffix('.txt'),
        image_path.with_suffix('.caption'),
        image_path.parent / f"{image_path.stem}.txt",
    ]
    
    for txt_path in txt_paths:
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    
    return None


def encode_text(
    processor,
    model,
    text: str,
    max_length: int = 512,
    device: torch.device = None,
) -> torch.Tensor:
    """
    编码纯文本为 embedding
    
    LongCat 训练时使用纯文本编码，不需要图片输入
    """
    # 使用 tokenizer 直接编码文本
    from longcat_image.utils.model_utils import split_quotation
    
    # 处理引号（与 LongCat pipeline 保持一致）
    text = text.strip('"') if text.startswith('"') and text.endswith('"') else text
    
    all_tokens = []
    for clean_prompt_sub, matched in split_quotation(text):
        if matched:
            for sub_word in clean_prompt_sub:
                tokens = processor.tokenizer(sub_word, add_special_tokens=False)['input_ids']
                all_tokens.extend(tokens)
        else:
            tokens = processor.tokenizer(clean_prompt_sub, add_special_tokens=False)['input_ids']
            all_tokens.extend(tokens)
    
    # 截断到最大长度
    all_tokens = all_tokens[:max_length]
    
    # Padding
    text_tokens_and_mask = processor.tokenizer.pad(
        {'input_ids': [all_tokens]},
        max_length=max_length,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    if device:
        text_tokens_and_mask = {k: v.to(device) for k, v in text_tokens_and_mask.items()}
    
    # 编码
    with torch.no_grad():
        outputs = model(
            input_ids=text_tokens_and_mask['input_ids'],
            attention_mask=text_tokens_and_mask['attention_mask'],
            output_hidden_states=True
        )
        # 使用最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]
        # 去掉 batch 维度
        embed = hidden_states.squeeze(0)
    
    return embed


def process_caption(
    image_path: Path,
    processor,
    model,
    output_dir: Path,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
    input_root: Path = None,
) -> bool:
    """处理单个文本描述"""
    # 获取 caption
    caption = get_caption(image_path)
    if caption is None:
        logger.warning(f"No caption found for {image_path}")
        return False
    
    # 编码纯文本（不需要图片）
    embed = encode_text(processor, model, caption, max_length, device)
    embed = embed.to(dtype=dtype)
    
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
    
    # 保存
    name = image_path.stem
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"
    
    # 使用 varlen 格式（与 Z-Image 兼容）
    sd = {f"varlen_vl_embed_{dtype_str}": embed.cpu()}
    save_file(sd, str(output_file))
    
    return True


def find_images(input_dir: str) -> List[Path]:
    """查找目录中的所有图片 (递归)"""
    input_path = Path(input_dir)
    extensions = ('.jpg', '.jpeg', '.png', '.webp')
    images = set()
    for ext in extensions:
        images.update(input_path.rglob(f'*{ext}'))
        images.update(input_path.rglob(f'*{ext.upper()}'))
    return sorted(list(images))


def main():
    parser = argparse.ArgumentParser(description="Cache text embeddings for LongCat-Image training")
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen2.5-VL model path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory (with .txt captions)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载文本编码器
    print(f"Loading Text Encoder: {args.text_encoder}", flush=True)
    processor, model = load_text_encoder(args.text_encoder, device, dtype)
    print("Text Encoder loaded successfully", flush=True)
    
    # 查找图片
    images = find_images(args.input_dir)
    total = len(images)
    print(f"Found {total} images", flush=True)
    print(f"Progress: 0/{total}", flush=True)
    
    # 处理
    success = 0
    skipped = 0
    
    for i, image_path in enumerate(images, 1):
        # 检查是否已存在
        name = image_path.stem
        
        # 计算预期输出路径
        if args.input_dir:
            try:
                rel_path = image_path.relative_to(args.input_dir)
                target_dir = output_dir / rel_path.parent
            except ValueError:
                target_dir = output_dir
        else:
            target_dir = output_dir
            
        output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"
        
        if args.skip_existing and output_file.exists():
            skipped += 1
            print(f"Progress: {i}/{total}", flush=True)
            continue
        
        try:
            if process_caption(image_path, processor, model, output_dir, args.max_length, device, dtype, input_root=Path(args.input_dir)):
                success += 1
            print(f"Progress: {i}/{total}", flush=True)
        except Exception as e:
            print(f"Error: {image_path}: {e}", flush=True)
            print(f"Progress: {i}/{total}", flush=True)
            continue
    
    print(f"Text encoding completed! Processed: {success}, Skipped: {skipped}", flush=True)
    
    # 清理显存
    del model
    del processor
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Text Encoder unloaded, GPU memory released", flush=True)


if __name__ == "__main__":
    main()
