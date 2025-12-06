#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ollama 图片标注脚本
支持多种视觉模型，自动断点续传
"""
import argparse
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 默认提示词模板
DEFAULT_PROMPT = """Describe this image in detail for AI image generation training. 
Focus on: subject, clothing, pose, expression, hair, accessories.
Output format: comma-separated tags in English.
Be concise and specific."""

DEFAULT_PROMPT_CN = """请为这张图片生成训练标注。
描述内容：主体特征、服装、姿势、表情、发型、配饰。
输出格式：逗号分隔的标签，中文为主。
简洁准确。"""


def parse_args():
    parser = argparse.ArgumentParser(description="Ollama 图片标注工具")
    parser.add_argument("--input_dir", type=str, required=True, help="图片目录")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="Ollama API 地址")
    parser.add_argument("--model", type=str, default="llava:13b", help="视觉模型名称")
    parser.add_argument("--prompt", type=str, default=None, help="自定义提示词")
    parser.add_argument("--prompt_file", type=str, default=None, help="提示词文件路径")
    parser.add_argument("--language", type=str, default="en", choices=["en", "cn"], help="输出语言")
    parser.add_argument("--trigger_word", type=str, default="", help="触发词（添加到标注开头）")
    parser.add_argument("--max_long_edge", type=int, default=1024, help="图片最长边限制")
    parser.add_argument("--downsample_ratio", type=float, default=0.5, help="下采样比例（0.5=减半）")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="跳过已标注图片")
    parser.add_argument("--enable_think", action="store_true", help="启用思考模式（qwen3等模型）")
    parser.add_argument("--timeout", type=int, default=180, help="API 超时时间（秒）")
    parser.add_argument("--delay", type=float, default=0.2, help="请求间隔（秒）")
    parser.add_argument("--batch_size", type=int, default=0, help="批量处理数量（0=全部）")
    return parser.parse_args()


def get_image_paths(input_dir: Path, skip_existing: bool = True) -> list:
    """获取待标注图片列表"""
    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]
    image_paths = []
    
    for ext in IMAGE_EXTENSIONS:
        for img_path in input_dir.rglob(f"*{ext}"):
            if skip_existing:
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists() and txt_path.stat().st_size > 0:
                    continue
            image_paths.append(img_path)
    
    # 也检查大写扩展名
    for ext in IMAGE_EXTENSIONS:
        for img_path in input_dir.rglob(f"*{ext.upper()}"):
            if skip_existing:
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists() and txt_path.stat().st_size > 0:
                    continue
            if img_path not in image_paths:
                image_paths.append(img_path)
    
    return sorted(image_paths)


def resize_image(img_path: Path, max_long_edge: int = 1024, downsample_ratio: float = 1.0) -> bytes:
    """调整图片大小并转换为 JPEG 字节"""
    with Image.open(img_path) as im:
        # 转换为 RGB
        if im.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', im.size, (255, 255, 255))
            if im.mode == 'P':
                im = im.convert('RGBA')
            if im.mode in ('RGBA', 'LA'):
                background.paste(im, mask=im.split()[-1])
                im = background
            else:
                im = im.convert('RGB')
        elif im.mode != 'RGB':
            im = im.convert('RGB')
        
        w, h = im.size
        
        # 应用下采样比例
        if downsample_ratio < 1.0:
            w = int(w * downsample_ratio)
            h = int(h * downsample_ratio)
            im = im.resize((w, h), Image.Resampling.BICUBIC)
        
        # 限制最长边
        if max(w, h) > max_long_edge:
            if w > h:
                new_w = max_long_edge
                new_h = int(h * max_long_edge / w)
            else:
                new_h = max_long_edge
                new_w = int(w * max_long_edge / h)
            im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 转换为 JPEG 字节
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def generate_caption(
    img_path: Path,
    ollama_url: str,
    model: str,
    prompt: str,
    max_long_edge: int,
    downsample_ratio: float,
    enable_think: bool,
    timeout: int
) -> Optional[str]:
    """生成单张图片标注"""
    try:
        # 准备图片
        jpeg_bytes = resize_image(img_path, max_long_edge, downsample_ratio)
        base64_img = base64.b64encode(jpeg_bytes).decode()
        
        # 处理思考模式
        actual_prompt = prompt
        if not enable_think and ('qwen3' in model.lower() or 'qwen-3' in model.lower()):
            actual_prompt = f"/no_think {prompt}"
        
        # 构建请求
        payload = {
            "model": model,
            "prompt": actual_prompt,
            "images": [base64_img],
            "stream": False,
            "context": [],
            "temperature": 0.7,
            "top_p": 0.9,
            "options": {
                "num_predict": 1024
            }
        }
        
        if not enable_think:
            payload["options"]["think"] = False
            payload["think"] = False
        
        # 发送请求
        resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        
        data = resp.json()
        caption = ""
        
        # 提取响应
        if "response" in data:
            caption = data["response"].strip()
        
        # 如果开启思考模式但 response 为空，尝试从 thinking 提取
        if not caption and enable_think and "thinking" in data and data["thinking"]:
            thinking = data["thinking"].strip()
            # 取最后一段作为结果
            paragraphs = [p.strip() for p in thinking.split('\n\n') if p.strip()]
            if paragraphs:
                caption = paragraphs[-1]
        
        if not caption:
            logger.warning(f"空响应: {img_path.name}")
            return None
        
        # 清理输出
        caption = caption.replace("```markdown", "").replace("```", "")
        import re
        caption = re.sub(r'<think>.*?</think>', '', caption, flags=re.DOTALL)
        caption = re.sub(r'<thinking>.*?</thinking>', '', caption, flags=re.DOTALL)
        caption = re.sub(r'\[img-\d+\]', '', caption)
        caption = re.sub(r'\n{3,}', '\n\n', caption)
        caption = caption.strip()
        
        return caption if caption else None
        
    except requests.exceptions.Timeout:
        logger.error(f"超时: {img_path.name}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"请求失败 ({img_path.name}): {e}")
        return None
    except Exception as e:
        logger.error(f"处理失败 ({img_path.name}): {e}")
        return None


def save_caption(img_path: Path, caption: str, trigger_word: str = ""):
    """保存标注文件"""
    if trigger_word.strip():
        caption = f"{trigger_word.strip()}, {caption}"
    
    txt_path = img_path.with_suffix(".txt")
    txt_path.write_text(caption, encoding="utf-8")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        logger.error(f"目录不存在: {input_dir}")
        sys.exit(1)
    
    # 确定提示词
    if args.prompt_file and Path(args.prompt_file).exists():
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT_CN if args.language == "cn" else DEFAULT_PROMPT
    
    # 获取图片列表
    image_paths = get_image_paths(input_dir, args.skip_existing)
    
    if not image_paths:
        logger.info("没有待标注的图片")
        print("Progress: 0/0", flush=True)
        return
    
    total = len(image_paths)
    if args.batch_size > 0:
        image_paths = image_paths[:args.batch_size]
        total = len(image_paths)
    
    logger.info(f"待标注: {total} 张图片")
    logger.info(f"模型: {args.model}")
    logger.info(f"API: {args.ollama_url}")
    
    print(f"Progress: 0/{total}", flush=True)
    
    success_count = 0
    fail_count = 0
    
    for i, img_path in enumerate(image_paths):
        # 输出当前文件（用于前端显示）
        print(f"[FILE] {img_path.name}", flush=True)
        sys.stdout.flush()
        
        caption = generate_caption(
            img_path,
            args.ollama_url,
            args.model,
            prompt,
            args.max_long_edge,
            args.downsample_ratio,
            args.enable_think,
            args.timeout
        )
        
        if caption:
            save_caption(img_path, caption, args.trigger_word)
            success_count += 1
            print(f"[OK] {img_path.name}", flush=True)
        else:
            fail_count += 1
            print(f"[FAIL] {img_path.name}", flush=True)
        
        # 输出进度（用于前端解析）
        print(f"Progress: {i+1}/{total}", flush=True)
        sys.stdout.flush()
        
        if args.delay > 0 and i < total - 1:
            time.sleep(args.delay)
    
    logger.info(f"完成! 成功: {success_count}, 失败: {fail_count}")
    print(f"Progress: {total}/{total}", flush=True)


if __name__ == "__main__":
    main()

