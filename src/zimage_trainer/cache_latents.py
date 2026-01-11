# -*- coding: utf-8 -*-
"""
Z-Image Latent Cache Script (Standalone)

将图片编码为 latent 并缓存到磁盘。

Usage:
    python -m zimage_trainer.cache_latents \
        --vae /path/to/vae \
        --input_dir /path/to/images \
        --output_dir /path/to/cache \
        --resolution 1024
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image

# 延迟导入 torch 和 CUDA 相关模块（避免多卡模式下的 CUDA 初始化冲突）
# 实际导入在需要时进行

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Z-Image architecture identifier
ARCHITECTURE = "zi"


def find_images(input_dir: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp')) -> List[Path]:
    """查找目录中的所有图片 (递归)"""
    input_path = Path(input_dir)
    images = set()
    for ext in extensions:
        images.update(input_path.rglob(f'*{ext}'))
        images.update(input_path.rglob(f'*{ext.upper()}'))
    return sorted(list(images))


def resize_image(image: Image.Image, resolution: int, bucket_no_upscale: bool = True) -> Image.Image:
    """调整图片大小，保持宽高比"""
    w, h = image.size
    
    # 计算目标尺寸
    aspect = w / h
    if aspect > 1:
        # 横向图片
        new_w = resolution
        new_h = int(resolution / aspect)
    else:
        # 纵向图片
        new_h = resolution
        new_w = int(resolution * aspect)
    
    # 对齐到 8 的倍数（VAE 要求）
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    
    # 不放大
    if bucket_no_upscale:
        new_w = min(new_w, w)
        new_h = min(new_h, h)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
    
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return image


def process_image(
    image_path: Path,
    vae,
    resolution: int,
    output_dir: Path,
    device,  # torch.device - 延迟类型检查
    dtype=None,  # torch.dtype - 延迟类型检查
    input_root: Path = None,
) -> None:
    """处理单张图片"""
    import torch
    from safetensors.torch import save_file
    
    if dtype is None:
        dtype = torch.bfloat16
    
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 调整大小
    image = resize_image(image, resolution)
    w, h = image.size
    
    # 转换为 tensor
    import numpy as np
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    # 归一化到 [-1, 1]
    img_tensor = img_tensor * 2.0 - 1.0
    img_tensor = img_tensor.to(device=device, dtype=dtype)
    
    # 编码
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()
    
    # 应用 scaling 和 shift (Pipeline format)
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
    latent = (latent - shift_factor) * scaling_factor
    
    # 保存
    latent = latent.cpu()
    F, H, W = 1, latent.shape[2], latent.shape[3]
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
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
    
    # 文件名格式: {name}_{WxH}_{arch}.safetensors
    name = image_path.stem
    output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}.safetensors"
    
    # 保存为 safetensors
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.squeeze(0)}
    save_file(sd, str(output_file))


def process_controlnet_pair(
    target_path: Path,
    control_path: Path,
    vae,
    resolution: int,
    output_dir: Path,
    device,
    dtype=None,
    input_root: Path = None,
) -> None:
    """处理 ControlNet 配对：目标图 + 条件图"""
    import torch
    import numpy as np
    from safetensors.torch import save_file
    
    if dtype is None:
        dtype = torch.bfloat16
    
    # 处理目标图 (编码为 latent)
    target_img = Image.open(target_path).convert('RGB')
    target_img = resize_image(target_img, resolution)
    w, h = target_img.size
    
    target_array = np.array(target_img).astype(np.float32) / 255.0
    target_tensor = torch.from_numpy(target_array).permute(2, 0, 1).unsqueeze(0)
    target_tensor = target_tensor * 2.0 - 1.0
    target_tensor = target_tensor.to(device=device, dtype=dtype)
    
    with torch.no_grad():
        target_latent = vae.encode(target_tensor).latent_dist.sample()
    
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
    target_latent = (target_latent - shift_factor) * scaling_factor
    target_latent = target_latent.cpu()
    
    # 处理条件图 (保持为 tensor，不编码)
    control_img = Image.open(control_path).convert('RGB')
    control_img = control_img.resize((w, h), Image.LANCZOS)  # 调整到与目标图相同大小
    
    control_array = np.array(control_img).astype(np.float32) / 255.0
    control_tensor = torch.from_numpy(control_array).permute(2, 0, 1)  # (3, H, W)
    
    # 计算输出路径
    if input_root:
        try:
            rel_path = target_path.relative_to(input_root)
            target_dir = output_dir / rel_path.parent
        except ValueError:
            target_dir = output_dir
    else:
        target_dir = output_dir
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存
    name = target_path.stem
    F, H, W = 1, target_latent.shape[2], target_latent.shape[3]
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
    output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}_controlnet.safetensors"
    sd = {
        f"latents_{F}x{H}x{W}_{dtype_str}": target_latent.squeeze(0),
        "control_image": control_tensor,
    }
    save_file(sd, str(output_file))


def process_img2img_pair(
    target_path: Path,
    source_path: Path,
    vae,
    resolution: int,
    output_dir: Path,
    device,
    dtype=None,
    input_root: Path = None,
) -> None:
    """处理 Img2Img 配对：目标图 + 源图 (都编码为 latent)"""
    import torch
    import numpy as np
    from safetensors.torch import save_file
    
    if dtype is None:
        dtype = torch.bfloat16
    
    def encode_image(image_path):
        img = Image.open(image_path).convert('RGB')
        img = resize_image(img, resolution)
        w, h = img.size
        
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor * 2.0 - 1.0
        img_tensor = img_tensor.to(device=device, dtype=dtype)
        
        with torch.no_grad():
            latent = vae.encode(img_tensor).latent_dist.sample()
        
        scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
        shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
        latent = (latent - shift_factor) * scaling_factor
        return latent.cpu(), w, h
    
    # 编码两张图
    target_latent, w, h = encode_image(target_path)
    source_latent, _, _ = encode_image(source_path)
    
    # 计算输出路径
    if input_root:
        try:
            rel_path = target_path.relative_to(input_root)
            target_dir = output_dir / rel_path.parent
        except ValueError:
            target_dir = output_dir
    else:
        target_dir = output_dir
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存
    name = target_path.stem
    F, H, W = 1, target_latent.shape[2], target_latent.shape[3]
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
    output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}_img2img.safetensors"
    sd = {
        f"target_latents_{F}x{H}x{W}_{dtype_str}": target_latent.squeeze(0),
        f"source_latents_{F}x{H}x{W}_{dtype_str}": source_latent.squeeze(0),
    }
    save_file(sd, str(output_file))


def worker_process(gpu_id: int, image_paths: List[Path], args, output_dir: Path, total_count: int, shared_counter, counter_lock):
    """单个 GPU worker 进程"""
    import os
    # 必须在导入 torch 前设置环境变量（避免 CUDA 初始化冲突）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 现在可以安全导入 torch
    import torch
    
    device = torch.device("cuda:0")  # 在这个进程中只能看到一张卡
    dtype = torch.bfloat16
    
    # 加载 VAE
    print(f"[GPU {gpu_id}] Loading VAE...", flush=True)
    from .utils.vae_utils import load_vae
    vae = load_vae(args.vae, device=device, dtype=dtype)
    print(f"[GPU {gpu_id}] VAE loaded, processing {len(image_paths)} images", flush=True)
    
    processed = 0
    for i, image_path in enumerate(image_paths):
        # 检查是否已存在
        name = image_path.stem
        existing = list(output_dir.glob(f"{name}_*_{ARCHITECTURE}.safetensors"))
        if args.skip_existing and existing:
            # 跳过但仍更新计数
            with counter_lock:
                shared_counter.value += 1
                current = shared_counter.value
            if current % 10 == 0 or current == total_count:
                print(f"Progress: {current}/{total_count}", flush=True)
            continue
        
        try:
            process_image(image_path, vae, args.resolution, output_dir, device, dtype, input_root=Path(args.input_dir))
            processed += 1
        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {image_path.name}: {e}", flush=True)
        
        # 更新共享计数器并输出进度
        with counter_lock:
            shared_counter.value += 1
            current = shared_counter.value
        
        # 每 10 张或最后一张输出进度
        if current % 10 == 0 or current == total_count:
            print(f"Progress: {current}/{total_count}", flush=True)
    
    # 清理
    del vae
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return processed


def main():
    parser = argparse.ArgumentParser(description="Cache latents for Z-Image training")
    parser.add_argument("--vae", type=str, required=True, help="VAE model path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--resolution", type=int, default=1024, help="Target resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs (0=auto detect)")
    
    # 新增: 训练模式 (影响缓存格式)
    parser.add_argument("--mode", type=str, default="text2img", 
                       choices=["text2img", "controlnet", "img2img", "omni"],
                       help="Training mode: text2img, controlnet, img2img, omni")
    
    # ControlNet 模式参数
    parser.add_argument("--control_dir", type=str, default=None,
                       help="ControlNet condition image directory (for controlnet mode)")
    
    # Img2Img 模式参数
    parser.add_argument("--source_dir", type=str, default=None,
                       help="Source image directory (for img2img mode)")
    
    # Omni 模式参数
    parser.add_argument("--condition_dirs", type=str, default=None,
                       help="Comma-separated condition image directories (for omni mode)")
    
    args = parser.parse_args()

    
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
    
    # 检测 GPU 数量（避免在主进程初始化 CUDA，防止 spawn 模式下的冲突）
    if args.num_gpus > 0:
        num_gpus = args.num_gpus
    else:
        # 使用 subprocess 调用 nvidia-smi 获取 GPU 数量，避免初始化 CUDA
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                num_gpus = len(result.stdout.strip().split('\n'))
            else:
                num_gpus = 1  # 默认单卡
        except Exception:
            num_gpus = 1  # 如果 nvidia-smi 不可用，默认单卡
    
    if num_gpus <= 1:
        # 单 GPU 模式（兼容原有逻辑）
        # 此时可以安全导入 torch
        import torch
        from .utils.vae_utils import load_vae
        
        print(f"Using single GPU mode", flush=True)
        print(f"Progress: 0/{total}", flush=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        
        print(f"Loading VAE: {args.vae}", flush=True)
        vae = load_vae(args.vae, device=device, dtype=dtype)
        print("VAE loaded successfully", flush=True)
        
        processed = 0
        skipped = 0
        
        for i, image_path in enumerate(images, 1):
            name = image_path.stem
            existing = list(output_dir.glob(f"{name}_*_{ARCHITECTURE}.safetensors"))
            if args.skip_existing and existing:
                skipped += 1
                print(f"Progress: {i}/{total}", flush=True)
                continue
            
            try:
                process_image(image_path, vae, args.resolution, output_dir, device, dtype, input_root=Path(args.input_dir))
                processed += 1
                print(f"Progress: {i}/{total}", flush=True)
            except Exception as e:
                print(f"Error: {image_path}: {e}", flush=True)
                print(f"Progress: {i}/{total}", flush=True)
        
        print(f"Latent caching completed! Processed: {processed}, Skipped: {skipped}", flush=True)
        
        del vae
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VAE unloaded, GPU memory released", flush=True)
    
    else:
        # 多 GPU 模式
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        print(f"🚀 Multi-GPU mode: using {num_gpus} GPUs", flush=True)
        print(f"Progress: 0/{total}", flush=True)
        
        # 分片
        chunk_size = (total + num_gpus - 1) // num_gpus
        chunks = []
        for i in range(num_gpus):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            if start < total:
                chunks.append((i, images[start:end]))
        
        print(f"Distributing {total} images across {len(chunks)} GPUs", flush=True)
        for gpu_id, chunk in chunks:
            print(f"  GPU {gpu_id}: {len(chunk)} images", flush=True)
        
        # 创建共享计数器和锁
        manager = mp.Manager()
        shared_counter = manager.Value('i', 0)
        counter_lock = manager.Lock()
        
        # 启动进程池
        mp.set_start_method('spawn', force=True)
        
        total_processed = 0
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {
                executor.submit(worker_process, gpu_id, chunk, args, output_dir, total, shared_counter, counter_lock): gpu_id
                for gpu_id, chunk in chunks
            }
            
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    processed = future.result()
                    total_processed += processed
                    print(f"[GPU {gpu_id}] Completed: {processed} images", flush=True)
                except Exception as e:
                    print(f"[GPU {gpu_id}] Worker error: {e}", flush=True)
        
        print(f"Multi-GPU latent caching completed! Total processed: {total_processed}", flush=True)


if __name__ == "__main__":
    main()
