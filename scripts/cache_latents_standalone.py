#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z-Image Latent Cache Script (Standalone - Multi-GPU Safe)

独立缓存脚本，避免触发 zimage_trainer/__init__.py 导致 CUDA 提前初始化。
使用独立子进程启动方式，确保 CUDA_VISIBLE_DEVICES 在 torch 导入前设置。

Usage:
    python scripts/cache_latents_standalone.py \
        --vae /path/to/vae \
        --input_dir /path/to/images \
        --output_dir /path/to/cache \
        --resolution 1024
"""

# === 重要：不要在模块顶层导入任何可能触发 CUDA 的库 ===
# torch, PIL, numpy, diffusers 等都必须延迟导入

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

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


def run_single_gpu_worker(gpu_id: int, vae_path: str, input_dir: str, output_dir: str, 
                          resolution: int, skip_existing: bool, image_paths: List[str],
                          progress_queue):
    """
    单个 GPU 的 worker 函数（在独立进程中运行）
    
    重要：这个函数会在 subprocess 中运行，CUDA_VISIBLE_DEVICES 已经在启动前设置
    """
    # 延迟导入所有可能触发 CUDA 的库
    import torch
    from PIL import Image
    import numpy as np
    from safetensors.torch import save_file
    from diffusers import AutoencoderKL
    
    device = torch.device("cuda:0")  # 只能看到一张卡
    dtype = torch.bfloat16
    
    # 加载 VAE
    print(f"[GPU {gpu_id}] Loading VAE...", flush=True)
    if os.path.isdir(vae_path):
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
    elif vae_path.endswith(".safetensors"):
        vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported VAE path: {vae_path}")
    
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    print(f"[GPU {gpu_id}] VAE loaded, processing {len(image_paths)} images", flush=True)
    
    output_path = Path(output_dir)
    input_root = Path(input_dir)
    processed = 0
    
    for i, img_path_str in enumerate(image_paths):
        image_path = Path(img_path_str)
        name = image_path.stem
        
        # 检查是否已存在
        existing = list(output_path.glob(f"{name}_*_{ARCHITECTURE}.safetensors"))
        if skip_existing and existing:
            progress_queue.put(("skip", gpu_id, 1))
            continue
        
        try:
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            w, h = image.size
            
            # 调整大小
            aspect = w / h
            if aspect > 1:
                new_w = resolution
                new_h = int(resolution / aspect)
            else:
                new_h = resolution
                new_w = int(resolution * aspect)
            
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            new_w = min(new_w, w)
            new_h = min(new_h, h)
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            
            if (new_w, new_h) != (w, h):
                image = image.resize((new_w, new_h), Image.LANCZOS)
            
            w, h = image.size
            
            # 转换为 tensor
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor * 2.0 - 1.0
            img_tensor = img_tensor.to(device=device, dtype=dtype)
            
            # 编码
            with torch.no_grad():
                latent = vae.encode(img_tensor).latent_dist.sample()
            
            # 应用 scaling 和 shift
            scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
            shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
            latent = (latent - shift_factor) * scaling_factor
            
            # 保存
            latent = latent.cpu()
            F, H, W = 1, latent.shape[2], latent.shape[3]
            dtype_str = "bf16"
            
            # 计算输出路径
            try:
                rel_path = image_path.relative_to(input_root)
                target_dir = output_path / rel_path.parent
            except ValueError:
                target_dir = output_path
            
            target_dir.mkdir(parents=True, exist_ok=True)
            output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}.safetensors"
            
            sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.squeeze(0)}
            save_file(sd, str(output_file))
            
            processed += 1
            progress_queue.put(("done", gpu_id, 1))
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {image_path.name}: {e}", flush=True)
            progress_queue.put(("error", gpu_id, 1))
    
    # 清理
    del vae
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    progress_queue.put(("finished", gpu_id, processed))


def spawn_gpu_worker(gpu_id: int, vae_path: str, input_dir: str, output_dir: str,
                     resolution: int, skip_existing: bool, image_paths: List[str],
                     progress_queue):
    """
    在设置 CUDA_VISIBLE_DEVICES 后 spawn 一个 worker 子进程
    """
    import multiprocessing as mp
    
    # 创建子进程，在子进程 fork/spawn 之前设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 使用 subprocess 启动独立 Python 进程（最安全的方式）
    import subprocess
    import json
    import tempfile
    
    # 将参数写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        params = {
            "gpu_id": gpu_id,
            "vae_path": vae_path,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "resolution": resolution,
            "skip_existing": skip_existing,
            "image_paths": [str(p) for p in image_paths]
        }
        json.dump(params, f)
        params_file = f.name
    
    # 构建内联 worker 脚本
    worker_script = f'''
import os
import sys
import json

# 设置 CUDA_VISIBLE_DEVICES（必须在 import torch 之前）
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

# 现在可以安全导入
import torch
from PIL import Image
import numpy as np
from safetensors.torch import save_file
from diffusers import AutoencoderKL
from pathlib import Path

ARCHITECTURE = "zi"

# 读取参数
with open(r"{params_file}", "r") as f:
    params = json.load(f)

gpu_id = params["gpu_id"]
vae_path = params["vae_path"]
input_dir = params["input_dir"]
output_dir = params["output_dir"]
resolution = params["resolution"]
skip_existing = params["skip_existing"]
image_paths = params["image_paths"]

device = torch.device("cuda:0")
dtype = torch.bfloat16

# 加载 VAE
print(f"[GPU {{gpu_id}}] Loading VAE...", flush=True)
if os.path.isdir(vae_path):
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
elif vae_path.endswith(".safetensors"):
    vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
else:
    raise ValueError(f"Unsupported VAE path: {{vae_path}}")

vae.to(device)
vae.eval()
vae.requires_grad_(False)
print(f"[GPU {{gpu_id}}] VAE loaded, processing {{len(image_paths)}} images", flush=True)

output_path = Path(output_dir)
input_root = Path(input_dir)
processed = 0
total = len(image_paths)

for i, img_path_str in enumerate(image_paths):
    image_path = Path(img_path_str)
    name = image_path.stem
    
    # 检查是否已存在
    existing = list(output_path.glob(f"{{name}}_*_{{ARCHITECTURE}}.safetensors"))
    if skip_existing and existing:
        continue
    
    try:
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        
        # 调整大小
        aspect = w / h
        if aspect > 1:
            new_w = resolution
            new_h = int(resolution / aspect)
        else:
            new_h = resolution
            new_w = int(resolution * aspect)
        
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        new_w = min(new_w, w)
        new_h = min(new_h, h)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        
        if (new_w, new_h) != (w, h):
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        w, h = image.size
        
        # 转换为 tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor * 2.0 - 1.0
        img_tensor = img_tensor.to(device=device, dtype=dtype)
        
        # 编码
        with torch.no_grad():
            latent = vae.encode(img_tensor).latent_dist.sample()
        
        # 应用 scaling 和 shift
        scaling_factor = getattr(vae.config, "scaling_factor", 0.3611)
        shift_factor = getattr(vae.config, "shift_factor", 0.1159)
        latent = (latent - shift_factor) * scaling_factor
        
        # 保存
        latent = latent.cpu()
        F, H, W = 1, latent.shape[2], latent.shape[3]
        dtype_str = "bf16"
        
        # 计算输出路径
        try:
            rel_path = image_path.relative_to(input_root)
            target_dir = output_path / rel_path.parent
        except ValueError:
            target_dir = output_path
        
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file = target_dir / f"{{name}}_{{w}}x{{h}}_{{ARCHITECTURE}}.safetensors"
        
        sd = {{f"latents_{{F}}x{{H}}x{{W}}_{{dtype_str}}": latent.squeeze(0)}}
        save_file(sd, str(output_file))
        
        processed += 1
        
    except Exception as e:
        print(f"[GPU {{gpu_id}}] Error: {{image_path.name}}: {{e}}", flush=True)

# 清理
del vae
import gc
gc.collect()
torch.cuda.empty_cache()

print(f"[GPU {{gpu_id}}] Completed: {{processed}} images", flush=True)

# 清理临时文件
os.remove(r"{params_file}")
'''
    
    # 启动子进程
    process = subprocess.Popen(
        [sys.executable, "-c", worker_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1
    )
    
    return process


def main():
    parser = argparse.ArgumentParser(description="Cache latents for Z-Image training (Multi-GPU Safe)")
    parser.add_argument("--vae", type=str, required=True, help="VAE model path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--resolution", type=int, default=1024, help="Target resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs (0=auto detect)")
    
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
    
    # 检测 GPU 数量（避免在主进程初始化 CUDA）
    if args.num_gpus > 0:
        num_gpus = args.num_gpus
    else:
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                num_gpus = len(result.stdout.strip().split('\n'))
            else:
                num_gpus = 1
        except Exception:
            num_gpus = 1
    
    if num_gpus <= 1:
        # 单 GPU 模式
        import torch
        from PIL import Image
        import numpy as np
        from safetensors.torch import save_file
        from diffusers import AutoencoderKL
        
        print(f"Using single GPU mode", flush=True)
        print(f"Progress: 0/{total}", flush=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        
        print(f"Loading VAE: {args.vae}", flush=True)
        if os.path.isdir(args.vae):
            vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype)
        else:
            vae = AutoencoderKL.from_single_file(args.vae, torch_dtype=dtype)
        vae.to(device)
        vae.eval()
        vae.requires_grad_(False)
        print("VAE loaded successfully", flush=True)
        
        processed = 0
        skipped = 0
        input_root = Path(args.input_dir)
        
        for i, image_path in enumerate(images, 1):
            name = image_path.stem
            existing = list(output_dir.glob(f"{name}_*_{ARCHITECTURE}.safetensors"))
            if args.skip_existing and existing:
                skipped += 1
                print(f"Progress: {i}/{total}", flush=True)
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
                w, h = image.size
                
                aspect = w / h
                if aspect > 1:
                    new_w = args.resolution
                    new_h = int(args.resolution / aspect)
                else:
                    new_h = args.resolution
                    new_w = int(args.resolution * aspect)
                
                new_w = (new_w // 8) * 8
                new_h = (new_h // 8) * 8
                new_w = min(new_w, w)
                new_h = min(new_h, h)
                new_w = (new_w // 8) * 8
                new_h = (new_h // 8) * 8
                
                if (new_w, new_h) != (w, h):
                    image = image.resize((new_w, new_h), Image.LANCZOS)
                
                w, h = image.size
                
                img_array = np.array(image).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor * 2.0 - 1.0
                img_tensor = img_tensor.to(device=device, dtype=dtype)
                
                with torch.no_grad():
                    latent = vae.encode(img_tensor).latent_dist.sample()
                
                scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
                shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
                latent = (latent - shift_factor) * scaling_factor
                
                latent = latent.cpu()
                F, H, W = 1, latent.shape[2], latent.shape[3]
                
                try:
                    rel_path = image_path.relative_to(input_root)
                    target_dir = output_dir / rel_path.parent
                except ValueError:
                    target_dir = output_dir
                
                target_dir.mkdir(parents=True, exist_ok=True)
                output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}.safetensors"
                
                sd = {f"latents_{F}x{H}x{W}_bf16": latent.squeeze(0)}
                save_file(sd, str(output_file))
                
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
        # 多 GPU 模式 - 使用 subprocess 启动独立进程
        print(f"🚀 Multi-GPU mode: using {num_gpus} GPUs", flush=True)
        print(f"Progress: 0/{total}", flush=True)
        
        # 分片
        chunk_size = (total + num_gpus - 1) // num_gpus
        chunks = []
        for i in range(num_gpus):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            if start < total:
                chunks.append((i, [str(p) for p in images[start:end]]))
        
        print(f"Distributing {total} images across {len(chunks)} GPUs", flush=True)
        for gpu_id, chunk in chunks:
            print(f"  GPU {gpu_id}: {len(chunk)} images", flush=True)
        
        # 启动所有 worker 进程
        processes = []
        for gpu_id, image_paths in chunks:
            p = spawn_gpu_worker(
                gpu_id, args.vae, args.input_dir, args.output_dir,
                args.resolution, args.skip_existing, image_paths, None
            )
            processes.append((gpu_id, p))
        
        # 收集输出并等待完成
        import threading
        import queue
        
        output_queue = queue.Queue()
        
        def read_output(gpu_id, process, q):
            for line in process.stdout:
                q.put((gpu_id, line.rstrip()))
            process.wait()
        
        threads = []
        for gpu_id, p in processes:
            t = threading.Thread(target=read_output, args=(gpu_id, p, output_queue), daemon=True)
            t.start()
            threads.append(t)
        
        # 输出进度
        completed = 0
        progress_count = 0
        while completed < len(processes):
            try:
                gpu_id, line = output_queue.get(timeout=0.1)
                print(f"[GPU {gpu_id}] {line}", flush=True)
                if "Completed:" in line:
                    completed += 1
                elif not line.startswith("[GPU"):
                    # 可能是进度信息
                    progress_count += 1
                    if progress_count % 10 == 0:
                        print(f"Progress: {progress_count}/{total}", flush=True)
            except queue.Empty:
                # 检查进程是否结束
                all_done = all(p.poll() is not None for _, p in processes)
                if all_done:
                    break
        
        # 等待所有线程结束
        for t in threads:
            t.join(timeout=1)
        
        print(f"Multi-GPU latent caching completed!", flush=True)


if __name__ == "__main__":
    main()
