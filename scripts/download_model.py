"""
多模型下载脚本 - 支持总进度显示

支持:
- Z-Image-Turbo (Tongyi-MAI/Z-Image-Turbo)
- LongCat-Image-Dev (meituan-longcat/LongCat-Image-Dev)

Usage:
    python download_model.py <local_dir> [model_id]
    
Examples:
    python download_model.py ./zimage_models                          # 默认下载 Z-Image
    python download_model.py ./zimage_models Tongyi-MAI/Z-Image-Turbo
    python download_model.py ./longcat_models meituan-longcat/LongCat-Image-Dev
"""


import sys
import os
import time
import threading
from pathlib import Path

# 模型映射
MODEL_MAP = {
    "zimage": "Tongyi-MAI/Z-Image-Turbo",
    "longcat": "meituan-longcat/LongCat-Image-Dev",
    # 别名
    "z-image": "Tongyi-MAI/Z-Image-Turbo",
    "z-image-turbo": "Tongyi-MAI/Z-Image-Turbo",
    "longcat-image": "meituan-longcat/LongCat-Image-Dev",
}

# 预估模型大小 (GB)
MODEL_SIZES = {
    "Tongyi-MAI/Z-Image-Turbo": 32.0,
    "meituan-longcat/LongCat-Image-Dev": 35.0,
}


def get_model_id(model_arg: str) -> str:
    """解析模型 ID"""
    lower_arg = model_arg.lower().replace("_", "-")
    if lower_arg in MODEL_MAP:
        return MODEL_MAP[lower_arg]
    return model_arg


def get_dir_size_gb(path: Path) -> float:
    """获取目录大小 (GB)"""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except Exception:
        pass
    return total / (1024 ** 3)


def format_size(gb: float) -> str:
    """格式化大小显示"""
    if gb < 1:
        return f"{gb * 1024:.1f} MB"
    return f"{gb:.2f} GB"


def check_connectivity(model_id: str) -> bool:
    """检查 ModelScope 连接性"""
    print(f"[INFO] Checking connectivity to ModelScope for {model_id}...")
    try:
        from modelscope.hub.api import HubApi
        api = HubApi()
        # 尝试获取文件列表作为连接测试
        api.get_model_files(model_id)
        print("[INFO] Connectivity check passed.")
        return True
    except Exception as e:
        print(f"[ERROR] Connectivity check failed: {e}")
        print("[TIP] Please check your network connection or proxy settings.")
        return False


def progress_monitor(model_dir: Path, expected_size_gb: float, stop_event: threading.Event):
    """后台线程监控下载进度"""
    last_size = 0
    last_time = time.time()
    start_time = time.time()
    
    while not stop_event.is_set():
        current_size = get_dir_size_gb(model_dir)
        current_time = time.time()
        
        # 计算速度
        time_delta = current_time - last_time
        if time_delta > 0:
            speed = (current_size - last_size) / time_delta  # GB/s
            speed_str = f"{speed * 1024:.1f} MB/s" if speed > 0 else "0.0 MB/s"
        else:
            speed_str = "0.0 MB/s"
        
        # 计算进度
        if expected_size_gb > 0:
            progress = min(current_size / expected_size_gb * 100, 99.9)
        else:
            progress = 0
        
        # 显示进度条
        bar_width = 30
        filled = int(bar_width * progress / 100)
        bar = "=" * filled + "-" * (bar_width - filled)
        
        # 状态描述
        elapsed = int(current_time - start_time)
        if speed <= 0 and elapsed > 10:
             status = "验证中..." if current_size > 0 else "准备中..."
        else:
             status = "下载中..."

        # 打印进度（使用换行符，确保 readline 能读取）
        msg = f"[PROGRESS] [{bar}] {progress:.1f}% | {format_size(current_size)}/{format_size(expected_size_gb)} | {speed_str} | {status}"
        print(msg, flush=True)
        
        last_size = current_size
        last_time = current_time
        
        time.sleep(1)


def main():
    # 设置 stdout 为无缓冲模式
    sys.stdout.reconfigure(encoding='utf-8')
    
    if len(sys.argv) < 2:
        print("Usage: python download_model.py <local_dir> [model_id]")
        print("\nSupported models:")
        print("  - zimage (default): Tongyi-MAI/Z-Image-Turbo (~32GB)")
        print("  - longcat: meituan-longcat/LongCat-Image-Dev (~35GB)")
        sys.exit(1)
    
    model_dir = Path(sys.argv[1])
    
    # 获取模型 ID
    if len(sys.argv) >= 3:
        model_id = get_model_id(sys.argv[2])
    else:
        model_id = MODEL_MAP["zimage"]
    
    # 获取预估大小
    expected_size = MODEL_SIZES.get(model_id, 30.0)
    
    print("=" * 60)
    print(f"Model ID: {model_id}")
    print(f"Target directory: {model_dir}")
    print(f"Expected size: ~{format_size(expected_size)}")
    print("=" * 60)
    
    # 检查连接性
    if not check_connectivity(model_id):
        sys.exit(1)

    print("")
    print("[INFO] Starting download process...")
    print("[INFO] Note: If you have downloaded files before, the script will verify them first.")
    print("[INFO] This verification phase consumes Disk I/O but NO Network traffic.")
    print("[INFO] Please wait patiently if the speed shows 0.0 MB/s initially.")
    print("")
    
    # 确保目录存在
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 启动进度监控线程
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=progress_monitor,
        args=(model_dir, expected_size, stop_event),
        daemon=True
    )
    monitor_thread.start()
    



    try:
        from modelscope.hub.snapshot_download import snapshot_download
        
        # 执行下载(阻塞调用)
        # 注意: snapshot_download 会输出自己的进度到 stderr/tqdm
        # 我们的监控线程通过文件大小变化来显示总进度
        print(f"Downloading Model from https://www.modelscope.cn to directory: {model_dir}", flush=True)
        snapshot_download(model_id, local_dir=str(model_dir))
        
        # 停止监控
        stop_event.set()
        monitor_thread.join(timeout=2)
        
        # 最终大小
        final_size = get_dir_size_gb(model_dir)
        
        print("")  # 换行
        print("")
        print("=" * 60)
        print(f"[PROGRESS] Download complete!")
        print(f"[PROGRESS] Total size: {format_size(final_size)}")
        print(f"[PROGRESS] Saved to: {model_dir}")
        print("=" * 60)
        
    except Exception as e:
        stop_event.set()
        monitor_thread.join(timeout=2)
        print("")
        print("")
        print("=" * 60)
        print(f"[ERROR] Download failed: {e}")
        print("=" * 60)
        # 写入 crash log
        with open("download_crash.log", "w") as f:
            f.write(f"Download failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("download_crash.log", "w") as f:
            f.write(f"Script crashed: {e}\n")
        print(f"[CRASH] Script crashed: {e}")
        sys.exit(1)
