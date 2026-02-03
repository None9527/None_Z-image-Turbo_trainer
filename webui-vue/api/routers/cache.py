from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import subprocess
import sys
import os
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from core.config import PROJECT_ROOT, get_model_path
from core import state

# 线程池用于文件 I/O 操作
_file_executor = ThreadPoolExecutor(max_workers=2)

router = APIRouter(prefix="/api/cache", tags=["cache"])

# Windows 兼容性：获取 subprocess 创建标志
def _get_subprocess_flags():
    """获取跨平台 subprocess 创建标志"""
    if os.name == 'nt':
        # Windows: 隐藏命令行窗口
        return subprocess.CREATE_NO_WINDOW
    return 0

class CacheGenerationRequest(BaseModel):
    datasetPath: str
    generateLatent: bool
    generateText: bool
    vaePath: str = ""  # 已弃用，后端根据 modelType 自动获取
    textEncoderPath: str = ""  # 已弃用，后端根据 modelType 自动获取
    modelType: str = "zimage"  # 模型类型: 仅 zimage
    resolution: int = 1024
    batchSize: int = 1
    maxSequenceLength: int = 512  # 文本编码器最大序列长度
    # 配对模式支持
    trainingMode: str = "text2img"  # text2img | img2img | omni | controlnet
    generateSiglip: bool = False  # Omni 模式生成 SigLIP 特征

# 存储待执行的 text cache 参数
_pending_text_cache: dict = {}

def _start_text_cache_after_latent():
    """等待 latent 完成后启动 text cache（后台线程）"""
    global _pending_text_cache
    
    if not _pending_text_cache:
        return
    
    # 等待 latent 完成
    if state.cache_latent_process:
        state.cache_latent_process.wait()
        state.add_log("Latent cache 完成，开始 Text cache...", "info")
    
    # 启动 text cache
    params = _pending_text_cache
    _pending_text_cache = {}
    
    if params:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        env["PYTHONUNBUFFERED"] = "1"
        
        # 使用保存的独立脚本路径
        text_script = params.get("text_script", str(PROJECT_ROOT / "scripts" / "cache_text_encoder_standalone.py"))
        
        cmd_text = [
            sys.executable, text_script,
            "--text_encoder", params["text_encoder"],
            "--input_dir", params["input_dir"],
            "--output_dir", params["output_dir"],
            "--max_length", str(params.get("max_sequence_length", 512)),
            "--skip_existing"
        ]
        
        state.cache_text_process = subprocess.Popen(
            cmd_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            creationflags=_get_subprocess_flags()
        )
        # 创建进程输出读取器
        from routers.websocket import parse_cache_progress
        state.start_process_reader(state.cache_text_process, "cache_text", parse_func=parse_cache_progress)
        state.add_log(f"Text cache started (PID: {state.cache_text_process.pid})", "info")

@router.post("/generate")
async def generate_cache(request: CacheGenerationRequest):
    """Generate latent and/or text encoder cache for a dataset
    
    重要：当同时请求 latent 和 text 时，会**顺序执行**（先 latent 后 text），
    避免低显存机器同时加载 VAE 和 Text Encoder 导致 OOM。
    """
    global _pending_text_cache
    
    # 检查是否有缓存任务正在运行
    if state.cache_latent_process and state.cache_latent_process.poll() is None:
        raise HTTPException(status_code=400, detail="Latent cache generation already in progress")
    if state.cache_text_process and state.cache_text_process.poll() is None:
        raise HTTPException(status_code=400, detail="Text cache generation already in progress")
    
    dataset_path = Path(request.datasetPath)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset path not found")
    
    # 根据模型类型获取正确的模型路径
    model_type = request.modelType
    vae_path = str(get_model_path(model_type, "vae"))
    text_encoder_path = str(get_model_path(model_type, "text_encoder"))
    
    # 检测配对数据集结构
    source_dir = dataset_path / "source"
    target_dir = dataset_path / "target"
    has_paired_structure = source_dir.exists() and target_dir.exists()
    training_mode = request.trainingMode
    
    # 自动检测配对结构
    if has_paired_structure and training_mode == "text2img":
        training_mode = "img2img"
        state.add_log("检测到 source/ + target/ 结构，自动切换到 Img2Img 模式", "info")
    
    state.add_log(f"模型类型: {model_type}, 训练模式: {training_mode}", "info")
    state.add_log(f"VAE: {vae_path}, Text Encoder: {text_encoder_path}", "info")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    
    started_tasks = []
    
    # Latent Cache (先执行)
    if request.generateLatent:
        # 验证 VAE 路径存在
        if not Path(vae_path).exists():
            raise HTTPException(status_code=400, detail=f"VAE path not found: {vae_path}")
        
        # 重置进度（清除上次遗留）
        state.reset_cache_progress("latent")
        
        # 缓存脚本（使用独立脚本，避免触发 __init__.py 导致 CUDA 初始化问题）
        latent_script = str(PROJECT_ROOT / "scripts" / "cache_latents_standalone.py")
        
        # 配对模式：分别缓存 source 和 target 目录
        if has_paired_structure and training_mode in ("img2img", "omni", "controlnet"):
            # 先缓存 source 目录
            cmd_source = [
                sys.executable, latent_script,
                "--vae", vae_path,
                "--input_dir", str(source_dir),
                "--output_dir", str(source_dir),
                "--resolution", str(request.resolution),
                "--batch_size", str(request.batchSize),
                "--skip_existing"
            ]
            # 再缓存 target 目录
            cmd_target = [
                sys.executable, latent_script,
                "--vae", vae_path,
                "--input_dir", str(target_dir),
                "--output_dir", str(target_dir),
                "--resolution", str(request.resolution),
                "--batch_size", str(request.batchSize),
                "--skip_existing"
            ]
            
            state.add_log(f"配对模式: 将分别缓存 source/ 和 target/ 目录", "info")
            
            # 启动 source 缓存（target 将在完成后启动）
            state.cache_latent_process = subprocess.Popen(
                cmd_source,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                creationflags=_get_subprocess_flags()
            )
            from routers.websocket import parse_cache_progress
            state.start_process_reader(state.cache_latent_process, "cache_latent", parse_func=parse_cache_progress)
            state.add_log(f"Source latent cache started (PID: {state.cache_latent_process.pid})", "info")
            
            # 后台等待 source 完成后启动 target
            def _start_target_cache():
                state.cache_latent_process.wait()
                state.add_log("Source 缓存完成，开始 Target 缓存...", "info")
                state.cache_latent_process = subprocess.Popen(
                    cmd_target,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    creationflags=_get_subprocess_flags()
                )
                state.start_process_reader(state.cache_latent_process, "cache_latent", parse_func=parse_cache_progress)
                state.add_log(f"Target latent cache started (PID: {state.cache_latent_process.pid})", "info")
            
            thread = threading.Thread(target=_start_target_cache, daemon=True)
            thread.start()
            started_tasks.append("latent (source + target)")
        else:
            # 标准模式：缓存整个目录
            cmd_latent = [
                sys.executable, latent_script,
                "--vae", vae_path,
                "--input_dir", str(dataset_path),
                "--output_dir", str(dataset_path),
                "--resolution", str(request.resolution),
                "--batch_size", str(request.batchSize),
                "--skip_existing"
            ]
            
            state.cache_latent_process = subprocess.Popen(
                cmd_latent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                creationflags=_get_subprocess_flags()
            )
            # 创建进程输出读取器
            from routers.websocket import parse_cache_progress
            state.start_process_reader(state.cache_latent_process, "cache_latent", parse_func=parse_cache_progress)
            state.add_log(f"Latent cache started (PID: {state.cache_latent_process.pid})", "info")
            started_tasks.append("latent")
    
    # Text Encoder Cache
    if request.generateText:
        # 验证 Text Encoder 路径存在
        if not Path(text_encoder_path).exists():
            raise HTTPException(status_code=400, detail=f"Text Encoder path not found: {text_encoder_path}")
        
        # 重置进度（清除上次遗留）
        state.reset_cache_progress("text")
        
        # text缓存脚本（使用独立脚本）
        text_script = str(PROJECT_ROOT / "scripts" / "cache_text_encoder_standalone.py")
        
        # 如果同时请求了 latent，则排队等待（顺序执行）
        if request.generateLatent:
            _pending_text_cache = {
                "text_encoder": text_encoder_path,  # 使用根据模型类型获取的路径
                "input_dir": str(dataset_path),
                "output_dir": str(dataset_path),
                "text_script": text_script,
                "max_sequence_length": request.maxSequenceLength
            }
            state.add_log("Text cache 已排队，将在 Latent cache 完成后自动开始", "info")
            started_tasks.append("text (queued)")
            
            # 启动后台线程等待 latent 完成
            thread = threading.Thread(target=_start_text_cache_after_latent, daemon=True)
            thread.start()
        else:
            # 只请求 text，直接执行
            cmd_text = [
                sys.executable, text_script,
                "--text_encoder", text_encoder_path,  # 使用根据模型类型获取的路径
                "--input_dir", str(dataset_path),
                "--output_dir", str(dataset_path),
                "--max_length", str(request.maxSequenceLength),
                "--skip_existing"
            ]
            
            state.cache_text_process = subprocess.Popen(
                cmd_text,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                creationflags=_get_subprocess_flags()
            )
            # 创建进程输出读取器
            from routers.websocket import parse_cache_progress
            state.start_process_reader(state.cache_text_process, "cache_text", parse_func=parse_cache_progress)
            state.add_log(f"Text cache started (PID: {state.cache_text_process.pid})", "info")
            started_tasks.append("text")
    
    # SigLIP Cache (Omni 模式专用)
    if request.generateSiglip and training_mode == "omni":
        conditions_dir = dataset_path / "conditions"
        if conditions_dir.exists():
            siglip_path = str(get_model_path(model_type, "siglip"))
            if siglip_path and Path(siglip_path).exists():
                siglip_script = str(PROJECT_ROOT / "src" / "zimage_trainer" / "cache_siglip.py")
                
                cmd_siglip = [
                    sys.executable, "-m", "zimage_trainer.cache_siglip",
                    "--siglip", siglip_path,
                    "--input_dir", str(conditions_dir),
                    "--output_dir", str(conditions_dir),
                    "--skip_existing"
                ]
                
                # SigLIP 缓存在 text 之后顺序执行（显存限制）
                def _start_siglip_after_all():
                    # 等待所有前置任务完成
                    if state.cache_latent_process:
                        state.cache_latent_process.wait()
                    if state.cache_text_process:
                        state.cache_text_process.wait()
                    
                    state.add_log("开始 SigLIP 特征缓存...", "info")
                    siglip_proc = subprocess.Popen(
                        cmd_siglip,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        bufsize=1,
                        cwd=str(PROJECT_ROOT),
                        creationflags=_get_subprocess_flags()
                    )
                    state.add_log(f"SigLIP cache started (PID: {siglip_proc.pid})", "info")
                    siglip_proc.wait()
                    state.add_log("SigLIP 缓存完成", "info")
                
                thread = threading.Thread(target=_start_siglip_after_all, daemon=True)
                thread.start()
                started_tasks.append("siglip (queued)")
                state.add_log("SigLIP cache 已排队，将在其他缓存完成后开始", "info")
            else:
                state.add_log("未找到 SigLIP 模型，跳过 SigLIP 缓存", "warning")
        else:
            state.add_log("未找到 conditions/ 目录，跳过 SigLIP 缓存", "warning")
    
    if not started_tasks:
        raise HTTPException(status_code=400, detail="No cache type selected")
    
    return {
        "success": True, 
        "message": f"Cache generation started: {', '.join(started_tasks)}",
        "tasks": started_tasks
    }

@router.post("/stop")
async def stop_cache():
    """Stop cache generation with proper cleanup"""
    import gc
    stopped = []
    
    if state.cache_latent_process and state.cache_latent_process.poll() is None:
        pid = state.cache_latent_process.pid
        state.cache_latent_process.terminate()
        try:
            state.cache_latent_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            state.cache_latent_process.kill()
            state.cache_latent_process.wait(timeout=5)
        state.cache_latent_process = None
        state.stop_process_reader("cache_latent")  # 停止读取器
        stopped.append("latent")
        state.add_log(f"Latent cache stopped (PID: {pid})", "warning")
    
    if state.cache_text_process and state.cache_text_process.poll() is None:
        pid = state.cache_text_process.pid
        state.cache_text_process.terminate()
        try:
            state.cache_text_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            state.cache_text_process.kill()
            state.cache_text_process.wait(timeout=5)
        state.cache_text_process = None
        state.stop_process_reader("cache_text")  # 停止读取器
        stopped.append("text")
        state.add_log(f"Text cache stopped (PID: {pid})", "warning")
    
    # 清理
    gc.collect()
    state.reset_cache_progress()
    
    return {"success": True, "stopped": stopped}

@router.get("/status")
async def get_cache_status():
    """Get cache generation status"""
    
    def get_process_status(process, name):
        if process is None:
            return {"status": "idle"}
        
        return_code = process.poll()
        if return_code is None:
            return {"status": "running"}
        elif return_code == 0:
            return {"status": "completed"}
        else:
            return {"status": "failed", "code": return_code}
    
    return {
        "latent": get_process_status(state.cache_latent_process, "latent"),
        "text": get_process_status(state.cache_text_process, "text")
    }

class CacheClearRequest(BaseModel):
    datasetPath: str
    clearLatent: bool = False
    clearText: bool = False
    modelType: str = "zimage"  # 模型类型

class CacheCheckRequest(BaseModel):
    datasetPath: str
    modelType: str = "zimage"  # 模型类型

@router.post("/check")
async def check_cache_status(request: CacheCheckRequest):
    """检查数据集的缓存完整性（异步，防止前端卡死）"""
    dataset_path = Path(request.datasetPath)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset path not found")
    
    # 获取模型对应的缓存后缀
    from .training import get_cache_suffixes
    latent_pattern, text_suffix = get_cache_suffixes(request.modelType)
    
    def _do_check():
        """同步检查函数，在线程中执行"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = []
        for f in dataset_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in image_extensions:
                images.append(f)
        
        total_images = len(images)
        latent_cached = 0
        text_cached = 0
        
        for img in images:
            stem = img.stem
            parent = img.parent
            
            # 检查 latent 缓存
            latent_files = list(parent.glob(f"{stem}{latent_pattern}"))
            if latent_files:
                latent_cached += 1
            
            # 检查 text 缓存
            text_files = list(parent.glob(f"{stem}{text_suffix}"))
            if text_files:
                text_cached += 1
        
        return {
            "total_images": total_images,
            "latent_cached": latent_cached,
            "text_cached": text_cached,
            "latent_complete": latent_cached >= total_images,
            "text_complete": text_cached >= total_images,
            "all_complete": latent_cached >= total_images and text_cached >= total_images
        }
    
    # 在线程池中执行，避免阻塞事件循环
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_file_executor, _do_check)
    return result

@router.post("/clear")
async def clear_cache(request: CacheClearRequest):
    """Clear latent and/or text encoder cache for a dataset（异步，防止前端卡死）"""
    dataset_path = Path(request.datasetPath)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset path not found")
    
    # 获取模型对应的缓存后缀用于删除
    from .training import get_cache_suffixes
    latent_pattern, text_suffix = get_cache_suffixes(request.modelType)
    
    # Define patterns to delete
    patterns = []
    if request.clearLatent:
        patterns.append(f"*{latent_pattern}")
    if request.clearText:
        patterns.append(f"*{text_suffix}")
    
    if not patterns:
        return {"success": True, "deleted": 0, "message": "No cache type selected"}
    
    def _do_clear():
        """同步删除函数，在线程中执行"""
        deleted_count = 0
        errors = []
        
        for pattern in patterns:
            for file in dataset_path.rglob(pattern):
                try:
                    file.unlink()
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"{file.name}: {str(e)}")
        
        return deleted_count, errors
    
    try:
        loop = asyncio.get_event_loop()
        deleted_count, errors = await loop.run_in_executor(_file_executor, _do_clear)
        
        state.add_log(f"Cleared {deleted_count} cache files", "info")
        
        return {
            "success": True, 
            "deleted": deleted_count, 
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
