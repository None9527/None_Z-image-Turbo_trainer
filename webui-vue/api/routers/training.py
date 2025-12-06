from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import json
import re
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

from core.config import CONFIGS_DIR, OUTPUT_BASE_DIR, MODEL_PATH, SaveConfigRequest, PROJECT_ROOT
from core import state

router = APIRouter(prefix="/api/training", tags=["training"])

# Windows 兼容性：获取 subprocess 创建标志
def _get_subprocess_flags():
    """获取跨平台 subprocess 创建标志"""
    if os.name == 'nt':
        return subprocess.CREATE_NO_WINDOW
    return 0

def get_default_config():
    """Get default training configuration"""
    return {
        "name": "default",
        "acrf": {
            "turbo_steps": 10,
            "shift": 3.0,
            "jitter_scale": 0.02
        },
        "network": {
            "dim": 16,
            "alpha": 16
        },
        "optimizer": {
            "type": "AdamW8bit"
        },
        "training": {
            "output_name": "zimage-lora",
            "learning_rate": 0.0001,
            "weight_decay": 0,
            "lr_scheduler": "cosine_with_restarts",
            "lr_warmup_steps": 100,
            "lr_num_cycles": 1,
            # Standard 模式参数
            "lambda_fft": 0,
            "lambda_cosine": 0,
            "snr_gamma": 5.0,
            # 损失模式
            "loss_mode": "standard",
            # 频域感知参数
            "alpha_hf": 1.0,
            "beta_lf": 0.2,
            # 风格结构参数
            "lambda_struct": 1.0,
            "lambda_light": 0.5,
            "lambda_color": 0.3,
            "lambda_tex": 0.5
        },
        "dataset": {
            "batch_size": 1,
            "shuffle": True,
            "enable_bucket": True,
            "datasets": []
        },
        "advanced": {
            "num_train_epochs": 10,
            "save_every_n_epochs": 1,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
            "seed": 42
        }
    }

@router.get("/system-paths")
async def get_system_paths():
    """Get system-wide model and output paths"""
    return {
        "model_path": MODEL_PATH,
        "output_base_dir": str(OUTPUT_BASE_DIR)
    }

@router.get("/defaults")
async def get_defaults():
    """Get default configuration for frontend store (flat structure)"""
    return {
        "modelPath": str(MODEL_PATH),
        "vaePath": str(MODEL_PATH / "vae"),
        "textEncoderPath": str(MODEL_PATH / "text_encoder"),
        "outputDir": str(OUTPUT_BASE_DIR),
        "outputName": "zimage-lora",
        "datasetConfigPath": "./dataset_config.toml",
        "cacheDir": "./cache",
        "epochs": 10,
        "batchSize": 1,
        "learningRate": 1e-4,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "warmupSteps": 100,
        "networkDim": 64,
        "networkAlpha": 64,
        "mixedPrecision": "bf16",
        "gradientCheckpointing": True,
        "gradientAccumulationSteps": 1,
        "maxGradNorm": 1.0,
        "seed": 42
    }

@router.get("/configs")
async def list_training_configs():
    """List all saved training configurations"""
    try:
        configs = []
        if CONFIGS_DIR.exists():
            for config_file in CONFIGS_DIR.glob("*.json"):
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    configs.append({
                        "name": config.get("name", config_file.stem),
                        "created": config.get("created", ""),
                        "modified": config.get("modified", "")
                    })
        return {"configs": configs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list configs: {str(e)}")

@router.get("/config/current")
async def get_current_config():
    """Get the most recently modified training configuration for preview"""
    try:
        if not CONFIGS_DIR.exists():
            return get_default_config()
        
        config_files = list(CONFIGS_DIR.glob("*.json"))
        if not config_files:
            return get_default_config()
        
        # 找最近修改的配置
        latest_file = max(config_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return get_default_config()

@router.get("/config/{config_name}")
async def get_training_config(config_name: str):
    """Get a specific training configuration"""
    try:
        if config_name == "default":
            return get_default_config()
        
        config_path = CONFIGS_DIR / f"{config_name}.json"
        if not config_path.exists():
            return get_default_config()
        
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")

@router.post("/config/save")
async def save_training_config(request: SaveConfigRequest):
    """Save a training configuration"""
    try:
        safe_name = re.sub(r'[^\w\-]', '_', request.name)
        config_path = CONFIGS_DIR / f"{safe_name}.json"
        
        config_data = request.config
        config_data["name"] = safe_name
        now = datetime.now().isoformat()
        
        if config_path.exists():
            config_data["modified"] = now
            with open(config_path, "r", encoding="utf-8") as f:
                old_config = json.load(f)
                config_data["created"] = old_config.get("created", now)
        else:
            config_data["created"] = now
            config_data["modified"] = now
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return {"success": True, "name": safe_name, "path": str(config_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")

@router.delete("/config/{config_name}")
async def delete_training_config(config_name: str):
    """Delete a training configuration"""
    try:
        if config_name == "default":
            raise HTTPException(status_code=400, detail="Cannot delete default config")
        
        config_path = CONFIGS_DIR / f"{config_name}.json"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Config not found")
        
        config_path.unlink()
        return {"success": True, "message": f"Config '{config_name}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete config: {str(e)}")

@router.get("/presets")
async def get_training_presets():
    """Get preset training configurations"""
    presets = [
        {
            "name": "快速测试",
            "description": "小批次、少epoch、快速迭代",
            "config": {
                **get_default_config(),
                "name": "fast_test",
                "advanced": {
                    **get_default_config()["advanced"],
                    "num_train_epochs": 3,
                    "gradient_accumulation_steps": 2
                }
            }
        },
        {
            "name": "标准训练",
            "description": "平衡的训练配置",
            "config": {
                **get_default_config(),
                "name": "standard",
                "lora": {
                    "network_dim": 16,
                    "network_alpha": 8.0
                }
            }
        },
        {
            "name": "高质量",
            "description": "大Rank、长训练、高质量输出",
            "config": {
                **get_default_config(),
                "name": "high_quality",
                "lora": {
                    "network_dim": 32,
                    "network_alpha": 16.0
                },
                "training": {
                    **get_default_config()["training"],
                    "learning_rate": 0.00005
                },
                "advanced": {
                    **get_default_config()["advanced"],
                    "num_train_epochs": 20,
                    "gradient_accumulation_steps": 8
                }
            }
        }
    ]
    return {"presets": presets}

def check_dataset_cache(config: Dict[str, Any]) -> Dict[str, Any]:
    """检查数据集缓存状态（同步函数）"""
    dataset_cfg = config.get("dataset", {})
    datasets = dataset_cfg.get("datasets", [])
    
    if not datasets:
        return {"has_cache": True, "message": "No datasets configured"}
    
    total_images = 0
    latent_cached = 0
    text_cached = 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    for ds in datasets:
        cache_dir = ds.get("cache_directory", "")
        if not cache_dir:
            continue
        
        dataset_path = Path(cache_dir)
        if not dataset_path.exists():
            continue
        
        # 统计图片和缓存
        for f in dataset_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in image_extensions:
                total_images += 1
                stem = f.stem
                parent = f.parent
                
                # 检查 latent 缓存
                if list(parent.glob(f"{stem}_*_zi.safetensors")):
                    latent_cached += 1
                
                # 检查 text 缓存 (格式: {name}_zi_te.safetensors)
                if list(parent.glob(f"{stem}_zi_te.safetensors")):
                    text_cached += 1
    
    has_cache = latent_cached > 0 and text_cached > 0
    
    return {
        "has_cache": has_cache,
        "total_images": total_images,
        "latent_cached": latent_cached,
        "text_cached": text_cached,
        "latent_missing": total_images - latent_cached,
        "text_missing": total_images - text_cached
    }


@router.post("/start")
async def start_training(config: Dict[str, Any]):
    """Start AC-RF training with accelerate launch
    
    如果缓存不完整，返回 needs_cache=True，前端应该先执行缓存
    """
    if state.training_process and state.training_process.poll() is None:
        raise HTTPException(status_code=400, detail="训练已在运行中")
    
    try:
        # 清空训练历史（新训练开始）
        state.clear_training_history()
        state.training_logs.clear()
        
        # 广播清空图表消息给前端
        from .websocket import manager
        import asyncio
        asyncio.create_task(manager.broadcast({
            "type": "training_reset",
            "training_history": state.get_training_history()  # 发送空的历史数据
        }))
        
        # 如果有生成模型加载，先卸载释放显存
        if state.pipeline is not None:
            state.add_log("检测到生成模型已加载，正在卸载以释放显存...", "info")
            import gc
            if state.current_lora_path:
                try:
                    state.pipeline.unload_lora_weights()
                except:
                    pass
                state.current_lora_path = None
            del state.pipeline
            state.pipeline = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            state.add_log("生成模型已卸载", "info")
        
        # 检查缓存状态
        cache_info = check_dataset_cache(config)
        if not cache_info["has_cache"]:
            state.add_log(f"缓存不完整: Latent {cache_info['latent_cached']}/{cache_info['total_images']}, Text {cache_info['text_cached']}/{cache_info['total_images']}", "warning")
            return {
                "success": False,
                "needs_cache": True,
                "total_images": cache_info["total_images"],
                "latent_cached": cache_info["latent_cached"],
                "text_cached": cache_info["text_cached"],
                "latent_missing": cache_info["latent_missing"],
                "text_missing": cache_info["text_missing"],
                "message": f"缓存不完整，需要先生成缓存"
            }
        
        state.add_log(f"缓存检查通过: {cache_info['latent_cached']} latent, {cache_info['text_cached']} text", "info")
        
        # 生成 TOML 配置文件
        config_path = CONFIGS_DIR / "current_training.toml"
        toml_content = generate_acrf_toml_config(config)
        
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(toml_content)
        
        state.add_log(f"AC-RF 配置已生成: {config_path}", "info")
        
        # 构建 accelerate launch 命令
        # 使用 scripts/train_acrf.py (AC-RF 训练脚本)
        python_exe = sys.executable
        train_script = PROJECT_ROOT / "scripts" / "train_acrf.py"
        
        mixed_precision = config.get("advanced", {}).get("mixed_precision", "bf16")
        
        cmd = [
            python_exe, "-m", "accelerate.commands.launch",
            "--mixed_precision", mixed_precision,
            str(train_script),
            "--config", str(config_path)
        ]
        
        state.add_log(f"启动命令: {' '.join(cmd)}", "info")
        state.add_log(f"混合精度: {mixed_precision}", "info")
        
        # 启动训练进程
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        state.training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            creationflags=_get_subprocess_flags()
        )
        
        state.add_log(f"AC-RF 训练进程已启动 (PID: {state.training_process.pid})", "success")
        
        return {
            "success": True,
            "message": "AC-RF 训练已启动",
            "pid": state.training_process.pid,
            "config_path": str(config_path)
        }
        
    except Exception as e:
        state.add_log(f"启动失败: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")


def generate_acrf_toml_config(config: Dict[str, Any]) -> str:
    """
    将前端配置转换为 AC-RF 训练 TOML 格式（包含 dataset 配置）
    
    匹配 scripts/train_acrf.py 的参数格式
    所有配置合并到一个 TOML 文件中
    """
    
    # 获取数据集配置
    dataset_cfg = config.get("dataset", {})
    datasets = dataset_cfg.get("datasets", [])
    
    toml_lines = [
        "# Z-Image AC-RF 训练配置 (自动生成)",
        f"# 生成时间: {datetime.now().isoformat()}",
        "",
        "[model]",
        f'dit = "{str(MODEL_PATH / "transformer").replace(chr(92), "/")}"',
        f'output_dir = "{str(OUTPUT_BASE_DIR).replace(chr(92), "/")}"',
        "",
        "[acrf]",
        f"turbo_steps = {config.get('acrf', {}).get('turbo_steps', 10)}",
        f"shift = {config.get('acrf', {}).get('shift', 3.0)}",
        f"jitter_scale = {config.get('acrf', {}).get('jitter_scale', 0.02)}",
        "",
        "[lora]",
        f"network_dim = {config.get('network', {}).get('dim', 8)}",
        f"network_alpha = {config.get('network', {}).get('alpha', 4.0)}",
        "",
        "[training]",
        f'output_name = "{config.get("training", {}).get("output_name", "zimage-lora")}"',
        f'optimizer_type = "{config.get("optimizer", {}).get("type", "AdamW8bit")}"',
        f"learning_rate = {config.get('training', {}).get('learning_rate', 0.0001)}",
        f"weight_decay = {config.get('training', {}).get('weight_decay', 0.01)}",
        f'lr_scheduler = "{config.get("training", {}).get("lr_scheduler", "constant")}"',
        f"lr_warmup_steps = {config.get('training', {}).get('lr_warmup_steps', 0)}",
        f"lr_num_cycles = {config.get('training', {}).get('lr_num_cycles', 1)}",
        # Standard 模式参数
        f"lambda_fft = {config.get('training', {}).get('lambda_fft', 0)}",
        f"lambda_cosine = {config.get('training', {}).get('lambda_cosine', 0)}",
        f"snr_gamma = {config.get('training', {}).get('snr_gamma', 5.0)}",
        # 损失模式
        f'loss_mode = "{config.get("training", {}).get("loss_mode", "standard")}"',
        # 频域感知参数
        f"alpha_hf = {config.get('training', {}).get('alpha_hf', 1.0)}",
        f"beta_lf = {config.get('training', {}).get('beta_lf', 0.2)}",
        # 风格结构参数
        f"lambda_struct = {config.get('training', {}).get('lambda_struct', 1.0)}",
        f"lambda_light = {config.get('training', {}).get('lambda_light', 0.5)}",
        f"lambda_color = {config.get('training', {}).get('lambda_color', 0.3)}",
        f"lambda_tex = {config.get('training', {}).get('lambda_tex', 0.5)}",
        f"num_train_epochs = {config.get('advanced', {}).get('num_train_epochs', 10)}",
        f"save_every_n_epochs = {config.get('advanced', {}).get('save_every_n_epochs', 1)}",
        f"gradient_accumulation_steps = {config.get('advanced', {}).get('gradient_accumulation_steps', 4)}",
        f'mixed_precision = "{config.get("advanced", {}).get("mixed_precision", "bf16")}"',
        f"seed = {config.get('advanced', {}).get('seed', 42)}",
        "",
        "[advanced]",
        f"max_grad_norm = {config.get('advanced', {}).get('max_grad_norm', 1.0)}",
        f"gradient_checkpointing = {'true' if config.get('advanced', {}).get('gradient_checkpointing', True) else 'false'}",
        "",
        "[optimization]",
        "auto_optimize = true",
        "",
        "# ============ Dataset 配置 ============",
        "[dataset]",
        f"batch_size = {dataset_cfg.get('batch_size', 1)}",
        f"shuffle = {'true' if dataset_cfg.get('shuffle', True) else 'false'}",
        f"enable_bucket = {'true' if dataset_cfg.get('enable_bucket', True) else 'false'}",
        "",
    ]
    
    # 添加数据集源（带完整配置）
    for ds in datasets:
        cache_dir = ds.get("cache_directory", "")
        if cache_dir:
            toml_lines.extend([
                "[[dataset.sources]]",
                f'cache_directory = "{cache_dir.replace(chr(92), "/")}"',
                f"num_repeats = {ds.get('num_repeats', 1)}",
                f"resolution_limit = {ds.get('resolution_limit', 1024)}",
                "",
            ])
    
    return "\n".join(toml_lines)


@router.post("/stop")
async def stop_training():
    """Stop training process with proper cleanup"""
    if state.training_process:
        pid = state.training_process.pid
        state.add_log(f"正在停止训练进程 (PID: {pid})...", "warning")
        
        # 先尝试优雅终止
        state.training_process.terminate()
        
        # 等待进程结束（最多10秒）
        try:
            state.training_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # 强制杀死
            state.training_process.kill()
            state.training_process.wait(timeout=5)
            state.add_log("进程被强制终止", "warning")
        
        state.training_process = None
        state.add_log("训练已停止，显存将被释放", "warning")
        
        # 清理 Python 端的缓存
        import gc
        gc.collect()
        
    return {"status": "stopped"}

@router.get("/status")
async def get_training_status():
    """Get current training status"""
    is_running = state.training_process is not None and state.training_process.poll() is None
    return {
        "running": is_running,
        "pid": state.training_process.pid if state.training_process else None
    }

@router.get("/logs")
async def get_logs():
    """Get training logs"""
    return {"logs": state.training_logs[-100:]}


@router.websocket("/ws")
async def training_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time training updates"""
    await websocket.accept()
    state.training_websockets.append(websocket)
    
    try:
        # 发送当前状态
        is_running = state.training_process is not None and state.training_process.poll() is None
        await websocket.send_json({
            "type": "status",
            "payload": {"running": is_running}
        })
        
        # 发送历史日志
        for log in state.training_logs[-50:]:
            await websocket.send_json({
                "type": "log",
                "payload": log
            })
        
        # 保持连接，等待消息或断开
        while True:
            try:
                # 读取训练进程输出（如果正在运行）
                if state.training_process and state.training_process.poll() is None:
                    if state.training_process.stdout:
                        line = state.training_process.stdout.readline()
                        if line:
                            state.add_log(line.strip(), "info")
                            await websocket.send_json({
                                "type": "log",
                                "payload": {"message": line.strip(), "level": "info"}
                            })
                
                # 检查训练是否完成
                if state.training_process and state.training_process.poll() is not None:
                    exit_code = state.training_process.poll()
                    if exit_code == 0:
                        await websocket.send_json({"type": "complete", "payload": {}})
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "payload": {"message": f"训练进程退出，代码: {exit_code}"}
                        })
                    state.training_process = None
                
                # 等待客户端消息（带超时）
                import asyncio
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    # 处理客户端消息（如心跳）
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    pass  # 超时继续循环
                    
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        state.add_log(f"WebSocket 错误: {str(e)}", "error")
    finally:
        if websocket in state.training_websockets:
            state.training_websockets.remove(websocket)
