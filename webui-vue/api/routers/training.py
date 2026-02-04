from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import json
import re
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

from core.config import CONFIGS_DIR, OUTPUT_BASE_DIR, MODEL_PATH, SaveConfigRequest, PROJECT_ROOT, get_model_path, LORA_PATH, FINETUNE_PATH, CONTROLNET_PATH, LOGS_PATH
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
        # 模型类型：zimage (仅支持)
        "model_type": "zimage",
        "acrf": {
            "turbo_steps": 10,
            "shift": 3.0,
            "jitter_scale": 0.02,
            # Min-SNR 加权参数（所有 loss 模式通用）
            "snr_gamma": 5.0,
            "snr_floor": 0.1,
            # 是否使用锚点采样
            "use_anchor": True,
            # 动态 shift
            "use_dynamic_shifting": True
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
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "lr_num_cycles": 1,
            # 基础损失权重
            "lambda_l1": 1.0,
            "lambda_cosine": 0.1,
            # 频域增强 (开关+权重+子参数)
            "enable_freq": False,
            "lambda_freq": 0.3,
            "alpha_hf": 1.0,
            "beta_lf": 0.2,
            # 风格学习 (开关+权重+子参数)
            "enable_style": False,
            "lambda_style": 0.3,
            "lambda_light": 0.5,
            "lambda_color": 0.3,
            # 兼容旧参数
            "lambda_fft": 0
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

def get_cache_suffixes(model_type: str = "zimage") -> tuple:
    """获取模型对应的缓存文件后缀
    
    Args:
        model_type: 模型类型 (仅支持 zimage)
    
    Returns:
        (latent_suffix_pattern, text_suffix)
    """
    # 固定返回 zimage 后缀
    return ("_*_zi.safetensors", "_zi_te.safetensors")


    
def check_dataset_cache(config: Dict[str, Any], model_type: str = "zimage") -> Dict[str, Any]:
    """检查数据集缓存状态
    
    Args:
        config: 训练配置
        model_type: 模型类型 (仅支持 zimage)
    """
    dataset_cfg = config.get("dataset", {})
    datasets = dataset_cfg.get("datasets", [])
    
    if not datasets:
        return {"has_cache": True, "message": "No datasets configured"}
    
    # 获取模型对应的缓存后缀
    latent_pattern, text_suffix = get_cache_suffixes(model_type)
    
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
        
        # 方法1: 统计源图片数量
        source_images = list(dataset_path.rglob("*"))
        source_images = [f for f in source_images if f.is_file() and f.suffix.lower() in image_extensions]
        total_images += len(source_images)
        
        # 方法2: 直接统计缓存文件数量 (更可靠)
        # Latent 缓存: *_*_zi.safetensors 或 *_*_lc.safetensors
        latent_files = list(dataset_path.rglob(f"*{latent_pattern}"))
        latent_cached += len(latent_files)
        
        # Text 缓存: *_zi_te.safetensors 或 *_lc_te.safetensors
        text_files = list(dataset_path.rglob(f"*{text_suffix}"))
        text_cached += len(text_files)
    
    # 如果没有源图片但有缓存，使用缓存数量作为基准
    # (用户可能只复制了缓存文件)
    if total_images == 0 and latent_cached > 0:
        total_images = latent_cached
    
    has_cache = latent_cached > 0 and text_cached > 0
    
    return {
        "has_cache": has_cache,
        "total_images": total_images,
        "latent_cached": latent_cached,
        "text_cached": text_cached,
        "latent_missing": max(0, total_images - latent_cached),
        "text_missing": max(0, total_images - text_cached),
        "model_type": model_type
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
        
        # 获取模型类型
        model_type = config.get("model_type", "zimage")
        
        # 检查缓存状态(传递model_type)
        cache_info = check_dataset_cache(config, model_type)
        if not cache_info.get("has_cache", False):
            state.add_log(f"缓存不完整 ({model_type}): Latent {cache_info.get('latent_cached', 0)}/{cache_info.get('total_images', 0)}, Text {cache_info.get('text_cached', 0)}/{cache_info.get('total_images', 0)}", "warning")
            return {
                "success": False,
                "needs_cache": True,
                "total_images": cache_info.get("total_images", 0),
                "latent_cached": cache_info.get("latent_cached", 0),
                "text_cached": cache_info.get("text_cached", 0),
                "latent_missing": cache_info.get("latent_missing", 0),
                "text_missing": cache_info.get("text_missing", 0),
                "message": f"缓存不完整，需要先生成缓存"
            }
        
        state.add_log(f"缓存检查通过: {cache_info.get('latent_cached', 0)} latent, {cache_info.get('text_cached', 0)} text", "info")
        
        # 获取模型类型
        model_type = config.get("model_type", "zimage")
        
        # 生成 TOML 配置文件
        config_path = CONFIGS_DIR / "current_training.toml"
        toml_content = generate_training_toml_config(config, model_type)
        
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(toml_content)
        
        state.add_log(f"训练配置已生成: {config_path} (模型: {model_type})", "info")
        
        # 构建 accelerate launch 命令
        python_exe = sys.executable
        mixed_precision = config.get("advanced", {}).get("mixed_precision", "bf16")
        
        # 多卡训练参数
        num_gpus = config.get("advanced", {}).get("num_gpus", 1)
        gpu_ids = config.get("advanced", {}).get("gpu_ids", "")  # 如 "0,1,2"
        
        # 根据训练类型选择对应的训练脚本
        training_type = config.get("training_type", "lora")
        condition_mode = config.get("condition_mode", "text2img")
        
        if training_type == "lora":
            # LoRA训练（默认）
            if condition_mode == "omni":
                train_script = PROJECT_ROOT / "scripts" / "train_zimage_omni.py"
                state.add_log("训练模式: LoRA + Omni (SigLIP多图条件)", "info")
            elif condition_mode == "img2img":
                train_script = PROJECT_ROOT / "scripts" / "train_zimage_img2img.py"
                state.add_log("训练模式: LoRA + Img2Img (图像转换)", "info")
            else:
                # text2img (默认)
                train_script = PROJECT_ROOT / "scripts" / "train_zimage_v2.py"
                state.add_log(f"训练模式: LoRA + Text2Img", "info")
        elif training_type == "finetune":
            # 全量微调 - 支持三种条件模式
            if condition_mode == "omni":
                train_script = PROJECT_ROOT / "scripts" / "train_full_finetune_omni.py"
                state.add_log("训练模式: Finetune + Omni (全量微调，显存需求高)", "warning")
            elif condition_mode == "img2img":
                train_script = PROJECT_ROOT / "scripts" / "train_full_finetune_img2img.py"
                state.add_log("训练模式: Finetune + Img2Img (全量微调，显存需求高)", "warning")
            else:
                # text2img (默认)
                train_script = PROJECT_ROOT / "scripts" / "train_full_finetune.py"
                state.add_log("训练模式: Finetune + Text2Img (全量微调，显存需求高)", "warning")
        elif training_type == "controlnet":
            # ControlNet训练 (独立模式，不受 condition_mode 影响)
            train_script = PROJECT_ROOT / "scripts" / "train_controlnet.py"
            state.add_log("训练模式: ControlNet (独立控制网络)", "info")
        else:
            # 默认使用LoRA + Text2Img
            train_script = PROJECT_ROOT / "scripts" / "train_zimage_v2.py"
            state.add_log(f"未知训练类型 {training_type}，使用默认LoRA脚本", "warning")
        
        # 构建 accelerate 参数
        accelerate_args = [
            python_exe, "-m", "accelerate.commands.launch",
            "--mixed_precision", mixed_precision,
        ]
        
        # 多卡配置
        if num_gpus > 1:
            accelerate_args.extend(["--multi_gpu", "--num_processes", str(num_gpus)])
            if gpu_ids:
                # 设置 CUDA_VISIBLE_DEVICES
                state.add_log(f"多卡训练: {num_gpus} GPUs, GPU IDs: {gpu_ids}", "info")
            else:
                state.add_log(f"多卡训练: {num_gpus} GPUs (自动选择)", "info")
        elif gpu_ids:
            # 单卡但指定了 ID
            state.add_log(f"单卡训练: GPU {gpu_ids}", "info")
        
        # 添加训练脚本和配置
        cmd = accelerate_args + [str(train_script), "--config", str(config_path)]
        
        state.add_log(f"启动命令: {' '.join(cmd)}", "info")
        state.add_log(f"模型类型: {model_type}, 混合精度: {mixed_precision}", "info")
        
        # 启动训练进程
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        # 设置 GPU ID
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids
            state.add_log(f"CUDA_VISIBLE_DEVICES={gpu_ids}", "info")
        
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
        
        # 创建进程输出读取器（独立线程持续读取，防止管道阻塞）
        from routers.websocket import parse_training_progress
        state.start_process_reader(
            state.training_process, 
            "training", 
            parse_func=parse_training_progress
        )
        
        state.add_log(f"Z-Image 训练进程已启动 (PID: {state.training_process.pid})", "success")
        
        return {
            "success": True,
            "message": "Z-Image 训练已启动",
            "pid": state.training_process.pid,
            "config_path": str(config_path),
            "model_type": model_type
        }
        
    except Exception as e:
        state.add_log(f"启动失败: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")


def generate_training_toml_config(config: Dict[str, Any], model_type: str = "zimage") -> str:
    """
    将前端配置转换为训练 TOML 格式
    
    Args:
        config: 前端配置
        model_type: 模型类型 (仅支持 zimage)
    
    Returns:
        TOML 配置字符串
    """
    
    # 获取训练类型（提前定义，后续多处使用）
    training_type = config.get("training_type", "lora")
    
    # 获取数据集配置
    dataset_cfg = config.get("dataset", {})
    datasets = dataset_cfg.get("datasets", [])
    
    
    toml_lines = [
        "# Z-Image 训练配置 (自动生成)",
        f"# 生成时间: {datetime.now().isoformat()}",
        "",
        "[general]",
        f'model_type = "{model_type}"',
        f'training_type = "{config.get("training_type", "lora")}"',
        f'condition_mode = "{config.get("condition_mode", "text2img")}"',
        f'dit = "{str(get_model_path(model_type, "transformer")).replace(chr(92), "/")}"',
        # 根据 training_type 设置输出子目录（相对于 OUTPUT_PATH 环境变量）
        f'output_dir = "{training_type if training_type in ["lora", "finetune", "controlnet"] else "lora"}"',
        "",
        "[acrf]",
        f"enable_turbo = {'false' if training_type == 'controlnet' else ('true' if config.get('acrf', {}).get('enable_turbo', True) else 'false')}",
        f"turbo_steps = {config.get('acrf', {}).get('turbo_steps', 10)}",
        f"shift = {config.get('acrf', {}).get('shift', 3.0)}",
        f"jitter_scale = {config.get('acrf', {}).get('jitter_scale', 0.02)}",
        f"snr_gamma = {config.get('acrf', {}).get('snr_gamma', 5.0)}",
        f"snr_floor = {config.get('acrf', {}).get('snr_floor', 0.1)}",
        f"use_anchor = {'false' if training_type == 'controlnet' else ('true' if config.get('acrf', {}).get('use_anchor', True) else 'false')}",
        f"use_dynamic_shifting = {'true' if config.get('acrf', {}).get('use_dynamic_shifting', True) else 'false'}",
        f"base_shift = {config.get('acrf', {}).get('base_shift', 0.5)}",
        f"max_shift = {config.get('acrf', {}).get('max_shift', 1.15)}",
        # RAFT 混合模式参数
        f"raft_mode = {'true' if config.get('acrf', {}).get('raft_mode', False) else 'false'}",
        f"free_stream_ratio = {config.get('acrf', {}).get('free_stream_ratio', 0.3)}",
        # L2 调度参数
        f'l2_schedule_mode = "{config.get("acrf", {}).get("l2_schedule_mode", "constant")}"',
        f"l2_initial_ratio = {config.get('acrf', {}).get('l2_initial_ratio', 0.3)}",
        f"l2_final_ratio = {config.get('acrf', {}).get('l2_final_ratio', 0.3)}",
        f'l2_milestones = "{config.get("acrf", {}).get("l2_milestones", "")}"',
        f"l2_include_anchor = {'true' if config.get('acrf', {}).get('l2_include_anchor', False) else 'false'}",
        f"l2_anchor_ratio = {config.get('acrf', {}).get('l2_anchor_ratio', 0.3)}",
        # Latent Jitter (构图突破)
        f"latent_jitter_scale = {config.get('acrf', {}).get('latent_jitter_scale', 0.0)}",
        # 时间步感知 Loss 权重
        f"enable_timestep_aware_loss = {'true' if config.get('acrf', {}).get('enable_timestep_aware_loss', False) else 'false'}",
        f"timestep_high_threshold = {config.get('acrf', {}).get('timestep_high_threshold', 0.7)}",
        f"timestep_low_threshold = {config.get('acrf', {}).get('timestep_low_threshold', 0.3)}",
        # 曲率惩罚
        f"enable_curvature = {'true' if config.get('acrf', {}).get('enable_curvature', False) else 'false'}",
        f"lambda_curvature = {config.get('acrf', {}).get('lambda_curvature', 0.05)}",
        f"curvature_interval = {config.get('acrf', {}).get('curvature_interval', 10)}",
        f"curvature_start_epoch = {config.get('acrf', {}).get('curvature_start_epoch', 0)}",
        # CFG Training
        f"cfg_training = {'true' if config.get('acrf', {}).get('cfg_training', False) else 'false'}",
        f"cfg_scale = {config.get('acrf', {}).get('cfg_scale', 7.0)}",
        f"cfg_training_ratio = {config.get('acrf', {}).get('cfg_training_ratio', 0.3)}",
        "",
    ]
    
    # [lora] 部分 - 根据继续训练模式决定输出内容
    lora_cfg = config.get("lora", {})
    resume_training = lora_cfg.get("resume_training", False)
    resume_lora_path = lora_cfg.get("resume_lora_path", "")
    
    if resume_training and resume_lora_path:
        # 继续训练模式：只输出 resume_lora 路径，不输出 rank/层设置
        toml_lines.extend([
            "[lora]",
            f'resume_lora = "{resume_lora_path.replace(chr(92), "/")}"',
            "# Rank 和层设置将从 LoRA 文件自动读取",
        ])
    else:
        # 新建 LoRA 模式：输出完整设置
        toml_lines.extend([
            "[lora]",
            f"network_dim = {config.get('network', {}).get('dim', 8)}",
            f"network_alpha = {config.get('network', {}).get('alpha', 4.0)}",
            f"train_adaln = {'true' if lora_cfg.get('train_adaln', False) else 'false'}",
            f"train_norm = {'true' if lora_cfg.get('train_norm', False) else 'false'}",
            f"train_single_stream = {'true' if lora_cfg.get('train_single_stream', False) else 'false'}",
        ])
    
    # [controlnet] 部分 - 仅在 training_type 为 controlnet 时输出
    training_type = config.get("training_type", "lora")
    if training_type == "controlnet":
        controlnet_cfg = config.get("controlnet", {})
        toml_lines.extend([
            "",
            "[controlnet]",
            f"conditioning_scale = {controlnet_cfg.get('conditioning_scale', 0.75)}",
            f'controlnet_path = "{controlnet_cfg.get("controlnet_path", "").replace(chr(92), "/")}"',
        ])
    
    toml_lines.extend([
        "",
        "[training]",
        f'output_name = "{config.get("training", {}).get("output_name", "zimage-lora")}"',
        f'optimizer_type = "{config.get("optimizer", {}).get("type", "AdamW8bit")}"',
        f"adafactor_relative_step = {'true' if config.get('optimizer', {}).get('relative_step', False) else 'false'}",
        f"learning_rate = {config.get('training', {}).get('learning_rate', 0.0001)}",
        f"weight_decay = {config.get('training', {}).get('weight_decay', 0.01)}",
        f'lr_scheduler = "{config.get("training", {}).get("lr_scheduler", "constant")}"',
        f"lr_warmup_steps = {config.get('training', {}).get('lr_warmup_steps', 0)}",
        f"lr_num_cycles = {config.get('training', {}).get('lr_num_cycles', 1)}",
        # 基础损失权重
        f"lambda_l1 = {config.get('training', {}).get('lambda_l1', 1.0)}",
        f"lambda_cosine = {config.get('training', {}).get('lambda_cosine', 0.1)}",
        # 频域增强 (开关+权重+子参数)
        f"enable_freq = {'true' if config.get('training', {}).get('enable_freq', False) else 'false'}",
        f"lambda_freq = {config.get('training', {}).get('lambda_freq', 0.3)}",
        f"alpha_hf = {config.get('training', {}).get('alpha_hf', 1.0)}",
        f"beta_lf = {config.get('training', {}).get('beta_lf', 0.2)}",
        # 风格学习 (开关+权重+子参数)
        f"enable_style = {'true' if config.get('training', {}).get('enable_style', False) else 'false'}",
        f"lambda_style = {config.get('training', {}).get('lambda_style', 0.3)}",
        f"lambda_light = {config.get('training', {}).get('lambda_light', 0.5)}",
        f"lambda_color = {config.get('training', {}).get('lambda_color', 0.3)}",
        f"num_train_epochs = {config.get('advanced', {}).get('num_train_epochs', 10)}",
        f"gradient_accumulation_steps = {config.get('advanced', {}).get('gradient_accumulation_steps', 4)}",
        f'mixed_precision = "{config.get("advanced", {}).get("mixed_precision", "bf16")}"',
        f"seed = {config.get('advanced', {}).get('seed', 42)}",
        "",
        "[advanced]",
        f"save_every_n_epochs = {config.get('advanced', {}).get('save_every_n_epochs', 1)}",
        f"max_grad_norm = {config.get('advanced', {}).get('max_grad_norm', 1.0)}",
        f"gradient_checkpointing = {'true' if config.get('advanced', {}).get('gradient_checkpointing', True) else 'false'}",
        f"blocks_to_swap = {config.get('advanced', {}).get('blocks_to_swap', 0)}",
        "",
        "[optimization]",
        "auto_optimize = true",
        "",
        "# ============ Dataset 配置 ============",
        "[dataset]",
        f"batch_size = {dataset_cfg.get('batch_size', 1)}",
        f"shuffle = {'true' if dataset_cfg.get('shuffle', True) else 'false'}",
        f"enable_bucket = {'true' if dataset_cfg.get('enable_bucket', True) else 'false'}",
        f"drop_text_ratio = {dataset_cfg.get('drop_text_ratio', 0.1)}",
        "",
    ])
    
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
    
    # 正则数据集配置（防止过拟合）
    reg_dataset = config.get("reg_dataset", {})
    reg_enabled = reg_dataset.get("enabled", False)
    reg_datasets = reg_dataset.get("datasets", [])
    
    if reg_enabled and reg_datasets:
        toml_lines.extend([
            "",
            "# ============ 正则数据集配置 (Regularization) ============",
            "[reg_dataset]",
            "enabled = true",
            f"weight = {reg_dataset.get('weight', 1.0)}",  # 正则数据权重
            f"ratio = {reg_dataset.get('ratio', 0.5)}",   # 正则数据占比 (0.5 = 1:1 混合)
            "",
        ])
        
        for rds in reg_datasets:
            reg_cache_dir = rds.get("cache_directory", "")
            if reg_cache_dir:
                toml_lines.extend([
                    "[[reg_dataset.sources]]",
                    f'cache_directory = "{reg_cache_dir.replace(chr(92), "/")}"',
                    f"num_repeats = {rds.get('num_repeats', 1)}",
                    "",
                ])
    
    return "\n".join(toml_lines)


@router.post("/stop")
async def stop_training():
    """Stop training process with proper cleanup (Kill Process Tree)"""
    if state.training_process:
        pid = state.training_process.pid
        
        # Safety Guard: Never kill self
        if pid == os.getpid():
            state.add_log("错误: 试图终止主进程，操作已拦截", "error")
            return {"status": "error", "message": "Cannot kill main process"}
            
        state.add_log(f"正在停止训练进程树 (PID: {pid})...", "warning")
        
        import psutil
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
            
            # Wait for termination
            psutil.wait_procs(children + [parent], timeout=5)
            
        except psutil.NoSuchProcess:
            state.add_log("进程已不存在", "warning")
        except Exception as e:
            state.add_log(f"停止进程树失败: {str(e)}", "error")
            # Fallback
            try:
                state.training_process.kill()
            except:
                pass
        
        state.training_process = None
        state.stop_process_reader("training")  # 停止读取器线程
        state.add_log("训练已停止，进程树已清理", "warning")
        
        # 清理 Python 端的缓存
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
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


# ============================================================================
# TensorBoard 日志查询 API
# ============================================================================

@router.get("/runs")
async def list_training_runs():
    """列出所有训练记录（从TensorBoard日志目录扫描）"""
    try:
        from utils.tensorboard_parser import list_training_runs as scan_runs
        
        # TensorBoard 日志保存在 output/logs 目录
        logs_dir = LOGS_PATH
        runs = scan_runs(logs_dir)
        
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"扫描训练记录失败: {str(e)}")


@router.delete("/runs/{run_name}")
async def delete_training_run(run_name: str):
    """删除指定训练记录的 TensorBoard 日志

    Args:
        run_name: 训练记录名称
    """
    import shutil
    
    logs_dir = LOGS_PATH / run_name
    
    if not logs_dir.exists():
        raise HTTPException(status_code=404, detail=f"训练记录 '{run_name}' 不存在")
    
    try:
        shutil.rmtree(logs_dir)
        return {"message": f"成功删除训练记录: {run_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.get("/scalars")
async def get_training_scalars(run: str = "", tag: str = "train/loss"):
    """获取指定训练记录的标量数据
    
    Args:
        run: 训练记录名称（默认使用最新）
        tag: 标量标签名称（默认 train/loss）
    """
    try:
        from utils.tensorboard_parser import get_scalar_data, list_training_runs as scan_runs
        
        logs_dir = LOGS_PATH
        
        # 如果未指定run，使用最新的
        if not run:
            runs = scan_runs(logs_dir)
            if not runs:
                return {"data": [], "run": None, "tag": tag}
            run = runs[0]["name"]
        
        logdir = str(logs_dir / run)
        data = get_scalar_data(logdir, tag)
        
        return {"data": data, "run": run, "tag": tag}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取标量数据失败: {str(e)}")


@router.get("/tags")
async def get_available_tags(run: str = ""):
    """获取指定训练记录的所有可用标量标签
    
    Args:
        run: 训练记录名称（默认使用最新）
    """
    try:
        from utils.tensorboard_parser import get_available_tags as fetch_tags, list_training_runs as scan_runs
        
        logs_dir = LOGS_PATH
        
        # 如果未指定run，使用最新的
        if not run:
            runs = scan_runs(logs_dir)
            if not runs:
                return {"tags": [], "run": None}
            run = runs[0]["name"]
        
        logdir = str(logs_dir / run)
        tags = fetch_tags(logdir)
        
        return {"tags": tags, "run": run}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取标签列表失败: {str(e)}")


@router.get("/all-scalars")
async def get_all_scalars(run: str = ""):
    """批量获取所有标量数据（减少HTTP请求）
    
    Args:
        run: 训练记录名称（默认使用最新）
    """
    try:
        from utils.tensorboard_parser import get_all_scalars as fetch_all, list_training_runs as scan_runs
        
        logs_dir = LOGS_PATH
        
        # 如果未指定run，使用最新的
        if not run:
            runs = scan_runs(logs_dir)
            if not runs:
                return {"scalars": {}, "run": None}
            run = runs[0]["name"]
        
        logdir = str(logs_dir / run)
        scalars = fetch_all(logdir)
        
        return {"scalars": scalars, "run": run}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量获取数据失败: {str(e)}")

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
