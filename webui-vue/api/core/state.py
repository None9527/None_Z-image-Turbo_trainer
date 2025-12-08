import subprocess
from typing import Optional, List, Any, Dict
from fastapi import WebSocket
from dataclasses import dataclass, field
from enum import Enum

class ProcessType(Enum):
    TRAINING = "training"
    DOWNLOAD = "download"
    CACHE_LATENT = "cache_latent"
    CACHE_TEXT = "cache_text"
    GENERATION = "generation"

@dataclass
class ProcessState:
    """进程状态"""
    process: Optional[subprocess.Popen] = None
    status: str = "idle"  # idle, running, completed, failed
    progress: float = 0.0
    current_file: str = ""
    total_files: int = 0
    processed_files: int = 0
    error: Optional[str] = None

# Global state - 统一进程管理
processes: Dict[ProcessType, ProcessState] = {
    ProcessType.TRAINING: ProcessState(),
    ProcessType.DOWNLOAD: ProcessState(),
    ProcessType.CACHE_LATENT: ProcessState(),
    ProcessType.CACHE_TEXT: ProcessState(),
    ProcessType.GENERATION: ProcessState(),
}

# Legacy compatibility
@property
def training_process():
    return processes[ProcessType.TRAINING].process

@training_process.setter
def training_process(value):
    processes[ProcessType.TRAINING].process = value

@property
def download_process():
    return processes[ProcessType.DOWNLOAD].process

@download_process.setter
def download_process(value):
    processes[ProcessType.DOWNLOAD].process = value

# 保持向后兼容
training_process: Optional[subprocess.Popen] = None
download_process: Optional[subprocess.Popen] = None
cache_latent_process: Optional[subprocess.Popen] = None
cache_text_process: Optional[subprocess.Popen] = None

training_websockets: List[WebSocket] = []

# Generation pipeline
pipeline: Any = None

# 当前加载的 LoRA 路径（用于智能缓存）
current_lora_path: Optional[str] = None

# Generation state
generation_status: Dict[str, Any] = {
    "running": False,
    "current_step": 0,
    "total_steps": 0,
    "progress": 0,
    "stage": "idle",  # idle, loading, generating, saving, completed, failed
    "message": "",
    "error": None
}

def update_generation_progress(current_step: int, total_steps: int, stage: str = "generating", message: str = ""):
    """更新生成进度"""
    generation_status["current_step"] = current_step
    generation_status["total_steps"] = total_steps
    generation_status["progress"] = round(current_step / total_steps * 100, 1) if total_steps > 0 else 0
    generation_status["stage"] = stage
    generation_status["message"] = message

def get_generation_status() -> Dict[str, Any]:
    """获取生成状态"""
    return generation_status.copy()

# Training logs (in-memory buffer)
training_logs: List[dict] = []

# Cache progress state (用于实时进度显示)
cache_progress: Dict[str, Any] = {
    "latent": {"current": 0, "total": 0, "progress": 0},
    "text": {"current": 0, "total": 0, "progress": 0}
}

# Download progress state (用于总进度显示)
download_progress: Dict[str, Any] = {
    "progress": 0,
    "downloaded_gb": 0,
    "total_gb": 32,  # 默认预估
    "speed": 0,
    "speed_unit": "MB"
}

def update_download_progress(**kwargs):
    """更新下载进度"""
    download_progress.update(kwargs)

def reset_download_progress():
    """重置下载进度"""
    download_progress["progress"] = 0
    download_progress["downloaded_gb"] = 0
    download_progress["total_gb"] = 32
    download_progress["speed"] = 0
    download_progress["speed_unit"] = "MB"

def get_download_progress() -> Dict[str, Any]:
    """获取下载进度"""
    return download_progress.copy()

def update_cache_progress(cache_type: str, **kwargs):
    """更新缓存进度"""
    if cache_type in cache_progress:
        cache_progress[cache_type].update(kwargs)

def reset_cache_progress(cache_type: str = None):
    """重置缓存进度"""
    if cache_type:
        cache_progress[cache_type] = {"current": 0, "total": 0, "progress": 0}
    else:
        cache_progress["latent"] = {"current": 0, "total": 0, "progress": 0}
        cache_progress["text"] = {"current": 0, "total": 0, "progress": 0}

# Training progress history (用于图表，刷新页面不丢失)
training_history: Dict[str, Any] = {
    "loss_history": [],      # EMA loss 历史
    "lr_history": [],        # 学习率历史
    "current_epoch": 0,
    "total_epochs": 0,
    "current_step": 0,
    "total_steps": 0,
    "learning_rate": 0,
    "loss": 0,
    "elapsed_time": 0,
    "estimated_remaining": 0,
}

def update_training_history(**kwargs):
    """更新训练历史数据"""
    for key, value in kwargs.items():
        if key in training_history:
            training_history[key] = value
    # 限制历史长度
    if len(training_history["loss_history"]) > 10000:
        training_history["loss_history"] = training_history["loss_history"][-5000:]
    if len(training_history["lr_history"]) > 10000:
        training_history["lr_history"] = training_history["lr_history"][-5000:]

def clear_training_history():
    """清空训练历史（新训练开始时调用）"""
    training_history["loss_history"] = []
    training_history["lr_history"] = []
    training_history["current_epoch"] = 0
    training_history["total_epochs"] = 0
    training_history["current_step"] = 0
    training_history["total_steps"] = 0
    training_history["learning_rate"] = 0
    training_history["loss"] = 0
    training_history["elapsed_time"] = 0
    training_history["estimated_remaining"] = 0

def get_training_history() -> Dict[str, Any]:
    """获取训练历史"""
    return training_history.copy()

# Dev mode flag
DEV_MODE: bool = False

def add_log(message: str, level: str = "info"):
    """Add a log entry to the in-memory buffer"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {"time": timestamp, "message": message, "level": level}
    training_logs.append(log_entry)
    # Keep only last 1000 logs
    if len(training_logs) > 1000:
        training_logs.pop(0)
    # Print to console for debugging
    print(f"[{level.upper()}] {timestamp} - {message}")

def get_process_status(process_type: ProcessType) -> Dict[str, Any]:
    """获取进程状态"""
    state = processes[process_type]
    if state.process is None:
        return {"status": "idle"}
    
    return_code = state.process.poll()
    if return_code is None:
        return {
            "status": "running",
            "progress": state.progress,
            "current_file": state.current_file,
            "total_files": state.total_files,
            "processed_files": state.processed_files
        }
    elif return_code == 0:
        return {"status": "completed", "progress": 100}
    else:
        return {"status": "failed", "code": return_code, "error": state.error}

def update_process_progress(process_type: ProcessType, **kwargs):
    """更新进程进度"""
    state = processes[process_type]
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)
