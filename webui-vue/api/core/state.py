import subprocess
import threading
import queue
from typing import Optional, List, Any, Dict, Callable
from fastapi import WebSocket
from dataclasses import dataclass, field
from enum import Enum


class ProcessOutputReader:
    """
    进程输出读取器 - 使用独立线程持续读取进程 stdout
    
    解决问题：WebSocket 断开后管道缓冲区满导致训练进程 print() 阻塞
    
    设计：
    - 独立线程持续读取 stdout，永不停止（直到进程结束）
    - 使用有界队列存储日志，防止内存无限增长
    - WebSocket 可读取队列中的日志进行广播
    """
    
    def __init__(self, process: subprocess.Popen, name: str = "process", 
                 max_queue_size: int = 1000, parse_func: Callable = None):
        self.process = process
        self.name = name
        self.parse_func = parse_func
        self.output_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True, name=f"reader-{name}")
        self.thread.start()
        print(f"[ProcessOutputReader] Started reader thread for {name}")
    
    def _read_loop(self):
        """独立线程：持续读取 stdout，永不阻塞主线程"""
        try:
            if not self.process or not self.process.stdout:
                return
            
            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break
                if not line:
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                # 解析进度信息
                progress = None
                if self.parse_func:
                    try:
                        progress = self.parse_func(line)
                    except:
                        pass
                
                # 放入队列
                item = {"line": line, "progress": progress}
                try:
                    self.output_queue.put_nowait(item)
                except queue.Full:
                    # 队列满时丢弃最旧的日志
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(item)
                    except:
                        pass
            
            print(f"[ProcessOutputReader] Reader thread for {self.name} finished (process ended)")
        except Exception as e:
            print(f"[ProcessOutputReader] Error in reader thread for {self.name}: {e}")
        finally:
            self.running = False
    
    def get_lines(self, max_lines: int = 50) -> List[Dict[str, Any]]:
        """获取队列中的日志行（非阻塞）"""
        lines = []
        while len(lines) < max_lines:
            try:
                lines.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return lines
    
    def stop(self):
        """停止读取线程"""
        self.running = False
        print(f"[ProcessOutputReader] Stopping reader thread for {self.name}")
    
    def is_alive(self) -> bool:
        """检查读取线程是否存活"""
        return self.thread.is_alive()


# 全局进程读取器管理
_process_readers: Dict[str, ProcessOutputReader] = {}

def start_process_reader(process: subprocess.Popen, name: str, parse_func: Callable = None) -> ProcessOutputReader:
    """启动进程输出读取器"""
    global _process_readers
    
    # 如果已存在同名读取器，先停止
    if name in _process_readers:
        _process_readers[name].stop()
    
    reader = ProcessOutputReader(process, name, parse_func=parse_func)
    _process_readers[name] = reader
    return reader

def get_process_reader(name: str) -> Optional[ProcessOutputReader]:
    """获取进程读取器"""
    return _process_readers.get(name)

def stop_process_reader(name: str):
    """停止进程读取器"""
    if name in _process_readers:
        _process_readers[name].stop()
        del _process_readers[name]

def stop_all_readers():
    """停止所有读取器"""
    for name in list(_process_readers.keys()):
        stop_process_reader(name)



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

# Generation pipeline (支持多模型)
pipelines: Dict[str, Any] = {}  # {model_type: pipeline}
pipeline: Any = None  # 保持向后兼容，指向当前活跃的 pipeline

# 当前加载的模型类型
current_model_type: Optional[str] = None

# 当前加载的 LoRA 路径（用于智能缓存，每个模型独立）
lora_cache: Dict[str, Optional[str]] = {}  # {model_type: lora_path}
current_lora_path: Optional[str] = None  # 向后兼容

def get_pipeline(model_type: str) -> Any:
    """获取指定模型的 pipeline"""
    return pipelines.get(model_type)

def set_pipeline(model_type: str, pipe: Any):
    """设置指定模型的 pipeline"""
    global pipeline, current_model_type
    pipelines[model_type] = pipe
    pipeline = pipe  # 向后兼容
    current_model_type = model_type

def get_lora_path(model_type: str) -> Optional[str]:
    """获取指定模型当前加载的 LoRA 路径"""
    return lora_cache.get(model_type)

def set_lora_path(model_type: str, path: Optional[str]):
    """设置指定模型的 LoRA 路径"""
    global current_lora_path
    lora_cache[model_type] = path
    current_lora_path = path  # 向后兼容

def clear_pipelines():
    """清理所有加载的 pipeline 以释放显存"""
    global pipelines, pipeline, current_model_type
    pipelines.clear()
    pipeline = None
    current_model_type = None
    print("[State] All pipelines cleared")

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
    "speed_unit": "MB",
    "model_type": "",  # 正在下载的模型类型
    "model_name": ""   # 正在下载的模型名称
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
    download_progress["model_type"] = ""
    download_progress["model_name"] = ""

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
    "start_timestamp": 0,    # 训练开始时间戳 (Unix timestamp, 秒)
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
    import time
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
    training_history["start_timestamp"] = time.time()  # 记录训练开始时间

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
