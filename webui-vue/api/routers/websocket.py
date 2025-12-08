"""
WebSocket 实时推送模块
统一管理所有监控通讯：GPU状态、系统信息、模型状态、下载、训练进度、缓存进度
"""

import asyncio
import subprocess
import json
import platform
from typing import Set, Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime
from pathlib import Path

from core.config import MODEL_PATH, PROJECT_ROOT
from core import state
from core.state import get_generation_status

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """WebSocket 连接管理器"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"[WebSocket] Client connected. Total clients: {len(self.active_connections)}")
        
        # 启动广播任务（如果还没启动）
        if not self._running:
            self._running = True
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        print(f"[WebSocket] Client disconnected. Total clients: {len(self.active_connections)}")
        
        # 如果没有连接了，停止广播
        if not self.active_connections and self._running:
            self._running = False
            if self._broadcast_task:
                self._broadcast_task.cancel()
    
    async def broadcast(self, message: Dict[str, Any]):
        """向所有连接广播消息"""
        if not self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        
        # 移除断开的连接
        self.active_connections -= disconnected
    
    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]):
        """向单个连接发送消息"""
        try:
            await websocket.send_json(message)
        except Exception:
            self.active_connections.discard(websocket)
    
    async def _broadcast_loop(self):
        """后台广播循环 - 定时推送状态更新"""
        while self._running:
            try:
                # 收集所有状态
                gpu_info = await get_gpu_info()
                download_status = get_download_status()
                training_status = get_training_status()
                cache_status = get_cache_status()
                generation_status = get_generation_status()
                
                # 广播状态更新
                await self.broadcast({
                    "type": "status_update",
                    "timestamp": datetime.now().isoformat(),
                    "gpu": gpu_info,
                    "download": download_status,
                    "training": training_status,
                    "cache": cache_status,
                    "generation": generation_status
                })
                
                # 读取训练进程输出
                if state.training_process and state.training_process.poll() is None:
                    await self._read_process_output(
                        state.training_process, "training_log", parse_training_progress
                    )
                
                # 读取下载进程输出
                if state.download_process and state.download_process.poll() is None:
                    await self._read_process_output(
                        state.download_process, "download_log", parse_download_progress
                    )
                
                # 读取缓存进程输出
                if state.cache_latent_process and state.cache_latent_process.poll() is None:
                    await self._read_process_output(
                        state.cache_latent_process, "cache_latent_log", parse_cache_progress
                    )
                
                if state.cache_text_process and state.cache_text_process.poll() is None:
                    await self._read_process_output(
                        state.cache_text_process, "cache_text_log", parse_cache_progress
                    )
                
                await asyncio.sleep(0.2)  # 每 200ms 更新一次（加快日志响应）
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Broadcast loop error: {e}")
                await asyncio.sleep(1)
    
    async def _read_process_output(self, process, log_type: str, parse_func):
        """读取进程输出并广播（批量读取多行）"""
        if not process or not process.stdout:
            return
        
        try:
            import sys
            
            # 批量读取多行（最多10行）
            lines_read = 0
            max_lines = 10
            
            if sys.platform != 'win32':
                # Unix: 使用 select 进行非阻塞检查
                import select
                while lines_read < max_lines:
                    readable, _, _ = select.select([process.stdout], [], [], 0)
                    if not readable:
                        break
                    line = process.stdout.readline()
                    if not line:
                        break
                    await self._process_line(line.strip(), log_type, parse_func)
                    lines_read += 1
            else:
                # Windows: 使用 PeekNamedPipe 检查数据可用性
                import ctypes
                from ctypes import wintypes
                import msvcrt
                
                kernel32 = ctypes.windll.kernel32
                
                try:
                    fd = process.stdout.fileno()
                    handle = msvcrt.get_osfhandle(fd)
                except:
                    return
                
                while lines_read < max_lines:
                    try:
                        # 检查管道中是否有数据
                        bytes_avail = wintypes.DWORD()
                        result = kernel32.PeekNamedPipe(
                            handle, None, 0, None,
                            ctypes.byref(bytes_avail), None
                        )
                        
                        if not result or bytes_avail.value == 0:
                            break  # 没有数据
                        
                        # 有数据，读取一行
                        line = process.stdout.readline()
                        if not line:
                            break
                        await self._process_line(line.strip(), log_type, parse_func)
                        lines_read += 1
                    except Exception:
                        break
                        
        except Exception as e:
            error_str = str(e)
            if "10038" not in error_str and "Bad file descriptor" not in error_str:
                pass  # 静默忽略
    
    async def _process_line(self, line: str, log_type: str, parse_func):
        """处理输出行"""
        if not line:
            return
        
        # 调试：打印收到的日志行
        if "cache" in log_type:
            print(f"[{log_type}] Line: {line}", flush=True)
        
        # 添加到日志
        state.add_log(line, "info")
        
        # 解析进度信息
        progress = parse_func(line) if parse_func else None
        
        # 更新训练历史（用于图表持久化）
        if log_type == "training_log" and progress:
            self._update_training_history(progress)
        
        # 更新缓存进度（用于进度条显示）
        if log_type == "cache_latent_log" and progress:
            self._update_cache_progress("latent", progress)
        elif log_type == "cache_text_log" and progress:
            self._update_cache_progress("text", progress)
        
        # 更新下载进度（用于总进度显示）
        if log_type == "download_log" and progress:
            self._update_download_progress(progress)
        
        await self.broadcast({
            "type": log_type,
            "timestamp": datetime.now().isoformat(),
            "message": line,
            "progress": progress
        })
    
    def _update_cache_progress(self, cache_type: str, progress: Dict[str, Any]):
        """更新缓存进度"""
        if progress.get("type") == "progress":
            state.update_cache_progress(cache_type, 
                current=progress.get("current", 0),
                total=progress.get("total", 0),
                progress=progress.get("percent", 0)
            )
            print(f"[Cache] Updated {cache_type}: {state.cache_progress[cache_type]}", flush=True)
        elif progress.get("type") == "percent":
            state.update_cache_progress(cache_type, progress=progress.get("value", 0))
    
    def _update_download_progress(self, progress: Dict[str, Any]):
        """更新下载进度"""
        if progress.get("type") == "total_progress":
            state.update_download_progress(
                progress=progress.get("percent", 0),
                downloaded_gb=progress.get("downloaded_gb", 0),
                total_gb=progress.get("total_gb", 32),
                speed=progress.get("speed", 0),
                speed_unit=progress.get("speed_unit", "MB")
            )
        elif progress.get("type") == "percent":
            state.update_download_progress(progress=progress.get("value", 0))
        elif progress.get("type") == "complete":
            state.update_download_progress(progress=100)
    
    def _update_training_history(self, progress: Dict[str, Any]):
        """更新训练历史数据（用于图表持久化）"""
        updates = {}
        
        if progress.get("epoch"):
            updates["current_epoch"] = progress["epoch"]["current"]
            updates["total_epochs"] = progress["epoch"]["total"]
        
        if progress.get("step"):
            updates["current_step"] = progress["step"]["current"]
            updates["total_steps"] = progress["step"]["total"]
        
        if progress.get("ema_loss") is not None:
            updates["loss"] = progress["ema_loss"]
            # 添加到历史（EMA loss 用于图表）
            state.training_history["loss_history"].append(progress["ema_loss"])
        elif progress.get("loss") is not None:
            updates["loss"] = progress["loss"]
            state.training_history["loss_history"].append(progress["loss"])
        
        if progress.get("learningRate") is not None:
            updates["learning_rate"] = progress["learningRate"]
            state.training_history["lr_history"].append(progress["learningRate"])
        
        if progress.get("time"):
            # 解析时间字符串
            def parse_time(s):
                parts = s.split(':')
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                return 0
            updates["elapsed_time"] = parse_time(progress["time"]["elapsed"])
            updates["estimated_remaining"] = parse_time(progress["time"]["remaining"])
        
        if updates:
            state.update_training_history(**updates)


# 全局连接管理器
manager = ConnectionManager()


async def broadcast_generation_progress(current_step: int, total_steps: int, stage: str = "generating", message: str = ""):
    """广播生成进度更新 - 供 generation.py 调用"""
    # 更新状态
    state.update_generation_progress(current_step, total_steps, stage, message)
    
    # 立即广播进度
    await manager.broadcast({
        "type": "generation_progress",
        "timestamp": datetime.now().isoformat(),
        "current_step": current_step,
        "total_steps": total_steps,
        "progress": round(current_step / total_steps * 100, 1) if total_steps > 0 else 0,
        "stage": stage,
        "message": message
    })


# 保存主事件循环引用
_main_loop = None

def set_main_loop(loop):
    """设置主事件循环引用"""
    global _main_loop
    _main_loop = loop

def sync_broadcast_generation_progress(current_step: int, total_steps: int, stage: str = "generating", message: str = ""):
    """同步版本的广播函数 - 供线程中回调使用"""
    import asyncio
    
    # 更新状态（直接更新全局字典）
    state.generation_status["running"] = True if stage not in ("completed", "failed", "idle") else False
    state.generation_status["current_step"] = current_step
    state.generation_status["total_steps"] = total_steps
    state.generation_status["progress"] = round(current_step / total_steps * 100, 1) if total_steps > 0 else 0
    state.generation_status["stage"] = stage
    state.generation_status["message"] = message
    
    progress_pct = state.generation_status["progress"]
    print(f"[Generation] {stage}: {message} - {progress_pct}% ({len(manager.active_connections)} ws clients) - state.running={state.generation_status['running']}")
    
    message_data = {
        "type": "generation_progress",
        "timestamp": datetime.now().isoformat(),
        "current_step": current_step,
        "total_steps": total_steps,
        "progress": progress_pct,
        "stage": stage,
        "message": message
    }
    
    # 从线程中安全地调度到主事件循环
    try:
        if _main_loop and _main_loop.is_running():
            # 使用 run_coroutine_threadsafe 从线程安全地调度
            future = asyncio.run_coroutine_threadsafe(
                manager.broadcast(message_data),
                _main_loop
            )
            # 等待一小段时间确保任务被调度
            try:
                future.result(timeout=0.1)
            except:
                pass  # 超时没关系，任务已经在队列中
        else:
            print(f"[Generation] Main loop not available")
    except Exception as e:
        print(f"Broadcast generation progress error: {e}")


def parse_training_progress(line: str) -> Optional[Dict[str, Any]]:
    """解析训练进度信息
    
    支持格式：
    1. [TRAINING_INFO] total_steps=3000 total_epochs=100  (启动时的总步数信息)
    2. [STEP] 50/3000 epoch=1/100 loss=0.1234 ema_loss=0.1200 lr=1.00e-04  (每步进度)
    3. tqdm 进度条格式 (备用)
    """
    import re
    
    result = {}
    
    # 优先匹配 [TRAINING_INFO] 格式（启动时的总步数信息）
    if '[TRAINING_INFO]' in line:
        total_steps_match = re.search(r'total_steps=(\d+)', line)
        total_epochs_match = re.search(r'total_epochs=(\d+)', line)
        if total_steps_match:
            result["step"] = {
                "current": 0,
                "total": int(total_steps_match.group(1))
            }
        if total_epochs_match:
            result["epoch"] = {
                "current": 0,
                "total": int(total_epochs_match.group(1))
            }
        return result if result else None
    
    # 优先匹配 [STEP] 格式（每步进度，最可靠）
    if '[STEP]' in line:
        step_match = re.search(r'\[STEP\]\s*(\d+)/(\d+)', line)
        if step_match:
            result["step"] = {
                "current": int(step_match.group(1)),
                "total": int(step_match.group(2))
            }
        epoch_match = re.search(r'epoch=(\d+)/(\d+)', line)
        if epoch_match:
            result["epoch"] = {
                "current": int(epoch_match.group(1)),
                "total": int(epoch_match.group(2))
            }
        loss_match = re.search(r'loss=([0-9.]+)', line)
        if loss_match:
            result["loss"] = float(loss_match.group(1))
        ema_match = re.search(r'ema_loss=([0-9.]+)', line)
        if ema_match:
            result["ema_loss"] = float(ema_match.group(1))
        lr_match = re.search(r'lr=([0-9.e+-]+)', line)
        if lr_match:
            try:
                result["learningRate"] = float(lr_match.group(1))
            except:
                pass
        return result if result else None
    
    # 先检查是否是模型加载日志（跳过）
    if any(keyword in line.lower() for keyword in ['loading', 'load', '加载', 'initializing', 'preparing']):
        # 只有同时包含 loss/lr 才继续解析，否则跳过
        if 'loss' not in line.lower() and 'lr=' not in line.lower():
            return None
    
    # 匹配 epoch 进度: "Epoch 1/10" 或 "epoch: 1/10"
    epoch_match = re.search(r'[Ee]poch[:\s]+(\d+)[/\s]+(\d+)', line)
    if epoch_match:
        result["epoch"] = {
            "current": int(epoch_match.group(1)),
            "total": int(epoch_match.group(2))
        }
    
    # 匹配 tqdm 进度条: "50%|█████| 50/100 [00:30<01:00" 或 "[1:30:00<2:00:00"
    # 只有这种格式才是真正的训练进度
    tqdm_match = re.search(r'(\d+)%\|[^|]+\|\s*(\d+)/(\d+)\s*\[(\d+:\d+(?::\d+)?)<(\d+:\d+(?::\d+)?)', line)
    if tqdm_match:
        result["step"] = {
            "current": int(tqdm_match.group(2)),
            "total": int(tqdm_match.group(3))
        }
        result["time"] = {
            "elapsed": tqdm_match.group(4),
            "remaining": tqdm_match.group(5)
        }
        result["percent"] = int(tqdm_match.group(1))
    # 注意：移除了简单的 "X/Y" 匹配，避免误解析模型加载日志
    
    # 匹配 loss: "loss=0.1234" 或 "loss: 0.1234"
    loss_match = re.search(r'loss[=:\s]+([0-9.]+)', line, re.IGNORECASE)
    if loss_match:
        result["loss"] = float(loss_match.group(1))
    
    # 匹配学习率: "lr=1.00e-04" 或 "lr: 1e-4"
    lr_match = re.search(r'lr[=:\s]+([0-9.e+-]+)', line, re.IGNORECASE)
    if lr_match:
        try:
            result["learningRate"] = float(lr_match.group(1))
        except:
            pass
    
    # 匹配 EMA loss: "ema=0.1234"
    ema_match = re.search(r'ema[=:\s]+([0-9.]+)', line, re.IGNORECASE)
    if ema_match:
        result["ema_loss"] = float(ema_match.group(1))
    
    return result if result else None


def parse_download_progress(line: str) -> Optional[Dict[str, Any]]:
    """解析下载进度信息"""
    import re
    
    # 优先匹配新格式: [PROGRESS] [...] 50.0% | 16.00 GB/32.00 GB | 10.5 MB/s
    progress_match = re.search(
        r'\[PROGRESS\].*?(\d+(?:\.\d+)?)%\s*\|\s*([0-9.]+)\s*(MB|GB)/([0-9.]+)\s*(MB|GB)\s*\|\s*([0-9.]+)\s*(MB|KB)/s',
        line, re.IGNORECASE
    )
    if progress_match:
        downloaded = float(progress_match.group(2))
        downloaded_unit = progress_match.group(3).upper()
        total = float(progress_match.group(4))
        total_unit = progress_match.group(5).upper()
        speed = float(progress_match.group(6))
        speed_unit = progress_match.group(7).upper()
        
        # 统一转换为 GB
        if downloaded_unit == "MB":
            downloaded /= 1024
        if total_unit == "MB":
            total /= 1024
            
        return {
            "type": "total_progress",
            "percent": float(progress_match.group(1)),
            "downloaded_gb": downloaded,
            "total_gb": total,
            "speed": speed,
            "speed_unit": speed_unit
        }
    
    # 匹配简化的新格式: [PROGRESS] [...] 50.0%
    simple_progress = re.search(r'\[PROGRESS\].*?(\d+(?:\.\d+)?)%', line)
    if simple_progress:
        return {
            "type": "percent",
            "value": float(simple_progress.group(1))
        }
    
    # 匹配下载完成: [PROGRESS] Download complete!
    if "[PROGRESS] Download complete" in line:
        return {
            "type": "complete",
            "percent": 100
        }
    
    # 匹配百分比: "50%" 或 "Downloading: 50%"
    percent_match = re.search(r'(\d+(?:\.\d+)?)%', line)
    if percent_match:
        return {
            "type": "percent",
            "value": float(percent_match.group(1))
        }
    
    # 匹配文件进度: "Processing 10 items: 50%"
    items_match = re.search(r'Processing\s+(\d+)\s+items?[:\s]+(\d+(?:\.\d+)?)%', line)
    if items_match:
        return {
            "type": "items",
            "total": int(items_match.group(1)),
            "percent": float(items_match.group(2))
        }
    
    # 匹配下载速度: "10.5 MB/s"
    speed_match = re.search(r'([0-9.]+)\s*(KB|MB|GB)/s', line, re.IGNORECASE)
    if speed_match:
        return {
            "type": "speed",
            "value": float(speed_match.group(1)),
            "unit": speed_match.group(2).upper()
        }
    
    # 匹配文件名: "Downloading [filename.safetensors]"
    file_match = re.search(r'Downloading\s+\[([^\]]+)\]', line)
    if file_match:
        return {
            "type": "file",
            "name": file_match.group(1)
        }
    
    return None


def parse_cache_progress(line: str) -> Optional[Dict[str, Any]]:
    """解析缓存进度信息"""
    import re
    
    # 匹配处理进度: "Progress: 10/100" 或 "10/100"
    progress_match = re.search(r'Progress:\s*(\d+)\s*/\s*(\d+)', line)
    if not progress_match:
        # 兼容旧格式
        progress_match = re.search(r'(\d+)\s*/\s*(\d+)', line)
    
    if progress_match:
        current = int(progress_match.group(1))
        total = int(progress_match.group(2))
        result = {
            "type": "progress",
            "current": current,
            "total": total,
            "percent": round(current / total * 100, 1) if total > 0 else 0
        }
        print(f"[Cache Progress] Parsed: {result}", flush=True)  # 调试日志
        return result
    
    # 匹配百分比: "50%"
    percent_match = re.search(r'(\d+(?:\.\d+)?)%', line)
    if percent_match:
        return {
            "type": "percent",
            "value": float(percent_match.group(1))
        }
    
    # 匹配文件处理: "Processing: filename.png"
    file_match = re.search(r'[Pp]rocessing[:\s]+(.+\.(?:png|jpg|jpeg|webp))', line)
    if file_match:
        return {
            "type": "file",
            "name": file_match.group(1)
        }
    
    # 匹配完成: "Completed" 或 "Done"
    if re.search(r'(?:completed|done|finished)', line, re.IGNORECASE):
        return {
            "type": "completed"
        }
    
    # 匹配跳过: "Skipping"
    if re.search(r'skip', line, re.IGNORECASE):
        return {
            "type": "skipped"
        }
    
    return None


async def get_gpu_info() -> Dict[str, Any]:
    """获取 GPU 信息"""
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5
            )
        )
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 5:
                memory_total = float(parts[1]) / 1024  # MB -> GB
                memory_used = float(parts[2]) / 1024
                return {
                    "name": parts[0],
                    "memoryTotal": round(memory_total, 1),
                    "memoryUsed": round(memory_used, 1),
                    "memoryPercent": round((memory_used / memory_total) * 100),
                    "utilization": int(parts[3]),
                    "temperature": int(parts[4])
                }
    except Exception as e:
        print(f"GPU info error: {e}")
    
    return {
        "name": "Unknown",
        "memoryTotal": 0,
        "memoryUsed": 0,
        "memoryPercent": 0,
        "utilization": 0,
        "temperature": 0
    }


# 缓存系统信息（不经常变化）
_cached_system_info: Optional[Dict[str, Any]] = None

async def get_system_info() -> Dict[str, Any]:
    """获取系统信息（带缓存）"""
    global _cached_system_info
    
    if _cached_system_info is not None:
        return _cached_system_info
    
    try:
        import torch
        import diffusers
        
        # xformers
        try:
            import xformers
            xformers_ver = xformers.__version__
        except ImportError:
            xformers_ver = "N/A"
        
        # accelerate
        try:
            import accelerate
            accelerate_ver = accelerate.__version__
        except ImportError:
            accelerate_ver = "N/A"
        
        # transformers
        try:
            import transformers
            transformers_ver = transformers.__version__
        except ImportError:
            transformers_ver = "N/A"
        
        # bitsandbytes
        try:
            import bitsandbytes
            bnb_ver = bitsandbytes.__version__
        except ImportError:
            bnb_ver = "N/A"
        
        cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"
        cudnn_ver = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
        
        _cached_system_info = {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "diffusers": diffusers.__version__,
            "xformers": xformers_ver,
            "accelerate": accelerate_ver,
            "transformers": transformers_ver,
            "bitsandbytes": bnb_ver,
            "cuda": cuda_ver,
            "cudnn": cudnn_ver,
            "platform": platform.platform()
        }
        return _cached_system_info
    except Exception as e:
        print(f"System info error: {e}")
        return {
            "python": "",
            "pytorch": "",
            "diffusers": "",
            "xformers": "",
            "accelerate": "",
            "transformers": "",
            "bitsandbytes": "",
            "cuda": "",
            "cudnn": "",
            "platform": ""
        }


def get_model_status_simple() -> Dict[str, Any]:
    """获取简化的模型状态"""
    try:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            return {"downloaded": False, "path": str(model_path), "size_gb": 0}
        
        # 计算目录大小
        total_size = 0
        for f in model_path.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
        
        size_gb = round(total_size / (1024**3), 2)
        
        # 检查关键文件是否存在
        key_files = ["model_index.json", "transformer", "vae", "text_encoder"]
        exists_count = sum(1 for f in key_files if (model_path / f).exists())
        
        return {
            "downloaded": exists_count >= 3,  # 至少3个关键组件存在
            "path": str(model_path),
            "size_gb": size_gb
        }
    except Exception as e:
        print(f"Model status error: {e}")
        return {"downloaded": False, "path": MODEL_PATH, "size_gb": 0}


def get_download_status() -> Dict[str, Any]:
    """获取下载状态"""
    if state.download_process is None:
        return {"status": "idle"}
    
    return_code = state.download_process.poll()
    
    if return_code is None:
        # 从 state 获取下载进度
        dp = state.get_download_progress()
        
        return {
            "status": "running",
            "progress": dp.get("progress", 0),
            "downloaded_size_gb": dp.get("downloaded_gb", 0),
            "total_size_gb": dp.get("total_gb", 32),
            "speed": dp.get("speed", 0),
            "speed_unit": dp.get("speed_unit", "MB")
        }
    elif return_code == 0:
        state.download_process = None
        state.reset_download_progress()
        return {"status": "completed", "progress": 100}
    else:
        state.download_process = None
        state.reset_download_progress()
        return {"status": "failed", "code": return_code}


def get_training_status() -> Dict[str, Any]:
    """获取训练状态
    
    状态：
    - idle: 未运行
    - loading: 模型加载中（进程运行但还没有 loss 数据）
    - running: 训练中（有 loss 数据）
    - completed: 训练完成
    - failed: 训练失败
    """
    if state.training_process is None:
        return {"status": "idle", "running": False}
    
    return_code = state.training_process.poll()
    
    if return_code is None:
        # 检查是否已经开始产生 loss 数据
        history = state.get_training_history()
        has_loss_data = len(history.get("loss_history", [])) > 0
        
        return {
            "status": "running" if has_loss_data else "loading",
            "running": True,
            "loading": not has_loss_data,  # 额外标记是否在加载中
            "pid": state.training_process.pid
        }
    elif return_code == 0:
        state.add_log("训练完成", "success")
        state.training_process = None  # 清除引用，防止重复添加日志
        return {"status": "completed", "running": False}
    else:
        # 获取详细错误信息
        error_msg = f"训练进程退出，代码: {return_code}"
        
        # 常见退出码解释
        error_hints = {
            1: "一般错误（可能是配置问题或代码异常）",
            2: "命令行参数错误",
            137: "进程被 SIGKILL 终止（可能是内存不足 OOM）",
            139: "段错误（SIGSEGV）",
            -9: "进程被强制终止（OOM Killer）",
            -15: "进程收到 SIGTERM 终止信号"
        }
        
        hint = error_hints.get(return_code, "未知错误")
        
        # 检查最近的日志是否有 OOM 信息
        recent_logs = state.training_logs[-20:] if state.training_logs else []
        for log in recent_logs:
            msg = log.get("message", "")
            if "CUDA out of memory" in msg or "OutOfMemoryError" in msg:
                hint = "CUDA 显存不足 - 请减小 batch_size 或增大 gradient_accumulation_steps"
                break
            elif "No valid cache files" in msg:
                hint = "未找到有效的缓存文件 - 请检查数据集路径"
                break
            elif "ModuleNotFoundError" in msg:
                hint = "缺少依赖模块 - 请检查安装"
                break
        
        state.add_log(f"{error_msg}: {hint}", "error")
        state.training_process = None  # 清除引用，防止重复添加日志
        
        return {
            "status": "failed", 
            "running": False, 
            "code": return_code,
            "message": error_msg,
            "hint": hint
        }


def get_cache_status() -> Dict[str, Any]:
    """获取缓存状态（包含进度信息）"""
    
    def _get_status(process, cache_type):
        if process is None:
            return {"status": "idle", "progress": 0, "current": 0, "total": 0}
        
        return_code = process.poll()
        progress_info = state.cache_progress.get(cache_type, {})
        
        if return_code is None:
            return {
                "status": "running", 
                "pid": process.pid,
                "progress": progress_info.get("progress", 0),
                "current": progress_info.get("current", 0),
                "total": progress_info.get("total", 0)
            }
        elif return_code == 0:
            return {"status": "completed", "progress": 100, "current": 0, "total": 0}
        else:
            return {"status": "failed", "code": return_code, "progress": 0, "current": 0, "total": 0}
    
    return {
        "latent": _get_status(state.cache_latent_process, "latent"),
        "text": _get_status(state.cache_text_process, "text")
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """统一的 WebSocket 端点 - 所有监控通讯"""
    await manager.connect(websocket)
    
    try:
        # 发送初始状态（包含完整信息）
        gpu_info = await get_gpu_info()
        system_info = await get_system_info()
        model_status = get_model_status_simple()
        
        await manager.send_personal(websocket, {
            "type": "init",
            "timestamp": datetime.now().isoformat(),
            "gpu": gpu_info,
            "system_info": system_info,
            "model_status": model_status,
            "download": get_download_status(),
            "training": get_training_status(),
            "cache": get_cache_status(),
            "generation": get_generation_status(),
            "logs": state.training_logs[-50:],  # 最近50条日志
            "training_history": state.get_training_history()  # 训练历史（图表数据）
        })
        
        # 保持连接，处理客户端消息
        while True:
            try:
                data = await websocket.receive_text()
                
                # 处理客户端命令
                if data == "ping":
                    await websocket.send_text("pong")
                elif data.startswith("{"):
                    # JSON 命令
                    cmd = json.loads(data)
                    await handle_ws_command(websocket, cmd)
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket receive error: {e}")
                break
                
    finally:
        manager.disconnect(websocket)


async def handle_ws_command(websocket: WebSocket, cmd: Dict[str, Any]):
    """处理 WebSocket 命令"""
    action = cmd.get("action")
    
    if action == "get_gpu":
        gpu_info = await get_gpu_info()
        await manager.send_personal(websocket, {
            "type": "gpu",
            "data": gpu_info
        })
    
    elif action == "get_status":
        # 获取所有状态
        gpu_info = await get_gpu_info()
        system_info = await get_system_info()
        model_status = get_model_status_simple()
        await manager.send_personal(websocket, {
            "type": "full_status",
            "gpu": gpu_info,
            "system_info": system_info,
            "model_status": model_status,
            "download": get_download_status(),
            "training": get_training_status(),
            "cache": get_cache_status(),
            "generation": get_generation_status()
        })
    
    elif action == "get_logs":
        count = cmd.get("count", 50)
        await manager.send_personal(websocket, {
            "type": "logs",
            "data": state.training_logs[-count:]
        })
    
    elif action == "clear_logs":
        state.training_logs.clear()
        await manager.send_personal(websocket, {
            "type": "logs_cleared"
        })
    
    elif action == "subscribe":
        # 订阅特定类型的更新（未来扩展用）
        topics = cmd.get("topics", [])
        await manager.send_personal(websocket, {
            "type": "subscribed",
            "topics": topics
        })
