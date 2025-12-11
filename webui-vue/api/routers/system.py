from fastapi import APIRouter, HTTPException
import subprocess
import sys
import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

from core.config import PROJECT_ROOT, MODEL_PATH, LONGCAT_MODEL_PATH, MODEL_PATHS, get_model_path
from core import state

# 导入新的模型检测和下载抽象层
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.model_detector import (
    create_model_detector,
    ModelStatus, ZImageDetector, LongCatDetector
)
from src.models.model_downloader import (
    create_model_downloader, get_model_download_info, DownloadStatus,
    ModelDownloadManager
)
from pydantic import BaseModel

router = APIRouter(prefix="/api/system", tags=["system"])

class ModelOpsRequest(BaseModel):
    model_type: str = "zimage"

# 多模型支持 - 模型规格定义
MODEL_SPECS = {
    "zimage": {
        "name": "Z-Image-Turbo",
        "model_id": "Tongyi-MAI/Z-Image-Turbo",
        "path_env": "MODEL_PATH",  # 使用 MODEL_PATH 环境变量
        "expected_files": {
            "model_index.json": {"required": True, "min_size": 100},
            "transformer": {
                "required": True,
                "files": {
                    "config.json": {"min_size": 100},
                    "diffusion_pytorch_model.safetensors.index.json": {"min_size": 1000},
                }
            },
            "vae": {
                "required": True,
                "files": {
                    "config.json": {"min_size": 100},
                    "diffusion_pytorch_model.safetensors": {"min_size": 100 * 1024 * 1024}
                }
            },
            "text_encoder": {
                "required": True,
                "files": {
                    "config.json": {"min_size": 100},
                    "model.safetensors.index.json": {"min_size": 1000},
                }
            },
            "tokenizer": {
                "required": True,
                "files": {
                    "tokenizer.json": {"min_size": 1000},
                    "tokenizer_config.json": {"min_size": 100},
                }
            },
            "scheduler": {
                "required": True,
                "files": {
                    "scheduler_config.json": {"min_size": 100},
                }
            }
        }
    },
    "longcat": {
        "name": "LongCat-Image-Dev",
        "model_id": "meituan-longcat/LongCat-Image-Dev",
        "path_env": "LONGCAT_MODEL_PATH",
        "default_path": "longcat_models",  # 默认下载目录
        "expected_files": {
            "model_index.json": {"required": True, "min_size": 100},
            "transformer": {
                "required": True,
                "files": {
                    # config.json 或 configuration.json 均可，降低严格度，只检查核心权重索引
                    "diffusion_pytorch_model.safetensors.index.json": {"min_size": 500},
                }
            },
            "vae": {
                "required": True,
                "files": {
                    "config.json": {"min_size": 100},
                    # VAE 约 335MB
                    "diffusion_pytorch_model.safetensors": {"min_size": 100 * 1024 * 1024},
                }
            },
            "text_encoder": {
                "required": True,
                "files": {
                    "config.json": {"min_size": 100},
                    # Text encoder 约 9.4GB，检查 index 文件
                    "model.safetensors.index.json": {"min_size": 500},
                }
            },
            "tokenizer": {
                "required": True,
                "files": {
                    "tokenizer.json": {"min_size": 100000},  # 约 2.7MB
                    "tokenizer_config.json": {"min_size": 100},
                }
            },
            "scheduler": {
                "required": True,
                "files": {
                    "scheduler_config.json": {"min_size": 100},
                }
            }
        }
    }
}

# 向后兼容 - 默认 Z-Image 规格
MODELSCOPE_MODEL_ID = MODEL_SPECS["zimage"]["model_id"]
EXPECTED_FILES = MODEL_SPECS["zimage"]["expected_files"]

@router.get("/status")
async def get_system_status():
    """Get overall system status"""
    if state.training_process and state.training_process.poll() is None:
        status = "training"
    else:
        status = "online"
    return {"status": status}

@router.get("/info")
async def get_system_info():
    """Get detailed system information"""
    import torch
    import diffusers
    import platform
    
    # Get Windows version properly
    if platform.system() == "Windows":
        win_ver = platform.win32_ver()
        # win32_ver returns (release, version, csd, ptype)
        # e.g., ('10', '10.0.22631', 'SP0', 'Multiprocessor Free')
        build = win_ver[1].split('.')[-1] if win_ver[1] else ''
        # Build >= 22000 is Windows 11
        if build and int(build) >= 22000:
            os_name = f"Windows 11 (Build {build})"
        else:
            os_name = f"Windows {win_ver[0]} (Build {build})"
    else:
        os_name = platform.platform()
    
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
    
    # CUDA / cuDNN
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"
    cudnn_ver = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "diffusers": diffusers.__version__,
        "xformers": xformers_ver,
        "accelerate": accelerate_ver,
        "transformers": transformers_ver,
        "bitsandbytes": bnb_ver,
        "cuda": cuda_ver,
        "cudnn": cudnn_ver,
        "platform": os_name
    }

# 移除硬编码的 MODEL_SPECS，改用 ModelDetector 作为唯一事实来源

@router.get("/model-status")
async def get_model_status(model_type: Optional[str] = None):
    """获取模型状态（统一使用 ModelDetector.validate_local）"""
    if model_type is None:
        model_type = "zimage"
    if model_type not in ["zimage", "longcat"]:
        model_type = "zimage"
        
    try:
        model_path = get_model_path(model_type, "base")
        detector = create_model_detector(model_type, model_path)
        
        # 本地 Schema 校验 (快)
        result = detector.validate_local()
        
        # 映射状态码
        status_map = {
            ModelStatus.VALID: "ready",
            ModelStatus.MISSING: "missing",
            ModelStatus.INCOMPLETE: "incomplete",
            ModelStatus.CORRUPTED: "incomplete",
            ModelStatus.DOWNLOADING: "downloading"
        }
        status_str = status_map.get(result["overall_status"], "missing")
        
        # 转换组件格式以兼容前端
        return {
            "status": status_str,
            "exists": result["overall_status"] != ModelStatus.MISSING,
            "valid": result["overall_status"] == ModelStatus.VALID,
            "details": {
                "components": result["components"],
                "missing_files": result["summary"]["missing_components"],
                "invalid_files": result["summary"]["corrupted_components"]
            },
            "path": str(model_path),
            "model_type": model_type,
            "model_name": detector.spec.name,
            "model_id": detector.spec.model_id
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "missing",
            "exists": False,
            "valid": False,
            "details": {"error": str(e)},
            "path": "",
            "model_type": model_type
        }

@router.post("/verify-model")
@router.get("/verify-model")
async def verify_model_integrity(request: Optional[ModelOpsRequest] = None, 
                               model_type: Optional[str] = None):
    """校验模型完整性（深度在线校验）"""
    if request is not None:
        model_type = request.model_type
    elif model_type is None:
        model_type = "zimage"

    try:
        model_path = get_model_path(model_type, "base")
        detector = create_model_detector(model_type, model_path)
        
        # 调用新实现的在线校验 (Async)
        result = await detector.validate_online()

        
        status_map = {
            ModelStatus.VALID: "ready",
            ModelStatus.MISSING: "missing", 
            ModelStatus.INCOMPLETE: "incomplete",
            ModelStatus.CORRUPTED: "incomplete"
        }
        status_str = status_map.get(result["overall_status"], "missing")
        
        valid_files_count = result["summary"]["valid_components"]
        total_files_count = result["summary"]["total_components"]

        return {
            "success": True,
            "model_type": model_type,
            "model_name": detector.spec.name,
            "model_id": detector.spec.model_id,
            "total_files": total_files_count,
            "valid_files": valid_files_count,
            "invalid_files": result["summary"]["corrupted_components"],
            "missing_files": result["summary"]["missing_components"],
            "is_complete": result["overall_status"] == ModelStatus.VALID,
            "status": status_str,
             # 为了兼容通过 /verify-model 获取详情的逻辑
             "details": {
                "components": result["components"],
                "missing_files": result["summary"]["missing_components"],
                "invalid_files": result["summary"]["corrupted_components"]
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False, 
            "status": "error", 
            "error": str(e),
            "message": str(e)
        }

@router.post("/verify-from-modelscope")
async def verify_from_modelscope():
    """从 ModelScope 在线校验模型完整性（旧版兼容）"""
    return await verify_model_integrity(ModelOpsRequest(model_type="zimage"))

@router.post("/download-model")
async def download_model(request: Optional[ModelOpsRequest] = None, model_type: Optional[str] = None):
    """下载模型（统一使用 ModelDetector 获取信息）"""
    if request is not None:
        actual_type = request.model_type
    elif model_type is not None:
        actual_type = model_type
    else:
        actual_type = "zimage"
    
    model_type = actual_type
    if state.download_process and state.download_process.poll() is None:
        raise HTTPException(status_code=400, detail="Download already in progress")

    try:
        # 使用 Detector 获取规格，不再查本地字典
        model_path = get_model_path(model_type, "base")
        detector = create_model_detector(model_type, model_path)
        spec = detector.spec
        model_id = spec.model_id
        
        if not model_id:
            raise HTTPException(status_code=400, detail=f"Model {model_type} does not support auto-download")
        
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 维持原有的脚本调用逻辑（最稳健的方式）
        download_script = PROJECT_ROOT / "scripts" / "download_fresh.py"
        if not download_script.exists():
            download_script = PROJECT_ROOT / "scripts" / "download_model.py"
        
        cmd = [
            sys.executable, 
            str(download_script),
            str(model_path),
            model_id
        ]
        
        state.add_log(f"开始下载 {spec.name} 模型...", "info")
        state.add_log(f"ModelScope ID: {model_id}", "info")
        state.add_log(f"下载目录: {model_path}", "info")
        
        state.update_download_progress(
            model_type=model_type,
            model_name=spec.name,
            total_gb=spec.size_gb
        )
        
        state.download_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        return {
            "success": True, 
            "message": f"{spec.name} 下载已启动",
            "model_type": model_type,
            "model_id": model_id,
            "path": str(model_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start download: {str(e)}")


@router.get("/download-status")
async def get_download_status():
    """Get status of the download process"""
    if state.download_process is None:
        return {"status": "idle"}
    
    return_code = state.download_process.poll()
    
    if return_code is None:
        return {"status": "running"}
    elif return_code == 0:
        # state.download_process = None  # 不要立即清除，防止日志未读完
        return {"status": "completed"}
    else:
        # state.download_process = None  # 不要立即清除，以便查看错误
        return {"status": "failed", "code": return_code}

@router.get("/gpu")
async def get_gpu_info():
    """Get GPU information using nvidia-smi (supports multi-GPU)"""
    gpus = []
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    try:
                        gpu_index = int(parts[0])
                        memory_total = float(parts[2]) / 1024  # Convert MB to GB
                        memory_used = float(parts[3]) / 1024
                        gpus.append({
                            "index": gpu_index,
                            "name": parts[1],
                            "memoryTotal": round(memory_total, 1),
                            "memoryUsed": round(memory_used, 1),
                            "memoryPercent": round((memory_used / memory_total) * 100) if memory_total > 0 else 0,
                            "utilization": int(parts[4]) if parts[4].strip() else 0,
                            "temperature": int(parts[5]) if parts[5].strip() else 0
                        })
                    except (ValueError, IndexError) as e:
                        print(f"[GPU] Failed to parse line '{line}': {e}")
                        continue
    except FileNotFoundError:
        print("[GPU] nvidia-smi not found")
    except Exception as e:
        print(f"[GPU] Error: {e}")
    
    # 返回格式兼容旧版（单卡）和新版（多卡）
    if len(gpus) == 0:
        # 无法检测到 GPU
        return {
            "name": "Unknown",
            "memoryTotal": 0,
            "memoryUsed": 0,
            "memoryPercent": 0,
            "utilization": 0,
            "temperature": 0,
            "gpus": [],
            "count": 0
        }
    elif len(gpus) == 1:
        # 单卡，保持向后兼容
        gpu = gpus[0]
        return {
            "name": gpu["name"],
            "memoryTotal": gpu["memoryTotal"],
            "memoryUsed": gpu["memoryUsed"],
            "memoryPercent": gpu["memoryPercent"],
            "utilization": gpu["utilization"],
            "temperature": gpu["temperature"],
            "gpus": gpus,
            "count": 1
        }
    else:
        # 多卡，汇总信息 + 详细列表
        total_memory = sum(g["memoryTotal"] for g in gpus)
        used_memory = sum(g["memoryUsed"] for g in gpus)
        avg_utilization = sum(g["utilization"] for g in gpus) // len(gpus)
        max_temp = max(g["temperature"] for g in gpus)
        
        return {
            "name": f"{len(gpus)}x {gpus[0]['name']}",
            "memoryTotal": round(total_memory, 1),
            "memoryUsed": round(used_memory, 1),
            "memoryPercent": round((used_memory / total_memory) * 100) if total_memory > 0 else 0,
            "utilization": avg_utilization,
            "temperature": max_temp,
            "gpus": gpus,
            "count": len(gpus)
        }

