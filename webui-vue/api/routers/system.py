from fastapi import APIRouter, HTTPException
import subprocess
import sys
import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

from core.config import PROJECT_ROOT, MODEL_PATH
from core import state

router = APIRouter(prefix="/api/system", tags=["system"])

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
        "name": "LongCat-Image",
        "model_id": None,  # 无自动下载
        "path_env": "LONGCAT_MODEL_PATH",
        "expected_files": {
            "transformer": {
                "required": True,
                "files": {
                    "config.json": {"min_size": 100},
                }
            },
            "vae": {
                "required": True,
                "files": {
                    "config.json": {"min_size": 100},
                }
            },
            "text_encoder": {
                "required": True,
                "files": {
                    "config.json": {"min_size": 100},
                }
            },
            "tokenizer": {
                "required": True,
                "files": {
                    "tokenizer.json": {"min_size": 1000},
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

def validate_json_file(file_path: Path) -> tuple[bool, str]:
    """验证JSON文件是否有效"""
    try:
        if not file_path.exists():
            return False, "文件不存在"
        
        if file_path.stat().st_size == 0:
            return False, "文件为空"
        
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True, "有效"
    except json.JSONDecodeError as e:
        return False, f"JSON解析错误: {str(e)}"
    except Exception as e:
        return False, f"读取错误: {str(e)}"

def check_safetensors_index_files(model_dir: Path, index_file_name: str = "model.safetensors.index.json") -> tuple[bool, dict]:
    """检查safetensors分片模型文件的完整性"""
    try:
        index_path = model_dir / index_file_name
        if not index_path.exists():
            return False, {"error": f"索引文件不存在: {index_file_name}"}
        
        # 验证索引文件
        is_valid, message = validate_json_file(index_path)
        if not is_valid:
            return False, {"error": f"索引文件无效: {message}"}
        
        import json
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        if "weight_map" not in index_data:
            return False, {"error": "索引文件缺少weight_map字段"}
        
        weight_map = index_data["weight_map"]
        required_files = set(weight_map.values())
        
        missing_files = []
        empty_files = []
        
        for file_name in required_files:
            file_path = model_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
            elif file_path.stat().st_size == 0:
                empty_files.append(file_name)
        
        result = {
            "total_files": len(required_files),
            "missing_files": missing_files,
            "empty_files": empty_files,
            "index_valid": True
        }
        
        if missing_files or empty_files:
            result["error"] = f"缺少文件: {missing_files}, 空文件: {empty_files}"
            return False, result
        
        return True, result
        
    except Exception as e:
        return False, {"error": f"检查分片文件时出错: {str(e)}"}

def check_model_component(model_path: Path, component_name: str) -> tuple[bool, dict]:
    """检查单个模型组件的完整性"""
    component_path = model_path / component_name
    result = {
        "exists": False,
        "valid": False,
        "details": {},
        "error": None
    }
    
    try:
        if not component_path.exists():
            result["error"] = "组件目录不存在"
            return False, result
        
        result["exists"] = True
        
        if component_name.endswith('.json'):
            # 直接检查JSON文件
            is_valid, message = validate_json_file(component_path)
            result["valid"] = is_valid
            result["details"]["json_valid"] = is_valid
            result["details"]["json_message"] = message
            if not is_valid:
                result["error"] = message
            return is_valid, result
        
        # 检查目录中的关键文件
        if component_name == "transformer":
            # Transformer可能有分片模型文件
            index_files = list(component_path.glob("*.safetensors.index.json"))
            if index_files:
                # 有分片模型索引文件
                is_complete, shard_info = check_safetensors_index_files(component_path, index_files[0].name)
                result["valid"] = is_complete
                result["details"]["shard_check"] = shard_info
                if not is_complete:
                    result["error"] = shard_info.get("error", "分片模型文件不完整")
                return is_complete, result
            else:
                # 检查单个模型文件
                model_files = list(component_path.glob("*.safetensors")) + list(component_path.glob("*.bin"))
                if not model_files:
                    result["error"] = "未找到模型文件"
                    return False, result
                
                # 检查文件大小
                model_file = model_files[0]
                file_size = model_file.stat().st_size
                result["details"]["model_file"] = str(model_file.name)
                result["details"]["file_size"] = file_size
                
                if file_size < 1024 * 1024:  # 小于1MB认为无效
                    result["error"] = "模型文件过小"
                    return False, result
                
                result["valid"] = True
                return True, result
        
        elif component_name == "text_encoder":
            # Text encoder也可能有分片文件
            index_files = list(component_path.glob("model.safetensors.index.json"))
            if index_files:
                is_complete, shard_info = check_safetensors_index_files(component_path)
                result["valid"] = is_complete
                result["details"]["shard_check"] = shard_info
                if not is_complete:
                    result["error"] = shard_info.get("error", "分片模型文件不完整")
                return is_complete, result
            else:
                # 检查单个模型文件
                model_files = list(component_path.glob("*.safetensors")) + list(component_path.glob("*.bin"))
                if not model_files:
                    result["error"] = "未找到模型文件"
                    return False, result
                
                model_file = model_files[0]
                file_size = model_file.stat().st_size
                result["details"]["model_file"] = str(model_file.name)
                result["details"]["file_size"] = file_size
                
                if file_size < 1024 * 1024:  # 小于1MB认为无效
                    result["error"] = "模型文件过小"
                    return False, result
                
                result["valid"] = True
                return True, result
        
        elif component_name in ["vae", "tokenizer"]:
            # VAE和Tokenizer通常有配置文件和模型文件
            config_files = list(component_path.glob("config.json"))
            model_files = list(component_path.glob("*.safetensors")) + list(component_path.glob("*.bin"))
            
            if config_files:
                config_valid, config_message = validate_json_file(config_files[0])
                result["details"]["config_valid"] = config_valid
                result["details"]["config_message"] = config_message
                if not config_valid:
                    result["error"] = f"配置文件无效: {config_message}"
                    return False, result
            
            if component_name == "vae" and model_files:
                # VAE应该有模型文件
                model_file = model_files[0]
                file_size = model_file.stat().st_size
                result["details"]["model_file"] = str(model_file.name)
                result["details"]["file_size"] = file_size
                
                if file_size < 1024 * 1024:  # 小于1MB认为无效
                    result["error"] = "模型文件过小"
                    return False, result
            
            elif component_name == "tokenizer":
                # Tokenizer应该有tokenizer.json
                tokenizer_files = list(component_path.glob("tokenizer.json"))
                if not tokenizer_files:
                    result["error"] = "未找到tokenizer.json文件"
                    return False, result
                
                tokenizer_valid, tokenizer_message = validate_json_file(tokenizer_files[0])
                result["details"]["tokenizer_valid"] = tokenizer_valid
                result["details"]["tokenizer_message"] = tokenizer_message
                if not tokenizer_valid:
                    result["error"] = f"Tokenizer文件无效: {tokenizer_message}"
                    return False, result
            
            result["valid"] = True
            return True, result
        
        elif component_name == "scheduler":
            # Scheduler应该有配置文件
            config_files = list(component_path.glob("scheduler_config.json"))
            if not config_files:
                result["error"] = "未找到scheduler_config.json文件"
                return False, result
            
            config_valid, config_message = validate_json_file(config_files[0])
            result["details"]["config_valid"] = config_valid
            result["details"]["config_message"] = config_message
            if not config_valid:
                result["error"] = f"Scheduler配置文件无效: {config_message}"
                return False, result
            
            result["valid"] = True
            return True, result
        
        else:
            # 其他组件，只要存在就认为是有效的
            result["valid"] = True
            return True, result
            
    except Exception as e:
        result["error"] = f"检查组件时出错: {str(e)}"
        return False, result

def get_modelscope_file_list() -> Optional[Dict[str, Any]]:
    """从 ModelScope 获取模型文件列表（用于校验）"""
    try:
        from modelscope.hub.api import HubApi
        api = HubApi()
        
        # 获取模型文件列表
        files = api.get_model_files(MODELSCOPE_MODEL_ID)
        return {
            "success": True,
            "files": files,
            "model_id": MODELSCOPE_MODEL_ID
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def check_component_enhanced(model_path: Path, component_name: str, expected: dict) -> dict:
    """增强的组件检查"""
    result = {
        "exists": False,
        "valid": False,
        "status": "missing",  # missing, incomplete, corrupted, valid
        "details": {},
        "error": None
    }
    
    try:
        if component_name.endswith('.json'):
            # 直接检查JSON文件
            file_path = model_path / component_name
            if not file_path.exists():
                result["error"] = "文件不存在"
                return result
            
            result["exists"] = True
            
            file_size = file_path.stat().st_size
            min_size = expected.get("min_size", 0)
            
            if file_size < min_size:
                result["status"] = "corrupted"
                result["error"] = f"文件过小 ({file_size} < {min_size} bytes)"
                return result
            
            is_valid, message = validate_json_file(file_path)
            if not is_valid:
                result["status"] = "corrupted"
                result["error"] = message
                return result
            
            result["valid"] = True
            result["status"] = "valid"
            result["details"]["file_size"] = file_size
            return result
        
        # 检查目录组件
        component_path = model_path / component_name
        if not component_path.exists():
            result["error"] = "目录不存在"
            return result
        
        result["exists"] = True
        expected_files = expected.get("files", {})
        
        missing_files = []
        corrupted_files = []
        valid_files = []
        
        for file_name, file_spec in expected_files.items():
            file_path = component_path / file_name
            
            if not file_path.exists():
                missing_files.append(file_name)
                continue
            
            file_size = file_path.stat().st_size
            min_size = file_spec.get("min_size", 0)
            
            if file_size < min_size:
                corrupted_files.append(f"{file_name} (大小异常: {file_size})")
                continue
            
            # 对于 JSON 文件，验证格式
            if file_name.endswith('.json'):
                is_valid, message = validate_json_file(file_path)
                if not is_valid:
                    corrupted_files.append(f"{file_name} (格式错误)")
                    continue
            
            valid_files.append(file_name)
        
        # 检查分片文件完整性
        index_files = list(component_path.glob("*.index.json"))
        for index_file in index_files:
            is_complete, shard_info = check_safetensors_index_files(component_path, index_file.name)
            if not is_complete:
                missing_shards = shard_info.get("missing_files", [])
                if missing_shards:
                    missing_files.extend([f"shard:{f}" for f in missing_shards[:3]])
                    if len(missing_shards) > 3:
                        missing_files.append(f"...还有{len(missing_shards)-3}个分片")
        
        result["details"] = {
            "expected_files": list(expected_files.keys()),
            "valid_files": valid_files,
            "missing_files": missing_files,
            "corrupted_files": corrupted_files
        }
        
        if missing_files:
            result["status"] = "incomplete"
            result["error"] = f"缺少文件: {', '.join(missing_files[:3])}"
        elif corrupted_files:
            result["status"] = "corrupted"
            result["error"] = f"文件损坏: {', '.join(corrupted_files[:2])}"
        else:
            result["valid"] = True
            result["status"] = "valid"
        
        return result
        
    except Exception as e:
        result["error"] = f"检查错误: {str(e)}"
        result["status"] = "error"
        return result

@router.get("/model-status")
async def get_model_status(model_type: str = "zimage"):
    """增强的模型状态检测 - 支持多模型类型
    
    Args:
        model_type: 模型类型 (zimage, longcat)
    """
    try:
        # 获取模型规格
        if model_type not in MODEL_SPECS:
            model_type = "zimage"
        
        spec = MODEL_SPECS[model_type]
        expected_files = spec["expected_files"]
        
        # 确定模型路径
        if model_type == "zimage":
            model_path = Path(MODEL_PATH)
        elif model_type == "longcat":
            # LongCat 使用环境变量或默认路径
            longcat_path = os.environ.get("LONGCAT_MODEL_PATH", "")
            if not longcat_path:
                # 尝试常见路径
                possible_paths = [
                    Path("D:/AI/LongCat-Image/models"),
                    Path(MODEL_PATH).parent / "longcat-image",
                ]
                for p in possible_paths:
                    if p.exists():
                        longcat_path = str(p)
                        break
            model_path = Path(longcat_path) if longcat_path else Path("")
        else:
            model_path = Path(MODEL_PATH)
        
        components = {}
        overall_valid = True
        
        if model_path.exists():
            for name, file_spec in expected_files.items():
                result = check_component_enhanced(model_path, name, file_spec)
                components[name] = result
                
                if file_spec.get("required", False) and not result["valid"]:
                    overall_valid = False
        else:
            # 路径不存在，所有组件标记为缺失
            for name, file_spec in expected_files.items():
                components[name] = {
                    "exists": False,
                    "valid": False,
                    "reason": "模型路径不存在"
                }
                if file_spec.get("required", False):
                    overall_valid = False
        
        # 统计
        total = len(expected_files)
        valid_count = sum(1 for c in components.values() if c["valid"])
        
        return {
            "exists": valid_count > 0,
            "valid": overall_valid,
            "details": components,
            "path": str(model_path),
            "model_type": model_type,
            "model_name": spec["name"],
            "model_id": spec.get("model_id"),
            "summary": {
                "total_components": total,
                "valid_components": valid_count,
                "invalid_components": [name for name, c in components.items() if c["exists"] and not c["valid"]],
                "missing_components": [name for name, c in components.items() if not c["exists"]]
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "exists": False,
            "valid": False,
            "error": str(e),
            "details": {},
            "path": str(MODEL_PATH),
            "model_type": model_type
        }

@router.post("/verify-from-modelscope")
async def verify_from_modelscope():
    """从 ModelScope 在线校验模型完整性"""
    try:
        model_path = Path(MODEL_PATH)
        
        # 尝试获取 ModelScope 文件列表
        ms_info = get_modelscope_file_list()
        
        if not ms_info.get("success"):
            return {
                "success": False,
                "error": f"无法连接 ModelScope: {ms_info.get('error')}",
                "offline_check": await get_model_status()
            }
        
        remote_files = ms_info.get("files", [])
        
        # 比对本地文件
        local_files = []
        missing_files = []
        size_mismatch = []
        
        for remote_file in remote_files:
            file_name = remote_file.get("Name", remote_file.get("name", ""))
            remote_size = remote_file.get("Size", remote_file.get("size", 0))
            
            local_path = model_path / file_name
            
            if local_path.exists():
                local_size = local_path.stat().st_size
                local_files.append({
                    "name": file_name,
                    "local_size": local_size,
                    "remote_size": remote_size,
                    "match": abs(local_size - remote_size) < 1024  # 允许1KB误差
                })
                
                if abs(local_size - remote_size) >= 1024:
                    size_mismatch.append(file_name)
            else:
                missing_files.append(file_name)
        
        return {
            "success": True,
            "model_id": MODELSCOPE_MODEL_ID,
            "total_remote_files": len(remote_files),
            "local_files": len(local_files),
            "missing_files": missing_files,
            "size_mismatch": size_mismatch,
            "is_complete": len(missing_files) == 0 and len(size_mismatch) == 0
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@router.post("/download-model")
async def download_model():
    """Download model from ModelScope"""
    if state.download_process and state.download_process.poll() is None:
        raise HTTPException(status_code=400, detail="Download already in progress")

    try:
        model_path = Path(MODEL_PATH)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Command to download using python script
        cmd = [
            sys.executable, 
            str(PROJECT_ROOT / "scripts" / "download_model.py"),
            str(model_path)
        ]
        
        # Start download in background
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
        
        return {"success": True, "message": "Download started"}
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
        state.download_process = None
        return {"status": "completed"}
    else:
        state.download_process = None
        return {"status": "failed", "code": return_code}

@router.get("/gpu")
async def get_gpu_info():
    """Get GPU information using nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=5
        )
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 5:
                memory_total = float(parts[1]) / 1024  # Convert MB to GB
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
