from fastapi import APIRouter
from typing import Optional, Dict, Any
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_model_path
from src.models.model_detector import (
    create_model_detector, auto_detect_model, check_model_integrity,
    ModelStatus, ZImageDetector, LongCatDetector
)
from src.models.model_downloader import (
    create_model_downloader, get_model_download_info, DownloadStatus,
    ModelDownloadManager
)

router = APIRouter(prefix="/api/system", tags=["system"])

# 新增：基于抽象层的模型检测和下载API

@router.get("/model-detector/detect/{model_type}")
async def detect_model_with_abstract_layer(model_type: str):
    """使用抽象层检测模型完整性和状态"""
    try:
        # 获取模型路径
        model_path = get_model_path(model_type, "base")
        
        # 使用新的检测器
        detector = create_model_detector(model_type, model_path)
        
        result = detector.check_model_integrity()
        
        # 转换为前端友好的格式
        return {
            "success": True,
            "model_type": model_type,
            "model_name": detector.spec.name,
            "model_id": detector.spec.model_id,
            "status": result["overall_status"].value,
            "path": result["model_path"],
            "components": result["components"],
            "summary": result["summary"],
            "model_info": detector.get_model_info()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_type": model_type
        }

@router.get("/model-detector/auto-detect")
async def auto_detect_model_type(path: Optional[str] = None):
    """自动检测模型类型"""
    try:
        if path:
            model_path = Path(path)
        else:
            # 尝试检测默认路径
            for model_type in ["zimage", "longcat"]:
                model_path = get_model_path(model_type, "base")
                if model_path.exists():
                    break
            else:
                return {
                    "success": False,
                    "error": "未找到模型路径"
                }
        
        # 自动检测模型类型
        detected_type = auto_detect_model(model_path)
        
        if detected_type:
            # 获取详细信息
            detector = create_model_detector(detected_type, model_path)
            result = detector.check_model_integrity()
            
            return {
                "success": True,
                "detected_type": detected_type,
                "model_name": detector.spec.name,
                "model_id": detector.spec.model_id,
                "status": result["overall_status"].value,
                "path": str(model_path),
                "components": result["components"],
                "summary": result["summary"]
            }
        else:
            return {
                "success": False,
                "error": "无法识别模型类型",
                "path": str(model_path)
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@router.get("/model-downloader/info/{model_type}")
async def get_model_download_info_abstract(model_type: str):
    """获取模型下载信息"""
    try:
        info = get_model_download_info(model_type)
        if not info:
            return {
                "success": False,
                "error": f"不支持的模型类型: {model_type}"
            }
        
        return {
            "success": True,
            "model_type": model_type,
            "name": info.get("name"),
            "model_id": info.get("model_id"),
            "size_gb": info.get("size_gb"),
            "description": info.get("description")
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@router.post("/model-downloader/start/{model_type}")
async def start_model_download_abstract(model_type: str, path: Optional[str] = None):
    """开始模型下载"""
    try:
        # 确定下载路径
        if not path:
            path = get_model_path(model_type, "base")
        else:
            path = Path(path)
        
        # 检查是否已经存在
        detector = create_model_detector(model_type, path)
        integrity_result = detector.check_model_integrity()
        
        if integrity_result["overall_status"] == ModelStatus.VALID:
            return {
                "success": False,
                "error": "模型已存在且完整",
                "model_type": model_type,
                "path": str(path)
            }
        
        # 创建下载器
        downloader = create_model_downloader(model_type, path)
        
        # 开始下载
        success = downloader.start_download()
        
        if success:
            return {
                "success": True,
                "model_type": model_type,
                "path": str(path),
                "status": downloader.progress.status.value,
                "progress_percent": downloader.progress.progress_percent
            }
        else:
            return {
                "success": False,
                "error": "无法启动下载",
                "model_type": model_type
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_type": model_type
        }

@router.get("/model-downloader/progress/{model_type}")
async def get_download_progress_abstract(model_type: str, path: Optional[str] = None):
    """获取下载进度"""
    try:
        if not path:
            path = get_model_path(model_type, "base")
        else:
            path = Path(path)
        
        # 获取下载器
        downloader = download_manager.get_downloader(model_type, path)
        
        if not downloader:
            return {
                "success": False,
                "error": "未找到下载任务",
                "model_type": model_type
            }
        
        progress = downloader.get_progress()
        
        return {
            "success": True,
            "model_type": model_type,
            "status": progress.status.value,
            "progress_percent": progress.progress_percent,
            "downloaded_mb": progress.downloaded_mb,
            "total_mb": progress.total_mb,
            "speed_mbps": progress.speed_mbps,
            "eta_seconds": progress.eta_seconds,
            "current_file": progress.current_file,
            "error_message": progress.error_message
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_type": model_type
        }

@router.post("/model-downloader/cancel/{model_type}")
async def cancel_model_download_abstract(model_type: str, path: Optional[str] = None):
    """取消模型下载"""
    try:
        if not path:
            path = get_model_path(model_type, "base")
        else:
            path = Path(path)
        
        # 获取下载器
        downloader = download_manager.get_downloader(model_type, path)
        
        if not downloader:
            return {
                "success": False,
                "error": "未找到下载任务",
                "model_type": model_type
            }
        
        # 取消下载
        downloader.cancel_download()
        
        return {
            "success": True,
            "model_type": model_type,
            "status": "cancelled"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_type": model_type
        }

@router.get("/model-detector/supported-models")
async def get_supported_models():
    """获取支持的模型列表"""
    try:
        from src.models.model_detector import ModelDetectorRegistry
        from src.models.model_downloader import download_manager
        
        # 获取检测器支持的模型
        detector_models = ModelDetectorRegistry.list_detectors()
        
        # 获取下载器支持的模型
        downloader_models = download_manager.list_supported_models()
        
        # 合并信息
        models_info = {}
        
        for model_type in set(detector_models + downloader_models):
            info = download_manager.get_model_info(model_type)
            models_info[model_type] = {
                "name": info.get("name", model_type),
                "model_id": info.get("model_id", ""),
                "size_gb": info.get("size_gb", 0),
                "description": info.get("description", ""),
                "supports_detection": model_type in detector_models,
                "supports_download": model_type in downloader_models
            }
        
        return {
            "success": True,
            "models": models_info
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }