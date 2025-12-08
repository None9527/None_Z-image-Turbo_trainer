# -*- coding: utf-8 -*-
"""
模型检测与下载抽象层 - 支持多模型结构的统一检测和下载

提供标准化的模型检测、完整性验证和下载管理功能，
从 ModelScope 动态获取文件列表进行校验，而不是硬编码。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """模型状态枚举"""
    MISSING = "missing"           # 完全缺失
    INCOMPLETE = "incomplete"     # 不完整
    CORRUPTED = "corrupted"       # 损坏
    VALID = "valid"               # 有效
    DOWNLOADING = "downloading"   # 下载中


@dataclass
class RemoteFileInfo:
    """远程文件信息"""
    path: str                    # 文件相对路径
    size: int = 0               # 文件大小(bytes)
    sha256: str = ""            # 校验码


@dataclass
class ModelSpec:
    """模型规格定义"""
    name: str                    # 模型名称
    model_id: str               # 远程模型ID (ModelScope)
    description: str            # 模型描述
    default_path: str           # 默认本地路径
    size_gb: float = 0.0        # 预估大小(GB)
    aliases: List[str] = field(default_factory=list)  # 别名列表


class ModelDetector(ABC):
    """模型检测器抽象基类 - 从 ModelScope 动态获取文件列表进行校验"""
    
    # 缓存远程文件列表
    _remote_files_cache: Dict[str, List[RemoteFileInfo]] = {}
    
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.spec = self.get_spec()
    
    @abstractmethod
    def get_spec(self) -> ModelSpec:
        """获取模型规格定义"""
        pass
    
    @abstractmethod
    def detect_model_type(self) -> Optional[str]:
        """检测模型类型，返回模型类型标识或None"""
        pass
    
    def get_remote_files(self, force_refresh: bool = False) -> List[RemoteFileInfo]:
        """从 ModelScope 获取远程文件列表"""
        model_id = self.spec.model_id
        
        # 检查缓存
        if not force_refresh and model_id in self._remote_files_cache:
            return self._remote_files_cache[model_id]
        
        try:
            from modelscope.hub.api import HubApi
            api = HubApi()
            
            # 获取文件列表
            files_info = api.get_model_files(model_id)
            
            remote_files = []
            for file_info in files_info:
                # file_info 可能是 dict 或其他格式
                if isinstance(file_info, dict):
                    path = file_info.get('Path', file_info.get('path', ''))
                    size = file_info.get('Size', file_info.get('size', 0))
                    sha256 = file_info.get('Sha256', file_info.get('sha256', ''))
                elif hasattr(file_info, 'path'):
                    path = file_info.path
                    size = getattr(file_info, 'size', 0)
                    sha256 = getattr(file_info, 'sha256', '')
                else:
                    # 字符串格式
                    path = str(file_info)
                    size = 0
                    sha256 = ''
                
                if path:  # 过滤空路径
                    remote_files.append(RemoteFileInfo(
                        path=path,
                        size=size,
                        sha256=sha256
                    ))
            
            # 缓存结果
            self._remote_files_cache[model_id] = remote_files
            logger.info(f"从 ModelScope 获取到 {len(remote_files)} 个文件")
            return remote_files
            
        except Exception as e:
            logger.warning(f"无法从 ModelScope 获取文件列表: {e}")
            return []
    
    def validate_file(self, remote_file: RemoteFileInfo) -> Tuple[bool, str]:
        """验证单个文件"""
        local_path = self.model_path / remote_file.path
        
        # 检查文件是否存在
        if not local_path.exists():
            return False, f"文件缺失: {remote_file.path}"
        
        # 检查文件大小（如果远程有大小信息）
        if remote_file.size > 0:
            local_size = local_path.stat().st_size
            if local_size != remote_file.size:
                return False, f"文件大小不匹配: 本地 {local_size} vs 远程 {remote_file.size}"
        
        return True, "有效"
    
    def check_model_integrity(self) -> Dict[str, Any]:
        """检查模型完整性 - 基于 ModelScope 远程文件列表"""
        results = {
            "model_type": self.__class__.__name__,
            "model_path": str(self.model_path),
            "overall_status": ModelStatus.MISSING,
            "components": {},
            "summary": {
                "total_components": 0,
                "valid_components": 0,
                "missing_components": [],
                "corrupted_components": []
            }
        }
        
        # 检查本地路径是否存在
        if not self.model_path.exists():
            results["overall_status"] = ModelStatus.MISSING
            results["error"] = "模型路径不存在"
            return results
        
        # 获取远程文件列表
        remote_files = self.get_remote_files()
        
        if not remote_files:
            # 无法获取远程列表，回退到基础检查
            return self._fallback_integrity_check(results)
        
        results["summary"]["total_components"] = len(remote_files)
        
        valid_count = 0
        missing_files = []
        corrupted_files = []
        
        # 检查每个远程文件
        for remote_file in remote_files:
            is_valid, message = self.validate_file(remote_file)
            
            local_exists = (self.model_path / remote_file.path).exists()
            status = ModelStatus.VALID if is_valid else (ModelStatus.MISSING if not local_exists else ModelStatus.CORRUPTED)
            
            results["components"][remote_file.path] = {
                "status": status.value,
                "valid": is_valid,
                "message": message,
                "expected_size": remote_file.size,
                "exists": local_exists
            }
            
            if is_valid:
                valid_count += 1
            elif not local_exists:
                missing_files.append(remote_file.path)
            else:
                corrupted_files.append(remote_file.path)
        
        # 确定整体状态
        if valid_count == len(remote_files):
            results["overall_status"] = ModelStatus.VALID
        elif valid_count == 0:
            results["overall_status"] = ModelStatus.MISSING
        elif missing_files:
            results["overall_status"] = ModelStatus.INCOMPLETE
        else:
            results["overall_status"] = ModelStatus.CORRUPTED
        
        # 更新摘要
        results["summary"]["valid_components"] = valid_count
        results["summary"]["missing_components"] = missing_files
        results["summary"]["corrupted_components"] = corrupted_files
        
        return results
    
    def _fallback_integrity_check(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """回退检查：无法获取远程列表时，检查基础文件"""
        # 检查关键文件是否存在
        key_files = ["model_index.json", "config.json"]
        key_dirs = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        
        valid_count = 0
        total_count = 0
        
        for f in key_files:
            total_count += 1
            path = self.model_path / f
            exists = path.exists()
            results["components"][f] = {
                "status": ModelStatus.VALID.value if exists else ModelStatus.MISSING.value,
                "valid": exists,
                "exists": exists,
                "message": "存在" if exists else "缺失"
            }
            if exists:
                valid_count += 1
            else:
                results["summary"]["missing_components"].append(f)
        
        for d in key_dirs:
            total_count += 1
            path = self.model_path / d
            exists = path.exists() and path.is_dir()
            # 检查目录是否有内容
            has_content = exists and any(path.iterdir()) if exists else False
            results["components"][d] = {
                "status": ModelStatus.VALID.value if has_content else (ModelStatus.MISSING.value if not exists else ModelStatus.INCOMPLETE.value),
                "valid": has_content,
                "exists": exists,
                "message": "有效" if has_content else ("缺失" if not exists else "目录为空")
            }
            if has_content:
                valid_count += 1
            else:
                results["summary"]["missing_components"].append(d)
        
        results["summary"]["total_components"] = total_count
        results["summary"]["valid_components"] = valid_count
        
        if valid_count == total_count:
            results["overall_status"] = ModelStatus.VALID
        elif valid_count == 0:
            results["overall_status"] = ModelStatus.MISSING
        else:
            results["overall_status"] = ModelStatus.INCOMPLETE
        
        results["note"] = "无法连接 ModelScope，使用本地基础检查"
        return results
    
    def get_download_progress(self) -> Optional[Dict[str, Any]]:
        """获取下载进度信息"""
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型基本信息"""
        return {
            "name": self.spec.name,
            "model_id": self.spec.model_id,
            "description": self.spec.description,
            "size_gb": self.spec.size_gb,
            "aliases": self.spec.aliases or [],
            "default_path": self.spec.default_path
        }


class ZImageDetector(ModelDetector):
    """Z-Image 模型检测器"""
    
    def get_spec(self) -> ModelSpec:
        return ModelSpec(
            name="Z-Image Turbo",
            model_id="Tongyi-MAI/Z-Image-Turbo",
            description="阿里Z-Image Turbo 10步加速模型",
            default_path="zimage_models",
            size_gb=32.0,
            aliases=["zimage", "z-image", "z-image-turbo"]
        )
    
    def detect_model_type(self) -> Optional[str]:
        """检测是否为Z-Image模型"""
        try:
            # 检查model_index.json中的模型类型
            model_index_path = self.model_path / "model_index.json"
            if model_index_path.exists():
                with open(model_index_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # Z-Image的特征检查
                if "_class_name" in config:
                    class_name = config["_class_name"].lower()
                    if "zimage" in class_name or "z-image" in class_name:
                        return "zimage"
                
                # 检查组件结构
                expected_components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
                if all(comp in config for comp in expected_components):
                    return "zimage"
            
            # 检查配置文件
            config_path = self.model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                if "model_name" in config and "z-image" in config["model_name"].lower():
                    return "zimage"
            
            return None
            
        except Exception:
            return None


class LongCatDetector(ModelDetector):
    """LongCat-Image 模型检测器"""
    
    def get_spec(self) -> ModelSpec:
        return ModelSpec(
            name="LongCat-Image",
            model_id="meituan-longcat/LongCat-Image",
            description="美团LongCat-Image基于FLUX架构",
            default_path="longcat_models",
            size_gb=35.0,
            aliases=["longcat", "longcat-image", "long_cat"]
        )
    
    def detect_model_type(self) -> Optional[str]:
        """检测是否为LongCat模型"""
        try:
            # 检查model_index.json
            model_index_path = self.model_path / "model_index.json"
            if model_index_path.exists():
                with open(model_index_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # LongCat的特征检查
                if "_class_name" in config:
                    class_name = config["_class_name"].lower()
                    if "longcat" in class_name or "long_cat" in class_name:
                        return "longcat"
                
                # 检查特殊组件
                if "transformer" in config and "text_encoder" in config:
                    # LongCat使用Qwen2作为text_encoder
                    text_encoder_path = self.model_path / "text_encoder" / "config.json"
                    if text_encoder_path.exists():
                        with open(text_encoder_path, 'r', encoding='utf-8') as f:
                            text_config = json.load(f)
                            if "qwen" in text_config.get("model_type", "").lower():
                                return "longcat"
            
            # 检查配置文件
            config_path = self.model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                if "model_name" in config and "longcat" in config["model_name"].lower():
                    return "longcat"
            
            return None
            
        except Exception:
            return None


class ModelDetectorRegistry:
    """模型检测器注册表"""
    
    _detectors: Dict[str, type] = {}
    
    @classmethod
    def register(cls, model_type: str):
        """注册检测器装饰器"""
        def decorator(detector_class: type):
            cls._detectors[model_type.lower()] = detector_class
            return detector_class
        return decorator
    
    @classmethod
    def get_detector(cls, model_type: str, model_path: Union[str, Path]) -> ModelDetector:
        """获取检测器实例"""
        detector_class = cls._detectors.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"未知的模型类型: {model_type}")
        return detector_class(model_path)
    
    @classmethod
    def list_detectors(cls) -> List[str]:
        """列出所有注册的检测器"""
        return list(cls._detectors.keys())
    
    @classmethod
    def auto_detect(cls, model_path: Union[str, Path]) -> Optional[str]:
        """自动检测模型类型"""
        model_path = Path(model_path)
        
        # 尝试所有注册的检测器
        for model_type, detector_class in cls._detectors.items():
            try:
                detector = detector_class(model_path)
                if detector.detect_model_type():
                    return model_type
            except Exception:
                continue
        
        return None


# 注册检测器
@ModelDetectorRegistry.register("zimage")
class RegisteredZImageDetector(ZImageDetector):
    pass

@ModelDetectorRegistry.register("longcat")
class RegisteredLongCatDetector(LongCatDetector):
    pass


def create_model_detector(model_type: str, model_path: Union[str, Path]) -> ModelDetector:
    """工厂函数：创建模型检测器"""
    return ModelDetectorRegistry.get_detector(model_type, model_path)


def auto_detect_model(model_path: Union[str, Path]) -> Optional[str]:
    """自动检测模型类型"""
    return ModelDetectorRegistry.auto_detect(model_path)


def check_model_integrity(model_type: str, model_path: Union[str, Path]) -> Dict[str, Any]:
    """检查模型完整性"""
    detector = create_model_detector(model_type, model_path)
    return detector.check_model_integrity()