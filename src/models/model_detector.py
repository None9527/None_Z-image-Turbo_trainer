# -*- coding: utf-8 -*-
"""
模型检测与下载抽象层 - 生产级异步 Schema 校验系统

特性：
1. 异步非阻塞 Hash 计算 (Async IO + ThreadPool)
2. 数据驱动 Schema 配置 (Data-Driven)
3. 健壮的临时文件过滤 (.part, .downloading)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import logging
from enum import Enum
import hashlib
import asyncio
import concurrent.futures

logger = logging.getLogger(__name__)

# --- Configuration (Simulated External Config) ---

MODEL_SCHEMAS = {
    "zimage": {
        "rules": [
            {"type": "file", "path": "model_index.json", "min_size": 10},
            # 权重文件校验 (防止仅分发 json)
            # Transformer (分片 -> 检查 index)
            {"type": "file", "path": "transformer/diffusion_pytorch_model.safetensors.index.json", "min_size": 10},
            {"type": "file", "path": "transformer/config.json", "min_size": 10},
            # Text Encoder (分片 -> 检查 index)
            {"type": "file", "path": "text_encoder/model.safetensors.index.json", "min_size": 10},
            {"type": "file", "path": "text_encoder/config.json", "min_size": 10},
            # VAE (单文件 -> 检查 safetensors, >100MB)
            {"type": "file", "path": "vae/diffusion_pytorch_model.safetensors", "min_size": 100 * 1024 * 1024},
            {"type": "file", "path": "vae/config.json", "min_size": 10},
        ]
    }
}


class ModelStatus(Enum):
    """模型状态枚举"""
    MISSING = "missing"           # 完全缺失
    INCOMPLETE = "incomplete"     # 不完整
    CORRUPTED = "corrupted"       # 损坏
    VALID = "valid"               # 有效
    DOWNLOADING = "downloading"   # 下载中 (暂未完全集成到 detector)


@dataclass
class ModelSpec:
    """模型规格定义"""
    name: str
    model_id: str
    description: str
    default_path: str
    size_gb: float
    aliases: List[str] = field(default_factory=list)


# --- Helper Functions ---

def _calculate_file_hash(file_path: Path, chunk_size=8192) -> str:
    """CPU 密集型哈希计算 (将在线程池中运行)"""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                sha256.update(chunk)
        return sha256.hexdigest().lower()
    except Exception as e:
        logger.error(f"Error hashing {file_path}: {e}")
        return ""

def _is_temp_file(path: Path) -> bool:
    """检查是否为下载临时文件"""
    suffix = path.suffix.lower()
    return suffix in ['.part', '.downloading', '.tmp', '.aria2']


# --- Schema Rules ---

class ValidationRule(ABC):
    @abstractmethod
    def validate(self, base_path: Path) -> Tuple[bool, Dict[str, Any]]:
        pass

@dataclass
class FileRule(ValidationRule):
    path: str
    required: bool = True
    min_size: int = 0
    
    def validate(self, base_path: Path) -> Tuple[bool, Dict[str, Any]]:
        file_path = base_path / self.path
        
        # 1. 基础存在性检查
        exists = file_path.exists()
        
        # 2. 临时文件/空文件过滤
        if exists:
            if _is_temp_file(file_path):
                exists = False # 把临时文件视为不存在，避免误判 Ready
            elif file_path.stat().st_size == 0:
                exists = False # 空文件视为不存在
        
        detail = {
            "type": "file",
            "path": self.path,
            "exists": exists,
            "valid": exists,
            "required": self.required,
            "message": "存在" if exists else "缺失"
        }
        
        # 3. 大小检查
        if exists and self.min_size > 0:
            size = file_path.stat().st_size
            if size < self.min_size:
                detail["valid"] = False
                detail["message"] = f"文件过小 ({size} < {self.min_size})"
        
        if not exists and self.required:
            detail["valid"] = False
            
        return detail["valid"], {self.path: detail}

@dataclass
class AlternativeRule(ValidationRule):
    paths: List[str]
    name: str
    required: bool = True
    
    def validate(self, base_path: Path) -> Tuple[bool, Dict[str, Any]]:
        found_any = False
        details = {}
        
        for path in self.paths:
            file_path = base_path / path
            exists = file_path.exists()
            
            # 同样应用临时文件过滤
            if exists:
                if _is_temp_file(file_path) or file_path.stat().st_size == 0:
                    exists = False
            
            if exists:
                found_any = True
                details[path] = {
                    "type": "alternative",
                    "group": self.name,
                    "path": path,
                    "exists": True,
                    "valid": True,
                    "message": "有效 (替代组)"
                }
        
        if not found_any and self.required:
            primary = self.paths[0]
            details[primary] = {
                "type": "alternative_missing",
                "group": self.name,
                "path": primary,
                "exists": False,
                "valid": False,
                "required": True,
                "message": f"缺失 (需 { ' 或 '.join(self.paths) } 之一)"
            }
            return False, details
            
        return True, details


class ModelDetector(ABC):
    """模型检测器基类"""
    
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.spec = self.get_spec()
        self._load_schema()
        
    @abstractmethod
    def get_spec(self) -> ModelSpec:
        pass
    
    def _load_schema(self):
        """加载配置化 Schema"""
        # 尝试从 MODEL_SCHEMAS 获取，如果没有则退化为空列表
        schema_data = MODEL_SCHEMAS.get(self.detect_model_type_key(), {"rules": []})
        self.rules = []
        for rule_cfg in schema_data.get("rules", []):
            rtype = rule_cfg.get("type")
            if rtype == "file":
                self.rules.append(FileRule(
                    path=rule_cfg["path"], 
                    required=rule_cfg.get("required", True),
                    min_size=rule_cfg.get("min_size", 0)
                ))
            elif rtype == "alternative":
                self.rules.append(AlternativeRule(
                    paths=rule_cfg["paths"],
                    name=rule_cfg["name"],
                    required=rule_cfg.get("required", True)
                ))

    def detect_model_type_key(self) -> str:
        # 简单映射，子类可覆盖
        return "zimage"

    def validate_local(self) -> Dict[str, Any]:
        """本地快速校验"""
        if not self.model_path.exists():
            return self._build_result(ModelStatus.MISSING, {}, ["根目录不存在"], [])
            
        details = {}
        missing = []
        corrupted = []
        all_valid = True
        
        for rule in self.rules:
            is_valid, rule_details = rule.validate(self.model_path)
            details.update(rule_details)
            if not is_valid:
                all_valid = False
                for k, v in rule_details.items():
                    if not v["valid"]:
                        # 简化的缺失/损坏判定
                        if not v["exists"]: missing.append(k)
                        else: corrupted.append(k)
        
        status = ModelStatus.VALID if all_valid else ModelStatus.INCOMPLETE
        if len(missing) > 0 and len(missing) >= len(self.rules): # 大部分缺失
             status = ModelStatus.MISSING

        return self._build_result(status, details, missing, corrupted)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型基本信息"""
        return {
            "name": self.spec.name,
            "model_id": self.spec.model_id,
            "description": self.spec.description,
            "size_gb": self.spec.size_gb,
            "default_path": self.spec.default_path
        }

    def _build_result(self, status: ModelStatus, components: Dict, missing: List, corrupted: List) -> Dict[str, Any]:
        """构建标准返回结果"""
        valid_count = len([c for c in components.values() if c.get("valid", False)])
        total_count = len(components)
        
        return {
            "model_type": self.spec.aliases[0] if self.spec.aliases else "unknown",
            "model_path": str(self.model_path),
            "overall_status": status,
            "components": components,
            "summary": {
                "total_components": total_count,
                "valid_components": valid_count,
                "missing_components": missing,
                "corrupted_components": corrupted
            }
        }

    async def validate_online(self) -> Dict[str, Any]:
        """异步在线深度校验 (Non-blocking)"""
        try:
            # 1. 异步获取文件列表 (run_in_executor 避免 api request 阻塞)
            loop = asyncio.get_running_loop()
            
            def _fetch_info():
                from modelscope.hub.api import HubApi
                return HubApi().get_model_files(self.spec.model_id)
            
            files_info = await loop.run_in_executor(None, _fetch_info)
            
        except Exception as e:
            return {
                "overall_status": ModelStatus.MISSING, 
                "error": f"ModelScope 连接失败: {e}",
                "components": {}, "summary": {}
            }

        remote_map = {}
        for f in files_info:
            if isinstance(f, dict):
                path = f.get('Path')
                if not path or any(x in path.lower() for x in ['readme', '.git', 'demo']): continue
                if f.get('Type') == 'tree': continue
                remote_map[path] = {"size": f.get("Size", 0), "sha256": f.get("Sha256", "")}
        
        details = {}
        missing = []
        corrupted = []
        valid_count = 0
        
        # 2. 并行 Hash 计算
        # 将 hash 任务分发到线程池
        check_tasks = []
        
        for path, info in remote_map.items():
            local_path = self.model_path / path
            check_tasks.append(self._verify_single_file(local_path, path, info))
            
        results = await asyncio.gather(*check_tasks)
        
        for res in results:
            path = res["path"]
            details[path] = res
            if res["valid"]: valid_count += 1
            elif not res["exists"]: missing.append(path)
            else: corrupted.append(path)

        status = ModelStatus.VALID
        if missing: status = ModelStatus.INCOMPLETE
        if corrupted: status = ModelStatus.CORRUPTED
        if len(missing) == len(remote_map): status = ModelStatus.MISSING
        
        return {
            "model_type": self.spec.aliases[0] if self.spec.aliases else "unknown",
            "model_path": str(self.model_path),
            "overall_status": status,
            "components": details,
            "summary": {
                "total_components": len(remote_map),
                "valid_components": valid_count,
                "missing_components": missing,
                "corrupted_components": corrupted
            }
        }

    async def _verify_single_file(self, local_path: Path, rel_path: str, info: Dict) -> Dict:
        """单文件验证逻辑"""
        res = {"path": rel_path, "type": "file", "valid": False, "exists": False, "message": ""}
        
        if not local_path.exists():
            res["message"] = "缺失"
            return res
            
        if _is_temp_file(local_path):
            res["message"] = "下载未完成"
            return res
            
        res["exists"] = True
        local_size = local_path.stat().st_size
        
        if info["size"] > 0 and local_size != info["size"]:
            res["message"] = f"大小不匹配 ({local_size} != {info['size']})"
            return res
            
        # 仅对大于 10MB 的文件进行 Hash 校验，且如果提供了 Hash
        if info["sha256"] and info["size"] > 10 * 1024 * 1024:
            loop = asyncio.get_running_loop()
            file_hash = await loop.run_in_executor(None, _calculate_file_hash, local_path)
            if file_hash != info["sha256"].lower():
                res["message"] = "Hash 校验失败"
                return res
        
        res["valid"] = True
        res["message"] = "完整"
        return res

    def check_model_integrity(self) -> Dict[str, Any]:
        """Sync wrapper for local check (Compat)"""
        return self.validate_local()


# --- Concrete Detectors ---

class ZImageDetector(ModelDetector):
    def get_spec(self) -> ModelSpec:
        return ModelSpec(
            name="Z-Image Turbo",
            # model_id="Tongyi-MAI/Z-Image-Turbo", 
            model_id="Tongyi-MAI/Z-Image-Turbo",
            description="阿里Z-Image Turbo 10步加速模型",
            default_path="zimage_models",
            size_gb=32.0,
            aliases=["zimage", "z-image"]
        )

# --- Registry ---

class ModelDetectorRegistry:
    _detectors = {
        "zimage": ZImageDetector,
    }
    
    @classmethod
    def get_detector(cls, model_type: str, model_path: Union[str, Path]) -> ModelDetector:
        detector_class = cls._detectors.get(model_type.lower(), ZImageDetector)
        return detector_class(model_path)

    @classmethod
    def list_detectors(cls) -> List[str]:
        return list(cls._detectors.keys())

    @classmethod
    def auto_detect(cls, path: Path) -> Optional[str]:
        """简单启发式检测"""
        # 检查 model_index.json 存在即认为是 zimage
        if (path / "model_index.json").exists():
            return "zimage"
        return None

def create_model_detector(model_type: str, model_path: Union[str, Path]) -> ModelDetector:
    return ModelDetectorRegistry.get_detector(model_type, model_path)