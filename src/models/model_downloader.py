# -*- coding: utf-8 -*-
"""
增强的模型下载抽象层 - 支持多模型下载管理

提供统一的模型下载接口，支持不同模型的下载策略和进度管理。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Union
from pathlib import Path
import logging
import os
import threading
import time
from enum import Enum

logger = logging.getLogger(__name__)


class DownloadStatus(Enum):
    """下载状态枚举"""
    PENDING = "pending"          # 等待中
    DOWNLOADING = "downloading"  # 下载中
    PAUSED = "paused"           # 暂停
    COMPLETED = "completed"     # 完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"      # 取消


@dataclass
class DownloadProgress:
    """下载进度信息"""
    status: DownloadStatus
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0      # 下载速度(bytes/second)
    eta_seconds: Optional[float] = None  # 预计剩余时间
    error_message: Optional[str] = None
    current_file: Optional[str] = None   # 当前下载的文件
    
    @property
    def progress_percent(self) -> float:
        """获取下载进度百分比"""
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, (self.downloaded_bytes / self.total_bytes) * 100)
    
    @property
    def downloaded_mb(self) -> float:
        """已下载大小(MB)"""
        return self.downloaded_bytes / (1024 * 1024)
    
    @property
    def total_mb(self) -> float:
        """总大小(MB)"""
        return self.total_bytes / (1024 * 1024)
    
    @property
    def speed_mbps(self) -> float:
        """下载速度(MB/s)"""
        return self.speed_bps / (1024 * 1024)


class ModelDownloader(ABC):
    """模型下载器抽象基类"""
    
    def __init__(self, model_id: str, local_path: Union[str, Path]):
        self.model_id = model_id
        self.local_path = Path(local_path)
        self.progress = DownloadProgress(DownloadStatus.PENDING)
        self._callbacks: List[Callable[[DownloadProgress], None]] = []
        self._stop_event = threading.Event()
        self._download_thread: Optional[threading.Thread] = None
    
    @abstractmethod
    def get_model_files(self) -> List[Dict[str, Any]]:
        """获取模型文件列表
        
        Returns:
            文件列表，每个文件包含:
            - path: 相对路径
            - size: 文件大小(bytes)
            - url: 下载URL
            - hash: 文件哈希(可选)
        """
        pass
    
    @abstractmethod
    def download_file(self, file_info: Dict[str, Any], target_path: Path) -> bool:
        """下载单个文件
        
        Args:
            file_info: 文件信息
            target_path: 目标路径
            
        Returns:
            下载是否成功
        """
        pass
    
    def add_progress_callback(self, callback: Callable[[DownloadProgress], None]):
        """添加进度回调"""
        self._callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[DownloadProgress], None]):
        """移除进度回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify_progress(self):
        """通知进度更新"""
        for callback in self._callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                logger.warning(f"进度回调出错: {e}")
    
    def _update_progress(self, **kwargs):
        """更新进度信息"""
        for key, value in kwargs.items():
            if hasattr(self.progress, key):
                setattr(self.progress, key, value)
        self._notify_progress()
    
    def start_download(self) -> bool:
        """开始下载"""
        if self._download_thread and self._download_thread.is_alive():
            logger.warning("下载已在进行中")
            return False
        
        self._stop_event.clear()
        self._download_thread = threading.Thread(target=self._download_worker)
        self._download_thread.start()
        return True
    
    def pause_download(self):
        """暂停下载"""
        if self.progress.status == DownloadStatus.DOWNLOADING:
            self._update_progress(status=DownloadStatus.PAUSED)
    
    def resume_download(self):
        """恢复下载"""
        if self.progress.status == DownloadStatus.PAUSED:
            self.start_download()
    
    def cancel_download(self):
        """取消下载"""
        self._stop_event.set()
        self._update_progress(status=DownloadStatus.CANCELLED)
    
    def _download_worker(self):
        """下载工作线程"""
        try:
            self._update_progress(status=DownloadStatus.DOWNLOADING)
            
            # 获取文件列表
            files = self.get_model_files()
            if not files:
                raise Exception("无法获取模型文件列表")
            
            total_size = sum(file.get("size", 0) for file in files)
            downloaded_size = 0
            start_time = time.time()
            
            self._update_progress(
                total_bytes=total_size,
                downloaded_bytes=downloaded_size
            )
            
            # 创建本地目录
            self.local_path.mkdir(parents=True, exist_ok=True)
            
            # 下载每个文件
            for i, file_info in enumerate(files):
                if self._stop_event.is_set():
                    self._update_progress(status=DownloadStatus.CANCELLED)
                    return
                
                file_path = Path(file_info["path"])
                target_path = self.local_path / file_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 跳过已存在的文件
                if target_path.exists() and target_path.stat().st_size == file_info.get("size", 0):
                    downloaded_size += file_info.get("size", 0)
                    continue
                
                self._update_progress(current_file=str(file_path))
                
                # 下载文件
                success = self.download_file(file_info, target_path)
                if not success:
                    raise Exception(f"下载失败: {file_path}")
                
                downloaded_size += file_info.get("size", 0)
                
                # 更新进度
                elapsed = time.time() - start_time
                if elapsed > 0:
                    speed = downloaded_size / elapsed
                    eta = (total_size - downloaded_size) / speed if speed > 0 else None
                else:
                    speed = 0
                    eta = None
                
                self._update_progress(
                    downloaded_bytes=downloaded_size,
                    speed_bps=speed,
                    eta_seconds=eta
                )
            
            # 下载完成
            self._update_progress(
                status=DownloadStatus.COMPLETED,
                current_file=None
            )
            
        except Exception as e:
            logger.error(f"下载出错: {e}")
            self._update_progress(
                status=DownloadStatus.FAILED,
                error_message=str(e)
            )
    
    def get_progress(self) -> DownloadProgress:
        """获取当前进度"""
        return self.progress


class ModelScopeDownloader(ModelDownloader):
    """ModelScope 模型下载器"""
    
    def __init__(self, model_id: str, local_path: Union[str, Path], revision: str = "master"):
        super().__init__(model_id, local_path)
        self.revision = revision
        self._api = None
    
    def _get_api(self):
        """获取ModelScope API实例"""
        if not self._api:
            try:
                from modelscope.hub.api import HubApi
                self._api = HubApi()
            except ImportError:
                raise ImportError("需要安装modelscope: pip install modelscope")
        return self._api
    
    def get_model_files(self) -> List[Dict[str, Any]]:
        """获取ModelScope模型文件列表"""
        try:
            api = self._get_api()
            files = api.get_model_files(self.model_id, revision=self.revision)
            
            file_list = []
            for file_info in files:
                if isinstance(file_info, dict):
                    file_list.append({
                        "path": file_info.get("Path", ""),
                        "size": file_info.get("Size", 0),
                        "url": file_info.get("DownloadUrl", ""),
                        "hash": file_info.get("Sha256", ""),
                        "type": "file"
                    })
            
            return file_list
            
        except Exception as e:
            logger.error(f"获取模型文件列表失败: {e}")
            return []
    
    def download_file(self, file_info: Dict[str, Any], target_path: Path) -> bool:
        """从ModelScope下载单个文件"""
        try:
            from modelscope.hub.file_download import model_file_download
            
            file_path = file_info.get("path", "")
            if not file_path:
                return False
            
            # 下载文件
            downloaded_path = model_file_download(
                model_id=self.model_id,
                file_path=file_path,
                local_dir=str(target_path.parent),
                revision=self.revision
            )
            
            # 重命名到目标路径
            if Path(downloaded_path).exists():
                Path(downloaded_path).rename(target_path)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"下载文件失败 {file_info.get('path', '')}: {e}")
            return False


class ModelDownloadManager:
    """模型下载管理器"""
    
    def __init__(self):
        self._downloaders: Dict[str, ModelDownloader] = {}
        self._model_specs = {
            "zimage": {
                "name": "Z-Image Turbo",
                "model_id": "Tongyi-MAI/Z-Image-Turbo",
                "size_gb": 32.0,
                "description": "阿里Z-Image Turbo 10步加速模型"
            }
        }
    
    def create_downloader(self, model_type: str, local_path: Union[str, Path], 
                         downloader_type: str = "modelscope") -> ModelDownloader:
        """创建模型下载器"""
        
        if model_type not in self._model_specs:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        spec = self._model_specs[model_type]
        model_id = spec["model_id"]
        
        if downloader_type == "modelscope":
            downloader = ModelScopeDownloader(model_id, local_path)
        else:
            raise ValueError(f"不支持的下载器类型: {downloader_type}")
        
        # 存储下载器实例
        key = f"{model_type}:{local_path}"
        self._downloaders[key] = downloader
        
        return downloader
    
    def get_downloader(self, model_type: str, local_path: Union[str, Path]) -> Optional[ModelDownloader]:
        """获取已创建的下载器"""
        key = f"{model_type}:{local_path}"
        return self._downloaders.get(key)
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """获取模型信息"""
        return self._model_specs.get(model_type, {})
    
    def list_supported_models(self) -> List[str]:
        """列出支持的模型类型"""
        return list(self._model_specs.keys())


# 全局下载管理器实例
download_manager = ModelDownloadManager()


def create_model_downloader(model_type: str, local_path: Union[str, Path], 
                           downloader_type: str = "modelscope") -> ModelDownloader:
    """工厂函数：创建模型下载器"""
    return download_manager.create_downloader(model_type, local_path, downloader_type)


def get_model_download_info(model_type: str) -> Dict[str, Any]:
    """获取模型下载信息"""
    return download_manager.get_model_info(model_type)