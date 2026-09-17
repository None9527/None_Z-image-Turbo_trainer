# -*- coding: utf-8 -*-
"""
Application Layer - System Use Cases

Orchestrates system monitoring and model management.
"""

import logging
from typing import List, Dict, Any, Optional

from ..domain.system.entities import (
    GPUInfo, SystemInfo, ModelInfo, DownloadProgress,
)
from ..domain.system.repositories import (
    IGPUMonitor, ISystemInfoProvider, IModelManager,
)

logger = logging.getLogger(__name__)


class GetSystemStatusUseCase:
    """Get combined system status (GPU + system info)."""

    def __init__(
        self,
        gpu_monitor: IGPUMonitor,
        sys_info: ISystemInfoProvider,
    ):
        self._gpu_monitor = gpu_monitor
        self._sys_info = sys_info

    def execute(self) -> Dict[str, Any]:
        gpus = self._gpu_monitor.get_gpu_info()
        info = self._sys_info.get_system_info()
        info.gpus = gpus

        return {
            "os": info.os,
            "python_version": info.python_version,
            "torch_version": info.torch_version,
            "cuda_version": info.cuda_version,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "memory_total": g.memory_total_gb,
                    "memory_used": g.memory_used_gb,
                    "memory_percent": round(g.memory_percent, 1),
                    "temperature": g.temperature,
                    "utilization": g.utilization,
                }
                for g in gpus
            ],
        }


class GetModelStatusUseCase:
    """Check local model status."""

    def __init__(self, model_mgr: IModelManager):
        self._model_mgr = model_mgr

    def execute(self, model_type: str = "zimage") -> ModelInfo:
        return self._model_mgr.get_model_status(model_type)


