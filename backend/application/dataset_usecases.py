# -*- coding: utf-8 -*-
"""
Application Layer - Dataset Use Cases

Orchestrates dataset operations across domain objects.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from urllib.parse import quote

from ..domain.dataset.entities import (
    DatasetStats, ImageInfo, ImageGroup, DatasetChannel,
    BucketInfo, CacheStatus,
)
from ..domain.dataset.repositories import (
    IDatasetRepository, IDatasetCacheRepository, IBucketCalculator,
)

logger = logging.getLogger(__name__)


class ScanDatasetUseCase:
    """Scan a dataset directory for images with pagination."""

    def __init__(self, repo: IDatasetRepository):
        self._repo = repo

    def execute(
        self, path: str, page: int = 1, page_size: int = 50
    ) -> Dict[str, Any]:
        valid, msg = self._repo.validate_path(path)
        if not valid:
            return {"success": False, "message": msg}

        dataset_type = self._repo.detect_type(path)

        # For multi-channel datasets, also return channels
        channels = []
        if dataset_type.value == "multi_channel":
            channels = self._repo.detect_channels(path)

        images, total = self._repo.scan_images(path, page, page_size)

        return {
            "success": True,
            "images": images,
            "total": total,
            "page": page,
            "page_size": page_size,
            "dataset_type": dataset_type.value,
            "channels": [ch.to_dict() for ch in channels],
        }


class ScanMultiChannelUseCase:
    """Scan a multi-channel dataset for ImageGroups."""

    def __init__(self, repo: IDatasetRepository):
        self._repo = repo

    def execute(
        self, path: str, page: int = 1, page_size: int = 50
    ) -> Dict[str, Any]:
        valid, msg = self._repo.validate_path(path)
        if not valid:
            return {"success": False, "message": msg}

        channels = self._repo.detect_channels(path)
        groups, total = self._repo.scan_groups(path, page, page_size)

        # Serialize groups for API response
        serialized_groups = []
        for grp in groups:
            g = {
                "id": grp.id,
                "caption": grp.caption,
                "target": {
                    "filename": grp.target.filename,
                    "path": grp.target.path,
                    "width": grp.target.width,
                    "height": grp.target.height,
                    "size": grp.target.size_bytes,
                    "thumbnailUrl": f"/api/dataset/image?path={quote(grp.target.path, safe='')}",
                } if grp.target else None,
                "channels": {},
            }
            for ch_name, img in grp.channels.items():
                g["channels"][ch_name] = {
                    "filename": img.filename,
                    "path": img.path,
                    "width": img.width,
                    "height": img.height,
                    "size": img.size_bytes,
                    "thumbnailUrl": f"/api/dataset/image?path={quote(img.path, safe='')}",
                }
            serialized_groups.append(g)

        return {
            "success": True,
            "groups": serialized_groups,
            "channels": [ch.to_dict() for ch in channels],
            "total": total,
            "page": page,
            "page_size": page_size,
        }


class GetDatasetStatsUseCase:
    """Get dataset statistics including cache coverage."""

    def __init__(
        self,
        repo: IDatasetRepository,
        cache_repo: Optional[IDatasetCacheRepository] = None,
    ):
        self._repo = repo
        self._cache_repo = cache_repo

    def execute(self, path: str) -> DatasetStats:
        return self._repo.get_stats(path)


class CalculateBucketsUseCase:
    """Calculate aspect ratio buckets for a dataset."""

    def __init__(self, calculator: IBucketCalculator):
        self._calculator = calculator

    def execute(
        self, path: str, batch_size: int = 4, resolution_limit: int = 1536
    ) -> List[BucketInfo]:
        return self._calculator.calculate_buckets(path, batch_size, resolution_limit)


