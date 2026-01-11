from fastapi import APIRouter, HTTPException, Form, UploadFile, File, BackgroundTasks
from pathlib import Path
from PIL import Image
import re
import shutil
import base64
import io
import requests
import asyncio
import threading
import urllib.parse
from typing import Optional
from pydantic import BaseModel

from core.config import DATASETS_DIR, DatasetScanRequest
from core import state

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

# 多模型缓存后缀配置 (仅 Z-Image)
CACHE_SUFFIXES = {
    "zimage": {"latent": "_zi.safetensors", "text": "_zi_te.safetensors", "latent_pattern": "*_zi.safetensors"},
}
ALL_LATENT_PATTERNS = ["*_zi.safetensors"]
ALL_TEXT_SUFFIXES = ["_zi_te.safetensors"]

# 文件列表缓存（避免每次翻页都重新遍历）
# 结构: {path: (files, total_size, timestamp)}
_dataset_cache: dict[str, tuple[list, int, float]] = {}
_cache_ttl = 60  # 缓存有效期（秒）

# 图片尺寸缓存（避免每次都读取图片头信息）
# 结构: {dataset_path: {filename: (width, height, mtime)}}
_dimension_cache: dict[str, dict] = {}

def _load_dimension_cache(dataset_path: Path) -> dict:
    """从 JSON 文件加载尺寸缓存"""
    cache_file = dataset_path / ".sizes_cache.json"
    if cache_file.exists():
        try:
            import json
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def _save_dimension_cache(dataset_path: Path, cache: dict):
    """保存尺寸缓存到 JSON 文件"""
    cache_file = dataset_path / ".sizes_cache.json"
    try:
        import json
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
    except:
        pass

def _get_image_dimensions(file: Path, dim_cache: dict) -> tuple[int, int]:
    """获取图片尺寸，优先使用缓存"""
    filename = file.name
    try:
        mtime = file.stat().st_mtime
    except:
        mtime = 0
    
    # 检查缓存是否有效
    if filename in dim_cache:
        cached = dim_cache[filename]
        if len(cached) >= 3 and cached[2] == mtime:
            return cached[0], cached[1]
    
    # 读取图片尺寸（只读取头信息，不加载整个图片）
    try:
        with Image.open(file) as img:
            w, h = img.size
            dim_cache[filename] = [w, h, mtime]
            return w, h
    except Exception as e:
        print(f"[Dataset] Failed to get dimensions for {filename}: {e}")
        return 0, 0

def _get_cached_files_and_size(path: Path) -> tuple[list, int]:
    """获取缓存的文件列表和总大小，过期则重新扫描"""
    import time
    path_str = str(path)
    now = time.time()
    
    if path_str in _dataset_cache:
        files, total_size, ts = _dataset_cache[path_str]
        if now - ts < _cache_ttl:
            return files, total_size
    
    # 重新扫描并计算总大小
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    all_files = []
    total_size = 0
    for file in path.rglob('*'):
        if file.is_file() and file.suffix.lower() in image_extensions:
            all_files.append(file)
            try:
                total_size += file.stat().st_size
            except:
                pass
    all_files.sort(key=lambda f: f.name.lower())
    
    _dataset_cache[path_str] = (all_files, total_size, now)
    return all_files, total_size


def _invalidate_dataset_cache(path: Path):
    """主动清除数据集缓存（上传/删除/创建后调用）"""
    path_str = str(path)
    if path_str in _dataset_cache:
        del _dataset_cache[path_str]
        print(f"[Cache] Invalidated cache for: {path}")
    # 同时清除父目录缓存（如果有子目录结构）
    parent_str = str(path.parent)
    if parent_str in _dataset_cache:
        del _dataset_cache[parent_str]


@router.post("/scan")
async def scan_dataset(request: DatasetScanRequest):
    """Scan a directory for images (带缓存的极速模式)
    
    优化策略：
    1. 文件列表缓存（翻页不重新遍历）
    2. asyncio.to_thread 不阻塞事件循环
    3. 恢复标注和文件大小显示
    """
    import asyncio
    
    path = Path(request.path)
    page = max(1, request.page)
    page_size = min(500, max(10, request.page_size))
    
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {request.path}")
    
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.path}")
    
    def _scan_files_sync():
        """同步文件扫描（在线程池中执行）"""
        # 使用缓存获取文件列表和总大小（翻页不重新遍历）
        all_files, cached_total_size = _get_cached_files_and_size(path)
        
        total_count = len(all_files)
        total_pages = (total_count + page_size - 1) // page_size
        
        # 分页
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_count)
        page_files = all_files[start_idx:end_idx]
        
        # 加载尺寸缓存
        dim_cache = _load_dimension_cache(path)
        cache_modified = False
        
        # 构建响应（恢复标注和缓存状态）
        images = []
        for file in page_files:
            try:
                stat = file.stat()
                size = stat.st_size
                
                # 读取标注文件
                caption = None
                caption_file = file.with_suffix('.txt')
                if caption_file.exists():
                    try:
                        caption = caption_file.read_text(encoding='utf-8').strip()
                    except:
                        pass
                
                # 检查缓存状态（只检查当前页 100 张，可接受的延迟）
                has_latent_cache = any(
                    list(file.parent.glob(f"{file.stem}_*{suffix}"))[:1]
                    for suffix in ["_zi.safetensors", "_lc.safetensors"]
                )
                has_text_cache = any(
                    (file.parent / f"{file.stem}{suffix}").exists()
                    for suffix in ALL_TEXT_SUFFIXES
                )
                
                # 获取图片尺寸（使用缓存系统）
                old_cache_size = len(dim_cache)
                img_width, img_height = _get_image_dimensions(file, dim_cache)
                if len(dim_cache) > old_cache_size:
                    cache_modified = True
                
                encoded_path = urllib.parse.quote(str(file), safe='')
                images.append({
                    "path": str(file),
                    "filename": file.name,
                    "width": img_width,
                    "height": img_height,
                    "size": size,
                    "caption": caption,
                    "hasLatentCache": has_latent_cache,
                    "hasTextCache": has_text_cache,
                    "thumbnailUrl": f"/api/dataset/thumbnail?path={encoded_path}"
                })
            except Exception:
                continue
        
        # 保存尺寸缓存（仅当有新增时）
        if cache_modified:
            _save_dimension_cache(path, dim_cache)
        
        return {
            "path": str(path),
            "name": path.name,
            "imageCount": total_count,
            "totalSize": cached_total_size,  # 使用缓存的总大小
            "images": images,
            "totalLatentCached": None,
            "totalTextCached": None,
            "pagination": {
                "page": page,
                "pageSize": page_size,
                "totalPages": total_pages,
                "totalCount": total_count,
                "hasNext": page < total_pages,
                "hasPrev": page > 1
            }
        }
    
    # 在线程池中执行，不阻塞事件循环
    return await asyncio.to_thread(_scan_files_sync)


@router.post("/stats")
async def get_dataset_stats(request: DatasetScanRequest):
    """快速获取数据集缓存统计
    
    使用更聪明的方法：直接 glob 扫描缓存文件数量，
    而不是对每个图片检查缓存是否存在。
    从 4000 次 glob 变为 2 次 glob，性能提升 1000 倍。
    """
    import asyncio
    
    path = Path(request.path)
    
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {request.path}")
    
    def _compute_stats_fast():
        """快速统计（直接扫描缓存文件数量）"""
        # 使用缓存获取图片总数
        all_files, _ = _get_cached_files_and_size(path)
        total_count = len(all_files)
        
        # 直接 glob 扫描 latent 缓存文件数量（只需 2 次 glob）
        latent_zi_files = set()
        latent_lc_files = set()
        
        # Z-Image latent 缓存: xxx_0512x0512_zi.safetensors 或 xxx_zi.safetensors
        for f in path.rglob("*_zi.safetensors"):
            # 提取原始图片名
            stem = f.stem  # xxx_0512x0512_zi
            parts = stem.rsplit('_', 2)
            if len(parts) >= 3:
                # 标准格式: name_RES_suffix
                original_name = parts[0]
                latent_zi_files.add(original_name)
            elif len(parts) == 2:
                # 兼容格式: name_suffix
                latent_zi_files.add(parts[0])
        
        # latent 缓存数 = ZI 缓存存在的图片数
        latent_cached_names = latent_zi_files
        print(f"[Stats] Latent cached: {len(latent_cached_names)} (ZI: {len(latent_zi_files)})")
        
        # 直接 glob 扫描 text 缓存文件数量
        text_zi_files = set()
        text_lc_files = set()
        
        for f in path.rglob("*_zi_te.safetensors"):
            # xxx_zi_te.safetensors -> xxx
            stem = f.stem  # xxx_zi_te
            if stem.endswith("_zi_te"):
                original_name = stem[:-6]
                text_zi_files.add(original_name)
        
        text_cached_names = text_zi_files
        print(f"[Stats] Text cached: {len(text_cached_names)} (ZI: {len(text_zi_files)})")
        
        return {
            "totalCount": total_count,
            "totalLatentCached": len(latent_cached_names),
            "totalTextCached": len(text_cached_names)
        }
    
    # 在线程池中执行，不阻塞事件循环
    return await asyncio.to_thread(_compute_stats_fast)

@router.get("/list")
async def list_datasets():
    """List all datasets in the datasets folder"""
    print(f"[Dataset] Scanning directory: {DATASETS_DIR}")
    print(f"[Dataset] Directory exists: {DATASETS_DIR.exists()}")
    
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = []
    for folder in DATASETS_DIR.iterdir():
        if folder.is_dir():
            # Count images
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
            image_count = sum(1 for f in folder.rglob('*') if f.suffix.lower() in image_extensions)
            
            datasets.append({
                "name": folder.name,
                "path": str(folder),
                "imageCount": image_count
            })
    
    print(f"[Dataset] Found {len(datasets)} datasets")
    return {
        "datasets": datasets, 
        "datasetsDir": str(DATASETS_DIR),
        "datasetsDirExists": DATASETS_DIR.exists()
    }

@router.post("/create")
async def create_dataset(name: str = Form(...)):
    """Create a new dataset folder"""
    # Sanitize name
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', name.strip())
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    
    dataset_path = DATASETS_DIR / safe_name
    if dataset_path.exists():
        raise HTTPException(status_code=400, detail="Dataset already exists")
    
    try:
        dataset_path.mkdir(parents=True, exist_ok=True)
        # 清除父目录缓存（新建数据集后列表需要更新）
        _invalidate_dataset_cache(DATASETS_DIR)
        return {"success": True, "path": str(dataset_path), "name": safe_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")

@router.post("/upload_batch")
async def upload_batch(
    dataset_name: str = Form(...),
    files: list[UploadFile] = File(...)
):
    """Upload multiple files to a dataset"""
    print(f"DEBUG: Received upload request for dataset '{dataset_name}' with {len(files)} files")
    
    # Sanitize dataset name
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', dataset_name.strip())
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
        
    dataset_path = DATASETS_DIR / safe_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    print(f"DEBUG: Created/Checked dataset directory: {dataset_path}")
    
    uploaded_count = 0
    errors = []
    
    for file in files:
        try:
            print(f"DEBUG: Processing file: {file.filename}, Type: {file.content_type}")
            # Skip non-image files (optional, but good practice)
            if not file.content_type.startswith('image/') and not file.filename.endswith('.txt') and not file.filename.endswith('.safetensors'):
                print(f"DEBUG: Skipping file {file.filename} due to type mismatch")
                continue
                
            file_path = dataset_path / file.filename
            
            # Ensure parent directory exists (for nested files)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"DEBUG: Saving to {file_path}")
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            uploaded_count += 1
        except Exception as e:
            print(f"ERROR: Failed to save {file.filename}: {e}")
            import traceback
            traceback.print_exc()
            errors.append(f"{file.filename}: {str(e)}")
            
    # 清除缓存以确保下次扫描获取最新数据
    _invalidate_dataset_cache(dataset_path)
    
    return {
        "success": True, 
        "uploaded": uploaded_count, 
        "dataset": safe_name,
        "errors": errors
    }

@router.post("/upload")
async def upload_files(
    dataset: str = Form(...),
    files: list[UploadFile] = File(...)
):
    """Upload files to an existing dataset"""
    print(f"DEBUG: Upload to dataset '{dataset}' with {len(files)} files")
    
    # 查找数据集路径
    dataset_path = DATASETS_DIR / dataset
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")
    
    uploaded = []
    errors = []
    
    for file in files:
        try:
            # 检查文件类型
            is_image = file.content_type and file.content_type.startswith('image/')
            is_txt = file.filename and file.filename.endswith('.txt')
            is_safetensors = file.filename and file.filename.endswith('.safetensors')
            
            if not (is_image or is_txt or is_safetensors):
                errors.append(f"{file.filename}: 不支持的文件类型")
                continue
            
            # 保存文件
            file_path = dataset_path / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded.append(file.filename)
            print(f"DEBUG: Saved {file.filename}")
            
        except Exception as e:
            print(f"ERROR: Failed to save {file.filename}: {e}")
            errors.append(f"{file.filename}: {str(e)}")
    
    # 清除缓存以确保下次扫描获取最新数据
    _invalidate_dataset_cache(dataset_path)
    
    return {
        "success": True,
        "uploaded": uploaded,
        "datasetPath": str(dataset_path),
        "errors": errors
    }

@router.get("/thumbnail")
async def get_thumbnail(path: str):
    """Get a thumbnail of an image with improved error handling and caching"""
    from fastapi.responses import Response
    import io
    import urllib.parse
    
    # URL 解码路径
    decoded_path = urllib.parse.unquote(path)
    img_path = Path(decoded_path)
    
    if not img_path.exists():
        print(f"[Thumbnail] File not found: {img_path}")
        raise HTTPException(status_code=404, detail=f"Image not found: {decoded_path}")
    
    try:
        with Image.open(img_path) as img:
            # 转换为 RGB 模式（处理 RGBA、P 等模式）
            if img.mode in ('RGBA', 'LA', 'P'):
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 生成缩略图
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            
            # 添加缓存头，缓存 1 小时
            return Response(
                content=buffer.read(), 
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "X-Content-Type-Options": "nosniff"
                }
            )
    except Exception as e:
        print(f"[Thumbnail] Error processing {img_path}: {e}")
        import traceback
        traceback.print_exc()
        # 返回一个占位图而不是报错
        try:
            # 生成一个简单的灰色占位图
            placeholder = Image.new('RGB', (200, 200), (64, 64, 64))
            buffer = io.BytesIO()
            placeholder.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            return Response(
                content=buffer.read(), 
                media_type="image/jpeg",
                headers={"X-Thumbnail-Error": "true"}
            )
        except:
            raise HTTPException(status_code=500, detail=str(e))

@router.get("/image")
async def get_image(path: str):
    """Get full size image"""
    from fastapi.responses import FileResponse
    
    img_path = Path(path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    if not img_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
        
    return FileResponse(str(img_path))

@router.get("/cached")
async def list_cached_datasets():
    """List all datasets available for training (both raw and cached)"""
    try:
        datasets = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        if DATASETS_DIR.exists():
            # 递归扫描所有包含图片的目录
            def scan_directory(dir_path: Path, prefix: str = ""):
                for item in dir_path.iterdir():
                    if item.is_dir():
                        # 检查是否包含图片
                        image_count = sum(
                            1 for f in item.iterdir() 
                            if f.is_file() and f.suffix.lower() in image_extensions
                        )
                        
                        # 检查是否有缓存文件 (支持多模型)
                        cache_count = sum(
                            len(list(item.glob(pattern)))
                            for pattern in ALL_LATENT_PATTERNS
                        )
                        
                        if image_count > 0 or cache_count > 0:
                            name = f"{prefix}{item.name}" if prefix else item.name
                            file_count = cache_count if cache_count > 0 else image_count
                            status = "已缓存" if cache_count > 0 else "图片"
                            
                            datasets.append({
                                "name": f"{name} ({file_count} {status})",
                                "path": str(item.absolute()),
                                "files": file_count,
                                "cached": cache_count > 0
                            })
                        
                        # 继续扫描子目录
                        scan_directory(item, f"{prefix}{item.name}/")
            
            scan_directory(DATASETS_DIR)
        
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")

@router.delete("/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset folder"""
    # Sanitize name to prevent directory traversal
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', dataset_name.strip())
    if not safe_name or safe_name == "." or safe_name == "..":
        raise HTTPException(status_code=400, detail="Invalid dataset name")
        
    dataset_path = DATASETS_DIR / safe_name
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    try:
        shutil.rmtree(dataset_path)
        # 清除缓存
        _invalidate_dataset_cache(dataset_path)
        _invalidate_dataset_cache(DATASETS_DIR)
        return {"success": True, "message": f"Dataset '{dataset_name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

class DeleteImagesRequest(BaseModel):
    paths: list[str]

@router.post("/delete-images")
async def delete_images(request: DeleteImagesRequest):
    """Delete specific images from a dataset"""
    deleted_count = 0
    errors = []
    affected_dirs = set()  # 跟踪受影响的目录
    
    for path_str in request.paths:
        try:
            path = Path(path_str)
            
            # Security check: ensure path is within DATASETS_DIR
            # Resolve to absolute path for comparison
            abs_path = path.resolve()
            try:
                abs_path.relative_to(DATASETS_DIR.resolve())
            except ValueError:
                errors.append(f"{path_str}: Access denied (outside datasets directory)")
                continue
                
            if abs_path.exists() and abs_path.is_file():
                # 记录受影响的目录
                affected_dirs.add(abs_path.parent)
                
                abs_path.unlink()
                
                # Also try to delete associated files (.txt, .safetensors cache)
                txt_path = abs_path.with_suffix('.txt')
                if txt_path.exists():
                    txt_path.unlink()
                
                # 删除所有模型类型的缓存文件
                for model_type, suffixes in CACHE_SUFFIXES.items():
                    # 删除 latent 缓存 (pattern: stem_WxH_suffix)
                    for cache_file in abs_path.parent.glob(f"{abs_path.stem}_*{suffixes['latent']}"):
                        cache_file.unlink()
                    
                    # 删除 text encoder 缓存
                    text_cache = abs_path.parent / f"{abs_path.stem}{suffixes['text']}"
                    if text_cache.exists():
                        text_cache.unlink()
                    
                deleted_count += 1
            else:
                errors.append(f"{path_str}: File not found")
                
        except Exception as e:
            errors.append(f"{path_str}: {str(e)}")
    
    # 清除所有受影响目录的缓存
    for dir_path in affected_dirs:
        _invalidate_dataset_cache(dir_path)
            
    return {
        "success": True,
        "deleted": deleted_count,
        "errors": errors
    }


# ============================================================================
# 分桶计算器
# ============================================================================

class BucketCalculateRequest(BaseModel):
    path: str
    batch_size: int = 4
    resolution_limit: int = 1536

@router.post("/calculate-buckets")
async def calculate_buckets(request: BucketCalculateRequest):
    """计算数据集的分桶分布（基于全量数据）"""
    import asyncio
    
    path = Path(request.path)
    batch_size = max(1, min(16, request.batch_size))
    limit = max(256, min(2048, request.resolution_limit))
    
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {request.path}")
    
    def _compute_buckets():
        # 使用缓存获取文件列表
        all_files, _ = _get_cached_files_and_size(path)
        
        # 加载尺寸缓存
        dim_cache = _load_dimension_cache(path)
        cache_modified = False
        
        # 按分辨率分组
        buckets: dict[str, dict] = {}
        
        for file in all_files:
            old_cache_size = len(dim_cache)
            w, h = _get_image_dimensions(file, dim_cache)
            if len(dim_cache) > old_cache_size:
                cache_modified = True
            
            if w == 0 or h == 0:
                continue
            
            # 应用分辨率限制
            if max(w, h) > limit:
                scale = limit / max(w, h)
                w = int(w * scale)
                h = int(h * scale)
            
            # 对齐到 8 的倍数
            w = (w // 8) * 8
            h = (h // 8) * 8
            
            key = f"{w}x{h}"
            if key not in buckets:
                buckets[key] = {"width": w, "height": h, "count": 0}
            buckets[key]["count"] += 1
        
        # 保存尺寸缓存
        if cache_modified:
            _save_dimension_cache(path, dim_cache)
        
        # 计算每个桶的批次数和丢弃数
        results = []
        max_count = max([b["count"] for b in buckets.values()]) if buckets else 1
        
        for key, bucket in buckets.items():
            batches = bucket["count"] // batch_size
            dropped = bucket["count"] % batch_size
            if batches == 0:
                dropped = bucket["count"]  # 没有完整批次，全部丢弃
            
            results.append({
                "width": bucket["width"],
                "height": bucket["height"],
                "aspectRatio": round(bucket["width"] / bucket["height"], 2),
                "count": bucket["count"],
                "batches": batches,
                "dropped": dropped,
                "percentage": round((bucket["count"] / max_count) * 100)
            })
        
        # 按图片数量排序
        results.sort(key=lambda x: x["count"], reverse=True)
        
        # 汇总统计
        total_images = sum(b["count"] for b in results)
        total_batches = sum(b["batches"] for b in results)
        dropped_images = sum(b["dropped"] for b in results)
        
        return {
            "buckets": results,
            "summary": {
                "totalImages": total_images,
                "totalBatches": total_batches,
                "droppedImages": dropped_images,
                "bucketCount": len(results)
            }
        }
    
    return await asyncio.to_thread(_compute_buckets)


# ============================================================================
# Ollama 标注功能
# ============================================================================

class OllamaTagRequest(BaseModel):
    dataset_path: str
    ollama_url: str = "http://localhost:11434"
    model: str = "llava:13b"
    prompt: str = "Describe this image in detail for AI training. Focus on: subject, clothing, pose, expression. Output: comma-separated tags."
    max_long_edge: int = 512  # 长边最大尺寸
    skip_existing: bool = True  # 跳过已有标注
    trigger_word: str = ""  # 触发词，添加到每个标注开头

# 标注状态
tagging_state = {
    "running": False,
    "total": 0,
    "completed": 0,
    "current_file": "",
    "errors": []
}

@router.get("/ollama/models")
async def get_ollama_models(ollama_url: str = "http://localhost:11434"):
    """获取 Ollama 可用模型列表"""
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return {"models": models, "success": True}
    except requests.exceptions.ConnectionError:
        return {"models": [], "success": False, "error": "无法连接到 Ollama 服务"}
    except Exception as e:
        return {"models": [], "success": False, "error": str(e)}

@router.get("/ollama/status")
async def get_tagging_status():
    """获取标注进度（排除 process 对象，它无法 JSON 序列化）"""
    return {
        "running": tagging_state.get("running", False),
        "total": tagging_state.get("total", 0),
        "completed": tagging_state.get("completed", 0),
        "current_file": tagging_state.get("current_file", ""),
        "errors": tagging_state.get("errors", [])
    }

@router.post("/ollama/stop")
async def stop_tagging():
    """停止标注"""
    global tagging_state
    tagging_state["running"] = False
    
    # 终止子进程
    process = tagging_state.get('process')
    if process and process.poll() is None:
        pid = process.pid
        print(f"[Tagging] Stopping process (PID: {pid})...")
        try:
            # 先优雅终止
            process.terminate()
            try:
                process.wait(timeout=5)
            except:
                # 强制杀死
                process.kill()
                process.wait(timeout=3)
            print(f"[Tagging] Process stopped (PID: {pid})")
        except Exception as e:
            print(f"[Tagging] Error stopping process: {e}")
        tagging_state['process'] = None
    
    return {"success": True, "message": "标注已停止"}


def run_tagging_subprocess(dataset_path: str, ollama_url: str, model: str, prompt: str, 
                           max_long_edge: int, trigger_word: str, skip_existing: bool):
    """在子进程中运行标注脚本"""
    global tagging_state
    import subprocess
    import sys
    import re
    import os
    
    # 构建命令
    script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "ollama_tagging.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--input_dir", dataset_path,
        "--ollama_url", ollama_url,
        "--model", model,
        "--prompt", prompt,
        "--max_long_edge", str(max_long_edge),
    ]
    
    if trigger_word:
        cmd.extend(["--trigger_word", trigger_word])
    if skip_existing:
        cmd.append("--skip_existing")
    
    print(f"[Tagging] Starting: {' '.join(cmd[:6])}...")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    try:
        # Windows: CREATE_NO_WINDOW + CREATE_NEW_PROCESS_GROUP 便于终止
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            creationflags = 0
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=env,
            creationflags=creationflags
        )
        
        tagging_state['process'] = process
        
        # 读取输出并解析进度
        progress_pattern = re.compile(r'Progress:\s*(\d+)/(\d+)')
        
        for line in iter(process.stdout.readline, ''):
            if not tagging_state["running"]:
                print("[Tagging] Stop requested, terminating...")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
                break
            
            line = line.strip()
            if not line:
                continue
            
            print(f"[Tagging] {line}", flush=True)
            
            # 解析进度 - Progress: x/y
            match = progress_pattern.search(line)
            if match:
                completed = int(match.group(1))
                total = int(match.group(2))
                tagging_state["completed"] = completed
                tagging_state["total"] = total
                continue
            
            # 解析当前文件 - [FILE] filename.jpg
            if line.startswith("[FILE]"):
                filename = line[6:].strip()
                tagging_state["current_file"] = filename
                continue
            
            # 解析成功 - [OK] filename.jpg  
            if line.startswith("[OK]"):
                continue
            
            # 解析失败 - [FAIL] filename.jpg
            if line.startswith("[FAIL]"):
                filename = line[6:].strip()
                tagging_state["errors"].append(filename)
                continue
        
        process.wait()
        
    except Exception as e:
        print(f"[Tagging] Error: {e}")
        tagging_state["errors"].append(str(e))
    finally:
        tagging_state["running"] = False
        tagging_state["current_file"] = ""
        tagging_state['process'] = None


@router.post("/ollama/tag")
async def start_tagging(request: OllamaTagRequest):
    """开始批量标注（使用独立脚本）"""
    global tagging_state
    
    if tagging_state["running"]:
        raise HTTPException(status_code=400, detail="标注任务正在进行中")
    
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集路径不存在")
    
    # 快速统计待标注图片数量
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    image_count = 0
    for ext in image_extensions:
        for img_path in dataset_path.rglob(f"*{ext}"):
            if request.skip_existing:
                txt_path = img_path.with_suffix('.txt')
                if txt_path.exists() and txt_path.stat().st_size > 0:
                    continue
            image_count += 1
        for img_path in dataset_path.rglob(f"*{ext.upper()}"):
            if request.skip_existing:
                txt_path = img_path.with_suffix('.txt')
                if txt_path.exists() and txt_path.stat().st_size > 0:
                    continue
            image_count += 1
    
    if image_count == 0:
        return {"success": True, "message": "没有需要标注的图片", "total": 0}
    
    # 初始化状态
    tagging_state = {
        "running": True,
        "total": image_count,
        "completed": 0,
        "current_file": "",
        "errors": [],
        "process": None
    }
    
    # 启动后台线程运行子进程
    thread = threading.Thread(
        target=run_tagging_subprocess,
        args=(
            str(dataset_path),
            request.ollama_url,
            request.model,
            request.prompt,
            request.max_long_edge,
            request.trigger_word,
            request.skip_existing
        ),
        daemon=True
    )
    thread.start()
    
    return {
        "success": True,
        "message": f"开始标注 {image_count} 张图片",
        "total": image_count
    }


# ============================================================================
# 以下为旧版内联实现（已弃用，保留供参考）
# ============================================================================

def resize_image_for_api(img_path: Path, max_long_edge: int) -> bytes:
    """缩放图片长边到指定尺寸，返回 JPEG 字节（参照用户脚本使用 getvalue）"""
    with Image.open(img_path) as img:
        # 打印原始图片信息用于调试
        print(f"[Ollama] Opening image: {img_path}")
        print(f"[Ollama] Original size: {img.size}, mode: {img.mode}")
        
        # 转换为 RGB
        if img.mode in ('RGBA', 'P', 'LA', 'L'):
            img = img.convert('RGB')
        
        w, h = img.size
        long_edge = max(w, h)
        
        if long_edge > max_long_edge:
            ratio = max_long_edge / long_edge
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.BICUBIC)
            print(f"[Ollama] Resized to: {new_size}")
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        # 使用 getvalue() 而非 read()，确保获取完整数据
        jpeg_bytes = buf.getvalue()
        print(f"[Ollama] JPEG bytes length: {len(jpeg_bytes)}")
        return jpeg_bytes

def generate_caption_ollama_sync(img_path: Path, ollama_url: str, model: str, prompt: str, max_long_edge: int, enable_think: bool = False) -> Optional[str]:
    """同步调用 Ollama API - 每次独立调用，不保留上下文
    
    Args:
        enable_think: 是否启用思考模式（某些模型如 deepseek、qwen3 支持）
                     如果模型不支持思考模式，建议设为 False
    """
    try:
        jpeg_bytes = resize_image_for_api(img_path, max_long_edge)
        base64_img = base64.b64encode(jpeg_bytes).decode()
        
        print(f"[Ollama] Processing: {img_path.name}")
        print(f"[Ollama] Image bytes: {len(jpeg_bytes)}, Base64 len: {len(base64_img)}")
        print(f"[Ollama] Model: {model}, URL: {ollama_url}, Think: {enable_think}")
        
        # 关键：context=[] 清除对话历史，确保每张图片独立处理
        # 对于 qwen3-vl，不启用思考时在 prompt 前加 /no_think
        actual_prompt = prompt
        if not enable_think and ('qwen3' in model.lower() or 'qwen-3' in model.lower()):
            actual_prompt = f"/no_think {prompt}"
            print(f"[Ollama] Added /no_think prefix for qwen3 model")
        
        payload = {
            "model": model,
            "prompt": actual_prompt,
            "images": [base64_img],
            "stream": False,
            "context": [],  # 清除上下文，避免累积图片
            "temperature": 0.7,
            "top_p": 0.9,
            "options": {
                "num_predict": 1024,  # 限制输出长度
            }
        }
        
        # 添加思考模式控制（某些模型如 deepseek、qwen3 支持）
        # 注意：不是所有模型都支持这个参数，不支持的模型会忽略它
        if not enable_think:
            # 尝试多种方式关闭思考模式
            payload["options"]["think"] = False  # Ollama 标准方式
            payload["think"] = False  # 备用方式
        
        print(f"[Ollama] Sending request to {ollama_url}/api/generate")
        resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=180)
        print(f"[Ollama] Response status: {resp.status_code}")
        resp.raise_for_status()
        
        data = resp.json()
        print(f"[Ollama] Response keys: {data.keys()}")
        
        caption = ""
        has_thinking = "thinking" in data and data["thinking"]
        
        # 检查 response 字段
        if "response" in data:
            caption = data["response"].strip()
            print(f"[Ollama] Raw response ({len(caption)} chars): {caption[:200] if caption else '(empty)'}...")
        
        # 如果 response 为空但有 thinking 字段
        if not caption and has_thinking:
            thinking_text = data["thinking"]
            print(f"[Ollama] Thinking content ({len(thinking_text)} chars)")
            
            if enable_think:
                # 用户开启了思考模式，模型输出了思考但没有最终答案
                # 这种情况下，思考内容可能就是结果（某些模型如此）
                # 尝试清理思考内容作为 caption
                import re
                # 移除常见的思考前缀
                cleaned = thinking_text.strip()
                # 移除思考过程的开头语
                cleaned = re.sub(r'^(好的|让我|我来|首先|我需要|接下来|然后)[，,。.：:\s]*', '', cleaned)
                # 取最后一段有意义的内容（通常是结论）
                paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
                if paragraphs:
                    # 取最后一段，如果太短就多取几段
                    last_para = paragraphs[-1]
                    if len(last_para) < 50 and len(paragraphs) > 1:
                        caption = '\n\n'.join(paragraphs[-2:])
                    else:
                        caption = last_para
                    print(f"[Ollama] Extracted from thinking: {caption[:100]}...")
                else:
                    print(f"[Ollama] Warning: Could not extract useful content from thinking")
                    return None
            else:
                # 用户关闭了思考模式，但模型仍然返回了 thinking
                print(f"[Ollama] Warning: Model requires thinking mode!")
                print(f"[Ollama] Tip: 此模型需要开启「思考模式」才能正常工作，请在前端开启")
                return None
        
        # 如果开启了思考但模型没返回 thinking 字段，可能模型不支持
        if enable_think and not has_thinking and not caption:
            print(f"[Ollama] Warning: Model may not support thinking mode")
            print(f"[Ollama] Tip: 此模型可能不支持思考模式，请尝试关闭「思考模式」")
            return None
        
        if caption:
            # 清理输出
            caption = caption.replace("```markdown", "").replace("```", "")
            import re
            # 移除思考标签内容（<think>...</think> 或 <thinking>...</thinking>）
            caption = re.sub(r'<think>.*?</think>', '', caption, flags=re.DOTALL)
            caption = re.sub(r'<thinking>.*?</thinking>', '', caption, flags=re.DOTALL)
            # 清理可能的 [img-X] 残留标记
            caption = re.sub(r'\[img-\d+\]', '', caption)
            # 清理多余的空行
            caption = re.sub(r'\n{3,}', '\n\n', caption)
            caption = caption.strip()
            
            if not caption:
                print(f"[Ollama] Warning: Empty caption after cleaning for {img_path.name}")
                print(f"[Ollama] Tip: If using qwen3-vl, try adding '/no_think' to prompt or use a different model")
                return None
            
            print(f"[Ollama] Final Caption: {caption[:100]}...")
            return caption
        else:
            print(f"[Ollama] No usable response for {img_path.name}")
            print(f"[Ollama] Full data: {data}")
        return None
    except requests.exceptions.Timeout:
        print(f"[Ollama] Timeout for {img_path.name} - try a smaller image or faster model")
        return None
    except Exception as e:
        print(f"[Ollama] Error for {img_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


class ResizeImagesRequest(BaseModel):
    dataset_path: str
    max_long_edge: int = 1024
    quality: int = 95  # JPEG 质量
    sharpen: float = 0.3  # 锐化强度 0-1

# 尝试导入 OpenCV
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

def resize_with_pil_hq(img: Image.Image, target_size: int, sharpen: float = 0.3) -> Image.Image:
    """使用 PIL 高质量多步下采样"""
    from PIL import ImageEnhance
    
    w, h = img.size
    
    # 计算目标尺寸
    if w >= h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    
    # 确保尺寸是偶数
    new_w = new_w if new_w % 2 == 0 else new_w + 1
    new_h = new_h if new_h % 2 == 0 else new_h + 1
    
    # 多步下采样：每次最多缩小到70%
    current_img = img
    current_w, current_h = w, h
    
    while current_w > new_w * 1.5 or current_h > new_h * 1.5:
        step_w = int(current_w * 0.7)
        step_h = int(current_h * 0.7)
        step_w = max(step_w, new_w)
        step_h = max(step_h, new_h)
        current_img = current_img.resize((step_w, step_h), Image.LANCZOS)
        current_w, current_h = step_w, step_h
    
    # 最终缩放
    result = current_img.resize((new_w, new_h), Image.LANCZOS)
    
    # 锐化恢复细节
    if sharpen > 0:
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(1.0 + sharpen)
    
    return result

def resize_with_cv2_hq(img: Image.Image, target_size: int, sharpen: float = 0.3) -> Image.Image:
    """使用 OpenCV INTER_AREA 高质量下采样 + USM锐化"""
    img_array = np.array(img)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    h, w = img_array.shape[:2]
    
    # 计算目标尺寸
    if w >= h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    
    new_w = new_w if new_w % 2 == 0 else new_w + 1
    new_h = new_h if new_h % 2 == 0 else new_h + 1
    
    # INTER_AREA 最佳抗锯齿下采样
    resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # USM 锐化
    if sharpen > 0:
        blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
        sharpened = cv2.addWeighted(resized, 1.0 + sharpen, blurred, -sharpen, 0)
        resized = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(resized)

def resize_high_quality(img: Image.Image, target_size: int, sharpen: float = 0.3) -> Image.Image:
    """高质量图像缩放，优先使用 OpenCV"""
    if HAS_CV2:
        return resize_with_cv2_hq(img, target_size, sharpen)
    return resize_with_pil_hq(img, target_size, sharpen)

# 缩放状态
resize_state = {
    "running": False,
    "total": 0,
    "completed": 0,
    "current_file": ""
}

@router.get("/resize/status")
async def get_resize_status():
    """获取缩放进度"""
    return resize_state

@router.post("/resize/stop")
async def stop_resize():
    """停止缩放"""
    resize_state["running"] = False
    return {"success": True}

def _resize_single_image(args):
    """单张图片缩放处理（进程池 worker 函数）"""
    img_path_str, max_long_edge, quality, sharpen = args
    img_path = Path(img_path_str)
    
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            long_edge = max(w, h)
            
            # 只处理超过目标尺寸的图片
            if long_edge > max_long_edge:
                # 转换为 RGB
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 使用高质量缩放
                img_resized = resize_high_quality(img, max_long_edge, sharpen=sharpen)
                
                # 保存（覆盖原文件）
                if img_path.suffix.lower() in ['.jpg', '.jpeg']:
                    img_resized.save(img_path, format='JPEG', quality=quality, subsampling=0)
                elif img_path.suffix.lower() == '.png':
                    img_resized.save(img_path, format='PNG', compress_level=1)
                elif img_path.suffix.lower() == '.webp':
                    img_resized.save(img_path, format='WEBP', quality=quality)
                return (True, img_path.name)
        return (True, img_path.name)
    except Exception as e:
        return (False, f"{img_path.name}: {str(e)}")

@router.post("/resize")
async def resize_images(request: ResizeImagesRequest):
    """批量缩放图片（多进程并行，按 CPU 核心数顶满跑）"""
    global resize_state
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if resize_state["running"]:
        raise HTTPException(status_code=400, detail="缩放任务正在进行中")
    
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集路径不存在")
    
    # 收集图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_paths = []
    
    for ext in image_extensions:
        for img_path in dataset_path.rglob(f"*{ext}"):
            image_paths.append(img_path)
    
    if not image_paths:
        return {"success": True, "message": "没有图片需要处理", "total": 0}
    
    # 初始化状态
    resize_state = {
        "running": True,
        "total": len(image_paths),
        "completed": 0,
        "current_file": ""
    }
    
    # 获取 CPU 核心数
    num_workers = multiprocessing.cpu_count()
    print(f"[Resize] 启动多进程并行处理，使用 {num_workers} 个 worker")
    
    # 准备参数
    args_list = [
        (str(img_path), request.max_long_edge, request.quality, request.sharpen)
        for img_path in image_paths
    ]
    
    # 后台多进程执行缩放
    def run_resize_parallel():
        global resize_state
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_resize_single_image, args): args[0] for args in args_list}
                
                for future in as_completed(futures):
                    if not resize_state["running"]:
                        # 取消剩余任务
                        for f in futures:
                            f.cancel()
                        break
                    
                    img_path = futures[future]
                    resize_state["current_file"] = Path(img_path).name
                    resize_state["completed"] += 1
                    
                    try:
                        success, msg = future.result()
                        if not success:
                            print(f"[Resize] Error: {msg}")
                    except Exception as e:
                        print(f"[Resize] Worker error: {e}")
        except Exception as e:
            print(f"[Resize] Pool error: {e}")
        finally:
            resize_state["running"] = False
            resize_state["current_file"] = ""
    
    # 在后台线程中运行进程池
    thread = threading.Thread(target=run_resize_parallel, daemon=True)
    thread.start()
    
    return {
        "success": True,
        "message": f"开始处理 {len(image_paths)} 张图片 (使用 {num_workers} 进程并行)",
        "total": len(image_paths),
        "workers": num_workers
    }

class SaveCaptionRequest(BaseModel):
    path: str  # 图片路径
    caption: str  # 标注内容

@router.post("/caption")
async def save_caption(request: SaveCaptionRequest):
    """保存单张图片的标注"""
    image_path = Path(request.path)
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="图片文件不存在")
    
    # 标注文件与图片同名，扩展名为 .txt
    caption_path = image_path.with_suffix('.txt')
    
    try:
        # 如果标注内容为空，删除标注文件
        if not request.caption.strip():
            if caption_path.exists():
                caption_path.unlink()
            return {"success": True, "message": "标注已删除"}
        
        # 保存标注
        caption_path.write_text(request.caption.strip(), encoding='utf-8')
        return {"success": True, "message": "标注已保存"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")

class DeleteCaptionsRequest(BaseModel):
    dataset_path: str

@router.post("/delete-captions")
async def delete_captions(request: DeleteCaptionsRequest):
    """删除数据集中所有 .txt 标注文件"""
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集路径不存在")
    
    deleted_count = 0
    errors = []
    
    for txt_file in dataset_path.rglob("*.txt"):
        try:
            txt_file.unlink()
            deleted_count += 1
        except Exception as e:
            errors.append(f"{txt_file.name}: {str(e)}")
    
    return {
        "success": True,
        "deleted": deleted_count,
        "errors": errors
    }

# 旧版 run_tagging_thread 和 start_tagging 已删除，使用新的 run_tagging_subprocess 实现
