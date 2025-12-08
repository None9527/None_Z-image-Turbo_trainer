# -*- coding: utf-8 -*-
"""
缓存工具 - 统一管理多模型缓存文件后缀

通过抽象层获取模型配置，避免硬编码后缀
"""

from pathlib import Path
from typing import List, Set
import sys

# 确保 src 在路径中
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.registry import get_adapter


# 缓存后缀映射（通过抽象层获取）
def get_cache_suffix(model_type: str) -> str:
    """
    获取指定模型类型的缓存后缀
    
    Args:
        model_type: 模型类型 (zimage, longcat, flux)
        
    Returns:
        缓存后缀字符串 (zi, lc, flux)
    """
    try:
        adapter = get_adapter(model_type)
        return adapter.config.cache_suffix
    except:
        # Fallback
        suffix_map = {
            "zimage": "zi",
            "longcat": "lc",
            "flux": "flux"
        }
        return suffix_map.get(model_type, "zi")


def get_all_cache_suffixes() -> List[str]:
    """
    获取所有支持的缓存后缀
    
    Returns:
        后缀列表 ['zi', 'lc', 'flux']
    """
    return ["zi", "lc", "flux"]


def find_latent_cache(image_stem: str, cache_dir: Path) -> bool:
    """
    检查是否存在任意模型的 latent 缓存
    
    Args:
        image_stem: 图片文件名（不含扩展名）
        cache_dir: 缓存目录
        
    Returns:
        是否存在缓存
    """
    for suffix in get_all_cache_suffixes():
        if any(cache_dir.glob(f"{image_stem}_*_{suffix}.safetensors")):
            return True
    return False


def find_text_cache(image_stem: str, cache_dir: Path) -> bool:
    """
    检查是否存在任意模型的 text 缓存
    
    Args:
        image_stem: 图片文件名（不含扩展名）
        cache_dir: 缓存目录
        
    Returns:
        是否存在缓存
    """
    for suffix in get_all_cache_suffixes():
        if (cache_dir / f"{image_stem}_{suffix}_te.safetensors").exists():
            return True
    return False


def find_all_cache_files(cache_dir: Path, cache_type: str = "all", model_type: str = None) -> List[Path]:
    """
    查找所有缓存文件
    
    Args:
        cache_dir: 缓存目录
        cache_type: 缓存类型 ("latent", "text", "all")
        model_type: 模型类型，None 表示所有模型
        
    Returns:
        缓存文件路径列表
    """
    files = []
    
    # 确定要搜索的后缀
    if model_type:
        suffixes = [get_cache_suffix(model_type)]
    else:
        suffixes = get_all_cache_suffixes()
    
    for suffix in suffixes:
        if cache_type in ["latent", "all"]:
            # Latent 缓存: {name}_{WxH}_{suffix}.safetensors
            # 需要排除 text 缓存 (*_te.safetensors)
            for file in cache_dir.rglob(f"*_{suffix}.safetensors"):
                if not file.name.endswith(f"_{suffix}_te.safetensors"):
                    files.append(file)
        
        if cache_type in ["text", "all"]:
            # Text 缓存: {name}_{suffix}_te.safetensors
            files.extend(cache_dir.rglob(f"*_{suffix}_te.safetensors"))
    
    return list(set(files))  # 去重


def delete_cache_files(cache_dir: Path, delete_latent: bool, delete_text: bool, model_type: str = None) -> tuple:
    """
    删除缓存文件
    
    Args:
        cache_dir: 缓存目录
        delete_latent: 是否删除 latent 缓存
        delete_text: 是否删除 text 缓存
        model_type: 模型类型，None 表示所有模型
        
    Returns:
        (删除数量, 错误列表)
    """
    deleted_count = 0
    errors = []
    
    # 查找要删除的文件
    files_to_delete = []
    
    if delete_latent:
        files_to_delete.extend(find_all_cache_files(cache_dir, "latent", model_type))
    
    if delete_text:
        files_to_delete.extend(find_all_cache_files(cache_dir, "text", model_type))
    
    # 执行删除
    for file in files_to_delete:
        try:
            file.unlink()
            deleted_count += 1
        except Exception as e:
            errors.append(str(file))
    
    return deleted_count, errors
