"""
模型完整性校验脚本 - 使用 ModelScope API 校验文件 hash

Usage:
    python verify_model.py <model_dir> [model_id]
    
Examples:
    python verify_model.py ./zimage_models
    python verify_model.py ./longcat_models meituan-longcat/LongCat-Image
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 模型映射
MODEL_MAP = {
    "zimage": "Tongyi-MAI/Z-Image-Turbo",
    "longcat": "meituan-longcat/LongCat-Image",
}


def calculate_sha256(file_path: Path) -> str:
    """计算文件的 SHA-256 hash"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_modelscope_file_list(model_id: str) -> Optional[Dict]:
    """从 ModelScope 获取模型文件列表（包含 hash）"""
    try:
        from modelscope.hub.api import HubApi
        api = HubApi()
        
        # 获取模型文件列表
        files = api.get_model_files(model_id)
        
        result = {}
        for file_info in files:
            if isinstance(file_info, dict):
                name = file_info.get('Name') or file_info.get('Path', '')
                sha256 = file_info.get('Sha256', '')
                size = file_info.get('Size', 0)
            else:
                # 可能是字符串格式
                name = str(file_info)
                sha256 = ''
                size = 0
            
            if name:
                result[name] = {
                    'sha256': sha256,
                    'size': size
                }
        
        return result
    except Exception as e:
        print(f"[ERROR] 无法获取 ModelScope 文件列表: {e}")
        return None


def verify_file(local_path: Path, expected_sha256: str, expected_size: int = 0) -> Tuple[bool, str]:
    """校验单个文件"""
    if not local_path.exists():
        return False, "文件不存在"
    
    local_size = local_path.stat().st_size
    
    # 如果有预期大小，先检查大小
    if expected_size > 0 and local_size != expected_size:
        return False, f"大小不匹配 (本地: {local_size}, 预期: {expected_size})"
    
    # 计算 hash
    if expected_sha256:
        local_sha256 = calculate_sha256(local_path)
        if local_sha256.lower() != expected_sha256.lower():
            return False, f"Hash 不匹配"
    
    return True, "校验通过"


def verify_model(model_dir: Path, model_id: str) -> Dict:
    """校验整个模型目录"""
    print(f"=" * 60)
    print(f"Model ID: {model_id}")
    print(f"Local directory: {model_dir}")
    print(f"=" * 60)
    print()
    
    # 获取远程文件列表
    print("[INFO] 正在从 ModelScope 获取文件列表...")
    remote_files = get_modelscope_file_list(model_id)
    
    if not remote_files:
        print("[ERROR] 无法获取远程文件列表，将使用本地大小检查")
        return {"success": False, "error": "无法连接 ModelScope"}
    
    print(f"[INFO] 远程文件数量: {len(remote_files)}")
    print()
    
    # 校验结果
    results = {
        "valid": [],
        "invalid": [],
        "missing": [],
        "extra": []
    }
    
    # 检查每个远程文件
    print("[INFO] 开始校验文件...")
    for rel_path, file_info in remote_files.items():
        local_path = model_dir / rel_path
        expected_sha256 = file_info.get('sha256', '')
        expected_size = file_info.get('size', 0)
        
        valid, reason = verify_file(local_path, expected_sha256, expected_size)
        
        if not local_path.exists():
            results["missing"].append(rel_path)
            print(f"  [MISSING] {rel_path}")
        elif valid:
            results["valid"].append(rel_path)
            # 只显示大文件的校验结果
            if expected_size > 10 * 1024 * 1024:  # > 10MB
                print(f"  [OK] {rel_path}")
        else:
            results["invalid"].append({"path": rel_path, "reason": reason})
            print(f"  [INVALID] {rel_path}: {reason}")
    
    # 检查本地多余文件
    for local_file in model_dir.rglob("*"):
        if local_file.is_file():
            rel_path = str(local_file.relative_to(model_dir)).replace("\\", "/")
            if rel_path not in remote_files and not rel_path.startswith("."):
                results["extra"].append(rel_path)
    
    # 汇总
    print()
    print("=" * 60)
    print("[SUMMARY]")
    print(f"  有效文件: {len(results['valid'])}")
    print(f"  无效文件: {len(results['invalid'])}")
    print(f"  缺失文件: {len(results['missing'])}")
    print(f"  多余文件: {len(results['extra'])}")
    print("=" * 60)
    
    is_complete = len(results['missing']) == 0 and len(results['invalid']) == 0
    
    if is_complete:
        print("[RESULT] ✓ 模型完整")
    else:
        print("[RESULT] ✗ 模型不完整，请重新下载")
        if results['missing']:
            print(f"  缺失: {', '.join(results['missing'][:5])}...")
        if results['invalid']:
            print(f"  无效: {', '.join([f['path'] for f in results['invalid'][:5]])}...")
    
    return {
        "success": True,
        "complete": is_complete,
        "results": results
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_model.py <model_dir> [model_id]")
        print("\nExamples:")
        print("  python verify_model.py ./zimage_models")
        print("  python verify_model.py ./longcat_models longcat")
        sys.exit(1)
    
    model_dir = Path(sys.argv[1])
    
    if not model_dir.exists():
        print(f"[ERROR] 目录不存在: {model_dir}")
        sys.exit(1)
    
    # 确定 model_id
    if len(sys.argv) >= 3:
        model_id_arg = sys.argv[2].lower().replace("_", "-")
        model_id = MODEL_MAP.get(model_id_arg, sys.argv[2])
    else:
        # 从目录名推断
        dir_name = model_dir.name.lower()
        if "zimage" in dir_name or "z-image" in dir_name:
            model_id = MODEL_MAP["zimage"]
        elif "longcat" in dir_name:
            model_id = MODEL_MAP["longcat"]
        else:
            print("[ERROR] 无法推断模型类型，请指定 model_id")
            sys.exit(1)
    
    result = verify_model(model_dir, model_id)
    
    sys.exit(0 if result.get("complete", False) else 1)


if __name__ == "__main__":
    main()

