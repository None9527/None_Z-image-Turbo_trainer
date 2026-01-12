"""
TensorBoard日志解析工具模块

提供TensorBoard event文件解析功能，支持：
- 列出训练记录
- 解析标量数据
- 获取可用标签列表
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import os


def list_training_runs(logs_base_dir: Path) -> List[Dict[str, Any]]:
    """
    扫描日志目录返回训练记录列表
    
    Args:
        logs_base_dir: 日志基础目录 (通常是 output/logs)
    
    Returns:
        训练记录列表，每个记录包含 name, start_time, logdir, event_files
    """
    runs = []
    
    if not logs_base_dir.exists():
        return runs
    
    # 扫描所有子目录
    for run_dir in logs_base_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        # 查找 event 文件
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue
        
        # 获取最早的 event 文件时间作为开始时间
        earliest_mtime = min(f.stat().st_mtime for f in event_files)
        start_time = datetime.fromtimestamp(earliest_mtime)
        
        runs.append({
            "name": run_dir.name,
            "start_time": start_time.isoformat(),
            "logdir": str(run_dir),
            "event_files": len(event_files)
        })
    
    # 按开始时间倒序排列（最新在前）
    runs.sort(key=lambda x: x["start_time"], reverse=True)
    return runs


def get_available_tags(logdir: str) -> List[str]:
    """
    获取指定训练记录的所有可用标量标签
    
    Args:
        logdir: 日志目录路径
    
    Returns:
        标签名称列表
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()
        
        return ea.Tags().get("scalars", [])
    except Exception as e:
        print(f"[TensorBoard] 获取标签失败: {e}")
        return []


def get_scalar_data(logdir: str, tag: str) -> List[Dict[str, Any]]:
    """
    解析指定标量数据
    
    Args:
        logdir: 日志目录路径
        tag: 标量标签名称 (如 'train/loss')
    
    Returns:
        标量数据列表，每个数据点包含 step, value, wall_time
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()
        
        if tag not in ea.Tags().get("scalars", []):
            return []
        
        events = ea.Scalars(tag)
        return [
            {
                "step": e.step,
                "value": e.value,
                "wall_time": e.wall_time
            }
            for e in events
        ]
    except Exception as e:
        print(f"[TensorBoard] 解析标量数据失败: {e}")
        return []


def get_all_scalars(logdir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    获取所有标量数据（批量获取，减少IO）
    
    Args:
        logdir: 日志目录路径
    
    Returns:
        字典，键为标签名，值为数据列表
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()
        
        result = {}
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            result[tag] = [
                {
                    "step": e.step,
                    "value": e.value,
                    "wall_time": e.wall_time
                }
                for e in events
            ]
        
        return result
    except Exception as e:
        print(f"[TensorBoard] 批量解析失败: {e}")
        return {}
