from fastapi import APIRouter, HTTPException
import asyncio
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional, List
from datetime import datetime
import json
import io
import base64
import sys
import queue
import threading
from pathlib import Path
from PIL import Image
import os

# ============================================================================
# 可选 torch 导入（支持无 GPU 环境开发）
# ============================================================================
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    print("[WARN] torch not available - generation features disabled")

from core.config import OUTPUTS_DIR, PROJECT_ROOT, GenerationRequest, DeleteHistoryRequest, LORA_PATH, FINETUNE_PATH, get_model_path
from core import state
from core.generation_core import GenerationParams, get_generator
from routers.websocket import sync_broadcast_generation_progress

# 添加src到路径以使用models抽象层
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 导入模型适配器注册表（也可能依赖 torch，需要保护）
try:
    from models.registry import get_adapter, list_adapters
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    get_adapter = None
    list_adapters = None

router = APIRouter(prefix="/api", tags=["generation"])


def load_pipeline_with_adapter(model_type: str = "zimage"):
    """使用抽象层加载本地模型 Pipeline (仅支持 Z-Image)"""
    model_path = get_model_path("zimage", "base")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=True,
    )
    
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    
    return pipe


@router.get("/generation/models")
async def get_available_models():
    """获取可用的生成模型列表 (仅 Z-Image)"""
    model_path = get_model_path("zimage", "base")
    model_info = {
        "type": "zimage",
        "name": "Z-Image Turbo",
        "path": str(model_path),
        "exists": model_path.exists(),
        "loaded": state.get_pipeline("zimage") is not None,
    }
    
    if model_path.exists():
        components = ["vae", "text_encoder", "transformer", "scheduler"]
        model_info["components"] = {}
        for comp in components:
            comp_path = model_path / comp
            model_info["components"][comp] = comp_path.exists()
    
    return {"models": [model_info]}


@router.get("/loras")
async def get_loras():
    """Scan for LoRA models in LORA_PATH directory (output/lora)"""
    loras = []
    
    print(f"[LoRA] Scanning directory: {LORA_PATH}")
    print(f"[LoRA] Directory exists: {LORA_PATH.exists()}")
    
    if LORA_PATH.exists():
        for root, _, files in os.walk(LORA_PATH):
            for file in files:
                if file.endswith(".safetensors"):
                    full_path = Path(root) / file
                    file_size = full_path.stat().st_size
                    rel_path = full_path.relative_to(LORA_PATH)
                    loras.append({
                        "name": str(rel_path),
                        "path": str(full_path),
                        "size": file_size
                    })
    
    return {
        "loras": sorted(loras, key=lambda x: x["name"]),
        "loraPath": str(LORA_PATH),
        "loraPathExists": LORA_PATH.exists()
    }


@router.get("/transformers")
async def get_transformers():
    """Scan for Transformer models (finetune outputs) in FINETUNE_PATH directory (output/finetune)"""
    transformers = []
    
    # Default transformer path
    default_transformer = get_model_path("zimage", "transformer")
    if default_transformer.exists():
        transformers.append({
            "name": "Default (原始模型)",
            "path": str(default_transformer),
            "size": 0,  # Directory
            "is_default": True
        })
    
    # Scan FINETUNE_PATH for finetune weights
    print(f"[Finetune] Scanning directory: {FINETUNE_PATH}")
    if FINETUNE_PATH.exists():
        for root, _, files in os.walk(FINETUNE_PATH):
            for file in files:
                if file.endswith(".safetensors"):
                    full_path = Path(root) / file
                    try:
                        rel_path = full_path.relative_to(FINETUNE_PATH)
                        transformers.append({
                            "name": str(rel_path),
                            "path": str(full_path),
                            "size": full_path.stat().st_size,
                            "is_default": False
                        })
                    except ValueError:
                        pass
    
    return {
        "transformers": sorted(transformers, key=lambda x: (not x["is_default"], x["name"])),
        "finetunePath": str(FINETUNE_PATH)
    }


@router.get("/loras/download")
async def download_lora(path: str):
    """Download a LoRA model file"""
    file_path = Path(path)
    
    try:
        file_path.resolve().relative_to(LORA_PATH.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not file_path.suffix == ".safetensors":
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream"
    )


@router.delete("/loras/delete")
async def delete_lora(path: str):
    """Delete a LoRA model file"""
    file_path = Path(path)
    
    try:
        file_path.resolve().relative_to(LORA_PATH.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not file_path.suffix == ".safetensors":
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        file_path.unlink()
        return {"success": True, "message": f"Deleted {file_path.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


@router.post("/generate")
async def generate_image(req: GenerationRequest):
    """生成图片 - 使用重构后的核心模块"""
    
    if not TORCH_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Generation unavailable: PyTorch not installed."
        )
    
    def _sync_generate():
        try:
            model_type = req.model_type.lower()
            generator = get_generator(OUTPUTS_DIR)
            
            # 获取或加载 Pipeline
            pipe = state.get_pipeline(model_type)
            if pipe is None:
                pipe = load_pipeline_with_adapter(model_type)
                state.set_pipeline(model_type, pipe)
            
            # 转换参数
            params = GenerationParams(
                prompt=req.prompt,
                model_type=model_type,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                lora_path=req.lora_path,
                lora_scale=req.lora_scale,
                comparison_mode=req.comparison_mode,
            )
            
            # 对比模式或普通模式
            if req.comparison_mode and req.lora_path:
                comparison_result = generator.generate_comparison(pipe, params)
                
                # 新格式: {images, composite, seed}
                # 为前端添加 data: 前缀
                images_with_prefix = []
                for img in comparison_result["images"]:
                    images_with_prefix.append({
                        "image": f"data:image/png;base64,{img['image']}",
                        "lora_path": img["lora_path"],
                        "lora_scale": img["lora_scale"],
                    })
                
                return {
                    "success": True,
                    "comparison_mode": True,
                    "images": images_with_prefix,
                    "seed": comparison_result["seed"],
                    "model_type": model_type,
                }
            else:
                result = generator.generate(pipe, params)
                
                return {
                    "success": True,
                    "image": f"data:image/png;base64,{result.base64}",
                    "filename": result.filename,
                    "timestamp": result.timestamp,
                    "seed": result.seed,
                    "model_type": model_type,
                    "comparison_mode": False,
                }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "message": str(e)}
    
    return await asyncio.to_thread(_sync_generate)


@router.post("/generate-stream")
async def generate_image_stream(req: GenerationRequest):
    """生成图片流式接口 (SSE) - 使用重构后的核心模块"""
    
    if not TORCH_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Generation unavailable: PyTorch not installed."
        )
    
    progress_queue = queue.Queue()
    result_holder = {"result": None, "error": None}
    
    def do_generate_with_queue():
        try:
            model_type = req.model_type.lower()
            generator = get_generator(OUTPUTS_DIR)
            
            progress_queue.put({
                "stage": "loading", 
                "step": 0, 
                "total": req.steps, 
                "message": f"Loading {model_type} model..."
            })
            
            pipe = state.get_pipeline(model_type)
            if pipe is None:
                pipe = load_pipeline_with_adapter(model_type)
                state.set_pipeline(model_type, pipe)
            
            # 转换参数
            params = GenerationParams(
                prompt=req.prompt,
                model_type=model_type,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                lora_path=req.lora_path,
                lora_scale=req.lora_scale,
                comparison_mode=req.comparison_mode,
            )
            
            # Progress callback
            def progress_callback(pipe_instance, step_index, timestep, callback_kwargs):
                step = step_index + 1
                progress_queue.put({
                    "stage": "generating", 
                    "step": step, 
                    "total": params.steps, 
                    "message": f"Step {step}/{params.steps}"
                })
                return callback_kwargs
            
            progress_queue.put({
                "stage": "generating", 
                "step": 0, 
                "total": params.steps, 
                "message": "Starting..."
            })
            
            # 对比模式或普通模式
            if req.comparison_mode and req.lora_path:
                comparison_result = generator.generate_comparison(pipe, params, progress_callback)
                
                # 新格式: {images, composite, seed}
                result_holder["result"] = {
                    "success": True,
                    "comparison_mode": True,
                    "images": comparison_result["images"],
                    "seed": comparison_result["seed"],
                    "model_type": model_type,
                }
            else:
                result = generator.generate(pipe, params, progress_callback)
                
                result_holder["result"] = {
                    "success": True,
                    "image": result.base64,
                    "filename": result.filename,
                    "timestamp": result.timestamp,
                    "seed": result.seed,
                    "model_type": model_type,
                    "comparison_mode": False,
                }
            
            progress_queue.put({
                "stage": "completed", 
                "step": params.steps, 
                "total": params.steps, 
                "message": "Completed!"
            })
            
        except Exception as e:
            result_holder["error"] = str(e)
            progress_queue.put({
                "stage": "error", 
                "step": 0, 
                "total": req.steps, 
                "message": f"Error: {str(e)}"
            })
        finally:
            progress_queue.put(None)  # End signal
    
    async def generate_sse():
        thread = threading.Thread(target=do_generate_with_queue)
        thread.start()
        
        try:
            timeout_counter = 0
            max_timeout = 300  # 5 分钟总超时
            
            while timeout_counter < max_timeout:
                try:
                    # 非阻塞获取，避免阻塞事件循环
                    item = progress_queue.get_nowait()
                    timeout_counter = 0  # 重置超时计数器
                    
                    if item is None:
                        break
                    
                    sync_broadcast_generation_progress(
                        item.get("step", 0),
                        item.get("total", 0),
                        item.get("stage", "generating"),
                        item.get("message", "")
                    )
                    yield f"data: {json.dumps(item)}\n\n"
                    
                    if item.get("stage") in ["completed", "error"]:
                        break
                        
                except queue.Empty:
                    # 队列为空时让出控制权，允许其他协程运行
                    await asyncio.sleep(0.1)
                    timeout_counter += 0.1
                    
            if result_holder["result"]:
                yield f"data: {json.dumps(result_holder['result'])}\n\n"
            elif result_holder["error"]:
                yield f"data: {json.dumps({'success': False, 'error': result_holder['error']})}\n\n"
            elif timeout_counter >= max_timeout:
                yield f"data: {json.dumps({'success': False, 'error': 'Generation timeout'})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'success': False, 'error': str(e)})}\n\n"
        finally:
            thread.join(timeout=5)
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/history")
async def get_generation_history():
    """Get generation history"""
    history = []
    generated_dir = OUTPUTS_DIR / "generated"
    
    if not generated_dir.exists():
        return {"history": []}
    
    for json_file in sorted(generated_dir.glob("*.json"), reverse=True)[:50]:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            image_file = json_file.with_suffix(".png")
            if image_file.exists():
                history.append({
                    "url": f"/api/history/image/{json_file.stem}",
                    "thumbnail": f"/api/history/image/{json_file.stem}?thumb=true",
                    "metadata": metadata
                })
        except Exception as e:
            print(f"Failed to read history {json_file}: {e}")
    
    return {"history": history}


@router.post("/history/delete")
async def delete_generation_history(req: DeleteHistoryRequest):
    """Delete specific generation history items"""
    generated_dir = OUTPUTS_DIR / "generated"
    deleted = []
    failed = []
    
    for timestamp in req.timestamps:
        try:
            json_file = generated_dir / f"{timestamp}.json"
            image_file = generated_dir / f"{timestamp}.png"
            
            if json_file.exists():
                json_file.unlink()
            if image_file.exists():
                image_file.unlink()
            deleted.append(timestamp)
        except Exception as e:
            failed.append({"timestamp": timestamp, "error": str(e)})
    
    return {
        "success": len(failed) == 0,
        "deleted": deleted,
        "failed": failed
    }


@router.get("/history/image/{timestamp}")
async def get_history_image(timestamp: str, thumb: bool = False):
    """Get a specific history image"""
    generated_dir = OUTPUTS_DIR / "generated"
    image_path = generated_dir / f"{timestamp}.png"
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    if thumb:
        # 生成缩略图
        from io import BytesIO
        img = Image.open(image_path)
        img.thumbnail((256, 256))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")
    
    return FileResponse(str(image_path), media_type="image/png")