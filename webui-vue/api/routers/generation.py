from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Optional, List
from datetime import datetime
import json
import io
import base64
import sys
import torch
from pathlib import Path
from PIL import Image
import os

from core.config import OUTPUTS_DIR, PROJECT_ROOT, GenerationRequest, DeleteHistoryRequest, LORA_PATH, get_model_path
from core import state
from routers.websocket import sync_broadcast_generation_progress

# 添加src到路径以使用models抽象层
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 导入模型适配器注册表
from models.registry import get_adapter, list_adapters

router = APIRouter(prefix="/api", tags=["generation"])

def load_pipeline_with_adapter(model_type: str):
    """使用抽象层加载本地模型 Pipeline"""
    model_path = get_model_path(model_type, "base")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    # 使用抽象层检测是否使用 local_files_only
    if model_type == "zimage":
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,  # 强制使用本地模型
        )
    elif model_type == "longcat":
        try:
            from longcat_image.pipelines.pipeline_longcat_image import LongCatImagePipeline
        except ImportError:
             # Fallback or try adding path again if needed, though sys.path should be set
            sys.path.append(str(PROJECT_ROOT / "src"))
            from longcat_image.pipelines.pipeline_longcat_image import LongCatImagePipeline

        pipe = LongCatImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,  # 强制使用本地模型
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if torch.cuda.is_available():
        pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    
    return pipe

@router.get("/generation/models")
async def get_available_models():
    """获取可用的生成模型列表"""
    models = []
    
    for model_type in ["zimage", "longcat"]:
        model_path = get_model_path(model_type, "base")
        model_info = {
            "type": model_type,
            "name": "Z-Image Turbo" if model_type == "zimage" else "LongCat Image",
            "path": str(model_path),
            "exists": model_path.exists(),
            "loaded": state.get_pipeline(model_type) is not None,
        }
        
        # 检查必要组件
        if model_path.exists():
            components = ["vae", "text_encoder", "transformer", "scheduler"]
            model_info["components"] = {}
            for comp in components:
                comp_path = model_path / comp
                model_info["components"][comp] = comp_path.exists()
        
        models.append(model_info)
    
    return {"models": models}

@router.get("/loras")
async def get_loras():
    """Scan for LoRA models in LORA_PATH directory"""
    loras = []
    
    for root, _, files in os.walk(LORA_PATH):
        for file in files:
            if file.endswith(".safetensors"):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(LORA_PATH)
                loras.append({
                    "name": str(rel_path),
                    "path": str(full_path),
                    "size": full_path.stat().st_size
                })
    
    return sorted(loras, key=lambda x: x["name"])

@router.get("/loras/download")
async def download_lora(path: str):
    """Download a LoRA model file"""
    file_path = Path(path)
    
    # 安全检查：确保文件在 LORA_PATH 内
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
    
    # 安全检查：确保文件在 LORA_PATH 内
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
    """生成图片 - 同步版本，支持多模型"""
    
    try:
        model_type = req.model_type.lower()
        pipe = state.get_pipeline(model_type)
        
        # 使用抽象层加载本地模型
        if pipe is None:
            pipe = load_pipeline_with_adapter(model_type)
            state.set_pipeline(model_type, pipe)
        
        # LoRA handling (每个模型独立管理)
        current_lora = state.get_lora_path(model_type)
        if req.lora_path:
            if current_lora != req.lora_path:
                if current_lora:
                    try:
                        pipe.unload_lora_weights()
                    except:
                        pass
                pipe.load_lora_weights(req.lora_path)
                state.set_lora_path(model_type, req.lora_path)
        else:
            if current_lora:
                try:
                    pipe.unload_lora_weights()
                except:
                    pass
                state.set_lora_path(model_type, None)
        
        # Seed
        generator = None
        actual_seed = req.seed
        if actual_seed != -1:
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(actual_seed)
        else:
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
            actual_seed = generator.seed()
        
        if req.lora_path:
            pipe.cross_attention_kwargs = {"scale": req.lora_scale}
        
        # 根据模型类型调用不同的生成接口
        if model_type == "zimage":
            image = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
                generator=generator,
            ).images[0]
        elif model_type == "longcat":
            # LongCat/FLUX 使用不同的参数
            image = pipe(
                prompt=req.prompt,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
                generator=generator,
            ).images[0]
        
        if pipe:
            pipe.cross_attention_kwargs = None
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        generated_dir = OUTPUTS_DIR / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{timestamp}.png"
        image_path = generated_dir / filename
        image.save(image_path)
        
        # Metadata (包含模型类型)
        metadata = req.dict()
        metadata["timestamp"] = timestamp
        metadata["seed"] = actual_seed
        with open(generated_dir / f"{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_base64}",
            "filename": filename,
            "timestamp": timestamp,
            "seed": actual_seed,
            "model_type": model_type
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "message": str(e)}


@router.post("/generate-stream")
async def generate_image_stream(req: GenerationRequest):
    """Generate image with streaming progress updates (SSE)"""
    from fastapi.responses import StreamingResponse
    import asyncio
    import queue
    import threading
    import time
    
    progress_queue = queue.Queue()
    result_holder = {"result": None, "error": None}
    
    def do_generate_with_queue():
        """Execute generation in thread, send progress through queue"""
        
        try:
            model_type = req.model_type.lower()
            progress_queue.put({"stage": "loading", "step": 0, "total": req.steps, "message": f"Loading {model_type} model..."})
            
            pipe = state.get_pipeline(model_type)
            
            # 使用抽象层加载本地模型
            if pipe is None:
                pipe = load_pipeline_with_adapter(model_type)
                state.set_pipeline(model_type, pipe)
            
            # LoRA handling (每个模型独立管理)
            current_lora = state.get_lora_path(model_type)
            if req.lora_path:
                if current_lora != req.lora_path:
                    if current_lora:
                        try:
                            pipe.unload_lora_weights()
                        except:
                            pass
                    pipe.load_lora_weights(req.lora_path)
                    state.set_lora_path(model_type, req.lora_path)
            else:
                if current_lora:
                    try:
                        pipe.unload_lora_weights()
                    except:
                        pass
                    state.set_lora_path(model_type, None)
            
            # Seed
            generator = None
            actual_seed = req.seed
            if actual_seed != -1:
                generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(actual_seed)
            else:
                generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
                actual_seed = generator.seed()
            
            # Progress callback
            def progress_callback(pipe_instance, step_index, timestep, callback_kwargs):
                step = step_index + 1
                progress_queue.put({"stage": "generating", "step": step, "total": req.steps, "message": f"Step {step}/{req.steps}"})
                return callback_kwargs
            
            progress_queue.put({"stage": "generating", "step": 0, "total": req.steps, "message": "Starting..."})
            
            if req.lora_path:
                pipe.cross_attention_kwargs = {"scale": req.lora_scale}
            
            # 根据模型类型调用不同的生成接口
            if model_type == "zimage":
                image = pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    width=req.width,
                    height=req.height,
                    generator=generator,
                    callback_on_step_end=progress_callback,
                ).images[0]
            elif model_type == "longcat":
                image = pipe(
                    prompt=req.prompt,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    width=req.width,
                    height=req.height,
                    generator=generator,
                    callback_on_step_end=progress_callback,
                ).images[0]
            
            if pipe:
                pipe.cross_attention_kwargs = None
            
            progress_queue.put({"stage": "saving", "step": req.steps, "total": req.steps, "message": "Saving..."})
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            generated_dir = OUTPUTS_DIR / "generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{timestamp}.png"
            image_path = generated_dir / filename
            image.save(image_path)
            
            # Metadata (包含模型类型)
            metadata = req.dict()
            metadata["timestamp"] = timestamp
            metadata["seed"] = actual_seed
            with open(generated_dir / f"{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            result_holder["result"] = {
                "success": True,
                "image": img_base64,
                "filename": filename,
                "timestamp": timestamp,
                "seed": actual_seed,
                "model_type": model_type
            }
            
            progress_queue.put({"stage": "completed", "step": req.steps, "total": req.steps, "message": "Completed!"})
            
        except Exception as e:
            result_holder["error"] = str(e)
            progress_queue.put({"stage": "error", "step": 0, "total": req.steps, "message": f"Error: {str(e)}"})
        finally:
            progress_queue.put(None)  # End signal
    
    def generate_sse():
        """Generate Server-Sent Events"""
        thread = threading.Thread(target=do_generate_with_queue)
        thread.start()
        
        try:
            while True:
                item = progress_queue.get(timeout=30)
                if item is None:
                    break
                
                # Send progress to WebSocket clients
                sync_broadcast_generation_progress(item)
                
                yield f"data: {json.dumps(item)}\n\n"
                
                if item.get("stage") in ["completed", "error"]:
                    break
                    
            # Send final result
            if result_holder["result"]:
                yield f"data: {json.dumps(result_holder['result'])}\n\n"
            elif result_holder["error"]:
                yield f"data: {json.dumps({'success': False, 'error': result_holder['error']})}\n\n"
                
        except queue.Empty:
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
    
    if generated_dir.exists():
        json_files = sorted(generated_dir.glob("*.json"), reverse=True)
        for json_file in json_files[:50]:  # Limit to recent 50
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                png_file = json_file.with_suffix(".png")
                if png_file.exists():
                    has_image = True
                    image_size = png_file.stat().st_size
                else:
                    has_image = False
                    
                # 构造前端需要的数据结构
                history_item = {
                    "url": f"/api/history/image/{data.get('timestamp', '')}",
                    "thumbnail": f"/api/history/image/{data.get('timestamp', '')}",
                    "metadata": data,
                    "has_image": has_image,
                    "image_size": image_size if has_image else 0
                }
                    
                history.append(history_item)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
    
    return {"history": history}

@router.post("/history/delete")
async def delete_generation_history(req: DeleteHistoryRequest):
    """Delete specific generation history items"""
    deleted = []
    errors = []
    
    generated_dir = OUTPUTS_DIR / "generated"
    
    for timestamp in req.timestamps:
        try:
            # Delete JSON metadata
            json_file = generated_dir / f"{timestamp}.json"
            if json_file.exists():
                json_file.unlink()
                
            # Delete PNG image
            png_file = generated_dir / f"{timestamp}.png"
            if png_file.exists():
                png_file.unlink()
                
            deleted.append(timestamp)
        except Exception as e:
            errors.append(f"{timestamp}: {str(e)}")
    
    return {
        "success": len(errors) == 0,
        "deleted": deleted,
        "errors": errors
    }

@router.get("/history/image/{timestamp}")
async def get_history_image(timestamp: str):
    """Get a specific history image"""
    png_file = OUTPUTS_DIR / "generated" / f"{timestamp}.png"
    
    if not png_file.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        path=str(png_file),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=\"{timestamp}.png\""}
    )