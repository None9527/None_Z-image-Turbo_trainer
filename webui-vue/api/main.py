"""
Z-Image Trainer Web API Server
FastAPI backend for the training web UI
"""

import argparse
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from core.config import OUTPUTS_DIR, WEBUI_DIR, PROJECT_ROOT
from core import state

# Note: Using system Python environment (conda/venv)
# If you need a specific venv, activate it before running this script

# Import routers
from routers import training, dataset, system, cache, websocket, system_abstract_layer, generation

app = FastAPI(title="Z-Image Trainer", version="1.0.0")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(training.router)
app.include_router(dataset.router)
app.include_router(system.router)
app.include_router(system_abstract_layer.router)  # 抽象层路由
app.include_router(generation.router)
app.include_router(cache.router)

# WebSocket endpoint - 必须在 SPA fallback 之前注册
from fastapi import WebSocket
from routers.websocket import manager, websocket_endpoint, set_main_loop
import asyncio

@app.on_event("startup")
async def startup_event():
    """应用启动时设置主事件循环引用"""
    loop = asyncio.get_running_loop()
    set_main_loop(loop)
    print("Main event loop registered for WebSocket broadcasts")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await websocket_endpoint(ws)

# Mount outputs directory
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Mount frontend static files
DIST_DIR = WEBUI_DIR / "dist"

# Mount assets directory for static files
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

# SPA fallback: serve index.html for all non-API routes
@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    """Serve SPA frontend - return index.html for all non-API routes"""
    if DIST_DIR.exists():
        index_path = DIST_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
    return {"error": "Frontend not found"}

if __name__ == "__main__":
    import uvicorn
    import os
    from dotenv import load_dotenv
    
    # 从 .env 读取配置
    load_dotenv(PROJECT_ROOT / ".env")
    
    # 从环境变量获取端口和主机（与 env.example 一致）
    default_port = int(os.getenv("TRAINER_PORT", "9198"))
    default_host = os.getenv("TRAINER_HOST", "0.0.0.0")
    
    parser = argparse.ArgumentParser(description="Start None Trainer API Server")
    parser.add_argument("--host", type=str, default=default_host, help=f"Host to bind to (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to bind to (default: {default_port})")
    parser.add_argument("--dev", action="store_true", help="Development mode")
    args = parser.parse_args()
    
    state.DEV_MODE = args.dev
    
    print(f"Starting None Trainer on http://{args.host}:{args.port}")
    if args.dev:
        uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)
