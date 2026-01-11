from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# ============================================================================
# Paths & Constants
# ============================================================================

# Project structure
API_DIR = Path(__file__).parent.parent
WEBUI_DIR = API_DIR.parent
PROJECT_ROOT = WEBUI_DIR.parent

# Load .env
load_dotenv(PROJECT_ROOT / ".env")

# 从 .env 读取路径配置 (支持相对路径)
def _resolve_path(env_key: str, default: str) -> Path:
    """解析路径，支持相对路径（相对于项目根目录）"""
    value = os.getenv(env_key, "")
    if not value:
        value = default
    if not value:  # 如果仍然为空，返回 None
        return None
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p

# ============================================================================
# Z-Image 模型路径
# ============================================================================
MODEL_PATH = _resolve_path("MODEL_PATH", "./zimage_models")
VAE_PATH = _resolve_path("VAE_PATH", "") or MODEL_PATH / "vae"
TEXT_ENCODER_PATH = _resolve_path("TEXT_ENCODER_PATH", "") or MODEL_PATH / "text_encoder"
TRANSFORMER_PATH = _resolve_path("TRANSFORMER_PATH", "") or MODEL_PATH / "transformer"

# ============================================================================
# 模型路径映射
# ============================================================================
MODEL_PATHS = {
    "zimage": {
        "base": MODEL_PATH,
        "vae": VAE_PATH,
        "text_encoder": TEXT_ENCODER_PATH,
        "transformer": TRANSFORMER_PATH,
    }
}

def get_model_path(model_type: str = "zimage", component: str = "base") -> Path:
    """获取指定模型的路径
    
    Args:
        model_type: 模型类型 (仅支持 zimage)
        component: 组件类型 (base, vae, text_encoder, transformer)
    
    Returns:
        对应路径
    """
    # 强制使用 zimage
    return MODEL_PATHS["zimage"].get(component, MODEL_PATHS["zimage"]["base"])

# ============================================================================
# 其他路径
# ============================================================================
LORA_PATH = _resolve_path("LORA_PATH", "./output")  # LoRA 模型输出目录
DATASETS_DIR = _resolve_path("DATASET_PATH", "./datasets")
GENERATION_OUTPUT_PATH = _resolve_path("GENERATION_OUTPUT_PATH", "./outputs")  # 图片生成输出路径

OUTPUTS_DIR = GENERATION_OUTPUT_PATH  # 图片生成输出目录
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_BASE_DIR = LORA_PATH  # LoRA 输出目录

# Create necessary directories
OUTPUTS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
LORA_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 启动时路径诊断日志
# ============================================================================
print("=" * 60)
print("[CONFIG] 路径配置诊断")
print("=" * 60)
print(f"[CONFIG] PROJECT_ROOT     = {PROJECT_ROOT}")
print(f"[CONFIG] .env 文件路径     = {PROJECT_ROOT / '.env'}")
print(f"[CONFIG] .env 文件存在     = {(PROJECT_ROOT / '.env').exists()}")
print("-" * 60)
print(f"[CONFIG] DATASET_PATH env = {os.getenv('DATASET_PATH', '<未设置>')}")
print(f"[CONFIG] DATASETS_DIR     = {DATASETS_DIR}")
print(f"[CONFIG] DATASETS_DIR 存在 = {DATASETS_DIR.exists()}")
print("-" * 60)
print(f"[CONFIG] LORA_PATH env    = {os.getenv('LORA_PATH', '<未设置>')}")
print(f"[CONFIG] LORA_PATH        = {LORA_PATH}")
print(f"[CONFIG] LORA_PATH 存在   = {LORA_PATH.exists()}")
print("=" * 60)

# ============================================================================
# Pydantic Models
# ============================================================================

class DatasetScanRequest(BaseModel):
    path: str
    page: int = 1  # 当前页码（从1开始）
    page_size: int = 100  # 每页数量

class CaptionRequest(BaseModel):
    path: str
    caption: Optional[str] = None

class CacheGenerateRequest(BaseModel):
    datasetPath: str
    generateLatent: bool = True
    generateText: bool = True
    vaePath: str = ""
    textEncoderPath: str = ""

class TrainingConfig(BaseModel):
    outputDir: str = "./output"
    outputName: str = "zimage-lora"
    modelPath: str = ""
    vaePath: str = ""
    textEncoderPath: str = ""
    datasetConfigPath: str = "./dataset_config.toml"
    cacheDir: str = "./cache"
    epochs: int = 10
    batchSize: int = 1
    learningRate: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmupSteps: int = 100
    networkDim: int = 64
    networkAlpha: int = 64
    mixedPrecision: str = "bf16"
    gradientCheckpointing: bool = True
    gradientAccumulationSteps: int = 1
    maxGradNorm: float = 1.0
    seed: int = 42

class ConfigSaveRequest(BaseModel):
    path: str
    config: TrainingConfig

class SaveConfigRequest(BaseModel):
    name: str
    config: Dict[str, Any]

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 9
    guidance_scale: float = 1.0
    seed: int = -1
    width: int = 1024
    height: int = 1024
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    comparison_mode: bool = False
    model_type: str = "zimage"  # 仅支持 zimage

class DeleteHistoryRequest(BaseModel):
    timestamps: List[str]
