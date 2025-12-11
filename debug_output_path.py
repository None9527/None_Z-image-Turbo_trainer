import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(r'd:\AI\None_Z-image-Turbo_trainer')
load_dotenv(PROJECT_ROOT / '.env')

def _resolve_path(env_key, default):
    value = os.getenv(env_key, '')
    if not value:
        value = default
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p

print("=" * 60)
print("检查 OUTPUT 路径配置")
print("=" * 60)

OUTPUT_PATH = os.getenv('OUTPUT_PATH', '')
LORA_PATH_ENV = os.getenv('LORA_PATH', '')

print(f"OUTPUT_PATH env value: '{OUTPUT_PATH}'")
print(f"LORA_PATH env value: '{LORA_PATH_ENV}'")

LORA_PATH = _resolve_path('OUTPUT_PATH', '') or _resolve_path('LORA_PATH', './output')
print(f"Resolved LORA_PATH: {LORA_PATH}")
print(f"Path exists: {LORA_PATH.exists() if LORA_PATH else False}")

# 检查生成的 TOML
import sys
sys.path.insert(0, str(PROJECT_ROOT / 'webui-vue' / 'api'))
from core.config import OUTPUT_BASE_DIR
print(f"OUTPUT_BASE_DIR: {OUTPUT_BASE_DIR}")
