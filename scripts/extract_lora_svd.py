
import os
import argparse
import logging
import sys
import re
import torch
from safetensors.torch import save_file
from diffusers import Transformer2DModel, UNet2DConditionModel

# Add src to path to support custom models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

# try to set the device to cpu to avoid potential memory issues during imports or small ops
torch.device('cpu')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# Global Imports & Config
# ==============================================================================

# Try to import custom model class
try:
    from zimage_trainer.models.transformer_z_image import ZImageTransformer2DModel
    HAS_ZIMAGE_MODEL = True
except ImportError:
    ZImageTransformer2DModel = None
    HAS_ZIMAGE_MODEL = False

# Try to import LoRA config for strict parity
try:
    from zimage_trainer.networks.lora import ZIMAGE_TARGET_NAMES, EXCLUDE_PATTERNS
    logger.info(f"✅ Loaded LoRA targets from implementation: {len(ZIMAGE_TARGET_NAMES)} targets, {len(EXCLUDE_PATTERNS)} excludes")
    logger.info(f"🎯 Targets: {ZIMAGE_TARGET_NAMES}")
    logger.info(f"🚫 Excludes: {EXCLUDE_PATTERNS}")
except ImportError:
    logger.warning("⚠️ Could not import ZIMAGE_TARGET_NAMES/EXCLUDE_PATTERNS. Using fallbacks.")
    ZIMAGE_TARGET_NAMES = ["to_q", "to_k", "to_v", "to_out", "feed_forward"]
    EXCLUDE_PATTERNS = [r".*embedder.*", r".*pad_token.*", r".*norm.*", r".*adaLN.*", r".*refiner.*", r".*final_layer.*"]

# ==============================================================================
# Helper Functions
# ==============================================================================

def should_process_key(key, target_names, exclude_patterns):
    """
    Determine if a state_dict key should be processed for LoRA extraction.
    """
    # 1. Check strict excludes (Regex)
    for pattern in exclude_patterns:
        if re.search(pattern, key): 
            return False, f"Matched exclude: {pattern}"
            
    # 2. Check strict includes (Substring)
    has_target = False
    for target in target_names:
        if target in key:
            has_target = True
            break
            
    if not has_target:
        return False, "Not in targets"
        
    return True, "OK"

# ==============================================================================
# Main Extraction Logic
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Extract LoRA from two models using SVD (Turbo - Base).")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model (e.g., normal model).")
    parser.add_argument("--turbo_model", type=str, required=True, help="Path to the target/turbo model (e.g., accelerated model).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the extracted LoRA (.safetensors).")
    parser.add_argument("--rank", type=int, default=64, help="Rank for SVD extraction. Default is 64.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation.")
    parser.add_argument("--subfolder", type=str, default=None, help="Subfolder for the model weights (e.g., 'transformer' or 'unet'). Defaults to None (root) or 'transformer' depending on detection.")
    parser.add_argument("--use_unet", action="store_true", help="Use UNet2DConditionModel instead of Transformer2DModel.")
    parser.add_argument("--use_zimage", action="store_true", help="Force use ZImageTransformer2DModel.")
    return parser.parse_args()

def extract_lora_svd(base_model_path: str, turbo_model_path: str, output_path: str, rank: int = 64, device: str = "cuda", subfolder: str = None, use_unet: bool = False, use_zimage: bool = False):
    """
    Extract LoRA weights by subtracting Base model from Turbo model and applying SVD.
    Lora = SVD(Turbo - Base)
    """
    # Fix for "Cannot generate a cpu tensor from a generator of type cuda"
    torch.manual_seed(0)
    
    # Select Model Class
    if use_unet:
        ModelClass = UNet2DConditionModel
    elif use_zimage and HAS_ZIMAGE_MODEL:
        logger.info("👉 Using custom ZImageTransformer2DModel")
        ModelClass = ZImageTransformer2DModel
    elif HAS_ZIMAGE_MODEL:
        logger.info("👉 ZImageTransformer2DModel found, using it by default. Use --use_unet to override if needed.")
        ModelClass = ZImageTransformer2DModel
    else:
        ModelClass = Transformer2DModel
    
    # --------------------------------------------------------------------------
    # Load Models (CPU Only, No Gradient)
    # --------------------------------------------------------------------------
    logger.info(f"🚀 Loading Base Model from: {base_model_path}")
    
    with torch.no_grad():
        try:
            with torch.device("cpu"):
                base_model = ModelClass.from_pretrained(base_model_path, subfolder=subfolder, torch_dtype=torch.float32, local_files_only=True)
        except Exception as e:
            logger.error(f"Failed to load base model with {ModelClass.__name__}: {e}")
            if subfolder is None and not use_unet:
                 logger.info("Retrying with subfolder='transformer'...")
                 try:
                    with torch.device("cpu"):
                        base_model = ModelClass.from_pretrained(base_model_path, subfolder="transformer", torch_dtype=torch.float32, local_files_only=True)
                 except Exception as e2:
                     logger.error(f"Retry failed: {e2}")
                     return
            else:
                return

        logger.info(f"✅ Base Model Loaded. Params: {sum(p.numel() for p in base_model.parameters())/1e6:.2f}M")

        logger.info(f"🚀 Loading Turbo Model from: {turbo_model_path}")
        try:
            with torch.device("cpu"):
                turbo_model = ModelClass.from_pretrained(turbo_model_path, subfolder=subfolder, torch_dtype=torch.float32, local_files_only=True)
        except Exception as e:
            logger.error(f"Failed to load turbo model: {e}")
            if subfolder is None and not use_unet:
                 logger.info("Retrying with subfolder='transformer'...")
                 try:
                    with torch.device("cpu"):
                        turbo_model = ModelClass.from_pretrained(turbo_model_path, subfolder="transformer", torch_dtype=torch.float32, local_files_only=True)
                 except Exception as e2:
                     logger.error(f"Retry failed: {e2}")
                     return
            else:
                return

    # --------------------------------------------------------------------------
    # Prepare Dictionaries & Validate
    # --------------------------------------------------------------------------
    lora_state_dict = {}
    base_dict = base_model.state_dict()
    turbo_dict = turbo_model.state_dict()
    
    logger.info(f"📊 Base Dict Size: {len(base_dict)}")
    logger.info(f"📊 Turbo Dict Size: {len(turbo_dict)}")
    
    base_keys = list(base_dict.keys())
    # turbo_keys = list(turbo_dict.keys())
    
    logger.info(f"🔑 Base Sample: {base_keys[:3]}")
    # logger.info(f"🔑 Turbo Sample: {turbo_keys[:3]}")
    
    common = set(base_dict.keys()) & set(turbo_dict.keys())
    logger.info(f"🔗 Common Keys: {len(common)}")
    
    if len(common) == 0:
        logger.error("❌ CRITICAL: No common keys found!")
        return

    # --------------------------------------------------------------------------
    # Extraction Loop
    # --------------------------------------------------------------------------
    logger.info(f"🧩 Starting SVD extraction (Rank: {rank})...")
    logger.info(f"   Mode: Store on CPU, Compute on {device}, SVD on CPU (Safe)")
    
    # Clean output path
    if not output_path.endswith(".safetensors"):
        output_path += ".safetensors"
        logger.info(f"👉 Auto-appending suffix: {output_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    processed_count = 0
    skipped_count = 0
    debug_skips = 0

    # Explicit iteration (no tqdm for now to ensure we see logs)
    iterable = base_dict.items()
    logger.info("🔍 Inspecting keys & Processing...")
    
    for key, val_base in iterable:
        # ABSOLUTE DEBUG
        if debug_skips < 3:
             logger.info(f"🔄 Iterating: {key}")

        if key not in turbo_dict:
             logger.warning(f"❌ Key missing in Turbo: {key}")
             continue
            
        # 1. Filter Check
        should_proc, reason = should_process_key(key, ZIMAGE_TARGET_NAMES, EXCLUDE_PATTERNS)
        if not should_proc:
            if debug_skips < 5:
                 logger.info(f"⏭️ [SKIP] {key} -> {reason}")
                 debug_skips += 1
            skipped_count += 1
            continue

        # 2. Dim Check
        if "weight" not in key or val_base.dim() < 2:
            # logger.debug(f"Skipping {key} (dim/bias)")
            continue
        
        # 3. Compute
        # .to(device) creates a copy
        w_base = val_base.to(device, dtype=torch.float32)
        w_turbo = turbo_dict[key].to(device, dtype=torch.float32)
        
        w_diff = w_turbo - w_base
        
        # Release inputs early
        del w_base
        del w_turbo

        # Handle Conv
        orig_shape = w_diff.shape
        if w_diff.dim() == 4:
            w_diff = w_diff.flatten(1)

        # SVD
        try:
            # Force SVD on CPU to avoid CUDA OOM/Crashes with large rank/matrices
            # Moving a single matrix to CPU is relatively cheap compared to stability benefits
            w_diff_cpu = w_diff.to("cpu", dtype=torch.float32)
            U, S, Vh = torch.linalg.svd(w_diff_cpu, full_matrices=False)
            del w_diff_cpu
        except RuntimeError as e:
            logger.warning(f"⚠️ SVD failed for {key}: {e}")
            del w_diff
            continue
        except Exception as e:
             logger.error(f"⚠️ Unexpected error during SVD for {key}: {e}")
             del w_diff
             continue
        
        del w_diff

        # Rank Truncate
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        dist_s = torch.sqrt(S)
        lora_down = torch.diag(dist_s) @ Vh
        lora_up = U @ torch.diag(dist_s)
        
        del U, S, Vh, dist_s

        # Reshape & Key Norm
        if len(orig_shape) == 4:
            lora_down = lora_down.reshape(rank, orig_shape[1], orig_shape[2], orig_shape[3])
            lora_up = lora_up.reshape(orig_shape[0], rank, 1, 1)

        base_key = key.replace(".weight", "")
        # Always use diffusion_model. for Comfy/Z-Image compatibility
        lora_key_base = f"diffusion_model.{base_key}"

        # Store to CPU
        lora_state_dict[f"{lora_key_base}.lora_down.weight"] = lora_down.to("cpu").half()
        lora_state_dict[f"{lora_key_base}.lora_up.weight"] = lora_up.to("cpu").half()
        lora_state_dict[f"{lora_key_base}.alpha"] = torch.tensor(rank).half()
        
        del lora_down
        del lora_up

        processed_count += 1
        if processed_count % 10 == 0:
            print(f".", end="", flush=True)

    print("") # Newline
    
    if processed_count == 0:
        logger.warning("❌ No layers processed! Check if targets/excludes are too strict.")
        return

    # Save
    logger.info(f"💾 Saving LoRA to {output_path}...")
    save_file(lora_state_dict, output_path)
    logger.info(f"🎉 Done! Saved {processed_count} layers. (Skipped {skipped_count})")

if __name__ == "__main__":
    try:
        args = parse_args()
        extract_lora_svd(
            base_model_path=args.base_model,
            turbo_model_path=args.turbo_model,
            output_path=args.output_path,
            rank=args.rank,
            device=args.device,
            subfolder=args.subfolder,
            use_unet=args.use_unet,
            use_zimage=args.use_zimage
        )
    except Exception as e:
        logger.critical(f"🔥 Fatal Error: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.warning("🛑 Interrupted by user.")
