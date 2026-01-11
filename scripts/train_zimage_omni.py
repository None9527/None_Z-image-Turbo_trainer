"""
Z-Image Omni Multi-Image Training Script

基于 train_zimage_v2.py 扩展的多图条件训练脚本。
支持使用多个条件图像和 SigLIP 视觉特征进行训练。

数据集格式:
    conditions/  - 条件图像目录 (每个样本可有多个条件图)
    targets/     - 目标图像
    metadata.jsonl - 包含 caption 和条件图列表

关键特性:
- 集成 SigLIP Vision Encoder
- 支持多图条件输入
- 实现 x_combined 和 image_noise_mask 构造
"""

import os
import sys
import argparse
import logging
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

# Local imports
from zimage_trainer.networks.lora import LoRANetwork, ZIMAGE_TARGET_NAMES
from zimage_trainer.acrf_trainer import ACRFTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Interrupt handler
_interrupted = False

def signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    logger.info("[INTERRUPT] Training will stop after current step...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Z-Image Omni Multi-Image Training")
    parser.add_argument("--config", type=str, required=True, help="TOML config path")
    
    # Model paths
    parser.add_argument("--dit", type=str, default=None, help="Transformer 模型路径")
    parser.add_argument("--vae", type=str, default=None, help="VAE 模型路径")
    parser.add_argument("--siglip", type=str, default=None, help="SigLIP 模型路径")
    
    # Training params
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_name", type=str, default="zimage_omni")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    
    # LoRA
    parser.add_argument("--network_dim", type=int, default=16)
    parser.add_argument("--network_alpha", type=float, default=16)
    parser.add_argument("--resume_lora", type=str, default=None)
    
    # Omni specific
    parser.add_argument("--max_condition_images", type=int, default=4,
        help="最大条件图数量")
    parser.add_argument("--freeze_siglip", type=bool, default=True,
        help="是否冻结 SigLIP 编码器")
    
    # AC-RF / Turbo
    parser.add_argument("--turbo_steps", type=int, default=10)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--use_dynamic_shift", type=bool, default=True)
    parser.add_argument("--jitter_scale", type=float, default=0.02)
    parser.add_argument("--enable_turbo", type=bool, default=True)
    
    # Loss weights
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_cosine", type=float, default=0.1)
    
    # SNR
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--snr_floor", type=float, default=0.1)
    
    # Optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW8bit")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    
    args = parser.parse_args()
    
    # Load config from TOML
    if args.config and os.path.exists(args.config):
        import toml
        config = toml.load(args.config)
        
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        lora_cfg = config.get("lora", {})
        omni_cfg = config.get("omni", {})
        acrf_cfg = config.get("acrf", {})
        
        # Model paths
        args.dit = model_cfg.get("dit", args.dit)
        args.vae = model_cfg.get("vae", args.vae)
        args.siglip = omni_cfg.get("siglip", args.siglip)
        args.output_dir = model_cfg.get("output_dir", args.output_dir)
        
        # Omni specific
        args.max_condition_images = omni_cfg.get("max_condition_images", args.max_condition_images)
        args.freeze_siglip = omni_cfg.get("freeze_siglip", args.freeze_siglip)
        
        # LoRA
        args.network_dim = lora_cfg.get("network_dim", args.network_dim)
        args.network_alpha = lora_cfg.get("network_alpha", args.network_alpha)
        args.resume_lora = lora_cfg.get("resume_lora", args.resume_lora)
        
        # Training
        args.output_name = training_cfg.get("output_name", args.output_name)
        args.num_train_epochs = training_cfg.get("num_train_epochs", args.num_train_epochs)
        args.learning_rate = training_cfg.get("learning_rate", args.learning_rate)
        
        # AC-RF
        args.turbo_steps = acrf_cfg.get("turbo_steps", args.turbo_steps)
        args.shift = acrf_cfg.get("shift", args.shift)
        args.use_dynamic_shift = acrf_cfg.get("use_dynamic_shift", args.use_dynamic_shift)
        
        # Loss
        args.lambda_l1 = training_cfg.get("lambda_l1", args.lambda_l1)
        args.lambda_cosine = training_cfg.get("lambda_cosine", args.lambda_cosine)
    
    return args


def prepare_x_combined(
    condition_latents: list,
    target_latent: torch.Tensor,
) -> list:
    """
    构造 Omni 输入: x_combined = [condition_latents...] + [target_latent]
    
    Args:
        condition_latents: 条件图 latents 列表
        target_latent: 目标图 latent (加噪后)
        
    Returns:
        x_combined 列表
    """
    return condition_latents + [target_latent]


def prepare_image_noise_mask(
    num_condition_images: int,
) -> list:
    """
    构造 image_noise_mask: 条件图=0 (干净), 目标图=1 (噪声)
    
    Args:
        num_condition_images: 条件图数量
        
    Returns:
        noise mask 列表
    """
    return [0] * num_condition_images + [1]


def main():
    global _interrupted
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    if args.seed is not None:
        set_seed(args.seed)
    
    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    logger.info("\n" + "=" * 60)
    logger.info("🌌 Z-Image Omni Multi-Image Training")
    logger.info("=" * 60)
    logger.info(f"📁 输出: {args.output_dir}/{args.output_name}")
    logger.info(f"🖼️ 最大条件图: {args.max_condition_images}")
    logger.info(f"⚡ 精度: {weight_dtype}")
    
    # =========================================================================
    # 1. Load Transformer
    # =========================================================================
    logger.info("\n[1/6] 加载 Transformer...")
    
    try:
        from zimage_trainer.models.transformer_z_image import ZImageTransformer2DModel
        logger.info("  ✓ 使用本地 ZImageTransformer2DModel")
    except ImportError:
        from diffusers import ZImageTransformer2DModel
        logger.warning("  ⚠ 使用 diffusers 默认版本")
    
    transformer = ZImageTransformer2DModel.from_pretrained(
        args.dit,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    transformer = transformer.to(accelerator.device)
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("  [CKPT] Gradient checkpointing enabled")
    
    transformer.train()
    
    # =========================================================================
    # 2. Load SigLIP Vision Encoder
    # =========================================================================
    logger.info("\n[2/6] 加载 SigLIP Vision Encoder...")
    
    siglip = None
    siglip_processor = None
    
    if args.siglip:
        try:
            from transformers import Siglip2VisionModel, Siglip2ImageProcessorFast
            siglip = Siglip2VisionModel.from_pretrained(args.siglip, torch_dtype=weight_dtype)
            siglip_processor = Siglip2ImageProcessorFast.from_pretrained(args.siglip)
            siglip = siglip.to(accelerator.device)
            
            if args.freeze_siglip:
                siglip.requires_grad_(False)
                siglip.eval()
                logger.info("  [FREEZE] SigLIP 已冻结")
            else:
                siglip.train()
            
            logger.info(f"  ✓ 加载 SigLIP: {args.siglip}")
        except Exception as e:
            logger.warning(f"  ⚠ SigLIP 加载失败: {e}")
    else:
        logger.warning("  ⚠ 未指定 SigLIP 模型，多图特征将不可用")
    
    # =========================================================================
    # 3. Apply LoRA
    # =========================================================================
    logger.info(f"\n[3/6] 创建 LoRA (rank={args.network_dim})...")
    
    network = LoRANetwork(
        unet=transformer,
        lora_dim=args.network_dim,
        alpha=args.network_alpha,
        multiplier=1.0,
        target_names=ZIMAGE_TARGET_NAMES,
    )
    network.apply_to(transformer)
    
    if args.resume_lora and os.path.exists(args.resume_lora):
        network.load_weights(args.resume_lora)
        logger.info(f"  [RESUME] 已加载 LoRA: {args.resume_lora}")
    
    network.to(accelerator.device, dtype=weight_dtype)
    
    transformer.requires_grad_(False)
    
    trainable_params = []
    for lora_module in network.lora_modules.values():
        trainable_params.extend(lora_module.get_trainable_params())
    
    param_count = sum(p.numel() for p in trainable_params)
    logger.info(f"  ✓ 参数量: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # =========================================================================
    # 4. Initialize AC-RF Trainer
    # =========================================================================
    logger.info("\n[4/6] 初始化 AC-RF Trainer...")
    
    use_dynamic_shift = getattr(args, 'use_dynamic_shift', True)
    if isinstance(use_dynamic_shift, str):
        use_dynamic_shift = use_dynamic_shift.lower() in ('true', '1', 'yes')
    
    acrf_trainer = ACRFTrainer(
        num_train_timesteps=1000,
        turbo_steps=args.turbo_steps,
        shift=args.shift,
        use_dynamic_shift=use_dynamic_shift,
    )
    acrf_trainer.verify_setup()
    
    # =========================================================================
    # 5. DataLoader (Omni 专用)
    # =========================================================================
    logger.info("\n[5/6] 加载数据集...")
    
    # TODO: 实现 Omni 专用 DataLoader
    # 需要加载: conditions (多条件图), target (目标图), caption
    logger.warning("  ⚠ Omni DataLoader 尚未实现，使用占位符")
    
    # =========================================================================
    # 6. Optimizer
    # =========================================================================
    logger.info("\n[6/6] 配置优化器...")
    
    if args.optimizer_type == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        except ImportError:
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            logger.warning("  ⚠ bitsandbytes 未安装，使用标准 AdamW")
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    logger.info(f"  ✓ {args.optimizer_type}, LR={args.learning_rate}")
    
    # =========================================================================
    # Training Loop (框架)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("🚧 Omni 训练循环框架已准备")
    logger.info("=" * 60)
    
    # 训练循环的关键步骤:
    # 1. 加载 condition_images (多条件图)
    # 2. 加载 target_latents (目标图 VAE encode)
    # 3. 对每个条件图计算 SigLIP 特征
    # 4. 构造 x_combined 和 image_noise_mask
    # 5. Transformer forward: pred = transformer(x_combined, t, embed, siglip_feats, noise_mask)
    # 6. 只计算目标图位置的 loss
    
    """
    训练伪代码:
    
    for batch in dataloader:
        condition_images = batch['condition_images']  # List[PIL.Image]
        target_latents = batch['target_latents']
        vl_embed = batch['vl_embed']
        
        batch_size = target_latents.shape[0]
        
        # 1. Encode condition images
        condition_latents = [vae.encode(img) for img in condition_images]
        
        # 2. Extract SigLIP features
        siglip_feats = [siglip(img) for img in condition_images]
        
        # 3. Add noise to target
        noise = torch.randn_like(target_latents)
        noisy_latents, timesteps, target_velocity = acrf_trainer.sample_batch(...)
        
        # 4. Construct x_combined
        x_combined = condition_latents + [noisy_latents]
        image_noise_mask = [0] * len(condition_latents) + [1]
        
        # 5. Forward pass
        model_pred = transformer(
            x=x_combined,
            t=timesteps,
            cap_feats=vl_embed,
            siglip_feats=siglip_feats + [None],
            image_noise_mask=image_noise_mask,
        )
        
        # 6. Compute loss (only on target position)
        loss = F.l1_loss(model_pred[-1], target_velocity)
    """
    
    logger.info("\n✅ Omni 训练脚本框架创建完成")
    logger.info("下一步: 实现 Omni DataLoader")


if __name__ == "__main__":
    main()
