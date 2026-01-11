"""
Z-Image Img2Img Training Script

基于 train_zimage_v2.py 扩展的图像转换训练脚本。
支持使用源图像和目标图像对进行训练。

数据集格式:
    source/    - 源图像 (输入)
    target/    - 目标图像 (期望输出)
    metadata.jsonl - 包含 caption 和可选的 strength_hint

关键特性:
- 继承 AC-RF 训练框架
- Strength 采样策略：训练时随机采样 strength
- 与 Img2Img Pipeline 行为一致
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
    parser = argparse.ArgumentParser(description="Z-Image Img2Img Training")
    parser.add_argument("--config", type=str, required=True, help="TOML config path")
    
    # Model paths
    parser.add_argument("--dit", type=str, default=None, help="Transformer 模型路径")
    parser.add_argument("--vae", type=str, default=None, help="VAE 模型路径")
    
    # Training params
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_name", type=str, default="zimage_img2img")
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
    
    # Img2Img specific
    parser.add_argument("--strength_min", type=float, default=0.3,
        help="Strength 最小值 (训练时随机采样)")
    parser.add_argument("--strength_max", type=float, default=0.9,
        help="Strength 最大值")
    
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
    
    # Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    
    args = parser.parse_args()
    
    # Load config from TOML
    if args.config and os.path.exists(args.config):
        import toml
        config = toml.load(args.config)
        
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        lora_cfg = config.get("lora", {})
        img2img_cfg = config.get("img2img", {})
        acrf_cfg = config.get("acrf", {})
        
        # Model paths
        args.dit = model_cfg.get("dit", args.dit)
        args.vae = model_cfg.get("vae", args.vae)
        args.output_dir = model_cfg.get("output_dir", args.output_dir)
        
        # Img2Img specific
        args.strength_min = img2img_cfg.get("strength_min", args.strength_min)
        args.strength_max = img2img_cfg.get("strength_max", args.strength_max)
        
        # LoRA
        args.network_dim = lora_cfg.get("network_dim", args.network_dim)
        args.network_alpha = lora_cfg.get("network_alpha", args.network_alpha)
        args.resume_lora = lora_cfg.get("resume_lora", args.resume_lora)
        
        # Training
        args.output_name = training_cfg.get("output_name", args.output_name)
        args.num_train_epochs = training_cfg.get("num_train_epochs", args.num_train_epochs)
        args.learning_rate = training_cfg.get("learning_rate", args.learning_rate)
        args.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", args.gradient_accumulation_steps)
        
        # AC-RF
        args.turbo_steps = acrf_cfg.get("turbo_steps", args.turbo_steps)
        args.shift = acrf_cfg.get("shift", args.shift)
        args.use_dynamic_shift = acrf_cfg.get("use_dynamic_shift", args.use_dynamic_shift)
        args.jitter_scale = acrf_cfg.get("jitter_scale", args.jitter_scale)
        args.enable_turbo = acrf_cfg.get("enable_turbo", args.enable_turbo)
        
        # Loss
        args.lambda_l1 = training_cfg.get("lambda_l1", args.lambda_l1)
        args.lambda_cosine = training_cfg.get("lambda_cosine", args.lambda_cosine)
        args.snr_gamma = training_cfg.get("snr_gamma", args.snr_gamma)
        
        # Optimizer
        args.optimizer_type = training_cfg.get("optimizer_type", args.optimizer_type)
        args.weight_decay = training_cfg.get("weight_decay", args.weight_decay)
    
    return args


def sample_strength(batch_size: int, strength_min: float, strength_max: float, device: torch.device) -> torch.Tensor:
    """
    随机采样 strength 值 (uniform)
    
    Args:
        batch_size: batch 大小
        strength_min: 最小 strength
        strength_max: 最大 strength
        device: 设备
        
    Returns:
        (batch_size,) 的 strength tensor
    """
    return torch.rand(batch_size, device=device) * (strength_max - strength_min) + strength_min


def get_timesteps_from_strength(
    strength: torch.Tensor,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """
    根据 strength 计算对应的 timestep
    
    与 Img2Img Pipeline 的逻辑一致:
    - strength=1.0 -> timestep=1000 (从纯噪声开始)
    - strength=0.0 -> timestep=0 (无变化)
    
    Args:
        strength: (batch_size,) strength 值
        num_train_timesteps: 训练总步数
        
    Returns:
        (batch_size,) 的 timestep 值
    """
    # t = strength * num_train_timesteps
    return strength * num_train_timesteps


def scale_noise_for_img2img(
    latents: torch.Tensor,
    noise: torch.Tensor,
    strength: torch.Tensor,
) -> torch.Tensor:
    """
    Img2Img 专用加噪方式
    
    与 FlowMatchEulerDiscreteScheduler.scale_noise 一致:
    z_t = (1 - sigma) * latents + sigma * noise
    
    其中 sigma = strength (在 [0, 1] 范围)
    
    Args:
        latents: 源图像 latents (x_0)
        noise: 标准高斯噪声
        strength: (batch_size,) strength 值
        
    Returns:
        加噪后的 latents
    """
    sigma = strength.view(-1, 1, 1, 1)
    return (1 - sigma) * latents + sigma * noise


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
    logger.info("🖼️ Z-Image Img2Img Training")
    logger.info("=" * 60)
    logger.info(f"📁 输出: {args.output_dir}/{args.output_name}")
    logger.info(f"💪 Strength 范围: [{args.strength_min}, {args.strength_max}]")
    logger.info(f"⚡ 精度: {weight_dtype}")
    
    # =========================================================================
    # 1. Load Transformer
    # =========================================================================
    logger.info("\n[1/5] 加载 Transformer...")
    
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
    # 2. Apply LoRA
    # =========================================================================
    logger.info(f"\n[2/5] 创建 LoRA (rank={args.network_dim})...")
    
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
    # 3. Initialize AC-RF Trainer
    # =========================================================================
    logger.info("\n[3/5] 初始化 AC-RF Trainer...")
    
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
    # 4. DataLoader (Img2Img 专用)
    # =========================================================================
    logger.info("\n[4/5] 加载数据集...")
    
    # TODO: 实现 Img2Img 专用 DataLoader
    # 需要同时加载 source (源图) 和 target (目标图)
    logger.warning("  ⚠ Img2Img DataLoader 尚未实现，使用占位符")
    
    # Placeholder - 需要实现 create_img2img_dataloader
    # dataloader = create_img2img_dataloader(args)
    
    # =========================================================================
    # 5. Optimizer and Scheduler
    # =========================================================================
    logger.info("\n[5/5] 配置优化器...")
    
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
    logger.info("🚧 Img2Img 训练循环框架已准备")
    logger.info("=" * 60)
    
    # 训练循环的关键步骤:
    # 1. 加载 source_latents (源图 VAE encode)
    # 2. 加载 target_latents (目标图 VAE encode)
    # 3. 随机采样 strength 值
    # 4. 使用 scale_noise_for_img2img 加噪
    # 5. 计算对应的 timestep
    # 6. Transformer forward
    # 7. 计算 loss: pred vs (noise - source_latents)
    
    """
    训练伪代码:
    
    for batch in dataloader:
        source_latents = batch['source_latents']  # 源图
        target_latents = batch['target_latents']  # 目标图
        vl_embed = batch['vl_embed']
        
        batch_size = source_latents.shape[0]
        noise = torch.randn_like(target_latents)
        
        # 随机采样 strength
        strength = sample_strength(batch_size, args.strength_min, args.strength_max, device)
        
        # Img2Img 加噪方式
        noisy_latents = scale_noise_for_img2img(target_latents, noise, strength)
        
        # 计算 timestep (与 Pipeline 一致)
        timesteps = get_timesteps_from_strength(strength, 1000)
        
        # 目标是从 noisy_latents 预测 velocity (noise - target_latents)
        target_velocity = noise - target_latents
        
        # Forward pass
        model_pred = transformer(noisy_latents, timesteps, vl_embed)
        
        # 计算 loss
        loss = F.l1_loss(model_pred, target_velocity)
    """
    
    logger.info("\n✅ Img2Img 训练脚本框架创建完成")
    logger.info("下一步: 实现 Img2Img DataLoader")


if __name__ == "__main__":
    main()
