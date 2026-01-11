"""
Z-Image ControlNet Training Script

基于 train_zimage_v2.py 扩展的 ControlNet 训练脚本。
支持使用控制图像条件进行训练。

数据集格式:
    source/    - 控制图像 (edge, depth, pose 等)
    target/    - 目标图像
    metadata.jsonl - 包含 caption 和 control_type

关键特性:
- 继承 AC-RF 训练框架
- 支持冻结 Transformer 只训练 ControlNet
- 支持 Dynamic Shift 和 CFG 训练
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
    parser = argparse.ArgumentParser(description="Z-Image ControlNet Training")
    parser.add_argument("--config", type=str, required=True, help="TOML config path")
    
    # Model paths
    parser.add_argument("--dit", type=str, default=None, help="Transformer 模型路径")
    parser.add_argument("--controlnet", type=str, default=None, help="ControlNet 模型路径")
    parser.add_argument("--vae", type=str, default=None, help="VAE 模型路径")
    
    # Training mode
    parser.add_argument("--freeze_transformer", type=bool, default=True,
        help="冻结 Transformer 只训练 ControlNet (推荐)")
    parser.add_argument("--train_lora", type=bool, default=False,
        help="同时训练 Transformer LoRA")
    
    # Training params
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_name", type=str, default="zimage_controlnet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    
    # ControlNet specific
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.75,
        help="ControlNet 条件强度")
    parser.add_argument("--control_type", type=str, default="canny",
        choices=["canny", "depth", "pose", "normal", "lineart", "seg", "multi"],
        help="控制类型")
    
    # AC-RF / Turbo
    parser.add_argument("--turbo_steps", type=int, default=10)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--use_dynamic_shift", type=bool, default=True)
    parser.add_argument("--jitter_scale", type=float, default=0.02)
    
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
        controlnet_cfg = config.get("controlnet", {})
        acrf_cfg = config.get("acrf", {})
        
        # Model paths
        args.dit = model_cfg.get("dit", args.dit)
        args.controlnet = controlnet_cfg.get("path", args.controlnet)
        args.vae = model_cfg.get("vae", args.vae)
        args.output_dir = model_cfg.get("output_dir", args.output_dir)
        
        # ControlNet specific
        args.freeze_transformer = controlnet_cfg.get("freeze_transformer", args.freeze_transformer)
        args.train_lora = controlnet_cfg.get("train_lora", args.train_lora)
        args.controlnet_conditioning_scale = controlnet_cfg.get("conditioning_scale", args.controlnet_conditioning_scale)
        args.control_type = controlnet_cfg.get("control_type", args.control_type)
        
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
        
        # Loss
        args.lambda_l1 = training_cfg.get("lambda_l1", args.lambda_l1)
        args.lambda_cosine = training_cfg.get("lambda_cosine", args.lambda_cosine)
        args.snr_gamma = training_cfg.get("snr_gamma", args.snr_gamma)
        
        # Optimizer
        args.optimizer_type = training_cfg.get("optimizer_type", args.optimizer_type)
        args.weight_decay = training_cfg.get("weight_decay", args.weight_decay)
    
    return args


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
    logger.info("🎛️ Z-Image ControlNet Training")
    logger.info("=" * 60)
    logger.info(f"📁 输出: {args.output_dir}/{args.output_name}")
    logger.info(f"🎮 控制类型: {args.control_type}")
    logger.info(f"⚡ 精度: {weight_dtype}")
    logger.info(f"🔒 冻结 Transformer: {args.freeze_transformer}")
    
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
    
    # Freeze or keep trainable
    if args.freeze_transformer:
        transformer.requires_grad_(False)
        transformer.eval()
        logger.info("  [FREEZE] Transformer 已冻结")
    else:
        transformer.train()
    
    # =========================================================================
    # 2. Load ControlNet
    # =========================================================================
    logger.info("\n[2/5] 加载 ControlNet...")
    
    from diffusers.models.controlnets import ZImageControlNetModel
    
    if args.controlnet and os.path.exists(args.controlnet):
        # Load from checkpoint
        controlnet = ZImageControlNetModel.from_single_file(
            args.controlnet,
            torch_dtype=weight_dtype,
        )
        logger.info(f"  ✓ 从 checkpoint 加载: {args.controlnet}")
    else:
        # Create from transformer
        controlnet = ZImageControlNetModel.from_transformer(None, transformer)
        logger.info("  ✓ 从 Transformer 初始化 ControlNet")
    
    controlnet = controlnet.to(accelerator.device, dtype=weight_dtype)
    controlnet.train()
    
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
    
    # Get ControlNet trainable params
    controlnet_params = list(controlnet.parameters())
    param_count = sum(p.numel() for p in controlnet_params if p.requires_grad)
    logger.info(f"  ✓ ControlNet 参数量: {param_count:,} ({param_count/1e6:.2f}M)")
    
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
    # 4. DataLoader (ControlNet 专用)
    # =========================================================================
    logger.info("\n[4/5] 加载数据集...")
    
    # TODO: 实现 ControlNet 专用 DataLoader
    # 需要同时加载 source (control image) 和 target (target image)
    logger.warning("  ⚠ ControlNet DataLoader 尚未实现，使用占位符")
    
    # Placeholder - 需要实现 create_controlnet_dataloader
    # dataloader = create_controlnet_dataloader(args)
    
    # =========================================================================
    # 5. Optimizer and Scheduler
    # =========================================================================
    logger.info("\n[5/5] 配置优化器...")
    
    trainable_params = controlnet_params
    
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
    logger.info("🚧 ControlNet 训练循环框架已准备")
    logger.info("待实现: ControlNet DataLoader 和训练循环")
    logger.info("=" * 60)
    
    # 训练循环的关键步骤:
    # 1. 加载 source_latents (控制图 VAE encode)
    # 2. 加载 target_latents (目标图 VAE encode)
    # 3. 生成 noisy_latents 和 timesteps
    # 4. ControlNet forward: controlnet_block_samples = controlnet(noisy, t, embed, control_image)
    # 5. Transformer forward: pred = transformer(noisy, t, embed, controlnet_block_samples=...)
    # 6. 计算 loss 并反向传播
    
    logger.info("\n✅ ControlNet 训练脚本框架创建完成")
    logger.info("下一步: 实现 ControlNet DataLoader")


if __name__ == "__main__":
    main()
