"""
Z-Image ControlNet Training Script (修复版)

基于 diffusers ZImageControlNetModel 的正确训练流程。
ControlNet 需要从 Transformer 共享关键模块，并将其输出注入 Transformer。

数据集格式:
    source/    - 控制图像 (edge, depth, pose 等)，预处理后的 VAE latents
    target/    - 目标图像，预处理后的 VAE latents
    metadata.jsonl - 包含 caption 和 control_type

使用方式:
    accelerate launch --mixed_precision bf16 scripts/train_zimage_controlnet.py \
        --config configs/current_training.toml
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
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import save_file

# Local imports
from zimage_trainer.acrf_trainer import ACRFTrainer
from zimage_trainer.utils.snr_utils import compute_snr_weights

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
    parser.add_argument("--controlnet", type=str, default=None, help="ControlNet 预训练权重路径")
    parser.add_argument("--vae", type=str, default=None, help="VAE 模型路径")
    
    # ControlNet specific

    parser.add_argument("--conditioning_scale", type=float, default=0.75,
        help="ControlNet 条件强度 (0-1)")
    
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
        
        general_cfg = config.get("general", {})
        training_cfg = config.get("training", {})
        controlnet_cfg = config.get("controlnet", {})
        acrf_cfg = config.get("acrf", {})
        advanced_cfg = config.get("advanced", {})
        
        # Model paths
        args.dit = general_cfg.get("dit", args.dit)
        args.vae = general_cfg.get("vae", args.vae)
        
        # ControlNet specific
        args.controlnet = controlnet_cfg.get("controlnet_path", args.controlnet)

        args.conditioning_scale = controlnet_cfg.get("conditioning_scale", args.conditioning_scale)
        
        # Training
        args.output_dir = general_cfg.get("output_dir", args.output_dir)
        args.output_name = training_cfg.get("output_name", args.output_name)
        args.num_train_epochs = advanced_cfg.get("num_train_epochs", args.num_train_epochs)
        args.learning_rate = training_cfg.get("learning_rate", args.learning_rate)
        args.gradient_accumulation_steps = advanced_cfg.get("gradient_accumulation_steps", args.gradient_accumulation_steps)
        args.save_every_n_epochs = advanced_cfg.get("save_every_n_epochs", args.save_every_n_epochs)
        args.seed = advanced_cfg.get("seed", args.seed)
        
        # AC-RF
        args.turbo_steps = acrf_cfg.get("turbo_steps", args.turbo_steps)
        args.shift = acrf_cfg.get("shift", args.shift)
        args.use_dynamic_shift = acrf_cfg.get("use_dynamic_shift", args.use_dynamic_shift)
        args.jitter_scale = acrf_cfg.get("jitter_scale", args.jitter_scale)
        args.enable_turbo = acrf_cfg.get("enable_turbo", args.enable_turbo)
        
        # Loss
        args.lambda_l1 = training_cfg.get("lambda_l1", args.lambda_l1)
        args.lambda_cosine = training_cfg.get("lambda_cosine", args.lambda_cosine)
        args.snr_gamma = training_cfg.get("snr_gamma", acrf_cfg.get("snr_gamma", args.snr_gamma))
        
        # Optimizer
        args.optimizer_type = training_cfg.get("optimizer_type", args.optimizer_type)
        args.weight_decay = training_cfg.get("weight_decay", args.weight_decay)
        
    return args


def save_controlnet_weights(controlnet, path: str, dtype=torch.bfloat16):
    """保存 ControlNet 模型权重（仅 ControlNet 专属部分）"""
    state_dict = {}
    # 只保存 ControlNet 特有的层，不保存从 Transformer 共享的模块
    for name, param in controlnet.named_parameters():
        if param.requires_grad and "control_" in name:
            state_dict[name] = param.data.to(dtype).cpu()
    save_file(state_dict, path)
    logger.info(f"[SAVE] 已保存 {len(state_dict)} 个 ControlNet 参数到 {path}")


def main():
    global _interrupted
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # seed=-1 表示完全随机（不设置固定种子）
    if args.seed is not None and args.seed >= 0:
        set_seed(args.seed)
        logger.info(f"🎲 固定种子: {args.seed}")
    else:
        logger.info("🎲 随机模式: 每次训练使用不同的随机状态")
    
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

    logger.info(f"💪 条件强度: {args.conditioning_scale}")
    logger.info(f"⚡ 精度: {weight_dtype}")
    logger.info("🔒 Transformer 自动冻结")
    
    # =========================================================================
    # 1. Load Transformer (主模型)
    # =========================================================================
    logger.info("\n[1/6] 加载 Transformer...")
    
    from zimage_trainer.models.zimage_model import ZImageModel
    model = ZImageModel.from_pretrained(args.dit, weight_dtype=weight_dtype)
    transformer = model.transformer.to(accelerator.device)
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("  [CKPT] Gradient checkpointing enabled")
    
    # 冻结 Transformer (ControlNet训练始终冻结主模型)
    transformer.requires_grad_(False)
    transformer.eval()
    logger.info("  [FREEZE] Transformer 已冻结")
    
    # =========================================================================
    # 2. Load ControlNet and share modules from Transformer
    # =========================================================================
    logger.info("\n[2/6] 加载 ControlNet...")
    
    # 尝试从本地 diffusers 导入
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../diffusers/src"))
        from diffusers.models.controlnets.controlnet_z_image import ZImageControlNetModel
        logger.info("  ✓ 使用本地 ZImageControlNetModel")
    except ImportError:
        try:
            from diffusers.models.controlnets import ZImageControlNetModel
        except ImportError:
            logger.error("  ❌ 无法导入 ZImageControlNetModel，请检查 diffusers 版本")
            return
    
    # 创建或加载 ControlNet
    if args.controlnet and os.path.exists(args.controlnet):
        # 从预训练权重加载
        controlnet = ZImageControlNetModel.from_pretrained(
            args.controlnet,
            torch_dtype=weight_dtype,
        )
        logger.info(f"  ✓ 从预训练加载: {args.controlnet}")
    else:
        # 从 Transformer 配置创建新的 ControlNet
        # 需要根据 Transformer 的配置创建匹配的 ControlNet
        controlnet = ZImageControlNetModel(
            control_layers_places=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18],  # 示例配置
            control_refiner_layers_places=[0, 1],
            control_in_dim=16,  # VAE latent channels
            dim=transformer.config.dim if hasattr(transformer, 'config') else 3840,
            n_heads=transformer.config.n_heads if hasattr(transformer, 'config') else 30,
            n_kv_heads=transformer.config.n_kv_heads if hasattr(transformer, 'config') else 30,
        )
        logger.info("  ✓ 创建新的 ControlNet")
    
    # 关键: 从 Transformer 共享模块
    controlnet = ZImageControlNetModel.from_transformer(controlnet, transformer)
    logger.info("  ✓ 已从 Transformer 共享模块")
    
    controlnet = controlnet.to(accelerator.device, dtype=weight_dtype)
    controlnet.train()
    
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
    
    # 统计 ControlNet 可训练参数
    controlnet_params = [p for p in controlnet.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in controlnet_params)
    logger.info(f"  ✓ ControlNet 可训练参数: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # =========================================================================
    # 3. Initialize AC-RF Trainer
    # =========================================================================
    logger.info("\n[3/6] 初始化 AC-RF Trainer...")
    
    use_dynamic_shift = getattr(args, 'use_dynamic_shift', True)
    if isinstance(use_dynamic_shift, str):
        use_dynamic_shift = use_dynamic_shift.lower() in ('true', '1', 'yes')
    
    acrf_trainer = ACRFTrainer(
        num_train_timesteps=1000,
        turbo_steps=args.turbo_steps,
        shift=args.shift,
        use_dynamic_shift=use_dynamic_shift,
    )
    logger.info(f"  ✓ Turbo Steps: {args.turbo_steps}, Shift: {args.shift}")
    
    # =========================================================================
    # 4. DataLoader (ControlNet 专用)
    # =========================================================================
    logger.info("\n[4/6] 加载数据集...")
    
    # TODO: 实现 ControlNet DataLoader
    # 结构: 每个样本需要包含:
    # - target_latents: 目标图像的 VAE latents
    # - control_latents: 控制图像的 VAE latents (canny/depth/pose etc.)
    # - vl_embed: 文本嵌入
    logger.warning("  ⚠ ControlNet DataLoader 尚未实现")
    logger.warning("  需要准备包含 source (控制图) 和 target (目标图) 的配对数据集")
    
    # Placeholder dataloader - 实际使用时需要实现
    # dataloader = create_controlnet_dataloader(args)
    dataloader = None
    
    if dataloader is None:
        logger.error("  ❌ DataLoader 未实现，无法继续训练")
        logger.info("\n" + "=" * 60)
        logger.info("📋 ControlNet 训练框架已准备完成")
        logger.info("=" * 60)
        logger.info("下一步: 实现 ControlNet DataLoader，需要:")
        logger.info("  1. 控制图像 (source/) - Canny/Depth/Pose 等处理后的图像")
        logger.info("  2. 目标图像 (target/) - 原始图像")
        logger.info("  3. 文本描述 (metadata.jsonl)")
        logger.info("  4. 两者的 VAE latents 缓存")
        return
    
    # =========================================================================
    # 5. Optimizer and Scheduler
    # =========================================================================
    logger.info("\n[5/6] 配置优化器...")
    
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
    
    # Prepare with accelerator
    optimizer, dataloader, _ = accelerator.prepare(optimizer, dataloader, None)
    
    max_train_steps = len(dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    # TensorBoard
    writer = None
    if accelerator.is_main_process:
        logging_dir = os.path.join(args.output_dir, "logs", args.output_name)
        os.makedirs(logging_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=logging_dir)
    
    # =========================================================================
    # 6. Training Loop
    # =========================================================================
    logger.info("\n[6/6] 开始训练...")
    logger.info("=" * 60)
    
    global_step = 0
    ema_loss = None
    ema_decay = 0.99
    
    for epoch in range(args.num_train_epochs):
        if _interrupted:
            logger.info("[EXIT] Training interrupted")
            if accelerator.is_main_process and global_step > 0:
                emergency_path = Path(args.output_dir) / f"{args.output_name}_interrupted.safetensors"
                save_controlnet_weights(controlnet, str(emergency_path), dtype=weight_dtype)
            break
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}", disable=not accelerator.is_main_process)
        
        for batch in pbar:
            if _interrupted:
                break
            
            with accelerator.accumulate(controlnet):
                # 获取数据
                target_latents = batch['target_latents'].to(accelerator.device, dtype=weight_dtype)
                control_latents = batch['control_latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']
                vl_embed = [v.to(accelerator.device, dtype=weight_dtype) for v in vl_embed]
                
                # Sample noise and timesteps
                noise = torch.randn_like(target_latents)
                noisy_latents, timesteps, target = acrf_trainer.sample_batch(
                    target_latents, noise, jitter_scale=args.jitter_scale, use_anchor=args.enable_turbo
                )
                
                # ControlNet Forward
                # 输入: noisy_latents, timesteps, text_embeds, control_image
                # 输出: controlnet_block_samples (dict)
                model_input = noisy_latents.unsqueeze(2)
                model_input_list = list(model_input.unbind(dim=0))
                t_norm = (1000 - timesteps) / 1000.0
                
                # 控制图像也需要转换为 list 格式
                control_input = control_latents.unsqueeze(2)
                control_input_list = list(control_input.unbind(dim=0))
                
                controlnet_block_samples = controlnet(
                    x=model_input_list,
                    t=t_norm.to(dtype=weight_dtype),
                    cap_feats=vl_embed,
                    control_context=control_input_list,
                    conditioning_scale=args.conditioning_scale,
                )
                
                # Transformer Forward (with ControlNet injection)
                with torch.no_grad() if args.freeze_transformer else torch.enable_grad():
                    pred_list = transformer(
                        x=model_input_list,
                        t=t_norm.to(dtype=weight_dtype),
                        cap_feats=vl_embed,
                        controlnet_block_samples=controlnet_block_samples,  # 注入 ControlNet 输出
                    )[0]
                pred = -torch.stack(pred_list, dim=0).squeeze(2)
                
                # Compute losses - ControlNet 使用标准 MSE L2 Loss
                snr_weights = compute_snr_weights(
                    timesteps, gamma=args.snr_gamma, floor=args.snr_floor
                ).to(weight_dtype)
                
                # MSE L2 Loss (标准 ControlNet 训练方法)
                mse_loss = F.mse_loss(pred, target, reduction='none')
                mse_loss = (mse_loss.mean(dim=(1, 2, 3)) * snr_weights).mean()
                total_loss = mse_loss
                
                # Backward
                total_loss = total_loss.float()
                accelerator.backward(total_loss)
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            # EMA loss
            loss_item = total_loss.item()
            if ema_loss is None:
                ema_loss = loss_item
            else:
                ema_loss = ema_decay * ema_loss + (1 - ema_decay) * loss_item
            
            # Log
            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{ema_loss:.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})
                
                if writer and global_step % 10 == 0:
                    writer.add_scalar("train/loss", ema_loss, global_step)
                    writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step)
        
        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % args.save_every_n_epochs == 0:
            save_path = Path(args.output_dir) / f"{args.output_name}_epoch{epoch+1}.safetensors"
            save_controlnet_weights(controlnet, str(save_path), dtype=weight_dtype)
    
    # Final save
    if accelerator.is_main_process:
        final_path = Path(args.output_dir) / f"{args.output_name}_final.safetensors"
        save_controlnet_weights(controlnet, str(final_path), dtype=weight_dtype)
    
    logger.info("\n[DONE] ControlNet Training complete!")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
