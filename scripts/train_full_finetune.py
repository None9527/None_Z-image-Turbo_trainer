"""
[FULL FEATURE] Z-Image Full Finetune Training Script

直接微调主模型（不使用LoRA），需要大量显存（40GB+）。
基于 train_zimage_v2.py，移除 LoRA 网络，直接训练主模型参数。

注意事项：
- 显存需求极高（约40GB+），推荐使用 A100/H100
- 训练速度较慢，但效果可能更好
- 输出为完整模型权重（非 LoRA）

Usage:
    accelerate launch --mixed_precision bf16 scripts/train_full_finetune.py \
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
from zimage_trainer.dataset.dataloader import create_dataloader, create_reg_dataloader, get_reg_config
from zimage_trainer.acrf_trainer import ACRFTrainer
from zimage_trainer.utils.snr_utils import compute_snr_weights
from zimage_trainer.utils.l2_scheduler import L2RatioScheduler, create_l2_scheduler_from_args
from zimage_trainer.utils.timestep_aware_loss import TimestepAwareLossScheduler, create_timestep_aware_scheduler_from_args
from zimage_trainer.losses.frequency_aware_loss import FrequencyAwareLoss
from zimage_trainer.losses.style_structure_loss import LatentStyleStructureLoss

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
    parser = argparse.ArgumentParser(description="Z-Image Full Finetune Training")
    parser.add_argument("--config", type=str, required=True, help="TOML config path")
    
    # Model
    parser.add_argument("--dit", type=str, default=None)
    parser.add_argument("--vae", type=str, default=None)
    
    # Training
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    
    # AC-RF / Turbo
    parser.add_argument("--turbo_steps", type=int, default=10)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--jitter_scale", type=float, default=0.02)
    parser.add_argument("--latent_jitter_scale", type=float, default=0.01)
    
    # SNR
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--snr_floor", type=float, default=0.1)
    
    # Loss weights
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_cosine", type=float, default=0.0)
    parser.add_argument("--enable_freq", type=bool, default=True)
    parser.add_argument("--lambda_freq", type=float, default=0.3)
    parser.add_argument("--alpha_hf", type=float, default=1.0)
    parser.add_argument("--beta_lf", type=float, default=0.2)
    parser.add_argument("--enable_style", type=bool, default=True)
    parser.add_argument("--lambda_style", type=float, default=0.3)
    parser.add_argument("--lambda_struct", type=float, default=1.0)
    
    # Style-structure sub-params
    parser.add_argument("--lambda_light", type=float, default=0.5)
    parser.add_argument("--lambda_color", type=float, default=0.3)
    parser.add_argument("--lambda_tex", type=float, default=0.5)
    
    # Curvature Penalty (曲率惩罚)
    parser.add_argument("--enable_curvature", type=bool, default=False)
    parser.add_argument("--lambda_curvature", type=float, default=0.05)
    parser.add_argument("--curvature_interval", type=int, default=10)
    parser.add_argument("--curvature_start_epoch", type=int, default=0)
    
    # Drop Text (保持低 CFG 能力)
    parser.add_argument("--drop_text_ratio", type=float, default=0.0)
    
    # CFG Training
    parser.add_argument("--cfg_training", type=bool, default=False)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--cfg_training_ratio", type=float, default=0.5)
    
    # Dynamic Shift
    parser.add_argument("--use_dynamic_shift", type=bool, default=True)
    parser.add_argument("--base_seq_len", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--base_shift", type=float, default=0.5)
    parser.add_argument("--max_shift", type=float, default=1.15)
    
    # Memory optimization
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--block_swap_enabled", type=bool, default=False)
    
    # Turbo / RAFT mode
    parser.add_argument("--enable_turbo", type=bool, default=True)
    parser.add_argument("--raft_mode", type=bool, default=False)
    parser.add_argument("--free_stream_ratio", type=float, default=0.3)
    
    # L2 Ratio Schedule
    parser.add_argument("--l2_schedule_mode", type=str, default="constant")
    parser.add_argument("--l2_initial_ratio", type=float, default=None)
    parser.add_argument("--l2_final_ratio", type=float, default=None)
    parser.add_argument("--l2_milestones", type=str, default="")
    parser.add_argument("--l2_include_anchor", type=bool, default=False)
    parser.add_argument("--l2_anchor_ratio", type=float, default=0.3)
    
    # Timestep-aware Loss
    parser.add_argument("--enable_timestep_aware_loss", type=bool, default=False)
    parser.add_argument("--timestep_high_threshold", type=float, default=0.7)
    parser.add_argument("--timestep_low_threshold", type=float, default=0.3)
    
    # Optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW8bit")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    
    # Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    
    args = parser.parse_args()
    
    # Load config from TOML
    if args.config:
        import toml
        config = toml.load(args.config)
        
        # General
        general_cfg = config.get("general", {})
        args.dit = general_cfg.get("dit", args.dit)
        args.vae = general_cfg.get("vae", args.vae)
        
        # Training
        training_cfg = config.get("training", {})
        args.output_dir = training_cfg.get("output_dir", args.output_dir)
        args.output_name = training_cfg.get("output_name", args.output_name)
        args.learning_rate = training_cfg.get("learning_rate", args.learning_rate)
        
        # Advanced
        advanced_cfg = config.get("advanced", {})
        args.num_train_epochs = advanced_cfg.get("num_train_epochs", args.num_train_epochs)
        args.gradient_accumulation_steps = advanced_cfg.get("gradient_accumulation_steps", args.gradient_accumulation_steps)
        args.max_grad_norm = advanced_cfg.get("max_grad_norm", args.max_grad_norm)
        args.save_every_n_epochs = advanced_cfg.get("save_every_n_epochs", args.save_every_n_epochs)
        args.seed = advanced_cfg.get("seed", args.seed)
        args.gradient_checkpointing = advanced_cfg.get("gradient_checkpointing", args.gradient_checkpointing)
        
        # AC-RF / ACRF
        acrf_cfg = config.get("acrf", {})
        args.turbo_steps = acrf_cfg.get("turbo_steps", args.turbo_steps)
        args.shift = acrf_cfg.get("shift", args.shift)
        args.jitter_scale = acrf_cfg.get("jitter_scale", args.jitter_scale)
        args.latent_jitter_scale = acrf_cfg.get("latent_jitter_scale", args.latent_jitter_scale)
        
        # SNR
        args.snr_gamma = training_cfg.get("snr_gamma", acrf_cfg.get("snr_gamma", args.snr_gamma))
        args.snr_floor = acrf_cfg.get("snr_floor", args.snr_floor)
        
        # Loss
        args.lambda_l1 = training_cfg.get("lambda_l1", args.lambda_l1)
        args.lambda_cosine = training_cfg.get("lambda_cosine", args.lambda_cosine)
        args.enable_freq = training_cfg.get("enable_freq", args.enable_freq)
        args.lambda_freq = training_cfg.get("lambda_freq", args.lambda_freq)
        args.alpha_hf = training_cfg.get("alpha_hf", args.alpha_hf)
        args.beta_lf = training_cfg.get("beta_lf", args.beta_lf)
        args.enable_style = training_cfg.get("enable_style", args.enable_style)
        args.lambda_style = training_cfg.get("lambda_style", args.lambda_style)
        args.lambda_struct = training_cfg.get("lambda_struct", args.lambda_struct)
        args.lambda_light = training_cfg.get("lambda_light", args.lambda_light)
        args.lambda_color = training_cfg.get("lambda_color", args.lambda_color)
        args.lambda_tex = training_cfg.get("lambda_tex", args.lambda_tex)
        
        # Memory
        args.blocks_to_swap = advanced_cfg.get("blocks_to_swap", args.blocks_to_swap)
        args.block_swap_enabled = args.blocks_to_swap > 0
        
        # Turbo / RAFT mode
        args.enable_turbo = acrf_cfg.get("enable_turbo", args.enable_turbo)
        args.raft_mode = acrf_cfg.get("raft_mode", args.raft_mode)
        args.free_stream_ratio = acrf_cfg.get("free_stream_ratio", args.free_stream_ratio)
        
        # L2 Schedule
        args.l2_schedule_mode = acrf_cfg.get("l2_schedule_mode", args.l2_schedule_mode)
        args.l2_initial_ratio = acrf_cfg.get("l2_initial_ratio", args.l2_initial_ratio)
        args.l2_final_ratio = acrf_cfg.get("l2_final_ratio", args.l2_final_ratio)
        args.l2_milestones = acrf_cfg.get("l2_milestones", args.l2_milestones)
        args.l2_include_anchor = acrf_cfg.get("l2_include_anchor", args.l2_include_anchor)
        args.l2_anchor_ratio = acrf_cfg.get("l2_anchor_ratio", args.l2_anchor_ratio)
        
        # Timestep-aware Loss
        args.enable_timestep_aware_loss = acrf_cfg.get("enable_timestep_aware_loss", args.enable_timestep_aware_loss)
        args.timestep_high_threshold = acrf_cfg.get("timestep_high_threshold", args.timestep_high_threshold)
        args.timestep_low_threshold = acrf_cfg.get("timestep_low_threshold", args.timestep_low_threshold)
        
        # Dynamic Shift
        args.use_dynamic_shift = acrf_cfg.get("use_dynamic_shift", args.use_dynamic_shift)
        args.base_seq_len = acrf_cfg.get("base_seq_len", args.base_seq_len)
        args.max_seq_len = acrf_cfg.get("max_seq_len", args.max_seq_len)
        args.base_shift = acrf_cfg.get("base_shift", args.base_shift)
        args.max_shift = acrf_cfg.get("max_shift", args.max_shift)
        
        # Curvature Penalty (曲率惩罚)
        args.enable_curvature = acrf_cfg.get("enable_curvature", getattr(args, 'enable_curvature', False))
        args.lambda_curvature = acrf_cfg.get("lambda_curvature", getattr(args, 'lambda_curvature', 0.05))
        args.curvature_interval = acrf_cfg.get("curvature_interval", getattr(args, 'curvature_interval', 10))
        args.curvature_start_epoch = acrf_cfg.get("curvature_start_epoch", getattr(args, 'curvature_start_epoch', 0))
        
        # CFG Training
        args.cfg_training = acrf_cfg.get("cfg_training", args.cfg_training)
        args.cfg_scale = acrf_cfg.get("cfg_scale", args.cfg_scale)
        args.cfg_training_ratio = acrf_cfg.get("cfg_training_ratio", args.cfg_training_ratio)
        
        # Optimizer
        args.optimizer_type = training_cfg.get("optimizer_type", args.optimizer_type)
        args.weight_decay = training_cfg.get("weight_decay", args.weight_decay)
        
        # Scheduler
        args.lr_scheduler = training_cfg.get("lr_scheduler", args.lr_scheduler)
        args.lr_warmup_steps = training_cfg.get("lr_warmup_steps", args.lr_warmup_steps)
        args.lr_num_cycles = training_cfg.get("lr_num_cycles", args.lr_num_cycles)
        
    return args


def save_transformer_weights(transformer, path: str, dtype=torch.bfloat16):
    """保存 Transformer 模型权重（仅保存可训练参数）"""
    state_dict = {}
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.data.to(dtype).cpu()
    save_file(state_dict, path)
    logger.info(f"[SAVE] 已保存 {len(state_dict)} 个参数到 {path}")


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
    logger.info("🔥 Z-Image Full Finetune Training (高显存模式)")
    logger.info("=" * 60)
    
    logger.warning("⚠️ 全量微调需要约 40GB+ 显存，请确认硬件支持")
    
    # =========================================================================
    # 1. Load Model (不冻结)
    # =========================================================================
    logger.info("\n[1/6] 加载 Transformer 模型...")
    
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
    
    # 不冻结模型，直接训练
    transformer.requires_grad_(True)
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("  [CKPT] Gradient checkpointing enabled")
    
    transformer.train()
    
    # 统计可训练参数
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    logger.info(f"  ✓ 可训练参数: {param_count:,} ({param_count/1e9:.2f}B)")
    
    # =========================================================================
    # 2. Block Swapper (可选)
    # =========================================================================
    block_swapper = None
    if args.blocks_to_swap > 0:
        from zimage_trainer.utils.block_swapper import create_block_swapper
        logger.info(f"\n[SWAP] Initializing Block Swapper...")
        block_swapper = create_block_swapper(
            blocks_to_swap=args.blocks_to_swap,
            device=accelerator.device,
            verbose=True,
        )
        transformer.set_block_swapper(block_swapper)
        logger.info("  [OK] Block Swapper attached")
    
    # =========================================================================
    # 3. AC-RF Trainer
    # =========================================================================
    logger.info("\n[2/6] 初始化 AC-RF Trainer...")
    
    use_dynamic_shift = getattr(args, 'use_dynamic_shift', True)
    if isinstance(use_dynamic_shift, str):
        use_dynamic_shift = use_dynamic_shift.lower() in ('true', '1', 'yes')
    use_dynamic_shift = bool(use_dynamic_shift)
    
    acrf_trainer = ACRFTrainer(
        num_train_timesteps=1000,
        turbo_steps=args.turbo_steps,
        shift=args.shift,
        use_dynamic_shift=use_dynamic_shift,
        base_seq_len=args.base_seq_len,
        max_seq_len=args.max_seq_len,
        base_shift=args.base_shift,
        max_shift=args.max_shift,
    )
    logger.info(f"  ✓ Turbo Steps: {args.turbo_steps}, Shift: {args.shift}")
    logger.info(f"  ✓ Dynamic Shift: {use_dynamic_shift}")
    
    # =========================================================================
    # 4. Loss Functions
    # =========================================================================
    logger.info("\n[3/6] 初始化损失函数...")
    
    freq_loss_fn = None
    if args.enable_freq:
        freq_loss_fn = FrequencyAwareLoss(alpha_hf=args.alpha_hf, beta_lf=args.beta_lf)
        logger.info(f"  ✓ Freq Loss: α_hf={args.alpha_hf}, β_lf={args.beta_lf}")
    
    style_loss_fn = None
    if args.enable_style:
        style_loss_fn = LatentStyleStructureLoss(
            lambda_light=args.lambda_light,
            lambda_color=args.lambda_color,
            lambda_tex=args.lambda_tex,
        )
        logger.info(f"  ✓ Style Loss: light={args.lambda_light}, color={args.lambda_color}")
    
    # Timestep-aware scheduler
    timestep_scheduler = None
    if args.enable_timestep_aware_loss:
        timestep_scheduler = create_timestep_aware_scheduler_from_args(args)
        logger.info("  ✓ Timestep-aware loss enabled")
    
    # =========================================================================
    # 5. DataLoader
    # =========================================================================
    logger.info("\n[4/6] 加载数据集...")
    args.dataset_config = args.config
    dataloader = create_dataloader(args)
    logger.info(f"  ✓ {len(dataloader)} batches")
    
    # 正则数据集
    reg_dataloader = create_reg_dataloader(args)
    reg_config = get_reg_config(args)
    reg_iterator = None
    if reg_dataloader:
        reg_weight = reg_config.get('weight', 1.0)
        reg_ratio = reg_config.get('ratio', 0.5)
        logger.info(f"  + 正则数据集: {len(reg_dataloader)} batches")
    else:
        reg_weight = 0.0
        reg_ratio = 0.0
    
    # =========================================================================
    # 6. Optimizer and Scheduler
    # =========================================================================
    logger.info("\n[5/6] 配置优化器...")
    
    # Full finetune 建议使用较低学习率
    if args.learning_rate > 1e-5:
        logger.warning(f"  ⚠️ 全量微调建议使用较低学习率 (当前: {args.learning_rate})")
    
    if args.optimizer_type == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        except ImportError:
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            logger.warning("  ⚠ bitsandbytes 未安装，使用标准 AdamW")
    elif args.optimizer_type == "Adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay,
            scale_parameter=False, relative_step=False
        )
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
    
    logger.info(f"  ✓ 总步数: {max_train_steps}")
    
    # TensorBoard
    writer = None
    if accelerator.is_main_process:
        logging_dir = os.path.join(args.output_dir, "logs", args.output_name)
        os.makedirs(logging_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=logging_dir)
        logger.info(f"TensorBoard: {logging_dir}")
    
    # =========================================================================
    # 7. Training Loop
    # =========================================================================
    logger.info("\n[6/6] 开始训练...")
    logger.info("=" * 60)
    
    l2_scheduler = create_l2_scheduler_from_args(args)
    
    global_step = 0
    ema_loss = None
    ema_decay = 0.99
    
    for epoch in range(args.num_train_epochs):
        if _interrupted:
            logger.info("[EXIT] Training interrupted")
            if accelerator.is_main_process and global_step > 0:
                emergency_path = Path(args.output_dir) / f"{args.output_name}_interrupted.safetensors"
                save_transformer_weights(transformer, str(emergency_path), dtype=weight_dtype)
            break
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}", disable=not accelerator.is_main_process)
        
        for batch in pbar:
            if _interrupted:
                break
            
            with accelerator.accumulate(transformer):
                latents = batch['latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']
                vl_embed = [v.to(accelerator.device, dtype=weight_dtype) for v in vl_embed]
                
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                noisy_latents, timesteps, target = acrf_trainer.sample_batch(
                    latents, noise, jitter_scale=args.jitter_scale, use_anchor=args.enable_turbo
                )
                
                # Forward pass
                model_input = noisy_latents.unsqueeze(2)
                if args.gradient_checkpointing:
                    model_input.requires_grad_(True)
                model_input_list = list(model_input.unbind(dim=0))
                t_norm = (1000 - timesteps) / 1000.0
                
                pred_list = transformer(
                    x=model_input_list,
                    t=t_norm.to(dtype=weight_dtype),
                    cap_feats=vl_embed,
                )[0]
                pred = -torch.stack(pred_list, dim=0).squeeze(2)
                
                # Compute losses
                snr_weights = compute_snr_weights(
                    timesteps, gamma=args.snr_gamma, floor=args.snr_floor
                ).to(weight_dtype)
                
                # L1 Loss
                l1_loss = F.l1_loss(pred, target, reduction='none')
                l1_loss = (l1_loss.mean(dim=(1, 2, 3)) * snr_weights).mean()
                total_loss = l1_loss * args.lambda_l1
                
                # Cosine Loss
                if args.lambda_cosine > 0:
                    cos_loss = 1.0 - F.cosine_similarity(
                        pred.flatten(1), target.flatten(1), dim=1
                    ).mean()
                    total_loss = total_loss + cos_loss * args.lambda_cosine
                
                # Freq Loss
                if freq_loss_fn is not None and args.lambda_freq > 0:
                    freq_loss = freq_loss_fn(pred, target)
                    total_loss = total_loss + freq_loss * args.lambda_freq
                
                # Style Loss
                if style_loss_fn is not None and args.lambda_style > 0:
                    style_loss = style_loss_fn(pred, target)
                    total_loss = total_loss + style_loss * args.lambda_style
                
                # RAFT L2 Loss
                if args.raft_mode:
                    current_ratio = l2_scheduler.get_ratio(epoch) if l2_scheduler else args.free_stream_ratio
                    l2_loss = F.mse_loss(pred, target, reduction='none')
                    l2_loss = (l2_loss.mean(dim=(1, 2, 3)) * snr_weights).mean()
                    total_loss = total_loss + l2_loss * current_ratio
                
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
            save_transformer_weights(transformer, str(save_path), dtype=weight_dtype)
    
    # Final save
    if accelerator.is_main_process:
        final_path = Path(args.output_dir) / f"{args.output_name}_final.safetensors"
        save_transformer_weights(transformer, str(final_path), dtype=weight_dtype)
    
    logger.info("\n[DONE] Full Finetune Training complete!")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
