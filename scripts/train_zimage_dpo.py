"""
[DPO] Z-Image DPO LoRA Training Script

Direct Preference Optimization training for Z-Image models.
Learns to generate images that align with human preferences by comparing
preferred vs rejected image pairs.

Key Features:
- Policy/Reference model sharing (LoRA toggle)
- Same noise & timestep for preference pairs (critical!)
- Sigmoid/Hinge/IPO DPO loss variants
- Optional SNR weighting
- TensorBoard logging with implicit accuracy metric

Usage:
    accelerate launch --mixed_precision bf16 scripts/train_zimage_dpo.py \
        --config configs/dpo_training.toml

Based on: Diffusion-DPO (https://arxiv.org/abs/2311.12908)
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

# Local imports
from zimage_trainer.networks.lora import LoRANetwork, ZIMAGE_TARGET_NAMES, ZIMAGE_ADALN_NAMES, EXCLUDE_PATTERNS
from zimage_trainer.dataset.dpo_dataset import create_dpo_dataloader
from zimage_trainer.acrf_trainer import ACRFTrainer
from zimage_trainer.losses.dpo_loss import DPOLoss, DPOLossWithSNR

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
    parser = argparse.ArgumentParser(description="Z-Image DPO LoRA Training")
    parser.add_argument("--config", type=str, required=True, help="TOML config path")
    
    # Model
    parser.add_argument("--dit", type=str, default=None)
    parser.add_argument("--vae", type=str, default=None)
    
    # Training
    parser.add_argument("--output_dir", type=str, default="output/dpo")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    
    # LoRA
    parser.add_argument("--network_dim", type=int, default=16)
    parser.add_argument("--network_alpha", type=float, default=16)
    parser.add_argument("--resume_lora", type=str, default=None,
        help="继续训练的 LoRA 路径 (.safetensors)，Rank 将从文件自动推断")
    
    # DPO Parameters
    parser.add_argument("--beta_dpo", type=float, default=2500.0,
        help="DPO β 正则化系数 (推荐 2000-5000)")
    parser.add_argument("--dpo_loss_type", type=str, default="sigmoid",
        choices=["sigmoid", "hinge", "ipo"],
        help="DPO 损失类型")
    parser.add_argument("--dpo_label_smoothing", type=float, default=0.0,
        help="DPO 标签平滑 (0-0.5)")
    
    # AC-RF / Turbo
    parser.add_argument("--turbo_steps", type=int, default=10)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--jitter_scale", type=float, default=0.02)
    
    # SNR
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--use_snr_weighting", type=bool, default=False,
        help="对 DPO 损失应用 SNR 权重")
    
    # Dynamic Shift
    parser.add_argument("--use_dynamic_shift", type=bool, default=True)
    parser.add_argument("--base_seq_len", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--base_shift", type=float, default=0.5)
    parser.add_argument("--max_shift", type=float, default=1.15)
    
    # Memory optimization
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    
    # Turbo mode
    parser.add_argument("--enable_turbo", type=bool, default=True)
    
    # LoRA Advanced
    parser.add_argument("--train_adaln", type=bool, default=False)
    
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
        
        general_cfg = config.get("general", {})
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        lora_cfg = config.get("lora", {})
        dpo_cfg = config.get("dpo", {})
        advanced_cfg = config.get("advanced", {})
        acrf_cfg = config.get("acrf", {})
        
        # Model
        args.dit = general_cfg.get("dit") or model_cfg.get("dit") or args.dit
        args.vae = general_cfg.get("vae") or model_cfg.get("vae") or args.vae
        args.output_dir = general_cfg.get("output_dir") or model_cfg.get("output_dir") or args.output_dir
        
        # Training
        if args.output_name is None:
            args.output_name = training_cfg.get("output_name", "zimage_dpo_lora")
        
        if args.num_train_epochs is None:
            args.num_train_epochs = training_cfg.get("num_train_epochs", 
                                    advanced_cfg.get("num_train_epochs", 10))
        
        if args.learning_rate is None:
            args.learning_rate = training_cfg.get("learning_rate", 5e-5)
        
        args.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps",
                                            advanced_cfg.get("gradient_accumulation_steps", args.gradient_accumulation_steps))
        
        args.seed = training_cfg.get("seed", advanced_cfg.get("seed", args.seed))
        
        if args.save_every_n_epochs is None:
            args.save_every_n_epochs = advanced_cfg.get("save_every_n_epochs", 1)
        
        args.gradient_checkpointing = training_cfg.get("gradient_checkpointing",
                                        advanced_cfg.get("gradient_checkpointing", args.gradient_checkpointing))
        
        # LoRA
        args.network_dim = lora_cfg.get("network_dim", args.network_dim)
        args.network_alpha = lora_cfg.get("network_alpha", args.network_alpha)
        args.resume_lora = lora_cfg.get("resume_lora", args.resume_lora)
        
        # DPO
        args.beta_dpo = dpo_cfg.get("beta", dpo_cfg.get("beta_dpo", args.beta_dpo))
        args.dpo_loss_type = dpo_cfg.get("loss_type", dpo_cfg.get("dpo_loss_type", args.dpo_loss_type))
        args.dpo_label_smoothing = dpo_cfg.get("label_smoothing", args.dpo_label_smoothing)
        args.use_snr_weighting = dpo_cfg.get("use_snr_weighting", args.use_snr_weighting)
        
        # AC-RF
        args.turbo_steps = acrf_cfg.get("turbo_steps", args.turbo_steps)
        args.shift = acrf_cfg.get("shift", args.shift)
        args.jitter_scale = acrf_cfg.get("jitter_scale", args.jitter_scale)
        args.enable_turbo = acrf_cfg.get("enable_turbo", args.enable_turbo)
        
        # SNR
        args.snr_gamma = training_cfg.get("snr_gamma", acrf_cfg.get("snr_gamma", args.snr_gamma))
        
        # Dynamic Shift
        args.use_dynamic_shift = acrf_cfg.get("use_dynamic_shift", args.use_dynamic_shift)
        args.base_seq_len = acrf_cfg.get("base_seq_len", args.base_seq_len)
        args.max_seq_len = acrf_cfg.get("max_seq_len", args.max_seq_len)
        args.base_shift = acrf_cfg.get("base_shift", args.base_shift)
        args.max_shift = acrf_cfg.get("max_shift", args.max_shift)
        
        # Memory
        args.blocks_to_swap = advanced_cfg.get("blocks_to_swap", args.blocks_to_swap)
        
        # LoRA Advanced
        args.train_adaln = lora_cfg.get("train_adaln", args.train_adaln)
        
        # Optimizer
        args.optimizer_type = training_cfg.get("optimizer_type", args.optimizer_type)
        args.weight_decay = training_cfg.get("weight_decay", args.weight_decay)
        
        # Scheduler
        args.lr_scheduler = training_cfg.get("lr_scheduler", args.lr_scheduler)
        args.lr_warmup_steps = training_cfg.get("lr_warmup_steps", args.lr_warmup_steps)
        args.lr_num_cycles = training_cfg.get("lr_num_cycles", args.lr_num_cycles)
        
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
    
    # Set seed
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
    logger.info("🎯 Z-Image DPO LoRA Training")
    logger.info("=" * 60)
    
    # Basic info
    logger.info(f"📁 输出: {args.output_dir}/{args.output_name}")
    logger.info(f"⚡ 精度: {weight_dtype}")
    
    # DPO config
    logger.info(f"\n📊 DPO 参数:")
    logger.info(f"   β: {args.beta_dpo} | Loss: {args.dpo_loss_type}")
    if args.dpo_label_smoothing > 0:
        logger.info(f"   Label Smoothing: {args.dpo_label_smoothing}")
    if args.use_snr_weighting:
        logger.info(f"   SNR Gamma: {args.snr_gamma}")
    
    # Training params
    logger.info(f"\n📋 训练参数:")
    logger.info(f"   Epochs: {args.num_train_epochs} | LR: {args.learning_rate} | Grad Accum: {args.gradient_accumulation_steps}")
    logger.info(f"   LoRA: rank={args.network_dim}, alpha={args.network_alpha}")
    
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
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("  [CKPT] Gradient checkpointing enabled")
    
    transformer.train()
    
    # =========================================================================
    # 2. Block Swapper (if needed)
    # =========================================================================
    if args.blocks_to_swap > 0:
        from zimage_trainer.utils.block_swapper import create_block_swapper
        logger.info(f"\n[SWAP] Initializing Block Swapper (blocks_to_swap={args.blocks_to_swap})...")
        block_swapper = create_block_swapper(
            blocks_to_swap=args.blocks_to_swap,
            device=accelerator.device,
            verbose=True,
        )
        transformer.set_block_swapper(block_swapper)
    
    # =========================================================================
    # 3. Apply LoRA
    # =========================================================================
    
    # Resume mode: infer rank from existing LoRA
    if args.resume_lora and os.path.exists(args.resume_lora):
        logger.info(f"\n[RESUME] 继续训练模式: {args.resume_lora}")
        from safetensors.torch import load_file
        state_dict = load_file(args.resume_lora)
        for key, value in state_dict.items():
            if "lora_down" in key and value.dim() == 2:
                args.network_dim = value.shape[0]
                logger.info(f"  [RESUME] 从权重推断 rank = {args.network_dim}")
                break
    
    logger.info(f"\n[2/6] 创建 LoRA (rank={args.network_dim})...")
    
    # Build target names
    target_names = list(ZIMAGE_TARGET_NAMES)
    exclude_patterns = list(EXCLUDE_PATTERNS)
    
    train_adaln = getattr(args, 'train_adaln', False)
    if isinstance(train_adaln, str):
        train_adaln = train_adaln.lower() in ('true', '1', 'yes')
    train_adaln = bool(train_adaln)
    
    if train_adaln:
        target_names.extend(ZIMAGE_ADALN_NAMES)
        exclude_patterns = [p for p in exclude_patterns if "adaLN" not in p]
        logger.info("  [LoRA] AdaLN 训练已启用")
    
    network = LoRANetwork(
        unet=transformer,
        lora_dim=args.network_dim,
        alpha=args.network_alpha,
        multiplier=1.0,
        target_names=target_names,
        exclude_patterns=exclude_patterns,
    )
    network.apply_to(transformer)
    
    # Load resume weights
    if args.resume_lora and os.path.exists(args.resume_lora):
        network.load_weights(args.resume_lora)
        logger.info(f"  [RESUME] 已加载 LoRA 权重: {os.path.basename(args.resume_lora)}")
    
    # Convert LoRA params to same dtype
    network.to(accelerator.device, dtype=weight_dtype)
    
    # Freeze base model
    transformer.requires_grad_(False)
    
    # Get trainable params
    trainable_params = []
    for lora_module in network.lora_modules.values():
        trainable_params.extend(lora_module.get_trainable_params())
    
    param_count = sum(p.numel() for p in trainable_params)
    logger.info(f"  ✓ 参数量: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # =========================================================================
    # 4. AC-RF Trainer
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
        base_seq_len=getattr(args, 'base_seq_len', 256),
        max_seq_len=getattr(args, 'max_seq_len', 4096),
        base_shift=getattr(args, 'base_shift', 0.5),
        max_shift=getattr(args, 'max_shift', 1.15),
    )
    acrf_trainer.verify_setup()
    
    # =========================================================================
    # 5. DPO Loss Function
    # =========================================================================
    logger.info("\n[4/6] 初始化 DPO Loss...")
    
    if args.use_snr_weighting:
        dpo_loss_fn = DPOLossWithSNR(
            beta=args.beta_dpo,
            loss_type=args.dpo_loss_type,
            snr_gamma=args.snr_gamma,
            label_smoothing=args.dpo_label_smoothing,
        )
        logger.info(f"  ✓ DPOLossWithSNR (β={args.beta_dpo}, type={args.dpo_loss_type})")
    else:
        dpo_loss_fn = DPOLoss(
            beta=args.beta_dpo,
            loss_type=args.dpo_loss_type,
            label_smoothing=args.dpo_label_smoothing,
        )
        logger.info(f"  ✓ DPOLoss (β={args.beta_dpo}, type={args.dpo_loss_type})")
    
    # =========================================================================
    # 6. DataLoader
    # =========================================================================
    logger.info("\n[5/6] 加载 DPO 数据集...")
    args.dataset_config = args.config
    dataloader = create_dpo_dataloader(args)
    logger.info(f"  ✓ {len(dataloader)} batches (preference pairs)")
    
    # =========================================================================
    # 7. Optimizer and Scheduler
    # =========================================================================
    logger.info("\n[6/6] 配置优化器...")
    logger.info(f"  ✓ {args.optimizer_type}, LR={args.learning_rate}")
    
    if args.optimizer_type == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        except ImportError:
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            logger.warning("  ⚠ bitsandbytes 未安装，使用标准 AdamW")
    elif args.optimizer_type == "Prodigy":
        try:
            from prodigyopt import Prodigy
            prodigy_lr = args.learning_rate if args.learning_rate >= 0.1 else 1.0
            optimizer = Prodigy(
                trainable_params, 
                lr=prodigy_lr,
                weight_decay=args.weight_decay,
                safeguard_warmup=True,
                use_bias_correction=True,
            )
            logger.info(f"  🧒 Prodigy 优化器 (自适应 LR)")
        except ImportError:
            logger.warning("  ⚠ prodigyopt 未安装，使用 AdamW")
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Prepare with accelerator
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)
    
    # Calculate max_train_steps
    max_train_steps = len(dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    logger.info(f"Total training steps: {max_train_steps}")
    
    # TensorBoard
    writer = None
    if accelerator.is_main_process:
        output_base = os.environ.get("OUTPUT_PATH", "")
        if not output_base:
            output_base = os.path.dirname(args.output_dir)
        logging_dir = os.path.join(output_base, "logs", args.output_name)
        os.makedirs(logging_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=logging_dir)
        logger.info(f"TensorBoard log directory: {logging_dir}")
    
    # =========================================================================
    # 8. Training Loop
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("🚀 开始 DPO 训练...")
    logger.info("=" * 60)
    
    global_step = 0
    ema_loss = None
    ema_acc = None
    ema_decay = 0.99
    
    for epoch in range(args.num_train_epochs):
        if _interrupted:
            logger.info("[EXIT] Training interrupted by user")
            if accelerator.is_main_process and global_step > 0:
                emergency_path = Path(args.output_dir) / f"{args.output_name}_interrupted_step{global_step}.safetensors"
                network.save_weights(str(emergency_path), dtype=weight_dtype)
                logger.info(f"[SAVE] Emergency checkpoint saved: {emergency_path}")
            break
        
        logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=True)):
            if _interrupted:
                if accelerator.is_main_process and global_step > 0:
                    emergency_path = Path(args.output_dir) / f"{args.output_name}_interrupted_step{global_step}.safetensors"
                    network.save_weights(str(emergency_path), dtype=weight_dtype)
                break
            
            with accelerator.accumulate(transformer):
                # Get data
                preferred_latents = batch['preferred_latents'].to(accelerator.device, dtype=weight_dtype)
                rejected_latents = batch['rejected_latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']
                vl_embed = [v.to(accelerator.device, dtype=weight_dtype) for v in vl_embed]
                
                batch_size = preferred_latents.shape[0]
                
                # CRITICAL: Same noise and timestep for both preferred and rejected
                noise = torch.randn_like(preferred_latents)
                
                # Sample timesteps using AC-RF (same for both)
                noisy_pref, timesteps, target_pref = acrf_trainer.sample_batch(
                    preferred_latents, noise, jitter_scale=args.jitter_scale, use_anchor=args.enable_turbo
                )
                noisy_rej, _, target_rej = acrf_trainer.sample_batch(
                    rejected_latents, noise, jitter_scale=0.0, use_anchor=args.enable_turbo
                )
                # Override timesteps to be identical
                noisy_rej = timesteps.view(-1, 1, 1, 1) * noise + (1 - timesteps.view(-1, 1, 1, 1) / 1000) * rejected_latents
                target_rej = noise - rejected_latents
                
                # Timestep normalization
                timesteps_normalized = (1000 - timesteps) / 1000.0
                timesteps_normalized = timesteps_normalized.to(dtype=weight_dtype)
                
                # ===== Policy Model Forward (LoRA enabled) =====
                # Preferred
                pref_input = noisy_pref.unsqueeze(2)
                if args.gradient_checkpointing:
                    pref_input.requires_grad_(True)
                pref_input_list = list(pref_input.unbind(dim=0))
                
                policy_pred_w_list = transformer(
                    x=pref_input_list,
                    t=timesteps_normalized,
                    cap_feats=vl_embed,
                )[0]
                policy_pred_w = -torch.stack(policy_pred_w_list, dim=0).squeeze(2)
                
                # Rejected
                rej_input = noisy_rej.unsqueeze(2)
                if args.gradient_checkpointing:
                    rej_input.requires_grad_(True)
                rej_input_list = list(rej_input.unbind(dim=0))
                
                policy_pred_l_list = transformer(
                    x=rej_input_list,
                    t=timesteps_normalized,
                    cap_feats=vl_embed,
                )[0]
                policy_pred_l = -torch.stack(policy_pred_l_list, dim=0).squeeze(2)
                
                # ===== Reference Model Forward (LoRA disabled) =====
                network.set_multiplier(0.0)  # Disable LoRA
                
                with torch.no_grad():
                    # Preferred
                    ref_pred_w_list = transformer(
                        x=pref_input_list,
                        t=timesteps_normalized,
                        cap_feats=vl_embed,
                    )[0]
                    ref_pred_w = -torch.stack(ref_pred_w_list, dim=0).squeeze(2)
                    
                    # Rejected
                    ref_pred_l_list = transformer(
                        x=rej_input_list,
                        t=timesteps_normalized,
                        cap_feats=vl_embed,
                    )[0]
                    ref_pred_l = -torch.stack(ref_pred_l_list, dim=0).squeeze(2)
                
                network.set_multiplier(1.0)  # Re-enable LoRA
                
                # ===== Compute DPO Loss =====
                if args.use_snr_weighting:
                    loss, info = dpo_loss_fn(
                        policy_pred_w, policy_pred_l,
                        ref_pred_w, ref_pred_l,
                        target_pref, target_rej,
                        timesteps=timesteps,
                        num_train_timesteps=1000,
                    )
                else:
                    loss, info = dpo_loss_fn(
                        policy_pred_w, policy_pred_l,
                        ref_pred_w, ref_pred_l,
                        target_pref, target_rej,
                    )
                
                # NaN check
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[NaN] Loss is NaN/Inf at step {global_step}")
                    optimizer.zero_grad()
                    continue
                
                # Backward
                loss = loss.float()
                accelerator.backward(loss)
            
            # Optimizer step
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Update EMA
                loss_val = loss.detach().item()
                acc_val = info['implicit_acc']
                
                if ema_loss is None:
                    ema_loss = loss_val
                    ema_acc = acc_val
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * loss_val
                    ema_acc = ema_decay * ema_acc + (1 - ema_decay) * acc_val
                
                current_lr = lr_scheduler.get_last_lr()[0]
                
                # Log
                if accelerator.is_main_process:
                    print(f"[STEP] {global_step}/{max_train_steps} epoch={epoch+1}/{args.num_train_epochs} "
                          f"dpo_loss={loss_val:.4f} ema_loss={ema_loss:.4f} "
                          f"acc={acc_val:.2%} ema_acc={ema_acc:.2%} lr={current_lr:.2e}", flush=True)
                    
                    if writer:
                        writer.add_scalar("train/dpo_loss", loss_val, global_step)
                        writer.add_scalar("train/ema_loss", ema_loss, global_step)
                        writer.add_scalar("train/implicit_acc", acc_val, global_step)
                        writer.add_scalar("train/ema_acc", ema_acc, global_step)
                        writer.add_scalar("train/learning_rate", current_lr, global_step)
                        writer.add_scalar("train/policy_diff", info['policy_diff'], global_step)
                        writer.add_scalar("train/ref_diff", info['ref_diff'], global_step)
        
        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % args.save_every_n_epochs == 0:
            save_path = Path(args.output_dir) / f"{args.output_name}_epoch{epoch+1}.safetensors"
            network.save_weights(str(save_path), dtype=weight_dtype)
            logger.info(f"[SAVE] Checkpoint saved: {save_path}")
    
    # Final save
    if accelerator.is_main_process:
        final_path = Path(args.output_dir) / f"{args.output_name}_final.safetensors"
        network.save_weights(str(final_path), dtype=weight_dtype)
        logger.info(f"[SAVE] Final model saved: {final_path}")
    
    logger.info("\n[DONE] DPO Training complete!")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
