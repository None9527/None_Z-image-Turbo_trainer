"""
[FULL FEATURE] Z-Image AC-RF Training Script

Complete AC-RF (Anchor-Constrained Rectified Flow) training with all features.
Based on verified Phase 1 core + full train_acrf.py feature set.

Key Features:
- Local model with use_reentrant=False for frozen + checkpointing
- AC-RF trainer with anchor timesteps and sigmas
- Min-SNR weighting
- Multiple loss functions (L1, Cosine, Frequency, Style-Structure)
- MemoryOptimizer with block swapping
- Proper dtype handling (BF16)

Usage:
    accelerate launch --mixed_precision bf16 scripts/train_zimage_v2.py \
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

# Local imports
from zimage_trainer.networks.lora import LoRANetwork
from zimage_trainer.dataset.dataloader import create_dataloader
from zimage_trainer.acrf_trainer import ACRFTrainer
from zimage_trainer.utils.snr_utils import compute_snr_weights
from zimage_trainer.losses.frequency_aware_loss import FrequencyAwareLoss
from zimage_trainer.losses.style_structure_loss import LatentStyleStructureLoss
from zimage_trainer.utils.memory_optimizer import MemoryOptimizer

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
    parser = argparse.ArgumentParser(description="Z-Image AC-RF Training")
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
    
    # LoRA
    parser.add_argument("--network_dim", type=int, default=16)
    parser.add_argument("--network_alpha", type=float, default=16)
    
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
    
    # Memory optimization
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--block_swap_enabled", type=bool, default=False)
    
    # Turbo / RAFT mode
    parser.add_argument("--enable_turbo", type=bool, default=True)
    parser.add_argument("--raft_mode", type=bool, default=False)
    parser.add_argument("--free_stream_ratio", type=float, default=0.3)
    
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
        
        # Apply config values
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        lora_cfg = config.get("lora", {})
        acrf_cfg = config.get("acrf", {})
        advanced_cfg = config.get("advanced", {})
        
        # Model
        args.dit = model_cfg.get("dit", args.dit)
        args.vae = model_cfg.get("vae", args.vae)
        args.output_dir = model_cfg.get("output_dir", args.output_dir)
        
        # Training
        if args.output_name is None:
            args.output_name = training_cfg.get("output_name", "zimage_lora")
            
        if args.num_train_epochs is None:
            args.num_train_epochs = training_cfg.get("num_train_epochs", 
                                    advanced_cfg.get("num_train_epochs", 10))
                                    
        if args.learning_rate is None:
            args.learning_rate = training_cfg.get("learning_rate", 1e-4)

        args.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps",
                                            advanced_cfg.get("gradient_accumulation_steps", args.gradient_accumulation_steps))
                                            
        if args.save_every_n_epochs is None:
            args.save_every_n_epochs = advanced_cfg.get("save_every_n_epochs", 1)
            
        args.gradient_checkpointing = training_cfg.get("gradient_checkpointing",
                                        advanced_cfg.get("gradient_checkpointing", args.gradient_checkpointing))
        
        # LoRA
        args.network_dim = lora_cfg.get("network_dim", args.network_dim)
        args.network_alpha = lora_cfg.get("network_alpha", args.network_alpha)
        
        # AC-RF
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
        # Style-structure sub-params
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
        
        # Optimizer
        args.optimizer_type = training_cfg.get("optimizer_type", args.optimizer_type)
        args.weight_decay = training_cfg.get("weight_decay", args.weight_decay)
        
        # Scheduler
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
    
    if args.seed is not None:
        set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("[START] Z-Image AC-RF Training")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Turbo steps: {args.turbo_steps}")
    logger.info(f"LoRA rank: {args.network_dim}")
    
    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    logger.info(f"Weight dtype: {weight_dtype}")
    
    # =========================================================================
    # 1. Load Transformer (LOCAL VERSION with use_reentrant=False)
    # =========================================================================
    logger.info("\n[LOAD] Loading Transformer...")
    
    try:
        from zimage_trainer.models.transformer_z_image import ZImageTransformer2DModel
        logger.info("  [LOCAL] Using modified ZImageTransformer2DModel (use_reentrant=False)")
    except ImportError:
        from diffusers import ZImageTransformer2DModel
        logger.warning("  [FALLBACK] Using diffusers ZImageTransformer2DModel")
    
    transformer = ZImageTransformer2DModel.from_pretrained(
        args.dit,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    transformer = transformer.to(accelerator.device)
    
    # Enable gradient checkpointing (BEFORE freeze)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("  [CKPT] Gradient checkpointing enabled")
    
    transformer.train()
    
    # =========================================================================
    # 2. Memory Optimizer (Optional)
    # =========================================================================
    memory_optimizer = None
    if args.blocks_to_swap > 0:
        logger.info(f"\n[MEM] Initializing MemoryOptimizer (blocks_to_swap={args.blocks_to_swap})...")
        memory_config = {
            'block_swap_enabled': True,
            'blocks_to_swap': args.blocks_to_swap,
            'checkpoint_optimization': 'basic' if args.gradient_checkpointing else 'none',
        }
        memory_optimizer = MemoryOptimizer(memory_config)
        memory_optimizer.start()
        logger.info("  [OK] MemoryOptimizer initialized")
    
    # =========================================================================
    # 3. Apply LoRA with proper dtype
    # =========================================================================
    logger.info(f"\n[SETUP] Creating LoRA (rank={args.network_dim})...")
    
    network = LoRANetwork(
        unet=transformer,
        lora_dim=args.network_dim,
        alpha=args.network_alpha,
        multiplier=1.0,
    )
    network.apply_to(transformer)
    
    # CRITICAL: Convert LoRA params to same dtype as model (BF16)
    network.to(accelerator.device, dtype=weight_dtype)
    logger.info(f"  [DTYPE] LoRA params converted to {weight_dtype}")
    
    # Freeze base model (LoRA params remain trainable)
    transformer.requires_grad_(False)
    logger.info("  [FREEZE] Base model frozen (LoRA trainable)")
    
    # Get only LoRA trainable params
    trainable_params = []
    for lora_module in network.lora_modules.values():
        trainable_params.extend(lora_module.get_trainable_params())
    
    param_count = sum(p.numel() for p in trainable_params)
    logger.info(f"  Trainable parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # =========================================================================
    # 4. AC-RF Trainer
    # =========================================================================
    logger.info("\n[INIT] Initializing AC-RF Trainer...")
    acrf_trainer = ACRFTrainer(
        num_train_timesteps=1000,
        turbo_steps=args.turbo_steps,
        shift=args.shift,
    )
    acrf_trainer.verify_setup()
    
    # =========================================================================
    # 5. Loss Functions
    # =========================================================================
    logger.info("\n[LOSS] Configuring loss functions...")
    logger.info(f"  [Basic] lambda_l1={args.lambda_l1}, lambda_cosine={args.lambda_cosine}")
    
    freq_loss_fn = None
    if args.enable_freq:
        freq_loss_fn = FrequencyAwareLoss(
            alpha_hf=args.alpha_hf,
            beta_lf=args.beta_lf,
        )
        logger.info(f"  [Freq] Enabled lambda={args.lambda_freq}, alpha_hf={args.alpha_hf}")
    
    style_loss_fn = None
    if args.enable_style:
        style_loss_fn = LatentStyleStructureLoss(
            lambda_struct=args.lambda_struct,
            lambda_light=args.lambda_light,
            lambda_color=args.lambda_color,
            lambda_tex=args.lambda_tex,
        )
        logger.info(f"  [Style] Enabled lambda={args.lambda_style}, struct={args.lambda_struct}, light={args.lambda_light}, color={args.lambda_color}, tex={args.lambda_tex}")
    
    # RAFT L2 混合模式
    # 确保 raft_mode 是布尔值 (TOML 可能返回字符串)
    if isinstance(args.raft_mode, str):
        args.raft_mode = args.raft_mode.lower() in ('true', '1', 'yes')
    args.raft_mode = bool(args.raft_mode)
    
    logger.info(f"  [RAFT] raft_mode={args.raft_mode} (type={type(args.raft_mode).__name__}), free_stream_ratio={args.free_stream_ratio}")
    if args.raft_mode:
        logger.info(f"  [RAFT] L2 混合模式 Enabled, free_stream_ratio={args.free_stream_ratio}")
    else:
        logger.info(f"  [RAFT] L2 混合模式 Disabled")
    
    # =========================================================================
    # 6. DataLoader
    # =========================================================================
    logger.info("\n[DATA] Loading dataset...")
    args.dataset_config = args.config
    dataloader = create_dataloader(args)
    logger.info(f"  Dataset size: {len(dataloader)} batches")
    
    # =========================================================================
    # 7. Optimizer and Scheduler
    # =========================================================================
    logger.info("\n[OPT] Setting up optimizer...")
    logger.info(f"  Type: {args.optimizer_type}, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}")
    
    if args.optimizer_type == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            logger.info("  Using AdamW8bit optimizer")
        except ImportError:
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            logger.info("  [Fallback] bitsandbytes not found, using standard AdamW")
    elif args.optimizer_type == "Adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay,
            scale_parameter=False, relative_step=False
        )
        logger.info("  Using Adafactor optimizer")
    else:  # AdamW
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        logger.info("  Using standard AdamW optimizer")
    
    max_train_steps = len(dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num Batches per Epoch = {len(dataloader)}")
    logger.info(f"  Gradient Accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total Optimization Steps = {max_train_steps}")
    
    # Prepare with accelerator
    optimizer, dataloader, lr_scheduler = accelerator.prepare(
        optimizer, dataloader, lr_scheduler
    )
    
    # =========================================================================
    # 8. Training Loop
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("[TARGET] Starting training")
    logger.info("=" * 60)
    
    global_step = 0
    ema_loss = None
    ema_decay = 0.99
    
    for epoch in range(args.num_train_epochs):
        if _interrupted:
            logger.info("[EXIT] Training interrupted by user")
            break
            
        logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=True)):
            if _interrupted:
                break
                
            with accelerator.accumulate(transformer):
                # Get data
                latents = batch['latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']
                vl_embed = [v.to(accelerator.device, dtype=weight_dtype) for v in vl_embed]
                
                batch_size = latents.shape[0]
                
                # Generate noise
                noise = torch.randn_like(latents)
                
                # AC-RF sampling (timestep with jitter)
                noisy_latents, timesteps, target_velocity = acrf_trainer.sample_batch(
                    latents, noise, jitter_scale=args.jitter_scale
                )
                
                # Latent jitter (optional)
                if args.latent_jitter_scale > 0:
                    latent_jitter = torch.randn_like(noisy_latents) * args.latent_jitter_scale
                    noisy_latents = noisy_latents + latent_jitter
                    target_velocity = noise - latents
                
                # Prepare model input - Z-Image expects List[Tensor(C, 1, H, W)]
                model_input = noisy_latents.unsqueeze(2)
                
                # CRITICAL: For frozen model + checkpointing, input must have requires_grad=True
                if args.gradient_checkpointing:
                    model_input.requires_grad_(True)
                
                model_input_list = list(model_input.unbind(dim=0))
                
                # Timestep normalization (Z-Image uses (1000-t)/1000)
                timesteps_normalized = (1000 - timesteps) / 1000.0
                timesteps_normalized = timesteps_normalized.to(dtype=weight_dtype)
                
                # Forward pass
                model_pred_list = transformer(
                    x=model_input_list,
                    t=timesteps_normalized,
                    cap_feats=vl_embed,
                )[0]
                
                # Stack outputs
                model_pred = torch.stack(model_pred_list, dim=0)
                model_pred = model_pred.squeeze(2)
                
                # Z-Image output is negated
                model_pred = -model_pred
                
                # =========================================================
                # Compute Losses
                # =========================================================
                
                # L1 Loss
                l1_loss_val = F.l1_loss(model_pred, target_velocity)
                loss = args.lambda_l1 * l1_loss_val
                loss_components = {'l1': l1_loss_val.item()}
                
                # Cosine Loss
                cos_loss_val = 0.0
                if args.lambda_cosine > 0:
                    cos_loss = 1 - F.cosine_similarity(
                        model_pred.flatten(1), target_velocity.flatten(1), dim=1
                    ).mean()
                    loss = loss + args.lambda_cosine * cos_loss
                    cos_loss_val = cos_loss.item()
                loss_components['cosine'] = cos_loss_val
                
                # Frequency Loss (requires noisy_latents and timesteps)
                freq_loss_val = 0.0
                if freq_loss_fn and args.lambda_freq > 0:
                    freq_loss = freq_loss_fn(model_pred, target_velocity, noisy_latents, timesteps, num_train_timesteps=1000)
                    loss = loss + args.lambda_freq * freq_loss
                    freq_loss_val = freq_loss.item()
                loss_components['freq'] = freq_loss_val
                
                # Style-Structure Loss (requires noisy_latents and timesteps)
                style_loss_val = 0.0
                if style_loss_fn and args.lambda_style > 0:
                    style_loss = style_loss_fn(model_pred, target_velocity, noisy_latents, timesteps, num_train_timesteps=1000)
                    loss = loss + args.lambda_style * style_loss
                    style_loss_val = style_loss.item()
                loss_components['style'] = style_loss_val
                
                # === RAFT: L2 混合模式 (锚点流 + 自由流) ===
                l2_loss_val = 0.0
                raft_mode = getattr(args, 'raft_mode', False)
                free_stream_ratio = getattr(args, 'free_stream_ratio', 0.3)
                
                # 第一步调试日志
                if global_step == 0:
                    logger.info(f"  [RAFT DEBUG] In loop: raft_mode={raft_mode}, free_stream_ratio={free_stream_ratio}")
                
                if raft_mode and free_stream_ratio > 0:
                    # 自由流: 全时间步均匀随机采样
                    free_sigmas = torch.rand(batch_size, device=latents.device, dtype=weight_dtype)
                    # Z-Image shift 变换
                    shift = args.shift if hasattr(args, 'shift') else 3.0
                    free_sigmas = (free_sigmas * shift) / (1 + (shift - 1) * free_sigmas)
                    free_sigmas = free_sigmas.clamp(0.001, 0.999)
                    
                    # 构造自由流加噪 latents
                    sigma_bc = free_sigmas.view(batch_size, 1, 1, 1)
                    free_noisy = sigma_bc * noise + (1 - sigma_bc) * latents
                    free_target = noise - latents  # v-prediction
                    
                    # 自由流前向传播 (参与梯度)
                    free_input = free_noisy.unsqueeze(2)
                    if args.gradient_checkpointing:
                        free_input.requires_grad_(True)
                    free_input_list = list(free_input.unbind(dim=0))
                    
                    free_t = 1000 * free_sigmas  # 转回 timestep
                    free_t_norm = (1000 - free_t) / 1000.0
                    free_t_norm = free_t_norm.to(dtype=weight_dtype)
                    
                    free_pred_list = transformer(
                        x=free_input_list,
                        t=free_t_norm,
                        cap_feats=vl_embed,
                    )[0]
                    
                    free_pred = torch.stack(free_pred_list, dim=0).squeeze(2)
                    
                    # Z-Image output is negated (与锚点流一致)
                    free_pred = -free_pred
                    
                    # 自由流 L2 损失
                    l2_loss = F.mse_loss(free_pred, free_target)
                    loss = loss + args.free_stream_ratio * l2_loss
                    l2_loss_val = l2_loss.item()
                loss_components['L2'] = l2_loss_val
                
                # SNR weighting
                snr_weights = compute_snr_weights(
                    timesteps,
                    snr_gamma=args.snr_gamma,
                    snr_floor=args.snr_floor,
                )
                snr_weights = snr_weights.to(device=loss.device, dtype=weight_dtype)
                loss = loss * snr_weights.mean()
                
                # NaN check
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[NaN] Loss is NaN/Inf at step {global_step}, skipping backward. Components: {loss_components}")
                    optimizer.zero_grad()
                    continue
                
                # Cast loss to float32 for stable backward
                loss = loss.float()
                
                # Backward pass with error handling
                try:
                    accelerator.backward(loss)
                except RuntimeError as e:
                    logger.error(f"[BACKWARD ERROR] Step {global_step}, Loss={loss.item():.4f}")
                    logger.error(f"  Components: {loss_components}")
                    logger.error(f"  Error: {e}")
                    # Check for OOM
                    if "out of memory" in str(e).lower():
                        logger.error("  [OOM] GPU out of memory. Try reducing batch_size or enabling blocks_to_swap.")
                    raise
                
                # Memory optimization step
                if memory_optimizer:
                    memory_optimizer.optimize_training_step()
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update EMA loss
                current_loss = loss.item()
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * current_loss
                
                if accelerator.sync_gradients:
                    global_step += 1
                    
                    # Get current learning rate
                    current_lr = lr_scheduler.get_last_lr()[0]
                    
                    # Print progress for frontend parsing (CRITICAL: exact format required)
                    l1 = loss_components.get('l1', 0)
                    cosine = loss_components.get('cosine', 0)
                    freq = loss_components.get('freq', 0)
                    style = loss_components.get('style', 0)
                    l2 = loss_components.get('L2', 0)
                    print(f"[STEP] {global_step}/{max_train_steps} epoch={epoch+1}/{args.num_train_epochs} loss={current_loss:.4f} ema={ema_loss:.4f} l1={l1:.4f} cos={cosine:.4f} freq={freq:.4f} style={style:.4f} L2={l2:.4f} lr={current_lr:.2e}", flush=True)
        
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
    
    logger.info("\n[DONE] Training complete!")


if __name__ == "__main__":
    main()
