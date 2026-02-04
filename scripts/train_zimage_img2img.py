"""
Z-Image Img2Img Training Script

基于 train_zimage_v2.py 的图像转换训练脚本。
支持使用源图像和目标图像对进行训练。

数据集格式 (预缓存):
    {name}_{WxH}_zi_img2img.safetensors - 包含 target_latents + source_latents
    {name}_zi_te.safetensors - Text Encoder 输出

关键特性:
- 使用 diffusers 官方 ZImageTransformer2DModel
- Strength 采样策略：训练时随机采样 strength
- 与 Img2Img Pipeline 行为一致

Usage:
    accelerate launch --mixed_precision bf16 scripts/train_zimage_img2img.py \
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
from torch.utils.data import DataLoader

# Local imports
from zimage_trainer.networks.lora import LoRANetwork, ZIMAGE_TARGET_NAMES, ZIMAGE_ADALN_NAMES, EXCLUDE_PATTERNS
from zimage_trainer.dataset.dataloader import Img2ImgDataset, BucketBatchSampler, collate_fn
from zimage_trainer.acrf_trainer import ACRFTrainer
from zimage_trainer.utils.snr_utils import compute_snr_weights
from zimage_trainer.utils.model_hooks import apply_all_optimizations

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
    parser.add_argument("--train_adaln", type=bool, default=False)
    
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
    
    # Memory optimization
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--attention_backend", type=str, default="flash")
    
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
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        lora_cfg = config.get("lora", {})
        img2img_cfg = config.get("img2img", {})
        acrf_cfg = config.get("acrf", {})
        advanced_cfg = config.get("advanced", {})
        
        # Model paths
        args.dit = general_cfg.get("dit") or model_cfg.get("dit") or args.dit
        args.vae = general_cfg.get("vae") or model_cfg.get("vae") or args.vae
        args.output_dir = general_cfg.get("output_dir") or model_cfg.get("output_dir") or args.output_dir
        
        # Img2Img specific
        args.strength_min = img2img_cfg.get("strength_min", args.strength_min)
        args.strength_max = img2img_cfg.get("strength_max", args.strength_max)
        
        # LoRA
        args.network_dim = lora_cfg.get("network_dim", args.network_dim)
        args.network_alpha = lora_cfg.get("network_alpha", args.network_alpha)
        args.resume_lora = lora_cfg.get("resume_lora", args.resume_lora)
        args.train_adaln = lora_cfg.get("train_adaln", args.train_adaln)
        
        # Training
        args.output_name = training_cfg.get("output_name", args.output_name)
        args.num_train_epochs = training_cfg.get("num_train_epochs", args.num_train_epochs)
        args.learning_rate = training_cfg.get("learning_rate", args.learning_rate)
        args.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", args.gradient_accumulation_steps)
        args.seed = training_cfg.get("seed", advanced_cfg.get("seed", args.seed))
        args.save_every_n_epochs = advanced_cfg.get("save_every_n_epochs", args.save_every_n_epochs)
        args.gradient_checkpointing = training_cfg.get("gradient_checkpointing", args.gradient_checkpointing)
        
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
        args.snr_floor = acrf_cfg.get("snr_floor", args.snr_floor)
        
        # Memory
        args.blocks_to_swap = advanced_cfg.get("blocks_to_swap", args.blocks_to_swap)
        args.attention_backend = advanced_cfg.get("attention_backend", args.attention_backend)
        
        # Optimizer
        args.optimizer_type = training_cfg.get("optimizer_type", args.optimizer_type)
        args.weight_decay = training_cfg.get("weight_decay", args.weight_decay)
        
        # Scheduler
        args.lr_scheduler = training_cfg.get("lr_scheduler", args.lr_scheduler)
        args.lr_warmup_steps = training_cfg.get("lr_warmup_steps", args.lr_warmup_steps)
        args.lr_num_cycles = training_cfg.get("lr_num_cycles", args.lr_num_cycles)
    
    return args


def create_img2img_dataloader(args) -> DataLoader:
    """创建 Img2Img 专用 DataLoader"""
    import toml
    
    config = toml.load(args.config)
    dataset_config = config.get('dataset', {})
    datasets = dataset_config.get('sources', [])
    
    if not datasets:
        cache_dir = dataset_config.get('cache_directory')
        if cache_dir:
            datasets = [{'cache_directory': cache_dir, 'num_repeats': 1}]
    
    if not datasets:
        raise ValueError("No datasets configured for Img2Img training")
    
    batch_size = dataset_config.get('batch_size', 4)
    num_workers = dataset_config.get('num_workers', 4)
    max_sequence_length = dataset_config.get('max_sequence_length', 512)
    enable_bucket = dataset_config.get('enable_bucket', True)
    
    dataset = Img2ImgDataset(
        datasets=datasets,
        max_sequence_length=max_sequence_length,
        cache_arch='zi',
    )
    
    def img2img_collate_fn(batch):
        """Img2Img 专用 collate"""
        latents = torch.stack([item['latents'] for item in batch])
        source_latents = torch.stack([item['source_latents'] for item in batch])
        vl_embeds = [item['vl_embed'] for item in batch]
        return {
            'latents': latents,
            'source_latents': source_latents,
            'vl_embed': vl_embeds,
        }
    
    if enable_bucket and hasattr(dataset, 'resolutions'):
        batch_sampler = BucketBatchSampler(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                               collate_fn=img2img_collate_fn, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               collate_fn=img2img_collate_fn, pin_memory=True, drop_last=True)
    
    return dataloader


def sample_strength(batch_size: int, strength_min: float, strength_max: float, device: torch.device) -> torch.Tensor:
    """随机采样 strength 值 (uniform)"""
    return torch.rand(batch_size, device=device) * (strength_max - strength_min) + strength_min


def scale_noise_for_img2img(latents: torch.Tensor, noise: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    """
    Img2Img 加噪方式: z_t = (1 - sigma) * latents + sigma * noise
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
    
    if args.seed is not None and args.seed >= 0:
        set_seed(args.seed)
        logger.info(f"🎲 固定种子: {args.seed}")
    
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
    # 1. Load Transformer (使用 diffusers 官方版本)
    # =========================================================================
    logger.info("\n[1/6] 加载 Transformer...")
    
    from diffusers import ZImageTransformer2DModel
    logger.info("  ✓ 使用 diffusers ZImageTransformer2DModel")
    
    transformer = ZImageTransformer2DModel.from_pretrained(
        args.dit,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    transformer = transformer.to(accelerator.device)
    
    # =========================================================================
    # 2. 应用优化 (通过 Hook)
    # =========================================================================
    optimization_results = apply_all_optimizations(
        transformer,
        blocks_to_swap=args.blocks_to_swap,
        attention_backend=args.attention_backend,
        gradient_checkpointing=args.gradient_checkpointing,
        device=accelerator.device,
        verbose=True,
    )
    block_swapper = optimization_results.get("block_swapper")
    
    transformer.train()
    
    # =========================================================================
    # 3. Apply LoRA
    # =========================================================================
    logger.info(f"\n[2/6] 创建 LoRA (rank={args.network_dim})...")
    
    # 动态构建 target_names
    target_names = list(ZIMAGE_TARGET_NAMES)
    exclude_patterns = list(EXCLUDE_PATTERNS)
    
    train_adaln = getattr(args, 'train_adaln', False)
    if isinstance(train_adaln, str):
        train_adaln = train_adaln.lower() in ('true', '1', 'yes')
    
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
    acrf_trainer.verify_setup()
    
    # =========================================================================
    # 5. DataLoader
    # =========================================================================
    logger.info("\n[4/6] 加载数据集...")
    
    dataloader = create_img2img_dataloader(args)
    logger.info(f"  ✓ {len(dataloader)} batches")
    
    # =========================================================================
    # 6. Optimizer and Scheduler
    # =========================================================================
    logger.info("\n[5/6] 配置优化器...")
    
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
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)
    
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
        output_base = os.environ.get("OUTPUT_PATH", "")
        if not output_base:
            output_base = os.path.dirname(args.output_dir)
        logging_dir = os.path.join(output_base, "logs", args.output_name)
        os.makedirs(logging_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=logging_dir)
    
    logger.info(f"  ✓ 训练轮数: {args.num_train_epochs}, 总步数: {max_train_steps}")
    
    # =========================================================================
    # 7. Training Loop
    # =========================================================================
    logger.info("\n[6/6] 开始训练...")
    logger.info("=" * 60)
    
    global_step = 0
    ema_loss = None
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
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)):
            if _interrupted:
                break
            
            with accelerator.accumulate(transformer):
                # Get data
                target_latents = batch['latents'].to(accelerator.device, dtype=weight_dtype)
                source_latents = batch['source_latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']
                vl_embed = [v.to(accelerator.device, dtype=weight_dtype) for v in vl_embed]
                
                batch_size = target_latents.shape[0]
                
                # Generate noise
                noise = torch.randn_like(target_latents)
                
                # 随机采样 strength
                strength = sample_strength(batch_size, args.strength_min, args.strength_max, accelerator.device)
                
                # Img2Img 专用加噪：从 source 出发
                # 加噪目标是 target_latents，但起点是 source_latents
                noisy_latents = scale_noise_for_img2img(source_latents, noise, strength)
                
                # 计算 timestep (strength -> timestep)
                timesteps = (strength * 1000).long()
                
                # 目标 velocity: 从 noisy 到 target
                target_velocity = noise - target_latents
                
                # Prepare model input
                model_input = noisy_latents.unsqueeze(2)
                if args.gradient_checkpointing:
                    model_input.requires_grad_(True)
                model_input_list = list(model_input.unbind(dim=0))
                
                # Timestep normalization
                timesteps_normalized = (1000 - timesteps.float()) / 1000.0
                timesteps_normalized = timesteps_normalized.to(dtype=weight_dtype)
                
                # Forward pass
                model_pred_list = transformer(
                    x=model_input_list,
                    t=timesteps_normalized,
                    cap_feats=vl_embed,
                )[0]
                
                model_pred = torch.stack(model_pred_list, dim=0).squeeze(2)
                model_pred = -model_pred  # Z-Image output is negated
                
                # Compute losses
                l1_loss = F.l1_loss(model_pred, target_velocity)
                loss = args.lambda_l1 * l1_loss
                
                if args.lambda_cosine > 0:
                    cos_loss = 1 - F.cosine_similarity(
                        model_pred.flatten(1), target_velocity.flatten(1), dim=1
                    ).mean()
                    loss = loss + args.lambda_cosine * cos_loss
                
                # SNR weighting
                snr_weights = compute_snr_weights(
                    timesteps=timesteps,
                    num_train_timesteps=1000,
                    snr_gamma=args.snr_gamma,
                    snr_floor=args.snr_floor,
                    prediction_type="v_prediction",
                )
                snr_mean = snr_weights.mean().to(device=loss.device, dtype=weight_dtype)
                loss = loss * snr_mean
                
                # NaN check
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[NaN] Loss is NaN/Inf at step {global_step}, skipping")
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
                
                # Update EMA loss
                loss_val = loss.detach().item()
                if ema_loss is None:
                    ema_loss = loss_val
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * loss_val
                
                current_lr = lr_scheduler.get_last_lr()[0]
                
                if accelerator.is_main_process:
                    print(f"[STEP] {global_step}/{max_train_steps} epoch={epoch+1}/{args.num_train_epochs} "
                          f"loss={loss_val:.4f} ema={ema_loss:.4f} lr={current_lr:.2e}", flush=True)
                    
                    if writer:
                        writer.add_scalar("train/loss", loss_val, global_step)
                        writer.add_scalar("train/ema_loss", ema_loss, global_step)
                        writer.add_scalar("train/learning_rate", current_lr, global_step)
        
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
    
    # Cleanup
    if block_swapper:
        block_swapper.remove_hooks()
    
    if writer:
        writer.close()
    
    logger.info("\n[DONE] Img2Img Training complete!")


if __name__ == "__main__":
    main()
