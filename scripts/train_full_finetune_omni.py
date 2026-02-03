"""
Z-Image Full Finetune Omni Multi-Image Training Script

全量微调版本的多图条件训练。
支持使用多个条件图像和 SigLIP 视觉特征进行全量微调训练。

注意事项:
- 显存需求极高（约40GB+），推荐使用 A100/H100
- 输出为完整模型权重（非 LoRA）
- 使用 diffusers 官方 ZImageTransformer2DModel (支持 Omni 模式)

Usage:
    accelerate launch --mixed_precision bf16 scripts/train_full_finetune_omni.py \
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
from safetensors.torch import save_file

# Local imports
from zimage_trainer.dataset.dataloader import OmniDataset, BucketBatchSampler
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
    parser = argparse.ArgumentParser(description="Z-Image Full Finetune Omni Training")
    parser.add_argument("--config", type=str, required=True, help="TOML config path")
    
    # Model paths
    parser.add_argument("--dit", type=str, default=None)
    parser.add_argument("--vae", type=str, default=None)
    parser.add_argument("--siglip", type=str, default=None)
    
    # Training params
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_name", type=str, default="finetune_omni")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    
    # Finetune 模块选择
    parser.add_argument("--trainable_modules", type=str, default="attention+mlp+adaln")
    parser.add_argument("--freeze_embeddings", type=bool, default=True)
    
    # Omni specific
    parser.add_argument("--max_condition_images", type=int, default=4)
    parser.add_argument("--freeze_siglip", type=bool, default=True)
    parser.add_argument("--condition_cache_dir", type=str, default=None)
    
    # AC-RF / Turbo
    parser.add_argument("--turbo_steps", type=int, default=10)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--use_dynamic_shift", type=bool, default=True)
    parser.add_argument("--jitter_scale", type=float, default=0.02)
    parser.add_argument("--enable_turbo", type=bool, default=True)
    
    # Loss weights
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_cosine", type=float, default=0.0)
    
    # SNR
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--snr_floor", type=float, default=0.1)
    
    # Memory optimization
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--attention_backend", type=str, default="flash")
    
    # Optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW8bit")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
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
        finetune_cfg = config.get("finetune", {})
        omni_cfg = config.get("omni", {})
        acrf_cfg = config.get("acrf", {})
        advanced_cfg = config.get("advanced", {})
        
        # Model paths
        args.dit = general_cfg.get("dit") or model_cfg.get("dit") or args.dit
        args.vae = general_cfg.get("vae") or model_cfg.get("vae") or args.vae
        args.siglip = omni_cfg.get("siglip") or model_cfg.get("siglip") or args.siglip
        args.output_dir = general_cfg.get("output_dir") or model_cfg.get("output_dir") or args.output_dir
        
        # Finetune specific
        args.trainable_modules = finetune_cfg.get("trainable_modules", args.trainable_modules)
        args.freeze_embeddings = finetune_cfg.get("freeze_embeddings", args.freeze_embeddings)
        
        # Omni specific
        args.max_condition_images = omni_cfg.get("max_condition_images", args.max_condition_images)
        args.freeze_siglip = omni_cfg.get("freeze_siglip", args.freeze_siglip)
        args.condition_cache_dir = omni_cfg.get("condition_cache_dir", args.condition_cache_dir)
        
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


def create_omni_dataloader(args) -> DataLoader:
    """创建 Omni 专用 DataLoader"""
    import toml
    
    config = toml.load(args.config)
    dataset_config = config.get('dataset', {})
    datasets = dataset_config.get('sources', [])
    
    if not datasets:
        cache_dir = dataset_config.get('cache_directory')
        if cache_dir:
            datasets = [{'cache_directory': cache_dir, 'num_repeats': 1}]
    
    if not datasets:
        raise ValueError("No datasets configured for Omni training")
    
    batch_size = dataset_config.get('batch_size', 4)
    num_workers = dataset_config.get('num_workers', 4)
    max_sequence_length = dataset_config.get('max_sequence_length', 512)
    enable_bucket = dataset_config.get('enable_bucket', True)
    condition_cache_dir = args.condition_cache_dir
    
    dataset = OmniDataset(
        datasets=datasets,
        max_sequence_length=max_sequence_length,
        cache_arch='zi',
        condition_cache_dir=condition_cache_dir,
    )
    
    def omni_collate_fn(batch):
        latents = torch.stack([item['latents'] for item in batch])
        vl_embeds = [item['vl_embed'] for item in batch]
        result = {'latents': latents, 'vl_embed': vl_embeds}
        if 'siglip_feats' in batch[0]:
            result['siglip_feats'] = [item.get('siglip_feats') for item in batch]
        return result
    
    if enable_bucket and hasattr(dataset, 'resolutions'):
        batch_sampler = BucketBatchSampler(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                               collate_fn=omni_collate_fn, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               collate_fn=omni_collate_fn, pin_memory=True, drop_last=True)
    
    return dataloader


def get_trainable_parameters(transformer, trainable_modules: str, freeze_embeddings: bool):
    """选择性解冻模块"""
    transformer.requires_grad_(False)
    
    trainable_params = []
    trainable_count = 0
    frozen_count = 0
    
    modules_to_train = set(trainable_modules.lower().replace(' ', '').split('+'))
    train_all = 'all' in modules_to_train
    train_attention = 'attention' in modules_to_train or train_all
    train_mlp = 'mlp' in modules_to_train or train_all
    train_adaln = 'adaln' in modules_to_train or train_all
    train_norm = 'norm' in modules_to_train or train_all
    
    for name, param in transformer.named_parameters():
        should_train = False
        name_lower = name.lower()
        
        if train_attention:
            if any(key in name_lower for key in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'to_q', 'to_k', 'to_v', 'to_out']):
                should_train = True
        
        if train_mlp:
            if any(key in name_lower for key in ['mlp', 'fc1', 'fc2', 'ffn', 'feed_forward', 'linear1', 'linear2']):
                should_train = True
        
        if train_adaln:
            if any(key in name_lower for key in ['adaln', 'scale_shift', 'modulation', 't_embedder', 'time_embed']):
                should_train = True
        
        if train_norm:
            if any(key in name_lower for key in ['norm', 'ln', 'layer_norm', 'layernorm', 'rmsnorm']):
                if not any(key in name_lower for key in ['adaln']):
                    should_train = True
        
        if freeze_embeddings:
            if any(key in name_lower for key in ['embed', 'embedding', 'pos_embed', 'patch_embed', 'x_embedder', 'cap_embedder']):
                should_train = False
        
        if should_train:
            param.requires_grad = True
            trainable_params.append(param)
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()
    
    return trainable_params, frozen_count, trainable_count


def save_transformer_weights(transformer, path: str, dtype=torch.bfloat16):
    """保存 Transformer 完整权重"""
    state_dict = transformer.state_dict()
    converted_state = {key: value.to(dtype).cpu() for key, value in state_dict.items()}
    save_file(converted_state, path)
    logger.info(f"[SAVE] 已保存完整模型 ({len(converted_state)} 个参数) 到 {path}")


def main():
    global _interrupted
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    if args.seed is not None and args.seed >= 0:
        set_seed(args.seed)
        logger.info(f"🎲 固定种子: {args.seed}")
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    logger.info("\n" + "=" * 60)
    logger.info("🌌 Z-Image Full Finetune Omni Multi-Image Training")
    logger.info("=" * 60)
    logger.info(f"📁 输出: {args.output_dir}/{args.output_name}")
    logger.info(f"🖼️ 最大条件图: {args.max_condition_images}")
    logger.warning("⚠️ 全量微调需要约 40GB+ 显存")
    
    # =========================================================================
    # 1. Load Transformer (使用 diffusers 官方版本，支持 Omni)
    # =========================================================================
    logger.info("\n[1/7] 加载 Transformer...")
    
    from diffusers import ZImageTransformer2DModel
    logger.info("  ✓ 使用 diffusers ZImageTransformer2DModel (Omni 支持)")
    
    transformer = ZImageTransformer2DModel.from_pretrained(
        args.dit,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    transformer = transformer.to(accelerator.device)
    
    has_omni_support = hasattr(transformer.config, 'siglip_feat_dim') and transformer.config.siglip_feat_dim is not None
    if has_omni_support:
        logger.info(f"  ✓ Omni 模式已启用 (siglip_feat_dim={transformer.config.siglip_feat_dim})")
    else:
        logger.warning("  ⚠ 模型不支持 Omni 模式，将使用标准训练模式")
    
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
    
    # =========================================================================
    # 3. Load SigLIP Vision Encoder (可选)
    # =========================================================================
    logger.info("\n[2/7] 加载 SigLIP Vision Encoder...")
    
    siglip = None
    
    if args.siglip and has_omni_support:
        try:
            from transformers import SiglipVisionModel
            siglip = SiglipVisionModel.from_pretrained(args.siglip, torch_dtype=weight_dtype)
            siglip = siglip.to(accelerator.device)
            
            freeze_siglip = args.freeze_siglip
            if isinstance(freeze_siglip, str):
                freeze_siglip = freeze_siglip.lower() in ('true', '1', 'yes')
            
            if freeze_siglip:
                siglip.requires_grad_(False)
                siglip.eval()
                logger.info("  [FREEZE] SigLIP 已冻结")
            
            logger.info(f"  ✓ 加载 SigLIP: {args.siglip}")
        except Exception as e:
            logger.warning(f"  ⚠ SigLIP 加载失败: {e}")
            siglip = None
    else:
        if not has_omni_support:
            logger.info("  跳过 (模型不支持 Omni)")
        else:
            logger.warning("  ⚠ 未指定 SigLIP 模型")
    
    # =========================================================================
    # 4. 选择性模块训练
    # =========================================================================
    logger.info(f"\n[3/7] 配置可训练模块 ({args.trainable_modules})...")
    
    trainable_params, frozen_count, trainable_count = get_trainable_parameters(
        transformer, args.trainable_modules, args.freeze_embeddings
    )
    
    total_params = frozen_count + trainable_count
    logger.info(f"  ✓ 可训练: {trainable_count:,} ({trainable_count/1e6:.2f}M, {100*trainable_count/total_params:.1f}%)")
    logger.info(f"  ✓ 冻结: {frozen_count:,} ({frozen_count/1e6:.2f}M)")
    
    transformer.train()
    
    # =========================================================================
    # 5. AC-RF Trainer
    # =========================================================================
    logger.info("\n[4/7] 初始化 AC-RF Trainer...")
    
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
    # 6. DataLoader
    # =========================================================================
    logger.info("\n[5/7] 加载数据集...")
    
    dataloader = create_omni_dataloader(args)
    logger.info(f"  ✓ {len(dataloader)} batches")
    
    # =========================================================================
    # 7. Optimizer and Scheduler
    # =========================================================================
    logger.info("\n[6/7] 配置优化器...")
    
    if args.learning_rate > 1e-5:
        logger.warning(f"  ⚠️ 全量微调建议使用较低学习率 (当前: {args.learning_rate})")
    
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
    
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)
    
    max_train_steps = len(dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    writer = None
    if accelerator.is_main_process:
        output_parent = os.path.dirname(args.output_dir)
        logging_dir = os.path.join(output_parent, "logs", args.output_name)
        os.makedirs(logging_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=logging_dir)
    
    logger.info(f"  ✓ 训练轮数: {args.num_train_epochs}, 总步数: {max_train_steps}")
    
    # =========================================================================
    # 8. Training Loop
    # =========================================================================
    logger.info("\n[7/7] 开始训练...")
    logger.info("=" * 60)
    
    global_step = 0
    ema_loss = None
    ema_decay = 0.99
    
    for epoch in range(args.num_train_epochs):
        if _interrupted:
            logger.info("[EXIT] Training interrupted by user")
            if accelerator.is_main_process and global_step > 0:
                emergency_path = Path(args.output_dir) / f"{args.output_name}_interrupted_step{global_step}.safetensors"
                save_transformer_weights(transformer, str(emergency_path), dtype=weight_dtype)
            break
        
        logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)):
            if _interrupted:
                break
            
            with accelerator.accumulate(transformer):
                latents = batch['latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']
                vl_embed = [v.to(accelerator.device, dtype=weight_dtype) for v in vl_embed]
                
                batch_size = latents.shape[0]
                
                siglip_feats = batch.get('siglip_feats')
                if siglip_feats is not None:
                    siglip_feats = [s.to(accelerator.device, dtype=weight_dtype) if s is not None else None 
                                   for s in siglip_feats]
                
                noise = torch.randn_like(latents)
                
                noisy_latents, timesteps, target_velocity = acrf_trainer.sample_batch(
                    latents, noise, jitter_scale=args.jitter_scale, use_anchor=args.enable_turbo
                )
                
                model_input = noisy_latents.unsqueeze(2)
                if args.gradient_checkpointing:
                    model_input.requires_grad_(True)
                model_input_list = list(model_input.unbind(dim=0))
                
                timesteps_normalized = (1000 - timesteps) / 1000.0
                timesteps_normalized = timesteps_normalized.to(dtype=weight_dtype)
                
                # Forward pass
                if has_omni_support and siglip_feats is not None:
                    x_omni = [[img] for img in model_input_list]
                    cap_feats_omni = [[cap] for cap in vl_embed]
                    siglip_feats_omni = [[sf] if sf is not None else None for sf in siglip_feats]
                    image_noise_mask = [[1] for _ in range(batch_size)]
                    
                    model_pred_list = transformer(
                        x=x_omni,
                        t=timesteps_normalized,
                        cap_feats=cap_feats_omni,
                        siglip_feats=siglip_feats_omni,
                        image_noise_mask=image_noise_mask,
                    )[0]
                else:
                    model_pred_list = transformer(
                        x=model_input_list,
                        t=timesteps_normalized,
                        cap_feats=vl_embed,
                    )[0]
                
                model_pred = torch.stack(model_pred_list, dim=0).squeeze(2)
                model_pred = -model_pred
                
                l1_loss = F.l1_loss(model_pred, target_velocity)
                loss = args.lambda_l1 * l1_loss
                
                if args.lambda_cosine > 0:
                    cos_loss = 1 - F.cosine_similarity(
                        model_pred.flatten(1), target_velocity.flatten(1), dim=1
                    ).mean()
                    loss = loss + args.lambda_cosine * cos_loss
                
                snr_weights = compute_snr_weights(
                    timesteps=timesteps,
                    num_train_timesteps=1000,
                    snr_gamma=args.snr_gamma,
                    snr_floor=args.snr_floor,
                    prediction_type="v_prediction",
                )
                snr_mean = snr_weights.mean().to(device=loss.device, dtype=weight_dtype)
                loss = loss * snr_mean
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[NaN] Loss is NaN/Inf at step {global_step}, skipping")
                    optimizer.zero_grad()
                    continue
                
                loss = loss.float()
                accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
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
        
        if accelerator.is_main_process and (epoch + 1) % args.save_every_n_epochs == 0:
            save_path = Path(args.output_dir) / f"{args.output_name}_epoch{epoch+1}.safetensors"
            save_transformer_weights(transformer, str(save_path), dtype=weight_dtype)
    
    if accelerator.is_main_process:
        final_path = Path(args.output_dir) / f"{args.output_name}_final.safetensors"
        save_transformer_weights(transformer, str(final_path), dtype=weight_dtype)
    
    if block_swapper:
        block_swapper.remove_hooks()
    
    if writer:
        writer.close()
    
    logger.info("\n[DONE] Full Finetune Omni Training complete!")


if __name__ == "__main__":
    main()
