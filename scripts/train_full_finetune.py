"""
[FULL FEATURE] Full Fine-tune Training Script for Z-Image-Turbo

全量微调脚本 - 训练 Transformer 的全部/部分参数
⚠️ 显存需求高：建议 24GB+ VRAM，推荐 48GB+

关键特性：
- 全参数微调，不使用 LoRA
- 支持选择性模块训练 (attention/mlp/norm)
- 与 LoRA 脚本功能一致：Freq/Style Loss、时间步感知、RAFT 模式
- 强制启用梯度检查点以节省显存

Usage:
    accelerate launch --mixed_precision bf16 scripts/train_full_finetune.py \\
        --config config/full_finetune_config.toml
"""

import os
import sys
import math
import gc
import signal
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file
from diffusers.optimization import get_scheduler

from zimage_trainer.acrf_trainer import ACRFTrainer
from zimage_trainer.utils.zimage_utils import load_transformer
from zimage_trainer.dataset.dataloader import create_dataloader, create_reg_dataloader, get_reg_config
from zimage_trainer.utils.snr_utils import compute_snr_weights
from zimage_trainer.utils.l2_scheduler import create_l2_scheduler_from_args
from zimage_trainer.utils.timestep_aware_loss import create_timestep_aware_scheduler_from_args
from zimage_trainer.losses.frequency_aware_loss import FrequencyAwareLoss
from zimage_trainer.losses.style_structure_loss import LatentStyleStructureLoss

import logging
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
    parser = argparse.ArgumentParser(description="Full Fine-tune 训练脚本（全量微调）")
    
    # 配置文件
    parser.add_argument("--config", type=str, help="TOML 配置文件路径")
    
    # 模型路径
    parser.add_argument("--dit", type=str, help="Transformer 模型路径")
    parser.add_argument("--dataset_config", type=str, help="数据集配置文件")
    parser.add_argument("--output_dir", type=str, default="output/full_finetune")
    parser.add_argument("--output_name", type=str, default="zimage-finetune")
    
    # AC-RF 参数
    parser.add_argument("--turbo_steps", type=int, default=10)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--jitter_scale", type=float, default=0.02)
    parser.add_argument("--latent_jitter_scale", type=float, default=0.0)
    parser.add_argument("--enable_turbo", type=bool, default=True)
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--snr_floor", type=float, default=0.1)
    
    # 全量微调专用参数
    parser.add_argument("--trainable_modules", type=str, default="attention",
                       choices=["all", "attention", "mlp", "norm"])
    parser.add_argument("--freeze_embeddings", type=bool, default=True)
    
    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    
    # Loss 参数 (与 LoRA 一致)
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_cosine", type=float, default=0.1)
    parser.add_argument("--enable_freq", type=bool, default=False)
    parser.add_argument("--lambda_freq", type=float, default=0.3)
    parser.add_argument("--alpha_hf", type=float, default=1.0)
    parser.add_argument("--beta_lf", type=float, default=0.2)
    parser.add_argument("--enable_style", type=bool, default=False)
    parser.add_argument("--lambda_style", type=float, default=0.3)
    parser.add_argument("--lambda_struct", type=float, default=1.0)
    parser.add_argument("--lambda_light", type=float, default=0.5)
    parser.add_argument("--lambda_color", type=float, default=0.3)
    parser.add_argument("--lambda_tex", type=float, default=0.5)
    
    # RAFT L2 混合模式
    parser.add_argument("--raft_mode", type=bool, default=False)
    parser.add_argument("--free_stream_ratio", type=float, default=0.3)
    parser.add_argument("--l2_schedule_mode", type=str, default="constant")
    parser.add_argument("--l2_initial_ratio", type=float, default=None)
    parser.add_argument("--l2_final_ratio", type=float, default=None)
    parser.add_argument("--l2_milestones", type=str, default="")
    
    # 时间步感知
    parser.add_argument("--enable_timestep_aware_loss", type=bool, default=False)
    parser.add_argument("--timestep_high_threshold", type=float, default=0.7)
    parser.add_argument("--timestep_low_threshold", type=float, default=0.3)
    
    # Optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW8bit")
    
    # Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    
    args = parser.parse_args()
    
    # 读取 TOML 配置
    if args.config:
        import toml
        config = toml.load(args.config)
        
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        acrf_cfg = config.get("acrf", {})
        finetune_cfg = config.get("finetune", {})
        advanced_cfg = config.get("advanced", {})
        
        # Model
        args.dit = model_cfg.get("dit", args.dit)
        args.output_dir = model_cfg.get("output_dir", args.output_dir)
        args.output_name = model_cfg.get("output_name", training_cfg.get("output_name", args.output_name))
        
        # AC-RF
        args.turbo_steps = acrf_cfg.get("turbo_steps", args.turbo_steps)
        args.shift = acrf_cfg.get("shift", args.shift)
        args.jitter_scale = acrf_cfg.get("jitter_scale", args.jitter_scale)
        args.latent_jitter_scale = acrf_cfg.get("latent_jitter_scale", args.latent_jitter_scale)
        args.enable_turbo = acrf_cfg.get("enable_turbo", args.enable_turbo)
        args.snr_gamma = training_cfg.get("snr_gamma", acrf_cfg.get("snr_gamma", args.snr_gamma))
        args.snr_floor = acrf_cfg.get("snr_floor", args.snr_floor)
        
        # Finetune
        args.trainable_modules = finetune_cfg.get("trainable_modules", args.trainable_modules)
        args.freeze_embeddings = finetune_cfg.get("freeze_embeddings", args.freeze_embeddings)
        
        # Training
        args.num_train_epochs = training_cfg.get("num_train_epochs", args.num_train_epochs)
        args.save_every_n_epochs = training_cfg.get("save_every_n_epochs", 
                                   advanced_cfg.get("save_every_n_epochs", args.save_every_n_epochs))
        args.learning_rate = training_cfg.get("learning_rate", args.learning_rate)
        args.weight_decay = training_cfg.get("weight_decay", args.weight_decay)
        args.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps",
                                           advanced_cfg.get("gradient_accumulation_steps", args.gradient_accumulation_steps))
        args.gradient_checkpointing = advanced_cfg.get("gradient_checkpointing", args.gradient_checkpointing)
        
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
        
        # RAFT
        args.raft_mode = acrf_cfg.get("raft_mode", args.raft_mode)
        args.free_stream_ratio = acrf_cfg.get("free_stream_ratio", args.free_stream_ratio)
        args.l2_schedule_mode = acrf_cfg.get("l2_schedule_mode", args.l2_schedule_mode)
        args.l2_initial_ratio = acrf_cfg.get("l2_initial_ratio", args.l2_initial_ratio)
        args.l2_final_ratio = acrf_cfg.get("l2_final_ratio", args.l2_final_ratio)
        
        # Timestep-aware
        args.enable_timestep_aware_loss = acrf_cfg.get("enable_timestep_aware_loss", 
                                          training_cfg.get("enable_timestep_aware_loss", args.enable_timestep_aware_loss))
        args.timestep_high_threshold = acrf_cfg.get("timestep_high_threshold", args.timestep_high_threshold)
        args.timestep_low_threshold = acrf_cfg.get("timestep_low_threshold", args.timestep_low_threshold)
        
        # Optimizer & Scheduler
        args.optimizer_type = training_cfg.get("optimizer_type", args.optimizer_type)
        args.lr_scheduler = training_cfg.get("lr_scheduler", args.lr_scheduler)
        args.lr_warmup_steps = training_cfg.get("lr_warmup_steps", args.lr_warmup_steps)
        args.lr_num_cycles = training_cfg.get("lr_num_cycles", args.lr_num_cycles)
    
    # 验证必要参数
    if not args.dit:
        parser.error("--dit is required")
    
    if not args.dataset_config and args.config:
        args.dataset_config = args.config
    
    # RAFT mode 类型转换
    if isinstance(args.raft_mode, str):
        args.raft_mode = args.raft_mode.lower() in ('true', '1', 'yes')
    args.raft_mode = bool(args.raft_mode)
        
    return args


def get_trainable_parameters(transformer, trainable_modules: str, freeze_embeddings: bool):
    """根据配置返回可训练的参数"""
    transformer.requires_grad_(False)
    
    trainable_params = []
    trainable_count = 0
    frozen_count = 0
    
    for name, param in transformer.named_parameters():
        should_train = False
        
        if trainable_modules == "all":
            should_train = True
        elif trainable_modules == "attention":
            if any(key in name.lower() for key in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
                should_train = True
        elif trainable_modules == "mlp":
            if any(key in name.lower() for key in ['mlp', 'fc1', 'fc2', 'ffn', 'feed_forward']):
                should_train = True
        elif trainable_modules == "norm":
            if any(key in name.lower() for key in ['norm', 'ln', 'layer_norm', 'layernorm']):
                should_train = True
        
        if freeze_embeddings and any(key in name.lower() for key in ['embed', 'embedding', 'pos_embed']):
            should_train = False
        
        if should_train:
            param.requires_grad = True
            trainable_params.append(param)
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()
    
    return trainable_params, frozen_count, trainable_count


def save_transformer_weights(transformer, save_path: Path, dtype=torch.float16):
    """保存 Transformer 权重为 safetensors 格式"""
    state_dict = {}
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.data.to(dtype).cpu()
    
    save_file(state_dict, str(save_path))
    logger.info(f"[SAVE] 保存权重: {save_path} ({len(state_dict)} tensors)")


def main():
    global _interrupted
    args = parse_args()
    
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
    
    # =========================================================================
    # 参数预览
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Z-Image Full Fine-tune Training")
    logger.info("=" * 60)
    
    logger.info(f"📁 输出: {args.output_dir}/{args.output_name}")
    logger.info(f"🎯 模式: {'Turbo (' + str(args.turbo_steps) + ' steps)' if args.enable_turbo else '标准 Flow Matching'}")
    logger.info(f"⚡ 精度: {weight_dtype}")
    
    logger.info(f"\n📋 训练参数:")
    logger.info(f"   Epochs: {args.num_train_epochs} | LR: {args.learning_rate} | Grad Accum: {args.gradient_accumulation_steps}")
    logger.info(f"   Trainable: {args.trainable_modules} | Freeze Embed: {args.freeze_embeddings}")
    logger.info(f"   Optimizer: {args.optimizer_type} | Scheduler: {args.lr_scheduler}")
    
    logger.info(f"\n⚙️ AC-RF 参数:")
    logger.info(f"   Shift: {args.shift} | Jitter: {args.jitter_scale} | Latent Jitter: {args.latent_jitter_scale}")
    logger.info(f"   SNR Gamma: {args.snr_gamma} | SNR Floor: {args.snr_floor}")
    if args.raft_mode:
        logger.info(f"   RAFT: ON (L2 ratio={args.free_stream_ratio})")
    
    loss_cfg = f"L1×{args.lambda_l1} + Cos×{args.lambda_cosine}"
    if args.enable_freq:
        loss_cfg += f" + Freq×{args.lambda_freq}(hf={args.alpha_hf},lf={args.beta_lf})"
    if args.enable_style:
        loss_cfg += f" + Style×{args.lambda_style}"
    logger.info(f"\n📊 Loss 配置:")
    logger.info(f"   {loss_cfg}")
    if args.enable_timestep_aware_loss:
        logger.info(f"   🎛 时间步感知: ON (早期重结构, 后期重纹理)")
    
    # =========================================================================
    # 1. Load Transformer
    # =========================================================================
    logger.info("\n[1/7] 加载 Transformer...")
    
    transformer = load_transformer(
        transformer_path=args.dit,
        device=accelerator.device,
        torch_dtype=weight_dtype,
    )
    logger.info(f"  ✓ 已加载: {args.dit}")
    
    # =========================================================================
    # 2. Configure Trainable Parameters
    # =========================================================================
    logger.info(f"\n[2/7] 配置可训练参数 ({args.trainable_modules})...")
    
    trainable_params, frozen_count, trainable_count = get_trainable_parameters(
        transformer, 
        args.trainable_modules, 
        args.freeze_embeddings
    )
    
    total_params = frozen_count + trainable_count
    logger.info(f"  ✓ 可训练: {trainable_count:,} ({trainable_count/1e6:.2f}M, {100*trainable_count/total_params:.1f}%)")
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing and hasattr(transformer, 'enable_gradient_checkpointing'):
        transformer.enable_gradient_checkpointing()
    
    transformer.train()
    
    # =========================================================================
    # 3. AC-RF Trainer
    # =========================================================================
    logger.info("\n[3/7] 初始化 AC-RF Trainer...")
    acrf_trainer = ACRFTrainer(
        num_train_timesteps=1000,
        turbo_steps=args.turbo_steps,
        shift=args.shift,
    )
    acrf_trainer.verify_setup()
    
    # =========================================================================
    # 4. Loss Functions
    # =========================================================================
    logger.info("\n[4/7] 初始化 Loss 函数...")
    
    freq_loss_fn = None
    if args.enable_freq:
        freq_loss_fn = FrequencyAwareLoss(
            alpha_hf=args.alpha_hf,
            beta_lf=args.beta_lf,
        )
    
    style_loss_fn = None
    if args.enable_style:
        style_loss_fn = LatentStyleStructureLoss(
            lambda_struct=args.lambda_struct,
            lambda_light=args.lambda_light,
            lambda_color=args.lambda_color,
            lambda_tex=args.lambda_tex,
        )
    
    # Timestep-aware scheduler
    timestep_aware_scheduler = create_timestep_aware_scheduler_from_args(args)
    
    # =========================================================================
    # 5. DataLoader
    # =========================================================================
    logger.info("\n[5/7] 加载数据集...")
    args.dataset_config = args.config
    dataloader = create_dataloader(args)
    logger.info(f"  ✓ {len(dataloader)} batches")
    
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
    logger.info("\n[6/7] 配置优化器...")
    logger.info(f"  ✓ {args.optimizer_type}, LR={args.learning_rate}")
    
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
    
    # Prepare with accelerator
    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)
    
    # Calculate max_train_steps
    max_train_steps = len(dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    logger.info(f"  ✓ 训练轮数: {args.num_train_epochs}, 总步数: {max_train_steps}")
    
    # =========================================================================
    # 7. Training Loop
    # =========================================================================
    logger.info("\n[7/7] 开始训练...")
    logger.info("=" * 60)
    
    # L2 scheduler
    l2_scheduler = create_l2_scheduler_from_args(args)
    
    global_step = 0
    ema_loss = None
    ema_decay = 0.99
    
    for epoch in range(args.num_train_epochs):
        if _interrupted:
            logger.info("[EXIT] Training interrupted by user")
            if accelerator.is_main_process and global_step > 0:
                emergency_path = Path(args.output_dir) / f"{args.output_name}_interrupted_step{global_step}.safetensors"
                save_transformer_weights(accelerator.unwrap_model(transformer), emergency_path, dtype=weight_dtype)
            break
        
        current_l2_ratio = l2_scheduler.get_ratio(epoch + 1) if l2_scheduler else args.free_stream_ratio
        
        if args.raft_mode:
            logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs} [L2={current_l2_ratio:.2f}]")
        else:
            logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        
        transformer.train()
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=True)):
            if _interrupted:
                if accelerator.is_main_process and global_step > 0:
                    emergency_path = Path(args.output_dir) / f"{args.output_name}_interrupted_step{global_step}.safetensors"
                    save_transformer_weights(accelerator.unwrap_model(transformer), emergency_path, dtype=weight_dtype)
                break
                
            with accelerator.accumulate(transformer):
                # Get data
                latents = batch['latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']
                vl_embed = [v.to(accelerator.device, dtype=weight_dtype) for v in vl_embed]
                
                batch_size = latents.shape[0]
                noise = torch.randn_like(latents)
                
                # AC-RF sampling
                noisy_latents, timesteps, target_velocity = acrf_trainer.sample_batch(
                    latents, noise, jitter_scale=args.jitter_scale, use_anchor=args.enable_turbo
                )
                
                # Latent jitter
                if args.latent_jitter_scale > 0:
                    latent_jitter = torch.randn_like(noisy_latents) * args.latent_jitter_scale
                    noisy_latents = noisy_latents + latent_jitter
                    target_velocity = noise - latents
                
                # Prepare model input
                model_input = noisy_latents.unsqueeze(2)
                if args.gradient_checkpointing:
                    model_input.requires_grad_(True)
                model_input_list = list(model_input.unbind(dim=0))
                
                timesteps_normalized = (1000 - timesteps) / 1000.0
                timesteps_normalized = timesteps_normalized.to(dtype=weight_dtype)
                
                # Forward pass
                model_pred_list = transformer(
                    x=model_input_list,
                    t=timesteps_normalized,
                    cap_feats=vl_embed,
                )[0]
                
                model_pred = torch.stack(model_pred_list, dim=0).squeeze(2)
                model_pred = -model_pred
                
                # =========================================================
                # Compute Losses
                # =========================================================
                
                ts_weights = None
                if timestep_aware_scheduler:
                    ts_weights = timestep_aware_scheduler.get_mean_weights(timesteps, num_train_timesteps=1000)
                
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
                
                # Frequency Loss
                freq_loss_val = 0.0
                if freq_loss_fn and args.lambda_freq > 0:
                    freq_loss = freq_loss_fn(model_pred, target_velocity, noisy_latents, timesteps, num_train_timesteps=1000)
                    freq_scale = ts_weights['lambda_freq_scale'] if ts_weights else 1.0
                    loss = loss + args.lambda_freq * freq_scale * freq_loss
                    freq_loss_val = freq_loss.item()
                loss_components['freq'] = freq_loss_val
                
                # Style Loss
                style_loss_val = 0.0
                if style_loss_fn and args.lambda_style > 0:
                    style_loss = style_loss_fn(model_pred, target_velocity, noisy_latents, timesteps, num_train_timesteps=1000)
                    style_scale = ts_weights['lambda_style_scale'] if ts_weights else 1.0
                    loss = loss + args.lambda_style * style_scale * style_loss
                    style_loss_val = style_loss.item()
                loss_components['style'] = style_loss_val
                
                # RAFT L2 混合模式
                l2_loss_val = 0.0
                if args.raft_mode and args.free_stream_ratio > 0:
                    free_sigmas = torch.rand(batch_size, device=latents.device, dtype=weight_dtype)
                    shift = args.shift
                    free_sigmas = (free_sigmas * shift) / (1 + (shift - 1) * free_sigmas)
                    free_sigmas = free_sigmas.clamp(0.001, 0.999)
                    
                    sigma_bc = free_sigmas.view(batch_size, 1, 1, 1)
                    free_noisy = sigma_bc * noise + (1 - sigma_bc) * latents
                    free_target = noise - latents
                    
                    free_input = free_noisy.unsqueeze(2)
                    if args.gradient_checkpointing:
                        free_input.requires_grad_(True)
                    free_input_list = list(free_input.unbind(dim=0))
                    
                    free_t = free_sigmas * 1000
                    free_t_norm = (1000 - free_t) / 1000.0
                    
                    free_pred_list = transformer(
                        x=free_input_list,
                        t=free_t_norm.to(dtype=weight_dtype),
                        cap_feats=vl_embed,
                    )[0]
                    
                    free_pred = -torch.stack(free_pred_list, dim=0).squeeze(2)
                    l2_loss = F.mse_loss(free_pred, free_target)
                    l2_loss_val = l2_loss.item()
                loss_components['L2'] = l2_loss_val
                
                # SNR weighting (与 LoRA 脚本一致)
                snr_weights = compute_snr_weights(
                    timesteps=timesteps,
                    num_train_timesteps=1000,
                    snr_gamma=args.snr_gamma,
                    snr_floor=args.snr_floor,
                    prediction_type="v_prediction",
                )
                snr_weights = snr_weights.to(device=loss.device, dtype=weight_dtype)
                anchor_loss_weighted = loss * snr_weights.mean()
                
                if l2_loss_val > 0:
                    loss = anchor_loss_weighted + current_l2_ratio * l2_loss
                else:
                    loss = anchor_loss_weighted
                
                # NaN check
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[NaN] Loss is NaN/Inf at step {global_step}, skipping")
                    optimizer.zero_grad()
                    continue
                
                loss = loss.float()
                accelerator.backward(loss)
            
            # Optimization step
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                current_loss = loss.item()
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * current_loss
                
                current_lr = lr_scheduler.get_last_lr()[0]
                
                if accelerator.is_main_process:
                    l1 = loss_components.get('l1', 0)
                    cosine = loss_components.get('cosine', 0)
                    freq = loss_components.get('freq', 0)
                    style = loss_components.get('style', 0)
                    l2 = loss_components.get('L2', 0)
                    print(f"[STEP] {global_step}/{max_train_steps} epoch={epoch+1}/{args.num_train_epochs} loss={current_loss:.4f} ema={ema_loss:.4f} l1={l1:.4f} cos={cosine:.4f} freq={freq:.4f} style={style:.4f} L2={l2:.4f} lr={current_lr:.2e}", flush=True)
            
            # Memory cleanup
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            if accelerator.is_main_process:
                save_path = Path(args.output_dir) / f"{args.output_name}_epoch{epoch+1}.safetensors"
                save_transformer_weights(
                    accelerator.unwrap_model(transformer), 
                    save_path, 
                    dtype=weight_dtype
                )
    
    # Save final model
    if accelerator.is_main_process:
        final_path = Path(args.output_dir) / f"{args.output_name}_final.safetensors"
        save_transformer_weights(
            accelerator.unwrap_model(transformer), 
            final_path, 
            dtype=weight_dtype
        )
    
    logger.info("\n" + "=" * 60)
    logger.info(f"[OK] 全量微调完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
