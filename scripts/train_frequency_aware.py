"""
[START] Frequency-Aware Training Script for Z-Image-Turbo

频域感知训练脚本 - 基于频域分离的解耦学习策略

核心策略：
- 高频增强：L1 Loss 强化纹理/边缘细节，让画面更锐利
- 低频锁定：Cosine Loss 锁定结构/光影方向，防止色偏和风格漂移

适用场景：
- 想要提升细节清晰度但不改变整体风格
- 微调时容易"顾此失彼"的情况
- 需要精细控制高频/低频学习程度

Usage:
    python scripts/train_frequency_aware.py --config config/freq_aware_config.toml
"""

import os
import sys
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from zimage_trainer.acrf_trainer import ACRFTrainer
from zimage_trainer.utils.zimage_utils import load_transformer
from zimage_trainer.networks.lora import LoRANetwork
from zimage_trainer.dataset.dataloader import create_dataloader
from zimage_trainer.utils.memory_optimizer import MemoryOptimizer
from zimage_trainer.utils.hardware_detector import HardwareDetector
from zimage_trainer.losses import FrequencyAwareLoss, AdaptiveFrequencyLoss

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Frequency-Aware 训练脚本")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, help="超参数配置文件路径 (.toml)")
    
    # 模型路径
    parser.add_argument("--dit", type=str, help="Transformer 模型路径")
    parser.add_argument("--dataset_config", type=str, help="数据集配置文件")
    parser.add_argument("--output_dir", type=str, default="output/freq_aware", help="输出目录")
    
    # AC-RF 参数
    parser.add_argument("--turbo_steps", type=int, default=10, help="Turbo 步数（锚点数量）")
    parser.add_argument("--shift", type=float, default=3.0, help="时间步 shift 参数")
    parser.add_argument("--jitter_scale", type=float, default=0.02, help="锚点抖动幅度")
    
    # LoRA 参数
    parser.add_argument("--network_dim", type=int, default=8, help="LoRA rank")
    parser.add_argument("--network_alpha", type=float, default=4.0, help="LoRA alpha")
    
    # 频域感知 Loss 参数
    parser.add_argument("--alpha_hf", type=float, default=1.0, 
                       help="高频增强权重 (推荐 0.5~1.0)")
    parser.add_argument("--beta_lf", type=float, default=0.2, 
                       help="低频锁定权重 (推荐 0.1~0.2)")
    parser.add_argument("--base_weight", type=float, default=1.0, 
                       help="基础 Loss 权重")
    parser.add_argument("--downsample_factor", type=int, default=4, 
                       help="低频提取降采样因子")
    parser.add_argument("--lf_magnitude_weight", type=float, default=0.0, 
                       help="低频幅度约束权重 (防止发灰，建议 0~0.1)")
    parser.add_argument("--adaptive_loss", action="store_true", 
                       help="启用自适应频域 Loss（动态调整权重）")
    
    # 训练参数
    parser.add_argument("--optimizer_type", type=str, default="AdamW", 
                       choices=["AdamW", "AdamW8bit", "Adafactor"])
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    
    # LR Scheduler 参数
    parser.add_argument("--lr_scheduler", type=str, default="constant", 
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Warmup 步数")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Cosine 调度器的循环次数")
    
    # Min-SNR 加权（与频域 Loss 配合使用）
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="Min-SNR gamma (0=禁用)")
    
    # 训练控制
    parser.add_argument("--num_train_epochs", type=int, default=10, help="训练 Epoch 数")
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="保存间隔 (Epoch)")
    parser.add_argument("--output_name", type=str, default="zimage-freq-lora", help="输出文件名")
    
    # 通用参数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    args = parser.parse_args()
    
    # 如果指定了配置文件，读取并覆盖默认值
    if args.config:
        try:
            import tomli
        except ImportError:
            import tomllib as tomli
            
        with open(args.config, "rb") as f:
            config = tomli.load(f)
        
        defaults = {}
        for section in config.values():
            if isinstance(section, dict):
                defaults.update(section)
            
        parser.set_defaults(**defaults)
        args = parser.parse_args()
        
    if not args.dit:
        parser.error("--dit is required")
    
    if not args.dataset_config and args.config:
        args.dataset_config = args.config
        
    return args


def main():
    args = parse_args()
    
    # 硬件检测
    logger.info("[DETECT] 正在进行硬件检测...")
    hardware_detector = HardwareDetector()
    hardware_detector.print_detection_summary()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    
    logger.info("="*60)
    logger.info("[START] 启动 Frequency-Aware 训练")
    logger.info("="*60)
    logger.info(f"🎨 训练策略: 频域分离解耦学习")
    logger.info(f"   高频增强权重 (alpha_hf): {args.alpha_hf}")
    logger.info(f"   低频锁定权重 (beta_lf): {args.beta_lf}")
    logger.info(f"   降采样因子: {args.downsample_factor}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"LoRA rank: {args.network_dim}")
    
    # 1. 加载模型
    logger.info("\n[LOAD] 加载 Transformer...")
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    transformer = load_transformer(
        transformer_path=args.dit,
        device=accelerator.device,
        torch_dtype=weight_dtype,
    )
    transformer.requires_grad_(False)
    transformer.train()
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("  [OK] 梯度检查点已启用")
    
    # 2. 创建 LoRA 网络
    logger.info(f"\n[SETUP] 创建 LoRA 网络 (rank={args.network_dim})...")
    network = LoRANetwork(
        unet=transformer,
        lora_dim=args.network_dim,
        alpha=args.network_alpha,
        multiplier=1.0,
    )
    network.apply_to(transformer)
    
    trainable_params = []
    for lora_module in network.lora_modules.values():
        trainable_params.extend(lora_module.get_trainable_params())
    
    lora_param_count = sum(p.numel() for p in trainable_params)
    logger.info(f"LoRA 可训练参数: {lora_param_count:,} ({lora_param_count/1e6:.2f}M)")
    
    # 3. 创建 AC-RF Trainer（用于采样）
    logger.info(f"\n[INIT] 初始化 AC-RF Trainer...")
    acrf_trainer = ACRFTrainer(
        num_train_timesteps=1000,
        turbo_steps=args.turbo_steps,
        shift=args.shift,
    )
    acrf_trainer.verify_setup()
    
    # 4. 创建频域感知 Loss
    logger.info(f"\n[LOSS] 初始化 Frequency-Aware Loss...")
    if args.adaptive_loss:
        loss_fn = AdaptiveFrequencyLoss(
            alpha_hf=args.alpha_hf,
            beta_lf=args.beta_lf,
            base_weight=args.base_weight,
            downsample_factor=args.downsample_factor,
            lf_magnitude_weight=args.lf_magnitude_weight,
        )
        logger.info("  [OK] 使用自适应频域 Loss（动态权重）")
    else:
        loss_fn = FrequencyAwareLoss(
            alpha_hf=args.alpha_hf,
            beta_lf=args.beta_lf,
            base_weight=args.base_weight,
            downsample_factor=args.downsample_factor,
            lf_magnitude_weight=args.lf_magnitude_weight,
        )
        logger.info("  [OK] 使用固定权重频域 Loss")
    
    # 5. 创建数据加载器
    logger.info("\n[DATA] 加载数据集...")
    dataloader = create_dataloader(args)
    logger.info(f"数据集大小: {len(dataloader)} batches")
    
    # 6. 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total Optimization Steps = {args.max_train_steps}")
    
    if accelerator.is_main_process:
        print(f"[TRAINING_INFO] total_steps={args.max_train_steps} total_epochs={args.num_train_epochs}", flush=True)

    # 7. 创建优化器
    logger.info(f"\n[SETUP] 初始化优化器: {args.optimizer_type}")
    
    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer_type == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params, 
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        except ImportError:
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        
    # 8. 创建学习率调度器
    from diffusers.optimization import get_scheduler
    logger.info(f"[SCHED] 初始化调度器: {args.lr_scheduler} (warmup={args.lr_warmup_steps})")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    # 9. Accelerator prepare
    transformer, network, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, network, optimizer, dataloader, lr_scheduler
    )
    
    # 10. 内存优化器
    memory_optimizer = MemoryOptimizer({'block_swap_enabled': False})
    memory_optimizer.start()
    
    # 11. 训练循环
    logger.info("\n" + "="*60)
    logger.info("[TARGET] 开始频域感知训练")
    logger.info("="*60)
    
    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, desc="Freq-Aware Training", disable=True)
    
    # EMA 平滑 loss
    ema_loss = None
    ema_decay = 0.99
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        transformer.train()
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(network):
                # 获取数据
                latents = batch['latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']
                
                if isinstance(vl_embed, list):
                    vl_embed = [tensor.to(accelerator.device, dtype=weight_dtype) for tensor in vl_embed]
                else:
                    vl_embed = vl_embed.to(accelerator.device, dtype=weight_dtype)
                
                # 生成噪声
                noise = torch.randn_like(latents)
                
                # AC-RF 采样
                noisy_latents, timesteps, target_velocity = acrf_trainer.sample_batch(
                    latents, noise, jitter_scale=args.jitter_scale
                )
                
                # 准备模型输入
                model_input = noisy_latents.unsqueeze(2)
                model_input_list = list(model_input.unbind(dim=0))
                
                # Timestep normalization
                timesteps_normalized = (1000 - timesteps) / 1000.0
                timesteps_normalized = timesteps_normalized.to(dtype=weight_dtype)
                
                # 前向传播
                model_pred_list = transformer(
                    x=model_input_list,
                    t=timesteps_normalized,
                    cap_feats=vl_embed,
                )[0]
                
                model_pred = torch.stack(model_pred_list, dim=0)
                model_pred = model_pred.squeeze(2)
                model_pred = -model_pred  # Z-Image 输出取负
                
                # 计算频域感知 Loss
                if args.adaptive_loss:
                    loss_fn.update_step(global_step)
                
                loss, loss_components = loss_fn(
                    pred_v=model_pred,
                    target_v=target_velocity,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    num_train_timesteps=1000,
                    return_components=True,
                )
                
                # 反向传播
                accelerator.backward(loss)
            
            # 梯度累积完成后执行优化步骤
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                global_step += 1
                
                # 更新 EMA loss
                current_loss = loss.item()
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * current_loss
                
                current_lr = lr_scheduler.get_last_lr()[0]
                
                # 打印进度（只让主进程打印）
                if accelerator.is_main_process:
                    base_l = loss_components["base_loss"].item()
                    hf_l = loss_components["loss_hf"].item()
                    lf_l = loss_components["loss_lf"].item()
                    
                    print(f"[STEP] {global_step}/{args.max_train_steps} epoch={epoch+1}/{args.num_train_epochs} "
                          f"loss={current_loss:.4f} ema={ema_loss:.4f} base={base_l:.4f} hf={hf_l:.4f} lf={lf_l:.4f} "
                          f"lr={current_lr:.2e}", flush=True)
            
            memory_optimizer.optimize_training_step()
                
        # Epoch 结束，保存检查点
        if (epoch + 1) % args.save_every_n_epochs == 0:
            save_path = Path(args.output_dir) / f"{args.output_name}_epoch{epoch+1}.safetensors"
            network.save_weights(save_path, dtype=weight_dtype)
            logger.info(f"\n[SAVE] 保存检查点 (Epoch {epoch+1}): {save_path}")
    
    # 保存最终模型
    final_path = Path(args.output_dir) / f"{args.output_name}_final.safetensors"
    network.save_weights(final_path, dtype=weight_dtype)
    
    memory_optimizer.stop()
    
    logger.info("\n" + "="*60)
    logger.info(f"[OK] 频域感知训练完成！")
    logger.info(f"最终模型: {final_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

