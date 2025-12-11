"""
[START] Style-Structure Transfer Training Script for Z-Image-Turbo

结构锁风格迁移训练脚本

核心功能：
输入一张普通画质图片，输出一张保持原图几何结构（Structure-Preserving），
但具有"大师级"光影、色调和纹理质感的图片。

技术路径：
- 图生图模式训练 (Img2Img Training)
- 自监督退化策略 (Self-Supervised Degradation)
- 频域分离损失 (Style-Structure Loss)

Loss 架构：
L_total = λ_struct * L_SSIM + λ_light * L_Moments_L + λ_color * L_Moments_ab + λ_tex * L_HighFreq

Usage:
    python scripts/train_style_transfer.py --config config/style_transfer_config.toml
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
from zimage_trainer.losses.style_structure_loss import LatentStyleStructureLoss
from zimage_trainer.utils.degradation import ImageDegradation, create_degradation_transform

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Style-Structure Transfer 训练脚本")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, help="超参数配置文件路径 (.toml)")
    
    # 模型路径
    parser.add_argument("--dit", type=str, help="Transformer 模型路径")
    parser.add_argument("--dataset_config", type=str, help="数据集配置文件")
    parser.add_argument("--output_dir", type=str, default="output/style_transfer", help="输出目录")
    
    # AC-RF 参数
    parser.add_argument("--turbo_steps", type=int, default=10, help="Turbo 步数")
    parser.add_argument("--shift", type=float, default=3.0, help="时间步 shift 参数")
    parser.add_argument("--jitter_scale", type=float, default=0.02, help="锚点抖动幅度")
    
    # LoRA 参数
    parser.add_argument("--network_dim", type=int, default=32, help="LoRA rank (风格迁移建议 32-128)")
    parser.add_argument("--network_alpha", type=float, default=16.0, help="LoRA alpha")
    
    # 风格结构 Loss 参数
    parser.add_argument("--lambda_struct", type=float, default=1.0, 
                       help="结构锁权重 (SSIM，防止脸崩)")
    parser.add_argument("--lambda_light", type=float, default=0.5, 
                       help="光影学习权重 (L通道统计)")
    parser.add_argument("--lambda_color", type=float, default=0.3, 
                       help="色调迁移权重 (ab通道统计)")
    parser.add_argument("--lambda_tex", type=float, default=0.5, 
                       help="质感增强权重 (高频L1)")
    parser.add_argument("--lambda_base", type=float, default=1.0, 
                       help="基础 v-prediction loss 权重")
    
    # 退化参数
    parser.add_argument("--degradation_strength", type=str, default="medium",
                       choices=["light", "medium", "heavy"],
                       help="退化强度预设")
    parser.add_argument("--enable_degradation", action="store_true", default=True,
                       help="启用自监督退化（图生图训练）")
    
    # 训练参数
    parser.add_argument("--optimizer_type", type=str, default="AdamW", 
                       choices=["AdamW", "AdamW8bit", "Adafactor"])
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    
    # LR Scheduler 参数
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Warmup 步数")
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    
    # 训练控制
    parser.add_argument("--num_train_epochs", type=int, default=10, help="训练 Epoch 数")
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="保存间隔")
    parser.add_argument("--output_name", type=str, default="zimage-style-lora", help="输出文件名")
    
    # 通用参数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    args = parser.parse_args()
    
    # 读取配置文件
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
    logger.info("[START] 启动 Style-Structure Transfer 训练")
    logger.info("="*60)
    logger.info(f"🎨 训练策略: 结构锁风格迁移")
    logger.info(f"   结构锁 (SSIM): {args.lambda_struct}")
    logger.info(f"   光影学习: {args.lambda_light}")
    logger.info(f"   色调迁移: {args.lambda_color}")
    logger.info(f"   质感增强: {args.lambda_tex}")
    logger.info(f"   退化强度: {args.degradation_strength}")
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
    
    # 3. 创建 AC-RF Trainer
    logger.info(f"\n[INIT] 初始化 AC-RF Trainer...")
    acrf_trainer = ACRFTrainer(
        num_train_timesteps=1000,
        turbo_steps=args.turbo_steps,
        shift=args.shift,
    )
    acrf_trainer.verify_setup()
    
    # 4. 创建风格结构 Loss
    logger.info(f"\n[LOSS] 初始化 Style-Structure Loss...")
    loss_fn = LatentStyleStructureLoss(
        lambda_struct=args.lambda_struct,
        lambda_light=args.lambda_light,
        lambda_color=args.lambda_color,
        lambda_tex=args.lambda_tex,
        lambda_base=args.lambda_base,
    )
    
    # 5. 创建退化变换
    if args.enable_degradation:
        logger.info(f"\n[DEGRADE] 初始化退化变换 (强度: {args.degradation_strength})...")
        degradation = create_degradation_transform(args.degradation_strength)
    else:
        degradation = None
        logger.info("[DEGRADE] 退化变换已禁用")
    
    # 6. 创建数据加载器
    logger.info("\n[DATA] 加载数据集...")
    dataloader = create_dataloader(args)
    logger.info(f"数据集大小: {len(dataloader)} batches")
    
    # 7. 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total Optimization Steps = {args.max_train_steps}")
    
    if accelerator.is_main_process:
        print(f"[TRAINING_INFO] total_steps={args.max_train_steps} total_epochs={args.num_train_epochs}", flush=True)

    # 8. 创建优化器
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
        
    # 9. 创建学习率调度器
    from diffusers.optimization import get_scheduler
    logger.info(f"[SCHED] 初始化调度器: {args.lr_scheduler} (warmup={args.lr_warmup_steps})")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    # 10. Accelerator prepare
    transformer, network, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, network, optimizer, dataloader, lr_scheduler
    )
    
    # 11. 内存优化器
    memory_optimizer = MemoryOptimizer({'block_swap_enabled': False})
    memory_optimizer.start()
    
    # 12. 训练循环
    logger.info("\n" + "="*60)
    logger.info("[TARGET] 开始结构锁风格迁移训练")
    logger.info("="*60)
    
    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, desc="Style-Transfer Training", disable=True)
    
    # EMA 平滑 loss
    ema_loss = None
    ema_decay = 0.99
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        transformer.train()
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(network):
                # 获取数据 (高质量目标)
                latents = batch['latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']
                
                if isinstance(vl_embed, list):
                    vl_embed = [tensor.to(accelerator.device, dtype=weight_dtype) for tensor in vl_embed]
                else:
                    vl_embed = vl_embed.to(accelerator.device, dtype=weight_dtype)
                
                # 生成噪声
                noise = torch.randn_like(latents)
                
                # 对于风格迁移训练，我们使用原始 latents 作为目标
                # 但可以对输入应用退化（在 latent 空间近似）
                target_latents = latents
                
                # AC-RF 采样（使用目标 latents）
                noisy_latents, timesteps, target_velocity = acrf_trainer.sample_batch(
                    target_latents, noise, jitter_scale=args.jitter_scale
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
                
                # 计算风格结构 Loss
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
                    struct_l = loss_components["loss_struct"].item()
                    light_l = loss_components["loss_light"].item()
                    color_l = loss_components["loss_color"].item()
                    tex_l = loss_components["loss_tex"].item()
                    
                    print(f"[STEP] {global_step}/{args.max_train_steps} epoch={epoch+1}/{args.num_train_epochs} "
                          f"loss={current_loss:.4f} ema={ema_loss:.4f} "
                          f"struct={struct_l:.4f} light={light_l:.4f} color={color_l:.4f} tex={tex_l:.4f} "
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
    logger.info(f"[OK] 结构锁风格迁移训练完成！")
    logger.info(f"最终模型: {final_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()


