"""
[START] Full Fine-tune Training Script for Z-Image-Turbo

全量微调脚本 - 训练 Transformer 的全部参数
⚠️ 显存需求高：建议 24GB+ VRAM，推荐 48GB+

关键特性：
- 全参数微调，不使用 LoRA
- 更小的学习率（1e-6 ~ 1e-5）
- 强制启用梯度检查点以节省显存
- 支持 DeepSpeed ZeRO 优化
"""

import os
import sys
import math
import gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file

from zimage_trainer.acrf_trainer import ACRFTrainer
from zimage_trainer.utils.zimage_utils import load_transformer
from zimage_trainer.dataset.dataloader import create_dataloader
from zimage_trainer.utils.memory_optimizer import MemoryOptimizer
from zimage_trainer.utils.hardware_detector import HardwareDetector

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Full Fine-tune 训练脚本（全量微调）")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, help="超参数配置文件路径 (.toml)")
    
    # 模型路径
    parser.add_argument("--dit", type=str, help="Transformer 模型路径")
    parser.add_argument("--dataset_config", type=str, help="数据集配置文件")
    parser.add_argument("--output_dir", type=str, default="output/full_finetune", help="输出目录")
    
    # AC-RF 参数
    parser.add_argument("--turbo_steps", type=int, default=10, help="Turbo 步数（锚点数量）")
    parser.add_argument("--shift", type=float, default=3.0, help="时间步 shift 参数")
    parser.add_argument("--jitter_scale", type=float, default=0.02, help="锚点抖动幅度")
    
    # 全量微调参数 - 学习率要小很多
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="学习率（全量微调建议 1e-6 ~ 1e-5）")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    
    # 训练参数
    parser.add_argument("--optimizer_type", type=str, default="AdamW8bit", 
                       choices=["AdamW", "AdamW8bit", "Adafactor"], 
                       help="优化器类型（推荐 AdamW8bit 节省显存）")
    
    # Adafactor 特有参数
    parser.add_argument("--adafactor_scale", action="store_true", help="Adafactor scale_parameter")
    parser.add_argument("--adafactor_relative", action="store_true", help="Adafactor relative_step")
    parser.add_argument("--adafactor_warmup", action="store_true", help="Adafactor warmup_init")
    
    # LR Scheduler 参数
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="学习率调度器（全量微调推荐 cosine）"
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Warmup 步数（全量微调建议更多 warmup）")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Cosine 调度器的循环次数")
    
    # Loss 参数
    parser.add_argument("--lambda_fft", type=float, default=0.05, help="FFT Loss 权重")
    parser.add_argument("--lambda_cosine", type=float, default=0.05, help="Cosine Loss 权重")
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="Min-SNR gamma (0=禁用, 推荐5.0)")
    
    # 训练控制 (Epoch 模式)
    parser.add_argument("--num_train_epochs", type=int, default=5, help="训练 Epoch 数（全量微调通常需要更少 epoch）")
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="保存间隔 (Epoch)")
    parser.add_argument("--output_name", type=str, default="zimage-finetune", help="输出文件名")
    
    # 全量微调专用参数
    parser.add_argument("--trainable_modules", type=str, default="all",
                       choices=["all", "attention", "mlp", "norm"],
                       help="可训练模块：all=全部, attention=仅注意力层, mlp=仅MLP层, norm=仅归一化层")
    parser.add_argument("--freeze_embeddings", action="store_true", default=True,
                       help="冻结嵌入层（推荐开启，减少过拟合风险）")
    
    # 兼容性保留
    parser.add_argument("--max_train_steps", type=int, default=None, help="最大训练步数 (自动计算)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积（全量微调建议更大）")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
                       help="混合精度（全量微调推荐 bf16）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 高级功能 - 全量微调必须开启梯度检查点
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, 
                       help="启用梯度检查点（全量微调强制开启）")
    
    # 显存优化
    parser.add_argument("--cpu_offload", action="store_true", help="CPU 卸载优化器状态")
    parser.add_argument("--activation_checkpointing", action="store_true", default=True,
                       help="激活检查点")
    
    # SDPA 参数
    parser.add_argument("--attention_backend", type=str, default="sdpa", 
        choices=["sdpa", "flash", "_flash_3"], help="注意力后端选择")
    parser.add_argument("--enable_flash_attention", action="store_true", help="启用Flash Attention")
    
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
        
    # 验证必要参数
    if not args.dit:
        parser.error("--dit is required")
    
    if not args.dataset_config and args.config:
        args.dataset_config = args.config
        
    return args


def get_trainable_parameters(transformer, trainable_modules: str, freeze_embeddings: bool):
    """
    根据配置返回可训练的参数
    
    Args:
        transformer: Transformer 模型
        trainable_modules: 可训练模块类型
        freeze_embeddings: 是否冻结嵌入层
    
    Returns:
        trainable_params: 可训练参数列表
        frozen_count: 冻结参数数量
        trainable_count: 可训练参数数量
    """
    # 先冻结所有参数
    transformer.requires_grad_(False)
    
    trainable_params = []
    trainable_count = 0
    frozen_count = 0
    
    for name, param in transformer.named_parameters():
        should_train = False
        
        if trainable_modules == "all":
            should_train = True
        elif trainable_modules == "attention":
            # 只训练注意力相关的层
            if any(key in name.lower() for key in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
                should_train = True
        elif trainable_modules == "mlp":
            # 只训练 MLP 层
            if any(key in name.lower() for key in ['mlp', 'fc1', 'fc2', 'ffn', 'feed_forward']):
                should_train = True
        elif trainable_modules == "norm":
            # 只训练归一化层
            if any(key in name.lower() for key in ['norm', 'ln', 'layer_norm', 'layernorm']):
                should_train = True
        
        # 冻结嵌入层
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
    """
    保存 Transformer 权重为 safetensors 格式
    """
    state_dict = {}
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.data.to(dtype).cpu()
    
    save_file(state_dict, str(save_path))
    logger.info(f"[SAVE] 保存权重: {save_path} ({len(state_dict)} tensors)")


def main():
    args = parse_args()
    
    # 显存检查
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[GPU] 检测到显存: {total_memory:.1f} GB")
        if total_memory < 20:
            logger.warning("⚠️ 显存不足 20GB，全量微调可能会 OOM！")
            logger.warning("   建议使用 LoRA 训练 (train_acrf.py) 或增加梯度累积步数")
    
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
    logger.info("[START] 启动 Full Fine-tune 训练（全量微调）")
    logger.info("="*60)
    logger.info(f"⚠️ 注意：全量微调需要大量显存，请确保 VRAM >= 24GB")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"Turbo 步数: {args.turbo_steps}")
    logger.info(f"可训练模块: {args.trainable_modules}")
    logger.info(f"冻结嵌入层: {args.freeze_embeddings}")
    logger.info(f"学习率: {args.learning_rate}")
    
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
    
    # 2. 配置可训练参数
    logger.info(f"\n[SETUP] 配置可训练参数 (模式: {args.trainable_modules})...")
    trainable_params, frozen_count, trainable_count = get_trainable_parameters(
        transformer, 
        args.trainable_modules, 
        args.freeze_embeddings
    )
    
    total_params = frozen_count + trainable_count
    logger.info(f"  总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
    logger.info(f"  可训练参数: {trainable_count:,} ({trainable_count/1e6:.2f}M, {100*trainable_count/total_params:.1f}%)")
    logger.info(f"  冻结参数: {frozen_count:,} ({frozen_count/1e6:.2f}M)")
    
    # 3. 强制启用梯度检查点
    logger.info("\n[MEM] 启用显存优化...")
    if hasattr(transformer, 'enable_gradient_checkpointing'):
        transformer.enable_gradient_checkpointing()
        logger.info("  [OK] 梯度检查点已启用")
    else:
        logger.warning("  [WARN] 模型不支持梯度检查点")
    
    transformer.train()
    
    # 4. 初始化内存优化器
    memory_config = {
        'block_swap_enabled': False,
        'checkpoint_optimization': 'full',
    }
    memory_optimizer = MemoryOptimizer(memory_config)
    memory_optimizer.start()
    
    # 5. 创建 AC-RF Trainer
    logger.info(f"\n[INIT] 初始化 AC-RF Trainer...")
    acrf_trainer = ACRFTrainer(
        num_train_timesteps=1000,
        turbo_steps=args.turbo_steps,
        shift=args.shift,
    )
    acrf_trainer.verify_setup()
    
    # 6. 创建数据加载器
    logger.info("\n[DATA] 加载数据集...")
    dataloader = create_dataloader(args)
    logger.info(f"数据集大小: {len(dataloader)} batches")
    
    # 7. 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num Batches per Epoch = {len(dataloader)}")
    logger.info(f"  Gradient Accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total Optimization Steps = {args.max_train_steps}")
    
    # 打印总步数供前端解析
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
            logger.info("  [OK] 使用 8-bit AdamW 节省显存")
        except ImportError:
            logger.warning("  [WARN] bitsandbytes 未安装，回退到标准 AdamW")
            optimizer = torch.optim.AdamW(
                trainable_params, 
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
    elif args.optimizer_type == "Adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            scale_parameter=args.adafactor_scale,
            relative_step=args.adafactor_relative,
            warmup_init=args.adafactor_warmup
        )
        logger.info("  [OK] 使用 Adafactor 节省显存")
    
    # 9. 创建学习率调度器
    from diffusers.optimization import get_scheduler
    logger.info(f"[SCHED] 初始化调度器: {args.lr_scheduler} (warmup={args.lr_warmup_steps})")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    # 10. Accelerator prepare
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    
    # 11. 训练循环
    logger.info("\n" + "="*60)
    logger.info("[TARGET] 开始全量微调训练")
    logger.info("="*60)
    
    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, desc="Full Fine-tune", disable=True)
    
    # EMA 平滑 loss
    ema_loss = None
    ema_decay = 0.99
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        transformer.train()
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(transformer):
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
                
                # Stack outputs
                model_pred = torch.stack(model_pred_list, dim=0)
                model_pred = model_pred.squeeze(2)
                model_pred = -model_pred
                
                # 计算损失
                loss = acrf_trainer.compute_loss(
                    model_output=model_pred,
                    target_velocity=target_velocity,
                    latents_noisy=noisy_latents,
                    timesteps=timesteps,
                    target_x0=latents,
                    lambda_fft=args.lambda_fft,
                    lambda_cosine=args.lambda_cosine,
                    snr_gamma=args.snr_gamma,
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
                
                # 打印进度
                print(f"[STEP] {global_step}/{args.max_train_steps} epoch={epoch+1}/{args.num_train_epochs} loss={current_loss:.4f} ema_loss={ema_loss:.4f} lr={current_lr:.2e}", flush=True)
            
            # 定期清理显存
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
        # Epoch 结束，保存检查点
        if (epoch + 1) % args.save_every_n_epochs == 0:
            save_path = Path(args.output_dir) / f"{args.output_name}_epoch{epoch+1}.safetensors"
            save_transformer_weights(
                accelerator.unwrap_model(transformer), 
                save_path, 
                dtype=weight_dtype
            )
    
    # 保存最终模型
    final_path = Path(args.output_dir) / f"{args.output_name}_final.safetensors"
    save_transformer_weights(
        accelerator.unwrap_model(transformer), 
        final_path, 
        dtype=weight_dtype
    )
    
    # 停止内存优化器
    memory_optimizer.stop()
    
    logger.info("\n" + "="*60)
    logger.info(f"[OK] 全量微调完成！")
    logger.info(f"最终模型: {final_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

