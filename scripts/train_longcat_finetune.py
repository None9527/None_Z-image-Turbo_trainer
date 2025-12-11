"""
[START] Full Fine-tune Training Script for LongCat-Image

全量微调脚本 - 训练 LongCat Transformer 的全部参数
⚠️ 显存需求高：建议 48GB+ VRAM，推荐 80GB+

关键特性：
- 全参数微调，不使用 LoRA
- 更小的学习率（1e-6 ~ 1e-5）
- 强制启用梯度检查点以节省显存
- 支持 FLUX 架构的 Flow Matching 训练
"""

import os
import sys
import math
import gc
import time
import signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局中断标志
_interrupted = False

def signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    logger.info("[SIGNAL] 收到中断信号，正在准备安全退出...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="LongCat Full Fine-tune 训练脚本（全量微调）")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, help="超参数配置文件路径 (.toml)")
    
    # 模型路径
    parser.add_argument("--dit", type=str, help="LongCat Transformer 模型路径")
    parser.add_argument("--dataset_config", type=str, help="数据集配置文件")
    parser.add_argument("--output_dir", type=str, default="output/longcat_finetune", help="输出目录")
    
    # 训练参数
    parser.add_argument("--turbo_steps", type=int, default=10, help="目标加速步数")
    parser.add_argument("--shift", type=float, default=0.5, help="Flow Matching shift 参数")
    
    # 锚点采样
    parser.add_argument("--use_anchor", type=bool, default=True, help="使用锚点时间步采样")
    parser.add_argument("--use_dynamic_shifting", type=bool, default=True, help="动态 shift")
    
    # 全量微调参数 - 学习率要小很多
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="学习率（全量微调建议 1e-6 ~ 1e-5）")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    
    # 训练参数
    parser.add_argument("--optimizer_type", type=str, default="AdamW8bit", 
                       choices=["AdamW", "AdamW8bit", "Adafactor"], 
                       help="优化器类型（推荐 AdamW8bit 节省显存）")
    
    # LR Scheduler 参数
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="学习率调度器（全量微调推荐 cosine）"
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Warmup 步数")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Cosine 调度器的循环次数")
    
    # Loss 参数
    parser.add_argument("--lambda_l1", type=float, default=1.0, help="L1 Loss 权重")
    parser.add_argument("--lambda_cosine", type=float, default=0.1, help="Cosine Loss 权重")
    parser.add_argument("--lambda_freq", type=float, default=0.3, help="Frequency Loss 权重")
    parser.add_argument("--enable_freq", type=bool, default=True, help="启用频域损失")
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="Min-SNR gamma")
    
    # 训练控制 (Epoch 模式)
    parser.add_argument("--num_train_epochs", type=int, default=5, help="训练 Epoch 数")
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="保存间隔 (Epoch)")
    parser.add_argument("--output_name", type=str, default="longcat-finetune", help="输出文件名")
    
    # 全量微调专用参数
    parser.add_argument("--trainable_modules", type=str, default="all",
                       choices=["all", "attention", "mlp", "norm", "single_stream", "dual_stream"],
                       help="可训练模块")
    parser.add_argument("--freeze_embeddings", action="store_true", default=True,
                       help="冻结嵌入层")
    
    # 兼容性保留
    parser.add_argument("--max_train_steps", type=int, default=None, help="最大训练步数 (自动计算)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    
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
    """根据配置返回可训练的参数"""
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
            if any(key in name.lower() for key in ['attn', 'to_q', 'to_k', 'to_v', 'to_out', 'add_k_proj', 'add_q_proj', 'add_v_proj']):
                should_train = True
        elif trainable_modules == "mlp":
            if any(key in name.lower() for key in ['mlp', 'ff.net', 'ff_context']):
                should_train = True
        elif trainable_modules == "norm":
            if any(key in name.lower() for key in ['norm', 'layer_norm']):
                should_train = True
        elif trainable_modules == "single_stream":
            if 'single_transformer_blocks' in name:
                should_train = True
        elif trainable_modules == "dual_stream":
            if 'transformer_blocks' in name and 'single' not in name:
                should_train = True
        
        # 冻结嵌入层
        if freeze_embeddings and any(key in name.lower() for key in ['embed', 'embedding', 'pos_embed', 'x_embedder', 'context_embedder']):
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
    
    # 显存检查
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[GPU] 检测到显存: {total_memory:.1f} GB")
        if total_memory < 40:
            logger.warning("⚠️ 显存不足 40GB，LongCat 全量微调可能会 OOM！")
            logger.warning("   建议使用 LoRA 训练 (train_longcat.py) 或增加梯度累积步数")
    
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
    logger.info("[START] 启动 LongCat Full Fine-tune 训练（全量微调）")
    logger.info("="*60)
    logger.info(f"⚠️ 注意：LongCat 全量微调需要大量显存，建议 VRAM >= 48GB")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"可训练模块: {args.trainable_modules}")
    logger.info(f"冻结嵌入层: {args.freeze_embeddings}")
    logger.info(f"学习率: {args.learning_rate}")
    
    # 1. 加载模型
    logger.info("\n[LOAD] 加载 LongCat Transformer...")
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    from longcat_image.models.longcat_image_dit import LongCatImageTransformer2DModel
    transformer = LongCatImageTransformer2DModel.from_pretrained(
        args.dit, 
        local_files_only=True
    ).to(accelerator.device, dtype=weight_dtype)
    
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
    
    # 3. 启用梯度检查点
    logger.info("\n[MEM] 启用显存优化...")
    if hasattr(transformer, 'enable_gradient_checkpointing'):
        transformer.enable_gradient_checkpointing()
        logger.info("  [OK] 梯度检查点已启用")
    
    transformer.train()
    
    # 4. 初始化 LongCat 训练适配器
    logger.info(f"\n[INIT] 初始化 LongCat Training Adapter...")
    from longcat_image.training.longcat_adapter import LongCatTrainingAdapter
    adapter = LongCatTrainingAdapter(
        num_train_timesteps=1000,
        turbo_steps=args.turbo_steps,
        base_shift=args.shift,
        use_dynamic_shifting=args.use_dynamic_shifting,
    )
    
    # 5. 创建数据加载器
    logger.info("\n[DATA] 加载数据集...")
    from longcat_image.dataset.longcat_dataloader import LongCatDataset
    
    try:
        import tomli
    except ImportError:
        import tomllib as tomli
    
    with open(args.dataset_config, "rb") as f:
        config = tomli.load(f)
    
    if 'dataset' in config:
        dataset_config = config['dataset']
        datasets = dataset_config.get('sources', dataset_config.get('datasets', []))
    else:
        datasets = config.get('datasets', [])
    
    if not datasets:
        cache_dir = config.get('cache_directory')
        if cache_dir:
            datasets = [{
                'cache_directory': cache_dir,
                'num_repeats': config.get('num_repeats', 1),
                'resolution_limit': config.get('resolution_limit', None)
            }]
        else:
            raise ValueError("No datasets configured")
    
    batch_size = config.get('batch_size', 1)
    num_workers = config.get('num_workers', 4)
    
    dataset = LongCatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    logger.info(f"数据集大小: {len(dataloader)} batches")
    
    # 6. 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    max_train_steps = args.max_train_steps
    
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num Batches per Epoch = {len(dataloader)}")
    logger.info(f"  Gradient Accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total Optimization Steps = {max_train_steps}")
    
    # 打印总步数供前端解析 (只让主进程打印)
    if accelerator.is_main_process:
        print(f"[TRAINING_INFO] total_steps={max_train_steps} total_epochs={args.num_train_epochs}", flush=True)
    
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
            scale_parameter=False,
            relative_step=False,
        )
        logger.info("  [OK] 使用 Adafactor 节省显存")
    
    # 8. 创建学习率调度器
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    # 9. Accelerator prepare
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    
    # 10. 训练循环
    logger.info("\n" + "="*60)
    logger.info("[TARGET] 开始 LongCat 全量微调训练")
    logger.info("="*60)
    
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
        
        logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        transformer.train()
        
        for step, batch in enumerate(dataloader):
            if _interrupted:
                break
                
            with accelerator.accumulate(transformer):
                # 获取数据
                latents = batch['latents'].to(accelerator.device, dtype=weight_dtype)
                text_embeds = batch['text_embeds']
                
                batch_size, C, H, W = latents.shape
                
                # 处理 text embeddings
                if isinstance(text_embeds, list):
                    text_embeds = torch.stack([t.to(accelerator.device, dtype=weight_dtype) for t in text_embeds])
                else:
                    text_embeds = text_embeds.to(accelerator.device, dtype=weight_dtype)
                
                # 生成噪声
                noise = torch.randn_like(latents)
                
                # 计算图像序列长度
                image_seq_len = (H // 2) * (W // 2)
                
                # 采样时间步
                timesteps, sigmas = adapter.sample_timesteps(
                    batch_size=batch_size,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    image_seq_len=image_seq_len,
                    use_anchor=args.use_anchor,
                )
                
                # 添加噪声
                noisy_latents = sigmas.view(-1, 1, 1, 1) * noise + (1 - sigmas.view(-1, 1, 1, 1)) * latents
                
                # 目标速度
                target_velocity = noise - latents
                
                # Pack latents
                packed_latents, pack_info = adapter.pack_latents(noisy_latents)
                
                if args.gradient_checkpointing:
                    packed_latents.requires_grad_(True)
                
                # 准备位置编码
                position_ids = adapter.get_position_ids(
                    batch_size=batch_size,
                    height=H,
                    width=W,
                    device=accelerator.device,
                )
                
                # 前向传播
                model_output = adapter.forward(
                    transformer=transformer,
                    hidden_states=packed_latents,
                    encoder_hidden_states=text_embeds,
                    timesteps=sigmas,
                    position_ids=position_ids,
                )
                
                # Unpack
                model_pred = adapter.unpack_latents(model_output, pack_info)
                model_pred = model_pred.to(dtype=weight_dtype)
                
                # 计算损失
                loss_components = {}
                
                # L1 Loss
                l1_loss = F.l1_loss(model_pred, target_velocity)
                loss_components['l1'] = l1_loss.item()
                
                # Cosine Loss
                cos_loss = 1 - F.cosine_similarity(model_pred.flatten(1), target_velocity.flatten(1), dim=1).mean()
                loss_components['cosine'] = cos_loss.item()
                
                # 总损失
                loss = args.lambda_l1 * l1_loss + args.lambda_cosine * cos_loss
                
                # Frequency Loss
                if args.enable_freq and args.lambda_freq > 0:
                    pred_fft = torch.fft.fft2(model_pred, norm='ortho')
                    target_fft = torch.fft.fft2(target_velocity, norm='ortho')
                    freq_loss = F.l1_loss(pred_fft.abs(), target_fft.abs())
                    loss = loss + args.lambda_freq * freq_loss
                    loss_components['freq'] = freq_loss.item()
                else:
                    loss_components['freq'] = 0.0
                
                loss_components['style'] = 0.0
                loss_components['L2'] = 0.0
                
                # 反向传播
                accelerator.backward(loss)
            
            # 梯度累积完成后执行优化步骤
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 更新 EMA loss
                current_loss = loss.item()
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * current_loss
                
                current_lr = lr_scheduler.get_last_lr()[0]
                
                # 打印进度 (只让主进程打印，避免多卡日志混乱)
                if accelerator.is_main_process:
                    l1 = loss_components.get('l1', 0)
                    cosine = loss_components.get('cosine', 0)
                    freq = loss_components.get('freq', 0)
                    style = loss_components.get('style', 0)
                    l2 = loss_components.get('L2', 0)
                    print(f"[STEP] {global_step}/{max_train_steps} epoch={epoch+1}/{args.num_train_epochs} loss={current_loss:.4f} ema={ema_loss:.4f} l1={l1:.4f} cos={cosine:.4f} freq={freq:.4f} style={style:.4f} L2={l2:.4f} lr={current_lr:.2e}", flush=True)
            
            # 定期清理显存
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Epoch 结束，保存检查点
        if accelerator.is_main_process and (epoch + 1) % args.save_every_n_epochs == 0:
            save_path = Path(args.output_dir) / f"{args.output_name}_epoch{epoch+1}.safetensors"
            save_transformer_weights(
                accelerator.unwrap_model(transformer), 
                save_path, 
                dtype=weight_dtype
            )
    
    # 保存最终模型
    if accelerator.is_main_process:
        final_path = Path(args.output_dir) / f"{args.output_name}_final.safetensors"
        save_transformer_weights(
            accelerator.unwrap_model(transformer), 
            final_path, 
            dtype=weight_dtype
        )
    
    logger.info("\n" + "="*60)
    logger.info(f"[OK] LongCat 全量微调完成！")
    if accelerator.is_main_process:
        logger.info(f"最终模型: {final_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
