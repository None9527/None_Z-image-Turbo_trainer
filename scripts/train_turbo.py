"""
[START] Turbo/Accelerated LoRA Training Script

通用的加速 LoRA 训练脚本，支持多种模型：
- Z-Image (Turbo)
- LongCat-Image
- 更多模型待扩展...

核心思路：
使用 Rectified Flow 方法，在 N 步锚点上训练 LoRA，
使非加速模型也能用少步采样生成高质量图像。
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

# 从新的模块化结构导入
from models import get_adapter, list_adapters, auto_detect_adapter, ModelAdapter
from zimage_trainer.dataset.dataloader import create_dataloader
from zimage_trainer.utils.memory_optimizer import MemoryOptimizer
from zimage_trainer.utils.snr_utils import compute_snr_weights

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="通用加速 LoRA 训练脚本")
    
    # 模型选择
    parser.add_argument("--model_type", type=str, default="auto",
        help=f"模型类型: auto, {', '.join(list_adapters())}。auto 会自动检测")
    parser.add_argument("--dit", type=str, required=True,
        help="Transformer 模型路径")
    parser.add_argument("--vae", type=str, default=None,
        help="VAE 路径（可选，默认使用模型目录下的 vae）")
    
    # 配置文件
    parser.add_argument("--config", type=str, help="配置文件路径 (.toml)")
    parser.add_argument("--dataset_config", type=str, help="数据集配置文件")
    parser.add_argument("--output_dir", type=str, default="output/turbo_lora",
        help="输出目录")
    parser.add_argument("--output_name", type=str, default="turbo-lora",
        help="输出文件名前缀")
    
    # 加速参数
    parser.add_argument("--turbo_steps", type=int, default=10,
        help="目标加速步数（锚点数量）")
    parser.add_argument("--shift", type=float, default=3.0,
        help="时间步 shift 参数")
    parser.add_argument("--jitter_scale", type=float, default=0.02,
        help="锚点抖动幅度")
    parser.add_argument("--use_anchor", action="store_true", default=True,
        help="使用锚点采样（默认开启）")
    parser.add_argument("--no_anchor", action="store_true",
        help="禁用锚点采样，使用连续时间步")
    
    # LoRA 参数
    parser.add_argument("--network_dim", type=int, default=32,
        help="LoRA rank")
    parser.add_argument("--network_alpha", type=float, default=16.0,
        help="LoRA alpha")
    parser.add_argument("--use_peft", action="store_true",
        help="使用 PEFT 库（推荐用于 LongCat）")
    
    # 训练参数
    parser.add_argument("--optimizer_type", type=str, default="AdamW8bit",
        choices=["AdamW", "AdamW8bit", "Adafactor"],
        help="优化器类型")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
        help="权重衰减")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                 "constant", "constant_with_warmup"],
        help="学习率调度器")
    parser.add_argument("--lr_warmup_steps", type=int, default=100,
        help="Warmup 步数")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Cosine 调度器循环次数")
    
    parser.add_argument("--num_train_epochs", type=int, default=10,
        help="训练轮数")
    parser.add_argument("--max_train_steps", type=int, default=None,
        help="最大训练步数（覆盖 epochs）")
    parser.add_argument("--save_every_n_epochs", type=int, default=1,
        help="每 N 轮保存一次")
    parser.add_argument("--save_every_n_steps", type=int, default=None,
        help="每 N 步保存一次")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="梯度累积步数")
    parser.add_argument("--seed", type=int, default=42,
        help="随机种子")
    
    # SNR 加权
    parser.add_argument("--snr_gamma", type=float, default=5.0,
        help="Min-SNR gamma (0=禁用)")
    parser.add_argument("--snr_floor", type=float, default=0.1,
        help="Min-SNR 保底权重")
    
    # 高级参数
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
        help="梯度裁剪阈值")
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="启用梯度检查点")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
        choices=["no", "fp16", "bf16"],
        help="混合精度")
    
    # 数据集参数
    parser.add_argument("--batch_size", type=int, default=1,
        help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
        help="数据加载线程数")
    
    args = parser.parse_args()
    
    # 处理配置文件
    if args.config:
        args = load_config_and_merge(args)
    
    # 处理 --no_anchor
    if args.no_anchor:
        args.use_anchor = False
    
    return args


def load_config_and_merge(args):
    """加载配置文件并合并参数"""
    try:
        import toml
    except ImportError:
        try:
            import tomli as toml
        except ImportError:
            logger.warning("toml/tomli not installed, skipping config file")
            return args
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = toml.load(f)
    
    # 合并配置（命令行参数优先）
    args_dict = vars(args)
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if key not in args_dict or args_dict[key] is None:
                    args_dict[key] = value
    
    return argparse.Namespace(**args_dict)


def create_optimizer(args, params):
    """创建优化器"""
    if args.optimizer_type == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
    elif args.optimizer_type == "Adafactor":
        from transformers import Adafactor
        optimizer = Adafactor(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            scale_parameter=False,
            relative_step=False,
        )
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    
    return optimizer


def create_lr_scheduler(args, optimizer, num_training_steps):
    """创建学习率调度器"""
    from diffusers.optimization import get_scheduler
    
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    return scheduler


def setup_lora(args, adapter: ModelAdapter, transformer):
    """设置 LoRA"""
    if args.use_peft:
        # 使用 PEFT 库
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=args.network_dim,
            lora_alpha=args.network_alpha,
            target_modules=adapter.get_lora_target_modules(),
            lora_dropout=0.0,
            bias="none",
            init_lora_weights="gaussian",
        )
        
        transformer = get_peft_model(transformer, lora_config)
        transformer.print_trainable_parameters()
        
        return transformer, None
    else:
        # 使用自定义 LoRA
        from zimage_trainer.networks.lora import LoRANetwork
        
        network = LoRANetwork(
            transformer,
            rank=args.network_dim,
            alpha=args.network_alpha,
            target_modules=adapter.get_lora_target_modules(),
        )
        network.apply_to()
        
        logger.info(f"LoRA parameters: {network.num_parameters():,}")
        
        return transformer, network


def train(args, adapter: ModelAdapter):
    """主训练循环"""
    # 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    
    if args.seed is not None:
        set_seed(args.seed)
    
    # 确定数据类型
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    device = accelerator.device
    
    # 加载模型
    logger.info(f"Loading {adapter.name} transformer...")
    transformer = adapter.load_transformer(
        args.dit,
        device="cpu",  # 先加载到 CPU
        dtype=weight_dtype,
    )
    
    # 设置 LoRA
    transformer, lora_network = setup_lora(args, adapter, transformer)
    
    # 梯度检查点
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # 创建数据加载器
    dataloader = create_dataloader(args)
    
    # 计算训练步数
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )
    
    # 创建优化器和调度器
    if lora_network is not None:
        params = lora_network.parameters()
    else:
        params = [p for p in transformer.parameters() if p.requires_grad]
    
    optimizer = create_optimizer(args, params)
    lr_scheduler = create_lr_scheduler(args, optimizer, args.max_train_steps)
    
    # Accelerate 准备
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    
    # 内存优化
    memory_optimizer = MemoryOptimizer(device)
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练信息
    logger.info("***** Running Turbo LoRA Training *****")
    logger.info(f"  Model type: {adapter.name}")
    logger.info(f"  Turbo steps: {args.turbo_steps}")
    logger.info(f"  Num examples: {len(dataloader.dataset)}")
    logger.info(f"  Num epochs: {args.num_train_epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps: {args.max_train_steps}")
    logger.info(f"  LoRA rank: {args.network_dim}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    
    # 训练循环
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        transformer.train()
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
            disable=not accelerator.is_local_main_process,
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(transformer):
                # 获取 latents 和 text embeddings
                latents = batch["latents"].to(device, dtype=weight_dtype)
                text_embeds = batch["vl_embed"].to(device, dtype=weight_dtype)
                
                batch_size = latents.shape[0]
                
                # 如果 latents 是列表（bucket 模式）
                if isinstance(latents, list):
                    latents = latents[0]
                    text_embeds = text_embeds[0]
                    batch_size = latents.shape[0]
                
                # 采样时间步
                timesteps, sigmas = adapter.sample_timesteps(
                    batch_size=batch_size,
                    device=device,
                    dtype=weight_dtype,
                    use_anchor=args.use_anchor,
                )
                
                # 采样噪声
                noise = torch.randn_like(latents)
                
                # 添加噪声
                noisy_latents = adapter.add_noise(latents, noise, sigmas)
                
                # 打包 latents
                packed_latents, pack_info = adapter.pack_latents(noisy_latents)
                
                # 准备位置编码
                latent_h, latent_w = latents.shape[2], latents.shape[3]
                text_len = text_embeds.shape[1]
                position_ids = adapter.prepare_position_ids(
                    latent_h, latent_w, text_len, device, weight_dtype
                )
                
                # 图像序列长度（用于动态 shift）
                image_seq_len = packed_latents.shape[1]
                
                # 模型前向
                model_pred = adapter.forward(
                    transformer=transformer,
                    hidden_states=packed_latents,
                    encoder_hidden_states=text_embeds,
                    timesteps=timesteps,
                    position_ids=position_ids,
                )
                
                # 解包预测
                model_pred = adapter.unpack_latents(model_pred, pack_info)
                
                # 计算目标
                target = adapter.compute_velocity_target(latents, noise)
                
                # 计算损失
                loss = adapter.compute_loss(model_pred, target, reduction="batch")
                
                # SNR 加权
                if args.snr_gamma > 0:
                    snr_weights = compute_snr_weights(
                        sigmas,
                        snr_gamma=args.snr_gamma,
                        snr_floor=args.snr_floor,
                    ).to(device)
                    loss = (loss * snr_weights).mean()
                else:
                    loss = loss.mean()
                
                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 更新进度条
            if accelerator.sync_gradients:
                global_step += 1
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })
                
                # 按步保存
                if args.save_every_n_steps and global_step % args.save_every_n_steps == 0:
                    save_checkpoint(
                        args, accelerator, transformer, lora_network,
                        output_dir, f"step-{global_step}"
                    )
            
            if global_step >= args.max_train_steps:
                break
        
        # 按轮保存
        if args.save_every_n_epochs and (epoch + 1) % args.save_every_n_epochs == 0:
            save_checkpoint(
                args, accelerator, transformer, lora_network,
                output_dir, f"epoch-{epoch + 1}"
            )
    
    # 最终保存
    save_checkpoint(
        args, accelerator, transformer, lora_network,
        output_dir, "final"
    )
    
    logger.info(f"Training complete! Models saved to {output_dir}")


def save_checkpoint(args, accelerator, transformer, lora_network, output_dir, name):
    """保存检查点"""
    if not accelerator.is_main_process:
        return
    
    save_path = output_dir / f"{args.output_name}_{name}.safetensors"
    
    if lora_network is not None:
        # 自定义 LoRA
        lora_network.save(str(save_path))
    else:
        # PEFT LoRA
        if hasattr(transformer, "save_pretrained"):
            peft_dir = output_dir / f"{args.output_name}_{name}_peft"
            transformer.save_pretrained(str(peft_dir))
            logger.info(f"Saved PEFT model to {peft_dir}")
    
    logger.info(f"Saved checkpoint: {save_path}")


def main():
    args = parse_args()
    
    # 自动检测或获取适配器
    if args.model_type == "auto":
        detected = auto_detect_adapter(args.dit)
        if detected:
            logger.info(f"Auto-detected model type: {detected}")
            args.model_type = detected
        else:
            logger.error("Could not auto-detect model type. Please specify --model_type")
            sys.exit(1)
    
    # 获取适配器
    adapter = get_adapter(
        args.model_type,
        turbo_steps=args.turbo_steps,
        shift=args.shift,
        jitter_scale=args.jitter_scale,
    )
    
    logger.info(f"Using adapter: {adapter}")
    
    # 开始训练
    train(args, adapter)


if __name__ == "__main__":
    main()

