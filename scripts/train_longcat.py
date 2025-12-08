"""
[START] LongCat-Image Training Script

独立的 LongCat-Image LoRA 训练脚本
基于 FLUX 架构，支持动态 shift 和锚点采样

关键特性：
- 使用 LongCatImageTransformer2DModel
- 支持动态 shift (根据图像尺寸自动调整)
- FLUX 风格 2x2 latent packing
- 3D 位置编码 (modality, h, w)
"""

import os
import sys
import math
import signal
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

# 显存优化
from zimage_trainer.utils.memory_optimizer import MemoryOptimizer

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局中断标志
_interrupted = False

def signal_handler(signum, frame):
    """SIGINT/SIGTERM 信号处理器"""
    global _interrupted
    _interrupted = True
    logger.info("\n[STOP] 收到停止信号，将在当前步骤完成后保存并退出...")

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="LongCat-Image LoRA 训练脚本")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, help="超参数配置文件路径 (.toml)")
    parser.add_argument("--model_type", type=str, default="longcat", help="模型类型 (忽略，仅用于兼容)")
    
    # 模型路径
    parser.add_argument("--dit", type=str, help="Transformer 模型路径")
    parser.add_argument("--dataset_config", type=str, help="数据集配置文件")
    parser.add_argument("--output_dir", type=str, default="output/longcat", help="输出目录")
    
    # LongCat 特有参数 - 动态 shift
    parser.add_argument("--turbo_steps", type=int, default=10, help="目标加速步数")
    parser.add_argument("--use_dynamic_shifting", type=bool, default=True, help="使用动态 shift")
    parser.add_argument("--base_shift", type=float, default=0.5, help="基础 shift 值")
    parser.add_argument("--max_shift", type=float, default=1.15, help="最大 shift 值")
    parser.add_argument("--jitter_scale", type=float, default=0.02, help="锚点抖动幅度")
    parser.add_argument("--use_anchor", type=bool, default=True, help="使用锚点采样")
    
    # LoRA 参数
    parser.add_argument("--network_dim", type=int, default=32, help="LoRA rank")
    parser.add_argument("--network_alpha", type=float, default=16.0, help="LoRA alpha")
    
    # 训练参数
    parser.add_argument("--optimizer_type", type=str, default="AdamW", 
        choices=["AdamW", "AdamW8bit", "Adafactor"], help="优化器类型")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    
    # LR Scheduler 参数
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="学习率调度器"
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Warmup 步数")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Cosine 调度器的循环次数")
    
    # Min-SNR 加权参数
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="Min-SNR gamma (0=禁用)")
    parser.add_argument("--snr_floor", type=float, default=0.1, help="Min-SNR 保底权重")
    
    # 训练控制
    parser.add_argument("--num_train_epochs", type=int, default=10, help="训练 Epoch 数")
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="保存间隔 (Epoch)")
    parser.add_argument("--output_name", type=str, default="longcat-lora", help="LoRA 输出文件名")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 高级功能
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="启用梯度检查点")
    parser.add_argument("--blocks_to_swap", type=int, default=0, 
        help="将多少个 transformer blocks 交换到 CPU，节省显存。"
             "16G显存建议设为 4-8，24G显存可不设置")
    
    # 数据加载参数
    parser.add_argument("--enable_bucket", action="store_true", default=True, help="启用分桶")
    parser.add_argument("--disable_bucket", action="store_true", help="禁用分桶")
    
    args = parser.parse_args()
    
    # 如果指定了配置文件，读取并覆盖默认值
    if args.config:
        import tomli
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
        parser.error("--dit is required (or set in config)")
    
    if not args.dataset_config and args.config:
        args.dataset_config = args.config
        
    return args


# ==================== LongCat Latent Dataset ====================

class LongCatLatentDataset(torch.utils.data.Dataset):
    """LongCat-Image Latent 数据集"""
    
    LATENT_ARCH = "lc"  # LongCat 使用 _lc.safetensors
    TE_SUFFIX = "_lc_te.safetensors"
    
    def __init__(self, datasets, shuffle=True):
        from safetensors.torch import load_file
        
        self.load_file = load_file
        self.shuffle = shuffle
        self.cache_files = []
        self.resolutions = []
        
        for ds_config in datasets:
            cache_dir = Path(ds_config['cache_directory'])
            repeats = ds_config.get('num_repeats', 1)
            resolution_limit = ds_config.get('resolution_limit', None)
            
            logger.info(f"Loading LongCat dataset from: {cache_dir} (repeats={repeats})")
            
            files, res_list = self._load_dataset(cache_dir, resolution_limit)
            
            if repeats > 1:
                files = files * repeats
                res_list = res_list * repeats
            
            self.cache_files.extend(files)
            self.resolutions.extend(res_list)
            
        if len(self.cache_files) == 0:
            raise ValueError("No valid LongCat cache files found")
            
        logger.info(f"Total LongCat samples: {len(self.cache_files)}")
    
    def _load_dataset(self, cache_dir, resolution_limit):
        files = []
        resolutions = []
        
        pattern = f"*_{self.LATENT_ARCH}.safetensors"
        latent_files = list(cache_dir.glob(pattern))
        
        for latent_path in latent_files:
            res = self._parse_resolution(latent_path.stem)
            
            if resolution_limit:
                h, w = res
                if max(h, w) > resolution_limit:
                    continue
            
            te_path = self._find_te_path(latent_path, cache_dir)
            
            if te_path and te_path.exists():
                files.append((latent_path, te_path))
                resolutions.append(res)
            
        return files, resolutions

    def _parse_resolution(self, name):
        parts = name.split('_')
        res = (1024, 1024)
        for part in parts:
            if 'x' in part and part.replace('x', '').isdigit():
                try:
                    w, h = map(int, part.split('x'))
                    res = (h, w)
                    break
                except:
                    pass
        return res

    def _find_te_path(self, latent_path, cache_dir):
        name = latent_path.stem
        parts = name.rsplit('_', 2)
        if len(parts) >= 3:
            base_name = parts[0]
        else:
            base_name = name.rsplit('_', 1)[0]
        
        return cache_dir / f"{base_name}{self.TE_SUFFIX}"
    
    def __len__(self):
        return len(self.cache_files)
    
    def __getitem__(self, idx):
        latent_path, te_path = self.cache_files[idx]
        
        # Load latent (LongCat: 16 channels)
        latent_data = self.load_file(str(latent_path))
        latent_key = next((k for k in latent_data.keys() if k.startswith('latents_')), None)
        if latent_key is None:
            raise ValueError(f"No latent key found in {latent_path}")
        latents = latent_data[latent_key]
        
        # 确保尺寸能被 2 整除 (for packing)
        C, H, W = latents.shape
        H_padded = ((H + 1) // 2) * 2
        W_padded = ((W + 1) // 2) * 2
        
        if H != H_padded or W != W_padded:
            latents = torch.nn.functional.pad(
                latents, 
                (0, W_padded - W, 0, H_padded - H),
                mode='reflect'
            )
        
        # Load text encoder output
        te_data = self.load_file(str(te_path))
        te_key = next((k for k in te_data.keys() if 'embed' in k.lower()), None)
        if te_key is None:
            raise ValueError(f"No embedding key found in {te_path}")
        text_embed = te_data[te_key]
        
        return {
            'latents': latents,
            'text_embed': text_embed,
        }


def longcat_collate_fn(batch):
    """LongCat 专用 collate 函数"""
    shapes = [item['latents'].shape for item in batch]
    all_same = all(s == shapes[0] for s in shapes)
    
    if all_same:
        latents = torch.stack([item['latents'] for item in batch])
    else:
        max_h = max(s[1] for s in shapes)
        max_w = max(s[2] for s in shapes)
        max_h = ((max_h + 1) // 2) * 2
        max_w = ((max_w + 1) // 2) * 2
        
        padded_latents = []
        for item in batch:
            lat = item['latents']
            c, h, w = lat.shape
            if h < max_h or w < max_w:
                lat = torch.nn.functional.pad(lat, (0, max_w - w, 0, max_h - h), value=0)
            padded_latents.append(lat)
        latents = torch.stack(padded_latents)
    
    text_embeds = [item['text_embed'] for item in batch]
    
    return {
        'latents': latents,
        'text_embeds': text_embeds,
    }


def create_longcat_dataloader(args):
    """创建 LongCat 数据加载器"""
    import tomli
    
    with open(args.dataset_config, "rb") as f:
        config = tomli.load(f)
    
    # 获取 dataset 配置
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
            }]
    
    if not datasets:
        raise ValueError("No datasets configured")
    
    batch_size = config.get('batch_size', 1)
    num_workers = config.get('num_workers', 4)
    
    dataset = LongCatLatentDataset(datasets=datasets)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=longcat_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader


# ==================== LongCat LoRA ====================

class LongCatLoRANetwork:
    """LongCat-Image LoRA 网络"""
    
    TARGET_MODULES = [
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        "ff.net.0.proj", "ff.net.2",
        "ff_context.net.0.proj", "ff_context.net.2",
    ]
    
    def __init__(self, transformer, lora_dim=32, alpha=16.0, dtype=None):
        self.transformer = transformer
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.lora_layers = {}
        self.dtype = dtype
        
        self._apply_lora()
    
    def _apply_lora(self):
        """应用 LoRA 到目标模块"""
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=self.lora_dim,
            lora_alpha=self.alpha,
            target_modules=self.TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
        )
        
        self.peft_model = get_peft_model(self.transformer, lora_config)
        logger.info(f"Applied LoRA to LongCat transformer")
        self.peft_model.print_trainable_parameters()
        
        # 将 LoRA 参数转换为正确的精度
        if self.dtype is not None:
            for name, param in self.peft_model.named_parameters():
                if param.requires_grad and "lora" in name.lower():
                    param.data = param.data.to(self.dtype)
    
    def get_trainable_params(self):
        """获取可训练参数"""
        return [p for p in self.peft_model.parameters() if p.requires_grad]
    
    def save_weights(self, save_path, dtype=None):
        """保存 LoRA 权重"""
        self.peft_model.save_pretrained(str(save_path.parent / save_path.stem))
        logger.info(f"Saved LoRA weights to {save_path.parent / save_path.stem}")


# ==================== 主训练逻辑 ====================

def main():
    args = parse_args()
    
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
    logger.info("[START] 启动 LongCat-Image LoRA 训练")
    logger.info("="*60)
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"Turbo 步数: {args.turbo_steps}")
    logger.info(f"动态 Shift: {args.use_dynamic_shifting}")
    logger.info(f"Base Shift: {args.base_shift}, Max Shift: {args.max_shift}")
    logger.info(f"LoRA rank: {args.network_dim}")
    
    # 1. 加载 LongCat Transformer
    logger.info("\n[LOAD] 加载 LongCat-Image Transformer...")
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # 使用 LongCatAdapter 加载
    from models.longcat.adapter import LongCatAdapter
    
    adapter = LongCatAdapter(
        turbo_steps=args.turbo_steps,
        use_dynamic_shifting=args.use_dynamic_shifting,
        base_shift=args.base_shift,
        max_shift=args.max_shift,
    )
    
    transformer = adapter.load_transformer(
        model_path=args.dit,
        device=accelerator.device,
        dtype=weight_dtype,
    )
    logger.info(f"Transformer dtype: {next(transformer.parameters()).dtype}")
    transformer.requires_grad_(False)
    transformer.train()
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("  [OK] 已启用梯度检查点")
    
    # 1.5 初始化显存优化器
    logger.info("\n[MEM] 初始化显存优化器...")
    if args.blocks_to_swap > 0:
        logger.info(f"  Blocks to swap: {args.blocks_to_swap}")
    memory_config = {
        'block_swap_enabled': args.blocks_to_swap > 0,
        'blocks_to_swap': args.blocks_to_swap,
        'memory_block_size': 256,
        'cpu_swap_buffer_size': 1024,
        'swap_threshold': 0.7,
        'swap_frequency': 5,
        'smart_prefetch': False,
        'swap_strategy': 'lru',
        'compressed_swap': False,
        'checkpoint_optimization': 'basic' if args.gradient_checkpointing else 'none',
    }
    memory_optimizer = MemoryOptimizer(memory_config)
    memory_optimizer.start()
    logger.info("  [OK] 显存优化器初始化完成")
    
    # 应用显存优化到 transformer
    if hasattr(transformer, 'apply_memory_optimization'):
        transformer.apply_memory_optimization(memory_optimizer)
        logger.info("  [INIT] 已应用显存优化策略")
    
    # 2. 创建 LoRA 网络
    logger.info(f"\n[SETUP] 创建 LongCat LoRA 网络 (rank={args.network_dim})...")
    network = LongCatLoRANetwork(
        transformer,
        lora_dim=args.network_dim,
        alpha=args.network_alpha,
        dtype=weight_dtype,  # 传入模型精度
    )
    
    # 检查 LoRA 层精度
    trainable_params = network.get_trainable_params()
    if trainable_params:
        logger.info(f"LoRA 参数 dtype: {trainable_params[0].dtype}")
    param_count = sum(p.numel() for p in trainable_params)
    logger.info(f"可训练参数: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # 3. 创建数据加载器
    logger.info("\n[DATA] 加载 LongCat 数据集...")
    dataloader = create_longcat_dataloader(args)
    logger.info(f"数据集大小: {len(dataloader)} batches")
    
    # 4. 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num Batches per Epoch = {len(dataloader)}")
    logger.info(f"  Gradient Accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total Optimization Steps = {max_train_steps}")
    
    print(f"[TRAINING_INFO] total_steps={max_train_steps} total_epochs={args.num_train_epochs}", flush=True)
    
    # 5. 创建优化器
    logger.info(f"\n[SETUP] 初始化优化器: {args.optimizer_type}")
    
    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer_type == "AdamW8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
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
        )
    
    # 6. 创建学习率调度器
    from diffusers.optimization import get_scheduler
    logger.info(f"[SCHED] 初始化调度器: {args.lr_scheduler}")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    # 7. Accelerator prepare
    transformer = network.peft_model
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    
    # 8. 训练循环
    logger.info("\n" + "="*60)
    logger.info("[TARGET] 开始训练")
    logger.info("="*60)
    
    global_step = 0
    ema_loss = None
    ema_decay = 0.99
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        transformer.train()
        
        for step, batch in enumerate(dataloader):
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
                
                # 计算图像序列长度 (用于动态 shift)
                image_seq_len = (H // 2) * (W // 2)  # packed 后的序列长度
                
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
                
                # Pack latents (FLUX 风格)
                packed_latents, pack_info = adapter.pack_latents(noisy_latents)
                
                # 准备位置编码
                text_len = text_embeds.shape[1] if text_embeds.dim() == 3 else text_embeds.shape[0]
                position_ids = adapter.prepare_position_ids(
                    latent_height=H,
                    latent_width=W,
                    text_len=text_len,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                
                # 前向传播
                model_output = adapter.forward(
                    transformer=transformer,
                    hidden_states=packed_latents,
                    encoder_hidden_states=text_embeds,
                    timesteps=timesteps,  # LongCat 内部会 * 1000
                    position_ids=position_ids,
                )
                
                # 调试信息（第一步时打印）
                if global_step == 0 and step == 0:
                    logger.info(f"[DEBUG] packed_latents dtype: {packed_latents.dtype}")
                    logger.info(f"[DEBUG] text_embeds dtype: {text_embeds.dtype}")
                    logger.info(f"[DEBUG] timesteps dtype: {timesteps.dtype}")
                    logger.info(f"[DEBUG] model_output dtype: {model_output.dtype}")
                    logger.info(f"[DEBUG] target_velocity dtype: {target_velocity.dtype}")
                
                # Unpack 输出
                model_pred = adapter.unpack_latents(model_output, pack_info)
                
                # 统一精度（模型可能输出 float32）
                model_pred = model_pred.to(dtype=weight_dtype)
                
                # 计算损失 (velocity prediction)
                loss = torch.nn.functional.mse_loss(model_pred, target_velocity)
                
                # 应用 SNR 加权
                if args.snr_gamma > 0:
                    snr = (1 - sigmas) ** 2 / (sigmas ** 2 + 1e-8)
                    snr_weights = torch.clamp(snr / args.snr_gamma, max=1.0)
                    snr_weights = torch.clamp(snr_weights, min=args.snr_floor)
                    loss = loss * snr_weights.mean()
                
                # 反向传播
                accelerator.backward(loss)
                
                # 即时进度反馈（每步都打印，让用户知道训练在进行）
                batch_loss = loss.item()
                print(f"[BATCH] epoch={epoch+1}/{args.num_train_epochs} batch={step+1}/{len(dataloader)} loss={batch_loss:.4f}", flush=True)
            
            # 优化器步进
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # EMA loss
                current_loss = loss.item()
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * current_loss
                
                current_lr = lr_scheduler.get_last_lr()[0]
                print(f"[STEP] {global_step}/{max_train_steps} epoch={epoch+1}/{args.num_train_epochs} loss={current_loss:.4f} ema_loss={ema_loss:.4f} lr={current_lr:.2e}", flush=True)
            
            # 执行内存优化 (清理缓存等)
            memory_optimizer.optimize_training_step()
            
            # 检查中断信号
            if _interrupted:
                logger.info(f"\n[STOP] 中断训练，保存当前进度...")
                interrupt_path = Path(args.output_dir) / f"{args.output_name}_interrupted_step{global_step}.safetensors"
                network.save_weights(interrupt_path, dtype=weight_dtype)
                logger.info(f"[SAVE] 已保存中断检查点: {interrupt_path}")
                memory_optimizer.stop()
                logger.info("[EXIT] 5秒后退出进程...")
                time.sleep(5)
                os._exit(0)
        
        # Epoch 结束，保存检查点
        if (epoch + 1) % args.save_every_n_epochs == 0:
            save_path = Path(args.output_dir) / f"{args.output_name}_epoch{epoch+1}.safetensors"
            network.save_weights(save_path, dtype=weight_dtype)
            logger.info(f"\n[SAVE] 保存检查点 (Epoch {epoch+1}): {save_path}")
    
    # 保存最终模型
    final_path = Path(args.output_dir) / f"{args.output_name}_final.safetensors"
    network.save_weights(final_path, dtype=weight_dtype)
    
    # 停止内存优化器并清理显存
    memory_optimizer.stop()
    
    # 清理 GPU 缓存
    del network, transformer, optimizer, lr_scheduler, dataloader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    logger.info("\n" + "="*60)
    logger.info(f"[OK] LongCat-Image 训练完成！")
    logger.info(f"最终模型: {final_path}")
    logger.info("="*60)
    
    # 5秒后强制退出进程，确保显存释放
    logger.info("\n[EXIT] 5秒后退出进程...")
    time.sleep(5)
    os._exit(0)


if __name__ == "__main__":
    main()
