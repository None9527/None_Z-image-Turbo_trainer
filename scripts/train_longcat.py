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
from zimage_trainer.utils.snr_utils import compute_snr_weights
from zimage_trainer.losses import FrequencyAwareLoss, LatentStyleStructureLoss
import torch.nn.functional as F

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
    
    # 损失权重参数
    parser.add_argument("--lambda_l1", type=float, default=1.0, help="Charbonnier/L1 Loss 权重")
    parser.add_argument("--lambda_cosine", type=float, default=0.1, help="Cosine Loss 权重")
    
    # 频域感知损失 (开关+权重+子参数)
    parser.add_argument("--enable_freq", action="store_true", help="启用频域感知损失")
    parser.add_argument("--lambda_freq", type=float, default=0.3, help="频域感知 Loss 权重")
    
    # 风格结构损失 (开关+权重+子参数)
    parser.add_argument("--enable_style", action="store_true", help="启用风格结构损失")
    parser.add_argument("--lambda_style", type=float, default=0.3, help="风格结构 Loss 权重")
    
    # L2 损失独立采样配置（全时间步随机采样，不使用锚点）
    parser.add_argument("--lambda_mse", type=float, default=0.0, help="L2/MSE Loss 权重 (0=禁用)")
    parser.add_argument("--mse_use_anchor", type=bool, default=False, help="L2 是否使用锚点 (False=全时间步随机)")
    
    # RAFT 混合模式参数 (同 batch 混合锚点流+自由流)
    parser.add_argument("--free_stream_ratio", type=float, default=0.3, help="自由流比例 (0.3=30%% 全时间步随机)")
    parser.add_argument("--raft_mode", action="store_true", help="启用 RAFT 同 batch 混合模式")
    
    # Latent Jitter: 空间抠动 (垂直于流线方向，真正改变构图的关键)
    parser.add_argument("--latent_jitter_scale", type=float, default=0.0, help="Latent 空间抠动幅度 (0=禁用, 推荐 0.03-0.05)")
    
    # 频域感知 Loss 子参数
    parser.add_argument("--alpha_hf", type=float, default=1.0, help="高频增强权重")
    parser.add_argument("--beta_lf", type=float, default=0.2, help="低频锁定权重")
    parser.add_argument("--lf_magnitude_weight", type=float, default=0.0, help="低频幅度约束")
    parser.add_argument("--downsample_factor", type=int, default=4, help="低频提取降采样因子")
    
    # 风格结构 Loss 子参数
    parser.add_argument("--lambda_struct", type=float, default=1.0, help="结构锁权重")
    parser.add_argument("--lambda_light", type=float, default=0.5, help="光影学习权重")
    parser.add_argument("--lambda_color", type=float, default=0.3, help="色调迁移权重")
    parser.add_argument("--lambda_tex", type=float, default=0.5, help="质感增强权重")
    
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
    
    # 确保布尔值正确解析 (TOML 可能返回字符串)
    for bool_arg in ['use_anchor', 'use_dynamic_shifting', 'gradient_checkpointing', 'raft_mode']:
        val = getattr(args, bool_arg, None)
        if isinstance(val, str):
            setattr(args, bool_arg, val.lower() in ('true', '1', 'yes'))
        elif val is not None:
            setattr(args, bool_arg, bool(val))
        
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
    )
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    
    logger.info("="*60)
    logger.info("[START] 启动 LongCat-Image LoRA 训练")
    logger.info("="*60)
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"Turbo 步数: {args.turbo_steps}")
    logger.info(f"锚点采样: {args.use_anchor}")
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
    
    # Enable gradient checkpointing BEFORE freeze (critical order)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("  [OK] 已启用梯度检查点")
    
    transformer.train()
    
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
    
    # 2. 创建 LoRA 网络 (BEFORE freeze - critical order)
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
    
    # Freeze base model AFTER LoRA is applied (critical order)
    # This ensures LoRA params remain trainable while base model is frozen
    for name, param in transformer.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
    logger.info("  [FREEZE] 基础模型已冻结 (LoRA 参数保持可训练)")
    
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
    
    # 8. 初始化损失函数 (基于权重判断)
    logger.info(f"\n[LOSS] 自由组合损失模式")
    logger.info(f"  [基础] lambda_l1={args.lambda_l1}, lambda_cosine={args.lambda_cosine}")
    
    frequency_loss_fn = None
    style_loss_fn = None
    
    # 频域感知损失 (开关控制)
    enable_freq = getattr(args, 'enable_freq', False)
    if enable_freq:
        frequency_loss_fn = FrequencyAwareLoss(
            alpha_hf=args.alpha_hf,
            beta_lf=args.beta_lf,
            base_weight=1.0,
            lf_magnitude_weight=args.lf_magnitude_weight,
            downsample_factor=args.downsample_factor,
        )
        logger.info(f"  [频域感知] ✅ 启用 lambda={args.lambda_freq}, alpha_hf={args.alpha_hf}")
    
    # 风格结构损失 (开关控制)
    enable_style = getattr(args, 'enable_style', False)
    if enable_style:
        style_loss_fn = LatentStyleStructureLoss(
            lambda_base=1.0,
            lambda_struct=args.lambda_struct,
            lambda_light=args.lambda_light,
            lambda_color=args.lambda_color,
            lambda_tex=args.lambda_tex,
        )
        logger.info(f"  [风格结构] ✅ 启用 lambda={args.lambda_style}, struct={args.lambda_struct}")
    
    # 9. 训练循环
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
                
                # === Latent Jitter: 空间抠动 (垂直于流线，改变构图的关键) ===
                latent_jitter_scale = getattr(args, 'latent_jitter_scale', 0.0)
                if latent_jitter_scale > 0:
                    # 在 x_t 上添加空间抖动
                    latent_jitter = torch.randn_like(noisy_latents) * latent_jitter_scale
                    noisy_latents = noisy_latents + latent_jitter
                    # target 保持指向真实 x_0 (noise - latents)
                
                # Pack latents (FLUX 风格)
                packed_latents, pack_info = adapter.pack_latents(noisy_latents)
                
                # 梯度检查点需要输入有梯度
                if args.gradient_checkpointing:
                    packed_latents.requires_grad_(True)
                
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

                
                # Unpack 输出
                model_pred = adapter.unpack_latents(model_output, pack_info)
                
                # 统一精度（模型可能输出 float32）
                model_pred = model_pred.to(dtype=weight_dtype)
                
                # 计算损失
                loss_components = {}
                
                # 计算 Min-SNR 权重
                if args.snr_gamma > 0:
                    snr_weights = compute_snr_weights(
                        timesteps=timesteps,
                        num_train_timesteps=1000,
                        snr_gamma=args.snr_gamma,
                        snr_floor=args.snr_floor,
                        prediction_type="v_prediction",
                    ).to(model_pred.device, dtype=torch.float32)
                else:
                    snr_weights = None
                
                # === Charbonnier Loss (Robust L1, 基础损失) ===
                diff = model_pred - target_velocity
                loss_l1 = torch.sqrt(diff**2 + 1e-6).mean()
                loss_components['l1'] = loss_l1.item()
                
                # === Cosine Loss (方向一致性) ===
                cos_loss_val = 0.0
                if args.lambda_cosine > 0:
                    pred_flat = model_pred.view(model_pred.shape[0], -1)
                    target_flat = target_velocity.view(target_velocity.shape[0], -1)
                    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
                    loss_cosine = 1.0 - cos_sim
                    cos_loss_val = loss_cosine.item()
                else:
                    loss_cosine = torch.tensor(0.0, device=model_pred.device)
                loss_components['cosine'] = cos_loss_val
                
                # === 自由组合损失 (权重控制) ===
                # 基础损失: L1 + Cosine
                loss = args.lambda_l1 * loss_l1 + args.lambda_cosine * loss_cosine
                
                # 可选: 频域感知损失
                if enable_freq and frequency_loss_fn is not None:
                    freq_loss, freq_comps = frequency_loss_fn(
                        pred_v=model_pred,
                        target_v=target_velocity,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        num_train_timesteps=1000,
                        return_components=True,
                    )
                    loss = loss + args.lambda_freq * freq_loss
                    loss_components['freq'] = freq_loss.item()
                
                # 可选: 风格结构损失
                if enable_style and style_loss_fn is not None:
                    style_loss, style_comps = style_loss_fn(
                        pred_v=model_pred,
                        target_v=target_velocity,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        num_train_timesteps=1000,
                        return_components=True,
                    )
                    loss = loss + args.lambda_style * style_loss
                    loss_components['style'] = style_loss.item()
                
                # === RAFT: 同 Batch 混合模式 (锚点流 + 自由流) ===
                raft_mode = getattr(args, 'raft_mode', False)
                free_stream_ratio = getattr(args, 'free_stream_ratio', 0.3)
                lambda_mse = getattr(args, 'lambda_mse', 0.0)
                
                if raft_mode and free_stream_ratio > 0:
                    # RAFT 模式: 同 batch 内混合计算自由流损失
                    
                    # 自由流: 全时间步随机采样
                    free_sigmas = torch.rand(batch_size, device=latents.device, dtype=latents.dtype)
                    # LongCat 动态 shift
                    mu = adapter._compute_mu(image_seq_len)
                    shift = torch.exp(torch.tensor(mu, device=latents.device))
                    free_sigmas = (free_sigmas * shift) / (1 + (shift - 1) * free_sigmas)
                    free_sigmas = free_sigmas.clamp(0.001, 0.999)
                    
                    # 构造自由流加噪 latents
                    sigma_broadcast = free_sigmas.view(batch_size, 1, 1, 1)
                    free_noisy = sigma_broadcast * noise + (1 - sigma_broadcast) * latents
                    free_target = noise - latents  # v-prediction
                    
                    # Pack 并前向传播 (参与梯度!)
                    free_packed, free_pack_info = adapter.pack_latents(free_noisy)
                    free_timesteps = free_sigmas
                    
                    free_output = adapter.forward(
                        transformer=transformer,
                        hidden_states=free_packed,
                        encoder_hidden_states=text_embeds,
                        timesteps=free_timesteps,
                        position_ids=position_ids,
                    )
                    free_pred = adapter.unpack_latents(free_output, free_pack_info)
                    free_pred = free_pred.to(dtype=weight_dtype)
                    
                    # 自由流 L2 损失
                    loss_free = F.mse_loss(free_pred, free_target)
                    
                    # RAFT 混合: loss_total = loss_anchor + ratio * loss_free
                    loss = loss + free_stream_ratio * loss_free
                    loss_components['loss_free'] = loss_free.item()
                
                elif lambda_mse > 0:
                    # 兼容旧版: 独立 L2 损失 (不参与梯度)
                    mse_use_anchor = getattr(args, 'mse_use_anchor', False)
                    if mse_use_anchor:
                        mse_pred = model_pred
                        mse_target = target_velocity
                    else:
                        mse_sigmas = torch.rand(batch_size, device=latents.device, dtype=latents.dtype)
                        mu = adapter._compute_mu(image_seq_len)
                        shift = torch.exp(torch.tensor(mu, device=latents.device))
                        mse_sigmas = (mse_sigmas * shift) / (1 + (shift - 1) * mse_sigmas)
                        mse_sigmas = mse_sigmas.clamp(0.001, 0.999)
                        
                        sigma_broadcast = mse_sigmas.view(batch_size, 1, 1, 1)
                        mse_noisy = sigma_broadcast * noise + (1 - sigma_broadcast) * latents
                        mse_target = noise - latents
                        
                        mse_packed, mse_pack_info = adapter.pack_latents(mse_noisy)
                        mse_timesteps = mse_sigmas
                        
                        with torch.no_grad():
                            mse_output = adapter.forward(
                                transformer=transformer,
                                hidden_states=mse_packed,
                                encoder_hidden_states=text_embeds,
                                timesteps=mse_timesteps,
                                position_ids=position_ids,
                            )
                        mse_pred = adapter.unpack_latents(mse_output, mse_pack_info)
                        mse_pred = mse_pred.to(dtype=weight_dtype)
                    
                    loss_mse = F.mse_loss(mse_pred, mse_target)
                    loss = loss + lambda_mse * loss_mse
                    loss_components['mse'] = loss_mse.item()
                
                # 应用 SNR 加权
                if snr_weights is not None:
                    loss = loss * snr_weights.mean()
                
                # 确保 Loss 为 Float32 (避免 Mixed Precision backward error)
                if loss.dtype != torch.float32:
                    loss = loss.to(dtype=torch.float32)
                
                # 反向传播
                accelerator.backward(loss)
            
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
                
                # 打印进度供前端解析
                l1 = loss_components.get('l1', 0)
                cosine = loss_components.get('cosine', 0)
                freq = loss_components.get('freq', 0)
                style = loss_components.get('style', 0)
                free = loss_components.get('loss_free', 0)
                print(f"[STEP] {global_step}/{max_train_steps} epoch={epoch+1}/{args.num_train_epochs} loss={current_loss:.4f} ema={ema_loss:.4f} l1={l1:.4f} cos={cosine:.4f} freq={freq:.4f} style={style:.4f} L2={free:.4f} lr={current_lr:.2e}", flush=True)
            
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
