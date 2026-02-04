"""
[FULL FEATURE] Z-Image AC-RF Training Script

Complete AC-RF (Anchor-Constrained Rectified Flow) training with all features.
Based on verified Phase 1 core + full train_acrf.py feature set.

Key Features:
- Local model with use_reentrant=False for frozen + checkpointing
- AC-RF trainer with anchor timesteps and sigmas
- Min-SNR weighting
- Multiple loss functions (L1, Cosine, Frequency, Style-Structure)
- MemoryOptimizer with block swapping
- Proper dtype handling (BF16)

Usage:
    accelerate launch --mixed_precision bf16 scripts/train_zimage_v2.py \
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

# Local imports
from zimage_trainer.networks.lora import LoRANetwork, ZIMAGE_TARGET_NAMES, ZIMAGE_ADALN_NAMES, EXCLUDE_PATTERNS
from zimage_trainer.dataset.dataloader import create_dataloader, create_reg_dataloader, get_reg_config
from zimage_trainer.acrf_trainer import ACRFTrainer
from zimage_trainer.utils.snr_utils import compute_snr_weights
from zimage_trainer.utils.l2_scheduler import L2RatioScheduler, create_l2_scheduler_from_args
from zimage_trainer.utils.timestep_aware_loss import TimestepAwareLossScheduler, create_timestep_aware_scheduler_from_args
from zimage_trainer.losses.frequency_aware_loss import FrequencyAwareLoss
from zimage_trainer.losses.style_structure_loss import LatentStyleStructureLoss

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
    parser = argparse.ArgumentParser(description="Z-Image AC-RF Training")
    parser.add_argument("--config", type=str, required=True, help="TOML config path")
    
    # Model
    parser.add_argument("--dit", type=str, default=None)
    parser.add_argument("--vae", type=str, default=None)
    
    # Training
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    
    # LoRA
    parser.add_argument("--network_dim", type=int, default=16)
    parser.add_argument("--network_alpha", type=float, default=16)
    parser.add_argument("--resume_lora", type=str, default=None,
        help="继续训练的 LoRA 路径 (.safetensors)，Rank 将从文件自动推断")
    
    # AC-RF / Turbo
    parser.add_argument("--turbo_steps", type=int, default=10)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--jitter_scale", type=float, default=0.02)
    parser.add_argument("--latent_jitter_scale", type=float, default=0.01)
    
    # SNR
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--snr_floor", type=float, default=0.1)
    
    # Loss weights
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_cosine", type=float, default=0.0)
    parser.add_argument("--enable_freq", type=bool, default=True)
    parser.add_argument("--lambda_freq", type=float, default=0.3)
    parser.add_argument("--alpha_hf", type=float, default=1.0)
    parser.add_argument("--beta_lf", type=float, default=0.2)
    parser.add_argument("--enable_style", type=bool, default=True)
    parser.add_argument("--lambda_style", type=float, default=0.3)
    parser.add_argument("--lambda_struct", type=float, default=1.0)
    
    # Style-structure sub-params
    parser.add_argument("--lambda_light", type=float, default=0.5)
    parser.add_argument("--lambda_color", type=float, default=0.3)
    parser.add_argument("--lambda_tex", type=float, default=0.5)
    
    # Curvature Penalty (曲率惩罚 - 鼓励更直的轨迹)
    parser.add_argument("--enable_curvature", type=bool, default=False,
        help="启用曲率惩罚 (鼓励锚点间匀速直线运动)")
    parser.add_argument("--lambda_curvature", type=float, default=0.05,
        help="曲率惩罚权重")
    parser.add_argument("--curvature_interval", type=int, default=10,
        help="每 N 步计算一次曲率惩罚 (减少计算开销)")
    parser.add_argument("--curvature_start_epoch", type=int, default=0,
        help="从第 N 个 epoch 开始启用曲率惩罚")
    
    # Drop Text (保持低 CFG 能力)
    parser.add_argument("--drop_text_ratio", type=float, default=0.0,
        help="丢弃文本条件的概率 (保持低 CFG 能力)，推荐 0.1")
    
    # CFG Training (Classifier-Free Guidance 训练)
    parser.add_argument("--cfg_training", type=bool, default=False,
        help="启用 CFG 训练 (正/负 prompt 对训练，与 Pipeline 一致)")
    parser.add_argument("--cfg_scale", type=float, default=7.0,
        help="CFG 训练使用的 guidance scale")
    parser.add_argument("--cfg_training_ratio", type=float, default=0.5,
        help="CFG 训练模式的比例 (0.5 = 50% 步数使用 CFG 模式)")
    
    # Dynamic Shift (与 Pipeline 保持一致)
    parser.add_argument("--use_dynamic_shift", type=bool, default=True,
        help="使用动态 shift (基于分辨率自动计算，与 Pipeline 一致)")
    parser.add_argument("--base_seq_len", type=int, default=256,
        help="动态 shift 基准序列长度")
    parser.add_argument("--max_seq_len", type=int, default=4096,
        help="动态 shift 最大序列长度")
    parser.add_argument("--base_shift", type=float, default=0.5,
        help="动态 shift 基准值")
    parser.add_argument("--max_shift", type=float, default=1.15,
        help="动态 shift 最大值")
    
    # Memory optimization
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--block_swap_enabled", type=bool, default=False)
    parser.add_argument("--attention_backend", type=str, default="flash",
        choices=["flash", "xformers", "sdpa", "math"],
        help="Attention backend (默认flash，如果显卡不支持会自动降级为sdpa)")
    
    # Turbo / RAFT mode
    parser.add_argument("--enable_turbo", type=bool, default=True)
    parser.add_argument("--raft_mode", type=bool, default=False)
    parser.add_argument("--free_stream_ratio", type=float, default=0.3)
    
    # L2 Ratio Schedule
    parser.add_argument("--l2_schedule_mode", type=str, default="constant",
        choices=["constant", "linear_increase", "linear_decrease", "step"],
        help="L2 ratio 调度模式")
    parser.add_argument("--l2_initial_ratio", type=float, default=None,
        help="L2 起始比例 (默认使用 free_stream_ratio)")
    parser.add_argument("--l2_final_ratio", type=float, default=None,
        help="L2 结束比例")
    parser.add_argument("--l2_milestones", type=str, default="",
        help="阶梯模式切换点 (epoch, 逗号分隔)")
    parser.add_argument("--l2_include_anchor", type=bool, default=False,
        help="L2 同时计算锚点时间步")
    parser.add_argument("--l2_anchor_ratio", type=float, default=0.3,
        help="L2 锚点时间步权重 (仅当 include_anchor=True 时生效)")
    
    # 时间步感知 Loss 权重
    parser.add_argument("--enable_timestep_aware_loss", type=bool, default=False,
        help="启用时间步分区动态 Loss 权重")
    parser.add_argument("--timestep_high_threshold", type=float, default=0.7,
        help="高噪声区阈值 (σ > 此值时重结构)")
    parser.add_argument("--timestep_low_threshold", type=float, default=0.3,
        help="低噪声区阈值 (σ < 此值时重纹理)")
    
    # LoRA 高级选项
    parser.add_argument("--train_adaln", type=bool, default=False,
        help="训练 AdaLN 调制层 (激进模式)")
    
    # Optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW8bit")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    
    # Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    
    args = parser.parse_args()
    
    # Load config from TOML
    if args.config:
        import toml
        config = toml.load(args.config)
        
        # Apply config values
        general_cfg = config.get("general", {})  # 新版 TOML 结构
        model_cfg = config.get("model", {})      # 旧版兼容
        training_cfg = config.get("training", {})
        lora_cfg = config.get("lora", {})
        acrf_cfg = config.get("acrf", {})
        advanced_cfg = config.get("advanced", {})
        
        # Model - 优先从 general，回退到 model
        args.dit = general_cfg.get("dit") or model_cfg.get("dit") or args.dit
        args.vae = general_cfg.get("vae") or model_cfg.get("vae") or args.vae
        args.output_dir = general_cfg.get("output_dir") or model_cfg.get("output_dir") or args.output_dir
        
        # Training
        if args.output_name is None:
            args.output_name = training_cfg.get("output_name", "zimage_lora")
            
        if args.num_train_epochs is None:
            args.num_train_epochs = training_cfg.get("num_train_epochs", 
                                    advanced_cfg.get("num_train_epochs", 10))
                                    
        if args.learning_rate is None:
            args.learning_rate = training_cfg.get("learning_rate", 1e-4)

        args.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps",
                                            advanced_cfg.get("gradient_accumulation_steps", args.gradient_accumulation_steps))
        
        # Seed (从 [training] 或 [advanced] 读取)
        args.seed = training_cfg.get("seed", advanced_cfg.get("seed", args.seed))
                                            
        if args.save_every_n_epochs is None:
            args.save_every_n_epochs = advanced_cfg.get("save_every_n_epochs", 1)
            
        args.gradient_checkpointing = training_cfg.get("gradient_checkpointing",
                                        advanced_cfg.get("gradient_checkpointing", args.gradient_checkpointing))
        
        # LoRA
        args.network_dim = lora_cfg.get("network_dim", args.network_dim)
        args.network_alpha = lora_cfg.get("network_alpha", args.network_alpha)
        args.resume_lora = lora_cfg.get("resume_lora", args.resume_lora)
        
        # AC-RF
        args.turbo_steps = acrf_cfg.get("turbo_steps", args.turbo_steps)
        args.shift = acrf_cfg.get("shift", args.shift)
        args.jitter_scale = acrf_cfg.get("jitter_scale", args.jitter_scale)
        args.latent_jitter_scale = acrf_cfg.get("latent_jitter_scale", args.latent_jitter_scale)
        
        # SNR
        args.snr_gamma = training_cfg.get("snr_gamma", acrf_cfg.get("snr_gamma", args.snr_gamma))
        args.snr_floor = acrf_cfg.get("snr_floor", args.snr_floor)
        
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
        # Style-structure sub-params
        args.lambda_light = training_cfg.get("lambda_light", args.lambda_light)
        args.lambda_color = training_cfg.get("lambda_color", args.lambda_color)
        args.lambda_tex = training_cfg.get("lambda_tex", args.lambda_tex)
        
        # Memory
        args.blocks_to_swap = advanced_cfg.get("blocks_to_swap", args.blocks_to_swap)
        args.block_swap_enabled = args.blocks_to_swap > 0
        
        # Turbo / RAFT mode
        args.enable_turbo = acrf_cfg.get("enable_turbo", args.enable_turbo)
        args.raft_mode = acrf_cfg.get("raft_mode", args.raft_mode)
        args.free_stream_ratio = acrf_cfg.get("free_stream_ratio", args.free_stream_ratio)
        
        # L2 Schedule
        args.l2_schedule_mode = acrf_cfg.get("l2_schedule_mode", args.l2_schedule_mode)
        args.l2_initial_ratio = acrf_cfg.get("l2_initial_ratio", args.l2_initial_ratio)
        args.l2_final_ratio = acrf_cfg.get("l2_final_ratio", args.l2_final_ratio)
        args.l2_milestones = acrf_cfg.get("l2_milestones", args.l2_milestones)
        args.l2_include_anchor = acrf_cfg.get("l2_include_anchor", args.l2_include_anchor)
        args.l2_anchor_ratio = acrf_cfg.get("l2_anchor_ratio", args.l2_anchor_ratio)
        
        # Curvature Penalty (曲率惩罚)
        args.enable_curvature = acrf_cfg.get("enable_curvature", getattr(args, 'enable_curvature', False))
        args.lambda_curvature = acrf_cfg.get("lambda_curvature", getattr(args, 'lambda_curvature', 0.05))
        args.curvature_interval = acrf_cfg.get("curvature_interval", getattr(args, 'curvature_interval', 10))
        args.curvature_start_epoch = acrf_cfg.get("curvature_start_epoch", getattr(args, 'curvature_start_epoch', 0))
        
        # Timestep-aware Loss
        args.enable_timestep_aware_loss = acrf_cfg.get("enable_timestep_aware_loss", 
                                          training_cfg.get("enable_timestep_aware_loss", args.enable_timestep_aware_loss))
        args.timestep_high_threshold = acrf_cfg.get("timestep_high_threshold", args.timestep_high_threshold)
        args.timestep_low_threshold = acrf_cfg.get("timestep_low_threshold", args.timestep_low_threshold)
        
        # CFG Training (Classifier-Free Guidance 训练)
        args.cfg_training = acrf_cfg.get("cfg_training", args.cfg_training)
        args.cfg_scale = acrf_cfg.get("cfg_scale", args.cfg_scale)
        args.cfg_training_ratio = acrf_cfg.get("cfg_training_ratio", args.cfg_training_ratio)
        
        # Dynamic Shift (与 Pipeline 保持一致)
        args.use_dynamic_shift = acrf_cfg.get("use_dynamic_shift", args.use_dynamic_shift)
        args.base_seq_len = acrf_cfg.get("base_seq_len", args.base_seq_len)
        args.max_seq_len = acrf_cfg.get("max_seq_len", args.max_seq_len)
        args.base_shift = acrf_cfg.get("base_shift", args.base_shift)
        args.max_shift = acrf_cfg.get("max_shift", args.max_shift)
        
        # LoRA 高级选项
        lora_cfg = config.get("lora", {})
        args.train_adaln = lora_cfg.get("train_adaln", args.train_adaln)
        
        # Optimizer
        args.optimizer_type = training_cfg.get("optimizer_type", args.optimizer_type)
        args.adafactor_relative_step = training_cfg.get("adafactor_relative_step", getattr(args, 'adafactor_relative_step', False))
        args.weight_decay = training_cfg.get("weight_decay", args.weight_decay)
        
        # Scheduler
        args.lr_scheduler = training_cfg.get("lr_scheduler", args.lr_scheduler)
        args.lr_warmup_steps = training_cfg.get("lr_warmup_steps", args.lr_warmup_steps)
        args.lr_num_cycles = training_cfg.get("lr_num_cycles", args.lr_num_cycles)
        
    return args


def main():
    global _interrupted
    args = parse_args()
    
    # 解析 output_dir：如果是相对路径（如 "lora"），则基于 OUTPUT_PATH 环境变量
    output_base = os.environ.get("OUTPUT_PATH", "")
    if output_base and not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(output_base, args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # seed=-1 表示完全随机（不设置固定种子）
    if args.seed is not None and args.seed >= 0:
        set_seed(args.seed)
        logger.info(f"🎲 固定种子: {args.seed}")
    else:
        logger.info("🎲 随机模式: 每次训练使用不同的随机状态")
    
    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Z-Image AC-RF Training")
    logger.info("=" * 60)
    
    # 基本信息
    logger.info(f"📁 输出: {args.output_dir}/{args.output_name}")
    logger.info(f"🎯 模式: {'Turbo (' + str(args.turbo_steps) + ' steps)' if args.enable_turbo else '标准 Flow Matching'}")
    logger.info(f"⚡ 精度: {weight_dtype}")
    
    # 训练参数
    logger.info(f"\n📋 训练参数:")
    logger.info(f"   Epochs: {args.num_train_epochs} | LR: {args.learning_rate} | Grad Accum: {args.gradient_accumulation_steps}")
    logger.info(f"   LoRA: rank={args.network_dim}, alpha={args.network_alpha}")
    logger.info(f"   Optimizer: {args.optimizer_type} | Scheduler: {args.lr_scheduler}")
    
    # AC-RF 参数
    logger.info(f"\n⚙️ AC-RF 参数:")
    logger.info(f"   Shift: {args.shift} | Jitter: {args.jitter_scale} | Latent Jitter: {args.latent_jitter_scale}")
    logger.info(f"   SNR Gamma: {args.snr_gamma} | SNR Floor: {args.snr_floor}")
    if args.raft_mode:
        logger.info(f"   RAFT: ON (L2 ratio={args.free_stream_ratio})")
    
    # Loss 配置
    loss_cfg = f"L1×{args.lambda_l1} + Cos×{args.lambda_cosine}"
    if args.enable_freq:
        loss_cfg += f" + Freq×{args.lambda_freq}(hf={args.alpha_hf},lf={args.beta_lf})"
    if args.enable_style:
        loss_cfg += f" + Style×{args.lambda_style}"
    logger.info(f"\n📊 Loss 配置:")
    logger.info(f"   {loss_cfg}")
    if getattr(args, 'enable_timestep_aware_loss', False):
        logger.info(f"   🎛 时间步感知: ON (早期重结构, 后期重纹理)")
    if getattr(args, 'enable_curvature', False):
        logger.info(f"   🔄 曲率惩罚: ON (λ={getattr(args, 'lambda_curvature', 0.05)}, interval={getattr(args, 'curvature_interval', 10)}, start_epoch={getattr(args, 'curvature_start_epoch', 0)})")
    
    logger.info("\n[1/7] 加载 Transformer...")
    
    # 直接使用 diffusers 官方 ZImageTransformer2DModel
    from diffusers import ZImageTransformer2DModel
    logger.info("  ✓ 使用 diffusers ZImageTransformer2DModel")
    
    transformer = ZImageTransformer2DModel.from_pretrained(
        args.dit,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    transformer = transformer.to(accelerator.device)
    
    # =========================================================================
    # 2. 应用优化 (通过 Hook，无需修改模型源码)
    # =========================================================================
    from zimage_trainer.utils.model_hooks import apply_all_optimizations
    
    # 获取 attention backend 配置 (默认优先使用 flash)
    attention_backend = getattr(args, 'attention_backend', 'flash')

    # 智能降级检测: 如果指定了 flash 但显卡不支持 (Compute Capability < 8.0)，自动降级为 sdpa
    if attention_backend == 'flash' and torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                logger.warning(f"⚠️ 检测到 GPU 架构 (Compute {major}.x) 不支持 FlashAttention (需 Ampere/Compute 8.0+)。")
                logger.warning("   🔄 自动降级为 'sdpa' (Scaled Dot Product Attention) 以保证正常运行。")
                attention_backend = 'sdpa'
        except Exception as e:
            logger.warning(f"⚠️ 无法检测 GPU 架构，保持默认设置: {e}")
    
    optimization_results = apply_all_optimizations(
        transformer,
        blocks_to_swap=args.blocks_to_swap,
        attention_backend=attention_backend,
        gradient_checkpointing=args.gradient_checkpointing,
        device=accelerator.device,
        verbose=True,
    )
    block_swapper = optimization_results.get("block_swapper")
    
    transformer.train()
    
    # =========================================================================
    # 3. Apply LoRA with proper dtype
    # =========================================================================
    
    # 继续训练模式：从已有 LoRA 文件推断 rank
    if args.resume_lora and os.path.exists(args.resume_lora):
        logger.info(f"\n[RESUME] 继续训练模式: {args.resume_lora}")
        from safetensors.torch import load_file
        state_dict = load_file(args.resume_lora)
        # 从权重推断 rank (找第一个 lora_down 权重)
        for key, value in state_dict.items():
            if "lora_down" in key and value.dim() == 2:
                args.network_dim = value.shape[0]  # down 的 out_features 就是 rank
                logger.info(f"  [RESUME] 从权重推断 rank = {args.network_dim}")
                break
        else:
            logger.warning("  [RESUME] 无法推断 rank，使用默认值")
    
    logger.info("\n[2/7] 创建 LoRA (rank={args.network_dim})...")
    
    # 动态构建 target_names 和 exclude_patterns
    target_names = list(ZIMAGE_TARGET_NAMES)
    exclude_patterns = list(EXCLUDE_PATTERNS)
    
    train_adaln = getattr(args, 'train_adaln', False)
    # 确保 train_adaln 是布尔值 (TOML 可能返回字符串)
    if isinstance(train_adaln, str):
        train_adaln = train_adaln.lower() in ('true', '1', 'yes')
    train_adaln = bool(train_adaln)
    
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
    
    # 继续训练模式：加载已有权重
    if args.resume_lora and os.path.exists(args.resume_lora):
        network.load_weights(args.resume_lora)
        logger.info(f"  [RESUME] 已加载 LoRA 权重: {os.path.basename(args.resume_lora)}")
    
    # CRITICAL: Convert LoRA params to same dtype as model (BF16)
    network.to(accelerator.device, dtype=weight_dtype)
    
    # Freeze base model (LoRA params remain trainable)
    transformer.requires_grad_(False)
    
    # Get only LoRA trainable params
    trainable_params = []
    for lora_module in network.lora_modules.values():
        trainable_params.extend(lora_module.get_trainable_params())
    
    param_count = sum(p.numel() for p in trainable_params)
    logger.info(f"  ✓ 参数量: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # =========================================================================
    # 4. AC-RF Trainer
    # =========================================================================
    logger.info("\n[3/7] 初始化 AC-RF Trainer...")
    
    # 解析 use_dynamic_shift 参数 (可能是字符串)
    use_dynamic_shift = getattr(args, 'use_dynamic_shift', True)
    if isinstance(use_dynamic_shift, str):
        use_dynamic_shift = use_dynamic_shift.lower() in ('true', '1', 'yes')
    use_dynamic_shift = bool(use_dynamic_shift)
    
    acrf_trainer = ACRFTrainer(
        num_train_timesteps=1000,
        turbo_steps=args.turbo_steps,
        shift=args.shift,
        use_dynamic_shift=use_dynamic_shift,
        base_seq_len=getattr(args, 'base_seq_len', 256),
        max_seq_len=getattr(args, 'max_seq_len', 4096),
        base_shift=getattr(args, 'base_shift', 0.5),
        max_shift=getattr(args, 'max_shift', 1.15),
    )
    acrf_trainer.verify_setup()
    
    # CFG 训练配置
    cfg_training = getattr(args, 'cfg_training', False)
    if isinstance(cfg_training, str):
        cfg_training = cfg_training.lower() in ('true', '1', 'yes')
    cfg_training = bool(cfg_training)
    cfg_scale = getattr(args, 'cfg_scale', 7.0)
    cfg_training_ratio = getattr(args, 'cfg_training_ratio', 0.5)
    
    if cfg_training:
        logger.info(f"  [CFG] CFG 训练已启用: scale={cfg_scale}, ratio={cfg_training_ratio}")
    
    # =========================================================================
    # 5. Loss Functions
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
    
    # RAFT L2 混合模式
    if isinstance(args.raft_mode, str):
        args.raft_mode = args.raft_mode.lower() in ('true', '1', 'yes')
    args.raft_mode = bool(args.raft_mode)
    
    # 时间步感知 Loss 权重调度器
    timestep_aware_scheduler = create_timestep_aware_scheduler_from_args(args)
    
    # =========================================================================
    # 6. DataLoader
    # =========================================================================
    logger.info("\n[5/7] 加载数据集...")
    args.dataset_config = args.config
    dataloader = create_dataloader(args)
    logger.info(f"  ✓ {len(dataloader)} batches")
    
    # 正则数据集加载 (防止过拟合)
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
    # 7. Optimizer and Scheduler
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
        relative_step = getattr(args, 'adafactor_relative_step', False)
        if isinstance(relative_step, str):
            relative_step = relative_step.lower() in ('true', '1', 'yes')
        relative_step = bool(relative_step)
        
        if relative_step:
            # 自适应学习率模式
            optimizer = Adafactor(
                trainable_params,
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
            )
            logger.info("  📊 Adafactor (自适应 LR 模式)")
        else:
            # 固定学习率模式
            optimizer = Adafactor(
                trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay,
                scale_parameter=False, relative_step=False
            )
    elif args.optimizer_type == "Prodigy":
        try:
            from prodigyopt import Prodigy
            # Prodigy 是自适应学习率优化器，建议 LR=1.0，内部自动调整
            prodigy_lr = args.learning_rate if args.learning_rate >= 0.1 else 1.0
            optimizer = Prodigy(
                trainable_params, 
                lr=prodigy_lr,
                weight_decay=args.weight_decay,
                safeguard_warmup=True,
                use_bias_correction=True,
            )
            logger.info(f"  🧒 Prodigy 优化器 (自适应 LR, d0=1.0)")
        except ImportError:
            logger.warning("  ⚠ prodigyopt 未安装 (pip install prodigyopt)，使用 AdamW")
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "Lion":
        try:
            from lion_pytorch import Lion
            optimizer = Lion(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            logger.info("  🦁 Lion 优化器 (显存低)")
        except ImportError:
            logger.warning("  ⚠ lion-pytorch 未安装 (pip install lion-pytorch)，使用 AdamW")
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "Lion8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Lion8bit(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            logger.info("  🦁 Lion8bit 优化器 (显存最低)")
        except ImportError:
            logger.warning("  ⚠ bitsandbytes 未安装，使用标准 AdamW")
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "AdamWFP8":
        try:
            from zimage_trainer.optimizers import AdamWFP8
            optimizer = AdamWFP8(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            logger.info("  🔢 AdamWFP8 优化器 (FP8 E4M3/E5M2 混合精度)")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"  ⚠ AdamWFP8 不可用 ({e})，使用标准 AdamW")
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == "AdamWBF16":
        try:
            from zimage_trainer.optimizers import AdamWBF16
            optimizer = AdamWBF16(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            logger.info("  🔢 AdamWBF16 优化器 (BF16 状态存储，显存节省 50%)")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"  ⚠ AdamWBF16 不可用 ({e})，使用标准 AdamW")
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:  # AdamW
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Prepare with accelerator FIRST (before calculating steps)
    optimizer, dataloader, lr_scheduler_placeholder = accelerator.prepare(
        optimizer, dataloader, None
    )
    
    # Calculate max_train_steps AFTER prepare (len(dataloader) is already divided by num_gpus)
    max_train_steps = len(dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    
    # 检查是否使用 Adafactor 自适应学习率模式
    use_adafactor_schedule = (
        args.optimizer_type == "Adafactor" and 
        getattr(args, 'adafactor_relative_step', False)
    )
    
    if use_adafactor_schedule:
        # Adafactor 自适应模式内部自动管理学习率，不需要外部 lr_scheduler
        # 注意：不能使用 AdafactorSchedule，因为 accelerator.prepare() 返回的是 AcceleratedOptimizer
        lr_scheduler = None
        logger.info("  📈 Adafactor 自适应 LR 模式（无外部调度器）")
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=max_train_steps,
            num_cycles=args.lr_num_cycles,
        )
    
    logger.info(f"Total training steps: {max_train_steps}")

    # Initialize TensorBoard writer
    writer = None
    if accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        # 日志统一写到 OUTPUT_PATH/logs 目录（优先使用环境变量，避免路径计算错误）
        output_base = os.environ.get("OUTPUT_PATH", "")
        if not output_base:
            # 回退：使用 output_dir 的父目录
            output_base = os.path.dirname(args.output_dir)
        logging_dir = os.path.join(output_base, "logs", args.output_name)
        os.makedirs(logging_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=logging_dir)
        logger.info(f"TensorBoard log directory: {logging_dir}")
    
    logger.info(f"  ✓ 训练轮数: {args.num_train_epochs}, 总步数: {max_train_steps}")
    
    # =========================================================================
    # 8. Training Loop
    # =========================================================================
    logger.info("\n[7/7] 开始训练...")
    logger.info("=" * 60)
    
    # 创建 L2 调度器
    l2_scheduler = create_l2_scheduler_from_args(args)
    
    global_step = 0
    micro_step = 0  # 实际 batch 计数器（用于曲率惩罚间隔）
    ema_loss = None
    ema_decay = 0.99
    last_curv_loss = 0.0  # 持久化曲率值（打印时使用）
    
    # Loss 累积变量（TensorBoard 标准做法）
    accumulated_loss = 0.0
    accumulated_l1 = 0.0
    accumulated_cos = 0.0
    accumulated_freq = 0.0
    accumulated_style = 0.0
    accumulated_l2 = 0.0
    accumulation_count = 0
    
    # Kohya 风格: 5步滑动平均 (avr_loss)
    loss_history = []
    
    for epoch in range(args.num_train_epochs):
        if _interrupted:
            logger.info("[EXIT] Training interrupted by user")
            # 紧急保存当前权重
            if accelerator.is_main_process and global_step > 0:
                emergency_path = Path(args.output_dir) / f"{args.output_name}_interrupted_step{global_step}.safetensors"
                network.save_weights(str(emergency_path), dtype=weight_dtype)
                logger.info(f"[SAVE] Emergency checkpoint saved: {emergency_path}")
            break
        
        # 获取当前 epoch 的 L2 ratio
        current_l2_ratio = l2_scheduler.get_ratio(epoch + 1) if l2_scheduler else getattr(args, 'free_stream_ratio', 0.3)
        
        # 只在 RAFT 模式启用时显示 L2 ratio
        if args.raft_mode:
            logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs} [L2={current_l2_ratio:.2f}]")
        else:
            logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=True)):
            if _interrupted:
                # 中途中断，保存当前进度
                if accelerator.is_main_process and global_step > 0:
                    emergency_path = Path(args.output_dir) / f"{args.output_name}_interrupted_step{global_step}.safetensors"
                    network.save_weights(str(emergency_path), dtype=weight_dtype)
                    logger.info(f"[SAVE] Emergency checkpoint saved: {emergency_path}")
                break
                
            with accelerator.accumulate(transformer):
                # Get data
                latents = batch['latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']
                vl_embed = [v.to(accelerator.device, dtype=weight_dtype) for v in vl_embed]
                
                batch_size = latents.shape[0]
                
                # === Drop Text (保持低 CFG 能力) ===
                # 以一定概率丢弃文本条件，让模型学习无条件生成新风格
                drop_text_ratio = getattr(args, 'drop_text_ratio', 0.0)
                if drop_text_ratio > 0 and torch.rand(1).item() < drop_text_ratio:
                    # 创建空文本嵌入 (全零或很小的值)
                    vl_embed = [torch.zeros_like(v) for v in vl_embed]
                
                # Generate noise
                noise = torch.randn_like(latents)
                
                # AC-RF sampling (timestep with jitter)
                # use_anchor=True: Turbo 锚点采样, use_anchor=False: 标准 Flow Matching
                noisy_latents, timesteps, target_velocity = acrf_trainer.sample_batch(
                    latents, noise, jitter_scale=args.jitter_scale, use_anchor=args.enable_turbo
                )
                
                # Latent jitter (optional)
                if args.latent_jitter_scale > 0:
                    latent_jitter = torch.randn_like(noisy_latents) * args.latent_jitter_scale
                    noisy_latents = noisy_latents + latent_jitter
                    target_velocity = noise - latents
                
                # Prepare model input - Z-Image expects List[Tensor(C, 1, H, W)]
                model_input = noisy_latents.unsqueeze(2)
                
                # CRITICAL: For frozen model + checkpointing, input must have requires_grad=True
                if args.gradient_checkpointing:
                    model_input.requires_grad_(True)
                
                model_input_list = list(model_input.unbind(dim=0))
                
                # Timestep normalization (Z-Image uses (1000-t)/1000)
                timesteps_normalized = (1000 - timesteps) / 1000.0
                timesteps_normalized = timesteps_normalized.to(dtype=weight_dtype)
                
                # === CFG 训练模式 ===
                # 以一定概率使用 CFG 模式: 同时计算正向和负向（空 prompt）的输出
                # 这样模型可以学习区分有条件和无条件生成
                use_cfg_mode = cfg_training and torch.rand(1).item() < cfg_training_ratio
                
                if use_cfg_mode:
                    # CFG 模式: 双倍 batch (正向 + 负向)
                    # 负向 prompt = 空 embedding (与 Pipeline 一致)
                    negative_embed = [torch.zeros_like(v) for v in vl_embed]
                    
                    # 合并输入: [positive + negative]
                    cfg_input_list = model_input_list + model_input_list  # 双倍输入
                    cfg_timesteps = timesteps_normalized.repeat(2)
                    cfg_embed = vl_embed + negative_embed  # 正向 + 负向
                    
                    # Forward pass (CFG 模式)
                    cfg_pred_list = transformer(
                        x=cfg_input_list,
                        t=cfg_timesteps,
                        cap_feats=cfg_embed,
                    )[0]
                    
                    # 分离正向和负向输出
                    pos_pred_list = cfg_pred_list[:batch_size]
                    neg_pred_list = cfg_pred_list[batch_size:]
                    
                    pos_pred = torch.stack(pos_pred_list, dim=0).squeeze(2)
                    neg_pred = torch.stack(neg_pred_list, dim=0).squeeze(2)
                    
                    # Z-Image output is negated
                    pos_pred = -pos_pred
                    neg_pred = -neg_pred
                    
                    # CFG 组合: pred = pos + scale * (pos - neg)
                    model_pred = pos_pred + cfg_scale * (pos_pred - neg_pred)
                    
                    # 目标也需要调整 (CFG 训练的目标是让组合后的输出接近原始目标)
                    # 简化处理: 仍使用原始 target_velocity
                else:
                    # 标准模式 (无 CFG)
                    model_pred_list = transformer(
                        x=model_input_list,
                        t=timesteps_normalized,
                        cap_feats=vl_embed,
                    )[0]
                    
                    # Stack outputs
                    model_pred = torch.stack(model_pred_list, dim=0)
                    model_pred = model_pred.squeeze(2)
                    
                    # Z-Image output is negated
                    model_pred = -model_pred
                
                # =========================================================
                # Compute Losses
                # =========================================================
                
                # 获取时间步感知权重 (如果启用)
                ts_weights = None
                if timestep_aware_scheduler:
                    ts_weights = timestep_aware_scheduler.get_mean_weights(timesteps, num_train_timesteps=1000)
                
                loss_components = {}
                
                # 初始化总损失为0，然后累加各分量
                loss = torch.tensor(0.0, device=model_pred.device, dtype=model_pred.dtype)
                
                # L1 Loss
                l1_loss_val = F.l1_loss(model_pred, target_velocity)
                if args.lambda_l1 > 0:
                    loss = loss + args.lambda_l1 * l1_loss_val
                    loss_components['l1'] = l1_loss_val.item()
                
                # Cosine Loss
                cos_loss_val = 0.0
                if args.lambda_cosine > 0:
                    cos_loss = 1 - F.cosine_similarity(
                        model_pred.flatten(1), target_velocity.flatten(1), dim=1
                    ).mean()
                    loss = loss + args.lambda_cosine * cos_loss
                    cos_loss_val = cos_loss.item()
                    loss_components['cosine'] = cos_loss_val
                
                # Frequency Loss (requires noisy_latents and timesteps)
                # 应用时间步感知权重缩放
                freq_loss_val = 0.0
                if freq_loss_fn and args.lambda_freq > 0:
                    freq_loss = freq_loss_fn(model_pred, target_velocity, noisy_latents, timesteps, num_train_timesteps=1000)
                    freq_scale = ts_weights['lambda_freq_scale'] if ts_weights else 1.0
                    loss = loss + args.lambda_freq * freq_scale * freq_loss
                    freq_loss_val = freq_loss.item()
                loss_components['freq'] = freq_loss_val
                
                # Style-Structure Loss (requires noisy_latents and timesteps)
                # 应用时间步感知权重缩放
                style_loss_val = 0.0
                if style_loss_fn and args.lambda_style > 0:
                    style_loss = style_loss_fn(model_pred, target_velocity, noisy_latents, timesteps, num_train_timesteps=1000)
                    style_scale = ts_weights['lambda_style_scale'] if ts_weights else 1.0
                    loss = loss + args.lambda_style * style_scale * style_loss
                    style_loss_val = style_loss.item()
                loss_components['style'] = style_loss_val
                
                # === RAFT: L2 混合模式 (锚点流 + 自由流) ===
                l2_loss_val = 0.0
                raft_mode = getattr(args, 'raft_mode', False)
                free_stream_ratio = getattr(args, 'free_stream_ratio', 0.3)
                
                if raft_mode and free_stream_ratio > 0:
                    # 自由流: 全时间步均匀随机采样
                    free_sigmas = torch.rand(batch_size, device=latents.device, dtype=weight_dtype)
                    # Z-Image shift 变换
                    shift = args.shift if hasattr(args, 'shift') else 3.0
                    free_sigmas = (free_sigmas * shift) / (1 + (shift - 1) * free_sigmas)
                    free_sigmas = free_sigmas.clamp(0.001, 0.999)
                    
                    # 构造自由流加噪 latents
                    sigma_bc = free_sigmas.view(batch_size, 1, 1, 1)
                    free_noisy = sigma_bc * noise + (1 - sigma_bc) * latents
                    free_target = noise - latents  # v-prediction
                    
                    # 自由流前向传播 (参与梯度)
                    free_input = free_noisy.unsqueeze(2)
                    if args.gradient_checkpointing:
                        free_input.requires_grad_(True)
                    free_input_list = list(free_input.unbind(dim=0))
                    
                    free_t = 1000 * free_sigmas  # 转回 timestep
                    free_t_norm = (1000 - free_t) / 1000.0
                    free_t_norm = free_t_norm.to(dtype=weight_dtype)
                    
                    free_pred_list = transformer(
                        x=free_input_list,
                        t=free_t_norm,
                        cap_feats=vl_embed,
                    )[0]
                    
                    free_pred = torch.stack(free_pred_list, dim=0).squeeze(2)
                    
                    # Z-Image output is negated (与锚点流一致)
                    free_pred = -free_pred
                    
                    # 自由流 L2 损失 (不参与 SNR 加权!)
                    l2_loss = F.mse_loss(free_pred, free_target)
                    l2_loss_val = l2_loss.item()
                    
                    # 如果 l2_include_anchor=True，额外在锚点上计算 L2
                    l2_include_anchor = getattr(args, 'l2_include_anchor', False)
                    if l2_include_anchor:
                        # 锚点 L2: 使用已有的 model_pred 和 target_velocity
                        # 权重由 l2_anchor_ratio 控制
                        l2_anchor_ratio = getattr(args, 'l2_anchor_ratio', 0.3)
                        anchor_l2 = F.mse_loss(model_pred, target_velocity)
                        l2_loss = l2_loss + (l2_anchor_ratio * anchor_l2)
                        l2_loss_val = l2_loss.item()
                        
                loss_components['L2'] = l2_loss_val
                
                # === SNR 加权策略 (v2 协作架构) ===
                # 对锚点流损失 (L1+Freq+Style) 和自由流 L2 统一应用 SNR 加权
                # 这确保了高噪区不会被 L2 主导，锚点约束保持有效
                
                snr_weights = compute_snr_weights(
                    timesteps=timesteps,  # 锚点流的 timesteps
                    num_train_timesteps=1000,
                    snr_gamma=args.snr_gamma,
                    snr_floor=args.snr_floor,
                    prediction_type="v_prediction",
                )
                snr_weights = snr_weights.to(device=loss.device, dtype=weight_dtype)
                snr_mean = snr_weights.mean()
                
                # 锚点流损失加权
                anchor_loss_weighted = loss * snr_mean
                
                # 自由流 L2 不加 SNR 权重（设计意图：L2 只区分是否包含锚点）
                if l2_loss_val > 0:
                    loss = anchor_loss_weighted + current_l2_ratio * l2_loss
                else:
                    loss = anchor_loss_weighted
                
                # NaN check
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[NaN] Loss is NaN/Inf at step {global_step}, skipping backward. Components: {loss_components}")
                    optimizer.zero_grad()
                    continue
                
                # === Curvature Penalty (曲率惩罚) ===
                # 鼓励相邻锚点间做匀速直线运动，减少跳跃误差
                curvature_loss_val = 0.0
                # 更新 micro-step 计数器（每个实际 batch +1）
                micro_step += 1
                
                if (getattr(args, 'enable_curvature', False) and 
                    args.lambda_curvature > 0 and
                    epoch >= getattr(args, 'curvature_start_epoch', 0) and
                    micro_step % getattr(args, 'curvature_interval', 10) == 0):
                    
                    # 获取当前锚点索引和 sigma
                    anchor_sigmas = acrf_trainer.anchor_sigmas  # 实际锚点 sigma 值
                    current_sigma = timesteps.float() / 1000.0  # 当前采样的 sigma
                    
                    # 动态计算 dt：使用实际锚点间距（自动适应不同步数和 shift）
                    # 使用平均锚点间距作为 dt
                    num_anchors = len(anchor_sigmas)
                    if num_anchors > 1:
                        # 计算平均间距（考虑 shift 变换后的非均匀分布）
                        dt = (anchor_sigmas[0] - anchor_sigmas[-1]).abs().item() / (num_anchors - 1)
                    else:
                        dt = 0.1  # 回退默认值
                    
                    sigma_plus = (current_sigma + dt).clamp(0.001, 0.999)
                    sigma_minus = (current_sigma - dt).clamp(0.001, 0.999)
                    
                    # 构造 t+dt 和 t-dt 的加噪 latents
                    sigma_plus_bc = sigma_plus.view(batch_size, 1, 1, 1)
                    sigma_minus_bc = sigma_minus.view(batch_size, 1, 1, 1)
                    
                    noisy_plus = sigma_plus_bc * noise + (1 - sigma_plus_bc) * latents
                    noisy_minus = sigma_minus_bc * noise + (1 - sigma_minus_bc) * latents
                    
                    # 前向传播 (不计算梯度，节省显存)
                    with torch.no_grad():
                        # t + dt
                        input_plus = noisy_plus.to(dtype=weight_dtype).unsqueeze(2)
                        input_plus_list = list(input_plus.unbind(dim=0))
                        t_plus_norm = (1000 - sigma_plus * 1000) / 1000.0
                        t_plus_norm = t_plus_norm.to(dtype=weight_dtype)
                        pred_plus = transformer(x=input_plus_list, t=t_plus_norm, cap_feats=vl_embed)[0]
                        pred_plus = -torch.stack(pred_plus, dim=0).squeeze(2)
                        
                        # t - dt
                        input_minus = noisy_minus.to(dtype=weight_dtype).unsqueeze(2)
                        input_minus_list = list(input_minus.unbind(dim=0))
                        t_minus_norm = (1000 - sigma_minus * 1000) / 1000.0
                        t_minus_norm = t_minus_norm.to(dtype=weight_dtype)
                        pred_minus = transformer(x=input_minus_list, t=t_minus_norm, cap_feats=vl_embed)[0]
                        pred_minus = -torch.stack(pred_minus, dim=0).squeeze(2)
                    
                    # 计算曲率 (二阶差分): curvature = v+ - 2v + v-
                    # 理想情况: curvature ≈ 0 (匀速直线运动)
                    # 方案 B: pred_plus/pred_minus 是常数，只有 model_pred 有梯度
                    # 物理意义: 惩罚当前预测偏离相邻时间步的线性插值
                    curvature = pred_plus.detach() - 2 * model_pred + pred_minus.detach()
                    curvature_loss = (curvature ** 2).mean()
                    
                    # 添加到总损失
                    loss = loss + args.lambda_curvature * curvature_loss
                    curvature_loss_val = curvature_loss.item()
                    last_curv_loss = curvature_loss_val  # 持久化保存
                
                loss_components['curvature'] = last_curv_loss  # 使用持久化值
                
                # 累积 loss 用于平均计算（TensorBoard 标准做法）
                accumulated_loss += loss.detach().float().item()
                accumulated_l1 += loss_components.get('l1', 0)
                accumulated_cos += loss_components.get('cosine', 0)
                accumulated_freq += loss_components.get('freq', 0)
                accumulated_style += loss_components.get('style', 0)
                accumulated_l2 += loss_components.get('L2', 0)
                accumulation_count += 1
                
                # Cast loss to float32 for stable backward
                loss = loss.float()
                
                # Backward pass with error handling
                try:
                    accelerator.backward(loss)
                except RuntimeError as e:
                    logger.error(f"[BACKWARD ERROR] Step {global_step}, Loss={loss.item():.4f}")
                    logger.error(f"  Components: {loss_components}")
                    logger.error(f"  Error: {e}")
                    # Check for OOM
                    if "out of memory" in str(e).lower():
                        logger.error("  [OOM] GPU out of memory. Try reducing batch_size or enabling blocks_to_swap.")
                    raise
                
            # 梯度累积完成后执行优化步骤 (在 accumulate 块外)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 计算累积期间的平均 loss（TensorBoard 标准做法）
                avg_loss = accumulated_loss / max(accumulation_count, 1)
                avg_l1 = accumulated_l1 / max(accumulation_count, 1)
                avg_cos = accumulated_cos / max(accumulation_count, 1)
                avg_freq = accumulated_freq / max(accumulation_count, 1)
                avg_style = accumulated_style / max(accumulation_count, 1)
                avg_l2 = accumulated_l2 / max(accumulation_count, 1)
                
                # 重置累积变量
                accumulated_loss = 0.0
                accumulated_l1 = 0.0
                accumulated_cos = 0.0
                accumulated_freq = 0.0
                accumulated_style = 0.0
                accumulated_l2 = 0.0
                accumulation_count = 0
                
                # Update EMA loss（使用平均值）
                if ema_loss is None:
                    ema_loss = avg_loss
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * avg_loss
                
                # Kohya 风格: 5步滑动平均
                loss_history.append(avg_loss)
                if len(loss_history) > 5:
                    loss_history.pop(0)
                avr_loss = sum(loss_history) / len(loss_history)
                
                # Get current learning rate
                if lr_scheduler is not None:
                    current_lr = lr_scheduler.get_last_lr()[0]
                else:
                    # Adafactor 自适应模式：尝试多种方式获取学习率
                    try:
                        # 方法1：尝试从 AcceleratedOptimizer 获取原始 optimizer
                        orig_opt = getattr(optimizer, 'optimizer', optimizer)
                        if hasattr(orig_opt, '_get_lr'):
                            # Adafactor 的 _get_lr 方法
                            group = orig_opt.param_groups[0]
                            param = group['params'][0]
                            if param in orig_opt.state:
                                current_lr = orig_opt._get_lr(group, orig_opt.state[param])
                            else:
                                current_lr = 0.0
                        else:
                            current_lr = optimizer.param_groups[0].get('lr', 0.0) or 0.0
                    except Exception:
                        current_lr = 0.0
                
                # Print progress for frontend parsing (CRITICAL: exact format required)
                # 只让主进程打印日志，避免多卡训练时日志混乱
                if accelerator.is_main_process:
                    curv = last_curv_loss
                    print(f"[STEP] {global_step}/{max_train_steps} epoch={epoch+1}/{args.num_train_epochs} loss={avg_loss:.4f} avr={avr_loss:.4f} ema={ema_loss:.4f} l1={avg_l1:.4f} cos={avg_cos:.4f} freq={avg_freq:.4f} style={avg_style:.4f} L2={avg_l2:.4f} curv={curv:.4f} lr={current_lr:.2e}", flush=True)
                    
                    if writer:
                        writer.add_scalar("train/loss", avg_loss, global_step)
                        writer.add_scalar("train/avr_loss", avr_loss, global_step)
                        writer.add_scalar("train/ema_loss", ema_loss, global_step)
                        writer.add_scalar("train/l1_loss", avg_l1, global_step)
                        writer.add_scalar("train/cosine_loss", avg_cos, global_step)
                        writer.add_scalar("train/freq_loss", avg_freq, global_step)
                        writer.add_scalar("train/style_loss", avg_style, global_step)
                        writer.add_scalar("train/l2_loss", avg_l2, global_step)
                        writer.add_scalar("train/curvature_loss", curv, global_step)
                        writer.add_scalar("train/learning_rate", current_lr, global_step)
                
                # ========== 正则训练步骤 (按比例执行) ==========
                # 正则化步骤在主训练步骤完成后独立执行，不参与梯度累积周期
                if reg_dataloader and reg_ratio > 0:
                    # 边界检查：reg_ratio 应在 (0, 1] 范围内
                    effective_reg_ratio = min(max(reg_ratio, 0.01), 1.0)
                    # 按比例决定是否执行正则步骤：ratio=0.5 表示每2步执行1次正则
                    reg_interval = max(1, int(1.0 / effective_reg_ratio))
                    if global_step % reg_interval == 0:
                        # 获取正则 batch
                        if reg_iterator is None:
                            reg_iterator = iter(reg_dataloader)
                        try:
                            reg_batch = next(reg_iterator)
                        except StopIteration:
                            reg_iterator = iter(reg_dataloader)
                            reg_batch = next(reg_iterator)
                        
                        # 正则前向传播 (独立步骤，不使用 accumulate 包装)
                        reg_latents = reg_batch['latents'].to(accelerator.device, dtype=weight_dtype)
                        reg_vl_embed = reg_batch['vl_embed']
                        reg_vl_embed = [v.to(accelerator.device, dtype=weight_dtype) for v in reg_vl_embed]
                        
                        reg_noise = torch.randn_like(reg_latents)
                        reg_noisy, reg_t, reg_target = acrf_trainer.sample_batch(
                            reg_latents, reg_noise, jitter_scale=args.jitter_scale, use_anchor=args.enable_turbo
                        )
                        
                        reg_input = reg_noisy.unsqueeze(2)
                        if args.gradient_checkpointing:
                            reg_input.requires_grad_(True)
                        reg_input_list = list(reg_input.unbind(dim=0))
                        reg_t_norm = (1000 - reg_t) / 1000.0
                        
                        reg_pred_list = transformer(
                            x=reg_input_list,
                            t=reg_t_norm.to(dtype=weight_dtype),
                            cap_feats=reg_vl_embed,
                        )[0]
                        reg_pred = -torch.stack(reg_pred_list, dim=0).squeeze(2)
                        
                        # 简单 L2 损失，保持模型原有能力
                        reg_loss = F.mse_loss(reg_pred, reg_target) * reg_weight
                        reg_loss = reg_loss.float()  # 与主损失一致，使用 float32 反向传播
                        
                        # 独立的优化步骤 (不参与梯度累积)
                        accelerator.backward(reg_loss)
                        accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                        optimizer.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()  # 修复：正则化步骤也需要更新学习率调度器
                        optimizer.zero_grad()
        
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
    
    logger.info("\n[DONE] Training complete!")


    if writer:
        writer.close()

if __name__ == "__main__":
    main()
