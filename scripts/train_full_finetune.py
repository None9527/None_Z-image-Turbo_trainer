"""
[FULL FEATURE] Z-Image Full Finetune Training Script

直接微调主模型（不使用LoRA），需要大量显存（40GB+）。
基于 train_zimage_v2.py，移除 LoRA 网络，直接训练主模型参数。

注意事项：
- 显存需求极高（约40GB+），推荐使用 A100/H100
- 训练速度较慢，但效果可能更好
- 输出为完整模型权重（非 LoRA）

Usage:
    accelerate launch --mixed_precision bf16 scripts/train_full_finetune.py \
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
from zimage_trainer.utils.lr_schedulers import get_scheduler_with_onecycle
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import save_file

# Local imports
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
    parser = argparse.ArgumentParser(description="Z-Image Full Finetune Training")
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
    
    # Finetune 模块选择
    parser.add_argument("--trainable_modules", type=str, default="attention+mlp+adaln",
                       help="可选: all, attention, mlp, attention+mlp, attention+mlp+adaln")
    parser.add_argument("--freeze_embeddings", type=bool, default=True)
    
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
    
    # Curvature Penalty (曲率惩罚)
    parser.add_argument("--enable_curvature", type=bool, default=False)
    parser.add_argument("--lambda_curvature", type=float, default=0.05)
    parser.add_argument("--curvature_interval", type=int, default=10)
    parser.add_argument("--curvature_start_epoch", type=int, default=0)
    
    # Drop Text (保持低 CFG 能力)
    parser.add_argument("--drop_text_ratio", type=float, default=0.0)
    
    # CFG Training
    parser.add_argument("--cfg_training", type=bool, default=False)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--cfg_training_ratio", type=float, default=0.5)
    
    # Dynamic Shift
    parser.add_argument("--use_dynamic_shift", type=bool, default=True)
    parser.add_argument("--base_seq_len", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--base_shift", type=float, default=0.5)
    parser.add_argument("--max_shift", type=float, default=1.15)
    
    # Memory optimization
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--block_swap_enabled", type=bool, default=False)
    
    # Turbo / RAFT mode
    parser.add_argument("--enable_turbo", type=bool, default=True)
    parser.add_argument("--raft_mode", type=bool, default=False)
    parser.add_argument("--free_stream_ratio", type=float, default=0.3)
    
    # L2 Ratio Schedule
    parser.add_argument("--l2_schedule_mode", type=str, default="constant")
    parser.add_argument("--l2_initial_ratio", type=float, default=None)
    parser.add_argument("--l2_final_ratio", type=float, default=None)
    parser.add_argument("--l2_milestones", type=str, default="")
    parser.add_argument("--l2_include_anchor", type=bool, default=False)
    parser.add_argument("--l2_anchor_ratio", type=float, default=0.3)
    
    # Timestep-aware Loss
    parser.add_argument("--enable_timestep_aware_loss", type=bool, default=False)
    parser.add_argument("--timestep_high_threshold", type=float, default=0.7)
    parser.add_argument("--timestep_low_threshold", type=float, default=0.3)
    
    # Optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW8bit")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    
    # Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    # OneCycleLR specific
    parser.add_argument("--lr_pct_start", type=float, default=0.1)
    parser.add_argument("--lr_div_factor", type=float, default=10.0)
    parser.add_argument("--lr_final_div_factor", type=float, default=100.0)
    
    args = parser.parse_args()
    
    # Load config from TOML
    if args.config:
        import toml
        config = toml.load(args.config)
        
        # Apply config values (与 train_zimage_v2.py 保持一致)
        general_cfg = config.get("general", {})
        training_cfg = config.get("training", {})
        acrf_cfg = config.get("acrf", {})
        advanced_cfg = config.get("advanced", {})
        
        # General / Model
        args.dit = general_cfg.get("dit", args.dit)
        args.vae = general_cfg.get("vae", args.vae)
        args.output_dir = general_cfg.get("output_dir", args.output_dir)
        
        # Training (从 [training] 和 [advanced] 双处回退读取)
        if args.output_name is None:
            args.output_name = training_cfg.get("output_name", "zimage_finetune")
            
        if args.num_train_epochs is None:
            args.num_train_epochs = training_cfg.get("num_train_epochs", 
                                    advanced_cfg.get("num_train_epochs", 10))
                                    
        if args.learning_rate is None:
            args.learning_rate = training_cfg.get("learning_rate", 1e-5)

        args.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps",
                                            advanced_cfg.get("gradient_accumulation_steps", args.gradient_accumulation_steps))
        
        # Seed (从 [training] 或 [advanced] 读取)
        args.seed = training_cfg.get("seed", advanced_cfg.get("seed", args.seed))
                                            
        if args.save_every_n_epochs is None:
            args.save_every_n_epochs = advanced_cfg.get("save_every_n_epochs", 1)
            
        args.gradient_checkpointing = training_cfg.get("gradient_checkpointing",
                                        advanced_cfg.get("gradient_checkpointing", args.gradient_checkpointing))
        args.max_grad_norm = advanced_cfg.get("max_grad_norm", args.max_grad_norm)
        
        # AC-RF / ACRF
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
        
        # Timestep-aware Loss
        args.enable_timestep_aware_loss = acrf_cfg.get("enable_timestep_aware_loss", args.enable_timestep_aware_loss)
        args.timestep_high_threshold = acrf_cfg.get("timestep_high_threshold", args.timestep_high_threshold)
        args.timestep_low_threshold = acrf_cfg.get("timestep_low_threshold", args.timestep_low_threshold)
        
        # Dynamic Shift
        args.use_dynamic_shift = acrf_cfg.get("use_dynamic_shifting", acrf_cfg.get("use_dynamic_shift", args.use_dynamic_shift))
        args.base_seq_len = acrf_cfg.get("base_seq_len", args.base_seq_len)
        args.max_seq_len = acrf_cfg.get("max_seq_len", args.max_seq_len)
        args.base_shift = acrf_cfg.get("base_shift", args.base_shift)
        args.max_shift = acrf_cfg.get("max_shift", args.max_shift)
        
        # Curvature Penalty (曲率惩罚)
        args.enable_curvature = acrf_cfg.get("enable_curvature", getattr(args, 'enable_curvature', False))
        args.lambda_curvature = acrf_cfg.get("lambda_curvature", getattr(args, 'lambda_curvature', 0.05))
        args.curvature_interval = acrf_cfg.get("curvature_interval", getattr(args, 'curvature_interval', 10))
        args.curvature_start_epoch = acrf_cfg.get("curvature_start_epoch", getattr(args, 'curvature_start_epoch', 0))
        
        # CFG Training
        args.cfg_training = acrf_cfg.get("cfg_training", args.cfg_training)
        args.cfg_scale = acrf_cfg.get("cfg_scale", args.cfg_scale)
        args.cfg_training_ratio = acrf_cfg.get("cfg_training_ratio", args.cfg_training_ratio)
        
        # Optimizer
        args.optimizer_type = training_cfg.get("optimizer_type", args.optimizer_type)
        args.adafactor_relative_step = training_cfg.get("adafactor_relative_step", getattr(args, 'adafactor_relative_step', False))
        args.weight_decay = training_cfg.get("weight_decay", args.weight_decay)
        
        # Scheduler
        args.lr_scheduler = training_cfg.get("lr_scheduler", args.lr_scheduler)
        args.lr_warmup_steps = training_cfg.get("lr_warmup_steps", args.lr_warmup_steps)
        args.lr_num_cycles = training_cfg.get("lr_num_cycles", args.lr_num_cycles)
        # OneCycleLR specific
        args.lr_pct_start = training_cfg.get("lr_pct_start", getattr(args, 'lr_pct_start', 0.1))
        args.lr_div_factor = training_cfg.get("lr_div_factor", getattr(args, 'lr_div_factor', 10.0))
        args.lr_final_div_factor = training_cfg.get("lr_final_div_factor", getattr(args, 'lr_final_div_factor', 100.0))
        
        # Finetune 模块选择
        finetune_cfg = config.get("finetune", {})
        args.trainable_modules = finetune_cfg.get("trainable_modules", 
                                   training_cfg.get("trainable_modules", args.trainable_modules))
        args.freeze_embeddings = finetune_cfg.get("freeze_embeddings", 
                                   training_cfg.get("freeze_embeddings", args.freeze_embeddings))
    return args


def get_trainable_parameters(transformer, trainable_modules: str, freeze_embeddings: bool = True):
    """
    根据配置返回可训练的参数
    
    Args:
        trainable_modules: 可选值
            - "all": 全部参数
            - "attention": 仅 Attention 层
            - "mlp": 仅 MLP/FFN 层
            - "attention+mlp": Attention + MLP
            - "attention+mlp+adaln": Attention + MLP + AdaLN (推荐)
        freeze_embeddings: 是否冻结 embedding 层
    """
    # 先冻结所有参数
    transformer.requires_grad_(False)
    
    trainable_params = []
    trainable_count = 0
    frozen_count = 0
    
    # 解析 trainable_modules
    modules_to_train = set(trainable_modules.lower().replace(' ', '').split('+'))
    train_all = 'all' in modules_to_train
    train_attention = 'attention' in modules_to_train or train_all
    train_mlp = 'mlp' in modules_to_train or train_all
    train_adaln = 'adaln' in modules_to_train or train_all
    train_norm = 'norm' in modules_to_train or train_all
    
    for name, param in transformer.named_parameters():
        should_train = False
        name_lower = name.lower()
        
        # Attention 层: q_proj, k_proj, v_proj, o_proj, to_q, to_k, to_v, to_out
        if train_attention:
            if any(key in name_lower for key in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'to_q', 'to_k', 'to_v', 'to_out']):
                should_train = True
        
        # MLP 层: mlp, fc1, fc2, ffn, feed_forward, linear1, linear2
        if train_mlp:
            if any(key in name_lower for key in ['mlp', 'fc1', 'fc2', 'ffn', 'feed_forward', 'linear1', 'linear2']):
                should_train = True
        
        # AdaLN 层: adaln, scale_shift, modulation, t_embedder (时间步相关)
        if train_adaln:
            if any(key in name_lower for key in ['adaln', 'scale_shift', 'modulation', 't_embedder', 'time_embed']):
                should_train = True
        
        # Norm 层 (非 AdaLN)
        if train_norm:
            if any(key in name_lower for key in ['norm', 'ln', 'layer_norm', 'layernorm', 'rmsnorm']):
                if not any(key in name_lower for key in ['adaln']):
                    should_train = True
        
        # 冻结 embedding 层
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
    """保存 Transformer 模型完整权重（使用 state_dict 确保与加载时键名一致）"""
    # 使用 state_dict() 而不是 named_parameters()，确保键名与 load_state_dict 兼容
    state_dict = transformer.state_dict()
    # 转换为指定 dtype 并移到 CPU
    converted_state = {}
    for key, value in state_dict.items():
        converted_state[key] = value.to(dtype).cpu()
    save_file(converted_state, path)
    logger.info(f"[SAVE] 已保存完整模型 ({len(converted_state)} 个参数) 到 {path}")


def main():
    global _interrupted
    args = parse_args()
    
    # 解析 output_dir：如果是相对路径，基于 OUTPUT_PATH 环境变量
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
    logger.info("🔥 Z-Image Full Finetune Training (高显存模式)")
    logger.info("=" * 60)
    
    logger.warning("⚠️ 全量微调需要约 40GB+ 显存，请确认硬件支持")
    
    # =========================================================================
    # 1. Load Model (不冻结)
    # =========================================================================
    logger.info("\n[1/6] 加载 Transformer 模型...")
    
    try:
        from zimage_trainer.models.transformer_z_image import ZImageTransformer2DModel
        logger.info("  ✓ 使用本地 ZImageTransformer2DModel")
    except ImportError:
        from diffusers import ZImageTransformer2DModel
        logger.warning("  ⚠ 使用 diffusers 默认版本")
    
    transformer = ZImageTransformer2DModel.from_pretrained(
        args.dit,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    transformer = transformer.to(accelerator.device)
    
    # =========================================================================
    # 2. 选择性模块训练
    # =========================================================================
    logger.info(f"\n[2/6] 配置可训练模块 ({args.trainable_modules})...")
    
    trainable_params, frozen_count, trainable_count = get_trainable_parameters(
        transformer, 
        args.trainable_modules, 
        args.freeze_embeddings
    )
    
    total_params = frozen_count + trainable_count
    logger.info(f"  ✓ 可训练: {trainable_count:,} ({trainable_count/1e6:.2f}M, {100*trainable_count/total_params:.1f}%)")
    logger.info(f"  ✓ 冻结: {frozen_count:,} ({frozen_count/1e6:.2f}M, {100*frozen_count/total_params:.1f}%)")
    
    # 先启用 gradient checkpointing (在 prepare 之前)
    if args.gradient_checkpointing:
        if hasattr(transformer, 'enable_gradient_checkpointing'):
            transformer.enable_gradient_checkpointing()
        else:
            transformer.gradient_checkpointing = True
        logger.info("  [CKPT] Gradient checkpointing enabled")
    
    transformer.train()
    
    # =========================================================================
    # 2. Block Swapper (可选)
    # =========================================================================
    block_swapper = None
    if args.blocks_to_swap > 0:
        from zimage_trainer.utils.block_swapper import create_block_swapper
        logger.info(f"\n[SWAP] Initializing Block Swapper...")
        block_swapper = create_block_swapper(
            blocks_to_swap=args.blocks_to_swap,
            device=accelerator.device,
            verbose=True,
        )
        transformer.set_block_swapper(block_swapper)
        logger.info("  [OK] Block Swapper attached")
    
    # =========================================================================
    # 3. AC-RF Trainer
    # =========================================================================
    logger.info("\n[2/6] 初始化 AC-RF Trainer...")
    
    use_dynamic_shift = getattr(args, 'use_dynamic_shift', True)
    if isinstance(use_dynamic_shift, str):
        use_dynamic_shift = use_dynamic_shift.lower() in ('true', '1', 'yes')
    use_dynamic_shift = bool(use_dynamic_shift)
    
    acrf_trainer = ACRFTrainer(
        num_train_timesteps=1000,
        turbo_steps=args.turbo_steps,
        shift=args.shift,
        use_dynamic_shift=use_dynamic_shift,
        base_seq_len=args.base_seq_len,
        max_seq_len=args.max_seq_len,
        base_shift=args.base_shift,
        max_shift=args.max_shift,
    )
    logger.info(f"  ✓ Turbo Steps: {args.turbo_steps}, Shift: {args.shift}")
    logger.info(f"  ✓ Dynamic Shift: {use_dynamic_shift}")
    
    # =========================================================================
    # 4. Loss Functions
    # =========================================================================
    logger.info("\n[3/6] 初始化损失函数...")
    
    # 解析布尔参数（TOML 可能返回字符串）
    enable_freq = args.enable_freq
    if isinstance(enable_freq, str):
        enable_freq = enable_freq.lower() in ('true', '1', 'yes')
    enable_freq = bool(enable_freq)
    
    enable_style = args.enable_style
    if isinstance(enable_style, str):
        enable_style = enable_style.lower() in ('true', '1', 'yes')
    enable_style = bool(enable_style)
    
    raft_mode = args.raft_mode
    if isinstance(raft_mode, str):
        raft_mode = raft_mode.lower() in ('true', '1', 'yes')
    raft_mode = bool(raft_mode)
    
    freq_loss_fn = None
    if enable_freq:
        freq_loss_fn = FrequencyAwareLoss(alpha_hf=args.alpha_hf, beta_lf=args.beta_lf)
        logger.info(f"  ✓ Freq Loss: α_hf={args.alpha_hf}, β_lf={args.beta_lf}")
    
    style_loss_fn = None
    if enable_style:
        style_loss_fn = LatentStyleStructureLoss(
            lambda_light=args.lambda_light,
            lambda_color=args.lambda_color,
            lambda_tex=args.lambda_tex,
        )
        logger.info(f"  ✓ Style Loss: light={args.lambda_light}, color={args.lambda_color}")
    
    # Timestep-aware scheduler
    timestep_scheduler = None
    if args.enable_timestep_aware_loss:
        timestep_scheduler = create_timestep_aware_scheduler_from_args(args)
        logger.info("  ✓ Timestep-aware loss enabled")
    
    # =========================================================================
    # 5. DataLoader
    # =========================================================================
    logger.info("\n[4/6] 加载数据集...")
    args.dataset_config = args.config
    dataloader = create_dataloader(args)
    logger.info(f"  ✓ {len(dataloader)} batches")
    
    # 正则数据集
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
    logger.info("\n[5/6] 配置优化器...")
    
    # Full finetune 建议使用较低学习率
    if args.learning_rate > 1e-5:
        logger.warning(f"  ⚠️ 全量微调建议使用较低学习率 (当前: {args.learning_rate})")
    
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
            optimizer = Adafactor(
                trainable_params,
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
            )
            logger.info("  📊 Adafactor (自适应 LR 模式)")
        else:
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
            logger.info(f"  🧒 Prodigy 优化器 (自适应 LR)")
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
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    logger.info(f"  ✓ {args.optimizer_type}, LR={args.learning_rate}")
    
    # Prepare with accelerator (不包装 transformer，减少显存开销)
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)
    
    # 显存调试日志
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"  [MEM] After prepare(): Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
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
        lr_scheduler = get_scheduler_with_onecycle(
            args.lr_scheduler,
            optimizer=optimizer,
            num_training_steps=max_train_steps,
            num_warmup_steps=args.lr_warmup_steps,
            num_cycles=args.lr_num_cycles,
            max_lr=args.learning_rate,
            pct_start=getattr(args, 'lr_pct_start', 0.1),
            div_factor=getattr(args, 'lr_div_factor', 10.0),
            final_div_factor=getattr(args, 'lr_final_div_factor', 100.0),
        )
        if args.lr_scheduler == "one_cycle":
            logger.info(f"  🚀 OneCycleLR: {args.learning_rate/getattr(args, 'lr_div_factor', 10.0):.2e} → {args.learning_rate:.2e} → {args.learning_rate/getattr(args, 'lr_final_div_factor', 100.0):.2e}")
    
    logger.info(f"  ✓ 总步数: {max_train_steps}")
    
    # TensorBoard
    writer = None
    if accelerator.is_main_process:
        # 日志统一写到 OUTPUT_PATH/logs 目录（优先使用环境变量，避免路径计算错误）
        output_base = os.environ.get("OUTPUT_PATH", "")
        if not output_base:
            output_base = os.path.dirname(args.output_dir)  # output/finetune -> output
        logging_dir = os.path.join(output_base, "logs", args.output_name)
        os.makedirs(logging_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=logging_dir)
        logger.info(f"TensorBoard: {logging_dir}")
    
    # =========================================================================
    # 7. Training Loop
    # =========================================================================
    logger.info("\n[6/6] 开始训练...")
    logger.info("=" * 60)
    
    l2_scheduler = create_l2_scheduler_from_args(args)
    
    global_step = 0
    ema_loss = None
    ema_decay = 0.99
    
    # Loss 累积变量（TensorBoard 标准做法，与 LoRA 脚本一致）
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
            logger.info("[EXIT] Training interrupted")
            if accelerator.is_main_process and global_step > 0:
                emergency_path = Path(args.output_dir) / f"{args.output_name}_interrupted.safetensors"
                save_transformer_weights(transformer, str(emergency_path), dtype=weight_dtype)
            break
        
        # 获取当前 epoch 的 L2 ratio
        current_l2_ratio = l2_scheduler.get_ratio(epoch + 1) if l2_scheduler else args.free_stream_ratio
        
        if raft_mode:
            logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs} [L2={current_l2_ratio:.2f}]")
        else:
            logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=True)):
            if _interrupted:
                break
            
            with accelerator.accumulate(transformer):
                latents = batch['latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']
                vl_embed = [v.to(accelerator.device, dtype=weight_dtype) for v in vl_embed]
                
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                noisy_latents, timesteps, target = acrf_trainer.sample_batch(
                    latents, noise, jitter_scale=args.jitter_scale, use_anchor=args.enable_turbo
                )
                
                # Forward pass (transformer 需要 list 输入)
                model_input = noisy_latents.unsqueeze(2)  # (B, C, 1, H, W)
                model_input_list = list(model_input.unbind(dim=0))  # List of (C, 1, H, W)
                t_norm = (1000 - timesteps) / 1000.0
                
                pred_list = transformer(
                    x=model_input_list,
                    t=t_norm.to(dtype=weight_dtype),
                    cap_feats=vl_embed,
                )[0]
                pred = -torch.stack(pred_list, dim=0).squeeze(2)
                
                # Compute losses
                snr_weights = compute_snr_weights(
                    timesteps, snr_gamma=args.snr_gamma, snr_floor=args.snr_floor
                ).to(weight_dtype).squeeze()  # (B,1,1,1) -> (B,)
                
                # 获取时间步感知权重 (如果启用，与 LoRA 脚本一致)
                ts_weights = None
                if timestep_scheduler:
                    ts_weights = timestep_scheduler.get_mean_weights(timesteps, num_train_timesteps=1000)
                
                # L1 Loss
                l1_loss_raw = F.l1_loss(pred, target, reduction='none')
                l1_loss = (l1_loss_raw.mean(dim=(1, 2, 3)) * snr_weights).mean()
                total_loss = l1_loss * args.lambda_l1
                l1_val = l1_loss.detach().float().item()
                
                # Cosine Loss
                cos_val = 0.0
                if args.lambda_cosine > 0:
                    cos_loss = 1.0 - F.cosine_similarity(
                        pred.flatten(1), target.flatten(1), dim=1
                    ).mean()
                    total_loss = total_loss + cos_loss * args.lambda_cosine
                    cos_val = cos_loss.detach().float().item()
                
                # Freq Loss (应用时间步感知权重缩放)
                freq_val = 0.0
                if freq_loss_fn is not None and args.lambda_freq > 0:
                    freq_loss = freq_loss_fn(pred, target, noisy_latents, timesteps, num_train_timesteps=1000)
                    freq_scale = ts_weights['lambda_freq_scale'] if ts_weights else 1.0
                    total_loss = total_loss + freq_loss * args.lambda_freq * freq_scale
                    freq_val = freq_loss.detach().float().item()
                
                # Style Loss (应用时间步感知权重缩放)
                style_val = 0.0
                if style_loss_fn is not None and args.lambda_style > 0:
                    style_loss = style_loss_fn(pred, target, noisy_latents, timesteps, num_train_timesteps=1000)
                    style_scale = ts_weights['lambda_style_scale'] if ts_weights else 1.0
                    total_loss = total_loss + style_loss * args.lambda_style * style_scale
                    style_val = style_loss.detach().float().item()
                
                # RAFT L2 Loss
                l2_val = 0.0
                if raft_mode:
                    l2_loss_raw = F.mse_loss(pred, target, reduction='none')
                    l2_loss = (l2_loss_raw.mean(dim=(1, 2, 3)) * snr_weights).mean()
                    total_loss = total_loss + l2_loss * current_l2_ratio
                    l2_val = l2_loss.detach().float().item()
                
                # 累积 loss（与 LoRA 脚本一致）
                accumulated_loss += total_loss.detach().float().item()
                accumulated_l1 += l1_val
                accumulated_cos += cos_val
                accumulated_freq += freq_val
                accumulated_style += style_val
                accumulated_l2 += l2_val
                accumulation_count += 1
                
                # Backward (保持 bf16，避免不必要的 dtype 转换)
                accelerator.backward(total_loss)
            
            # 梯度累积完成后执行优化步骤 (在 accumulate 块外)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 计算累积期间的平均 loss（与 LoRA 脚本一致）
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
                
                # Update EMA loss
                if ema_loss is None:
                    ema_loss = avg_loss
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * avg_loss
                
                # Kohya 风格: 5步滑动平均
                loss_history.append(avg_loss)
                if len(loss_history) > 5:
                    loss_history.pop(0)
                avr_loss = sum(loss_history) / len(loss_history)
                
                # 打印日志（与 LoRA 脚本格式完全一致 + Kohya avr_loss）
                if accelerator.is_main_process:
                    if lr_scheduler is not None:
                        current_lr = lr_scheduler.get_last_lr()[0]
                    else:
                        # Adafactor 自适应模式：尝试多种方式获取学习率
                        try:
                            orig_opt = getattr(optimizer, 'optimizer', optimizer)
                            if hasattr(orig_opt, '_get_lr'):
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
                    print(f"[STEP] {global_step}/{max_train_steps} epoch={epoch+1}/{args.num_train_epochs} loss={avg_loss:.4f} avr={avr_loss:.4f} ema={ema_loss:.4f} l1={avg_l1:.4f} cos={avg_cos:.4f} freq={avg_freq:.4f} style={avg_style:.4f} L2={avg_l2:.4f} lr={current_lr:.2e}", flush=True)
                    
                    if writer:
                        writer.add_scalar("train/loss", avg_loss, global_step)
                        writer.add_scalar("train/avr_loss", avr_loss, global_step)
                        writer.add_scalar("train/ema_loss", ema_loss, global_step)
                        writer.add_scalar("train/l1_loss", avg_l1, global_step)
                        writer.add_scalar("train/cosine_loss", avg_cos, global_step)
                        writer.add_scalar("train/freq_loss", avg_freq, global_step)
                        writer.add_scalar("train/style_loss", avg_style, global_step)
                        writer.add_scalar("train/l2_loss", avg_l2, global_step)
                        writer.add_scalar("train/learning_rate", current_lr, global_step)
        
        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % args.save_every_n_epochs == 0:
            save_path = Path(args.output_dir) / f"{args.output_name}_epoch{epoch+1}.safetensors"
            save_transformer_weights(transformer, str(save_path), dtype=weight_dtype)
    
    # Final save
    if accelerator.is_main_process:
        final_path = Path(args.output_dir) / f"{args.output_name}_final.safetensors"
        save_transformer_weights(transformer, str(final_path), dtype=weight_dtype)
    
    logger.info("\n[DONE] Full Finetune Training complete!")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
