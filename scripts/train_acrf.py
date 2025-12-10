"""
[START] AC-RF Training Script for Z-Image-Turbo

独立的 Anchor-Constrained Rectified Flow 训练脚本
用于 Z-Image-Turbo 模型的 LoRA 微调实验

关键特性：
- 保持 Turbo 模型的直线加速结构
- 只在关键锚点时间步训练
- 直接回归速度向量而非预测噪声
"""

import os
import sys
import math
import signal
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
import torch.nn.functional as F
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
from zimage_trainer.utils.snr_utils import compute_snr_weights, print_anchor_snr_weights
from zimage_trainer.losses import FrequencyAwareLoss, LatentStyleStructureLoss

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局中断标志
_interrupted = False

def signal_handler(signum, frame):
    """信号处理器"""
    global _interrupted
    _interrupted = True
    logger.info("\n[STOP] 收到停止信号，将在当前步骤完成后保存并退出...")

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="AC-RF 训练脚本")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, help="超参数配置文件路径 (.toml)")
    
    # 模型路径
    parser.add_argument("--dit", type=str, help="Transformer 模型路径")
    parser.add_argument("--dataset_config", type=str, help="数据集配置文件")
    parser.add_argument("--output_dir", type=str, default="output/acrf", help="输出目录")
    
    # AC-RF 参数
    parser.add_argument("--turbo_steps", type=int, default=10, help="Turbo 步数（锚点数量）")
    parser.add_argument("--shift", type=float, default=3.0, help="时间步 shift 参数")
    parser.add_argument("--jitter_scale", type=float, default=0.02, help="锚点抖动幅度")
    
    # LoRA 参数
    parser.add_argument("--network_dim", type=int, default=8, help="LoRA rank")
    parser.add_argument("--network_alpha", type=float, default=4.0, help="LoRA alpha")
    
    # 训练参数
    parser.add_argument("--optimizer_type", type=str, default="AdamW", choices=["AdamW", "AdamW8bit", "Adafactor"], help="优化器类型")
    # Adafactor 特有参数
    parser.add_argument("--adafactor_scale", action="store_true", help="Adafactor scale_parameter")
    parser.add_argument("--adafactor_relative", action="store_true", help="Adafactor relative_step")
    parser.add_argument("--adafactor_warmup", action="store_true", help="Adafactor warmup_init")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    
    # LR Scheduler 参数
    parser.add_argument("--lr_scheduler", type=str, default="constant", 
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="学习率调度器"
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Warmup 步数")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Cosine 调度器的循环次数")
    
    # Min-SNR 加权参数（统一应用于所有 loss 模式）
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="Min-SNR gamma (0=禁用, 推荐5.0)")
    parser.add_argument("--snr_floor", type=float, default=0.1, help="Min-SNR 保底权重 (10步模型关键参数，推荐0.1)")
    
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
    # 推荐 0.03-0.05，配合 target = x0 - x_t_perturbed
    parser.add_argument("--latent_jitter_scale", type=float, default=0.0, help="Latent 空间抠动幅度 (0=禁用, 推荐 0.03-0.05)")
    
    # 频域感知 Loss 子参数
    parser.add_argument("--alpha_hf", type=float, default=1.0, help="高频增强权重")
    parser.add_argument("--beta_lf", type=float, default=0.2, help="低频锁定权重")
    parser.add_argument("--lf_magnitude_weight", type=float, default=0.0, help="低频幅度约束")
    parser.add_argument("--downsample_factor", type=int, default=4, help="低频提取降采样因子")
    
    # 风格结构 Loss 子参数
    parser.add_argument("--lambda_struct", type=float, default=1.0, help="结构锁权重 (SSIM)")
    parser.add_argument("--lambda_light", type=float, default=0.5, help="光影学习权重 (L通道统计)")
    parser.add_argument("--lambda_color", type=float, default=0.3, help="色调迁移权重 (ab通道统计)")
    parser.add_argument("--lambda_tex", type=float, default=0.5, help="质感增强权重 (高频L1)")
    
    # 训练控制 (Epoch 模式)
    parser.add_argument("--num_train_epochs", type=int, default=10, help="训练 Epoch 数")
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="保存间隔 (Epoch)")
    parser.add_argument("--output_name", type=str, default="zimage-lora", help="LoRA 输出文件名")
    
    # 兼容性保留 (会被自动覆盖)
    parser.add_argument("--max_train_steps", type=int, default=None, help="最大训练步数 (自动计算)")
    parser.add_argument("--save_every_n_steps", type=int, default=None, help="保存间隔 (步数)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 高级功能
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--blocks_to_swap", type=int, default=0, 
        help="将多少个 transformer blocks 交换到 CPU，节省显存。"
             "16G显存建议设为 4-8，24G显存可不设置")
    
    # 自动优化功能
    parser.add_argument("--auto_optimize", action="store_true", default=True, help="启用自动硬件优化")
    
    # 数据加载参数
    parser.add_argument("--enable_bucket", action="store_true", default=True, help="启用分桶 (按分辨率分组)")
    parser.add_argument("--disable_bucket", action="store_true", help="禁用分桶 (所有图片必须相同尺寸)")
    
    # SDPA (Scaled Dot-Product Attention) 参数
    parser.add_argument("--attention_backend", type=str, default="sdpa", 
        choices=["sdpa", "flash", "_flash_3"], help="注意力后端选择")
    parser.add_argument("--enable_flash_attention", action="store_true", help="启用Flash Attention")
    parser.add_argument("--sdpa_optimize_level", type=str, default="auto",
        choices=["fast", "memory_efficient", "auto"], help="SDPA优化级别")
    parser.add_argument("--use_memory_efficient_attention", action="store_true", default=True, help="使用内存高效注意力")
    parser.add_argument("--attention_dropout", type=float, default=0.0, help="注意力dropout率")
    parser.add_argument("--force_deterministic", action="store_true", help="强制确定性计算")
    parser.add_argument("--sdpa_min_seq_length", type=int, default=512, help="SDPA最小序列长度阈值")
    parser.add_argument("--sdpa_batch_size_threshold", type=int, default=4, help="SDPA批量大小阈值")
    
    # Block Swapping (块交换技术) 参数
    parser.add_argument("--block_swap_enabled", action="store_true", help="启用块交换技术")
    parser.add_argument("--block_swap_block_size", type=int, default=256, help="块交换内存块大小")
    parser.add_argument("--block_swap_cpu_buffer_size", type=int, default=1024, help="块交换CPU缓冲区大小 (MB)")
    parser.add_argument("--block_swap_swap_threshold", type=float, default=0.7, help="块交换阈值 (0.1-0.9)")
    parser.add_argument("--block_swap_swap_strategy", type=str, default="lru", choices=["fifo", "lru", "priority"], help="块交换策略")
    parser.add_argument("--block_swap_compression_enabled", action="store_true", help="启用块交换压缩")
    parser.add_argument("--block_swap_prefetch_enabled", action="store_true", help="启用块交换预取")
    parser.add_argument("--activation_checkpoint_block_size", type=int, default=64, help="激活检查点块大小")
    parser.add_argument("--memory_monitoring_enabled", action="store_true", help="启用内存监控")
    parser.add_argument("--memory_swap_frequency", type=int, default=5, help="内存交换频率")
    parser.add_argument("--memory_pool_strategy", type=str, default="conservative",
        choices=["none", "conservative", "aggressive"], help="内存池策略")
    
    # 文本序列长度参数
    parser.add_argument("--max_sequence_length", type=int, default=512, help="文本编码器最大序列长度 (需与缓存时一致)")
    
    args = parser.parse_args()
    
    # 如果指定了配置文件，读取并覆盖默认值
    if args.config:
        import tomli
        with open(args.config, "rb") as f:
            config = tomli.load(f)
            
        # 扁平化 config 字典以便映射
        flat_config = {}
        for section in config.values():
            flat_config.update(section)
            
        # 更新 args (仅当命令行未指定时使用 config 值，或者直接覆盖？通常命令行优先级更高)
        # 这里我们实现：Config 覆盖默认值，命令行覆盖 Config
        
        # 1. 设置 Config 中的值
        for key, value in flat_config.items():
            # 只有当 args 中存在该属性且命令行未显式指定（这里比较难判断是否显式指定，
            # 简化起见，我们假设如果 config 有值就用 config 的，除非 args 是 None）
            # 更稳健的做法是：argparse default 设为 None，然后手动处理 defaults
            if hasattr(args, key):
                setattr(args, key, value)
    
    # 再次解析命令行参数以确保命令行参数优先级最高 (需要稍微重构，或者简单地只用 config)
    # 简单实现：如果提供了 config，就用 config 的值覆盖 args 的默认值
    # 但这样命令行参数就无效了。
    
    # 更好的实现：
    # 1. Parse args 得到命令行参数
    # 2. Load config
    # 3. 如果命令行参数是默认值，且 config 中有值，则使用 config 的值
    # 但 argparse 不容易区分"默认值"和"用户输入的值"。
    
    # 这种情况下，通常建议：如果用了 --config，就主要依赖 config。
    # 或者，我们手动检查 sys.argv
    
    # 让我们采用最简单的策略：Config 文件作为"新的默认值"
    if args.config:
        # 重新解析，这次将 config 中的值作为 default
        import tomli
        with open(args.config, "rb") as f:
            config = tomli.load(f)
        
        defaults = {}
        for section in config.values():
            defaults.update(section)
            
        parser.set_defaults(**defaults)
        args = parser.parse_args() # 再次解析，这样命令行参数会覆盖 config (作为 defaults)
        
    # 验证必要参数
    if not args.dit:
        parser.error("--dit is required (or set in config)")
    
    # dataset_config 可选：如果没有指定，使用主配置文件
    if not args.dataset_config and args.config:
        args.dataset_config = args.config  # 使用主配置文件中的 [dataset] 部分
        
    return args


def main():
    args = parse_args()
    
    # 硬件检测和自动优化
    logger.info("[DETECT] 正在进行硬件检测...")
    hardware_detector = HardwareDetector()
    hardware_detector.print_detection_summary()
    
    # 如果启用了自动优化，则应用优化配置
    if args.auto_optimize:
            logger.info("[TARGET] 启用自动硬件优化...")
            
            # 如果配置是简化配置，应用自动优化
            if args.config:
                try:
                    # 尝试导入tomli（TOML解析库）
                    try:
                        import tomli
                        with open(args.config, "rb") as f:
                            config = tomli.load(f)
                    except ImportError:
                        # 如果没有tomli，使用tomllib（Python 3.11+内置）
                        import tomllib
                        with open(args.config, "rb") as f:
                            config = tomllib.load(f)
                    
                    # 如果检测到是简化配置，应用自动优化
                    if 'optimization' in config and config['optimization'].get('auto_optimize', False):
                        logger.info("[CONFIG] 检测到简化配置，开始自动优化...")
                        
                        # 获取手动覆盖设置（如果有）
                        manual_gpu_tier = config['optimization'].get('gpu_tier')
                        manual_gpu_memory = config['optimization'].get('gpu_memory_gb')
                        
                        # 应用手动覆盖（如果有）
                        if manual_gpu_tier:
                            hardware_detector.gpu_info['gpu_tier'] = manual_gpu_tier
                            logger.info(f"[SETUP] 手动设置GPU级别: {manual_gpu_tier}")
                        
                        if manual_gpu_memory:
                            hardware_detector.gpu_info['memory_total'] = manual_gpu_memory
                            logger.info(f"[SETUP] 手动设置GPU显存: {manual_gpu_memory}GB")
                        
                        # 保存用户在 [advanced] 部分设置的值
                        user_advanced = config.get('advanced', {})
                        
                        # 应用优化配置
                        optimized_config = hardware_detector.get_optimized_config({})
                        
                        # 更新args对象（但保留用户显式设置的值）
                        for key, value in optimized_config.items():
                            if hasattr(args, key):
                                # 如果用户在 [advanced] 中设置了该值，则使用用户的值
                                if key in user_advanced:
                                    logger.info(f"   {key}: {user_advanced[key]} (用户设置)")
                                    setattr(args, key, user_advanced[key])
                                else:
                                    setattr(args, key, value)
                        
                        logger.info("[OK] 自动硬件优化完成")
                
                except Exception as e:
                    logger.warning(f"[WARN] 配置文件解析失败，使用默认优化: {e}")
                    # 使用默认优化配置
                    optimized_config = hardware_detector.get_optimized_config({})
                    for key, value in optimized_config.items():
                        if hasattr(args, key):
                            setattr(args, key, value)
                    logger.info("[OK] 使用默认硬件优化配置")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # 获取分布式训练信息
    world_size = getattr(accelerator, 'num_processes', None)
    rank = getattr(accelerator, 'rank', None)
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    
    logger.info("="*60)
    logger.info("[START] 启动 AC-RF 训练")
    logger.info("="*60)
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"Turbo 步数: {args.turbo_steps}")
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
    # =========================================================================
    # Refactored Model Loaded - No Monkey Patch Needed
    # =========================================================================

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        transformer.train()
        # NOTE: Freeze is done AFTER LoRA is applied (see below)
        logger.info("  [MEM] 梯度检查点已启用 (Gradient Checkpointing Enabled)")
    else:
         # Legacy unfreeze
         pass

    # =========================================================================
    
    # 1.1 配置SDPA (Scaled Dot-Product Attention)
    logger.info("\n[INIT] 配置 SDPA 注意力后端...")
    logger.info(f"  注意力后端: {args.attention_backend}")
    logger.info(f"  优化级别: {args.sdpa_optimize_level}")
    logger.info(f"  内存高效注意力: {args.use_memory_efficient_attention}")
    logger.info(f"  注意力dropout: {args.attention_dropout}")
    
    # 配置注意力后端
    if hasattr(transformer, 'set_attention_backend'):
        try:
            if args.enable_flash_attention:
                # 如果启用了flash attention，尝试切换后端
                if args.attention_backend == "sdpa":
                    # 检查硬件支持
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0).upper()
                        if "A100" in gpu_name or "H100" in gpu_name:
                            transformer.set_attention_backend("_flash_3")
                            logger.info("  [OK] 硬件检测：已启用 Flash Attention 3")
                        elif "RTX" in gpu_name or "4090" in gpu_name or "4080" in gpu_name:
                            transformer.set_attention_backend("flash")
                            logger.info("  [OK] 硬件检测：已启用 Flash Attention 2")
                        else:
                            logger.info("  [WARN] 硬件不支持Flash Attention，使用默认SDPA")
                    else:
                        logger.info("  [WARN] 未检测到CUDA，使用默认SDPA")
                else:
                    transformer.set_attention_backend(args.attention_backend)
                    logger.info(f"  [OK] 已设置注意力后端为: {args.attention_backend}")
        except Exception as e:
            logger.warning(f"  [WARN] 设置注意力后端失败: {e}")
            logger.info("  [FALLBACK] 继续使用默认SDPA实现")
    
    # 配置SDPA环境变量
    if args.force_deterministic:
        os.environ['TORCH_DETERMINISTIC'] = '1'
        logger.info("  [LOCK] 已启用确定性计算")
    
    if args.sdpa_optimize_level == "memory_efficient":
        os.environ['TORCH_CUDA_MEMORY_POOL'] = 'memory_efficient'
        logger.info("  [MEM] 已启用内存优化模式")
    
    # 初始化内存优化器
    logger.info(f"\n[MEM] 初始化内存优化器...")
    if args.blocks_to_swap > 0:
        logger.info(f"  Blocks to swap: {args.blocks_to_swap}")
    memory_config = {
        'block_swap_enabled': args.block_swap_enabled or args.blocks_to_swap > 0,
        'blocks_to_swap': args.blocks_to_swap,
        'memory_block_size': args.block_swap_block_size,
        'cpu_swap_buffer_size': args.block_swap_cpu_buffer_size,
        'swap_threshold': args.block_swap_swap_threshold,
        'swap_frequency': args.memory_swap_frequency,
        'smart_prefetch': args.block_swap_prefetch_enabled,
        'swap_strategy': args.block_swap_swap_strategy,
        'compressed_swap': args.block_swap_compression_enabled,
        'checkpoint_optimization': 'basic' if args.gradient_checkpointing else 'none',
    }
    memory_optimizer = MemoryOptimizer(memory_config)
    memory_optimizer.start()
    logger.info(f"  [OK] 内存优化器初始化完成")
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("  [FALLBACK] 已启用梯度检查点")
        
    # 应用内存优化到transformer
    if hasattr(transformer, 'apply_memory_optimization'):
        transformer.apply_memory_optimization(memory_optimizer)
        logger.info("  [INIT] 已应用内存优化策略")
        
    # 2. 创建 LoRA 网络
    logger.info(f"\n[SETUP] 创建 LoRA 网络 (rank={args.network_dim})...")
    network = LoRANetwork(
        unet=transformer,
        lora_dim=args.network_dim,
        alpha=args.network_alpha,
        multiplier=1.0,
    )
    network.apply_to(transformer)
    
    # 关键: 先应用 LoRA，再冻结底模 (LoRA 参数不会被冻结)
    if args.gradient_checkpointing:
        transformer.requires_grad_(False)  # 冻结底模
        logger.info("  [FREEZE] 底模已冻结 (Base model frozen, LoRA trainable)")
    
    # 只获取 LoRA 层的参数，不包括原始模型
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
    
    # 3.5. 打印 Min-SNR 配置和锚点权重分布
    snr_gamma = getattr(args, 'snr_gamma', 5.0)
    snr_floor = getattr(args, 'snr_floor', 0.1)
    logger.info(f"\n[SNR] Min-SNR 配置: gamma={snr_gamma}, floor={snr_floor}")
    if snr_gamma > 0:
        print_anchor_snr_weights(
            turbo_steps=args.turbo_steps,
            shift=args.shift,
            snr_gamma=snr_gamma,
            snr_floor=snr_floor,
        )
    
    # 3.6. 创建高级损失函数 (基于开关判断)
    logger.info(f"\n[LOSS] 自由组合损失模式")
    logger.info(f"  [基础] lambda_l1={args.lambda_l1}, lambda_cosine={args.lambda_cosine}")
    
    frequency_loss_fn = None
    style_loss_fn = None
    
    # 频域感知损失 (开关控制)
    enable_freq = getattr(args, 'enable_freq', False)
    if enable_freq:
        logger.info(f"  [频域感知] ✅ 启用 lambda={args.lambda_freq}, alpha_hf={args.alpha_hf}, beta_lf={args.beta_lf}")
        frequency_loss_fn = FrequencyAwareLoss(
            alpha_hf=args.alpha_hf,
            beta_lf=args.beta_lf,
            base_weight=1.0,
            downsample_factor=args.downsample_factor,
            lf_magnitude_weight=args.lf_magnitude_weight,
        )
    
    # 风格结构损失 (开关控制)
    enable_style = getattr(args, 'enable_style', False)
    if enable_style:
        logger.info(f"  [风格结构] ✅ 启用 lambda={args.lambda_style}, struct={args.lambda_struct}")
        style_loss_fn = LatentStyleStructureLoss(
            lambda_struct=args.lambda_struct,
            lambda_light=args.lambda_light,
            lambda_color=args.lambda_color,
            lambda_tex=args.lambda_tex,
            lambda_base=1.0,
        )
    
    # 4. 创建数据加载器
    logger.info("\n📊 加载数据集...")
    dataloader = create_dataloader(args)
    logger.info(f"数据集大小: {len(dataloader)} batches")
    
    # 5. 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num Batches per Epoch = {len(dataloader)}")
    logger.info(f"  Gradient Accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total Optimization Steps = {args.max_train_steps}")
    
    # 打印总步数供前端解析（关键！tqdm 的 \r 输出无法被 readline 捕获）
    print(f"[TRAINING_INFO] total_steps={args.max_train_steps} total_epochs={args.num_train_epochs}", flush=True)

    # 6. 创建优化器
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
            raise ImportError("请先安装 bitsandbytes 以使用 AdamW8bit 优化器")
    elif args.optimizer_type == "Adafactor":
        from transformers.optimization import Adafactor
        logger.info(f"  Adafactor 配置: scale={args.adafactor_scale}, relative={args.adafactor_relative}, warmup={args.adafactor_warmup}")
        optimizer = Adafactor(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            scale_parameter=args.adafactor_scale,
            relative_step=args.adafactor_relative,
            warmup_init=args.adafactor_warmup
        )
    else:
        raise ValueError(f"不支持的优化器类型: {args.optimizer_type}")
        
    # 7. 创建学习率调度器
    # 注意：lr_scheduler.step() 只在优化器步骤时调用（sync_gradients 时）
    # 所以 num_warmup_steps 和 num_training_steps 应该是优化器步数，不需要乘以梯度累积
    from diffusers.optimization import get_scheduler
    logger.info(f"[SCHED] 初始化调度器: {args.lr_scheduler} (warmup={args.lr_warmup_steps}, total_steps={args.max_train_steps}, cycles={args.lr_num_cycles})")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
    )
    
    # 7. Accelerator prepare
    transformer, network, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, network, optimizer, dataloader, lr_scheduler
    )
    
    # 8. 训练循环
    logger.info("\n" + "="*60)
    logger.info("[TARGET] 开始训练")
    logger.info("="*60)
    
    global_step = 0
    # 禁用 tqdm 显示，改用 [STEP] 格式输出（避免日志重复）
    progress_bar = tqdm(total=args.max_train_steps, desc="Training", disable=True)
    
    # EMA 平滑 loss（用于显示趋势，不影响训练）
    ema_loss = None
    ema_decay = 0.99  # 平滑系数
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        transformer.train()
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(network):
                # 获取数据
                latents = batch['latents'].to(accelerator.device, dtype=weight_dtype)
                vl_embed = batch['vl_embed']  # List of tensors
                
                # 确保 vl_embed 中的所有张量都在正确的设备上
                if isinstance(vl_embed, list):
                    vl_embed = [tensor.to(accelerator.device, dtype=weight_dtype) for tensor in vl_embed]
                else:
                    vl_embed = vl_embed.to(accelerator.device, dtype=weight_dtype)
                
                # 生成噪声
                noise = torch.randn_like(latents)
                
                # AC-RF 采样 (时间步 jitter)
                noisy_latents, timesteps, target_velocity = acrf_trainer.sample_batch(
                    latents, noise, jitter_scale=args.jitter_scale
                )
                
                # === Latent Jitter: 空间抠动 (垂直于流线，改变构图的关键) ===
                latent_jitter_scale = getattr(args, 'latent_jitter_scale', 0.0)
                if latent_jitter_scale > 0:
                    # 在 x_t 上添加空间抖动，把状态“推离”完美流线
                    latent_jitter = torch.randn_like(noisy_latents) * latent_jitter_scale
                    noisy_latents_perturbed = noisy_latents + latent_jitter
                    
                    # 关键: 重新计算 target，指向真实 x_0 (Ground Truth)
                    # v_target = x_0 - x_t_perturbed (不是 Teacher 输出!)
                    target_velocity = noise - latents  # v = epsilon - x0 (RF 公式)
                    # 但输入是扰动后的 x_t
                    noisy_latents = noisy_latents_perturbed
                
                # 准备模型输入
                # Z-Image expects List[Tensor(C, 1, H, W)]
                model_input = noisy_latents.unsqueeze(2)  # (B, C, 1, H, W)
                
                # 关键: 梯度检查点需要输入有梯度 (与 LongCat 相同模式)
                if args.gradient_checkpointing:
                    model_input.requires_grad_(True)
                    
                model_input_list = list(model_input.unbind(dim=0))
                
                # Timestep normalization (Z-Image uses (1000-t)/1000)
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
                model_pred = model_pred.squeeze(2)  # (B, C, H, W)
                
                # Z-Image 输出是负的
                model_pred = -model_pred
                
                # 根据损失模式计算损失
                loss_components = {}
                
                # === Charbonnier Loss (Robust L1, 基础损失) ===
                diff = model_pred - target_velocity
                loss_l1 = torch.sqrt(diff**2 + 1e-6).mean()
                loss_components['l1'] = loss_l1.item()
                
                # === Cosine Loss (方向一致性) ===
                pred_flat = model_pred.view(model_pred.shape[0], -1)
                target_flat = target_velocity.view(target_velocity.shape[0], -1)
                cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
                loss_cosine = 1.0 - cos_sim
                loss_components['cosine'] = loss_cosine.item()
                
                # 计算 Min-SNR 权重（统一应用于所有 loss 模式）
                if snr_gamma > 0:
                    snr_weights = compute_snr_weights(
                        timesteps=timesteps,
                        num_train_timesteps=1000,
                        snr_floor=snr_floor,
                        prediction_type="v_prediction",
                    ).to(model_pred.device, dtype=weight_dtype)
                else:
                    snr_weights = None
                
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
                    batch_size = latents.shape[0]
                    
                    # 自由流: 全时间步随机采样
                    free_sigmas = torch.rand(batch_size, device=latents.device, dtype=latents.dtype)
                    shift = args.shift
                    free_sigmas = (free_sigmas * shift) / (1 + (shift - 1) * free_sigmas)
                    free_sigmas = free_sigmas.clamp(0.001, 0.999)
                    
                    # 构造自由流加噪 latents
                    sigma_broadcast = free_sigmas.view(batch_size, 1, 1, 1)
                    free_noisy = sigma_broadcast * noise + (1 - sigma_broadcast) * latents
                    free_target = noise - latents  # v-prediction
                    
                    # 自由流前向传播 (参与梯度!)
                    free_input = free_noisy.unsqueeze(2)
                    free_input_list = list(free_input.unbind(dim=0))
                    free_t_norm = (1000 - free_sigmas * 1000) / 1000.0
                    free_t_norm = free_t_norm.to(dtype=weight_dtype)
                    
                    free_pred_list = transformer(
                        x=free_input_list,
                        t=free_t_norm,
                        cap_feats=vl_embed,
                    )[0]
                    free_pred = torch.stack(free_pred_list, dim=0).squeeze(2)
                    free_pred = -free_pred  # Z-Image 负号
                    
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
                        batch_size = latents.shape[0]
                        mse_sigmas = torch.rand(batch_size, device=latents.device, dtype=latents.dtype)
                        shift = args.shift
                        mse_sigmas = (mse_sigmas * shift) / (1 + (shift - 1) * mse_sigmas)
                        mse_sigmas = mse_sigmas.clamp(0.001, 0.999)
                        
                        sigma_broadcast = mse_sigmas.view(batch_size, 1, 1, 1)
                        mse_noisy = sigma_broadcast * noise + (1 - sigma_broadcast) * latents
                        mse_target = noise - latents
                        
                        mse_input = mse_noisy.unsqueeze(2)
                        mse_input_list = list(mse_input.unbind(dim=0))
                        mse_t_norm = (1000 - mse_sigmas * 1000) / 1000.0
                        mse_t_norm = mse_t_norm.to(dtype=weight_dtype)
                        
                        with torch.no_grad():
                            mse_pred_list = transformer(
                                x=mse_input_list,
                                t=mse_t_norm,
                                cap_feats=vl_embed,
                            )[0]
                        mse_pred = torch.stack(mse_pred_list, dim=0).squeeze(2)
                        mse_pred = -mse_pred
                    
                    loss_mse = F.mse_loss(mse_pred, mse_target)
                    loss = loss + lambda_mse * loss_mse
                    loss_components['mse'] = loss_mse.item()
                
                # 应用 SNR 加权
                if snr_weights is not None:
                    loss = loss * snr_weights.mean()
                
                # 强制转换为 Float32 以兼容 Accelerate 的 backward (BF16 混合精度修复)
                loss = loss.float()
                
                # 反向传播
                accelerator.backward(loss)
            
            # 只在梯度累积完成后执行优化步骤
            if accelerator.sync_gradients:
                # 梯度裁剪
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                # 优化器步进
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Cleanup transformer gradients (since we unfroze it but don't optimize it)
                transformer.zero_grad()
                
                # 更新进度
                progress_bar.update(1)
                global_step += 1
                
                # 更新 EMA loss（平滑显示，减少跳动的视觉干扰）
                current_loss = loss.item()
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * current_loss
                
                # 获取当前学习率
                current_lr = lr_scheduler.get_last_lr()[0]
                
                # 打印进度供前端解析
                l1 = loss_components.get('l1', 0)
                cosine = loss_components.get('cosine', 0)
                freq = loss_components.get('freq', 0)
                style = loss_components.get('style', 0)
                free = loss_components.get('loss_free', 0)
                print(f"[STEP] {global_step}/{args.max_train_steps} epoch={epoch+1}/{args.num_train_epochs} loss={current_loss:.4f} ema={ema_loss:.4f} l1={l1:.4f} cos={cosine:.4f} freq={freq:.4f} style={style:.4f} free={free:.4f} lr={current_lr:.2e}", flush=True)
                
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
            
            # 显式清理显存 (防止 16G 显卡显存泄露)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    # 保存最终模型
    final_path = Path(args.output_dir) / f"{args.output_name}_final.safetensors"
    network.save_weights(final_path, dtype=weight_dtype)
    
    # 停止内存优化器并清理显存
    memory_optimizer.stop()
    
    # 清理 GPU 缓存
    del network, transformer, optimizer, lr_scheduler, dataloader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    logger.info("\n" + "="*60)
    logger.info(f"[OK] 训练完成！")
    logger.info(f"最终模型: {final_path}")
    logger.info("="*60)
    
    # 5秒后强制退出进程，确保显存释放
    logger.info("\n[EXIT] 5秒后退出进程...")
    time.sleep(5)
    os._exit(0)


if __name__ == "__main__":
    main()
