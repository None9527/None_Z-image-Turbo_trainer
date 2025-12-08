"""
硬件检测和自动优化配置模块
自动检测 GPU 类型并优化训练参数

支持检测:
- GPU 类型和显存
- xformers 可用性
- Flash Attention 支持
- SDPA 支持
"""
import torch
import psutil
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# xformers 检测
XFORMERS_AVAILABLE = False
XFORMERS_VERSION = None

try:
    import xformers
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    XFORMERS_VERSION = getattr(xformers, "__version__", "unknown")
except ImportError:
    pass
except Exception:
    pass

class HardwareDetector:
    """硬件检测和自动优化器"""
    
    def __init__(self):
        self.gpu_info = self.detect_gpu()
        self.cpu_info = self.detect_cpu()
        self.memory_info = self.detect_memory()
        self.xformers_info = self.detect_xformers()
        self.attention_info = self.detect_attention_backends()
        
    def detect_gpu(self) -> Dict[str, Any]:
        """检测 GPU 信息"""
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": "CPU",
            "compute_capability": None,
            "memory_total": 0,
            "memory_free": 0,
            "gpu_tier": "unknown"
        }
        
        if not torch.cuda.is_available():
            logger.info("[INFO] No CUDA GPU detected, will use CPU training")
            return gpu_info
            
        # 获取主 GPU 信息
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        gpu_info["compute_capability"] = torch.cuda.get_device_properties(0).major
        gpu_info["compute_capability"] = (gpu_info["compute_capability"], torch.cuda.get_device_properties(0).minor)
        
        # 计算可用内存
        gpu_info["memory_free"] = gpu_info["memory_total"] - torch.cuda.memory_allocated() / (1024**3)
        
        # GPU 分级
        gpu_info["gpu_tier"] = self._classify_gpu_tier(gpu_info["device_name"], gpu_info["memory_total"])
        
        logger.info(f"🖥️ 检测到 GPU: {gpu_info['device_name']}")
        logger.info(f"[VRAM] GPU Memory: {gpu_info['memory_total']:.1f}GB (Free: {gpu_info['memory_free']:.1f}GB)")
        logger.info(f"[TIER] GPU Tier: {gpu_info['gpu_tier']}")
        
        return gpu_info
    
    def detect_cpu(self) -> Dict[str, Any]:
        """检测 CPU 信息"""
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        return {
            "count": cpu_count,
            "frequency": cpu_freq.max if cpu_freq else None,
            "arch": "x86_64"
        }
    
    def detect_memory(self) -> Dict[str, Any]:
        """检测系统内存信息"""
        memory = psutil.virtual_memory()
        
        return {
            "total": memory.total / (1024**3),  # GB
            "available": memory.available / (1024**3),  # GB
            "percent": memory.percent
        }
    
    def detect_xformers(self) -> Dict[str, Any]:
        """检测 xformers 可用性和功能"""
        info = {
            "available": XFORMERS_AVAILABLE,
            "version": XFORMERS_VERSION,
            "memory_efficient_attention": False,
            "flash_attention": False,
            "cutlass": False,
        }
        
        if not XFORMERS_AVAILABLE:
            logger.info("[WARN] xformers not installed")
            return info
        
        try:
            info["memory_efficient_attention"] = hasattr(xops, "memory_efficient_attention")
            
            if hasattr(xops, "MemoryEfficientAttentionFlashAttentionOp"):
                info["flash_attention"] = True
            
            if hasattr(xops, "MemoryEfficientAttentionCutlassOp"):
                info["cutlass"] = True
            
            logger.info(f"[OK] xformers {XFORMERS_VERSION} available")
            if info["flash_attention"]:
                logger.info("   [OK] Flash Attention supported")
            if info["cutlass"]:
                logger.info("   [OK] CUTLASS supported")
                
        except Exception as e:
            logger.warning(f"xformers 功能检测失败: {e}")
        
        return info
    
    def detect_attention_backends(self) -> Dict[str, Any]:
        """检测所有可用的注意力后端"""
        backends = {
            "torch_sdpa": False,
            "xformers": XFORMERS_AVAILABLE,
            "flash_attention_2": False,
            "recommended": "torch",
        }
        
        # 检查 PyTorch SDPA
        try:
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                backends["torch_sdpa"] = True
        except Exception:
            pass
        
        # 检查 Flash Attention 2
        try:
            import flash_attn
            backends["flash_attention_2"] = True
        except ImportError:
            pass
        
        # 推荐后端
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            
            # SM80+ (A100, H100, RTX 30xx/40xx)
            if cc[0] >= 8:
                if XFORMERS_AVAILABLE and self.xformers_info.get("flash_attention"):
                    backends["recommended"] = "xformers"
                elif backends["flash_attention_2"]:
                    backends["recommended"] = "flash_attention_2"
                elif backends["torch_sdpa"]:
                    backends["recommended"] = "torch_sdpa"
            # SM70+ (V100, T4, RTX 20xx)
            elif cc[0] >= 7:
                if XFORMERS_AVAILABLE:
                    backends["recommended"] = "xformers"
                elif backends["torch_sdpa"]:
                    backends["recommended"] = "torch_sdpa"
            else:
                if backends["torch_sdpa"]:
                    backends["recommended"] = "torch_sdpa"
        
        logger.info(f"[RECOMMEND] Attention backend: {backends['recommended']}")
        return backends
    
    def _classify_gpu_tier(self, device_name: str, memory_total: float) -> str:
        """
        基于显存的严格分级 (用户要求: 32G/24G/16G 分级，<16G 不支持)
        
        Args:
            device_name: GPU 名称
            memory_total: 显存大小 (GB)
            
        Returns:
            str: 'tier_s', 'tier_a', 'tier_b', 'unsupported'
        """
        # 门槛下调 1G，因为系统报告的显存略小于标称值 (如 24G 报告为 23.5G)
        if memory_total >= 31:
            return "tier_s"        # 32GB级 (A100/H100/Pro 6000/A6000/5090) - 全开
        elif memory_total >= 23:
            return "tier_a"        # 24GB级 (3090/4090)
        elif memory_total >= 15:
            return "tier_b"        # 16GB级 (4080/4070TiS/A4000)
        else:
            return "unsupported"   # <15GB - 不支持
    
    def get_optimized_config(self, config=None):
        """
        基于硬件生成动态优化配置
        """
        if config is None:
            config = {}
            
        memory_gb = self.gpu_info['memory_total']
        gpu_tier = self.gpu_info['gpu_tier']
        
        if gpu_tier == 'unsupported':
            logger.warning(f"[WARN] Detected VRAM ({memory_gb:.1f}GB) below minimum (16GB). Extreme mode enabled.")
            # 极限压榨模式：尝试使用更多显存，虽然风险高
            return {
                'mixed_precision': 'fp16',
                'gradient_checkpointing': True,
                'memory_efficient_preprocessing': True,
                'max_memory_mb': int(memory_gb * 1024 * 1.0), # 100% 显存
                'spda_enabled': False,
                'block_swap_enabled': True,
                'block_swap_block_size': 1024, # 增大块大小
                'block_swap_max_cache_blocks': int(memory_gb * 80), # 增加缓存块
                'block_swap_swap_threshold': 0.99, # 99% 阈值 (极限)
                'dataloader_num_workers': 4, 
            }

        # 1. 确定混合精度类型
        cc_major, cc_minor = self.gpu_info.get("compute_capability", (0, 0))
        cc = float(f"{cc_major}.{cc_minor}")
        use_bf16 = cc >= 8.0
        
        # 基础配置
        optimized = {
            'mixed_precision': 'bf16' if use_bf16 else 'fp16',
            'gradient_checkpointing': True,
            'memory_efficient_preprocessing': True,
            'memory_monitoring_enabled': True,
            'comp_cache_compress': True,
        }
        
        # 动态计算 max_memory_mb (极限压榨: 100%)
        safe_memory_ratio = 1.0
        optimized['max_memory_mb'] = int(memory_gb * 1024 * safe_memory_ratio)
        
        # 基于 Tier 的配置
        if gpu_tier == 'tier_s':
            # Tier S (32GB+: A100/H100/5090): 全性能模式
            # 显存充裕，不需要任何压缩技术
            optimized.update({
                # ❌ 关闭所有压缩/交换功能
                'block_swap_enabled': False,
                'block_swap_block_size': 0,
                'block_swap_cpu_buffer_size': 0,
                'block_swap_swap_threshold': 0,
                
                # ❌ 关闭 SPDA - 单卡无意义
                'spda_enabled': False,
                
                # ✅ 使用最高效的注意力后端
                'sdpa_enabled': True,
                'sdpa_flash_attention': True,
                'attention_backend': 'sdpa',
                
                # max_grad_norm 由用户在 toml 中指定，不自动覆盖
                'dataloader_num_workers': 16,
                
                # xformers 作为备选
                'xformers_enabled': self.xformers_info.get('available', False),
            })
            
        elif gpu_tier == 'tier_a':
            # Tier A (24GB级: 3090/4090): 高性能模式
            # 24GB 跑 LoRA 是"富裕仗"，不需要 Block Swap 等极限压缩
            optimized.update({
                # ❌ 关闭 Block Swap - 24GB 显存足够，开启只会拖慢速度
                'block_swap_enabled': False,
                'block_swap_block_size': 0,
                'block_swap_cpu_buffer_size': 0,
                'block_swap_swap_threshold': 0,
                
                # ❌ 关闭 SPDA - 单卡无意义，只增加开销
                'spda_enabled': False,
                
                # ✅ 使用 PyTorch 原生 SDPA (自动调用 Flash Attention)
                'sdpa_enabled': True,
                'sdpa_flash_attention': True,
                'attention_backend': 'sdpa',  # 优先使用原生 SDPA
                
                # max_grad_norm 由用户在 toml 中指定，不自动覆盖
                'dataloader_num_workers': 8,
                
                # xformers 作为备选
                'xformers_enabled': self.xformers_info.get('available', False),
            })
            
        elif gpu_tier == 'tier_b':
            # Tier B (16GB级: 4080/4070TiS/P100): 精简模式
            # 16GB 跑 LoRA 足够，但需要精简配置，避免不必要的内存开销
            optimized.update({
                # ❌ 关闭 Block Swap - 16GB LoRA 训练不需要，开启反而可能卡死
                # Block Swap 的监控线程和内存交换在边界条件下可能导致死锁
                'block_swap_enabled': False,
                'block_swap_block_size': 0,
                'block_swap_cpu_buffer_size': 0,
                'block_swap_swap_threshold': 0,
                
                # ❌ 关闭 SPDA - 单卡无意义
                'spda_enabled': False,
                
                # ✅ 使用原生 SDPA (内存效率最高)
                'sdpa_enabled': True,
                'sdpa_flash_attention': True,
                'attention_backend': 'sdpa',
                
                # ✅ 强制启用梯度检查点 - 16GB 必须开启
                'gradient_checkpointing': True,
                
                # ✅ 减少数据加载器线程数 - 节省 CPU 内存，避免与 GPU 竞争
                'dataloader_num_workers': 2,
                
                # ✅ 增加梯度累积 - 减少单次显存峰值
                'gradient_accumulation_steps': 4,
                
                # xformers 作为备选
                'xformers_enabled': self.xformers_info.get('available', False),
            })
            
        # SPDA World Size
        if optimized.get('spda_enabled', False):
            optimized['spda_world_size'] = torch.cuda.device_count()
        else:
            optimized['spda_world_size'] = 1
            
        # 输出优化的配置
        # 根据 tier 显示不同的模式描述
        mode_desc = {
            'tier_s': 'Full Performance (no compression)',
            'tier_a': 'High Performance (LoRA optimized)',
            'tier_b': 'Balanced (light block swap)',
            'unsupported': 'Extreme (risky)'
        }.get(gpu_tier, 'Unknown')
        logger.info(f"[CONFIG] Hardware tier: {gpu_tier.upper()} (VRAM: {memory_gb:.1f}GB) - {mode_desc}")
        for key, value in optimized.items():
            logger.info(f"   {key}: {value}")
        
        return optimized
    
    def print_detection_summary(self):
        """打印硬件检测摘要"""
        # Use logger instead of print to avoid Windows GBK encoding issues
        logger.info("")
        logger.info("=" * 60)
        logger.info("[Hardware Detection Report]")
        logger.info("=" * 60)
        logger.info(f"GPU: {self.gpu_info['device_name']}")
        logger.info(f"VRAM: {self.gpu_info['memory_total']:.1f}GB")
        logger.info(f"GPU Tier: {self.gpu_info['gpu_tier']}")
        logger.info(f"CPU: {self.cpu_info['count']} cores")
        logger.info(f"System Memory: {self.memory_info['total']:.1f}GB")
        logger.info(f"Available Memory: {self.memory_info['available']:.1f}GB")
        logger.info("-" * 60)
        logger.info("Attention Backend:")
        xf_status = f"[OK] {self.xformers_info.get('version', '')}" if self.xformers_info.get('available') else "[NO]"
        logger.info(f"  xformers: {xf_status}")
        logger.info(f"  PyTorch SDPA: {'[OK]' if self.attention_info.get('torch_sdpa') else '[NO]'}")
        logger.info(f"  Flash Attention 2: {'[OK]' if self.attention_info.get('flash_attention_2') else '[NO]'}")
        logger.info(f"  Recommended: {self.attention_info.get('recommended', 'torch')}")
        logger.info("=" * 60)