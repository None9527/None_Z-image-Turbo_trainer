"""
AC-RF (Anchor-Constrained Rectified Flow) Trainer for Z-Image-Turbo

基于 Rectified Flow 的 Turbo 模型微调方案。
关键特性：
1. 保持直线结构 - 使用线性插值而非 DDPM 弯曲路径
2. 离散锚点采样 - 只在 Turbo 模型的有效时间步训练
3. 速度回归 - 直接预测 velocity 而非 noise

时间步定义：
- Z-Image 使用 sigma ∈ [0, 1]
- sigma=0: 图像端 (x_0)
- sigma=1: 噪声端 (x_1)
- 线性插值: x_t = sigma * noise + (1 - sigma) * image
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ACRFTrainer:
    """
    Anchor-Constrained Rectified Flow Trainer
    
    在保留 Turbo 模型直线加速特性的同时，学习新的目标分布。
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        turbo_steps: int = 10,
        shift: float = 3.0,  # Z-Image 官方值
    ):
        """
        Args:
            num_train_timesteps: 训练时间步总数 (默认 1000)
            turbo_steps: Turbo 模型步数/锚点数量 (默认 10)
            shift: 时间步 shift 参数 (Z-Image 官方值 3.0)
        """
        self.num_train_timesteps = num_train_timesteps
        self.turbo_steps = turbo_steps
        self.shift = shift
        
        # 计算锚点 (离散采样点)
        # 对于 4-step Turbo: 我们需要在关键时间点训练
        # Z-Image scheduler 从 sigma=1.0 逐步降到 sigma=0.0
        self._compute_anchors()
        
        logger.info(f"[START] AC-RF Trainer 初始化完成")
        logger.info(f"   锚点 sigmas: {self.anchor_sigmas.tolist()}")
        logger.info(f"   对应 timesteps: {self.anchor_timesteps.tolist()}")
    
    def _compute_anchors(self):
        """
        从真实 Scheduler 获取离散锚点
        
        使用 FlowMatchEulerDiscreteScheduler 确保与推理时完全一致
        """
        from diffusers import FlowMatchEulerDiscreteScheduler
        
        # 创建与 Z-Image 完全一致的 scheduler
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=self.shift,
        )
        
        # 调用 set_timesteps 获取真实锚点
        scheduler.set_timesteps(
            num_inference_steps=self.turbo_steps,
            device="cpu"
        )
        
        # 提取 timesteps 和 sigmas
        timesteps = scheduler.timesteps.cpu()
        sigmas = scheduler.sigmas.cpu()
        
        # 保存锚点（排除终点 0）
        self.anchor_timesteps = timesteps
        self.anchor_sigmas = sigmas[:-1]  # 排除最后的 0
        
        logger.info(f"[OK] 从真实 Scheduler 加载锚点")
        logger.info(f"   Timesteps: {self.anchor_timesteps.tolist()}")
        logger.info(f"   Sigmas: {self.anchor_sigmas.tolist()}")

    def sample_batch(
        self, 
        latents: torch.Tensor, 
        noise: torch.Tensor,
        jitter_scale: float = 0.02,
        stratified: bool = True,  # 分层采样，减少 loss 跳动
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        为当前 batch 采样训练数据
        
        Args:
            latents: 原图 latents (x_0)
            noise: 噪声 (x_1)
            jitter_scale: 锚点抖动幅度
            stratified: 是否使用分层采样（推荐开启，减少 loss 波动）
            
        Returns:
            noisy_latents: 加噪后的 latents (x_t)
            timesteps: 对应的时间步
            target_velocity: 预测目标 (v)
        """
        batch_size = latents.shape[0]
        device = latents.device
        
        # 1. 选择锚点索引
        if stratified and batch_size >= self.turbo_steps:
            # 分层采样：确保每个锚点至少被采样一次
            base_indices = torch.arange(self.turbo_steps, device=device)
            repeats = batch_size // self.turbo_steps
            remainder = batch_size % self.turbo_steps
            indices = torch.cat([
                base_indices.repeat(repeats),
                base_indices[torch.randperm(self.turbo_steps, device=device)[:remainder]]
            ])
            indices = indices[torch.randperm(batch_size, device=device)]
        else:
            # 随机采样（batch_size 小时）
            indices = torch.randint(0, self.turbo_steps, (batch_size,), device=device)
        
        # 2. 获取对应的 sigma 并添加抖动
        # 抖动用于"增厚"流形，防止过拟合于精确的锚点
        # 确保 dtype 与 latents 一致，避免混合精度问题
        dtype = latents.dtype
        base_sigmas = self.anchor_sigmas.to(device=device, dtype=dtype)[indices]
        jitter = torch.randn_like(base_sigmas) * jitter_scale
        sampled_sigmas = (base_sigmas + jitter).clamp(0.001, 0.999) # 避免 0 和 1
        
        # 3. 计算对应的 timestep
        # Z-Image: timestep = sigma * 1000
        sampled_timesteps = sampled_sigmas * self.num_train_timesteps
        
        # 4. 扩展维度以匹配 latents shape (B, C, H, W)
        # 注意: Z-Image latents 是 (B, C, H, W)
        sigma_broadcast = sampled_sigmas.view(batch_size, 1, 1, 1)
        
        # 5. 线性插值构造 x_t
        # Z-Image: x_t = sigma * noise + (1 - sigma) * image
        # 注意: 这里 image 是 latents (x_0), noise 是 x_1
        noisy_latents = sigma_broadcast * noise + (1 - sigma_broadcast) * latents
        
        # 6. 计算目标速度
        # RF ODE: dx/dt = v(x, t)
        # 从 image(sigma=0) 到 noise(sigma=1)，速度 v = noise - image
        target_velocity = noise - latents
        
        return noisy_latents, sampled_timesteps, target_velocity

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target_velocity: torch.Tensor,
        latents_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        target_x0: torch.Tensor,
        lambda_charbonnier: float = 1.0,
        lambda_cosine: float = 0.1,
        lambda_fft: float = 0.1,
        snr_gamma: float = 5.0,  # Min-SNR 加权参数，0 表示禁用
        snr_floor: float = 0.1,  # Min-SNR 保底权重（10步模型关键参数）
    ) -> torch.Tensor:
        """
        计算 AC-RF 损失 (带 Floored Min-SNR 加权)
        
        Args:
            model_output: 模型预测的速度 v_pred
            target_velocity: 目标速度 v_target (noise - latents)
            latents_noisy: 加噪后的 latents (x_t)
            timesteps: 时间步
            target_x0: 原始 latents (x_0)
            lambda_charbonnier: Charbonnier Loss 权重
            lambda_cosine: Cosine Loss 权重
            lambda_fft: FFT Loss 权重
            snr_gamma: Min-SNR gamma 参数 (推荐 5.0)，设为 0 禁用
            snr_floor: Min-SNR 保底权重 (推荐 0.1)，确保高噪区参与训练
        """
        # 计算 Floored Min-SNR 权重
        # 标准 Min-SNR 在高噪区权重过低，导致 10 步模型无法学习构图
        # Floored Min-SNR 增加保底权重，确保每一步都参与训练
        if snr_gamma > 0:
            sigmas = timesteps.float() / self.num_train_timesteps
            sigmas_clamped = sigmas.clamp(min=0.001, max=0.999)
            snr = ((1 - sigmas_clamped) / sigmas_clamped) ** 2
            
            # v-prediction: weight = min(SNR, γ) / (SNR + 1)
            clipped_snr = torch.clamp(snr, max=snr_gamma)
            snr_weight = clipped_snr / (snr + 1)
            
            # Floored Min-SNR: 增加保底权重
            if snr_floor > 0:
                snr_weight = torch.maximum(snr_weight, torch.tensor(snr_floor, device=snr_weight.device))
            
            snr_weight = snr_weight.view(-1, 1, 1, 1)
        else:
            snr_weight = 1.0
        
        # 1. Charbonnier Loss (Robust L1) + SNR 加权
        # loss = sqrt((v_pred - v_target)^2 + eps^2)
        diff = model_output - target_velocity
        loss_per_sample = torch.sqrt(diff**2 + 1e-6)
        loss_charbonnier = (loss_per_sample * snr_weight).mean()
        
        # 2. Cosine Directional Loss
        # 鼓励速度方向一致
        # loss = 1 - cosine_similarity(v_pred, v_target)
        # Flatten to (B, -1) for cosine similarity
        pred_flat = model_output.view(model_output.shape[0], -1)
        target_flat = target_velocity.view(target_velocity.shape[0], -1)
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        # 使用张量减法避免 Python int/float 导致的类型提升
        one = torch.ones(1, device=model_output.device, dtype=model_output.dtype)
        loss_cosine = one.squeeze() - cos_sim
        
        # 3. FFT Loss (频域一致性)
        if lambda_fft > 0:
            # 推导预测的原图 (Predicted x0)
            # Z-Image 公式: x_t = sigma * v + x_0  =>  x_0 = x_t - sigma * v
            
            # 计算 sigma
            sigmas = timesteps / self.num_train_timesteps
            sigmas_broad = sigmas.view(-1, 1, 1, 1)
            
            # 重建 x0
            pred_x0 = latents_noisy - sigmas_broad * model_output
            
            # 计算 FFT Loss
            loss_fft = self._compute_fft_loss(pred_x0, target_x0)
        else:
            # 保持与 model_output 相同的 dtype，避免混合精度问题
            loss_fft = torch.zeros(1, device=model_output.device, dtype=model_output.dtype).squeeze()
        
        # 4. 总损失
        total_loss = (
            lambda_charbonnier * loss_charbonnier + 
            lambda_cosine * loss_cosine + 
            lambda_fft * loss_fft
        )
        
        return total_loss

    def _compute_fft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        高通补偿 FFT Loss - 只监督基础 loss 覆盖不到的高频区域
        
        设计理念：
        - 基础 loss (Charbonnier) 已经很好地监督了低频和中频
        - FFT Loss 专注于补偿高频细节，与基础 loss 互补而非竞争
        
        实现：
        1. 高通滤波：只保留归一化频率 > 0.25 的高频分量
        2. 平滑过渡：使用 sigmoid 平滑边界，避免硬截断伪影
        3. 对数压缩：log(1+amp) 让微弱高频信号可见
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        B, C, H, W = pred.shape
        original_dtype = pred.dtype
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        
        try:
            # 1. 2D FFT + shift (零频移到中心)
            pred_fft = torch.fft.fft2(pred_fp32, norm='ortho')
            target_fft = torch.fft.fft2(target_fp32, norm='ortho')
            pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
            target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))
            
            # 2. 构建高通滤波器
            # 计算归一化频率距离 [0, 1]
            # 使用 float32 计算滤波器（FFT 也是 float32），避免类型问题
            freq_y = torch.linspace(-0.5, 0.5, H, device=pred.device, dtype=torch.float32).view(-1, 1)
            freq_x = torch.linspace(-0.5, 0.5, W, device=pred.device, dtype=torch.float32).view(1, -1)
            freq_dist = torch.sqrt(freq_y ** 2 + freq_x ** 2)  # [0, ~0.707]
            freq_dist = freq_dist / 0.707  # 归一化到 [0, 1]
            
            # 高通滤波器：只保留高频 (freq > cutoff)
            # 使用 sigmoid 平滑过渡，避免硬截断
            # cutoff=0.25 表示只监督最外圈 75% 的高频区域
            # sharpness=12 控制过渡带宽度
            cutoff = 0.25
            sharpness = 12.0
            highpass_mask = torch.sigmoid((freq_dist - cutoff) * sharpness)
            highpass_mask = highpass_mask.view(1, 1, H, W)
            
            # 3. 提取幅度 + 对数压缩
            eps = 1e-8
            pred_amp = torch.log1p(pred_fft.abs() + eps)
            target_amp = torch.log1p(target_fft.abs() + eps)
            
            # 4. 只在高频区域计算 loss
            diff = (pred_amp - target_amp).abs()
            masked_diff = diff * highpass_mask
            
            # 5. 计算 loss (只对高频区域求均值)
            # 避免被零填充的低频区域稀释
            highpass_sum = highpass_mask.sum()
            if highpass_sum > 0:
                loss = (masked_diff.sum()) / (highpass_sum * B * C)
            else:
                loss = masked_diff.mean()
            
            return loss.to(original_dtype)
            
        except RuntimeError as e:
            if "powers of two" in str(e) or "cuFFT" in str(e):
                logger.debug(f"FFT loss disabled for dimensions {H}x{W}: {e}")
                return torch.tensor(0.0, device=pred.device, dtype=original_dtype)
            else:
                raise

    def verify_setup(self):
        """打印配置信息"""
        logger.info(f"训练时间步总数: {self.num_train_timesteps}")
        logger.info(f"Turbo 步数: {self.turbo_steps}")
        logger.info(f"Shift 参数: {self.shift}")
        
        # 一次性打印所有锚点
        anchor_info = " | ".join([f"{i}:{sigma:.3f}" for i, sigma in enumerate(self.anchor_sigmas)])
        logger.info(f"锚点配置 ({len(self.anchor_sigmas)}个): {anchor_info}")
        logger.info("=" * 60)


if __name__ == "__main__":
    # 验证脚本
    logging.basicConfig(level=logging.INFO)
    
    # 创建 trainer（使用 Z-Image 官方 shift 值）
    trainer = ACRFTrainer(turbo_steps=10, shift=3.0)
    trainer.verify_setup()
    
    # 测试 batch 采样
    batch_size = 4
    latents = torch.randn(batch_size, 16, 128, 128)
    noise = torch.randn_like(latents)
    
    noisy, timesteps, target = trainer.sample_batch(latents, noise)
    
    print(f"\n[OK] 测试通过！")
    print(f"输入形状: {noisy.shape}")
    print(f"Timesteps: {timesteps}")
    print(f"目标形状: {target.shape}")

