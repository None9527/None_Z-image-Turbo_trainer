"""
AdamW with FP8 E5M2 State Storage

Uses PyTorch native float8_e5m2 for optimizer state (m, v) storage.
Provides better precision than bitsandbytes uint8 quantization while
maintaining similar memory savings (~75% reduction vs fp32).

Key differences from bitsandbytes:
- bitsandbytes: uint8 linear quantization + absmax scaling
- This: float8_e5m2 native format, natural dynamic range

Usage:
    optimizer = AdamWFP8(model.parameters(), lr=1e-4)
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional, Tuple, Dict, Any


class AdamWFP8(Optimizer):
    """
    AdamW optimizer with FP8 E5M2 state storage.
    
    Stores momentum (m) and variance (v) in float8_e5m2 format to reduce
    memory usage. Computation is done in fp32 for numerical stability.
    
    Memory comparison for 100M parameters:
    - AdamW fp32:     800MB (2 states × 4 bytes)
    - AdamW8bit:      200MB (2 states × 1 byte + scales)
    - AdamWFP8:       200MB (2 states × 1 byte, no extra scales)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-4)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.0)
        amsgrad: Whether to use AMSGrad variant (default: False)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)
        
        # Check FP8 support
        self._fp8_dtype = torch.float8_e5m2
        self._check_fp8_support()
    
    def _check_fp8_support(self):
        """Verify PyTorch FP8 support is available."""
        if not hasattr(torch, 'float8_e5m2'):
            raise RuntimeError(
                "PyTorch float8_e5m2 not available. "
                "Requires PyTorch >= 2.1. Current version: " + torch.__version__
            )
    
    def _to_fp8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to FP8 E5M2 format."""
        # Clamp to FP8 e5m2 representable range to avoid inf/nan
        # e5m2 max value is ~57344
        clamped = tensor.clamp(-57000, 57000)
        return clamped.to(self._fp8_dtype)
    
    def _from_fp8(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Convert FP8 tensor back to target dtype (usually fp32)."""
        return tensor.to(dtype)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamWFP8 does not support sparse gradients")
                
                amsgrad = group['amsgrad']
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Initialize in FP8 (zeros)
                    state['exp_avg'] = torch.zeros_like(p, dtype=self._fp8_dtype)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=self._fp8_dtype)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, dtype=self._fp8_dtype)
                
                state['step'] += 1
                
                # Upcast FP8 states to computation dtype (fp32/bf16)
                compute_dtype = p.dtype if p.dtype in [torch.float32, torch.float64] else torch.float32
                
                exp_avg = self._from_fp8(state['exp_avg'], compute_dtype)
                exp_avg_sq = self._from_fp8(state['exp_avg_sq'], compute_dtype)
                
                if amsgrad:
                    max_exp_avg_sq = self._from_fp8(state['max_exp_avg_sq'], compute_dtype)
                
                # Bias correction
                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Decoupled weight decay (AdamW style)
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if amsgrad:
                    # Maintains max of all 2nd moment running avg till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use max for normalization
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                    # Downcast back to FP8
                    state['max_exp_avg_sq'] = self._to_fp8(max_exp_avg_sq)
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Downcast states back to FP8 for storage
                state['exp_avg'] = self._to_fp8(exp_avg)
                state['exp_avg_sq'] = self._to_fp8(exp_avg_sq)
        
        return loss
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict with FP8 states converted to fp32 for saving."""
        state_dict = super().state_dict()
        
        # Convert FP8 states to fp32 for compatibility
        for param_id, param_state in state_dict['state'].items():
            for key in ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']:
                if key in param_state and param_state[key].dtype == self._fp8_dtype:
                    param_state[key] = param_state[key].to(torch.float32)
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict, converting fp32 states to FP8."""
        # Convert fp32 states to FP8 before loading
        for param_id, param_state in state_dict['state'].items():
            for key in ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']:
                if key in param_state and param_state[key].dtype != self._fp8_dtype:
                    param_state[key] = self._to_fp8(param_state[key])
        
        super().load_state_dict(state_dict)
