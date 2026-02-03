"""
AdamW with BF16 State Storage

Uses torch.bfloat16 for optimizer state (m, v) storage.
BF16 offers better dynamic range than FP16 and no scaling needed.

Memory comparison for 100M parameters:
- AdamW fp32:     800MB (2 states × 4 bytes)
- AdamWBF16:      400MB (2 states × 2 bytes)
- AdamWFP8:       200MB (2 states × 1 byte + scales)

Usage:
    optimizer = AdamWBF16(model.parameters(), lr=1e-4)
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, Dict, Any


class AdamWBF16(Optimizer):
    """
    AdamW optimizer with BF16 state storage.
    
    Stores momentum (m) and variance (v) in bfloat16 format to reduce
    memory usage by 50% compared to fp32. Computation is done in fp32
    for numerical stability.
    
    BF16 advantages over FP8:
    - No scaling needed (same exponent range as fp32)
    - More mantissa bits (7 vs 2-3)
    - Better numerical stability
    
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
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
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
                    raise RuntimeError("AdamWBF16 does not support sparse gradients")
                
                amsgrad = group['amsgrad']
                state = self.state[p]
                
                # State initialization - directly in BF16
                if len(state) == 0:
                    state['step'] = 0
                    # BF16 zeros - saves 50% memory vs fp32
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.bfloat16)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, dtype=torch.bfloat16)
                
                state['step'] += 1
                step = state['step']
                
                # Upcast BF16 states to fp32 for computation
                exp_avg = state['exp_avg'].to(torch.float32)
                exp_avg_sq = state['exp_avg_sq'].to(torch.float32)
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq'].to(torch.float32)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Decoupled weight decay (AdamW style)
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Cast grad to fp32 for computation
                grad_fp32 = grad.to(torch.float32)
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)
                
                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                
                # Update parameters (cast to param dtype)
                update = exp_avg / denom * step_size
                p.data.add_(update.to(p.dtype), alpha=-1)
                
                # Downcast states back to BF16 for storage
                state['exp_avg'] = exp_avg.to(torch.bfloat16)
                state['exp_avg_sq'] = exp_avg_sq.to(torch.bfloat16)
                if amsgrad:
                    state['max_exp_avg_sq'] = max_exp_avg_sq.to(torch.bfloat16)
        
        return loss
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict with BF16 states converted to fp32 for saving."""
        state_dict = super().state_dict()
        
        # Convert BF16 states to fp32 for compatibility
        for param_id, param_state in state_dict['state'].items():
            for key in ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']:
                if key in param_state and param_state[key].dtype == torch.bfloat16:
                    param_state[key] = param_state[key].to(torch.float32)
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict, converting fp32 states to BF16."""
        # Convert fp32 states to BF16 before loading
        for param_id, param_state in state_dict['state'].items():
            for key in ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']:
                if key in param_state and param_state[key].dtype == torch.float32:
                    param_state[key] = param_state[key].to(torch.bfloat16)
        
        super().load_state_dict(state_dict)
