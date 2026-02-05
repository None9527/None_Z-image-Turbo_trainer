"""
Learning Rate Scheduler Utilities

Provides unified scheduler creation supporting both diffusers schedulers
and PyTorch's OneCycleLR for FP8 training optimization.
"""

from typing import Optional, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR


def get_scheduler_with_onecycle(
    scheduler_type: str,
    optimizer: Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    num_cycles: int = 1,
    # OneCycleLR specific
    max_lr: Optional[float] = None,
    pct_start: float = 0.1,
    div_factor: float = 10.0,
    final_div_factor: float = 100.0,
    anneal_strategy: str = "cos",
):
    """
    Create a learning rate scheduler with OneCycleLR support.
    
    For OneCycleLR, the lr schedule is:
        - Initial lr = max_lr / div_factor
        - Peak lr = max_lr (reached at pct_start of total steps)
        - Final lr = max_lr / final_div_factor
    
    Args:
        scheduler_type: One of "one_cycle", "constant", "linear", "cosine", 
                       "cosine_with_restarts", "constant_with_warmup"
        optimizer: PyTorch optimizer
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps (for non-OneCycleLR schedulers)
        num_cycles: Number of cycles (for cosine_with_restarts)
        max_lr: Maximum learning rate for OneCycleLR (uses optimizer's lr if None)
        pct_start: Percentage of training for warmup phase (OneCycleLR)
        div_factor: Initial lr divisor (OneCycleLR): initial_lr = max_lr/div_factor
        final_div_factor: Final lr divisor (OneCycleLR): final_lr = max_lr/final_div_factor
        anneal_strategy: Annealing strategy for OneCycleLR ("cos" or "linear")
    
    Returns:
        Learning rate scheduler
    """
    
    if scheduler_type == "one_cycle":
        # Get max_lr from optimizer if not specified
        if max_lr is None:
            max_lr = optimizer.defaults.get("lr", 1e-4)
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=num_training_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy=anneal_strategy,
        )
        return scheduler
    
    # Fallback to diffusers schedulers
    from diffusers.optimization import get_scheduler
    
    scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    
    return scheduler


def get_onecycle_lr_at_step(
    step: int,
    total_steps: int,
    max_lr: float,
    pct_start: float = 0.1,
    div_factor: float = 10.0,
    final_div_factor: float = 100.0,
) -> float:
    """
    Calculate OneCycleLR learning rate at a specific step.
    Useful for logging/visualization without actually creating a scheduler.
    
    Args:
        step: Current step
        total_steps: Total training steps
        max_lr: Maximum learning rate
        pct_start: Warmup percentage
        div_factor: Initial divisor
        final_div_factor: Final divisor
    
    Returns:
        Learning rate at the given step
    """
    import math
    
    initial_lr = max_lr / div_factor
    final_lr = max_lr / final_div_factor
    warmup_steps = int(total_steps * pct_start)
    
    if step < warmup_steps:
        # Warmup phase: linear increase from initial_lr to max_lr
        progress = step / warmup_steps
        return initial_lr + (max_lr - initial_lr) * progress
    else:
        # Annealing phase: cosine decay from max_lr to final_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return final_lr + (max_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * progress))
