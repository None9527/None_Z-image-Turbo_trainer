"""
DPO Loss - Direct Preference Optimization for Diffusion Models

Implements DPO loss function for Flow Matching / v-prediction based models.
Based on: https://arxiv.org/abs/2311.12908 (Diffusion-DPO)

Key insight: Instead of training a reward model, DPO directly optimizes
the policy to prefer "winning" samples over "losing" samples using the
implicit reward formulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class DPOLoss(nn.Module):
    """
    Diffusion DPO Loss - Direct Preference Optimization for diffusion models.
    
    Works by comparing policy model predictions against a reference model
    on preferred (winning) vs rejected (losing) image pairs.
    
    Core formula:
        policy_diff = mse(policy_pred_w, target_w) - mse(policy_pred_l, target_l)
        ref_diff = mse(ref_pred_w, target_w) - mse(ref_pred_l, target_l)
        logits = ref_diff - policy_diff
        loss = -log(sigmoid(beta * logits))
    
    Args:
        beta: DPO regularization coefficient. Higher = stronger preference learning.
              Recommended range: 2000-5000 for diffusion models.
        loss_type: Type of DPO loss function.
            - 'sigmoid': Standard DPO loss (default)
            - 'hinge': Hinge-style loss
            - 'ipo': Identity Preference Optimization
        label_smoothing: Optional label smoothing factor (0-0.5)
    """
    
    def __init__(
        self,
        beta: float = 2500.0,
        loss_type: str = "sigmoid",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        
        if loss_type not in ["sigmoid", "hinge", "ipo"]:
            raise ValueError(f"Unknown loss_type: {loss_type}. Use 'sigmoid', 'hinge', or 'ipo'.")
    
    def forward(
        self,
        policy_pred_w: torch.Tensor,
        policy_pred_l: torch.Tensor,
        ref_pred_w: torch.Tensor,
        ref_pred_l: torch.Tensor,
        target_w: torch.Tensor,
        target_l: torch.Tensor,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.
        
        Args:
            policy_pred_w: Policy model prediction for preferred (winning) samples [B, C, H, W]
            policy_pred_l: Policy model prediction for rejected (losing) samples [B, C, H, W]
            ref_pred_w: Reference model prediction for preferred samples [B, C, H, W]
            ref_pred_l: Reference model prediction for rejected samples [B, C, H, W]
            target_w: Target velocity for preferred samples [B, C, H, W]
            target_l: Target velocity for rejected samples [B, C, H, W]
            reduction: 'mean' | 'none'
        
        Returns:
            loss: Scalar DPO loss (or per-sample if reduction='none')
            info: Dictionary with diagnostic metrics
        """
        # Compute MSE losses per sample (reduce over spatial dims, keep batch)
        # Shape: [B]
        policy_loss_w = F.mse_loss(policy_pred_w, target_w, reduction="none").mean(dim=list(range(1, policy_pred_w.dim())))
        policy_loss_l = F.mse_loss(policy_pred_l, target_l, reduction="none").mean(dim=list(range(1, policy_pred_l.dim())))
        
        ref_loss_w = F.mse_loss(ref_pred_w, target_w, reduction="none").mean(dim=list(range(1, ref_pred_w.dim())))
        ref_loss_l = F.mse_loss(ref_pred_l, target_l, reduction="none").mean(dim=list(range(1, ref_pred_l.dim())))
        
        # Compute loss differences
        # Positive model_diff means policy prefers winning over losing (good!)
        policy_diff = policy_loss_w - policy_loss_l
        ref_diff = ref_loss_w - ref_loss_l
        
        # DPO logits: how much better is policy at distinguishing w/l compared to ref
        # Positive logits = policy is learning the preference
        logits = ref_diff - policy_diff
        
        # Compute loss based on type
        if self.loss_type == "sigmoid":
            # Standard DPO: maximize probability of preferring w over l
            if self.label_smoothing > 0:
                # Smoothed labels: instead of (1, 0), use (1-eps, eps)
                loss = -1.0 * (
                    (1 - self.label_smoothing) * F.logsigmoid(self.beta * logits) +
                    self.label_smoothing * F.logsigmoid(-self.beta * logits)
                )
            else:
                loss = -1.0 * F.logsigmoid(self.beta * logits)
        
        elif self.loss_type == "hinge":
            # Hinge loss: margin-based approach
            loss = torch.relu(1.0 - self.beta * logits)
        
        elif self.loss_type == "ipo":
            # Identity Preference Optimization
            # Directly regress logits toward 1/(2*beta)
            losses = (logits - 1.0 / (2.0 * self.beta)) ** 2
            loss = losses
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Reduction
        if reduction == "mean":
            loss = loss.mean()
        
        # Compute implicit accuracy (how often policy correctly prefers w over l)
        with torch.no_grad():
            implicit_acc = (logits > 0).float().mean().item()
            # Account for ties
            implicit_acc += 0.5 * (logits == 0).float().mean().item()
        
        # Diagnostic info
        info = {
            "implicit_acc": implicit_acc,
            "policy_diff": policy_diff.mean().item(),
            "ref_diff": ref_diff.mean().item(),
            "logits_mean": logits.mean().item(),
            "policy_loss_w": policy_loss_w.mean().item(),
            "policy_loss_l": policy_loss_l.mean().item(),
            "ref_loss_w": ref_loss_w.mean().item(),
            "ref_loss_l": ref_loss_l.mean().item(),
        }
        
        return loss, info


class DPOLossWithSNR(DPOLoss):
    """
    DPO Loss with SNR (Signal-to-Noise Ratio) weighting.
    
    Applies timestep-dependent weighting to better balance loss
    contributions across the denoising trajectory.
    """
    
    def __init__(
        self,
        beta: float = 2500.0,
        loss_type: str = "sigmoid",
        snr_gamma: float = 5.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__(beta, loss_type, label_smoothing)
        self.snr_gamma = snr_gamma
    
    def forward(
        self,
        policy_pred_w: torch.Tensor,
        policy_pred_l: torch.Tensor,
        ref_pred_w: torch.Tensor,
        ref_pred_l: torch.Tensor,
        target_w: torch.Tensor,
        target_l: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        num_train_timesteps: int = 1000,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss with optional SNR weighting.
        
        Additional Args:
            timesteps: Timestep for each sample [B]
            num_train_timesteps: Total number of training timesteps
        """
        loss, info = super().forward(
            policy_pred_w, policy_pred_l,
            ref_pred_w, ref_pred_l,
            target_w, target_l,
            reduction="none",
        )
        
        # Apply SNR weighting if timesteps provided
        if timesteps is not None:
            from zimage_trainer.utils.snr_utils import compute_snr_weights
            snr_weights = compute_snr_weights(
                timesteps=timesteps,
                num_train_timesteps=num_train_timesteps,
                snr_gamma=self.snr_gamma,
                prediction_type="v_prediction",
            )
            snr_weights = snr_weights.to(loss.device, dtype=loss.dtype)
            loss = loss * snr_weights
        
        if reduction == "mean":
            loss = loss.mean()
        
        return loss, info
