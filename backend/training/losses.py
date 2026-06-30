"""
Custom Loss Functions for Solar Flare Prediction
==================================================
All losses return a scalar tensor suitable for .backward().

FLARE-SPECIFIC CHALLENGES:
  - Extreme class imbalance (flare samples < 5% of data)
  - Event is rare but MUST be detected (asymmetric cost)
  - Lead time matters: an alert 1 minute before peak is less useful
    than one 10 minutes before → incorporate lead time into loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy with class weighting for extreme imbalance.

    Weights: w₁ (flare) >> w₀ (non-flare)
    Typical ratio: w₁/w₀ = number_of_nonflare / number_of_flare

    Usage:
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=10.0)
        loss = loss_fn(logits, targets)
    """

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.where(targets > 0.5, self.pos_weight, 1.0)
        loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples, focuses on hard misclassifications.

    FL(pₜ) = -αₜ (1 - pₜ)^γ log(pₜ)

    γ = 0  → standard cross-entropy
    γ = 2  → strong focus on hard examples (recommended for class imbalance)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


class LeadTimeAwareLoss(nn.Module):
    """
    Loss that penalizes late predictions more than early ones.

    For flare events, the loss is scaled by:
        w_lead = exp(-lead_time / tau)

    where lead_time is the time to flare peak.
    Short lead time → larger penalty.

    This encourages the model to predict flares earlier,
    which is the operational requirement (>10 min lead time).
    """

    def __init__(self, tau: float = 300.0, base_loss: str = "focal"):
        super().__init__()
        self.tau = tau
        if base_loss == "focal":
            self.base_loss = FocalLoss()
        else:
            self.base_loss = WeightedBCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                lead_times: torch.Tensor = None) -> torch.Tensor:
        base = self.base_loss(logits, targets)
        if lead_times is None:
            return base
        # lead_times > 0 → flare sample, lead_times == 0 → non-flare
        lead_weight = torch.exp(-lead_times / self.tau)
        lead_weight = torch.where(lead_times > 0, lead_weight, 1.0)
        weighted = base * lead_weight
        return weighted.mean()


class QuantileLoss(nn.Module):
    """
    Pinball (quantile) loss for uncertainty estimation.

    Produces prediction intervals instead of point forecasts.
    Used for conformal prediction calibration.

    Loss = max(q * (y - ŷ), (1-q) * (ŷ - y)) for quantile q.
    """

    def __init__(self, quantiles: list = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, targets: torch.Tensor,
                quantile_idx: int) -> torch.Tensor:
        q = self.quantiles[quantile_idx]
        errors = targets - preds
        loss = torch.max(q * errors, (q - 1) * errors)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combines multiple losses:
      - Classification loss (focal) for flare/no-flare
      - Quantile loss for uncertainty intervals
      - Lead time penalty for late detections
    """

    def __init__(self, focal_alpha: float = 0.75, focal_gamma: float = 2.0,
                 lead_tau: float = 300.0):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.lead_aware = LeadTimeAwareLoss(tau=lead_tau)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                lead_times: torch.Tensor = None) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        lead_loss = self.lead_aware(logits, targets, lead_times)
        return 0.7 * focal_loss + 0.3 * lead_loss
