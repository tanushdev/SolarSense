"""
Bayesian Uncertainty Quantification
=====================================
Uncertainty estimation methods for solar flare predictions.

TWO COMPLEMENTARY APPROACHES:
  1. MC Dropout (model-agnostic, no extra training)
     - Apply dropout at inference time
     - Run N forward passes
     - Mean = prediction, Std = epistemic uncertainty
  2. Conformal Prediction (distribution-free)
     - Calibrate on validation set
     - Guaranteed 90% coverage (finite-sample valid)
     - Produces prediction intervals

USAGE:
    uncertainty = MCDropout(model, n_passes=50)
    result = uncertainty.predict(X)
    # Returns: probability, uncertainty, lower_bound, upper_bound

    calibrator = ConformalCalibrator()
    calibrator.fit(val_preds, val_labels)
    intervals = calibrator.predict(test_preds)
"""

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


class MCDropout:
    """
    Monte Carlo Dropout for epistemic uncertainty estimation.

    Uses dropout at inference time to sample from approximate posterior.
    N forward passes → empirical distribution over predictions.

    Usage:
        mc = MCDropout(model, n_passes=50)
        result = mc.predict(X_tensor)
    """

    def __init__(self, model: torch.nn.Module, n_passes: int = 50):
        self.model = model
        self.n_passes = n_passes
        self._training_mode = None

    def predict(self, x_soft: torch.Tensor,
                x_hard: torch.Tensor) -> dict:
        """
        Run MC Dropout inference.

        Parameters
        ----------
        x_soft : (B, T, F) or (T, F)
        x_hard : (B, T, F) or (T, F)

        Returns
        -------
        dict with probability, uncertainty, lower_bound, upper_bound
        """
        was_training = self.model.training
        self.model.train()  # Enable dropout

        if x_soft.dim() == 2:
            x_soft = x_soft.unsqueeze(0)
        if x_hard.dim() == 2:
            x_hard = x_hard.unsqueeze(0)

        all_probs = []
        with torch.no_grad():
            for _ in range(self.n_passes):
                logits = self.model(x_soft, x_hard)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        self.model.train(was_training)

        all_probs = np.stack(all_probs, axis=0)  # (N, B)
        mean_prob = np.mean(all_probs, axis=0)
        std_prob = np.std(all_probs, axis=0)
        lower = np.percentile(all_probs, 5, axis=0)
        upper = np.percentile(all_probs, 95, axis=0)

        return {
            "probability": mean_prob.squeeze(),
            "uncertainty": std_prob.squeeze(),
            "lower_bound": lower.squeeze(),
            "upper_bound": upper.squeeze(),
        }


class ConformalCalibrator:
    """
    Conformal Prediction calibration.

    Provides distribution-free, finite-sample valid prediction intervals.
    Guarantees that P(y_true ∈ interval) ≥ 1 - α for any α ∈ (0,1).

    Usage:
        calibrator = ConformalCalibrator(alpha=0.1)
        calibrator.fit(val_probs, val_labels)
        intervals = calibrator.predict(test_probs)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_threshold = None
        self._fitted = False

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        """
        Calibrate on validation set.

        Parameters
        ----------
        val_probs : (N,) predicted probabilities on validation set
        val_labels : (N,) ground truth binary labels
        """
        # Nonconformity scores: |y - p|
        scores = np.abs(val_labels - val_probs)
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_threshold = np.quantile(scores, q_level, method="higher")
        self._fitted = True
        logger.info("Conformal: q_threshold={:.4f} (alpha={}, n={})",
                    self.q_threshold, self.alpha, n)

    def predict(self, probs: np.ndarray) -> dict:
        """
        Compute prediction intervals at calibrated confidence level.

        Returns
        -------
        dict with lower_bound, upper_bound for each input
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call .fit() first.")
        lower = np.clip(probs - self.q_threshold, 0, 1)
        upper = np.clip(probs + self.q_threshold, 0, 1)
        return {
            "lower_bound": lower,
            "upper_bound": upper,
            "q_threshold": self.q_threshold,
        }

    def coverage(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute empirical coverage of prediction intervals."""
        intervals = self.predict(probs)
        covered = (labels >= intervals["lower_bound"]) & \
                  (labels <= intervals["upper_bound"])
        return covered.mean()
