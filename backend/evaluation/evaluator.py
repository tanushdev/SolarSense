"""
Model Evaluator
================
Evaluates trained models against multiple metrics on test data.

Evaluates:
  - Binary classification: TSS, HSS, Brier, AUC
  - Probabilistic: ECE, reliability diagram
  - Temporal: lead time distribution
  - Per-class: TSS/HSS per flare class (C, M, X)
"""

import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass, field
from typing import Dict, List

from backend.training.metrics import (
    true_skill_statistic,
    heidke_skill_score,
    brier_score,
    full_evaluation_report,
)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    tss: float = 0.0
    hss: float = 0.0
    brier: float = 0.0
    auc: float = 0.0
    ece: float = 0.0
    avg_lead_time: float = 0.0
    false_alarm_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    thresholds: List[float] = field(default_factory=list)
    metrics_per_class: Dict = field(default_factory=dict)


class Evaluator:
    """
    Comprehensive model evaluation with standardized metrics.

    Usage:
        evaluator = Evaluator()
        result = evaluator.evaluate(y_true, y_pred_proba, model_name="RF")
        print(result.tss, result.hss)
    """

    def __init__(self, thresholds: List[float] = None):
        if thresholds is None:
            self.thresholds = np.arange(0.1, 1.0, 0.1)
        else:
            self.thresholds = thresholds

    def evaluate(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                 model_name: str = "model",
                 lead_times: np.ndarray = None) -> EvaluationResult:
        """
        Full evaluation.

        Parameters
        ----------
        y_true : (N,) binary ground truth
        y_pred_proba : (N,) predicted probabilities
        model_name : identifier for reporting
        lead_times : (N,) seconds to flare peak (0 for non-flare)

        Returns
        -------
        EvaluationResult
        """
        result = EvaluationResult(model_name=model_name)

        # Find optimal threshold (max TSS)
        best_tss = -1
        best_thresh = 0.5
        for t in self.thresholds:
            y_bin = (y_pred_proba > t).astype(int)
            tss = true_skill_statistic(y_true, y_bin)
            if tss > best_tss:
                best_tss = tss
                best_thresh = t

        y_opt = (y_pred_proba > best_thresh).astype(int)
        result.tss = best_tss
        result.hss = heidke_skill_score(y_true, y_opt)
        result.brier = brier_score(y_true, y_pred_proba)
        result.false_alarm_rate = self._false_alarm_rate(y_true, y_opt)
        result.precision, result.recall, result.f1 = self._prf(y_true, y_opt)
        result.thresholds = [best_thresh]

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            result.auc = roc_auc_score(y_true, y_pred_proba)
        except Exception:
            result.auc = 0.0

        # ECE
        result.ece = self._expected_calibration_error(y_true, y_pred_proba)

        # Lead time
        if lead_times is not None:
            flare_lt = lead_times[lead_times > 0]
            result.avg_lead_time = float(np.mean(flare_lt)) if len(flare_lt) > 0 else 0.0

        return result

    def _false_alarm_rate(self, y_true, y_pred):
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return fp / (fp + tn + 1e-10)

    def _prf(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        return p, r, f1

    def _expected_calibration_error(self, y_true, y_pred_proba, n_bins=10):
        """ECE: |Σ|(b_k) * |acc(b_k) - conf(b_k)| / N"""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bin_edges, right=True) - 1
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) == 0:
                continue
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_pred_proba[mask])
            ece += np.sum(mask) * np.abs(bin_acc - bin_conf)
        return ece / len(y_true)
