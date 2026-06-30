"""Tests for training modules."""

import pytest
import numpy as np
import torch

from backend.training.losses import (
    WeightedBCEWithLogitsLoss,
    FocalLoss,
    LeadTimeAwareLoss,
    CombinedLoss,
)
from backend.training.callbacks import FlareMetricTracker, GradientClippingCallback
from backend.training.metrics import (
    true_skill_statistic,
    heidke_skill_score,
    brier_score,
    full_evaluation_report,
)
from backend.training.experiment_logger import ExperimentLogger


class TestLosses:
    def test_weighted_bce(self):
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=10.0)
        logits = torch.tensor([2.0, -2.0, 0.5, -0.5])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_focal_loss(self):
        loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
        logits = torch.tensor([3.0, -3.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_lead_time_aware_loss(self):
        loss_fn = LeadTimeAwareLoss(tau=300.0)
        logits = torch.tensor([1.0, -1.0, 0.5])
        targets = torch.tensor([1.0, 0.0, 1.0])
        lead_times = torch.tensor([60.0, 0.0, 600.0])
        loss = loss_fn(logits, targets, lead_times)
        assert loss.item() > 0


class TestMetrics:
    def test_tss_perfect(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        tss = true_skill_statistic(y_true, y_pred)
        assert tss == pytest.approx(1.0)

    def test_tss_random(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        tss = true_skill_statistic(y_true, y_pred)
        assert tss == pytest.approx(0.0, abs=0.01)

    def test_hss_perfect(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        hss = heidke_skill_score(y_true, y_pred)
        assert hss == pytest.approx(1.0)

    def test_brier_score(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2])
        brier = brier_score(y_true, y_pred)
        assert 0 < brier < 1


class TestExperimentLogger:
    def test_log_and_finalize(self, tmp_path):
        logger = ExperimentLogger(
            experiment_name="test_exp",
            base_dir=str(tmp_path),
        )
        logger.log_params({"lr": 0.001, "epochs": 10})
        logger.log_metrics({"tss": 0.85, "hss": 0.72})
        logger.finalize("completed")
        assert (tmp_path / "test_exp" / "experiment.json").exists()
        assert (tmp_path / "test_exp" / "metrics.csv").exists()
