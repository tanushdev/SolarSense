"""Tests for all model implementations."""

import pytest
import numpy as np
import pandas as pd
import torch

from backend.models.nowcaster.threshold_detector import ThresholdNowcaster
from backend.models.forecaster.random_forest_forecaster import RandomForestForecaster
from backend.models.bayesian.uncertainty import MCDropout, ConformalCalibrator


@pytest.fixture
def sample_classification_data():
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 10)
    y = (X[:, 0] + X[:, 1] > 1).astype(float)
    return X, y


class TestThresholdNowcaster:
    def _make_df(self, n=1000):
        idx = pd.date_range("2024-07-06", periods=n, freq="5s", tz="UTC")
        return pd.DataFrame({
            "soft_flux": np.random.exponential(15, n),
            "hard_flux": np.random.exponential(5, n),
        }, index=idx)

    def test_detect_returns_events(self):
        nowcaster = ThresholdNowcaster()
        df = self._make_df()
        df.iloc[300:320] = [50, 100]  # Clear spike
        events = nowcaster.detect(df)
        assert len(events) >= 1

    def test_to_catalog_format(self):
        nowcaster = ThresholdNowcaster()
        df = self._make_df()
        df.iloc[300:320] = [50, 100]  # Add flare spike
        events = nowcaster.detect(df)
        assert len(events) >= 1, "Detector should find events with spike"
        catalog = nowcaster.to_catalog(events)
        assert "start_time" in catalog.columns
        assert "peak_time" in catalog.columns
        assert "flare_class" in catalog.columns


class TestRandomForest:
    def test_fit_predict(self, sample_classification_data):
        X, y = sample_classification_data
        model = RandomForestForecaster()
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert len(probs) == len(X)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_uncertainty(self, sample_classification_data):
        X, y = sample_classification_data
        model = RandomForestForecaster()
        model.fit(X, y)
        result = model.predict_with_uncertainty(X)
        assert "probability" in result
        assert "uncertainty" in result
        assert "lower_bound" in result
        assert "upper_bound" in result


class TestBayesianUncertainty:
    def test_conformal_calibration(self):
        np.random.seed(42)
        val_probs = np.random.uniform(0, 1, 500)
        val_labels = (val_probs > 0.7).astype(float)
        calibrator = ConformalCalibrator(alpha=0.1)
        calibrator.fit(val_probs, val_labels)
        test_probs = np.random.uniform(0, 1, 200)
        result = calibrator.predict(test_probs)
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert result["q_threshold"] > 0
