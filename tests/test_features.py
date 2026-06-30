"""Tests for all feature extractors."""

import pytest
import numpy as np
import pandas as pd

from backend.features.physics_features import PhysicsFeatureExtractor
from backend.features.spectral_features import SpectralFeatureExtractor
from backend.features.temporal_features import TemporalFeatureExtractor
from backend.features.statistical_features import StatisticalFeatureExtractor
from backend.features.feature_pipeline import FeaturePipeline


@pytest.fixture
def sample_data():
    """Generate 1 hour of synthetic dual-channel data at 5s cadence."""
    n = 720
    idx = pd.date_range("2024-07-06", periods=n, freq="5s", tz="UTC")
    np.random.seed(42)
    soft = 15.0 + np.random.normal(0, 0.1, n)  # Background ~15 counts
    hard = 1.0 + np.random.exponential(0.5, n)  # Background ~1 count
    # Add a flare-like spike at t=300
    spike = np.exp(-np.linspace(0, 5, 120)**2 / 2) * 100
    soft[300:420] += spike
    hard[300:420] += spike * 5
    return pd.DataFrame({
        "soft_flux": soft,
        "hard_flux": hard,
        "soft_counts": soft,
        "hard_counts": hard,
    }, index=idx)


class TestPhysicsFeatures:
    def test_extract_all(self, sample_data):
        extractor = PhysicsFeatureExtractor()
        feats = extractor.extract_all(sample_data)
        assert len(feats) > 0
        assert "hardness_ratio" in feats.columns
        assert "d_soft_flux_dt" in feats.columns
        assert "d_hard_flux_dt" in feats.columns
        assert "spectral_index" in feats.columns

    def test_hardness_ratio_spike_during_flare(self, sample_data):
        extractor = PhysicsFeatureExtractor()
        feats = extractor.extract_all(sample_data)
        # Hardness ratio should be higher during the flare
        pre_hr = feats["hardness_ratio"].iloc[280:300].mean()
        during_hr = feats["hardness_ratio"].iloc[310:330].mean()
        assert during_hr > pre_hr


class TestSpectralFeatures:
    def test_extract_all(self, sample_data):
        extractor = SpectralFeatureExtractor()
        feats = extractor.extract_all(sample_data)
        assert len(feats) > 0
        assert any("fft" in c for c in feats.columns)
        assert any("wavelet" in c for c in feats.columns)


class TestTemporalFeatures:
    def test_extract_all(self, sample_data):
        extractor = TemporalFeatureExtractor()
        feats = extractor.extract_all(sample_data)
        assert len(feats) > 0
        assert "hour_sin" in feats.columns
        assert "doy_sin" in feats.columns


class TestStatisticalFeatures:
    def test_extract_all(self, sample_data):
        extractor = StatisticalFeatureExtractor()
        feats = extractor.extract_all(sample_data)
        assert len(feats) > 0
        assert any("iqr" in c for c in feats.columns)
        assert any("acf" in c for c in feats.columns)


class TestFeaturePipeline:
    def test_pipeline_runs(self, sample_data):
        pipeline = FeaturePipeline()
        feats = pipeline.extract_all(sample_data)
        assert len(feats) > 0
        assert len(feats.columns) >= 10  # Expect at least 10 features total
