"""End-to-end integration tests for the full SolarSense pipeline."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


@pytest.fixture(scope="module")
def client():
    from backend.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_parquet(tmp_path):
    idx = pd.date_range("2024-07-01", periods=200, freq="s")
    np.random.seed(42)
    data = {
        "soft_flux": np.random.rand(200) * 10 + 10,
        "hard_flux": np.random.rand(200) * 5 + 2,
        "hardness_ratio": np.random.rand(200) * 0.3,
        "soft_flux_fft_1": np.random.randn(200),
        "soft_flux_fft_2": np.random.randn(200),
        "hard_flux_fft_1": np.random.randn(200),
        "hard_flux_fft_2": np.random.randn(200),
        "soft_flux_wavelet_1": np.random.randn(200),
        "hard_flux_wavelet_1": np.random.randn(200),
        "soft_flux_roll_mean_10": np.random.rand(200) * 10 + 10,
        "soft_flux_roll_std_10": np.random.rand(200) * 2,
        "hard_flux_roll_mean_10": np.random.rand(200) * 5 + 2,
        "hard_flux_roll_std_10": np.random.rand(200),
        "temp_approximate": np.random.rand(200) * 5 + 5,
        "emission_measure": np.random.rand(200) * 1e49,
        "cross_corr_lag": np.random.randn(200),
        "dynamic_lag": np.random.randn(200),
        "spectral_index": np.random.rand(200) * 0.5 + 1.5,
        "quality_solexs": np.zeros(200),
        "quality_hel1os": np.zeros(200),
        "data_gap": np.zeros(200),
        "artifact_flag": np.zeros(200),
    }
    df = pd.DataFrame(data, index=idx)
    path = tmp_path / "test_merged.parquet"
    df.to_parquet(path)
    return path


# ── API Tests ─────────────────────────────────────────────────────────

class TestAPI:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body

    def test_forecast(self, client):
        resp = client.get("/forecast")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("flare_probability", "predicted_class",
                     "lead_time_minutes", "alert_level", "model",
                     "similar_events", "physics_reason"):
            assert key in body, f"Missing key: {key}"
        assert 0 <= body["flare_probability"] <= 1

    def test_predict_post(self, client):
        payload = {
            "timestamps": [1.0, 2.0, 3.0],
            "soft_flux": [10.0, 10.5, 11.0],
            "hard_flux": [2.0, 2.5, 3.0],
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "flare_probability" in body

    def test_nowcast(self, client):
        payload = {
            "timestamps": [1.0, 2.0, 3.0],
            "soft_flux": [10.0, 10.5, 11.0],
            "hard_flux": [2.0, 2.5, 3.0],
        }
        resp = client.post("/nowcast", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "is_flare" in body

    def test_history(self, client):
        resp = client.get("/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_lightcurve(self, client):
        resp = client.get("/lightcurve?hours=6")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_noaa(self, client):
        resp = client.get("/noaa")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "flux_class" in body

    def test_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("tss", "hss", "roc_auc", "model", "total_predictions"):
            assert key in body

    def test_models(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        body = resp.json()
        assert "active_model" in body
        assert "available_models" in body

    def test_alerts(self, client):
        resp = client.get("/alerts")
        assert resp.status_code == 200
        body = resp.json()
        assert "alert" in body
        assert "alert_level" in body
        assert "above_threshold" in body
        assert "validation_status" in body

    def test_validation_report(self, client):
        resp = client.get("/validation/report")
        assert resp.status_code == 200
        body = resp.json()
        assert "detection_rate" in body


# ── Pipeline Tests ────────────────────────────────────────────────────

class TestPipeline:
    def test_reader_imports(self):
        from backend.data import fits_utils, solexs_reader, hel1os_reader
        assert hasattr(fits_utils, "is_lightcurve_file")
        assert hasattr(solexs_reader, "SoLEXSReader")
        assert hasattr(hel1os_reader, "HEL1OSReader")

    def test_features_smoke(self, sample_parquet):
        from backend.features.feature_pipeline import FeaturePipeline
        df = pd.read_parquet(sample_parquet)
        pipe = FeaturePipeline()
        result = pipe.extract_all(df.copy())
        assert result is not None
        assert len(result) > 0

    def test_prediction_persistence(self, tmp_path):
        from backend.services.prediction_store import PredictionStore
        db_path = tmp_path / "test_predictions.db"
        store = PredictionStore(str(db_path))
        pred = {
            "prediction_id": "test_id_001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "flare_probability": 0.85,
            "uncertainty": 0.10,
            "predicted_class": "M",
            "lead_time_minutes": 12.5,
            "alert_level": "ORANGE",
            "model": "xgboost",
            "threshold": 0.5,
            "forecast_horizon_minutes": 30,
            "data_timestamp": "2024-07-01T12:00:00",
            "dataset_version": "abc123",
            "feature_version": "def456",
            "config_version": "ghi789",
            "git_commit": "main",
            "model_tag": "benchmark",
            "validation_status": "pending",
            "similar_events": [{"date": "2024-05-11", "flare_class": "M2.3",
                                "similarity": 0.95, "lead_time": 10.0,
                                "description": "Test event"}],
            "physics_reason": "Hard X-ray flux increasing • Hardness ratio rising",
        }
        store.save_prediction(pred, execution_time_ms=45.2)
        recent = store.get_recent_predictions(limit=5)
        assert len(recent) >= 1
        assert recent[0]["prediction_id"] == "test_id_001"
        stats = store.get_stats()
        assert stats["total_predictions"] >= 1

    def test_calibrator(self):
        from backend.evaluation.calibration import Calibrator
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 1000)
        y_prob = np.clip(y_true + np.random.randn(1000) * 0.3, 0.01, 0.99)
        calibrator = Calibrator()
        calibrator.fit(y_true, y_prob)
        results = calibrator.evaluate(y_true, y_prob)
        assert "chosen_method" in results
        assert results["chosen_method"] in ("platt", "isotonic", "none")
        cal_probs = calibrator.predict_proba(y_prob)
        assert len(cal_probs) == len(y_prob)
        assert np.all((cal_probs >= 0) & (cal_probs <= 1))

    def test_versioning(self):
        from backend.services.versioning import PredictionVersion
        v = PredictionVersion(model_name="xgboost", model_tag="benchmark", threshold=0.5)
        d = v.to_dict()
        assert "dataset_version" in d
        assert "model_name" in d
        assert d["model_name"] == "xgboost"
        assert len(v.to_id()) == 12

    def test_forecast_horizon_labels(self):
        from backend.pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.active_horizon = 30
        pipeline.label_mode = "fixed_window"

        idx = pd.date_range("2024-07-01", periods=100, freq="min")
        df = pd.DataFrame({"soft_flux": np.random.rand(100) * 10 + 10,
                            "hard_flux": np.random.rand(100) * 5 + 2}, index=idx)

        pipeline.catalog = pd.DataFrame([
            {"peak_time": "2024-07-01 01:00:00",
             "start_time": "2024-07-01 00:45:00",
             "end_time": "2024-07-01 01:15:00",
             "flare_class": "C"}
        ])
        result = pipeline.build_labels(df)
        assert "flare_label" in result
        assert "lead_time" in result
        assert result["flare_label"].sum() > 0
        assert result["lead_time"].max() > 0

    def test_full_pipeline_smoke(self, sample_parquet, tmp_path):
        from backend.features.feature_pipeline import FeaturePipeline
        df = pd.read_parquet(sample_parquet)
        pipe = FeaturePipeline()
        featured = pipe.extract_all(df.copy())
        assert len(featured) > 0

        from backend.services.prediction_store import PredictionStore
        db_path = tmp_path / "e2e_predictions.db"
        store = PredictionStore(str(db_path))
        store.save_prediction({
            "prediction_id": "e2e_test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "flare_probability": 0.12,
            "predicted_class": "B",
            "lead_time_minutes": 0.0,
            "alert_level": "GREEN",
            "model": "test",
            "data_timestamp": "",
            "threshold": 0.5,
            "forecast_horizon_minutes": 30,
        })
        assert store.get_stats()["total_predictions"] >= 1


# ── Error Handling Tests ──────────────────────────────────────────────

class TestErrorHandling:
    def test_noaa_fallback(self, client):
        resp = client.get("/noaa")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("live", "cached", "stale", "initializing")

    def test_invalid_predict(self, client):
        payload = {"timestamps": [], "soft_flux": [], "hard_flux": []}
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (200, 422)

    def test_missing_lightcurve_param(self, client):
        resp = client.get("/lightcurve")
        assert resp.status_code == 200

    def test_validation_no_dates(self, client):
        resp = client.get("/validation/report")
        assert resp.status_code == 200


# ── New Feature Tests ────────────────────────────────────────────

class TestNewFeatures:
    def test_leakage_audit_script(self):
        """Leakage audit: verify no rolling(center=True) in physics_features."""
        # Confirm the fix: physics_features no longer uses center=True
        from backend.features import physics_features
        import inspect
        source = inspect.getsource(physics_features)
        # The old code had `center=True` — verify it was changed to `center=False`
        assert 'center=True' not in source, "Found leakage: rolling(center=True) in physics_features"
        # Verify the fix is present
        assert 'center=False' in source

    def test_auto_validator_imports(self):
        from backend.validation.auto_validator import AutoValidator, validate_pending_predictions
        assert AutoValidator is not None
        assert validate_pending_predictions is not None

    def test_dataset_statistics_exists(self, tmp_path):
        from backend.pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        # Create small test data to avoid loading large parquet
        idx = pd.date_range("2024-07-01", periods=100, freq="min")
        dummy = pd.DataFrame({"soft_flux": np.random.rand(100),
                                "hard_flux": np.random.rand(100)}, index=idx)
        pipeline.train_df = dummy
        pipeline.val_df = dummy.copy()
        pipeline.test_df = dummy.copy()
        pipeline.catalog = pd.DataFrame(columns=["peak_time", "start_time", "end_time", "flare_class"])
        stats = pipeline.compute_dataset_statistics()
        assert "total_samples" in stats
        assert "feature_count" in stats

    def test_calibrator_save_load(self, tmp_path):
        from backend.evaluation.calibration import Calibrator
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 500)
        y_prob = np.clip(y_true + np.random.randn(500) * 0.3, 0.01, 0.99)
        c = Calibrator()
        c.fit(y_true, y_prob)
        save_path = tmp_path / "calibrator_test.pkl"
        c.save(str(save_path))
        assert save_path.exists()

        c2 = Calibrator()
        c2.load(str(save_path))
        assert c2.chosen_method == c.chosen_method
