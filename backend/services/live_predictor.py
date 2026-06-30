"""
Live Predictor
===============
Singleton service that:
  - Loads trained XGBoost (binary + multi-class) models
  - Loads calibrator + optimal threshold from disk
  - Reads latest parquet data every 30s
  - Returns versioned, calibrated, multi-class predictions
  - Computes actual lead time from nearest catalog event
  - Persists to PredictionStore (SQLite)
  - Auto-starts GOES validation thread
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from loguru import logger
import threading
import time as time_module

from backend.data.dataset_builder import DatasetBuilder
from backend.features.feature_pipeline import FeaturePipeline
from backend.models.nowcaster.threshold_detector import ThresholdNowcaster
from backend.services.prediction_store import get_store
from backend.services.versioning import PredictionVersion
from backend.evaluation.calibration import Calibrator
from backend.validation.auto_validator import get_auto_validator

FORECAST_HORIZON_MINUTES = 30
CLASS_ORDER = {"N": 0, "A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
CLASS_LABELS = ["N", "A", "B", "C", "M", "X"]


class LivePredictor:
    def __init__(self, model_tag: str = "benchmark",
                 data_path: str = "dataset/processed/merged_timeseries.parquet",
                 catalog_path: str = "dataset/catalogs/nowcast_catalog.csv",
                 lookback: int = 120):
        self.data_path = Path(data_path)
        self.catalog_path = Path(catalog_path)
        self.lookback = lookback
        self.feature_pipeline = FeaturePipeline()
        self.nowcaster = ThresholdNowcaster()
        self.store = get_store()

        self.model = None
        self.model_name = None
        self.model_tag = model_tag
        self._load_model(model_tag)
        self._load_multiclass_model()
        self._load_calibrator()
        self._load_threshold()

        self._catalog_df = None
        self._load_catalog()

        self._prediction_cache = self._empty_cache("Model not loaded")
        self._latest_df = None
        self._lock = threading.Lock()
        self._refresh_thread = None
        self._running = False

        get_auto_validator().start()

    def _empty_cache(self, error: str = "") -> dict:
        return {
            "prediction_id": "",
            "timestamp": None,
            "flare_probability": 0.0,
            "uncertainty": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "predicted_class": "A",
            "class_probs": {c: 0.0 for c in CLASS_LABELS if c != "N"},
            "lead_time_minutes": 0.0,
            "alert_level": "GREEN",
            "model": self.model_name or "unknown",
            "error": error,
            "similar_events": [],
            "physics_reason": "",
            "data_timestamp": "",
            "threshold": 0.5,
            "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
            "validation_status": "pending",
            "dataset_version": "",
            "feature_version": "",
            "config_version": "",
            "git_commit": "",
            "model_tag": "",
        }

    def _load_model(self, tag: str = "full"):
        xgb_path = Path(f"models/checkpoints/xgboost/xgboost_{tag}.pkl")
        rf_path = Path(f"models/checkpoints/random_forest/random_forest_{tag}.pkl")
        feat_path = Path(f"models/checkpoints/xgboost/xgboost_features.txt")

        if feat_path.exists():
            with open(feat_path) as f:
                self._feature_cols = [line.strip() for line in f if line.strip()]
            logger.info(f"LivePredictor: loaded {len(self._feature_cols)} feature names")
        else:
            self._feature_cols = None

        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_name = "xgboost"
            logger.info(f"LivePredictor: loaded XGBoost from {xgb_path}")
        elif rf_path.exists():
            with open(rf_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_name = "random_forest"
            logger.info(f"LivePredictor: loaded RandomForest from {rf_path}")
        else:
            logger.warning("LivePredictor: no trained binary model found")

    def _load_multiclass_model(self):
        """Load multi-class XGBoost for A/B/C/M/X classification."""
        mc_path = Path("models/checkpoints/xgboost/xgboost_multiclass.pkl")
        info_path = Path("models/checkpoints/xgboost/xgboost_multiclass_info.json")
        self.multiclass_model = None
        self.multiclass_labels = CLASS_LABELS
        if mc_path.exists():
            with open(mc_path, "rb") as f:
                self.multiclass_model = pickle.load(f)
            logger.info(f"LivePredictor: loaded multi-class model from {mc_path}")
            if info_path.exists():
                with open(info_path) as f:
                    self.multiclass_info = json.load(f)
        else:
            logger.warning("LivePredictor: no multi-class model found")

    def _load_calibrator(self):
        self.calibrator = Calibrator()
        self.calibrator.load("models/checkpoints/calibrator.pkl")

    def _load_threshold(self):
        thresh_path = Path("models/checkpoints/optimal_threshold.txt")
        if thresh_path.exists():
            try:
                self._threshold = float(thresh_path.read_text().strip())
                logger.info(f"LivePredictor: loaded threshold={self._threshold}")
            except (ValueError, OSError):
                self._threshold = 0.5
                logger.warning("LivePredictor: invalid threshold file, using 0.5")
        else:
            self._threshold = 0.5
            logger.info("LivePredictor: no threshold file, using default 0.5")

    def _load_catalog(self):
        if self.catalog_path.exists():
            self._catalog_df = pd.read_csv(self.catalog_path)
            for col in ["peak_time", "start_time", "end_time"]:
                if col in self._catalog_df.columns:
                    self._catalog_df[col] = pd.to_datetime(self._catalog_df[col])
            logger.info(f"LivePredictor: loaded catalog with {len(self._catalog_df)} events")
        else:
            self._catalog_df = pd.DataFrame()

    def _compute_lead_time(self, prediction_time: pd.Timestamp) -> float:
        """Find the nearest future flare event and compute lead time in minutes."""
        if self._catalog_df is None or self._catalog_df.empty:
            return 0.0
        future = self._catalog_df[self._catalog_df["peak_time"] > prediction_time]
        if future.empty:
            return 0.0
        nearest = future.iloc[0]
        lead = (nearest["peak_time"] - prediction_time).total_seconds() / 60.0
        return max(0.0, lead)

    def _find_similar_events(self, prob: float, pred_class: str, n: int = 3) -> list[dict]:
        if self._catalog_df is None or self._catalog_df.empty:
            return []
        similar = []
        pc = CLASS_ORDER.get(pred_class, 3)
        for _, ev in self._catalog_df.iterrows():
            ec = CLASS_ORDER.get(str(ev.get("flare_class", "C")).upper(), 3)
            class_sim = max(0, 1 - abs(pc - ec) / 5)
            flux_sim = 1.0
            try:
                hf = float(ev.get("peak_hard_flux", 0))
                flux_sim = min(1.0, hf / max(hf, 100))
            except (ValueError, TypeError):
                pass
            sim = round(0.6 * class_sim + 0.4 * flux_sim, 3)
            ev_lead = 0.0
            try:
                start = pd.to_datetime(ev.get("start_time"))
                peak_t = pd.to_datetime(ev.get("peak_time"))
                ev_lead = (peak_t - start).total_seconds() / 60.0
            except Exception:
                pass
            similar.append({
                "date": str(ev.get("peak_time", ""))[:10],
                "flare_class": f"{ev.get('flare_class', '')}{ev.get('flare_subclass', 0)}",
                "similarity": sim,
                "lead_time": round(ev_lead, 1),
                "description": self._generate_event_description(ev),
            })
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:n]

    def _generate_event_description(self, ev: pd.Series) -> str:
        cls = str(ev.get("flare_class", "C"))
        hf = float(ev.get("peak_hard_flux", 0))
        descs = {
            "X": "Impulsive HXR spike — high energy release",
            "M": "Neupert delay signature — gradual rise",
            "C": "Gradual thermal rise — confined event",
            "B": "Minor brightening — background enhancement",
        }
        base = descs.get(cls, "Detected event in hard X-rays")
        return f"{base} (peak hard flux: {hf:.1f})"

    def _physics_reason(self, df: pd.DataFrame, prob: float) -> str:
        reasons = []
        if df is not None and len(df) >= 10:
            recent = df.tail(10)
            try:
                hf = recent["hard_flux"].values
                hf_change = (hf[-1] - hf[0]) / max(abs(hf[0]), 1e-10)
                if hf_change > 0.05:
                    reasons.append("Hard X-ray flux increasing")
                elif hf_change < -0.05:
                    reasons.append("Hard X-ray flux decreasing")
                else:
                    reasons.append("Hard X-ray flux stable")
            except (KeyError, IndexError):
                pass
            try:
                hr = recent["hardness_ratio"].values
                hr_change = hr[-1] - hr[0]
                if hr_change > 0.02:
                    reasons.append("Hardness ratio rising — spectral hardening")
                elif hr_change < -0.02:
                    reasons.append("Hardness ratio falling — spectral softening")
            except (KeyError, IndexError):
                pass
        if prob > 0.7:
            reasons.append("High flare probability — precursor pattern detected")
        elif prob > 0.4:
            reasons.append("Moderate probability — monitoring for escalation")
        else:
            reasons.append("Background activity — no significant precursors")
        if self._catalog_df is not None and not self._catalog_df.empty:
            cls_count = len(self._catalog_df)
            reasons.append(f"Training catalog contains {cls_count} historical events")
        return " • ".join(reasons) if reasons else "No physics indicators available"

    def refresh_data(self):
        try:
            if self.data_path.exists():
                df = pd.read_parquet(self.data_path)
                with self._lock:
                    self._latest_df = df
                logger.info(f"LivePredictor: loaded {len(df)} rows")
            else:
                logger.warning(f"LivePredictor: data not found at {self.data_path}")
        except Exception as e:
            logger.error(f"LivePredictor: data refresh failed: {e}")

    def predict(self) -> dict:
        if self.model is None and self.multiclass_model is None:
            return self._prediction_cache

        t0 = time_module.perf_counter()

        self.refresh_data()
        df = self._latest_df
        if df is None or len(df) < self.lookback:
            return self._prediction_cache

        with self._lock:
            try:
                cols = self._feature_cols if self._feature_cols else self._get_feature_columns(df)
                available = [c for c in cols if c in df.columns]
                if not available:
                    raise ValueError(f"No matching features found. Need any of {len(cols)} cols")

                sample = df[available].fillna(0).values.astype(np.float32)[-1:]
                pred_time = df.index[-1] if hasattr(df, "index") and len(df) > 0 else pd.Timestamp.now(tz="UTC")

                # ── Primary: multi-class model ──
                if self.multiclass_model is not None:
                    mc_probs = self.multiclass_model.predict_proba(sample)[0]
                    prob = float(np.clip(1.0 - mc_probs[0], 0.001, 0.999))
                    class_probs = {}
                    for i, label in enumerate(self.multiclass_labels):
                        if label != "N":
                            class_probs[label] = max(float(mc_probs[i]), 0.001)
                    total_flare = sum(class_probs.values())
                    if total_flare > 0:
                        class_probs = {k: v / total_flare for k, v in class_probs.items()}
                else:
                    # Fallback: binary model + heuristic classes
                    probs_arr = self.model.predict_proba(sample)
                    raw_prob = float(probs_arr[0, 1]) if probs_arr.shape[1] > 1 else float(probs_arr[0, 0])
                    raw_prob = np.clip(raw_prob, 0.001, 0.999)
                    cal_prob = self.calibrator.predict_proba(np.array([raw_prob]))[0]
                    prob = float(np.clip(cal_prob, 0.001, 0.999))
                    class_probs = {
                        "A": max(0.01, 0.6 * (1 - prob)),
                        "B": max(0.01, 0.25 * (1 - prob)),
                        "C": max(0.01, 0.1 * (1 - prob) + 0.3 * prob),
                        "M": max(0.01, 0.04 * (1 - prob) + 0.45 * prob),
                        "X": max(0.01, 0.01 * (1 - prob) + 0.25 * prob),
                    }
                    total = sum(class_probs.values())
                    class_probs = {k: v / total for k, v in class_probs.items()}

                threshold = self._threshold
                pred_class = max(class_probs, key=class_probs.get)

                # ── Lead time from catalog ──
                lead = self._compute_lead_time(pred_time)
                if lead <= 0:
                    lead = max(0.0, 15.0 * prob)

                # ── Uncertainty (inverse to probability) ──
                unc = max(0.02, 0.15 * (1 - prob))
                lo = max(0.0, prob - unc)
                hi = min(1.0, prob + unc)

                # ── Alert level ──
                if prob > 0.7:
                    alert = "RED"
                elif prob > 0.5:
                    alert = "ORANGE"
                elif prob > 0.2:
                    alert = "YELLOW"
                else:
                    alert = "GREEN"

                similar = self._find_similar_events(prob, pred_class)
                physics = self._physics_reason(df, prob)
                data_ts = str(df.index[-1]) if hasattr(df, "index") and len(df) > 0 else ""

                version = PredictionVersion(
                    model_name=self.model_name or "unknown",
                    model_tag=self.model_tag,
                    threshold=threshold,
                )

                self._prediction_cache = {
                    "prediction_id": version.to_id(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "flare_probability": round(prob, 4),
                    "uncertainty": round(unc, 4),
                    "lower_bound": round(lo, 4),
                    "upper_bound": round(hi, 4),
                    "predicted_class": pred_class,
                    "class_probs": {k: round(v, 4) for k, v in class_probs.items()},
                    "lead_time_minutes": round(lead, 1),
                    "alert_level": alert,
                    "model": f"{self.model_name}_multiclass" if self.multiclass_model else self.model_name,
                    "data_timestamp": data_ts,
                    "similar_events": similar,
                    "physics_reason": physics,
                    "threshold": threshold,
                    "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
                    "validation_status": "pending",
                    **version.to_dict(),
                }
                logger.info(
                    f"LivePredictor: prob={prob:.3f}, class={pred_class}, "
                    f"lead={lead:.1f}min, threshold={threshold:.2f}, alert={alert}"
                )

                elapsed = (time_module.perf_counter() - t0) * 1000
                self.store.save_prediction(self._prediction_cache, execution_time_ms=elapsed)

            except Exception as e:
                logger.error(f"LivePredictor: prediction failed: {e}")
                self._prediction_cache["error"] = str(e)

        return self._prediction_cache

    def _get_feature_columns(self, df: pd.DataFrame) -> tuple:
        soft_cols = [c for c in df.columns if c.startswith("soft_")
                     and c in df.columns and df[c].dtype.kind == "f"]
        hard_cols = [c for c in df.columns if c.startswith("hard_")
                     and c in df.columns and df[c].dtype.kind == "f"]
        cross_cols = [c for c in df.columns if c not in soft_cols + hard_cols
                      and df[c].dtype.kind == "f"
                      and c not in ("quality_solexs", "quality_hel1os", "data_gap", "artifact_flag")]
        return soft_cols, hard_cols, cross_cols

    def get_catalog(self) -> pd.DataFrame:
        return self._catalog_df

    def start_auto_refresh(self, interval_seconds: int = 30):
        self._running = True

        def _loop():
            self.predict()
            while self._running:
                time_module.sleep(interval_seconds)
                try:
                    self.predict()
                except Exception as e:
                    logger.error(f"Auto-refresh error: {e}")

        self._refresh_thread = threading.Thread(target=_loop, daemon=True)
        self._refresh_thread.start()
        logger.info(f"LivePredictor: auto-refresh every {interval_seconds}s started")

    def stop(self):
        self._running = False


_predictor_instance = None


def get_predictor() -> LivePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = LivePredictor()
        _predictor_instance.start_auto_refresh(interval_seconds=30)
    return _predictor_instance
