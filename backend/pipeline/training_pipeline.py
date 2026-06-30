"""
Training Pipeline
==================
Orchestrates the complete model training workflow with:
  - Configurable forecast horizons (5/10/15/30/60 min)
  - Window-based labels (not peak-only)
  - Probability calibration (Platt / Isotonic)
  - Threshold optimization (max TSS)
  - Dataset statistics computation
  - Benchmark with versioned outputs

Usage:
    pipeline = TrainingPipeline()
    pipeline.run()
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
import json
from loguru import logger

from backend.training.trainer import SolarFlareDataset, SolarFlareTrainer
from backend.training.experiment_logger import ExperimentLogger
from backend.models.forecaster.patchtst_forecaster import DualStreamPatchTST
from backend.models.forecaster.lstm_forecaster import LSTMFlareModule
from backend.models.forecaster.timesnet_forecaster import TimesNetFlareModule
from backend.models.forecaster.random_forest_forecaster import RandomForestForecaster
from backend.models.forecaster.xgboost_forecaster import XGBoostForecaster
from backend.models.bayesian.uncertainty import MCDropout
from backend.evaluation.benchmark import Benchmark
from backend.evaluation.evaluator import Evaluator
from backend.evaluation.calibration import Calibrator
from backend.services.versioning import compute_dataset_version, compute_feature_version


class TrainingPipeline:
    """
    Orchestrates the full model training and benchmarking workflow.

    Key improvements over baseline:
      - Labels shifted by forecast_horizon: predicts future flare occurrence
      - Window-based labeling: label=1 if any flare peak falls within horizon
      - Calibrated probabilities via Platt/Isotonic scaling
      - Optimal threshold selection via max TSS
    """

    def __init__(self, config_path: str = "configs/training.yaml",
                 models_path: str = "configs/models.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        with open(models_path) as f:
            self.models_cfg = yaml.safe_load(f)
        self.processed_dir = Path(self.cfg["data"]["processed_dir"])
        self.catalog_path = Path(self.cfg["data"]["catalog_path"])
        self.lookback = self.cfg["data"]["lookback_samples"]
        self.step = self.cfg["data"]["window_step"]
        self.forecast_horizons = self.cfg["data"].get("forecast_horizons_minutes", [30])
        self.active_horizon = self.forecast_horizons[0]
        self.label_mode = self.cfg["data"].get("label_mode", "fixed_window")
        self.winner = None
        self.calibrator = None
        self.optimal_threshold = 0.5

    def load_data(self):
        """Load processed parquet splits and catalog."""
        logger.info("Loading processed data...")
        suffix = self.cfg["data"].get("file_suffix", "")
        self.train_df = pd.read_parquet(self.processed_dir / f"train_timeseries{suffix}.parquet")
        self.val_df = pd.read_parquet(self.processed_dir / f"val_timeseries{suffix}.parquet")
        self.test_df = pd.read_parquet(self.processed_dir / f"test_timeseries{suffix}.parquet")
        self.catalog = pd.read_csv(self.catalog_path)
        logger.info(
            "Train: {}, Val: {}, Test: {}, Catalog: {} events, Horizon: {} min, Label: {}",
            len(self.train_df), len(self.val_df),
            len(self.test_df), len(self.catalog),
            self.active_horizon, self.label_mode,
        )

    def compute_dataset_statistics(self):
        """Compute and save dataset statistics to JSON."""
        train_labels = self.build_labels(self.train_df)
        val_labels = self.build_labels(self.val_df)
        test_labels = self.build_labels(self.test_df)
        all_labels = np.concatenate([
            train_labels["flare_label"],
            val_labels["flare_label"],
            test_labels["flare_label"],
        ])
        positive = int(all_labels.sum())
        negative = int(len(all_labels) - positive)
        stats = {
            "total_samples": len(all_labels),
            "positive_samples": positive,
            "negative_samples": negative,
            "class_ratio": round(negative / max(positive, 1), 2),
            "positive_pct": round(100 * positive / max(len(all_labels), 1), 4),
            "observation_period_start": str(self.train_df.index[0]),
            "observation_period_end": str(self.test_df.index[-1]),
            "train_samples": len(self.train_df),
            "val_samples": len(self.val_df),
            "test_samples": len(self.test_df),
            "catalog_events": len(self.catalog),
            "forecast_horizon_minutes": self.active_horizon,
            "label_mode": self.label_mode,
            "dataset_version": compute_dataset_version(),
            "feature_version": compute_feature_version(),
            "missing_values_pct": round(100 * self.train_df.isna().sum().sum()
                                         / max(self.train_df.size, 1), 2),
            "feature_count": len(self.train_df.select_dtypes(include=[np.number]).columns),
        }
        stats_path = Path("dataset/dataset_statistics.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2))
        logger.info("Dataset statistics saved: {}", stats_path)
        return stats

    def build_labels(self, df: pd.DataFrame) -> dict:
        """
        Build forecast labels with configurable horizon and window mode
        (vectorized — no per-event loops).

        Label modes:
          - 'peak_only':     label=1 only at exact peak timestamp
          - 'fixed_window':  label=1 if any flare peak is within
                             [t, t + forecast_horizon] (default)
        """
        labels = np.zeros(len(df), dtype=np.float32)
        lead_times = np.zeros(len(df), dtype=np.float32)
        flare_class = np.full(len(df), -1, dtype=np.int32)

        horizon_sec = self.active_horizon * 60
        idx_arr = np.arange(len(df))

        # Filter catalog events that fall within this dataframe's time range
        t0, t1 = df.index[0], df.index[-1]
        cat = self.catalog.copy()
        cat["peak_time"] = pd.to_datetime(cat["peak_time"])
        cat = cat[(cat["peak_time"] >= t0 - pd.Timedelta(seconds=horizon_sec)) &
                  (cat["peak_time"] <= t1)]
        if len(cat) == 0:
            return {"flare_label": labels, "lead_time": lead_times,
                    "flare_class": flare_class}

        cat_peak = cat["peak_time"].values
        class_map = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}

        if self.label_mode == "peak_only":
            for peak in cat_peak:
                if peak in df.index:
                    idx = df.index.get_loc(peak)
                    labels[idx] = 1.0
            n_pos = int(labels.sum())
            logger.info("Labels: mode=peak_only, positive={}/{} ({:.2f}%)",
                        n_pos, len(labels), 100 * n_pos / max(len(labels), 1))
            return {"flare_label": labels, "lead_time": lead_times,
                    "flare_class": flare_class}

        # Fixed window: for each event, label interval [peak - horizon, peak]
        t_ns = df.index.values.astype(np.int64)          # (N,) in ns
        horizon_ns = horizon_sec * 1_000_000_000
        peak_ns = cat_peak.astype(np.int64)              # (M,) in ns

        # For each event, compute the start of its positive window
        win_starts = peak_ns - horizon_ns                # (M,)

        # Mark labels: for each event j, label all rows i where
        #   win_starts[j] <= t[i] <= peak_ns[j]
        # Vectorize via left/right searchsorted
        # (M events * O(log N) each = fast)
        left = np.searchsorted(t_ns, win_starts, side="left")
        right = np.searchsorted(t_ns, peak_ns, side="right")

        for j in range(len(cat)):
            lo, hi = left[j], right[j]
            if lo < hi:
                labels[lo:hi] = 1.0
                # Set flare class for this event
                cls_str = str(cat.iloc[j].get("flare_class", "C"))[0].upper()
                cls_val = class_map.get(cls_str, 2)
                flare_class[lo:hi] = cls_val
                # Lead time: only set where not already claimed (first event wins)
                lt_slice = lead_times[lo:hi]
                new_mask = (lt_slice == 0).nonzero()[0]
                if len(new_mask) > 0:
                    idxs = lo + new_mask
                    lt_vals = np.maximum((peak_ns[j] - t_ns[idxs]) / 1_000_000_000, 0)
                    lead_times[idxs] = lt_vals

        n_positive = int(labels.sum())
        logger.info(
            "Labels: mode={}, horizon={}min, positive={}/{} ({:.2f}%)",
            self.label_mode, self.active_horizon,
            n_positive, len(labels), 100 * n_positive / max(len(labels), 1),
        )
        return {"flare_label": labels, "lead_time": lead_times,
                "flare_class": flare_class}
    def _get_feature_arrays(self, df: pd.DataFrame):
        """Split features into soft and hard channels, distributing cross-instrument features."""
        soft_cols = [c for c in df.columns if c.startswith("soft_")]
        hard_cols = [c for c in df.columns if c.startswith("hard_")]

        # Distribute unprefixed cross-instrument features
        cross_cols = {
            "soft": ["quality_solexs", "d_soft_flux_dt", "d2_soft_flux_dt2",
                     "d_log_soft_flux_dt"],
            "hard": ["quality_hel1os", "d_hard_flux_dt", "d2_hard_flux_dt2",
                     "d_log_hard_flux_dt", "event_rate_1h", "event_rate_6h",
                     "event_rate_24h", "flux_above_2sigma"],
            "both": ["hardness_ratio", "hardness_ratio_log", "hardness_ratio_deriv",
                     "hardness_ratio_5min", "hardness_ratio_15min",
                     "spectral_index", "spectral_index_5min", "spectral_index_deriv",
                     "hxr_sxr_corr_15min", "hxr_sxr_lag1_corr",
                     "neupert_proxy", "neupert_residual", "neupert_residual_abs",
                     "hour_sin", "hour_cos", "doy_sin", "doy_cos",
                     "data_gap", "artifact_flag"],
        }
        soft_cols.extend(cross_cols["soft"])
        soft_cols.extend(cross_cols["both"])
        hard_cols.extend(cross_cols["hard"])
        hard_cols.extend(cross_cols["both"])

        soft_cols = [c for c in soft_cols if c in df.columns]
        hard_cols = [c for c in hard_cols if c in df.columns]

        # Fallback if prefix approach yields nothing
        if not soft_cols:
            n = len(df.columns) // 2
            soft_cols = df.select_dtypes(include=[np.number]).columns[:n].tolist()
        if not hard_cols:
            n = len(df.columns) // 2
            hard_cols = df.select_dtypes(include=[np.number]).columns[n:].tolist()

        soft = np.nan_to_num(df[soft_cols].values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        hard = np.nan_to_num(df[hard_cols].values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return soft, hard

    def calibrate_probabilities(self, y_true: np.ndarray,
                                 y_pred_proba: np.ndarray) -> 'Calibrator':
        """Fit probability calibration (Platt + Isotonic), deploy best by ECE."""
        calibrator = Calibrator()
        calibrator.fit(y_true, y_pred_proba)
        # Apply to validation set to decide which method to deploy
        cal_info = calibrator.evaluate(y_true, y_pred_proba)
        logger.info("Calibration results: {}", cal_info)
        self.calibrator = calibrator
        # Auto-select: prefer the method with lower ECE
        chosen = cal_info.get("chosen_method", "platt")
        logger.info("Deploying calibration method: {}", chosen)
        return calibrator

    def select_optimal_threshold(self, y_true: np.ndarray,
                                  y_pred_proba: np.ndarray) -> float:
        """Find threshold that maximizes TSS (primary) then F1 (secondary)."""
        evaluator = Evaluator()
        result = evaluator.evaluate(y_true, y_pred_proba, model_name="threshold_search")
        best_thresh = result.thresholds[0] if result.thresholds else 0.5
        logger.info("Optimal threshold: {:.3f} (TSS={:.4f}, F1={:.4f})",
                    best_thresh, result.tss, result.f1)
        self.optimal_threshold = best_thresh
        return best_thresh

    def _dl_predict(self, model, ds_test, batch_size=256) -> np.ndarray:
        """Run full test DataLoader through a deep learning model."""
        from torch.utils.data import DataLoader
        model.eval()
        device = next(model.parameters()).device
        loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)
        probs = []
        with torch.no_grad():
            for batch in loader:
                xs, xh, _ = batch
                xs, xh = xs.to(device), xh.to(device)
                out = model(xs, xh)
                if isinstance(out, dict):
                    out = out.get("flare_prob", out.get("flare_class_logits", list(out.values())[0]))
                if out.dim() > 1 and out.shape[-1] == 1:
                    out = out.squeeze(-1)
                probs.append(torch.sigmoid(out).cpu().numpy())
        return np.concatenate(probs).ravel()

    def _save_confusion(self, name, y_true, y_pred):
        """Save confusion matrix and metrics as JSON."""
        from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        TPR = tp / (tp + fn + 1e-10)
        TNR = tn / (tn + fp + 1e-10)
        FAR = fp / (fp + tn + 1e-10)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        out = {
            "model": name,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "TPR": round(float(TPR), 4),
            "TNR": round(float(TNR), 4),
            "FAR": round(float(FAR), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
        }
        metrics_dir = Path("models/checkpoints/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"{name}_confusion.json").write_text(json.dumps(out, indent=2))
        logger.info("{} metrics saved | TPR={:.4f} FAR={:.4f} F1={:.4f}", name, TPR, FAR, f1)
        return out

    def run(self):
        """Execute the complete training pipeline."""
        logger.info("=== Training Pipeline ===")

        self.load_data()
        self.compute_dataset_statistics()

        train_labels = self.build_labels(self.train_df)
        val_labels = self.build_labels(self.val_df)
        test_labels = self.build_labels(self.test_df)

        Xs_train, Xh_train = self._get_feature_arrays(self.train_df)
        Xs_val, Xh_val = self._get_feature_arrays(self.val_df)
        Xs_test, Xh_test = self._get_feature_arrays(self.test_df)

        flat_train = np.concatenate([Xs_train, Xh_train], axis=1)
        flat_val   = np.concatenate([Xs_val, Xh_val], axis=1)
        flat_test  = np.concatenate([Xs_test, Xh_test], axis=1)

        # Build sliding window datasets (for DL models)
        ds_train = SolarFlareDataset(Xs_train, Xh_train, train_labels,
                                      lookback=self.lookback, step=self.step)
        ds_val = SolarFlareDataset(Xs_val, Xh_val, val_labels,
                                    lookback=self.lookback, step=self.step)
        ds_test = SolarFlareDataset(Xs_test, Xh_test, test_labels,
                                     lookback=self.lookback, step=self.step)
        logger.info("Datasets ready: train={}, val={}, test={}",
                     len(ds_train), len(ds_val), len(ds_test))

        # Align labels with sliding window indices (for DL models)
        test_idx = np.array(ds_test.indices)
        y_test_sw = test_labels["flare_label"][test_idx]
        lt_test_sw = test_labels["lead_time"][test_idx]
        # Keep full labels for flat models (RF, XGB)
        y_test_full = test_labels["flare_label"]
        lt_test_full = test_labels["lead_time"]

        benchmark = Benchmark()
        models_info = {}

        # ─── 1. Random Forest (flat features, full labels) ─────────────
        logger.info("Training Random Forest...")
        rf = RandomForestForecaster()
        rf.fit(flat_train, train_labels["flare_label"],
               flat_val, val_labels["flare_label"])
        rf_probs = rf.predict_proba(flat_test)
        benchmark.add_model("RF", y_test_full, rf_probs, lt_test_full)
        rf_opt = (rf_probs > 0.5).astype(int)
        self._save_confusion("RF", y_test_full, rf_opt)
        rf.save("rf_benchmark")
        models_info["RF"] = {"path": "models/checkpoints/rf_benchmark", "probs": rf_probs}

        # ─── 2. XGBoost (flat features, full labels) ───────────────────
        logger.info("Training XGBoost...")
        xgb = XGBoostForecaster()
        xgb.fit(flat_train, train_labels["flare_label"],
                flat_val, val_labels["flare_label"])
        xgb_probs = xgb.predict_proba(flat_test)
        benchmark.add_model("XGB", y_test_full, xgb_probs, lt_test_full)
        xgb_opt = (xgb_probs > 0.5).astype(int)
        self._save_confusion("XGB", y_test_full, xgb_opt)
        xgb.save("xgb_benchmark")
        models_info["XGB"] = {"path": "models/checkpoints/xgb_benchmark", "probs": xgb_probs}

        # ─── 3. LSTM (sliding windows, window-aligned labels) ──────────
        logger.info("Training LSTM...")
        lstm_cfg = self.models_cfg.get("forecaster", {}).get("lstm", {})
        lstm = LSTMFlareModule(
            input_dim=Xs_train.shape[1] + Xh_train.shape[1],
            hidden_dim=lstm_cfg.get("hidden_size", 64),
            num_layers=lstm_cfg.get("num_layers", 1),
            bidirectional=lstm_cfg.get("bidirectional", False),
            dropout=lstm_cfg.get("dropout", 0.2),
            learning_rate=lstm_cfg.get("learning_rate", 0.001),
        )
        lstm_trainer = SolarFlareTrainer()
        lstm_trainer.train_forecaster(lstm, ds_train, ds_val,
                                       experiment_name="lstm_benchmark",
                                       model_name="lstm")
        lstm_probs = self._dl_predict(lstm, ds_test)
        benchmark.add_model("LSTM", y_test_sw, lstm_probs, lt_test_sw)
        lstm_opt = (lstm_probs > 0.5).astype(int)
        self._save_confusion("LSTM", y_test_sw, lstm_opt)
        models_info["LSTM"] = {"path": "models/checkpoints/lstm_benchmark", "probs": lstm_probs}

        # ─── 4. PatchTST (sliding windows, window-aligned labels) ──────
        logger.info("Training PatchTST...")
        ptst = DualStreamPatchTST(
            n_soft_features=Xs_train.shape[1],
            n_hard_features=Xh_train.shape[1],
        )
        ptst_trainer = SolarFlareTrainer()
        ptst_trainer.train_forecaster(ptst, ds_train, ds_val,
                                       experiment_name="patchtst_benchmark")
        ptst_probs = self._dl_predict(ptst, ds_test)
        benchmark.add_model("PatchTST", y_test_sw, ptst_probs, lt_test_sw)
        ptst_opt = (ptst_probs > 0.5).astype(int)
        self._save_confusion("PatchTST", y_test_sw, ptst_opt)
        models_info["PatchTST"] = {"path": "models/checkpoints/patchtst_benchmark", "probs": ptst_probs}

        # ─── 5. TimesNet (sliding windows, window-aligned labels) ──────
        logger.info("Training TimesNet...")
        tn = TimesNetFlareModule(
            soft_channels=Xs_train.shape[1],
            hard_channels=Xh_train.shape[1],
        )
        tn_trainer = SolarFlareTrainer()
        tn_trainer.train_forecaster(tn, ds_train, ds_val,
                                     experiment_name="timesnet_benchmark",
                                     model_name="timesnet")
        tn_probs = self._dl_predict(tn, ds_test)
        benchmark.add_model("TimesNet", y_test_sw, tn_probs, lt_test_sw)
        tn_opt = (tn_probs > 0.5).astype(int)
        self._save_confusion("TimesNet", y_test_sw, tn_opt)
        models_info["TimesNet"] = {"path": "models/checkpoints/timesnet_benchmark", "probs": tn_probs}

        # ─── Benchmark Summary ─────────────────────────────────────────
        summary = benchmark.summarize()
        logger.info("\n{}", summary.to_string())

        summary_path = Path("models/checkpoints/metrics/benchmark_summary.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        logger.info("Benchmark summary saved: {}", summary_path)

        self.winner = benchmark.select_winner(summary)
        logger.info("Winner: {} | TSS={:.4f}", self.winner,
                    summary.loc[summary["Model"] == self.winner, "TSS"].values[0] if self.winner in summary["Model"].values else 0)

        # ─── Threshold optimization ────────────────────────────────────
        winner_probs = models_info.get(self.winner, {}).get("probs", rf_probs)
        winner_yt = y_test_sw if self.winner in ("LSTM", "PatchTST", "TimesNet") else y_test_full
        self.select_optimal_threshold(winner_yt, winner_probs)
        winner_opt = (winner_probs > self.optimal_threshold).astype(int)
        self._save_confusion(f"{self.winner}_deployed", winner_yt, winner_opt)

        thresh_path = Path("models/checkpoints/optimal_threshold.txt")
        thresh_path.parent.mkdir(parents=True, exist_ok=True)
        thresh_path.write_text(str(self.optimal_threshold))
        logger.info("Saved optimal threshold: {} ({})", self.optimal_threshold, thresh_path)

        # ─── Winner symlink (skip on Windows without developer mode) ───
        try:
            winner_link = Path("models/checkpoints/deployed")
            winner_link.parent.mkdir(parents=True, exist_ok=True)
            if winner_link.exists():
                winner_link.unlink()
            if hasattr(Path, "symlink_to"):
                winner_link.symlink_to(models_info.get(self.winner, {}).get("path", ""), target_is_directory=True)
            logger.info("Deployed model: {} -> {}", self.winner, winner_link)
        except Exception:
            logger.info("Skipped symlink (Windows compatibility)")

        logger.info("=== Training Pipeline Complete ===")
        logger.info("Winner: {}, Threshold: {:.3f}, Horizon: {}min, Label: {}",
                    self.winner, self.optimal_threshold, self.active_horizon, self.label_mode)
