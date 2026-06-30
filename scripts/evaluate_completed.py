import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from backend.training.trainer import SolarFlareDataset, SolarFlareTrainer
from backend.models.forecaster.lstm_forecaster import LSTMFlareModule
from backend.models.forecaster.random_forest_forecaster import RandomForestForecaster
from backend.models.forecaster.xgboost_forecaster import XGBoostForecaster
from backend.evaluation.benchmark import Benchmark
from backend.evaluation.evaluator import Evaluator

def main():
    logger.info("=== Loading Configurations and Data ===")
    with open("configs/training.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("configs/models.yaml") as f:
        models_cfg = yaml.safe_load(f)

    suffix = cfg["data"].get("file_suffix", "")
    processed_dir = Path(cfg["data"]["processed_dir"])
    catalog_path = Path(cfg["data"]["catalog_path"])
    lookback = cfg["data"]["lookback_samples"]
    step = cfg["data"]["window_step"]

    train_df = pd.read_parquet(processed_dir / f"train_timeseries{suffix}.parquet")
    val_df = pd.read_parquet(processed_dir / f"val_timeseries{suffix}.parquet")
    test_df = pd.read_parquet(processed_dir / f"test_timeseries{suffix}.parquet")
    catalog = pd.read_csv(catalog_path)

    # Feature definitions
    sc = [c for c in train_df.columns if c.startswith("soft_")]
    hc = [c for c in train_df.columns if c.startswith("hard_")]
    cross = {
        "soft": ["quality_solexs", "d_soft_flux_dt", "d2_soft_flux_dt2", "d_log_soft_flux_dt"],
        "hard": ["quality_hel1os", "d_hard_flux_dt", "d2_hard_flux_dt2", "d_log_hard_flux_dt",
                 "event_rate_1h", "event_rate_6h", "event_rate_24h", "flux_above_2sigma"],
        "both": ["hardness_ratio", "hardness_ratio_log", "hardness_ratio_deriv",
                 "hardness_ratio_5min", "hardness_ratio_15min", "spectral_index",
                 "spectral_index_5min", "spectral_index_deriv", "hxr_sxr_corr_15min",
                 "hxr_sxr_lag1_corr", "neupert_proxy", "neupert_residual",
                 "neupert_residual_abs", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
                 "data_gap", "artifact_flag"]
    }
    sc.extend(cross["soft"])
    sc.extend(cross["both"])
    hc.extend(cross["hard"])
    hc.extend(cross["both"])
    sc = [c for c in sc if c in train_df.columns]
    hc = [c for c in hc if c in train_df.columns]

    # Build sequence dataset for LSTM
    Xs_test = test_df[sc].values.astype(np.float32)
    Xh_test = test_df[hc].values.astype(np.float32)
    flat_test = np.concatenate([Xs_test, Xh_test], axis=1)

    horizon_min = cfg["data"].get("forecast_horizons_minutes", [5])[0]
    horizon_sec = horizon_min * 60
    class_map = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}

    def build_labels(df):
        labels = np.zeros(len(df), dtype=np.float32)
        lead_times = np.zeros(len(df), dtype=np.float32)
        flare_class = np.full(len(df), -1, dtype=np.int32)
        t0, t1 = df.index[0], df.index[-1]
        cat = catalog.copy()
        cat["peak_time"] = pd.to_datetime(cat["peak_time"])
        cat = cat[(cat["peak_time"] >= t0 - pd.Timedelta(seconds=horizon_sec)) & (cat["peak_time"] <= t1)]
        if len(cat) == 0:
            return {"flare_label": labels, "lead_time": lead_times, "flare_class": flare_class}
        peak_ns = cat["peak_time"].values.astype(np.int64)
        t_ns = df.index.values.astype(np.int64)
        horizon_ns = horizon_sec * 1_000_000_000
        left = np.searchsorted(t_ns, peak_ns - horizon_ns, side="left")
        right = np.searchsorted(t_ns, peak_ns, side="right")
        for j in range(len(cat)):
            lo, hi = left[j], right[j]
            if lo < hi:
                labels[lo:hi] = 1.0
                cls_str = str(cat.iloc[j].get("flare_class", "C"))[0].upper()
                flare_class[lo:hi] = class_map.get(cls_str, 2)
                lt_slice = lead_times[lo:hi]
                new_mask = (lt_slice == 0).nonzero()[0]
                if len(new_mask) > 0:
                    idxs = lo + new_mask
                    lead_times[idxs] = np.maximum((peak_ns[j] - t_ns[idxs]) / 1_000_000_000, 0)
        return {"flare_label": labels, "lead_time": lead_times, "flare_class": flare_class}

    test_labels = build_labels(test_df)
    ds_test = SolarFlareDataset(Xs_test, Xh_test, test_labels, lookback, step)
    
    test_idx = np.array(ds_test.indices)
    y_test_sw = test_labels["flare_label"][test_idx]
    lt_test_sw = test_labels["lead_time"][test_idx]
    y_test_full = test_labels["flare_label"]
    lt_test_full = test_labels["lead_time"]

    benchmark = Benchmark()
    models_info = {}

    # 1. Evaluate Random Forest
    logger.info("=== Loading and Evaluating Random Forest ===")
    rf = RandomForestForecaster()
    rf.load("rf_benchmark")
    rf_probs = rf.predict_proba(flat_test)
    benchmark.add_model("RF", y_test_full, rf_probs, lt_test_full)
    models_info["RF"] = {"path": "models/checkpoints/random_forest", "probs": rf_probs}

    # 2. Evaluate XGBoost
    logger.info("=== Loading and Evaluating XGBoost ===")
    xgb = XGBoostForecaster()
    xgb.load("xgb_benchmark")
    xgb_probs = xgb.predict_proba(flat_test)
    benchmark.add_model("XGB", y_test_full, xgb_probs, lt_test_full)
    models_info["XGB"] = {"path": "models/checkpoints/xgboost", "probs": xgb_probs}

    # 3. Evaluate LSTM
    logger.info("=== Loading and Evaluating LSTM ===")
    lstm_cfg = models_cfg.get("forecaster", {}).get("lstm", {})
    lstm = LSTMFlareModule(
        input_dim=Xs_test.shape[1] + Xh_test.shape[1],
        hidden_dim=lstm_cfg.get("hidden_size", 64),
        num_layers=lstm_cfg.get("num_layers", 1),
        bidirectional=lstm_cfg.get("bidirectional", False),
        dropout=lstm_cfg.get("dropout", 0.2),
        learning_rate=lstm_cfg.get("learning_rate", 0.001)
    )
    lstm.load_state_dict(torch.load("models/checkpoints/lstm_benchmark/best.ckpt", map_location="cpu"))
    
    # Run LSTM inference
    from torch.utils.data import DataLoader
    lstm.eval()
    lstm.to("cuda")
    loader = DataLoader(ds_test, batch_size=256, shuffle=False)
    lstm_probs = []
    with torch.no_grad():
        for batch in loader:
            xs, xh, _ = batch
            xs, xh = xs.to("cuda"), xh.to("cuda")
            out = lstm(xs, xh)
            if isinstance(out, dict):
                out = out.get("flare_prob", out.get("flare_logits", list(out.values())[0]))
            if out.dim() > 1 and out.shape[-1] == 1:
                out = out.squeeze(-1)
            lstm_probs.append(torch.sigmoid(out).cpu().numpy())
    lstm_probs = np.concatenate(lstm_probs).ravel()
    
    benchmark.add_model("LSTM", y_test_sw, lstm_probs, lt_test_sw)
    models_info["LSTM"] = {"path": "models/checkpoints/lstm_benchmark", "probs": lstm_probs}

    # Generate summary
    summary = benchmark.summarize()
    logger.info("\n{}", summary.to_string())

    summary_path = Path("models/checkpoints/metrics/benchmark_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Benchmark summary saved: {}", summary_path)

    winner = benchmark.select_winner(summary)
    
    # Select optimal threshold
    winner_probs = models_info[winner]["probs"]
    winner_yt = y_test_sw if winner == "LSTM" else y_test_full
    
    evaluator = Evaluator()
    result = evaluator.evaluate(winner_yt, winner_probs, model_name="threshold_search")
    best_thresh = result.thresholds[0] if result.thresholds else 0.5
    
    thresh_path = Path("models/checkpoints/optimal_threshold.txt")
    thresh_path.write_text(str(best_thresh))
    logger.info("Saved optimal threshold: {} to {}", best_thresh, thresh_path)

    # Deployed model symlink simulation
    try:
        winner_link = Path("models/checkpoints/deployed")
        if winner_link.exists():
            winner_link.unlink()
        # Create a simple deployed text config indicating deployed winner
        deployed_meta = {"deployed_model": winner, "model_path": models_info[winner]["path"]}
        with open("models/checkpoints/deployed_meta.json", "w") as fm:
            import json
            json.dump(deployed_meta, fm, indent=2)
        logger.info("Created deployment metadata pointing to {}", winner)
    except Exception as e:
        logger.warning("Symlink / deployment meta error: {}", e)

    logger.info("=== Evaluation and Selection Complete ===")

if __name__ == "__main__":
    main()
