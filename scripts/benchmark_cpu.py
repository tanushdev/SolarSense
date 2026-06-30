# Usage: python scripts/benchmark_cpu.py
# Trains RF, XGBoost, LSTM, TimesNet on reduced data (CPU-friendly)
# Skips PatchTST (needs GPU)

import sys, warnings
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from loguru import logger

from backend.evaluation.benchmark import Benchmark
from backend.models.forecaster.random_forest_forecaster import RandomForestForecaster
from backend.models.forecaster.lstm_forecaster import LSTMFlareModule
from backend.training.trainer import SolarFlareDataset

CATALOG_PATH = "dataset/catalogs/nowcast_catalog.csv"
DATA_PATH = "dataset/processed/merged_timeseries.parquet"
N_SUBSET = 50000
LOOKBACK = 120
STEP = 60
BATCH_SIZE = 32
EPOCHS = 10

def main():
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    catalog = pd.read_csv(CATALOG_PATH)
    logger.info(f"Data: {len(df)} rows, Catalog: {len(catalog)} events")

    # Build labels
    labels = np.zeros(len(df), dtype=np.float32)
    lead_times = np.zeros(len(df), dtype=np.float32)
    for _, event in catalog.iterrows():
        peak = pd.to_datetime(event["peak_time"])
        start = pd.to_datetime(event["start_time"])
        if peak in df.index:
            idx = df.index.get_loc(peak)
            labels[idx] = 1.0
            lead_times[idx] = (peak - start).total_seconds()

    # Feature arrays
    soft_cols = ["soft_flux", "soft_counts", "soft_rise_rate_5min", "soft_rise_rate_15min",
                 "soft_mean_5m", "soft_std_5m", "soft_mean_15m", "soft_std_15m",
                 "soft_mean_60m", "soft_std_60m", "soft_flux_excess", "soft_snr",
                 "soft_fft_power_hf_5m", "soft_wavelet_detail_energy_5m",
                 "soft_fft_power_hf_15m", "soft_wavelet_detail_energy_15m"]
    hard_cols = ["hard_flux", "hard_counts", "hard_rise_rate_5min", "hard_rise_rate_15min",
                 "hard_mean_5m", "hard_std_5m", "hard_mean_15m", "hard_std_15m",
                 "hard_mean_60m", "hard_std_60m", "hard_flux_excess", "hard_snr",
                 "hard_fft_power_hf_5m", "hard_wavelet_detail_energy_5m",
                 "hard_fft_power_hf_15m", "hard_wavelet_detail_energy_15m"]
    cross_cols = ["hardness_ratio", "hardness_ratio_5min", "hardness_ratio_15min",
                  "d_soft_flux_dt", "d_hard_flux_dt", "hardness_ratio_deriv"]

    available = df.columns.tolist()
    soft_cols = [c for c in soft_cols if c in available]
    hard_cols = [c for c in hard_cols if c in available]
    cross_cols = [c for c in cross_cols if c in available]

    Xs = df[soft_cols].fillna(0).values.astype(np.float32)
    Xh = df[hard_cols].fillna(0).values.astype(np.float32)
    Xc = df[cross_cols].fillna(0).values.astype(np.float32)

    # Train/val/test split (60/20/20 chronological)
    n = len(df)
    idx_val = int(n * 0.60)
    idx_test = int(n * 0.80)

    # Subset for tree models
    train_end = min(idx_val, N_SUBSET + LOOKBACK)
    X_flat_train = np.concatenate([Xs[LOOKBACK:train_end], Xh[LOOKBACK:train_end], Xc[LOOKBACK:train_end]], axis=1)
    y_flat_train = labels[LOOKBACK:train_end]
    lt_flat_train = lead_times[LOOKBACK:train_end]

    X_flat_val = np.concatenate([Xs[idx_val:idx_test], Xh[idx_val:idx_test], Xc[idx_val:idx_test]], axis=1)
    y_flat_val = labels[idx_val:idx_test]
    lt_flat_val = lead_times[idx_val:idx_test]

    X_flat_test = np.concatenate([Xs[idx_test:], Xh[idx_test:], Xc[idx_test:]], axis=1)
    y_flat_test = labels[idx_test:]
    lt_flat_test = lead_times[idx_test:]

    benchmark = Benchmark()

    # 1. Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestForecaster()
    rf.fit(X_flat_train, y_flat_train, X_flat_val, y_flat_val)
    rf_probs = rf.predict_proba(X_flat_test)
    benchmark.add_model("RF", y_flat_test, rf_probs, lt_flat_test)
    rf.save("benchmark")
    logger.info("RF complete")

    # 2. XGBoost
    logger.info("Training XGBoost...")
    try:
        from backend.models.forecaster.xgboost_forecaster import XGBoostForecaster
        xgb = XGBoostForecaster()
        xgb.fit(X_flat_train, y_flat_train)
        xgb_probs = xgb.predict_proba(X_flat_test)
        benchmark.add_model("XGB", y_flat_test, xgb_probs, lt_flat_test)
        xgb.save("benchmark")
        logger.info("XGBoost complete")
    except Exception as e:
        logger.warning(f"XGBoost failed: {e}")

    # 3. LSTM (sliding windows, reduced)
    logger.info("Training LSTM...")
    n_train_windows = 3000
    ds_train = SolarFlareDataset(Xs[:idx_val], Xh[:idx_val],
                                  {"flare_label": labels[:idx_val], "lead_time": lead_times[:idx_val]},
                                  lookback=LOOKBACK, step=STEP)
    ds_val = SolarFlareDataset(Xs[idx_val:idx_test], Xh[idx_val:idx_test],
                                {"flare_label": labels[idx_val:idx_test], "lead_time": lead_times[idx_val:idx_test]},
                                lookback=LOOKBACK, step=STEP)

    n_train = min(n_train_windows, len(ds_train))
    n_val = min(500, len(ds_val))

    from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
    train_subset = Subset(ds_train, list(range(n_train)))
    val_subset = Subset(ds_val, list(range(n_val)))

    flare_labels_sub = np.array([ds_train[i][2]["flare_label"] for i in range(n_train)])
    weights = np.where(flare_labels_sub > 0, 10.0, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    lstm = LSTMFlareModule(input_dim=Xs.shape[1] + Xh.shape[1], hidden_dim=128, num_layers=2)
    import lightning as L
    trainer = L.Trainer(max_epochs=EPOCHS, accelerator="auto", log_every_n_steps=50, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(lstm, train_loader, val_loader)

    # LSTM evaluation — sliding window on test set
    ds_test = SolarFlareDataset(Xs[idx_test:], Xh[idx_test:],
                                 {"flare_label": labels[idx_test:], "lead_time": lead_times[idx_test:]},
                                 lookback=LOOKBACK, step=STEP)
    n_test = min(1000, len(ds_test))
    lstm.eval()
    lstm_probs = []
    lstm_y = []
    lstm_lt = []
    with torch.no_grad():
        for i in range(n_test):
            xs, xh, y = ds_test[i]
            logit = lstm(xs.unsqueeze(0), xh.unsqueeze(0))
            lstm_probs.append(torch.sigmoid(logit).item())
            lstm_y.append(y["flare_label"].item())
            lstm_lt.append(y["lead_time"].item())
    benchmark.add_model("LSTM", np.array(lstm_y), np.array(lstm_probs), np.array(lstm_lt))
    Path("models/checkpoints/lstm_benchmark").mkdir(parents=True, exist_ok=True)
    torch.save(lstm.state_dict(), "models/checkpoints/lstm_benchmark/best.ckpt")
    logger.info("LSTM complete")

    # 4. TimesNet (reduced)
    logger.info("Training TimesNet...")
    try:
        from backend.models.forecaster.timesnet_forecaster import TimesNetFlareModule
        timesnet = TimesNetFlareModule(lookback=LOOKBACK, d_model=32, d_ff=64, soft_channels=len(soft_cols), hard_channels=len(hard_cols))
        trainer2 = L.Trainer(max_epochs=EPOCHS, accelerator="auto", log_every_n_steps=50, logger=False, enable_checkpointing=False, enable_progress_bar=False)
        trainer2.fit(timesnet, train_loader, val_loader)

        timesnet.eval()
        tn_probs = []
        tn_y = []
        tn_lt = []
        with torch.no_grad():
            for i in range(n_test):
                xs, xh, y = ds_test[i]
                logit = timesnet(xs.unsqueeze(0), xh.unsqueeze(0))
                tn_probs.append(torch.sigmoid(logit).item())
                tn_y.append(y["flare_label"].item())
                tn_lt.append(y["lead_time"].item())
        benchmark.add_model("TimesNet", np.array(tn_y), np.array(tn_probs), np.array(tn_lt))
        Path("models/checkpoints/timesnet_benchmark").mkdir(parents=True, exist_ok=True)
        torch.save(timesnet.state_dict(), "models/checkpoints/timesnet_benchmark/best.ckpt")
        logger.info("TimesNet complete")
    except Exception as e:
        logger.warning(f"TimesNet failed: {e}")

    # Summary
    summary = benchmark.summarize()
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)

    if len(summary) > 0:
        winner = benchmark.select_winner(summary)
        print(f"\nWINNER: {winner}")
        with open("models/checkpoints/benchmark_winner.txt", "w") as f:
            f.write(winner)
    logger.info("CPU benchmark complete")

if __name__ == "__main__":
    main()
