import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

# Add root directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.models.forecaster.patchtst_forecaster import DualStreamPatchTST
from backend.training.trainer import SolarFlareDataset, SolarFlareTrainer

def build_labels(df: pd.DataFrame, df_catalog: pd.DataFrame, lookahead_samples: int = 720) -> dict:
    """
    Compile machine learning labels from the event catalog.
    
    Parameters
    ----------
    df: timeseries DataFrame
    df_catalog: events catalog DataFrame
    """
    logger.info("Compiling target labels from catalog...")
    n_samples = len(df)
    timestamps = df.index
    
    flare_label = np.zeros(n_samples, dtype=np.float32)
    flare_class = np.zeros(n_samples, dtype=np.float32)
    lead_time = np.zeros(n_samples, dtype=np.float32)
    hazard = np.zeros((n_samples, 60), dtype=np.float32)
    
    # Class map (0-indexed for cross_entropy)
    class_map = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    
    # Parse event times
    events = []
    for _, row in df_catalog.iterrows():
        events.append({
            "start": pd.to_datetime(row["start_time"], utc=True),
            "peak": pd.to_datetime(row["peak_time"], utc=True),
            "end": pd.to_datetime(row["end_time"], utc=True),
            "class": class_map.get(row["flare_class"], 1),
        })
        
    for i, ts in enumerate(timestamps):
        # Look ahead up to lookahead_samples (e.g. 1 hour = 720 samples at 5s)
        end_idx = min(i + lookahead_samples, n_samples - 1)
        future_ts = timestamps[end_idx]
        
        # Check if any event starts in this lookahead horizon
        next_event = None
        for ev in events:
            if ev["start"] > ts and ev["start"] <= future_ts:
                next_event = ev
                break
                
        if next_event:
            flare_label[i] = 1.0
            flare_class[i] = float(next_event["class"])
            # minutes to peak
            time_to_peak = (next_event["peak"] - ts).total_seconds() / 60.0
            lead_time[i] = float(time_to_peak)
            
            # Hazard target (binary sequence for each of next 60 minutes)
            for m in range(60):
                future_minute_ts = ts + pd.Timedelta(minutes=m)
                if next_event["start"] <= future_minute_ts <= next_event["end"]:
                    hazard[i, m] = 1.0
                    
    return {
        "flare_label": flare_label,
        "flare_class": flare_class,
        "lead_time": lead_time,
        "hazard": hazard
    }

def main():
    logger.info("Loading processed splits...")
    train_df = pd.read_parquet("dataset/processed/train_timeseries.parquet")
    val_df = pd.read_parquet("dataset/processed/val_timeseries.parquet")
    # Try the pipeline-generated catalog first, fall back to versioned catalog
    catalog_path = Path("dataset/catalogs/nowcast_catalog.csv")
    if not catalog_path.exists():
        catalog_path = Path("dataset/catalogs/nowcast_catalog_v3.csv")
    df_catalog = pd.read_csv(catalog_path)
    
    # Feature columns
    soft_cols = [c for c in train_df.columns if "soft" in c]
    hard_cols = [c for c in train_df.columns if "hard" in c or "hxr" in c or "neupert" in c]
    
    logger.info("Found {} soft features and {} hard features", len(soft_cols), len(hard_cols))
    
    X_train_soft = np.nan_to_num(train_df[soft_cols].values, nan=0.0)
    X_train_hard = np.nan_to_num(train_df[hard_cols].values, nan=0.0)
    
    X_val_soft = np.nan_to_num(val_df[soft_cols].values, nan=0.0)
    X_val_hard = np.nan_to_num(val_df[hard_cols].values, nan=0.0)
    
    train_labels = build_labels(train_df, df_catalog)
    val_labels = build_labels(val_df, df_catalog)
    
    # Take a subsample of data to train quickly for validation/deployment verification
    # Using lookback=1440, stride step=60 (every 5 minutes) to train fast
    train_dataset = SolarFlareDataset(X_train_soft, X_train_hard, train_labels, lookback=1440, step=60)
    val_dataset = SolarFlareDataset(X_val_soft, X_val_hard, val_labels, lookback=1440, step=60)
    
    logger.info("Dataset prepared | Train windows: {} | Val windows: {}", len(train_dataset), len(val_dataset))
    
    # Initialize PatchTST Module
    model = DualStreamPatchTST(
        n_soft_features=len(soft_cols),
        n_hard_features=len(hard_cols)
    )
    
    # Initialize Trainer and train
    trainer = SolarFlareTrainer()
    
    logger.info("Starting training of PatchTST...")
    # Overwrite configuration epochs to 1 epoch for rapid testing during deployment run
    trainer.cfg["forecaster"]["patchtst"]["epochs"] = 1
    
    # Train
    trained_model = trainer.train_forecaster(model, train_dataset, val_dataset, experiment_name="patchtst_deploy_run")
    
    # Save the checkpoint manually
    checkpoint_dir = Path("models/checkpoints/patchtst")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), checkpoint_dir / "best.ckpt")
    logger.info("Model training completed and checkpoint saved successfully!")

if __name__ == "__main__":
    main()
