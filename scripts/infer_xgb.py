"""
Batch Inference with XGBoost
=============================
Load the trained XGBoost model and run predictions on new data.

Usage:
    python scripts/infer_xgb.py                         # runs on test set
    python scripts/infer_xgb.py --input path/to/data.parquet  # custom data
    python scripts/infer_xgb.py --output results.csv
"""

import sys, argparse, warnings, yaml, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models.forecaster.xgboost_forecaster import XGBoostForecaster


def get_feature_cols(df):
    """Recreate the 154-feature column list used during training."""
    sc = [c for c in df.columns if c.startswith("soft_")]
    hc = [c for c in df.columns if c.startswith("hard_")]
    cross = {"soft":["quality_solexs","d_soft_flux_dt","d2_soft_flux_dt2","d_log_soft_flux_dt"],
             "hard":["quality_hel1os","d_hard_flux_dt","d2_hard_flux_dt2","d_log_hard_flux_dt",
                     "event_rate_1h","event_rate_6h","event_rate_24h","flux_above_2sigma"],
             "both":["hardness_ratio","hardness_ratio_log","hardness_ratio_deriv",
                     "hardness_ratio_5min","hardness_ratio_15min","spectral_index",
                     "spectral_index_5min","spectral_index_deriv","hxr_sxr_corr_15min",
                     "hxr_sxr_lag1_corr","neupert_proxy","neupert_residual",
                     "neupert_residual_abs","hour_sin","hour_cos","doy_sin","doy_cos",
                     "data_gap","artifact_flag"]}
    sc.extend(cross["soft"]); sc.extend(cross["both"])
    hc.extend(cross["hard"]); hc.extend(cross["both"])
    sc = [c for c in sc if c in df.columns]
    hc = [c for c in hc if c in df.columns]
    return sc + hc  # 154 features total


def main():
    parser = argparse.ArgumentParser(description="XGBoost batch inference")
    parser.add_argument("--input", default=None,
                        help="Input parquet file (default: test set)")
    parser.add_argument("--output", default="predictions.csv",
                        help="Output CSV path")
    parser.add_argument("--model-tag", default="xgb_benchmark",
                        help="Model checkpoint tag")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold")
    args = parser.parse_args()

    # Load model
    print(f"Loading XGBoost model (tag={args.model_tag})...")
    model = XGBoostForecaster()
    model.load(tag=args.model_tag)

    # Get input data
    if args.input:
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_parquet("dataset/processed/test_timeseries_60s.parquet")
    print(f"Loaded {len(df)} rows from {args.input or 'test set'}")

    # Build features
    feat_cols = get_feature_cols(df)
    print(f"Using {len(feat_cols)} feature columns")
    X = df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict
    print("Running inference...")
    probs = model.predict_proba(X)
    preds = (probs > args.threshold).astype(int)

    # Save results
    results = pd.DataFrame({
        "probability": probs,
        "prediction": preds,
    })
    results.to_csv(args.output, index=False)
    print(f"Saved {len(results)} predictions to {args.output}")

    # Stats
    n_flare = int(preds.sum())
    print(f"Predictions: {n_flare}/{len(preds)} flare ({100*n_flare/len(preds):.1f}%)")
    print(f"Probability stats: mean={probs.mean():.4f} median={np.median(probs):.4f} "
          f"max={probs.max():.4f}")


if __name__ == "__main__":
    main()
