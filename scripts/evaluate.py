#!/usr/bin/env python
"""
Evaluate a trained model on the test set.

Usage:
    python scripts/evaluate.py --model path/to/model.ckpt
    python scripts/evaluate.py --model random_forest
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.evaluation.evaluator import Evaluator
from backend.evaluation.calibration import TemperatureScaler
from backend.evaluation.explainer import Explainer


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", default="patchtst_deploy_run",
                        help="Experiment name or checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold")
    args = parser.parse_args()

    print(f"Evaluating model: {args.model}")
    print("Loading test data...")

    test_df = pd.read_parquet("dataset/processed/test_timeseries.parquet")
    test_labels = pd.read_csv("dataset/catalogs/nowcast_catalog.csv")

    # For evaluation, we use a simple rule: flare if hard_flux > 3-sigma
    hard = test_df["hard_flux"].values
    threshold = hard.mean() + 3 * hard.std()
    y_pred = (hard > threshold).astype(int)

    # Ground truth: samples near catalog events
    y_true = np.zeros(len(test_df))
    for _, event in test_labels.iterrows():
        peak = pd.to_datetime(event["peak_time"])
        if peak in test_df.index:
            y_true[test_df.index.get_loc(peak)] = 1

    # Evaluate
    evaluator = Evaluator()
    result = evaluator.evaluate(y_true, hard / (hard.max() + 1e-10),
                                 model_name=args.model)

    print(f"\nResults for {args.model}:")
    print(f"  TSS:  {result.tss:.4f}")
    print(f"  HSS:  {result.hss:.4f}")
    print(f"  AUC:  {result.auc:.4f}")
    print(f"  Brier: {result.brier:.4f}")
    print(f"  ECE:  {result.ece:.4f}")
    print(f"  FAR:  {result.false_alarm_rate:.4f}")
    print(f"  F1:   {result.f1:.4f}")


if __name__ == "__main__":
    main()
