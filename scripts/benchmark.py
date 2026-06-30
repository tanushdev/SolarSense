#!/usr/bin/env python
"""
Benchmark all models on the test set and select the best one.

Compares: Random Forest, XGBoost, LSTM, PatchTST, TimesNet
Uses: TSS, HSS, Brier, AUC, ECE, Lead Time

Usage:
    python scripts/benchmark.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.evaluation.benchmark import Benchmark


def main():
    print("Benchmarking all models...")
    print("This script requires all models to be pre-trained.")
    print("Run scripts/train.py first to train all models.")

    benchmark = Benchmark()
    summary = benchmark.summarize()
    if len(summary) == 0:
        print("No model results found. Train models first.")
    else:
        print(summary.to_string())
        winner = benchmark.select_winner(summary)
        print(f"Recommended model: {winner}")


if __name__ == "__main__":
    main()
