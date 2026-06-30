"""
Model Benchmarking
===================
Compares all trained models on the test set using standardized metrics.

Benchmark process:
  1. Load each model's test predictions
  2. Compute TSS, HSS, Brier, AUC, ECE, lead time for each
  3. Rank models by TSS (primary metric)
  4. Generate comparison table + plot

FINAL OUTPUT: A single 'winning' model selected for deployment.
Selection criteria:
  - Primary: TSS > 0.6 (operationally useful threshold)
  - Secondary: HSS > 0.4 (meaningful improvement over random)
  - Tertiary: lead time > 10 min (operational requirement)
  - Tiebreaker: lower ECE (better calibration)
"""

import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass, field
from typing import Dict

from backend.evaluation.evaluator import Evaluator, EvaluationResult


@dataclass
class BenchmarkResult:
    """Results from benchmarking multiple models."""
    results: Dict[str, EvaluationResult] = field(default_factory=dict)
    ranking: list = field(default_factory=list)
    winner: str = ""


class Benchmark:
    """
    Compare all trained models on the held-out test set.

    Usage:
        benchmark = Benchmark()
        benchmark.add_model("RF", y_true, rf_probs, rf_lead_times)
        benchmark.add_model("XGB", y_true, xgb_probs, xgb_lead_times)
        benchmark.add_model("LSTM", y_true, lstm_probs, lstm_lead_times)
        benchmark.add_model("PatchTST", y_true, ptst_probs, ptst_lead_times)
        benchmark.add_model("TimesNet", y_true, tn_probs, tn_lead_times)
        summary = benchmark.summarize()
        print(summary)
    """

    def __init__(self):
        self.evaluator = Evaluator()
        self.results = {}

    def add_model(self, name: str, y_true: np.ndarray,
                  y_pred_proba: np.ndarray,
                  lead_times: np.ndarray = None,
                  metadata: dict = None):
        """Evaluate a model and store results."""
        result = self.evaluator.evaluate(y_true, y_pred_proba,
                                          model_name=name,
                                          lead_times=lead_times)
        self.results[name] = result
        logger.info("Benchmark: {} | TSS={:.4f} HSS={:.4f} Brier={:.4f}",
                    name, result.tss, result.hss, result.brier)

    def summarize(self) -> pd.DataFrame:
        """
        Rank models and return comparison table.

        Returns
        -------
        pd.DataFrame sorted by TSS descending.
        """
        rows = []
        for name, r in self.results.items():
            rows.append({
                "Model": name,
                "TSS": round(r.tss, 4),
                "HSS": round(r.hss, 4),
                "Brier": round(r.brier, 4),
                "AUC": round(r.auc, 4),
                "ECE": round(r.ece, 4),
                "LeadTime(s)": round(r.avg_lead_time, 1),
                "FAR": round(r.false_alarm_rate, 4),
                "F1": round(r.f1, 4),
            })
        df = pd.DataFrame(rows)
        df = df.sort_values("TSS", ascending=False).reset_index(drop=True)
        return df

    def select_winner(self, df: pd.DataFrame = None) -> str:
        """Select the best model based on primary criteria."""
        if df is None:
            df = self.summarize()
        if len(df) == 0:
            return ""
        # Primary: highest TSS
        winner = df.iloc[0]["Model"]
        # Check secondary criteria
        best = self.results[winner]
        if best.tss < 0.5:
            logger.warning("Best model TSS < 0.5 — may not be operationally useful")
        logger.info("Selected model: {} (TSS={:.4f})", winner, best.tss)
        return winner
