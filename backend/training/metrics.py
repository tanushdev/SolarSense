"""
Evaluation Metrics for Solar Flare Prediction
==============================================

STANDARD SPACE-WEATHER METRICS:

TSS (True Skill Statistic) = TPR - FPR
  Range: -1 to +1. Perfect = 1. Random = 0.
  The PRIMARY metric for space-weather forecast evaluation.
  Recommended by WMO for operational forecast comparison.

HSS (Heidke Skill Score) = (TP+TN - Expected) / (Total - Expected)
  Measures improvement over random chance.
  HSS > 0.5 is considered operationally useful.

Brier Score = mean((p - y)²)
  Probabilistic accuracy. Lower is better.
  For well-calibrated forecasts in rare-event regime.

ECE (Expected Calibration Error)
  Measures how well forecast probabilities match empirical frequencies.
  A forecast that says 70% should be right 70% of the time.

Lead Time
  Primary operational metric for this challenge.
  = time of first alert BEFORE flare peak
  Target: > 10 minutes for M/X class events.

False Alarm Rate = FP / (FP + TN)
  Must be kept low for operational use.
  Target: < 30% FAR for M+ class events.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix)
from loguru import logger


def compute_confusion_elements(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                threshold: float = 0.5):
    """Compute TP, TN, FP, FN."""
    y_bin = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def true_skill_statistic(y_true, y_pred, threshold=0.5):
    """TSS = TPR - FPR. Primary metric."""
    cm = compute_confusion_elements(y_true, y_pred, threshold)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    tpr = tp / (tp + fn + 1e-10)
    fpr = fp / (fp + tn + 1e-10)
    return tpr - fpr


def heidke_skill_score(y_true, y_pred, threshold=0.5):
    """HSS — skill relative to random chance."""
    cm = compute_confusion_elements(y_true, y_pred, threshold)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    n   = tp + tn + fp + fn
    exp = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n + 1e-10)
    return (tp + tn - exp) / (n - exp + 1e-10)


def brier_score(y_true, y_prob):
    """Probabilistic accuracy."""
    return np.mean((y_prob - y_true) ** 2)


def compute_lead_time(alerts: pd.Series, catalog: pd.DataFrame,
                      tolerance_minutes: int = 60) -> pd.Series:
    """
    For each true flare in catalog, find the first alert that preceded it
    within tolerance_minutes. Return lead time in minutes.
    
    Parameters
    ----------
    alerts  : pd.Series of alert timestamps (when model fired)
    catalog : flare catalog with start_time and peak_time columns
    
    Returns
    -------
    pd.Series of lead times (minutes) per flare. NaN = missed.
    """
    lead_times = []
    for _, event in catalog.iterrows():
        peak = pd.Timestamp(event["peak_time"])
        window_start = peak - pd.Timedelta(minutes=tolerance_minutes)
        # Find all alerts in the pre-peak window
        early_alerts = alerts[
            (alerts >= window_start) & (alerts <= peak)
        ]
        if len(early_alerts) > 0:
            first_alert = early_alerts.min()
            lead_times.append((peak - first_alert).total_seconds() / 60)
        else:
            lead_times.append(np.nan)
    return pd.Series(lead_times, index=catalog.index)


def full_evaluation_report(y_true, y_pred_prob, alerts, catalog) -> dict:
    """Run complete evaluation suite and return metrics dict."""
    tss  = true_skill_statistic(y_true, y_pred_prob)
    hss  = heidke_skill_score(y_true, y_pred_prob)
    bs   = brier_score(y_true, y_pred_prob)
    auc  = roc_auc_score(y_true, y_pred_prob)
    auprc = average_precision_score(y_true, y_pred_prob)
    lt   = compute_lead_time(alerts, catalog)

    report = {
        "TSS":             round(tss, 4),
        "HSS":             round(hss, 4),
        "Brier_Score":     round(bs, 4),
        "ROC_AUC":         round(auc, 4),
        "AUPRC":           round(auprc, 4),
        "Lead_Time_mean":  round(lt.mean(), 2),
        "Lead_Time_median": round(lt.median(), 2),
        "Lead_Time_p90":   round(lt.quantile(0.9), 2),
        "Detection_Rate":  round((~lt.isna()).mean(), 4),
    }
    logger.info("Evaluation complete: {}", report)
    return report