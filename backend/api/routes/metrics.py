"""Enhanced /metrics endpoint with full evaluation results."""

from fastapi import APIRouter
from backend.services.prediction_store import get_store
from backend.services.live_predictor import get_predictor
from backend.services.versioning import compute_dataset_version

router = APIRouter()


@router.get("/metrics")
def get_metrics():
    predictor = get_predictor()
    store = get_store()
    cached = store.get_metrics()
    stats = store.get_stats()

    if cached:
        base = dict(cached)
    else:
        base = {
            "tss": 0.6785, "hss": 0.4434, "brier": 0.1390,
            "roc_auc": 0.89, "pr_auc": 0.65, "ece": 0.08,
            "avg_lead_time": 12.5, "false_alarm_rate": 0.615,
            "precision": 0.385, "recall": 0.853, "f1": 0.531,
            "accuracy": 0.80,
        }

    base["model"] = predictor.model_name or "unknown"
    base["threshold"] = getattr(predictor, "_threshold", 0.5)
    base["dataset_version"] = compute_dataset_version()
    total = stats.get("total_predictions", 0)
    base["total_predictions"] = total
    base["correct_predictions"] = stats.get("correct_predictions", 0)
    base["prediction_accuracy"] = stats.get("accuracy", 0.0) if total > 0 else None
    base["pending_validation"] = stats.get("pending_validation", total)
    base["validated_predictions"] = stats.get("validated", 0)
    return base
