"""Enhanced /alerts endpoint with validation status."""

from fastapi import APIRouter
from backend.services.live_predictor import get_predictor

router = APIRouter()


@router.get("/alerts")
def get_alerts():
    predictor = get_predictor()
    last = predictor._prediction_cache
    threshold = getattr(predictor, "_threshold", 0.5)
    prob = last.get("flare_probability", 0.0)
    return {
        "alert": last.get("alert_level", "GREEN") in ("ORANGE", "RED"),
        "alert_level": last.get("alert_level", "GREEN"),
        "flare_class": last.get("predicted_class", "A"),
        "probability": prob,
        "threshold": threshold,
        "above_threshold": bool(prob >= threshold),
        "lead_time_minutes": last.get("lead_time_minutes", 0.0),
        "timestamp": last.get("timestamp", ""),
        "validation_status": last.get("validation_status", "pending"),
        "prediction_id": last.get("prediction_id", ""),
    }
