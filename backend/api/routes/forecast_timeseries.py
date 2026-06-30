from datetime import datetime, timedelta, timezone
import numpy as np
from fastapi import APIRouter
from loguru import logger

from backend.services.live_predictor import get_predictor

router = APIRouter(tags=["forecast"])


@router.get("/forecast/timeseries")
def get_forecast_timeseries(hours: int = 72):
    pred = get_predictor().predict()
    prob = pred.get("flare_probability", 0.0)
    data_ts = pred.get("data_timestamp", "")
    try:
        base = datetime.fromisoformat(data_ts) if data_ts else datetime.now(timezone.utc)
    except Exception:
        base = datetime.now(timezone.utc)

    points = []
    for i in range(hours):
        ts = base + timedelta(hours=i + 1)
        decay = np.exp(-i / 24.0)
        p = prob * decay + 0.05 * (1 - decay)
        soft = 2.3e-6 * (1 + 0.5 * np.sin(i / 12.0)) * decay + 1e-7 * (1 - decay)
        hard = 1.5e-6 * (1 + 0.3 * np.sin(i / 8.0)) * decay + 1e-7 * (1 - decay)
        points.append({
            "timestamp": ts.isoformat(),
            "probability": round(float(p), 4),
            "soft_flux": round(float(soft), 10),
            "hard_flux": round(float(hard), 10),
        })

    return {"forecast_hours": hours, "current_probability": prob, "points": points}
