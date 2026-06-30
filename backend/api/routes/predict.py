"""Predict endpoint — probabilistic flare forecast with uncertainty."""

import numpy as np
import pandas as pd
from fastapi import APIRouter
from loguru import logger
from datetime import datetime, timezone

from backend.api.schemas import ForecastResponse, SimilarEvent
from backend.services.live_predictor import get_predictor

router = APIRouter(tags=["predict"])


@router.get("/forecast", response_model=ForecastResponse)
@router.get("/predict", response_model=ForecastResponse)
def get_forecast():
    pred = get_predictor().predict()
    return _to_response(pred)


@router.post("/forecast", response_model=ForecastResponse)
@router.post("/predict", response_model=ForecastResponse)
def post_forecast():
    pred = get_predictor().predict()
    return _to_response(pred)


def _to_response(pred: dict) -> ForecastResponse:
    events_raw = pred.get("similar_events", [])
    events = [SimilarEvent(**ev) for ev in events_raw] if events_raw else []
    return ForecastResponse(
        timestamp=datetime.fromisoformat(pred.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        flare_probability=pred.get("flare_probability", 0.0),
        uncertainty=pred.get("uncertainty", 0.0),
        lower_bound=pred.get("lower_bound", 0.0),
        upper_bound=pred.get("upper_bound", 1.0),
        predicted_class=pred.get("predicted_class", "A"),
        class_probs=pred.get("class_probs", {"A": 1.0, "B": 0.0, "C": 0.0, "M": 0.0, "X": 0.0}),
        lead_time_minutes=pred.get("lead_time_minutes", 0.0),
        alert_level=pred.get("alert_level", "GREEN"),
        similar_events=events,
        physics_reason=pred.get("physics_reason", ""),
        model=pred.get("model", "unknown"),
        data_timestamp=pred.get("data_timestamp", ""),
    )
