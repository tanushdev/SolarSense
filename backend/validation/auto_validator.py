"""
Automatic GOES Validation
==========================
Background thread that validates past predictions against real GOES data.

Workflow:
  1. Query PredictionStore for predictions with validation_status='pending'
     whose forecast window has expired (timestamp + horizon has passed).
  2. Fetch GOES flare catalog for the relevant time window.
  3. If GOES reports a flare within tolerance → correct=1
     If prediction says flare but GOES doesn't → correct=0 (false alarm)
     If prediction says no-flare but GOES does → correct=0 (miss)
  4. Update prediction record in SQLite.
"""

import threading
import time as time_module
from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger
import json
import urllib.request
import urllib.error

from backend.services.prediction_store import get_store

GOES_FLARES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json"
POLL_INTERVAL = 120
TOLERANCE_MINUTES = 15
FORECAST_WINDOW_BUFFER_MINUTES = 5


def _fetch_goes_flares() -> list[dict]:
    """Fetch latest GOES flare list from NOAA."""
    try:
        req = urllib.request.Request(GOES_FLARES_URL, headers={"User-Agent": "SolarSense-AI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as e:
        logger.warning(f"GOES validation fetch failed: {e}")
        return []


def _match_goes_event(prediction: dict, goes_events: list[dict]) -> tuple:
    """
    Check if any GOES event matches the prediction's forecast window.

    Returns (validation_status, goes_class, goes_time, correct).
    """
    try:
        pred_time = datetime.fromisoformat(prediction["timestamp"])
        horizon_minutes = prediction.get("forecast_horizon_minutes", 30)
        window_end = pred_time + timedelta(minutes=horizon_minutes + FORECAST_WINDOW_BUFFER_MINUTES)
        window_start = pred_time - timedelta(minutes=5)  # small tolerance before
        prob = prediction.get("probability", 0.0)
        threshold = prediction.get("threshold", 0.5)
        predicted_flare = prob >= threshold

        for event in goes_events:
            event_time_str = event.get("peak_time", event.get("begin_time", ""))
            if not event_time_str:
                continue
            try:
                event_time = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            if window_start <= event_time <= window_end:
                goes_class = event.get("class", "UNKNOWN")
                is_major = goes_class and goes_class[0] in ("C", "M", "X")

                if predicted_flare and is_major:
                    return ("validated", goes_class, event_time_str, 1)
                elif not predicted_flare and is_major:
                    return ("missed", goes_class, event_time_str, 0)
                elif predicted_flare and not is_major:
                    return ("validated", goes_class, event_time_str, 0)

        if predicted_flare:
            return ("false_alarm", None, None, 0)

        return ("validated", None, None, 1)

    except Exception as e:
        logger.error(f"GOES match error: {e}")
        return ("error", None, None, None)


def validate_pending_predictions():
    """Check all pending predictions against GOES data."""
    store = get_store()
    pending = store.get_pending_validations(limit=100)
    if not pending:
        return

    goes_events = _fetch_goes_flares()
    if not goes_events:
        logger.debug("No GOES events fetched for validation")
        return

    now = datetime.now(timezone.utc)
    validated = 0
    for pred in pending:
        try:
            pred_time = datetime.fromisoformat(pred["timestamp"])
            horizon = pred.get("forecast_horizon_minutes", 30)
            # Only validate if forecast window has expired
            if pred_time + timedelta(minutes=horizon + FORECAST_WINDOW_BUFFER_MINUTES) < now:
                status, goes_cls, goes_time, correct = _match_goes_event(pred, goes_events)
                store.update_validation(
                    prediction_id=pred["prediction_id"],
                    status=status,
                    goes_class=goes_cls,
                    goes_time=goes_time,
                    correct=correct,
                )
                validated += 1
        except Exception as e:
            logger.error(f"Validation error for prediction {pred.get('prediction_id')}: {e}")

    if validated > 0:
        logger.info(f"Auto-validation: {validated} predictions validated")


class AutoValidator:
    """Background thread for automatic GOES validation."""

    def __init__(self):
        self._running = False
        self._thread = None

    def start(self, interval_seconds: int = POLL_INTERVAL):
        self._running = True

        def _loop():
            logger.info("AutoValidator: started (poll every {}s)", interval_seconds)
            while self._running:
                try:
                    validate_pending_predictions()
                except Exception as e:
                    logger.error(f"AutoValidator error: {e}")
                time_module.sleep(interval_seconds)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False


_validator_instance = None


def get_auto_validator() -> AutoValidator:
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = AutoValidator()
    return _validator_instance
