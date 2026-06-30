"""Enhanced /models endpoint with benchmark scores and training metadata."""

from fastapi import APIRouter
from backend.services.live_predictor import get_predictor
from backend.services.prediction_store import get_store
from pathlib import Path
import yaml

router = APIRouter()

MODEL_DIR = Path("models/checkpoints")
CONFIG_PATH = Path("configs/training.yaml")


@router.get("/models")
def get_models():
    predictor = get_predictor()
    store = get_store()

    available = []
    for subdir in sorted(MODEL_DIR.iterdir()):
        if subdir.is_dir():
            pkgs = list(subdir.glob("*.pkl"))
            for p in pkgs:
                tag = p.stem.replace(f"{subdir.name}_", "")
                available.append({
                    "name": subdir.name,
                    "tag": tag,
                    "path": str(p),
                })
    if not available:
        available = [
            {"name": "xgboost", "tag": "benchmark", "path": ""},
            {"name": "random_forest", "tag": "benchmark", "path": ""},
        ]

    # Load training config for horizon info
    horizon = 30
    label_mode = "fixed_window"
    if CONFIG_PATH.exists():
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
        horizons = cfg.get("data", {}).get("forecast_horizons_minutes", [30])
        horizon = horizons[0] if horizons else 30
        label_mode = cfg.get("data", {}).get("label_mode", "fixed_window")

    threshold_path = MODEL_DIR / "optimal_threshold.txt"
    threshold = 0.5
    if threshold_path.exists():
        try:
            threshold = float(threshold_path.read_text().strip())
        except (ValueError, OSError):
            pass

    calibrator_path = MODEL_DIR / "calibrator.pkl"
    calibration = "none"
    if calibrator_path.exists():
        import pickle
        with open(calibrator_path, "rb") as f:
            cal_data = pickle.load(f)
            calibration = cal_data.get("chosen_method", "none")

    return {
        "active_model": predictor.model_name or "unknown",
        "available_models": available,
        "threshold": threshold,
        "calibration": calibration,
        "forecast_horizon_minutes": horizon,
        "label_mode": label_mode,
        "total_predictions": store.get_stats().get("total_predictions", 0),
    }
