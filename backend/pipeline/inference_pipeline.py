"""
Inference Pipeline
===================
Serves predictions from the best-performing trained model.

Endpoints (used by FastAPI /predict):
  1. Preprocess: resample, align, extract features
  2. Predict: run MC Dropout inference with uncertainty
  3. Calibrate: apply temperature scaling
  4. Retrieve: query memory bank for similar events
  5. Return: probability, interval, similar events, alert level

Usage:
    from backend.pipeline.inference_pipeline import InferencePipeline
    pipeline = InferencePipeline(model_path="models/checkpoints/best.ckpt")
    result = pipeline.predict(soft_flux, hard_flux)
"""

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pathlib import Path
import yaml

from backend.models.forecaster.patchtst_forecaster import DualStreamPatchTST
from backend.models.bayesian.uncertainty import MCDropout, ConformalCalibrator
from backend.models.memory.flare_memory import FlareMemoryBank
from backend.features.feature_pipeline import FeaturePipeline


class InferencePipeline:
    """
    End-to-end inference pipeline for deployed model.

    Usage:
        pipeline = InferencePipeline()
        result = pipeline.predict(soft_flux_array, hard_flux_array)
        # Returns dict with probability, uncertainty, alert_level, similar_events
    """

    def __init__(self, model_path: str = None,
                 config_path: str = "configs/training.yaml",
                 models_config: str = "configs/models.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        with open(models_config) as f:
            self.models_cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.feature_pipeline = FeaturePipeline()
        self.mc_dropout = MCDropout(self.model, n_passes=50)

        # Try to load calibrator
        self.calibrator = ConformalCalibrator()

        # Try to load memory bank
        try:
            self.memory = FlareMemoryBank()
            self.memory.load("models/memory")
        except Exception:
            self.memory = None

        logger.info("InferencePipeline ready on {}", self.device)

    def _load_model(self, model_path: str = None):
        """Load trained model from checkpoint."""
        if model_path is None:
            model_path = "models/checkpoints/patchtst_deploy_run/best.ckpt"

        model = DualStreamPatchTST()
        if Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device,
                               weights_only=False)
            if "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            else:
                model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            logger.info("Loaded model from: {}", model_path)
        else:
            logger.warning("No checkpoint found at {}. Using untrained model.", model_path)
        return model

    def _extract_features(self, soft_flux: np.ndarray,
                          hard_flux: np.ndarray) -> tuple:
        """
        Build feature vectors from raw flux arrays.

        Returns
        -------
        (soft_features, hard_features) as torch tensors
        """
        df = pd.DataFrame({
            "soft_flux": soft_flux,
            "hard_flux": hard_flux,
            "timestamp_utc": pd.date_range("now", periods=len(soft_flux),
                                            freq="5s", tz="UTC"),
        }).set_index("timestamp_utc")

        features = self.feature_pipeline.extract_all(df)
        df_full = pd.concat([df, features], axis=1).fillna(0)

        soft_cols = [c for c in df_full.columns if c.startswith("soft_")]
        hard_cols = [c for c in df_full.columns if c.startswith("hard_")]

        soft = torch.tensor(df_full[soft_cols].values, dtype=torch.float32)
        hard = torch.tensor(df_full[hard_cols].values, dtype=torch.float32)
        return soft.unsqueeze(0), hard.unsqueeze(0)

    def _alert_level(self, probability: float, uncertainty: float) -> str:
        """Map probability + uncertainty to alert level."""
        if uncertainty > self.cfg.get("alert", {}).get("uncertainty_max_acceptable", 0.15):
            return "LOW_CONFIDENCE"
        if probability >= 0.8:
            return "RED"
        elif probability >= 0.6:
            return "ORANGE"
        elif probability >= 0.3:
            return "YELLOW"
        else:
            return "GREEN"

    def predict(self, soft_flux: np.ndarray,
                hard_flux: np.ndarray) -> dict:
        """
        Run full inference: features → MC Dropout → calibrate → retrieve.

        Parameters
        ----------
        soft_flux : (T,) SoLEXS soft X-ray flux
        hard_flux : (T,) HEL1OS hard X-ray flux

        Returns
        -------
        dict with probability, uncertainty, bounds, alert_level, similar_events
        """
        x_soft, x_hard = self._extract_features(soft_flux, hard_flux)
        mc_result = self.mc_dropout.predict(x_soft, x_hard)

        prob = float(mc_result["probability"])
        unc = float(mc_result["uncertainty"])

        # Retrieve similar events
        similar = []
        if self.memory is not None:
            try:
                combined = np.concatenate([
                    x_soft.numpy().flatten(),
                    x_hard.numpy().flatten(),
                ]).reshape(1, -1)
                similar = self.memory.query(combined, top_k=5)
            except Exception as e:
                logger.debug("Memory query failed: {}", e)

        return {
            "probability": prob,
            "uncertainty": unc,
            "lower_bound": float(mc_result["lower_bound"]),
            "upper_bound": float(mc_result["upper_bound"]),
            "alert_level": self._alert_level(prob, unc),
            "similar_events": similar,
        }
