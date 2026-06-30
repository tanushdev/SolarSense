"""
Abstract Base Model
===================
ALL models in SolarSense-AI inherit from BaseModel.
This enforces:
  - Consistent interface (fit, predict, predict_proba, uncertainty)
  - Mandatory logging of all experiments
  - Consistent checkpoint saving
  - Bayesian uncertainty output format
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
import mlflow
from loguru import logger
from pathlib import Path
import yaml


class BaseModel(ABC):
    """Base class for all SolarSense-AI models."""

    def __init__(self, model_name: str,
                 config_path: str = "configs/models.yaml"):
        self.model_name = model_name
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.checkpoint_dir = Path("models/checkpoints") / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Model initialized: {}", model_name)

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        """Train the model. Log metrics to MLflow."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability array shape (N,) or (N, n_classes)."""
        pass

    def predict_with_uncertainty(self, X: np.ndarray,
                                 n_passes: int = 50) -> dict:
        """
        Returns dict:
          probability  : mean prediction across MC Dropout passes
          uncertainty  : std of predictions (epistemic uncertainty)
          lower_bound  : 5th percentile
          upper_bound  : 95th percentile
        Override in subclasses that support MC Dropout.
        """
        probs = self.predict_proba(X)
        return {
            "probability":  probs,
            "uncertainty":  np.zeros_like(probs),
            "lower_bound":  probs,
            "upper_bound":  probs,
        }

    def save(self, tag: str = "best"):
        path = self.checkpoint_dir / f"{self.model_name}_{tag}.pt"
        torch.save(self.state_dict(), path)
        logger.info("Saved checkpoint: {}", path)

    def load(self, tag: str = "best"):
        path = self.checkpoint_dir / f"{self.model_name}_{tag}.pt"
        self.load_state_dict(torch.load(path))
        logger.info("Loaded checkpoint: {}", path)