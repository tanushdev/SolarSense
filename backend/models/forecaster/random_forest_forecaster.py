"""
Random Forest Forecaster
=========================
Baseline model for solar flare prediction.

WHY RANDOM FOREST:
  - Handles high-dimensional feature spaces (80+ features) without overfitting
  - Naturally captures non-linear feature interactions
  - Provides feature importance for physical interpretation
  - Fast to train (minutes vs hours for deep learning)
  - Good uncertainty proxy via tree variance

PHYSICS ADVANTAGE:
  RF feature importance directly answers: "Which physical precursor
  signal is most predictive?" — e.g., hardness_ratio vs spectral_index.
  This guides space physicists toward the most informative features.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
import yaml
from pathlib import Path
import pickle

from backend.models.base_model import BaseModel


class RandomForestForecaster(BaseModel):
    """
    Random Forest classifier for flare/no-flare prediction.

    Usage:
        model = RandomForestForecaster()
        model.fit(X_train, y_train, X_val, y_val)
        probs = model.predict_proba(X_test)
        uncertainty = model.predict_with_uncertainty(X_test)
    """

    def __init__(self, model_name: str = "random_forest",
                 config_path: str = "configs/models.yaml"):
        super().__init__(model_name, config_path)
        cfg = self.config.get("forecaster", {})
        rf_cfg = cfg.get("random_forest", {})
        self.model = RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 300),
            max_depth=rf_cfg.get("max_depth", 20),
            min_samples_split=rf_cfg.get("min_samples_split", 5),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            class_weight=rf_cfg.get("class_weight", "balanced_subsample"),
            n_jobs=rf_cfg.get("n_jobs", -1),
            random_state=42,
            verbose=0,
        )
        self._is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        logger.info("RF: Training with {} samples, {} features",
                    len(X_train), X_train.shape[1])
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("RF: Training complete")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict_with_uncertainty(self, X: np.ndarray,
                                 n_passes: int = None) -> dict:
        """
        Use tree variance as uncertainty proxy.
        Each tree in the forest gives a vote → std of votes = uncertainty.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet.")
        tree_probs = np.array([tree.predict_proba(X)[:, 1]
                               for tree in self.model.estimators_])
        mean_prob = np.mean(tree_probs, axis=0)
        std_prob = np.std(tree_probs, axis=0)
        lower = np.percentile(tree_probs, 5, axis=0)
        upper = np.percentile(tree_probs, 95, axis=0)
        return {
            "probability": mean_prob,
            "uncertainty": std_prob,
            "lower_bound": lower,
            "upper_bound": upper,
        }

    def feature_importance(self) -> dict:
        """Return feature importance scores."""
        if not self._is_fitted:
            return {}
        return {f"feat_{i}": imp
                for i, imp in enumerate(self.model.feature_importances_)}

    def save(self, tag: str = "best"):
        path = self.checkpoint_dir / f"{self.model_name}_{tag}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Saved RF model: {}", path)

    def load(self, tag: str = "best"):
        path = self.checkpoint_dir / f"{self.model_name}_{tag}.pkl"
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._is_fitted = True
        logger.info("Loaded RF model: {}", path)
