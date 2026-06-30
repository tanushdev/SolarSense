"""
XGBoost Forecaster
===================
Gradient-boosted tree model for solar flare prediction.

WHY XGBOOST:
  - Often outperforms RF on tabular data by sequentially correcting errors
  - Built-in handling of missing values
  - Regularization to prevent overfitting on rare events
  - Feature importance + SHAP compatibility for explainability
  - Typically the strongest non-deep-learning baseline for flare forecasting

PHYSICS ADVANTAGE:
  XGBoost handles the extreme class imbalance naturally via
  scale_pos_weight and can learn decision boundaries that separate
  quiet-Sun noise from genuine precursor patterns.
"""

import numpy as np
from loguru import logger
import yaml
from pathlib import Path
import pickle

from backend.models.base_model import BaseModel


try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")


class XGBoostForecaster(BaseModel):
    """
    XGBoost classifier for flare/no-flare prediction.

    Usage:
        model = XGBoostForecaster()
        model.fit(X_train, y_train, X_val, y_val)
        probs = model.predict_proba(X_test)
    """

    def __init__(self, model_name: str = "xgboost",
                 config_path: str = "configs/models.yaml"):
        super().__init__(model_name, config_path)
        if not _XGB_AVAILABLE:
            raise ImportError("XGBoost is required. pip install xgboost")

        cfg = self.config.get("forecaster", {})
        xgb_cfg = cfg.get("xgboost", {})

        # Compute scale_pos_weight from class balance
        scale = xgb_cfg.get("scale_pos_weight", 10)

        self.model = xgb.XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 500),
            max_depth=xgb_cfg.get("max_depth", 8),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=scale,
            eval_metric=xgb_cfg.get("eval_metric", "aucpr"),
            early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 20),
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        self._is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        logger.info("XGB: Training with {} samples, {} features",
                    len(X_train), X_train.shape[1])
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self._is_fitted = True
        logger.info("XGB: Training complete")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> dict:
        if not self._is_fitted:
            return {}
        imp = self.model.feature_importances_
        return {f"feat_{i}": v for i, v in enumerate(imp)}

    def save(self, tag: str = "best"):
        path = self.checkpoint_dir / f"{self.model_name}_{tag}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Saved XGB model: {}", path)

    def load(self, tag: str = "best"):
        path = self.checkpoint_dir / f"{self.model_name}_{tag}.pkl"
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._is_fitted = True
        logger.info("Loaded XGB model: {}", path)
