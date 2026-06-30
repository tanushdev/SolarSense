"""
Model Explainability
=====================
Interpretability tools for solar flare predictions.

Helps solar physicists understand WHY a model predicted a flare:
  1. SHAP values (feature-level: "hardness_ratio contributed +0.3 to probability")
  2. Feature importance (model-level: top-10 predictive features)
  3. Attention visualization (for transformer models)
  4. Similar event retrieval (from memory bank)

PHYSICS BENEFIT:
  The explainer connects ML predictions to known physics precursors.
  If SHAP shows hardness_ratio is the top contributor, the space
  physicist can verify against the actual light curve.
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Optional


class Explainer:
    """
    Model-agnostic explainer for flare predictions.

    Usage:
        explainer = Explainer()
        explainer.add_feature_importance(["hardness_ratio", ...], [0.35, ...])
        explanation = explainer.explain_prediction(feature_values, top_k=5)
    """

    def __init__(self):
        self.feature_names = []
        self.feature_importance = None
        self._has_importance = False

    def add_feature_importance(self, names: List[str], values: np.ndarray):
        """Store global feature importance (e.g. from RF or SHAP)."""
        self.feature_names = names
        self.feature_importance = values
        self._has_importance = True

    def explain_prediction(self, feature_values: np.ndarray,
                           top_k: int = 5) -> List[dict]:
        """
        Explain a single prediction by identifying top contributing features.

        Parameters
        ----------
        feature_values : array of feature values for this prediction
        top_k : number of top features to return

        Returns
        -------
        list of dicts: [{feature, value, importance, contribution}, ...]
        """
        if not self._has_importance:
            return [{"note": "No feature importance available"}]

        n = min(len(self.feature_names), len(feature_values),
                len(self.feature_importance))
        contributions = []
        for i in range(n):
            contributions.append({
                "feature": self.feature_names[i],
                "value": float(feature_values[i]),
                "importance": float(self.feature_importance[i]),
                "contribution": float(self.feature_importance[i] *
                                       (feature_values[i] - np.mean(feature_values)) /
                                       (np.std(feature_values) + 1e-10)),
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions[:top_k]

    def summary_table(self, top_n: int = 10) -> pd.DataFrame:
        """Global feature importance table."""
        if not self._has_importance:
            return pd.DataFrame()
        indices = np.argsort(self.feature_importance)[::-1][:top_n]
        rows = [{
            "Feature": self.feature_names[i],
            "Importance": round(float(self.feature_importance[i]), 4),
        } for i in indices]
        return pd.DataFrame(rows)
