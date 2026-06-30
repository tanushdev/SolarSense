"""
Probability Calibration for Solar Flare Prediction
====================================================
Implements Platt Scaling and Isotonic Regression to calibrate
raw XGBoost probabilities. Uses ECE to select the best method.

Platt Scaling:    P(y=1|x) = 1 / (1 + exp(A * raw + B))
Isotonic:         Non-parametric monotone fit (more flexible)

Both methods are fit on held-out validation data (never test).
The method with lower ECE is deployed automatically.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from loguru import logger
from pathlib import Path
import pickle


class Calibrator:
    """Fit, evaluate, and deploy probability calibration."""

    def __init__(self):
        self.platt_model = None
        self.isotonic_model = None
        self.chosen_method = "none"
        self._deployed = None

    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """
        Fit both Platt and Isotonic calibration.

        Parameters
        ----------
        y_true : (N,) binary labels
        y_pred_proba : (N,) raw model probabilities
        """
        # Platt Scaling — logistic regression on log-odds
        log_odds = np.clip(np.log(y_pred_proba + 1e-15) -
                           np.log(1 - y_pred_proba + 1e-15), -20, 20).reshape(-1, 1)
        self.platt_model = LogisticRegression(C=1e10, solver="lbfgs")
        self.platt_model.fit(log_odds, y_true)

        # Isotonic Regression
        self.isotonic_model = IsotonicRegression(out_of_bounds="clip")
        self.isotonic_model.fit(y_pred_proba, y_true)

        logger.info("Calibrator: Platt + Isotonic fitted on {} samples", len(y_true))

    def predict_proba(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply deployed calibration method."""
        if self._deployed is not None:
            return self._deployed(raw_proba)
        return raw_proba

    def _deploy_platt(self, raw_proba: np.ndarray) -> np.ndarray:
        log_odds = np.clip(np.log(raw_proba + 1e-15) -
                           np.log(1 - raw_proba + 1e-15), -20, 20).reshape(-1, 1)
        return self.platt_model.predict_proba(log_odds)[:, 1]

    def _deploy_isotonic(self, raw_proba: np.ndarray) -> np.ndarray:
        return self.isotonic_model.transform(raw_proba)

    def evaluate(self, y_true: np.ndarray,
                 y_pred_proba: np.ndarray) -> dict:
        """
        Compare Platt vs Isotonic on validation data.

        Returns dict with ECE and Brier for both methods plus winner.
        """
        def _ece(y_true, y_prob, n_bins=10):
            bins = np.linspace(0, 1, n_bins + 1)
            idx = np.digitize(y_prob, bins, right=True) - 1
            ece = 0.0
            for i in range(n_bins):
                mask = idx == i
                if mask.sum() == 0:
                    continue
                ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
            return ece / len(y_true)

        def _brier(y_true, y_prob):
            return np.mean((y_prob - y_true) ** 2)

        platt_probs = self._deploy_platt(y_pred_proba)
        iso_probs = self._deploy_isotonic(y_pred_proba)

        results = {
            "raw": {"ece": _ece(y_true, y_pred_proba), "brier": _brier(y_true, y_pred_proba)},
            "platt": {"ece": _ece(y_true, platt_probs), "brier": _brier(y_true, platt_probs)},
            "isotonic": {"ece": _ece(y_true, iso_probs), "brier": _brier(y_true, iso_probs)},
        }

        # Choose method with lowest ECE
        raw_ece = results["raw"]["ece"]
        platt_ece = results["platt"]["ece"]
        iso_ece = results["isotonic"]["ece"]

        if platt_ece <= iso_ece and platt_ece < raw_ece:
            self.chosen_method = "platt"
            self._deployed = self._deploy_platt
        elif iso_ece < raw_ece:
            self.chosen_method = "isotonic"
            self._deployed = self._deploy_isotonic
        else:
            self.chosen_method = "none"
            self._deployed = None

        results["chosen_method"] = self.chosen_method
        logger.info("Calibration: raw_ece={:.4f}, platt_ece={:.4f}, iso_ece={:.4f}, chosen={}",
                    raw_ece, platt_ece, iso_ece, self.chosen_method)
        return results

    def save(self, path: str = "models/checkpoints/calibrator.pkl"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "platt": self.platt_model,
                "isotonic": self.isotonic_model,
                "chosen_method": self.chosen_method,
            }, f)
        logger.info("Calibrator saved to {}", path)

    def load(self, path: str = "models/checkpoints/calibrator.pkl"):
        p = Path(path)
        if not p.exists():
            logger.warning("Calibrator not found at {}", path)
            return
        with open(p, "rb") as f:
            data = pickle.load(f)
        self.platt_model = data["platt"]
        self.isotonic_model = data["isotonic"]
        self.chosen_method = data["chosen_method"]
        if self.chosen_method == "platt":
            self._deployed = self._deploy_platt
        elif self.chosen_method == "isotonic":
            self._deployed = self._deploy_isotonic
        logger.info("Calibrator loaded from {} (method: {})", path, self.chosen_method)
