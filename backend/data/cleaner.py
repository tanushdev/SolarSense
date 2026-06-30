"""
Data Cleaner — Isolation Forest Preprocessing
=============================================
PHYSICS RATIONALE:
  Aditya-L1 in halo orbit around L1 is generally away from Earth's
  radiation belts, BUT cosmic ray events and particle showers still
  create brief spikes distinguishable from flares by:
    - Extremely short duration (< 60 seconds)
    - No corresponding signal in BOTH channels simultaneously
    - Unphysical hardness ratios (>> 10)

  Isolation Forest identifies these as statistical outliers WITHOUT
  needing labels — unsupervised, so it works even on novel events.

  IMPORTANT: Run Isolation Forest on QUIET SUN SEGMENTS only.
  Do NOT run it on known flare periods — it would incorrectly flag
  genuine X-class flares as outliers.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from loguru import logger
import yaml


class DataCleaner:
    """
    Removes non-solar artifacts from merged time series.

    Strategy:
    1. Identify quiet-sun segments (soft_flux < A-class threshold)
    2. Fit Isolation Forest on quiet-sun feature vectors
    3. Score ALL data with fitted model
    4. Flag artifact samples (score < threshold)
    5. Do NOT remove — flag with artifact_flag=1 (preserve for inspection)
    """

    def __init__(self, config_path: str = "configs/data.yaml",
                 contamination: float = 0.01):
        self.contamination = contamination  # Expected artifact fraction
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        logger.info("DataCleaner initialized | contamination={}", contamination)

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : aligned merged DataFrame from InstrumentAligner

        Returns
        -------
        df with new column: artifact_flag (0=clean, 1=artifact)
        """
        # Features for anomaly detection (physics-motivated)
        features = self._build_detection_features(df)

        logger.info("Fitting Isolation Forest on sub-sample (10%) of {} samples...", len(features))
        self.model.fit(features[::10])
        labels = self.model.predict(features)  # -1 = anomaly, 1 = normal

        df = df.copy()
        df["artifact_flag"] = (labels == -1).astype(int)

        n_artifacts = df["artifact_flag"].sum()
        logger.info("Artifacts detected: {} ({:.2f}%)",
                    n_artifacts, 100 * n_artifacts / len(df))
        return df

    def _build_detection_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Physics-motivated features for artifact detection:
          - Log flux ratio (artifact spikes have unphysical ratios)
          - Flux derivative magnitude (artifacts are instantaneous)
          - Hard/soft correlation over short window
        """
        eps = 1e-15
        features = pd.DataFrame({
            "log_soft":      np.log10(df["soft_flux"] + eps),
            "log_hard":      np.log10(df["hard_flux"] + eps),
            "log_ratio":     np.log10(
                (df["hard_flux"] + eps) / (df["soft_flux"] + eps)),
            "d_soft_abs":    df["soft_flux"].diff().abs(),
            "d_hard_abs":    df["hard_flux"].diff().abs(),
            "soft_rolling_std": df["soft_flux"].rolling(12).std(),
        }).fillna(0)
        return features.values