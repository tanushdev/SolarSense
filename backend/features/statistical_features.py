"""
Statistical Feature Engineering
================================
Statistical/moment-based features from X-ray flux distributions:
  - Higher-order moments (skewness, kurtosis) over multiple windows
  - Quantile ranges (IQR, 90-10 spread)
  - Autocorrelation at various lags
  - Partial autocorrelation
  - Hurst exponent (long-range dependence)

PHYSICS RATIONALE:
  The waiting-time distribution of solar flares follows a power-law,
  indicating a self-organized critical (SOC) state. As the coronal
  magnetic field approaches instability, the fluctuation statistics
  of X-ray flux deviate from Gaussian — increased kurtosis signals
  intermittent burst activity before visible flare onset.
"""

import numpy as np
import pandas as pd
from scipy import signal
from loguru import logger


class StatisticalFeatureExtractor:
    """
    Extracts statistical features from aligned X-ray flux data.
    """

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all statistical features.

        Parameters
        ----------
        df : DataFrame with [soft_flux, hard_flux] columns
        """
        features = {}
        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
            if col not in df.columns:
                continue
            features.update(self._quantile_features(df[col], ch))
            features.update(self._autocorrelation_features(df[col], ch))
            features.update(self._hurst_exponent(df[col], ch))
        return pd.DataFrame(features, index=df.index)

    def _quantile_features(self, series: pd.Series, prefix: str) -> dict:
        """
        Rolling quantile-based features.

        IQR and interquantile ratios are more robust to outliers
        than raw moments. The interquantile range widening is a
        known pre-flare indicator (increased variability).
        """
        results = {}
        windows = [(60, "5m"), (180, "15m"), (720, "1h")]
        for w, label in windows:
            rolled = series.rolling(w)
            q10 = rolled.quantile(0.10)
            q50 = rolled.quantile(0.50)
            q90 = rolled.quantile(0.90)
            results[f"{prefix}_iqr_{label}"]       = q90 - q10
            results[f"{prefix}_q90_q10_ratio_{label}"] = q90 / (q10 + 1e-10)
            results[f"{prefix}_q50_{label}"]       = q50
            # Coefficient of variation (std/mean) — normalized variability
            results[f"{prefix}_cv_{label}"]        = rolled.std() / (rolled.mean() + 1e-10)
        return results

    def _autocorrelation_features(self, series: pd.Series, prefix: str) -> dict:
        """
        Rolling autocorrelation at various lags.

        PHYSICS: During quiet Sun, X-ray flux is dominated by Poisson
        noise (low autocorrelation). As a flare builds up, the flux
        becomes increasingly correlated (rising trend → high autocorrelation
        at short lags). Decreasing autocorrelation at longer lags may
        signal the onset of turbulent (intermittent) processes.
        """
        results = {}
        window = 180  # 15 min
        for lag, lag_label in [(1, "lag1"), (6, "lag30s"), (12, "lag1m"), (60, "lag5m")]:
            corr = series.rolling(window).corr(series.shift(lag))
            results[f"{prefix}_acf_{lag_label}"] = corr
        return results

    def _hurst_exponent(self, series: pd.Series, prefix: str) -> dict:
        """
        Rolling Hurst exponent estimate using rescaled range (R/S) analysis.

        H = 0.5  → Brownian motion (random walk, no trend)
        H > 0.5  → persistent trending behavior (positive autocorrelation)
        H < 0.5  → mean-reverting (anti-persistent)

        PHYSICS:
          Quiet corona: H ~ 0.5 (stochastic)
          Pre-flare buildup: H > 0.6 (trending — flux rises systematically)
          Flaring: H ~ 0.8+ (strong persistence)
          Post-flare: H < 0.5 (mean reversion as flux decays)

        NOTE: This is a simplified R/S estimator. Full R/S analysis
        is O(n^2). Here we use a rolling window approximation.
        """
        results = {}
        w = 720  # 1-hour window for stability
        def _rs_hurst(x):
            x = np.asarray(x, dtype=np.float64)
            n = len(x)
            if n < 10 or np.std(x) == 0:
                return 0.5
            mean = np.mean(x)
            y = x - mean
            z = np.cumsum(y)
            r = np.max(z) - np.min(z)
            s = np.std(x, ddof=1)
            h = np.log(r / (s + 1e-10)) / np.log(n / 2.0)
            return np.clip(h, 0.0, 1.0)

        results[f"{prefix}_hurst_1h"] = series.rolling(w).apply(_rs_hurst, raw=True)
        return results
