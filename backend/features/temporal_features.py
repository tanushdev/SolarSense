"""
Temporal Feature Engineering
============================
Features derived from time structure of X-ray flux:
  - Time since last event / flare
  - Time to next prediction horizon
  - Day/night, orbital phase indicators
  - Seasonal indicators (solar cycle phase proxy)
  - Running event count / event rate

PHYSICS RATIONALE:
  Flare occurrence is not random — it clusters in active regions that
  rotate across the solar disk. The time since the last flare in a given
  region is a known predictor (the "loading-unloading" model).
  The Carrington rotation period (~27 days) creates periodic modulation.
"""

import numpy as np
import pandas as pd
from loguru import logger


class TemporalFeatureExtractor:
    """
    Extracts temporal/calendar features from aligned X-ray data.
    """

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all temporal features.

        Parameters
        ----------
        df : DataFrame with DatetimeIndex
        """
        features = {}
        features.update(self._time_of_day_indicators(df))
        features.update(self._event_accumulation(df))
        features.update(self._running_event_rate(df))
        return pd.DataFrame(features, index=df.index)

    def _time_of_day_indicators(self, df: pd.DataFrame) -> dict:
        """
        Encode time-of-day and day-of-year as cyclical features.

        PHYSICS: Solar active region visibility varies with Earth's rotation,
        creating diurnal modulation in observed flux. Also, the ~27-day
        Carrington rotation modulates flare rates due to active region
        passage across the visible disk.
        """
        idx = df.index
        # Hour of day (0-23) → sin/cos encoding
        hour_angle = 2 * np.pi * idx.hour / 23.0
        # Day of year (0-365) → sin/cos encoding
        doy_angle = 2 * np.pi * idx.dayofyear / 365.0

        return {
            "hour_sin":        np.sin(hour_angle),
            "hour_cos":        np.cos(hour_angle),
            "doy_sin":         np.sin(doy_angle),
            "doy_cos":         np.cos(doy_angle),
        }

    def _event_accumulation(self, df: pd.DataFrame) -> dict:
        """
        Running count of flare events over lookback windows.

        Also tracks whether flux is currently elevated relative to
        the recent baseline (indicating possible ongoing event).
        """
        results = {}
        # Use hard_flux derivative sign changes as simple event markers
        if "hard_flux" in df.columns:
            hard = df["hard_flux"]
            # Binary: is current value above a sliding threshold?
            threshold = hard.rolling(720).mean() + 2 * hard.rolling(720).std()
            results["flux_above_2sigma"] = (hard > threshold).astype(float)

        for w_min, label in [(15, "15m"), (60, "1h"), (360, "6h"), (1440, "24h")]:
            w = int(w_min * 60 / 5)
            if "hard_flux" in df.columns:
                results[f"hard_flux_accum_{label}"] = df["hard_flux"].rolling(w).sum()
                results[f"hard_flux_mean_{label}"] = df["hard_flux"].rolling(w).mean()
            if "soft_flux" in df.columns:
                results[f"soft_flux_accum_{label}"] = df["soft_flux"].rolling(w).sum()
        return results

    def _running_event_rate(self, df: pd.DataFrame) -> dict:
        """
        Running count of threshold-crossing events as a proxy for
        flare event rate.

        An event is counted when hard_flux exceeds 3-sigma above the
        rolling mean in a sustained manner.
        """
        results = {}
        if "hard_flux" not in df.columns:
            return {}
        hard = df["hard_flux"]
        mean_3h = hard.rolling(2160).mean()
        std_3h = hard.rolling(2160).std()
        threshold = mean_3h + 3 * std_3h
        above = (hard > threshold).astype(int)
        # Count transitions (event starts) over various windows
        for w_min, label in [(60, "1h"), (360, "6h"), (1440, "24h")]:
            w = int(w_min * 60 / 5)
            results[f"event_rate_{label}"] = above.rolling(w).sum() / (w_min / 60.0)
        return results
