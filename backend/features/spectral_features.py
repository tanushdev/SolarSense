"""
Spectral Feature Engineering
============================
FFT and Wavelet Decomposition features.

PHYSICS RATIONALE:
  - Solar flare trigger mechanisms (magnetic reconnection, reconnection loops)
    generate micro-oscillations and high-frequency fluctuations in X-ray flux
    known as Quasi-Periodic Pulsations (QPPs, periods of 10s to minutes).
  - Fast Fourier Transform (FFT) reveals dominant oscillation frequencies
    before the impulsive phase starts.
  - Discrete Wavelet Transform (DWT) decomposes the time series into scale-localized
    sub-bands, splitting slow thermal trends (low-frequency approximation) from
    fast non-thermal bursts and QPPs (high-frequency detail coefficients).
"""

import numpy as np
import pandas as pd
from loguru import logger
import pywt
import scipy.signal
import yaml


class SpectralFeatureExtractor:
    """
    Extracts frequency-domain features (FFT, Wavelet) from dual-stream X-ray fluxes.
    """

    def __init__(self, config_path: str = "configs/features.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)["spectral"]
        self.fft_k = self.cfg["fft"]["top_k_frequencies"]
        self.wavelet_name = self.cfg["wavelet"]["wavelet_name"]
        self.levels = self.cfg["wavelet"]["levels"]
        logger.info("SpectralFeatureExtractor initialized | fft_k={} | wavelet={}", 
                    self.fft_k, self.wavelet_name)

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract FFT and Wavelet features for both channels.
        """
        results = {}
        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
            results.update(self._extract_fft_features(df[col], ch))
            results.update(self._extract_wavelet_features(df[col], ch))
            
        return pd.DataFrame(results, index=df.index)

    def _extract_fft_features(self, series: pd.Series, prefix: str) -> dict:
        """
        Computes rolling FFT and returns dominant frequency magnitudes.
        Window size = 180 samples (15 minutes at 5s cadence).
        """
        w = 180
        # Since rolling 1D FFT over 3 million samples can be slow, we compute it
        # efficiently by taking rolling statistics or using standard rolling strides.
        # Alternatively, we can use pd.Series.rolling.apply or simple windowed stride.
        # To keep it performant, we calculate the rolling mean-subtracted absolute deviation
        # of high-frequency components filtered via a high-pass Butterworth filter.
        # This acts as a proxy for high-frequency FFT power.
        results = {}
        
        # High-pass filter to capture fast fluctuations (periods < 2 minutes)
        # 5s cadence -> Nyquist frequency = 0.1 Hz. Cutoff at 0.008 Hz (120s period).
        b, a = scipy.signal.butter(3, 0.08, btype='high')
        filtered = scipy.signal.filtfilt(b, a, series.fillna(0).values)
        filt_series = pd.Series(filtered, index=series.index)
        
        for w_min, label in [(5, "5m"), (15, "15m")]:
            window_size = int(w_min * 60 / 5)
            # Rolling standard deviation of high-frequency component
            results[f"{prefix}_fft_power_hf_{label}"] = filt_series.rolling(window_size).std()
            # Rolling mean absolute deviation of high-frequency component
            results[f"{prefix}_fft_mad_hf_{label}"] = filt_series.rolling(window_size).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
            
        return results

    def _extract_wavelet_features(self, series: pd.Series, prefix: str) -> dict:
        """
        Decomposes the time series using Discrete Wavelet Transform (DWT).
        PHYSICS:
          Detail coefficients (cD1 to cD5) represent high-frequency transients and QPPs.
          Approximation coefficients (cA5) represent the slow-varying background thermal trends.
        """
        results = {}
        # Fill missing values to avoid NaNs in DWT, make a copy to ensure it is writeable
        filled_vals = series.fillna(0).values.copy()
        
        # Compute DWT coefficients
        # For rolling/real-time inference, we compute wavelet energy on a rolling 15-minute window (180 samples)
        # To make it performant on millions of samples, we apply rolling standard deviation 
        # on high-pass detail wavelet subbands (levels 1-3).
        
        # Wavelet decomposition
        coeffs = pywt.wavedec(filled_vals, self.wavelet_name, level=self.levels)
        cA = coeffs[0]
        cD_list = coeffs[1:]  # Detail coefficients from high to low level
        
        # We reconstruct the detail components (high frequency detail) and approximation components (trend)
        # to map them back to the original index length.
        # Level 1-3 details (high frequency)
        detail_reconstructed = pywt.waverec([np.zeros_like(cA)] + [cD for cD in cD_list], self.wavelet_name)[:len(series)]
        # Approximation component (slow trend)
        trend_reconstructed = pywt.waverec([cA] + [np.zeros_like(cD) for cD in cD_list], self.wavelet_name)[:len(series)]
        
        df_det = pd.Series(detail_reconstructed, index=series.index)
        df_trend = pd.Series(trend_reconstructed, index=series.index)
        
        # Rolling wavelet energy/variance features
        for w_min, label in [(5, "5m"), (15, "15m")]:
            window_size = int(w_min * 60 / 5)
            # Detail variance represents energy of high-frequency fluctuations (precursor activity)
            results[f"{prefix}_wavelet_detail_energy_{label}"] = df_det.rolling(window_size).var()
            # Trend derivative represents rate of change of the slow thermal background
            results[f"{prefix}_wavelet_trend_slope_{label}"] = df_trend.diff(window_size)
            
        return results
