"""
Physics Feature Engineering
============================
Every feature below has:
  1. Physical justification (why it helps forecast flares)
  2. Formula
  3. Expected behavior before/during flares
  4. Implementation

PHYSICS PRIMER:
  Solar flares have a two-phase structure in X-rays:
  
  IMPULSIVE PHASE (hard X-rays peak here)
    - Duration: seconds to minutes
    - HEL1OS hard X-ray flux rises rapidly
    - Non-thermal electron bremsstrahlung
    - d(HXR)/dt is large and positive
    
  GRADUAL PHASE (soft X-rays peak here)
    - Duration: minutes to hours
    - SoLEXS soft X-ray flux peaks AFTER hard X-rays (Neupert Effect)
    - Thermal emission from heated plasma
    - d(SXR)/dt follows integral of HXR (Neupert relation)

  PRECURSOR SIGNATURES (before any flare is visible):
    - Subtle HXR rise (< 3-sigma) appearing 5–30 min before peak
    - Hardness ratio begins drifting upward
    - Cross-correlation lag between channels decreases
    - Matrix Profile discord score increases
"""

import numpy as np
import pandas as pd
from loguru import logger
import yaml


class PhysicsFeatureExtractor:
    """
    Extracts physics-grounded features from aligned SoLEXS + HEL1OS data.
    
    All features are computed at each time step and returned as a
    DataFrame aligned to the input index.
    """

    def __init__(self, config_path: str = "configs/features.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.eps = float(self.cfg["physics"]["hardness_ratio"]["epsilon"])

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all enabled feature extractors.
        
        Input: aligned DataFrame with [soft_flux, hard_flux, soft_counts, hard_counts]
        Output: DataFrame with all physics features appended
        """
        features = {}
        features.update(self._hardness_ratio(df))
        features.update(self._flux_derivatives(df))
        features.update(self._spectral_index(df))
        features.update(self._rise_decay_rates(df))
        features.update(self._rolling_statistics(df))
        features.update(self._cross_correlation_features(df))
        features.update(self._background_subtracted(df))
        features.update(self._neupert_proxy(df))
        return pd.DataFrame(features, index=df.index)

    # ─── FEATURE 1: Hardness Ratio ─────────────────────────────────────────
    def _hardness_ratio(self, df: pd.DataFrame) -> dict:
        """
        Hardness Ratio = hard_flux / soft_flux

        PHYSICS: Measures spectral hardness of X-ray emission.
          - Quiet sun:  HR ≈ 0.01–0.1 (soft thermal emission dominates)
          - Pre-flare:  HR starts increasing as non-thermal component grows
          - Flare peak: HR spikes (hard X-ray impulsive peak)
          - Post-flare: HR decreases as plasma cools and softens

        This is ONE OF THE MOST IMPORTANT PRECURSOR FEATURES.
        A sustained HR increase of > 20% over 10 minutes preceding a flare
        has been reported in literature.
        """
        hr = df["hard_flux"] / (df["soft_flux"] + self.eps)
        return {
            "hardness_ratio":        hr,
            "hardness_ratio_log":    np.log10(hr + self.eps),
            "hardness_ratio_deriv":  hr.diff(),
            "hardness_ratio_5min":   hr.rolling(60).mean(),   # 60*5s = 5 min
            "hardness_ratio_15min":  hr.rolling(180).mean(),
        }

    # ─── FEATURE 2: Flux Derivatives ───────────────────────────────────────
    def _flux_derivatives(self, df: pd.DataFrame) -> dict:
        """
        dF/dt and d²F/dt² for both channels.

        PHYSICS:
          - dF_hard/dt > 0 is the primary impulsive phase indicator
          - d²F_soft/dt² > 0 before the SXR peak indicates acceleration
          - Neupert: d(SXR)/dt ∝ HXR (integral relation)
          - A positive d²(SXR)/dt² 10–20 min before SXR peak is a
            forecast-relevant precursor

        Smoothing BEFORE differencing is critical — raw derivatives
        on noisy photon counting data are meaningless.
        """
        smooth_w = self.cfg["physics"]["flux_derivative"]["smoothing_window"]

        results = {}
        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
            smoothed = df[col].rolling(smooth_w, center=False).mean()
            log_flux = np.log10(smoothed + self.eps)
            results[f"d_{ch}_flux_dt"]     = smoothed.diff()
            results[f"d2_{ch}_flux_dt2"]   = smoothed.diff().diff()
            results[f"d_log_{ch}_flux_dt"] = log_flux.diff()   # relative rate
        return results

    # ─── FEATURE 3: Spectral Index ─────────────────────────────────────────
    def _spectral_index(self, df: pd.DataFrame) -> dict:
        """
        Spectral Index: slope of log(flux) vs log(energy)

        PHYSICS: Power-law spectrum: F(E) ∝ E^(-γ)
          γ = spectral index
          - Thermal (quiet sun): γ ≈ 4–6
          - Non-thermal (impulsive flare): γ ≈ 2–4 (harder spectrum)
          - Decreasing γ signals non-thermal acceleration = flare onset

        Approximated using two-point estimate between
        SoLEXS mean energy (~5 keV) and HEL1OS mean energy (~20 keV).
        """
        E_soft  = 5.0   # keV (approximate SoLEXS band center)
        E_hard  = 20.0  # keV (approximate HEL1OS band center)

        log_F_soft = np.log10(df["soft_flux"] + self.eps)
        log_F_hard = np.log10(df["hard_flux"] + self.eps)
        gamma = (log_F_hard - log_F_soft) / (np.log10(E_hard) - np.log10(E_soft))

        return {
            "spectral_index":        gamma,
            "spectral_index_5min":   gamma.rolling(60).mean(),
            "spectral_index_deriv":  gamma.diff(),
        }

    # ─── FEATURE 4: Rise and Decay Rates ───────────────────────────────────
    def _rise_decay_rates(self, df: pd.DataFrame) -> dict:
        """
        Rise Rate: how fast flux is increasing in recent N minutes
        Decay Rate: how fast flux is decreasing after peak

        PHYSICS:
          - Fast rise (> 1 decade/5 min) = impulsive X-class flare
          - Slow rise (< 1 decade/30 min) = C-class gradual event
          - Rise rate asymmetry between HXR and SXR channels gives
            impulsive fraction → severity proxy
        """
        results = {}
        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
            log_flux = np.log10(df[col] + self.eps)
            for w_min in [5, 15]:
                w = int(w_min * 60 / 5)  # convert minutes to samples
                change = log_flux - log_flux.shift(w)
                results[f"{ch}_rise_rate_{w_min}min"] = change.clip(lower=0)
                results[f"{ch}_decay_rate_{w_min}min"] = change.clip(upper=0).abs()
        return results

    # ─── FEATURE 5: Rolling Statistics ─────────────────────────────────────
    def _rolling_statistics(self, df: pd.DataFrame) -> dict:
        """
        Rolling mean, std, skewness, kurtosis at multiple windows.
        
        PHYSICS: Kurtosis spike in hard X-ray channel is a known
        precursor of impulsive phase onset (Georgoulis et al.).
        """
        results = {}
        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
            for w_min, w_label in [(5, "5m"), (15, "15m"), (60, "60m")]:
                w = int(w_min * 60 / 5)
                rolled = df[col].rolling(w)
                results[f"{ch}_mean_{w_label}"]  = rolled.mean()
                results[f"{ch}_std_{w_label}"]   = rolled.std()
                results[f"{ch}_skew_{w_label}"]  = rolled.skew()
                results[f"{ch}_kurt_{w_label}"]  = rolled.kurt()
        return results

    # ─── FEATURE 6: Cross-Correlation Features ─────────────────────────────
    def _cross_correlation_features(self, df: pd.DataFrame) -> dict:
        """
        Rolling cross-correlation between hard and soft X-ray channels.

        PHYSICS (NEUPERT EFFECT):
          During quiet sun: HXR and SXR are decorrelated (different sources)
          During pre-flare: correlation begins increasing as same energetic
            event starts producing both thermal and non-thermal emission
          During flare: HXR peaks first, SXR peaks 1–10 min later

          The LAG at maximum cross-correlation gives the Neupert delay.
          A decreasing lag over time is an early warning signal.
        """
        w = 180   # 15-minute rolling window
        results = {}
        
        log_soft = np.log10(df["soft_flux"] + self.eps)
        log_hard = np.log10(df["hard_flux"] + self.eps)
        
        # Rolling Pearson correlation (zero-lag)
        results["hxr_sxr_corr_15min"] = log_soft.rolling(w).corr(log_hard)
        
        # Lag-1 sample correlation (HXR leads SXR by 5 seconds?)
        results["hxr_sxr_lag1_corr"] = log_hard.rolling(w).corr(
            log_soft.shift(1))
        
        return results

    # ─── FEATURE 7: Background-Subtracted Flux ─────────────────────────────
    def _background_subtracted(self, df: pd.DataFrame) -> dict:
        """
        Remove pre-flare background to get flare-excess flux.

        Background = rolling 10th percentile over 60 minutes.
        This is the standard method used in GOES processing.
        """
        results = {}
        w = int(60 * 60 / 5)   # 60-minute window in samples
        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
            bg = df[col].rolling(w, center=False).quantile(0.10)
            excess = (df[col] - bg).clip(lower=0)
            results[f"{ch}_flux_excess"]     = excess
            results[f"{ch}_flux_excess_log"] = np.log10(excess + self.eps)
            results[f"{ch}_snr"]             = excess / (bg + self.eps)
        return results

    # ─── FEATURE 8: Neupert Proxy ──────────────────────────────────────────
    def _neupert_proxy(self, df: pd.DataFrame) -> dict:
        """
        Neupert Relation: SXR(t) ∝ ∫₀ᵗ HXR(t') dt'

        The Neupert Effect states: the soft X-ray time profile is approximately
        the time integral of the hard X-ray time profile.
        Deviation from this relation is physically meaningful.

        Neupert_proxy = cumulative sum of HXR (running integral)
        Neupert_residual = SXR - Neupert_proxy (normalized)
        A positive Neupert_residual emerging before SXR peak
        indicates additional thermal heating beyond the impulsive phase.

        This is a novel feature. Validate carefully on known events.
        """
        hard_excess = (df["hard_flux"] -
                       df["hard_flux"].rolling(720).quantile(0.10)).clip(lower=0)
        neupert_proxy = hard_excess.rolling(120).sum()  # 10-min running integral
        neupert_proxy_norm = neupert_proxy / (neupert_proxy.rolling(720).max() + self.eps)

        soft_norm = df["soft_flux"] / (df["soft_flux"].rolling(720).max() + self.eps)
        neupert_residual = soft_norm - neupert_proxy_norm

        return {
            "neupert_proxy":        neupert_proxy_norm,
            "neupert_residual":     neupert_residual,
            "neupert_residual_abs": neupert_residual.abs(),
        }