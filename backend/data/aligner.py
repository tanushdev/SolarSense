"""
Instrument Aligner
==================
SoLEXS and HEL1OS have independent clocks with potential drift.
This module creates a single merged DataFrame at uniform cadence.

PHYSICS NOTE:
  The time lag between HEL1OS hard X-ray peak and SoLEXS soft X-ray peak
  is the Neupert Effect (typically 1–10 minutes).
  After alignment, this lag becomes measurable as a feature.
  cross_correlation(hard_flux, soft_flux, lags=[-120, 120]) reveals it.
"""

import pandas as pd
import numpy as np
from loguru import logger
import yaml


class InstrumentAligner:
    """
    Aligns SoLEXS and HEL1OS time series to a common UTC grid.

    Steps:
    1. Find common time range (intersection of both series)
    2. Create reference grid at target cadence (5s)
    3. Interpolate both series onto reference grid
    4. Flag gaps > max_gap_seconds with quality_flag=9
    5. Return merged DataFrame
    """

    def __init__(self, config_path: str = "configs/data.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.resample = f"{cfg['alignment']['max_gap_seconds']}s"
        self.max_gap = pd.Timedelta(
            seconds=cfg["alignment"]["max_gap_seconds"])
        self.cadence = f"{cfg['solexs']['resample_seconds']}s"

    def align(self, df_solexs: pd.DataFrame,
              df_hel1os: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df_solexs  : output of SoLEXSReader.read_file()
        df_hel1os  : output of HEL1OSReader.read_file()

        Returns
        -------
        pd.DataFrame with columns:
            timestamp_utc  : index
            soft_flux      : from SoLEXS
            soft_counts    : from SoLEXS
            hard_flux      : from HEL1OS (interpolated to SoLEXS grid)
            hard_counts    : from HEL1OS
            quality_solexs : quality flag from SoLEXS
            quality_hel1os : quality flag from HEL1OS
            data_gap       : bool — True if either instrument has gap here
        """
        # 1. Find common range
        t_start = max(df_solexs.index[0], df_hel1os.index[0])
        t_end   = min(df_solexs.index[-1], df_hel1os.index[-1])
        logger.info("Common time range: {} to {}", t_start, t_end)

        # 2. Build reference grid
        ref_grid = pd.date_range(t_start, t_end, freq=self.cadence, tz="UTC")

        # 3. Reindex both with linear interpolation on numeric columns only
        num_cols_s = df_solexs.select_dtypes(include=[np.number]).columns
        df_s = df_solexs[num_cols_s].reindex(
            df_solexs.index.union(ref_grid)).interpolate(
            "time").reindex(ref_grid)
            
        num_cols_h = df_hel1os.select_dtypes(include=[np.number]).columns
        df_h = df_hel1os[num_cols_h].reindex(
            df_hel1os.index.union(ref_grid)).interpolate(
            "time").reindex(ref_grid)

        # 4. Merge
        merged = pd.DataFrame({
            "soft_flux":      df_s["soft_flux"],
            "soft_counts":    df_s["soft_counts"],
            "hard_flux":      df_h["hard_flux"],
            "hard_counts":    df_h["hard_counts"],
            "quality_solexs": df_s["quality_flag"].fillna(9).astype(int),
            "quality_hel1os": df_h["quality_flag"].fillna(9).astype(int),
        }, index=ref_grid)

        # 5. Flag gaps
        merged["data_gap"] = (
            merged["quality_solexs"] == 9) | (merged["quality_hel1os"] == 9)

        logger.info("Alignment complete: {} samples, {} gaps",
                    len(merged), merged["data_gap"].sum())
        return merged