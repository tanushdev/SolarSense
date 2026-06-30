"""
HEL1OS FITS Reader
==================
HEL1OS measures hard X-rays in 10–150 keV.

Physical context:
  Hard X-rays (10–150 keV) from HEL1OS trace non-thermal bremsstrahlung
  from energetic electrons accelerated during the impulsive phase.
  CRITICAL PHYSICS: Hard X-rays PEAK BEFORE soft X-rays.
  This is the Neupert Effect — HXR precedes SXR peak by 1–10 minutes.
  This precursor is the primary physical basis for forecasting.
  An algorithm that detects HXR rise before SXR rise gains lead time.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from pathlib import Path
from loguru import logger
from typing import Union, List, Optional
import yaml


class HEL1OSReader:
    """
    Reads and parses HEL1OS Level-1 FITS files from ISRO ISSDC PRADAN.

    Key difference from SoLEXSReader:
      - Energy band: 10–150 keV (hard X-rays, non-thermal)
      - Output column: hard_flux instead of soft_flux
      - Physics: HXR peaks 1–10 minutes before SXR peak (Neupert Effect)
    """

    REQUIRED_COLUMNS = ["TIME", "FLUX", "COUNTS"]
    QUALITY_GOOD = 0

    def __init__(self, config_path: str = "configs/data.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["hel1os"]
        self.hdu_ext = self.config["fits_extension"]
        self.resample = f"{self.config['resample_seconds']}s"
        logger.info("HEL1OSReader initialized | resample={}", self.resample)

    def read_file(self, fits_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read a single HEL1OS FITS file.

        Parameters
        ----------
        fits_path : path to .fits file

        Returns
        -------
        pd.DataFrame with columns:
            timestamp_utc   : datetime64[ns, UTC]
            hard_flux       : float64  (counts/s/keV, background included)
            hard_counts     : float64  (raw detector counts)
            quality_flag    : int      (0=good)
            source_file     : str      (origin filename for traceability)
        """
        fits_path = Path(fits_path)
        logger.info("Reading HEL1OS FITS: {}", fits_path.name)

        base_time = None
        try:
            with fits.open(fits_path) as hdul:
                self._validate_header(hdul[0].header, fits_path.name)
                table = hdul[self.hdu_ext].data
                df = self._table_to_dataframe(table, fits_path.name)
                
                # Get start time from primary header
                if "ISOSTART" in hdul[0].header:
                    base_time = pd.to_datetime(hdul[0].header["ISOSTART"], utc=True)
                elif "DATE_OBS" in hdul[self.hdu_ext].header:
                    date_str = hdul[self.hdu_ext].header["DATE_OBS"]
                    time_str = hdul[self.hdu_ext].header.get("TIME_OBS", "00:00:00")
                    base_time = pd.to_datetime(f"{date_str}T{time_str}", utc=True)
        except Exception as e:
            logger.error("Failed to read {}: {}", fits_path.name, e)
            raise

        df = self._convert_timestamps(df, base_time)
        df = self._resample(df)
        df = self._validate_flux(df)
        logger.info("HEL1OS file loaded: {} rows, {} to {}",
                    len(df), df.index[0], df.index[-1])
        return df

    def read_directory(self, dir_path: Union[str, Path],
                       sort: bool = True) -> pd.DataFrame:
        """
        Read all FITS files in a directory, concatenate chronologically.
        """
        dir_path = Path(dir_path)
        # Only read lightcurve files — skip evt.fits (137 MB each), gti, hk, spectra
        files = sorted(dir_path.rglob("lightcurve_cdte*.fits")) if sort else list(dir_path.rglob("lightcurve_cdte*.fits"))
            
        if not files:
            raise FileNotFoundError(f"No valid HEL1OS FITS files found in {dir_path}")
            
        logger.info("Reading {} HEL1OS files from {}", len(files), dir_path)
        dfs = []
        for f in files:
            try:
                dfs.append(self.read_file(f))
            except Exception as e:
                logger.warning("Skipping file {}: {}", f.name, e)
                
        if not dfs:
            raise ValueError(f"Failed to read any valid files in {dir_path}")
            
        merged = pd.concat(dfs).sort_index()
        # Drop duplicate timestamps by keeping the first occurrence
        merged = merged[~merged.index.duplicated(keep="first")]
        logger.info("HEL1OS directory loaded: {} total rows", len(merged))
        return merged

    def _validate_header(self, header: fits.Header, fname: str):
        """Check mandatory FITS keywords."""
        required_keys = ["TELESCOP", "INSTRUME", "DATE-OBS"]
        missing = [k for k in required_keys if k not in header]
        if missing:
            logger.warning("Missing FITS keywords {} in {}", missing, fname)

    def _table_to_dataframe(self, table, fname: str) -> pd.DataFrame:
        """Convert FITS binary table to DataFrame."""
        col_map = self.config["columns"]
        
        # Check table columns. HEL1OS files use different column layouts:
        #   lightcurve_cdte/czt: [MJD, ISOT, CTR, STAT_ERR]
        #   evt:                [mjd, hlsobt, currtemp, chn, ener, recnum, utc-isot]
        #   spectra:            complex multi-extension tables
        # Let's map dynamically to match table layout.
        time_col = None
        for candidate in [col_map["time"], "TIME", "MJD", "mjd", "TSTART", "SPEC_NUM"]:
            if candidate in table.names:
                time_col = candidate
                break
        if not time_col:
            raise KeyError(f"No valid time column found in table columns: {table.names}")

        flux_col = None
        for candidate in [col_map.get("flux", "FLUX"), "FLUX", "CTR", "COUNTS", "SPEC_NUM", "ener"]:
            if candidate in table.names:
                flux_col = candidate
                break
                
        counts_col = None
        for candidate in [col_map.get("counts", "COUNTS"), "COUNTS", "CTR", "SPEC_NUM"]:
            if candidate in table.names:
                counts_col = candidate
                break

        # If a column is a multi-dimensional array (e.g. spectrum channel arrays),
        # take the mean or sum across channels to get a single flux time-series value.
        time_data = table[time_col]
        flux_data = table[flux_col]
        counts_data = table[counts_col] if counts_col else flux_data

        if hasattr(flux_data, 'shape') and len(flux_data.shape) > 1:
            flux_data = np.mean(flux_data, axis=1)
        if hasattr(counts_data, 'shape') and len(counts_data.shape) > 1:
            counts_data = np.sum(counts_data, axis=1)

        df = pd.DataFrame({
            "time_raw":     time_data,
            "hard_flux":    np.asarray(flux_data, dtype=np.float64),
            "hard_counts":  np.asarray(counts_data, dtype=np.float64),
            "quality_flag": np.asarray(table["QUALITY"], dtype=int) if "QUALITY" in table.names else 0,
            "source_file":  fname,
        })
        return df

    def _convert_timestamps(self, df: pd.DataFrame, base_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Convert time column to UTC DatetimeIndex."""
        try:
            # If the column is MJD, convert from Modified Julian Date
            val = df["time_raw"].values
            if len(val) > 0 and val[0] > 50000 and val[0] < 80000:
                t = Time(val, format="mjd", scale="utc")
                df.index = pd.DatetimeIndex(t.to_datetime(), tz="UTC",
                                            name="timestamp_utc")
            elif base_time is not None:
                # Add relative seconds to the base time
                df.index = base_time + pd.to_timedelta(val, unit="s")
            else:
                # Fallback to standard conversion
                df.index = pd.to_datetime(val, utc=True)
        except Exception:
            df.index = pd.to_datetime(df["time_raw"], utc=True)
        df = df.drop(columns=["time_raw"])
        return df

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to uniform cadence."""
        numeric_cols = ["hard_flux", "hard_counts"]
        df_rs = df[numeric_cols].resample(self.resample).mean()
        df_rs["quality_flag"] = df["quality_flag"].resample(
            self.resample).max()
        df_rs["source_file"] = df["source_file"].resample(
            self.resample).first()
        return df_rs

    def _validate_flux(self, df: pd.DataFrame) -> pd.DataFrame:
        """Physical sanity checks on flux values."""
        FLUX_MIN = 1e-3
        FLUX_MAX = 1e8
        n_bad = ((df["hard_flux"] < FLUX_MIN) |
                 (df["hard_flux"] > FLUX_MAX)).sum()
        if n_bad > 0:
            logger.warning("HEL1OS: {} physically implausible flux values flagged", n_bad)
            df.loc[(df["hard_flux"] < FLUX_MIN) |
                   (df["hard_flux"] > FLUX_MAX), "quality_flag"] = 3
        return df