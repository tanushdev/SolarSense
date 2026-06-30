"""
SoLEXS FITS Reader
==================
Reads ISRO Aditya-L1 SoLEXS Level-1 FITS files.

SoLEXS measures soft X-rays in 1.6–12.0 keV.
Level-1 FITS structure (from ISRO documentation):
  - HDU 0: Primary header (mission metadata)
  - HDU 1: Binary table (TIME, FLUX, COUNTS, ENERGY columns)

Physical context:
  Soft X-rays (1.6–12 keV) from SoLEXS trace the thermal emission from
  hot coronal plasma (T > 10^7 K). During a flare, soft X-ray flux rises
  on timescales of minutes (the gradual phase) and is the standard
  classification band (GOES 1-8 Angstrom equivalent).
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from pathlib import Path
from loguru import logger
from typing import Union, List, Optional
import yaml


class SoLEXSReader:
    """
    Reads and parses SoLEXS Level-1 FITS files from ISRO ISSDC PRADAN.

    Usage:
        reader = SoLEXSReader(config_path="configs/data.yaml")
        df = reader.read_file("dataset/raw/solexs/solexs_20240101.fits")
        df_merged = reader.read_directory("dataset/raw/solexs/")
    """

    REQUIRED_COLUMNS = ["TIME", "FLUX", "COUNTS"]
    QUALITY_GOOD = 0

    def __init__(self, config_path: str = "configs/data.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["solexs"]
        self.hdu_ext = self.config["fits_extension"]
        self.resample = f"{self.config['resample_seconds']}s"
        logger.info("SoLEXSReader initialized | resample={}", self.resample)

    @staticmethod
    def _open_fits(path: Path):
        """Open .fits or .fits.gz or .lc.gz files."""
        return fits.open(str(path))

    def read_file(self, fits_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read a single SoLEXS FITS file.

        Parameters
        ----------
        fits_path : path to .fits file

        Returns
        -------
        pd.DataFrame with columns:
            timestamp_utc   : datetime64[ns, UTC]
            soft_flux       : float64  (W/m²/keV, background included)
            soft_counts     : float64  (raw detector counts)
            energy_mean     : float64  (keV)
            quality_flag    : int      (0=good, 1=particle event, 2=eclipse)
            source_file     : str      (origin filename for traceability)
        """
        fits_path = Path(fits_path)
        logger.info("Reading SoLEXS FITS: {}", fits_path.name)

        try:
            with self._open_fits(fits_path) as hdul:
                self._validate_header(hdul[0].header, fits_path.name)
                table = hdul[self.hdu_ext].data
                df = self._table_to_dataframe(table, fits_path.name)
        except Exception as e:
            logger.error("Failed to read {}: {}", fits_path.name, e)
            raise

        df = self._convert_timestamps(df)
        df = self._resample(df)
        df = self._validate_flux(df)
        logger.info("SoLEXS file loaded: {} rows, {} to {}",
                    len(df), df.index[0], df.index[-1])
        return df

    def read_directory(self, dir_path: Union[str, Path],
                       sort: bool = True) -> pd.DataFrame:
        """
        Read all FITS files in a directory, concatenate chronologically.
        """
        dir_path = Path(dir_path)
        # Search for lightcurve files — .fits, .fits.gz, or .lc.gz
        files = sorted(dir_path.rglob("*_lc.fits")) if sort else list(dir_path.rglob("*_lc.fits"))
        if not files:
            files = sorted(dir_path.rglob("*.fits")) if sort else list(dir_path.rglob("*.fits"))
            if not files:
                # Try gzipped lightcurve files (ISSDC format)
                files = sorted(dir_path.rglob("*.lc.gz")) if sort else list(dir_path.rglob("*.lc.gz"))
            # Filter out known non-lightcurve files
            files = [f for f in files if "gti" not in f.name and "pi" not in f.name]
            
        if not files:
            raise FileNotFoundError(f"No FITS files found in {dir_path}")
        logger.info("Reading {} SoLEXS files from {}", len(files), dir_path)
        dfs = [self.read_file(f) for f in files]
        merged = pd.concat(dfs).sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]
        logger.info("SoLEXS directory loaded: {} total rows", len(merged))
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
        
        time_col = col_map["time"] if col_map["time"] in table.names else "TIME"
        counts_col = col_map["counts"] if col_map["counts"] in table.names else "COUNTS"
        
        # SoLEXS _lc.fits has [TIME, COUNTS]; _pi.fits has [TSTART, ... , CHANNEL(2D), COUNTS(2D)]
        # Use counts as flux if no explicit FLUX column exists
        flux_col = col_map["flux"] if col_map["flux"] in table.names else None
        
        soft_counts = np.asarray(table[counts_col], dtype=np.float64)
        soft_flux = np.asarray(table[flux_col], dtype=np.float64) if flux_col else soft_counts.copy()
        
        # Convert 2D spectra arrays to 1D (sum across channels for spectra files)
        if len(soft_flux.shape) > 1:
            soft_flux = np.sum(soft_flux, axis=1)
        if len(soft_counts.shape) > 1:
            soft_counts = np.sum(soft_counts, axis=1)
        
        df = pd.DataFrame({
            "time_raw":     table[time_col],
            "soft_flux":    soft_flux,
            "soft_counts":  soft_counts,
            "energy_mean":  np.asarray(table[col_map.get("energy", "ENERGY")], dtype=np.float64)
                            if "energy" in col_map and col_map["energy"] in table.names
                            else np.nan,
            "quality_flag": np.asarray(table["QUALITY"], dtype=int)
                            if "QUALITY" in table.names else 0,
            "source_file":  fname,
        })
        return df

    def _convert_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert SoLEXS time column to UTC DatetimeIndex.
        SoLEXS TIME in these files is already stored as standard Unix epoch seconds.
        """
        try:
            t = Time(df["time_raw"].values, format="unix", scale="utc")
            df.index = pd.DatetimeIndex(t.to_datetime(), tz="UTC",
                                         name="timestamp_utc")
        except Exception:
            # Fallback: try parsing as ISO string
            df.index = pd.to_datetime(df["time_raw"], utc=True)
        df = df.drop(columns=["time_raw"])
        return df

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to uniform cadence. Use mean for flux, sum for counts."""
        numeric_cols = ["soft_flux", "soft_counts", "energy_mean"]
        df_rs = df[numeric_cols].resample(self.resample).mean()
        df_rs["quality_flag"] = df["quality_flag"].resample(
            self.resample).max()   # Propagate worst quality
        df_rs["source_file"] = df["source_file"].resample(
            self.resample).first()
        return df_rs

    def _validate_flux(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Physical sanity checks on flux values.
        SoLEXS soft X-ray background: ~1e-8 W/m²/keV
        Maximum observed X-class: ~1e-3 W/m²/keV
        """
        FLUX_MIN = 1e-12
        FLUX_MAX = 1e-2
        n_bad = ((df["soft_flux"] < FLUX_MIN) |
                 (df["soft_flux"] > FLUX_MAX)).sum()
        if n_bad > 0:
            logger.warning("SoLEXS: {} physically implausible flux values flagged", n_bad)
            df.loc[(df["soft_flux"] < FLUX_MIN) |
                   (df["soft_flux"] > FLUX_MAX), "quality_flag"] = 3
        return df