"""
FITS Utilities
==============
Shared utilities for FITS file reading across SoLEXS and HEL1OS readers.

Provides:
  - Column mapping resolution (handles varying column names across file versions)
  - Header validation with configurable required keys
  - Timestamp conversion (MJD, Unix, ISO)
  - 2D array reduction (spectra channels → 1D)
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from pathlib import Path
from loguru import logger
from typing import Union, List, Optional


def resolve_column(table, candidates: List[str]) -> Optional[str]:
    """
    Return the first column name from candidates that exists in the table.

    Parameters
    ----------
    table : FITS table (BinTableHDU data)
    candidates : Ordered list of column name candidates

    Returns
    -------
    str or None
    """
    for c in candidates:
        if c in table.names:
            return c
    return None


def validate_header(header: fits.Header, fname: str,
                    required_keys: List[str] = None):
    """
    Validate required FITS header keywords.

    Parameters
    ----------
    header : FITS header
    fname : Filename (for logging)
    required_keys : List of mandatory keywords (default: TELESCOP, INSTRUME, DATE-OBS)
    """
    if required_keys is None:
        required_keys = ["TELESCOP", "INSTRUME", "DATE-OBS"]
    missing = [k for k in required_keys if k not in header]
    if missing:
        logger.warning("Missing FITS keywords {} in {}", missing, fname)


def reduce_2d_array(data: np.ndarray, method: str = "sum") -> np.ndarray:
    """
    Reduce a 2D array (e.g. spectra channels x time) to 1D.

    Parameters
    ----------
    data : np.ndarray
        Input array, possibly 2D or higher.
    method : str
        'sum' → sum across channels (use for total counts)
        'mean' → mean across channels (use for mean flux)
    """
    if data is None:
        return None
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim <= 1:
        return arr
    if method == "sum":
        return np.sum(arr, axis=1)
    elif method == "mean":
        return np.mean(arr, axis=1)
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def convert_mjd_to_datetime(mjd_values: np.ndarray) -> pd.DatetimeIndex:
    """Convert Modified Julian Date array to UTC DatetimeIndex."""
    t = Time(mjd_values, format="mjd", scale="utc")
    return pd.DatetimeIndex(t.to_datetime(), tz="UTC", name="timestamp_utc")


def convert_unix_to_datetime(unix_values: np.ndarray) -> pd.DatetimeIndex:
    """Convert Unix epoch seconds array to UTC DatetimeIndex."""
    t = Time(unix_values, format="unix", scale="utc")
    return pd.DatetimeIndex(t.to_datetime(), tz="UTC", name="timestamp_utc")


def get_base_time_from_header(hdul: fits.HDUList) -> Optional[pd.Timestamp]:
    """
    Extract base observation time from primary FITS header.
    Tries ISOSTART, DATE_OBS+TIME_OBS, DATE-OBS.

    Parameters
    ----------
    hdul : FITS HDUList

    Returns
    -------
    pd.Timestamp or None
    """
    header = hdul[0].header
    if "ISOSTART" in header:
        return pd.to_datetime(header["ISOSTART"], utc=True)
    if "DATE_OBS" in header:
        date_str = header["DATE_OBS"]
        time_str = header.get("TIME_OBS", "00:00:00")
        if time_str == "00:00:00" and "TIME-OBS" in header:
            time_str = header["TIME-OBS"]
        return pd.to_datetime(f"{date_str}T{time_str}", utc=True)
    return None


def is_lightcurve_file(filepath: Path) -> bool:
    """
    Heuristic to check if a FITS file contains lightcurve data.
    Skips gti, evt (with tstart/tstop), and spectra-only files.

    Parameters
    ----------
    filepath : Path to .fits file

    Returns
    -------
    bool
    """
    fname = filepath.name.lower()
    if "_lc" in fname or "lightcurve" in fname:
        return True
    if "_gti" in fname or "_evt" in fname:
        return False
    return True
