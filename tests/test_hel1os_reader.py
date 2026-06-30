"""Tests for HEL1OS FITS reader."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.time import Time

from backend.data.hel1os_reader import HEL1OSReader


@pytest.fixture
def hel1os_reader():
    return HEL1OSReader(config_path="configs/data.yaml")


@pytest.fixture
def sample_hel1os_lc(tmp_path):
    """Generate a synthetic HEL1OS lightcurve FITS file."""
    n = 100
    mjd_start = 60450.0
    mjd = np.linspace(mjd_start, mjd_start + 0.01, n)
    ctr = np.random.poisson(50, n).astype(np.float32)
    stat_err = np.sqrt(ctr).astype(np.float32)
    isot = [Time(m, format="mjd", scale="utc").isot for m in mjd]

    col1 = fits.Column(name="MJD", format="D", array=mjd)
    col2 = fits.Column(name="ISOT", format="A24", array=isot)
    col3 = fits.Column(name="CTR", format="E", array=ctr)
    col4 = fits.Column(name="STAT_ERR", format="E", array=stat_err)
    hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4])

    header = fits.Header()
    header["TELESCOP"] = "HEL1OS"
    header["INSTRUME"] = "HEL1OS"
    header["DATE-OBS"] = "2024-07-06"

    path = tmp_path / "hel1os_lightcurve_cdte_test.fits"
    fits.writeto(path, np.array([]), header=header)
    fits.append(path, hdu.data, hdu.header)
    return path


class TestHEL1OSReader:
    def test_read_lightcurve(self, hel1os_reader, sample_hel1os_lc):
        df = hel1os_reader.read_file(sample_hel1os_lc)
        assert isinstance(df, pd.DataFrame)
        assert "hard_flux" in df.columns
        assert "hard_counts" in df.columns
        assert len(df) > 0

    def test_timestamp_is_utc(self, hel1os_reader, sample_hel1os_lc):
        df = hel1os_reader.read_file(sample_hel1os_lc)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_flux_positive(self, hel1os_reader, sample_hel1os_lc):
        df = hel1os_reader.read_file(sample_hel1os_lc)
        assert (df["hard_flux"].fillna(0) >= 0).all()

    def test_resampling(self, hel1os_reader, sample_hel1os_lc):
        df = hel1os_reader.read_file(sample_hel1os_lc)
        diffs = df.index.to_series().diff().dropna().dt.total_seconds()
        assert diffs.nunique() <= 3  # At most 3 unique cadences after resampling
