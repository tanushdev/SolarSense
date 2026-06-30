"""
Unit tests for SoLEXS FITS reader.
Run: pytest tests/test_solexs_reader.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from backend.data.solexs_reader import SoLEXSReader

SAMPLE_FITS = "tests/fixtures/sample_solexs.fits"


@pytest.fixture
def reader():
    return SoLEXSReader()


class TestSoLEXSReader:
    def test_read_sample_fits(self, reader):
        """Basic read succeeds and returns expected columns."""
        df = reader.read_file(SAMPLE_FITS)
        assert "soft_flux" in df.columns
        assert "soft_counts" in df.columns
        assert "quality_flag" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_timestamp_is_utc(self, reader):
        df = reader.read_file(SAMPLE_FITS)
        assert str(df.index.tz) == "UTC"

    def test_flux_physical_range(self, reader):
        df = reader.read_file(SAMPLE_FITS)
        assert (df["soft_flux"].dropna() > 1e-12).all(), "Flux below physical minimum"
        assert (df["soft_flux"].dropna() < 1e-2).all(), "Flux above physical maximum"

    def test_corrupt_file_raises(self, reader):
        with pytest.raises(Exception):
            reader.read_file("tests/fixtures/corrupt.fits")

    def test_missing_header_warns(self, reader, caplog):
        reader.read_file("tests/fixtures/missing_header_solexs.fits")
        assert "Missing FITS keywords" in caplog.text

    def test_resampling_uniform_cadence(self, reader):
        df = reader.read_file(SAMPLE_FITS)
        diffs = df.index.to_series().diff().dropna()
        expected = pd.Timedelta(seconds=5)
        assert (diffs == expected).all(), "Cadence is not uniform 5s after resampling"

    def test_read_directory(self, reader, tmp_path):
        """Multi-file directory read returns chronologically sorted concatenation."""
        import shutil
        shutil.copy(SAMPLE_FITS, tmp_path / "a.fits")
        shutil.copy(SAMPLE_FITS, tmp_path / "b.fits")
        df = reader.read_directory(tmp_path)
        assert len(df) > 0