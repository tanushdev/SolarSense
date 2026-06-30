#!/usr/bin/env python
"""
Full data pipeline runner.
Run this once after downloading FITS files from ISSDC PRADAN.

Steps:
  1. Read all SoLEXS FITS → parquet
  2. Read all HEL1OS FITS → parquet
  3. Align both streams to common UTC grid
  4. Run Isolation Forest cleaning
  5. Extract all physics + spectral features
  6. Compute Matrix Profile / statistical features
  7. Save merged feature DataFrame
  8. Run ThresholdNowcaster to generate nowcast catalog
  9. Save ready-to-train datasets (train/val/test splits)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# Add root folder to path so backend can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.data.solexs_reader import SoLEXSReader
from backend.data.hel1os_reader import HEL1OSReader
from backend.data.aligner import InstrumentAligner
from backend.data.cleaner import DataCleaner
from backend.features.physics_features import PhysicsFeatureExtractor
from backend.features.spectral_features import SpectralFeatureExtractor
from backend.models.nowcaster.threshold_detector import ThresholdNowcaster


def main():
    logger.info("Starting SolarSense-AI Unified Data Pipeline Builder")
    
    # Paths
    solexs_raw = Path("dataset/raw/solexs")
    hel1os_raw = Path("dataset/raw/hel1os")
    processed_dir = Path("dataset/processed")
    catalog_dir = Path("dataset/catalogs")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read SoLEXS FITS
    logger.info("Step 1: Reading SoLEXS raw FITS data...")
    solexs_reader = SoLEXSReader()
    df_solexs = solexs_reader.read_directory(solexs_raw)
    solexs_out = processed_dir / "solexs_timeseries.parquet"
    df_solexs.to_parquet(solexs_out, compression="snappy")
    logger.info("Saved SoLEXS time series to: {}", solexs_out)

    # 2. Read HEL1OS FITS
    logger.info("Step 2: Reading HEL1OS raw FITS data...")
    hel1os_reader = HEL1OSReader()
    df_hel1os = hel1os_reader.read_directory(hel1os_raw)
    hel1os_out = processed_dir / "hel1os_timeseries.parquet"
    df_hel1os.to_parquet(hel1os_out, compression="snappy")
    logger.info("Saved HEL1OS time series to: {}", hel1os_out)

    # 3. Align both streams to common UTC grid
    logger.info("Step 3: Temporal alignment of dual streams...")
    aligner = InstrumentAligner()
    df_aligned = aligner.align(df_solexs, df_hel1os)
    
    # 4. Run Isolation Forest cleaning
    logger.info("Step 4: Running Isolation Forest anomaly cleaning...")
    cleaner = DataCleaner(contamination=0.01)
    df_cleaned = cleaner.fit_predict(df_aligned)

    # 5. Extract all physics + spectral features
    logger.info("Step 5: Extracting physics features...")
    extractor = PhysicsFeatureExtractor()
    df_features = extractor.extract_all(df_cleaned)
    
    logger.info("Step 5b: Extracting spectral features (FFT and Wavelets)...")
    spectral_extractor = SpectralFeatureExtractor()
    df_spectral = spectral_extractor.extract_all(df_cleaned)
    
    # Combine original clean fluxes and flags with engineered features
    df_final = pd.concat([df_cleaned, df_features, df_spectral], axis=1)
    
    # Drop any duplicate columns if they exist
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    # Save merged feature DataFrame
    merged_out = processed_dir / "merged_timeseries.parquet"
    df_final.to_parquet(merged_out, compression="snappy")
    logger.info("Saved merged feature dataset to: {}", merged_out)

    # 8. Run ThresholdNowcaster to generate nowcast catalog
    logger.info("Step 8: Generating Threshold Nowcast catalog...")
    nowcaster = ThresholdNowcaster()
    events = nowcaster.detect(df_final[["soft_flux", "hard_flux"]])
    df_catalog = nowcaster.to_catalog(events)
    logger.info("df_catalog shape: {}", df_catalog.shape)
    
    catalog_out = catalog_dir / "nowcast_catalog.csv"
    df_catalog.to_csv(catalog_out, index=False)
    logger.info("Saved detected nowcast event catalog to: {} | size={}", catalog_out, os.path.getsize(catalog_out))

    # 9. Save ready-to-train datasets (train/val/test splits)
    # Split chronologically: Train (first 60%), Val (next 20%), Test (last 20%)
    logger.info("Step 9: Chronological dataset partitioning...")
    n_samples = len(df_final)
    idx_val = int(n_samples * 0.60)
    idx_test = int(n_samples * 0.80)
    
    train_split = df_final.iloc[:idx_val]
    val_split = df_final.iloc[idx_val:idx_test]
    test_split = df_final.iloc[idx_test:]
    
    train_split.to_parquet(processed_dir / "train_timeseries.parquet", compression="snappy")
    val_split.to_parquet(processed_dir / "val_timeseries.parquet", compression="snappy")
    test_split.to_parquet(processed_dir / "test_timeseries.parquet", compression="snappy")
    
    logger.info("Splits generated:")
    logger.info("  Train: {} samples ({} to {})", len(train_split), train_split.index[0], train_split.index[-1])
    logger.info("  Val:   {} samples ({} to {})", len(val_split), val_split.index[0], val_split.index[-1])
    logger.info("  Test:  {} samples ({} to {})", len(test_split), test_split.index[0], test_split.index[-1])
    
    logger.info("SolarSense-AI Data Pipeline completed successfully!")


if __name__ == "__main__":
    main()