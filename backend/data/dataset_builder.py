"""
Dataset Builder
===============
Orchestrates the full data processing pipeline:
  1. Read raw FITS files (SoLEXS + HEL1OS)
  2. Align to common UTC grid
  3. Clean artifacts (Isolation Forest)
  4. Extract all features
  5. Generate nowcast catalog
  6. Split into train/val/test chronologically
  7. Save to parquet
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import yaml

from backend.data.solexs_reader import SoLEXSReader
from backend.data.hel1os_reader import HEL1OSReader
from backend.data.aligner import InstrumentAligner
from backend.data.cleaner import DataCleaner
from backend.features.physics_features import PhysicsFeatureExtractor
from backend.features.spectral_features import SpectralFeatureExtractor
from backend.features.temporal_features import TemporalFeatureExtractor
from backend.features.statistical_features import StatisticalFeatureExtractor
from backend.features.feature_pipeline import FeaturePipeline
from backend.models.nowcaster.threshold_detector import ThresholdNowcaster


class DatasetBuilder:
    """
    Full pipeline orchestrator for building the training dataset.

    Usage:
        builder = DatasetBuilder()
        builder.run()
    """

    def __init__(self, data_config: str = "configs/data.yaml",
                 features_config: str = "configs/features.yaml"):
        with open(data_config) as f:
            self.data_cfg = yaml.safe_load(f)
        with open(features_config) as f:
            self.feat_cfg = yaml.safe_load(f)

        self.processed_dir = Path(self.data_cfg["storage"]["processed_dir"])
        self.catalog_dir = Path(self.data_cfg["storage"]["catalog_dir"])
        self.format = self.data_cfg["storage"]["format"]
        self.compression = self.data_cfg["storage"]["compression"]

    def run(self):
        """Execute the complete dataset building pipeline."""
        logger.info("DatasetBuilder: Starting pipeline")

        # 1. Read raw FITS
        logger.info("Step 1: Reading SoLEXS FITS...")
        solexs_reader = SoLEXSReader()
        solexs_raw = Path(self.data_cfg["solexs"]["raw_dir"])
        df_solexs = solexs_reader.read_directory(solexs_raw)
        self._save("solexs_timeseries", df_solexs)

        logger.info("Step 1: Reading HEL1OS FITS...")
        hel1os_reader = HEL1OSReader()
        hel1os_raw = Path(self.data_cfg["hel1os"]["raw_dir"])
        df_hel1os = hel1os_reader.read_directory(hel1os_raw)
        self._save("hel1os_timeseries", df_hel1os)

        # 2. Align
        logger.info("Step 2: Aligning dual streams...")
        aligner = InstrumentAligner()
        df_aligned = aligner.align(df_solexs, df_hel1os)

        # 3. Clean
        logger.info("Step 3: Cleaning artifacts...")
        cleaner = DataCleaner(contamination=0.01)
        df_cleaned = cleaner.fit_predict(df_aligned)

        # 4. Extract all features
        logger.info("Step 4: Extracting features...")
        pipeline = FeaturePipeline()
        df_features = pipeline.extract_all(df_cleaned)
        df_final = pd.concat([df_cleaned, df_features], axis=1)
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]
        self._save("merged_timeseries", df_final)

        # 5. Nowcast catalogs — challenge-compliant independent detection
        logger.info("Step 5: Independent flare detection on each instrument...")
        nowcaster = ThresholdNowcaster()
        df_detect = df_final[["soft_flux", "hard_flux", "soft_counts", "hard_counts"]]
        cat_solexs, cat_hel1os, cat_master = nowcaster.detect_all(df_detect)

        for name, cat in [("solexs_catalog", cat_solexs),
                          ("hel1os_catalog", cat_hel1os),
                          ("nowcast_catalog", cat_master)]:
            path = self.catalog_dir / f"{name}.csv"
            cat.to_csv(path, index=False)
            logger.info("Saved catalog: {} ({} events)", path, len(cat))

        # 6. Split chronologically
        logger.info("Step 6: Splitting train/val/test...")
        n = len(df_final)
        idx_val = int(n * 0.60)
        idx_test = int(n * 0.80)
        for name, split in [("train", df_final.iloc[:idx_val]),
                            ("val", df_final.iloc[idx_val:idx_test]),
                            ("test", df_final.iloc[idx_test:])]:
            self._save(f"{name}_timeseries", split)
            logger.info("  {}: {} samples", name, len(split))

        logger.info("DatasetBuilder: Pipeline complete!")

    def _save(self, name: str, df: pd.DataFrame):
        """Save DataFrame to parquet."""
        path = self.processed_dir / f"{name}.{self.format}"
        df.to_parquet(path, compression=self.compression)
        logger.info("Saved {} ({} rows, {} cols)", path, len(df), len(df.columns))
