"""
Dataset Pipeline
=================
End-to-end pipeline: raw FITS files → cleaned, feature-engineered DataFrame.

Steps:
  1. Read SoLEXS + HEL1OS FITS
  2. Align to common 5s UTC grid
  3. Clean artifacts (Isolation Forest)
  4. Extract all features (physics, spectral, temporal, statistical)
  5. Save intermediate + final parquet files
  6. Generate nowcast catalog
  7. Create train/val/test splits

Usage:
    python -c "from backend.pipeline.dataset_pipeline import DatasetPipeline; DatasetPipeline().run()"
"""

from pathlib import Path
import pandas as pd
from loguru import logger

from backend.data.dataset_builder import DatasetBuilder


class DatasetPipeline:
    """
    Wraps DatasetBuilder for use as a callable pipeline stage.
    """

    def __init__(self, data_config: str = "configs/data.yaml",
                 features_config: str = "configs/features.yaml"):
        self.builder = DatasetBuilder(data_config, features_config)

    def run(self):
        """Execute the full dataset pipeline."""
        logger.info("=== Dataset Pipeline ===")
        self.builder.run()
        logger.info("=== Dataset Pipeline Complete ===")
        return self
