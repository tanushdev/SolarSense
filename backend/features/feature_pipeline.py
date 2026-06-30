"""
Feature Pipeline
================
Orchestrates all feature extractors (physics, spectral, temporal, statistical).
Provides a single entry point that runs all enabled extractors in sequence.

This is the ONLY file that models/training code should import for features.
"""

import pandas as pd
from loguru import logger
import yaml

from backend.features.physics_features import PhysicsFeatureExtractor
from backend.features.spectral_features import SpectralFeatureExtractor
from backend.features.temporal_features import TemporalFeatureExtractor
from backend.features.statistical_features import StatisticalFeatureExtractor


class FeaturePipeline:
    """
    Aggregates all feature extractors into a single pipeline.

    Usage:
        pipeline = FeaturePipeline()
        df_features = pipeline.extract_all(df_aligned)
        df_full = pd.concat([df_aligned, df_features], axis=1)
    """

    def __init__(self, config_path: str = "configs/features.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.extractors = {
            "physics":     PhysicsFeatureExtractor(config_path),
            "spectral":    SpectralFeatureExtractor(config_path),
            "temporal":    TemporalFeatureExtractor(),
            "statistical": StatisticalFeatureExtractor(),
        }
        logger.info("FeaturePipeline initialized with {} extractors", len(self.extractors))

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all feature extractors and return concatenated feature DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Aligned and cleaned DataFrame with [soft_flux, hard_flux] columns.

        Returns
        -------
        pd.DataFrame with all features, same index as input.
        """
        all_features = []
        for name, extractor in self.extractors.items():
            logger.debug("Running {} extractor...", name)
            try:
                feats = extractor.extract_all(df)
                all_features.append(feats)
            except Exception as e:
                logger.error("Feature extractor {} failed: {}", name, e)
                raise

        if not all_features:
            logger.warning("No features extracted!")
            return pd.DataFrame(index=df.index)

        result = pd.concat(all_features, axis=1)
        logger.info("FeaturePipeline: extracted {} features", len(result.columns))
        return result
