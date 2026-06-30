"""Continue pipeline — streamed per-month parquet, catalog on flux-only subset."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from backend.data.aligner import InstrumentAligner
from backend.data.cleaner import DataCleaner
from backend.features.feature_pipeline import FeaturePipeline
from backend.models.nowcaster.threshold_detector import ThresholdNowcaster


def main():
    logger.info("=== Continuing Pipeline ===")
    processed_dir = Path("dataset/processed")
    catalog_dir = Path("dataset/catalogs")

    df_solexs = pd.read_parquet(processed_dir / "solexs_timeseries.parquet")
    df_hel1os = pd.read_parquet(processed_dir / "hel1os_timeseries.parquet")
    logger.info(f"SoLEXS: {len(df_solexs)} rows, HEL1OS: {len(df_hel1os)} rows")

    aligner = InstrumentAligner()
    df_aligned = aligner.align(df_solexs, df_hel1os)
    del df_solexs, df_hel1os

    cleaner = DataCleaner(contamination=0.01)
    df_cleaned = cleaner.fit_predict(df_aligned)
    del df_aligned

    dates = df_cleaned.index.date
    unique_dates = sorted(set(dates))
    chunk_size = 30
    chunks = [unique_dates[i:i+chunk_size] for i in range(0, len(unique_dates), chunk_size)]

    merged_dir = processed_dir / "merged_chunks"
    merged_dir.mkdir(exist_ok=True)

    # Keep flux-only subset for catalog generation (fits in memory)
    flux_parts = []

    for ci, chunk_dates in enumerate(chunks):
        sd, ed = chunk_dates[0], chunk_dates[-1]
        mask = (df_cleaned.index.date >= sd) & (df_cleaned.index.date <= ed)
        chunk = df_cleaned.loc[mask]
        logger.info(f"Chunk {ci+1}/{len(chunks)}: {sd} to {ed} ({len(chunk)} rows)")

        pipeline = FeaturePipeline()
        feats = pipeline.extract_all(chunk)
        merged = pd.concat([chunk, feats], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]

        fpath = merged_dir / f"merged_chunk_{ci:03d}.parquet"
        merged.to_parquet(fpath, compression="snappy")
        logger.info(f"  Saved {fpath} ({merged.shape})")

        flux_parts.append(merged[["soft_flux", "hard_flux", "soft_counts", "hard_counts"]])
        del merged, feats, chunk

    # Catalog on flux-only subset (small)
    logger.info("Generating catalogs from flux-only subset...")
    df_flux = pd.concat(flux_parts).sort_index()
    del flux_parts
    logger.info(f"  Flux subset: {len(df_flux)} rows")

    nowcaster = ThresholdNowcaster()
    cat_solexs, cat_hel1os, cat_master = nowcaster.detect_all(df_flux)
    del df_flux

    for name, cat in [("solexs_catalog", cat_solexs),
                      ("hel1os_catalog", cat_hel1os),
                      ("nowcast_catalog", cat_master)]:
        cat.to_csv(catalog_dir / f"{name}.csv", index=False)
        logger.info(f"Saved {name}: {len(cat)} events")

    # Create train/val/test splits from chunks
    logger.info("Creating train/val/test splits...")
    n_chunks = len(chunks)
    val_start = int(n_chunks * 0.60)
    test_start = int(n_chunks * 0.80)

    for split_name, ci_start, ci_end in [
        ("train", 0, val_start),
        ("val", val_start, test_start),
        ("test", test_start, n_chunks)
    ]:
        parts = []
        for ci in range(ci_start, ci_end):
            fpath = merged_dir / f"merged_chunk_{ci:03d}.parquet"
            parts.append(pd.read_parquet(fpath))
        split_df = pd.concat(parts)
        split_path = processed_dir / f"{split_name}_timeseries.parquet"
        split_df.to_parquet(split_path, compression="snappy")
        logger.info(f"  {split_name}: {len(split_df)} rows -> {split_path}")
        del split_df, parts

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
