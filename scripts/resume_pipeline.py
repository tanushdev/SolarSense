"""Resume after feature extraction — catalog gen (fast merge) + splits."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from backend.models.nowcaster.threshold_detector import ThresholdNowcaster


def main():
    logger.info("=== Resuming Pipeline: catalog + splits ===")
    processed_dir = Path("dataset/processed")
    catalog_dir = Path("dataset/catalogs")
    merged_dir = processed_dir / "merged_chunks"

    if not merged_dir.exists():
        logger.error("No merged_chunks dir — run continue_pipeline.py first")
        return

    chunk_files = sorted(merged_dir.glob("merged_chunk_*.parquet"))
    logger.info(f"Found {len(chunk_files)} merged chunks")

    # Load flux-only subset for catalog gen (fits in memory)
    flux_parts = []
    for fpath in chunk_files:
        df = pd.read_parquet(fpath, columns=["soft_flux", "hard_flux", "soft_counts", "hard_counts"])
        flux_parts.append(df)

    logger.info("Generating catalogs from flux-only subset...")
    df_flux = pd.concat(flux_parts).sort_index()
    del flux_parts
    logger.info(f"  Flux subset: {len(df_flux)} rows")

    nowcaster = ThresholdNowcaster()
    cat_solexs, cat_hel1os, cat_master = nowcaster.detect_all(df_flux)
    del df_flux

    cat_dir = Path("dataset/catalogs")
    cat_dir.mkdir(exist_ok=True)
    for name, cat in [("solexs_catalog", cat_solexs),
                      ("hel1os_catalog", cat_hel1os),
                      ("nowcast_catalog", cat_master)]:
        path = cat_dir / f"{name}.csv"
        cat.to_csv(path, index=False)
        logger.info(f"Saved {name}: {path} ({len(cat)} events)")

    # Train/val/test splits
    logger.info("Creating train/val/test splits...")
    n_chunks = len(chunk_files)
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
