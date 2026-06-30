"""Downsample train/val/test parquet files to 60s cadence (every 12th row).
Streams in batches to keep memory under control."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from loguru import logger

BATCH_SIZE = 500_000  # rows per batch


def downsample(src: Path, dst: Path):
    if dst.exists():
        logger.info(f"{dst.name} exists, skipping")
        return

    logger.info(f"Reading {src}...")
    pf = pq.ParquetFile(src)
    total = pf.metadata.num_rows
    n_cols = len(pf.schema_arrow.names)
    logger.info(f"  {total} rows, {n_cols} columns")

    writer = None
    for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
        # Take every 12th row from this batch
        df = batch.to_pandas().iloc[::12]
        table = pa.Table.from_pandas(df)

        if writer is None:
            writer = pq.ParquetWriter(dst, table.schema, compression="snappy")
        writer.write_table(table)

        if (i + 1) % 5 == 0:
            logger.info(f"  processed {min((i+1)*BATCH_SIZE, total)} / {total} rows")

    if writer:
        writer.close()
    sz = dst.stat().st_size / 1e9
    logger.info(f"Saved {dst} ({sz:.2f} GB)")


if __name__ == "__main__":
    proc = Path("dataset/processed")
    for name in ["train", "val", "test"]:
        src = proc / f"{name}_timeseries.parquet"
        dst = proc / f"{name}_timeseries_60s.parquet"
        downsample(src, dst)
    logger.info("Done")
