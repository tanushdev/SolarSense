"""Replace inf/nan with 0 in all 60s-cadence parquet files (streaming batches)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from loguru import logger

BATCH_SIZE = 250_000


def fix_file(src: Path):
    tmp = src.with_suffix(".tmp.parquet")
    if tmp.exists():
        tmp.unlink()

    pf = pq.ParquetFile(src)
    total = pf.metadata.num_rows
    logger.info(f"Fixing {src.name} ({total} rows)...")

    writer = None
    for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
        df = batch.to_pandas()

        # Replace inf/nan with 0 in all numeric columns
        num_cols = [c for c in df.columns if df[c].dtype.kind in 'fc']
        if num_cols:
            df[num_cols] = np.nan_to_num(df[num_cols].values, nan=0.0, posinf=0.0, neginf=0.0)

        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="snappy")
        writer.write_table(table)

        if (i + 1) % 10 == 0:
            logger.info(f"  processed {min((i+1)*BATCH_SIZE, total)} / {total}")

    if writer:
        writer.close()
    del pf
    import gc; gc.collect()
    src.unlink()
    tmp.rename(src)
    sz = src.stat().st_size / 1e9
    logger.info(f"  Done ({sz:.2f} GB)")


if __name__ == "__main__":
    proc = Path("dataset/processed")
    for name in ["train", "val", "test"]:
        fix_file(proc / f"{name}_timeseries_60s.parquet")
    logger.info("All files fixed")
