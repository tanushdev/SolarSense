"""Lightcurve endpoint — real X-ray flux data from processed parquet."""

import pandas as pd
from fastapi import APIRouter
from pathlib import Path

router = APIRouter()

DATA_PATH = Path("dataset/processed/merged_timeseries.parquet")


@router.get("/lightcurve")
def get_lightcurve(hours: float = 6):
    if not DATA_PATH.exists():
        return []
    try:
        df = pd.read_parquet(DATA_PATH)
        if hours > 0:
            cutoff = df.index[-1] - pd.Timedelta(hours=hours)
            df = df[df.index >= cutoff]
        step = max(1, len(df) // 1000)
        df_sample = df.iloc[::step]
        records = []
        for ts, row in df_sample.iterrows():
            records.append({
                "timestamp": ts.isoformat(),
                "soft_flux": float(row.get("soft_flux", 0)),
                "hard_flux": float(row.get("hard_flux", 0)),
            })
        return records
    except Exception:
        return []
