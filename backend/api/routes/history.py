"""History endpoint — past events and predictions."""

import pandas as pd
from fastapi import APIRouter
from pathlib import Path

router = APIRouter()


@router.get("/history")
def get_history():
    catalog_path = Path("dataset/catalogs/nowcast_catalog.csv")
    if not catalog_path.exists():
        return []
    try:
        df = pd.read_csv(catalog_path)
        return df.to_dict(orient="records")
    except Exception:
        return []
