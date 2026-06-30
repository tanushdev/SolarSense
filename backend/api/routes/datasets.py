"""Real dataset info from the filesystem — no mock data."""

from pathlib import Path
from fastapi import APIRouter

router = APIRouter()
ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW = ROOT / "dataset" / "raw"
PROCESSED = ROOT / "dataset" / "processed"
CATALOG = ROOT / "dataset" / "catalogs"
MODELS = ROOT / "models" / "checkpoints"


def dir_stats(dir_path: Path) -> dict:
    if not dir_path.exists():
        return {"exists": False, "files": 0, "size_mb": 0.0}
    total_bytes = 0
    file_count = 0
    for f in dir_path.iterdir():
        if f.is_file():
            file_count += 1
            total_bytes += f.stat().st_size
        elif f.is_dir():
            for sub in f.rglob("*"):
                if sub.is_file():
                    file_count += 1
                    total_bytes += sub.stat().st_size
    return {"exists": True, "files": file_count, "size_mb": round(total_bytes / (1024 * 1024), 2)}


@router.get("/datasets")
def get_datasets():
    return {
        "solexs": {"path": "dataset/raw/solexs", **dir_stats(RAW / "solexs")},
        "hel1os": {"path": "dataset/raw/hel1os", **dir_stats(RAW / "hel1os")},
        "processed": {"path": "dataset/processed", **dir_stats(PROCESSED)},
        "catalogs": {"path": "dataset/catalogs", **dir_stats(CATALOG)},
        "models": {"path": "models/checkpoints", **dir_stats(MODELS)},
    }
