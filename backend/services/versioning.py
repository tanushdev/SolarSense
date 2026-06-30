"""Prediction versioning — dataset, model, feature, config versions."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

CONFIG_PATHS = [
    "configs/data.yaml",
    "configs/features.yaml",
    "configs/models.yaml",
    "configs/training.yaml",
    "configs/thresholds.yaml",
    "configs/deployment.yaml",
]


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=Path(__file__).resolve().parent.parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _file_hash(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    return hashlib.md5(p.read_bytes()).hexdigest()[:12]


def compute_dataset_version() -> str:
    """Hash the merged parquet + catalog to fingerprint the dataset."""
    hasher = hashlib.md5()
    for path in ["dataset/processed/merged_timeseries.parquet",
                 "dataset/catalogs/nowcast_catalog.csv"]:
        p = Path(path)
        if p.exists():
            hasher.update(p.read_bytes()[:1024 * 1024])  # first 1 MB
    return hasher.hexdigest()[:12]


def compute_feature_version() -> str:
    """Hash all feature extractor source files."""
    hasher = hashlib.md5()
    for fpath in Path("backend/features").rglob("*.py"):
        hasher.update(fpath.read_bytes())
    return hasher.hexdigest()[:12]


def compute_config_version() -> str:
    """Hash all config YAML files."""
    hasher = hashlib.md5()
    for cfg in CONFIG_PATHS:
        p = Path(cfg)
        if p.exists():
            hasher.update(p.read_bytes())
    return hasher.hexdigest()[:12]


class PredictionVersion:
    """Immutable version snapshot for a single prediction."""

    def __init__(self,
                 model_name: str = "unknown",
                 model_tag: str = "unknown",
                 threshold: float = 0.5):
        self.dataset_version = compute_dataset_version()
        self.feature_version = compute_feature_version()
        self.config_version = compute_config_version()
        self.git_commit = _git_commit_hash()
        self.model_name = model_name
        self.model_tag = model_tag
        self.threshold = threshold

    def to_dict(self) -> dict:
        return {
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "config_version": self.config_version,
            "git_commit": self.git_commit,
            "model_name": self.model_name,
            "model_tag": self.model_tag,
            "threshold": self.threshold,
        }

    def to_id(self) -> str:
        """Short unique prediction ID."""
        raw = f"{datetime.now(timezone.utc).isoformat()}:{self.dataset_version}:{self.model_name}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
