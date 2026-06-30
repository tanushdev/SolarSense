"""
Experiment Logger
===================
Logs all experiments (hyperparameters, metrics, artifacts) for
reproducibility and comparison.

Supports multiple backends:
  - MLflow (primary, for rich UI comparison)
  - Local JSON log (fallback, always enabled)
  - CSV log (for spreadsheet analysis)

Logging per experiment:
  - git commit hash
  - full config (merged data + features + models)
  - all metrics (TSS, HSS, Brier, lead time per horizon)
  - model checkpoints
  - predictions on test set (for ensemble stacking)
  - training time, GPU memory
"""

import json
import os
from datetime import datetime
from pathlib import Path
from loguru import logger
from typing import Optional


class ExperimentLogger:
    """
    Logs experiment metadata, configs, and metrics to multiple backends.

    Usage:
        exp_logger = ExperimentLogger(experiment_name="rf_benchmark_01")
        exp_logger.log_params({"n_estimators": 100, "max_depth": 10})
        exp_logger.log_metrics({"tss": 0.72, "hss": 0.65})
        exp_logger.log_artifact("models/checkpoints/rf_benchmark_01/best.pt")
        exp_logger.finalize(status="completed")
    """

    def __init__(self, experiment_name: str,
                 base_dir: str = "experiments",
                 mlflow_enabled: bool = False):
        self.name = experiment_name
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_enabled = mlflow_enabled
        self._params = {}
        self._metrics = {}
        self._artifacts = []
        self._start_time = datetime.utcnow()
        self._status = "running"

        # Local JSON log
        self._json_path = self.exp_dir / "experiment.json"
        self._csv_path = self.exp_dir / "metrics.csv"

        logger.info("ExperimentLogger initialized: {}", experiment_name)

    def log_params(self, params: dict):
        """Log hyperparameters."""
        self._params.update(params)
        self._flush()

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics (can be called multiple times per epoch)."""
        entry = {"timestamp": datetime.utcnow().isoformat(),
                 "step": step,
                 **metrics}
        # Append to CSV
        import csv
        file_exists = self._csv_path.exists()
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)
        self._metrics.update(metrics)
        self._flush()

    def log_artifact(self, path: str):
        """Log a file path as an artifact."""
        if os.path.exists(path):
            self._artifacts.append(path)
            logger.info("Logged artifact: {}", path)

    def finalize(self, status: str = "completed"):
        """Mark experiment as completed/failed."""
        self._status = status
        self._end_time = datetime.utcnow()
        self._flush()
        duration = (self._end_time - self._start_time).total_seconds()
        logger.info("Experiment {} {} | Duration: {:.1f}s",
                    self.name, status, duration)

    def _flush(self):
        """Write current state to JSON file."""
        record = {
            "experiment": self.name,
            "status": self._status,
            "start_time": self._start_time.isoformat(),
            "end_time": getattr(self, "_end_time", datetime.utcnow()).isoformat(),
            "params": self._params,
            "metrics": self._metrics,
            "artifacts": self._artifacts,
        }
        with open(self._json_path, "w") as f:
            json.dump(record, f, indent=2, default=str)
