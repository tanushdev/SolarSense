"""SQLite-backed prediction persistence with full versioning."""

import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

DB_PATH = Path("backend/data/predictions.db")


class PredictionStore:
    def __init__(self, db_path: str = str(DB_PATH)):
        self._path = db_path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT UNIQUE,
                timestamp TEXT NOT NULL,
                probability REAL,
                predicted_class TEXT,
                confidence REAL,
                threshold REAL,
                lead_time_minutes REAL,
                alert_level TEXT,
                model_name TEXT,
                similar_events TEXT,
                physics_reason TEXT,
                data_timestamp TEXT,
                execution_time_ms REAL,
                dataset_version TEXT,
                feature_version TEXT,
                config_version TEXT,
                git_commit TEXT,
                model_tag TEXT,
                forecast_horizon_minutes REAL DEFAULT 30,
                validation_status TEXT DEFAULT 'pending',
                actual_outcome TEXT,
                correct INTEGER,
                goes_event_class TEXT,
                goes_event_time TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics_cache (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                metrics TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"PredictionStore: initialized at {self._path}")

    def save_prediction(self, pred: dict, execution_time_ms: float = 0.0):
        try:
            conn = self._conn()
            conn.execute("""
                INSERT OR IGNORE INTO predictions
                    (prediction_id, timestamp, probability, predicted_class,
                     confidence, threshold, lead_time_minutes, alert_level,
                     model_name, similar_events, physics_reason, data_timestamp,
                     execution_time_ms, dataset_version, feature_version,
                     config_version, git_commit, model_tag,
                     forecast_horizon_minutes, validation_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred.get("prediction_id", ""),
                pred.get("timestamp", datetime.now(timezone.utc).isoformat()),
                pred.get("flare_probability", 0.0),
                pred.get("predicted_class", ""),
                1.0 - pred.get("uncertainty", 0.5),
                pred.get("threshold", 0.5),
                pred.get("lead_time_minutes", 0.0),
                pred.get("alert_level", "GREEN"),
                pred.get("model", "unknown"),
                json.dumps(pred.get("similar_events", [])),
                pred.get("physics_reason", ""),
                pred.get("data_timestamp", ""),
                round(execution_time_ms, 2),
                pred.get("dataset_version", ""),
                pred.get("feature_version", ""),
                pred.get("config_version", ""),
                pred.get("git_commit", ""),
                pred.get("model_tag", ""),
                pred.get("forecast_horizon_minutes", 30),
                pred.get("validation_status", "pending"),
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"PredictionStore: save failed: {e}")

    def update_validation(self, prediction_id: str, status: str,
                          goes_class: Optional[str] = None,
                          goes_time: Optional[str] = None,
                          correct: Optional[int] = None):
        try:
            conn = self._conn()
            conn.execute("""
                UPDATE predictions SET
                    validation_status = ?,
                    goes_event_class = ?,
                    goes_event_time = ?,
                    correct = ?
                WHERE prediction_id = ?
            """, (status, goes_class, goes_time, correct, prediction_id))
            conn.commit()
        except Exception as e:
            logger.error(f"PredictionStore: validation update failed: {e}")

    def get_pending_validations(self, limit: int = 50) -> list[dict]:
        try:
            conn = self._conn()
            rows = conn.execute(
                "SELECT * FROM predictions WHERE validation_status = 'pending' "
                "ORDER BY id ASC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"PredictionStore: pending query failed: {e}")
            return []

    def get_recent_predictions(self, limit: int = 100) -> list[dict]:
        try:
            conn = self._conn()
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"PredictionStore: query failed: {e}")
            return []

    def get_stats(self) -> dict:
        try:
            conn = self._conn()
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            correct = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE correct = 1"
            ).fetchone()[0] or 0
            incorrect = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE correct = 0"
            ).fetchone()[0] or 0
            pending = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE validation_status = 'pending'"
            ).fetchone()[0] or 0
            validated = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE validation_status = 'validated'"
            ).fetchone()[0] or 0
            recent = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return {
                "total_predictions": total,
                "correct_predictions": correct,
                "incorrect_predictions": incorrect,
                "pending_validation": pending,
                "validated": validated,
                "accuracy": round(correct / max(total, 1), 4),
                "last_prediction": dict(recent) if recent else None,
            }
        except Exception as e:
            logger.error(f"PredictionStore: stats failed: {e}")
            return {}

    def save_metrics(self, metrics: dict):
        try:
            conn = self._conn()
            conn.execute("""
                INSERT OR REPLACE INTO metrics_cache (id, metrics, updated_at)
                VALUES (1, ?, ?)
            """, (json.dumps(metrics), datetime.now(timezone.utc).isoformat()))
            conn.commit()
        except Exception as e:
            logger.error(f"PredictionStore: save_metrics failed: {e}")

    def get_metrics(self) -> Optional[dict]:
        try:
            conn = self._conn()
            row = conn.execute("SELECT metrics FROM metrics_cache WHERE id = 1").fetchone()
            if row:
                return json.loads(row["metrics"])
            return None
        except Exception as e:
            logger.error(f"PredictionStore: get_metrics failed: {e}")
            return None


_store_instance = None


def get_store() -> PredictionStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = PredictionStore()
    return _store_instance
