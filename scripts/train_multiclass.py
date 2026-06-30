"""
Train Multi-Class Flare Classifier
====================================
Trains an XGBoost multi-class model to predict flare class
(A/B/C/M/X) in addition to binary flare/no-flare.

Usage:
    python scripts/train_multiclass.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import timedelta
from loguru import logger
import yaml

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.error("XGBoost not installed!")


CLASS_MAP = {0: "N", 1: "A", 2: "B", 3: "C", 4: "M", 5: "X"}
CLASS_INV = {"N": 0, "A": 1, "B": 2, "C": 3, "M": 4, "X": 5}


def build_multiclass_labels(df: pd.DataFrame, catalog: pd.DataFrame,
                            horizon_minutes: int = 30) -> np.ndarray:
    """Build multi-class labels for each sample.

    0 = no flare within horizon
    1 = A-class flare within horizon
    2 = B-class
    3 = C-class
    4 = M-class
    5 = X-class
    """
    labels = np.zeros(len(df), dtype=np.int32)
    horizon_sec = horizon_minutes * 60

    for _, event in catalog.iterrows():
        peak = pd.to_datetime(event["peak_time"])
        cls = str(event["flare_class"]).upper().strip()
        if cls not in CLASS_INV or cls == "N":
            continue
        cls_code = CLASS_INV[cls]

        window_start = peak - timedelta(seconds=horizon_sec)
        window_mask = (df.index >= window_start) & (df.index <= peak)
        window_indices = np.where(window_mask)[0]

        for idx in window_indices:
            if labels[idx] == 0 or cls_code > labels[idx]:
                labels[idx] = cls_code

    return labels


def main():
    logger.info("=== Multi-Class Flare Classifier Training ===")

    data_path = Path("dataset/processed/merged_timeseries.parquet")
    catalog_path = Path("dataset/catalogs/nowcast_catalog.csv")
    feat_path = Path("models/checkpoints/xgboost/xgboost_features.txt")

    if not _XGB_AVAILABLE:
        return

    logger.info("Loading data...")
    df = pd.read_parquet(data_path)
    catalog = pd.read_csv(catalog_path)
    logger.info(f"Data: {len(df)} rows, Catalog: {len(catalog)} events")

    with open(feat_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    horizon = 30
    logger.info(f"Building multi-class labels (horizon={horizon}min, classes=N/A/B/C/M/X)...")
    labels = build_multiclass_labels(df, catalog, horizon)

    class_counts = {CLASS_MAP[k]: int((labels == k).sum()) for k in range(6)}
    logger.info(f"Class distribution: {class_counts}")

    available_features = [c for c in feature_cols if c in df.columns]
    logger.info(f"Features available: {len(available_features)}/{len(feature_cols)}")

    X = df[available_features].fillna(0).values.astype(np.float32)
    y = labels

    # Subsample training data for speed — keep all positive samples
    pos_mask = y > 0
    neg_indices = np.where(~pos_mask)[0]
    pos_indices = np.where(pos_mask)[0]
    np.random.seed(42)
    neg_subsample = np.random.choice(neg_indices, size=min(100000, len(neg_indices)), replace=False)
    sample_idx = np.concatenate([pos_indices, neg_subsample])
    np.random.shuffle(sample_idx)
    X, y = X[sample_idx], y[sample_idx]
    logger.info(f"Subsampled to {len(X)} rows ({int(pos_mask.sum())} positive)")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    weights = np.zeros(6, dtype=np.float32)
    for i in range(6):
        count = (y_train == i).sum()
        weights[i] = len(y_train) / (6 * max(count, 1))
    sample_weight = weights[y_train]

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=6,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=10,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    logger.info("Training multi-class XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight=sample_weight,
        verbose=False,
    )

    logger.info("Evaluating...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred,
                                    target_names=[CLASS_MAP[i] for i in range(6)],
                                    zero_division=0)
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=[CLASS_MAP[i] for i in range(6)], zero_division=0)}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")

    out_dir = Path("models/checkpoints/xgboost")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "xgboost_multiclass.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved multi-class model")

    info = {
        "num_classes": 6,
        "class_mapping": CLASS_MAP,
        "class_inv": CLASS_INV,
        "features": available_features,
        "horizon_minutes": horizon,
        "class_counts": class_counts,
    }
    import json
    with open(out_dir / "xgboost_multiclass_info.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Saved multi-class model info")

    logger.info("=== Multi-Class Training Complete ===")


if __name__ == "__main__":
    main()
