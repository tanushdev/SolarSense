"""
Master Training Loop
=====================
Orchestrates the complete training pipeline:
  1. Load processed data
  2. Build sliding window datasets
  3. Train nowcaster (threshold + CNN)
  4. Generate nowcast catalog from training data
  5. Build survival labels from catalog
  6. Train forecaster ensemble (PatchTST + TimesNet + LSTM)
  7. Train survival model
  8. Build FAISS memory bank
  9. Evaluate full pipeline
  10. Log all experiments to MLflow

SLIDING WINDOW CONSTRUCTION:
  For each timestep t with label y(t):
    X[t] = feature matrix over [t - lookback, t]
    y[t] = label at t + forecast_horizon

  To prevent data leakage:
    TRAIN:      all data before 2024-06-01
    VALIDATION: 2024-06-01 to 2024-09-01
    TEST:       2024-09-01 onwards
    
    NO random splitting — time series must respect temporal order.
    A validation event that happens to share features with a training
    event is NOT leakage; what matters is no future labels in training.

CLASS IMBALANCE:
  Flare events are rare (~1–5% of all samples are flare samples).
  Strategies:
    1. Weighted sampling (oversample flare windows)
    2. Focal Loss (down-weight easy negatives)
    3. Class weights in loss functions
    Avoid SMOTE for time series — it destroys temporal structure.
"""

import numpy as np
import pandas as pd
import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader, WeightedRandomSampler
import lightning as L
from lightning.pytorch.callbacks import (EarlyStopping, ModelCheckpoint,
                                          LearningRateMonitor)
import mlflow
import mlflow.pytorch
import yaml
from loguru import logger
from pathlib import Path


class SolarFlareDataset(torch.utils.data.Dataset):
    """
    Sliding window time series dataset.
    
    Yields: (x_soft, x_hard, y_dict) tuples
    where y_dict contains: flare_label, flare_class, lead_time
    """

    def __init__(self, soft_features: np.ndarray,
                 hard_features: np.ndarray,
                 labels: dict,
                 lookback: int = 1440,
                 step: int = 12):   # Step=12 → new window every 60s
        self.X_soft    = soft_features
        self.X_hard    = hard_features
        self.labels    = labels
        self.lookback  = lookback
        self.step      = step
        self.indices   = list(range(lookback, len(soft_features), step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end   = self.indices[idx]
        start = end - self.lookback
        x_soft  = torch.tensor(self.X_soft[start:end],  dtype=torch.float32)
        x_hard  = torch.tensor(self.X_hard[start:end],  dtype=torch.float32)
        y = {k: torch.tensor(v[end], dtype=torch.float32)
             for k, v in self.labels.items()}
        return x_soft, x_hard, y


class SolarFlareTrainer:
    """
    Manages the full training pipeline for all models.
    """

    def __init__(self, config_path: str = "configs/models.yaml",
                 data_config: str = "configs/data.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Trainer initialized | device={}", self.device)

    class NumpyBatchLoader:
        """Lazy batch loader using sliding_window_view for fast contiguous reads.
        Quacks like a DataLoader for Lightning's internal checks."""
        def __init__(self, X_soft, X_hard, labels, indices, lookback, batch_size, sampler=None):
            # Pre-computed rolling window views (W, L, F) — zero-copy stride manipulation
            # sliding_window_view appends the window dim last: (W, F, L), transpose to (W, L, F)
            self._lookback = lookback
            self._win_soft = np.lib.stride_tricks.sliding_window_view(
                X_soft, lookback, axis=0).transpose(0, 2, 1)
            self._win_hard = np.lib.stride_tricks.sliding_window_view(
                X_hard, lookback, axis=0).transpose(0, 2, 1)
            self._labels = labels
            # Alignment: sliding_window_view[i] = X[i:i+L], dataset starts at index=lookback
            self._window_pos = indices - lookback
            self._batch_size = batch_size
            self._sampler = sampler
            self._n_batches = len(indices) // batch_size
            # Lightning checks these DataLoader attrs
            self.batch_sampler = type('_', (), {'sampler': sampler})()
            self.sampler = sampler
            self.num_workers = 0
            self.dataset = None
            self.prefetch_factor = None
            self.persistent_workers = False
            self.timeout = 0

        def __len__(self):
            return self._n_batches

        def __iter__(self):
            return self._batches()

        def _batches(self):
            # Always iterate over DATASET indices (0..N-1), never window positions
            n = len(self._window_pos)
            it = iter(self._sampler) if self._sampler is not None else iter(range(n))
            batch = []
            for idx in it:
                batch.append(int(idx))
                if len(batch) == self._batch_size:
                    yield self._load(batch)
                    batch = []
            if batch:
                yield self._load(batch)

        def _load(self, batch_idx):
            # batch_idx: list of dataset indices → convert to window positions
            pos = self._window_pos[batch_idx]  # (batch_size,)
            x_soft = torch.tensor(self._win_soft[pos], dtype=torch.float32)
            x_hard = torch.tensor(self._win_hard[pos], dtype=torch.float32)
            label_idx = pos + self._lookback
            y = {k: torch.tensor(v[label_idx], dtype=torch.float32)
                 for k, v in self._labels.items()}
            return x_soft, x_hard, y

    def train_forecaster(self, model, dataset_train, dataset_val,
                         experiment_name: str = "patchtst_run_01",
                         model_name: str = "patchtst"):
        """Train a forecaster model with full logging."""
        cfg = self.cfg["forecaster"].get(model_name, self.cfg["forecaster"]["patchtst"])

        # Weighted sampler for class imbalance (direct numpy slice, not 777K __getitem__)
        train_indices = np.array(dataset_train.indices)
        val_indices   = np.array(dataset_val.indices)
        flare_labels  = dataset_train.labels["flare_label"][train_indices]
        weights       = np.where(flare_labels > 0, 10.0, 1.0)
        sampler       = WeightedRandomSampler(weights, len(weights))

        bs = cfg["batch_size"]
        lookback = dataset_train.lookback  # match dataset's lookback, not model config

        train_loader = self.NumpyBatchLoader(
            dataset_train.X_soft, dataset_train.X_hard, dataset_train.labels,
            train_indices, lookback, bs, sampler)
        val_loader = self.NumpyBatchLoader(
            dataset_val.X_soft, dataset_val.X_hard, dataset_val.labels,
            val_indices, lookback, bs)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"]
                          if "early_stopping_patience" in cfg else 15,
                          mode="min"),
            ModelCheckpoint(dirpath=f"models/checkpoints/{experiment_name}",
                            filename="best",
                            monitor="val_loss",
                            save_top_k=1),
        ]

        trainer = L.Trainer(
            max_epochs=cfg["epochs"],
            callbacks=callbacks,
            accelerator=self.device,
            log_every_n_steps=50,
            deterministic=True,
            logger=False,
        )
        trainer.fit(model, train_loader, val_loader)

        checkpoint_dir = Path(f"models/checkpoints/{experiment_name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "best.ckpt")
        return model