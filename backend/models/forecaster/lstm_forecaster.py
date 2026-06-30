"""
LSTM Forecaster
================
Bidirectional LSTM for time series flare prediction.

WHY LSTM:
  - Naturally captures temporal dependencies in X-ray flux time series
  - Bidirectional processing sees both past and (near-)future context
  - 3-layer architecture captures multi-timescale patterns
    (short impulsive, medium gradual, long-term background trends)
  - Handles variable-length sequences via padding

PHYSICS ADVANTAGE:
  The LSTM hidden state at each time step encodes the recent history
  of both soft and hard X-ray channels. This is analogous to how a
  solar physicist visually integrates the last N minutes of light curve
  data to assess flare risk — the LSTM learns this integration.

ARCHITECTURE:
  Input: (batch, lookback, n_features)
    → Bidirectional LSTM (hidden=256, layers=3)
    → Dropout(0.3)
    → FC(512 → 256)
    → FC(256 → 1) with sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from loguru import logger
import numpy as np


class LSTMFlareModule(L.LightningModule):
    """
    LightningModule for LSTM-based flare forecasting.

    Usage:
        model = LSTMFlareModule(input_dim=81, hidden_dim=256,
                                num_layers=3, learning_rate=0.001)
        trainer.fit(model, train_loader, val_loader)
        probs = torch.sigmoid(model(X))
    """

    def __init__(self, input_dim: int = 81, hidden_dim: int = 256,
                 num_layers: int = 3, bidirectional: bool = True,
                 dropout: float = 0.3, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        n_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_out_dim = hidden_dim * n_directions
        self.fc1 = nn.Linear(lstm_out_dim, lstm_out_dim // 2)
        self.fc2 = nn.Linear(lstm_out_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x_soft: torch.Tensor, x_hard: torch.Tensor) -> torch.Tensor:
        # Concatenate soft and hard features along feature dimension
        x = torch.cat([x_soft, x_hard], dim=-1)  # (B, T, F_soft + F_hard)
        lstm_out, _ = self.lstm(x)
        # Use the last time step output
        last = lstm_out[:, -1, :]
        out = self.dropout(F.relu(self.fc1(last)))
        logits = self.fc2(out).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        x_soft, x_hard, y = batch
        logits = self.forward(x_soft, x_hard)
        loss = self.criterion(logits, y["flare_label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_soft, x_hard, y = batch
        logits = self.forward(x_soft, x_hard)
        loss = self.criterion(logits, y["flare_label"])
        self.log("val_loss", loss, prog_bar=True)
        return {"logits": logits.detach(), "labels": y["flare_label"].detach()}

    def predict_step(self, batch, batch_idx):
        x_soft, x_hard, _ = batch
        logits = self.forward(x_soft, x_hard)
        return torch.sigmoid(logits)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return [opt], [scheduler]
