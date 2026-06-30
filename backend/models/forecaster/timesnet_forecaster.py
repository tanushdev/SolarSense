"""
TimesNet Forecaster
====================
TimesNet: Temporal 2D-Variation Modeling for Time Series Forecasting.

Reference: Wu et al., "TimesNet: Temporal 2D-Variation Modeling
           for General Time Series Analysis", ICLR 2023.

KEY IDEA:
  Transform 1D time series into 2D tensors by folding along
  multiple periods (FFT-discovered). Apply 2D convolutions to
  capture inter-period and intra-period variations simultaneously.

  For flare forecasting, this captures:
    - Intra-period: fast impulsive variations (seconds to minutes)
    - Inter-period: slow background evolution (minutes to hours)

SIMPLIFIED IMPLEMENTATION:
  Full TimesNet is complex. This implements the core idea:
    1. FFT to find top-k dominant periods
    2. Fold 1D into 2D for each period
    3. Inception-style 2D conv blocks
    4. Adaptive aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from loguru import logger


class InceptionBlock(nn.Module):
    """Multi-scale 2D convolution block."""

    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=padding)
            )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = sum(conv(x) for conv in self.convs)
        return F.relu(self.bn(out))


class TimesNetFlareModule(L.LightningModule):
    """
    TimesNet LightningModule for dual-stream flare forecasting.

    Architecture:
        1. Separate 1D convs for soft and hard X-ray streams
        2. TimesNet 2D blocks on each stream
        3. Cross-attention fusion (same as DualStreamPatchTST)
        4. Classification head

    Usage:
        model = TimesNetFlareModule()
        trainer.fit(model, train_loader, val_loader)
    """

    def __init__(self, lookback: int = 1440, d_model: int = 64,
                 d_ff: int = 128, top_k: int = 5, num_kernels: int = 6,
                 dropout: float = 0.1, learning_rate: float = 0.001,
                 soft_channels: int = 16, hard_channels: int = 16):
        super().__init__()
        self.save_hyperparameters()

        self.lookback = lookback
        self.d_model = d_model
        self.top_k = top_k
        self.learning_rate = learning_rate

        # Input projections for soft and hard streams
        self.soft_proj = nn.Linear(soft_channels, d_model)
        self.hard_proj = nn.Linear(hard_channels, d_model)

        # TimesNet 2D blocks (shared weights for efficiency)
        self.inception = InceptionBlock(d_model, d_model)
        self.conv_out = nn.Conv2d(d_model, d_model, kernel_size=1)

        # Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4,
                                                 batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Output head
        self.fc1 = nn.Linear(d_model * 2, d_ff)
        self.fc2 = nn.Linear(d_ff, 1)
        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.BCEWithLogitsLoss()

    def _timesnet_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        TimesNet core: FFT → period detection → 2D folding → Inception convs.

        x: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = x.shape

        # FFT to find periods
        xf = torch.fft.rfft(x, dim=1)
        freqs = torch.abs(xf).mean(dim=-1)  # (B, T//2+1)
        # Top-k frequencies (excluding DC)
        freqs[:, 0] = 0
        topk_freqs = torch.topk(freqs, self.top_k, dim=1).indices
        periods = (T / (topk_freqs + 1).float()).long().clamp(min=2)

        # Process each period
        outputs = []
        for b in range(B):
            period = periods[b, 0].item()
            period = min(period, T)

            # Fold 1D → 2D: (T, D) → (P, T//P, D)
            pad_len = (period - T % period) % period
            x_pad = F.pad(x[b], (0, 0, 0, pad_len))
            folded = x_pad.reshape(1, -1, period, D)                # (1, N, P, D)
            folded = folded.permute(0, 3, 1, 2)                     # (1, D, N, P)

            # Inception conv + pooling
            out = self.inception(folded)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = out.view(D)                                        # (D,)
            outputs.append(out)

        result = torch.stack(outputs, dim=0)  # (B, D)
        return result.unsqueeze(1).repeat(1, T, 1)  # (B, T, D)

    def forward(self, x_soft: torch.Tensor, x_hard: torch.Tensor) -> torch.Tensor:
        # Standardize along sequence dimension to prevent gradient explosion
        soft_mean = x_soft.mean(dim=1, keepdim=True)
        soft_std  = x_soft.std(dim=1, keepdim=True) + 1e-5
        x_soft = (x_soft - soft_mean) / soft_std

        hard_mean = x_hard.mean(dim=1, keepdim=True)
        hard_std  = x_hard.std(dim=1, keepdim=True) + 1e-5
        x_hard = (x_hard - hard_mean) / hard_std

        # Project to d_model
        s = self.soft_proj(x_soft)   # (B, T, D)
        h = self.hard_proj(x_hard)   # (B, T, D)

        # TimesNet transform
        s_tn = self._timesnet_block(s)
        h_tn = self._timesnet_block(h)

        # Cross-attention fusion
        fused, _ = self.cross_attn(s_tn, h_tn, h_tn)
        fused = self.layer_norm(fused + s_tn)

        # Global pooling + classification (concatenate both streams)
        pooled_s = s_tn.mean(dim=1)
        pooled_h = h_tn.mean(dim=1)
        pooled = torch.cat([pooled_s, pooled_h], dim=-1)
        out = self.dropout(F.relu(self.fc1(pooled)))
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
