"""
PatchTST Forecaster — Dual Stream
===================================
Paper: "A Time Series is Worth 64 Words" (Nie et al., 2023)

WHY PatchTST FOR SOLAR FLARES:
  PatchTST divides the input time series into patches (like ViT for images)
  and applies self-attention across patches, not individual timesteps.
  
  This is physically motivated:
    - Flare precursors span 5–30 MINUTES, not individual 5-second samples
    - Patch size of 16 samples × 5s = 80 seconds per patch
    - With 120-minute lookback, we have ~90 patches
    - Attention across patches captures multi-minute correlations naturally
  
  DUAL-STREAM ARCHITECTURE:
    - SoLEXS encoder (soft X-ray patches)
    - HEL1OS encoder (hard X-ray patches)
    - Cross-attention fusion layer (learns the Neupert relationship)
    - Multi-task prediction heads:
        1. Binary flare/no-flare classifier
        2. Flare class regressor (A/B/C/M/X continuous)
        3. Lead time regressor (minutes to peak)
        4. Survival hazard head (instantaneous flare probability)
    - MC Dropout for uncertainty estimation

TENSOR SHAPES:
  Input:  (batch, seq_len, n_features)
          batch=32, seq_len=1440 (120min at 5s), n_features=40 (per channel)
  
  After patching:
          (batch, n_patches, patch_len)
          n_patches = (1440 - 16) / 8 + 1 = 179
  
  After encoder:
          (batch, n_patches, d_model) = (32, 179, 128)
  
  After cross-attention fusion:
          (batch, n_patches, d_model) = (32, 179, 128)
  
  Output heads:
    flare_prob:   (batch, 1)       sigmoid → P(flare in horizon)
    flare_class:  (batch, 5)       softmax → P(A/B/C/M/X)
    lead_time:    (batch, 1)       regression → minutes to peak
    hazard:       (batch, horizon) hazard function at each future step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L
from einops import rearrange, repeat
import mlflow
import yaml
from loguru import logger


class PatchEmbedding(nn.Module):
    """Divide time series into patches and project to d_model."""
    def __init__(self, patch_len: int, stride: int,
                 in_channels: int, d_model: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.proj      = nn.Linear(patch_len * in_channels, d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = x.unfold(1, self.patch_len, self.stride)
        # x: (batch, n_patches, channels, patch_len)
        x = rearrange(x, 'b n c p -> b n (c p)')
        return self.dropout(self.proj(x))  # (batch, n_patches, d_model)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between SoLEXS and HEL1OS patch sequences.
    Query from soft channel, Key/Value from hard channel.
    Learns the physical relationship: when does HXR predict SXR?
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn     = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, soft_patches, hard_patches):
        # Query = soft (what does SXR need to know from HXR?)
        # Key/Value = hard (the HXR carries precursor information)
        fused, attn_weights = self.attn(
            query=soft_patches,
            key=hard_patches,
            value=hard_patches
        )
        return self.norm(soft_patches + self.dropout(fused)), attn_weights


class DualStreamPatchTST(L.LightningModule):
    """
    Dual-stream PatchTST for solar flare forecasting.
    
    Two independent PatchTST encoders (one per instrument channel)
    fused via cross-attention, with multi-task output heads.
    MC Dropout enabled for uncertainty estimation.
    """

    def __init__(self, config_path: str = "configs/models.yaml",
                 n_soft_features: int = 20,
                 n_hard_features: int = 20):
        super().__init__()
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["forecaster"]["patchtst"]
        
        self.patch_len  = cfg["patch_len"]
        self.stride     = cfg["stride"]
        self.d_model    = cfg["d_model"]
        self.n_heads    = cfg["n_heads"]
        self.n_layers   = cfg["n_layers"]
        self.dropout    = cfg["dropout"]
        self.lr         = cfg["learning_rate"]

        # ── Soft X-ray encoder (SoLEXS) ─────────────────────────────
        self.soft_patch_embed = PatchEmbedding(
            self.patch_len, self.stride,
            n_soft_features, self.d_model, self.dropout)
        
        soft_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads,
            dim_feedforward=cfg["d_ff"], dropout=self.dropout,
            batch_first=True)
        self.soft_encoder = nn.TransformerEncoder(
            soft_layer, num_layers=self.n_layers)

        # ── Hard X-ray encoder (HEL1OS) ──────────────────────────────
        self.hard_patch_embed = PatchEmbedding(
            self.patch_len, self.stride,
            n_hard_features, self.d_model, self.dropout)
        
        hard_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads,
            dim_feedforward=cfg["d_ff"], dropout=self.dropout,
            batch_first=True)
        self.hard_encoder = nn.TransformerEncoder(
            hard_layer, num_layers=self.n_layers)

        # ── Cross-attention fusion ────────────────────────────────────
        self.cross_attention = CrossAttentionFusion(
            self.d_model, self.n_heads, self.dropout)
        
        # ── Pooling ───────────────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ── Multi-task output heads ───────────────────────────────────
        hidden = self.d_model * 2   # Concatenated soft + fused
        self.head_flare_prob   = nn.Linear(hidden, 1)   # Binary
        self.head_flare_class  = nn.Linear(hidden, 5)   # A/B/C/M/X
        self.head_lead_time    = nn.Linear(hidden, 1)   # Minutes
        self.head_hazard       = nn.Linear(hidden, 60)  # 60-step hazard

        # MC Dropout — KEEP dropout=True during inference for uncertainty
        self.mc_dropout = nn.Dropout(p=self.dropout)

        self.save_hyperparameters()

    def forward(self, x_soft, x_hard, return_attention=False):
        """
        Parameters
        ----------
        x_soft : (batch, seq_len, n_soft_features)
        x_hard : (batch, seq_len, n_hard_features)
        
        Returns
        -------
        dict with keys: flare_prob, flare_class_logits, lead_time, hazard
        """
        # Standardize along sequence dimension to prevent gradient explosion
        soft_mean = x_soft.mean(dim=1, keepdim=True)
        soft_std  = x_soft.std(dim=1, keepdim=True) + 1e-5
        x_soft = (x_soft - soft_mean) / soft_std

        hard_mean = x_hard.mean(dim=1, keepdim=True)
        hard_std  = x_hard.std(dim=1, keepdim=True) + 1e-5
        x_hard = (x_hard - hard_mean) / hard_std

        # Encode each stream independently
        soft_patches = self.soft_patch_embed(x_soft)
        hard_patches = self.hard_patch_embed(x_hard)
        
        soft_enc = self.soft_encoder(soft_patches)
        hard_enc = self.hard_encoder(hard_patches)

        # Cross-attention: soft queries hard (HXR → SXR causal)
        fused, attn_weights = self.cross_attention(soft_enc, hard_enc)

        # Pool: (batch, n_patches, d_model) → (batch, d_model)
        soft_pooled  = self.pool(soft_enc.transpose(1, 2)).squeeze(-1)
        fused_pooled = self.pool(fused.transpose(1, 2)).squeeze(-1)

        # Concatenate soft and fused representations
        combined = torch.cat([soft_pooled, fused_pooled], dim=-1)
        combined = self.mc_dropout(combined)   # MC Dropout here

        logits = self.head_flare_prob(combined)
        outputs = {
            "flare_logits":       logits,
            "flare_prob":         torch.sigmoid(logits),
            "flare_class_logits": self.head_flare_class(combined),
            "lead_time":          F.softplus(self.head_lead_time(combined)),
            "hazard":             torch.sigmoid(self.head_hazard(combined)),
        }
        if return_attention:
            outputs["attention_weights"] = attn_weights
        return outputs

    def predict_with_uncertainty(self, x_soft, x_hard,
                                 n_passes: int = 50) -> dict:
        """
        Monte Carlo Dropout inference.
        Keep model in train mode to activate dropout.
        Run N forward passes, return mean + std.
        """
        self.train()   # IMPORTANT: activates dropout
        with torch.no_grad():
            preds = [self(x_soft, x_hard)["flare_prob"].cpu().numpy()
                     for _ in range(n_passes)]
        self.eval()
        preds = np.stack(preds, axis=0)  # (n_passes, batch, 1)
        return {
            "probability":  preds.mean(axis=0),
            "uncertainty":  preds.std(axis=0),
            "lower_bound":  np.percentile(preds, 5, axis=0),
            "upper_bound":  np.percentile(preds, 95, axis=0),
        }

    def training_step(self, batch, batch_idx):
        x_soft, x_hard, y = batch
        out = self(x_soft, x_hard)
        loss = self._compute_loss(out, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_soft, x_hard, y = batch
        out = self(x_soft, x_hard)
        loss = self._compute_loss(out, y)
        self.log("val_loss", loss, prog_bar=True)

    def _compute_loss(self, out, y):
        """
        Multi-task loss:
          L = λ₁ * BCE(flare_prob) +
              λ₂ * CrossEntropy(flare_class) +
              λ₃ * MSE(lead_time) +
              λ₄ * PhysicsLoss(hazard)
        """
        y_flare  = y["flare_label"].float()
        y_class  = y["flare_class"].long().clamp(min=0)  # -1 → 0 for safety
        y_lead   = y["lead_time"].float()

        logits = out.get("flare_logits", out.get("flare_prob"))
        if "flare_logits" in out:
            bce = F.binary_cross_entropy_with_logits(
                logits.squeeze(), y_flare)
        else:
            bce = F.binary_cross_entropy(
                logits.squeeze(), y_flare)

        # Cross-entropy on flare samples only (mask non-flare)
        flare_mask = y_flare > 0
        if flare_mask.sum() > 0:
            ce = F.cross_entropy(
                out["flare_class_logits"][flare_mask],
                y_class[flare_mask])
        else:
            ce = torch.tensor(0., device=y_flare.device)

        mse  = F.mse_loss(
            out["lead_time"].squeeze()[flare_mask],
            y_lead[flare_mask]) if flare_mask.sum() > 0 else torch.tensor(0., device=y_flare.device)

        # Physics loss: hazard should be monotonically increasing before peak
        hazard = out["hazard"]
        phys = F.relu(hazard[:, :-1] - hazard[:, 1:]).mean()

        return bce + 0.5 * ce + 0.3 * mse + 0.1 * phys

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
        return [opt], [sched]