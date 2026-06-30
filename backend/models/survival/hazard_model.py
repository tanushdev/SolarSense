"""
Neural Survival Model — Time-to-Flare
======================================
Instead of binary: "Will a flare occur in the next 30 minutes?"
We model: "What is the instantaneous probability (hazard) of a flare
          at time t, given no flare has occurred before t?"

This is the survival analysis formulation.

PHYSICAL INTERPRETATION:
  h(t) = hazard rate at time t = P(flare at t | no flare before t)
  S(t) = survival function = P(no flare before t) = exp(-∫₀ᵗ h(s)ds)
  
  A rising hazard over 10–30 minutes IS the precursor signal.
  The forecast lead time = time from h(t) > threshold until actual flare.

ADVANTAGES OVER BINARY CLASSIFICATION:
  1. No fixed time window — the model decides when the risk becomes critical
  2. Continuous probability over time, not just yes/no
  3. Lead time emerges naturally from the hazard curve shape
  4. Handles censored events (observation ends before flare)

IMPLEMENTATION:
  Uses a neural Cox Proportional Hazard model.
  Features (physics features + PatchTST embeddings) map to log-hazard.
  Loss: Cox partial likelihood (Breslow approximation for ties).

LABELS:
  For each 5-second sample t:
    - time_to_next_flare: minutes until next detected flare (from nowcast catalog)
    - event_occurred: 1 if a flare occurred within the observation window
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from loguru import logger


class NeuralCoxHazard(nn.Module):
    """
    Neural Cox PH model for time-to-flare estimation.
    log h(t | x) = log h₀(t) + f(x; θ)
    where f(x; θ) is the neural network mapping features to log-hazard ratio.
    """

    def __init__(self, n_features: int,
                 hidden_layers=(128, 64, 32),
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = n_features
        for hidden in hidden_layers:
            layers += [
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))  # log-hazard ratio
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (batch, 1) — log hazard ratio


def cox_partial_likelihood_loss(log_hazard, durations, events):
    """
    Breslow approximation of Cox partial likelihood.
    
    Parameters
    ----------
    log_hazard : (N,) predicted log hazard ratios
    durations  : (N,) time to event or censoring (minutes)
    events     : (N,) 1=event occurred, 0=censored
    
    Returns
    -------
    Negative partial log-likelihood (minimize this)
    """
    # Sort by duration descending
    sort_idx   = torch.argsort(durations, descending=True)
    log_h      = log_hazard[sort_idx]
    evt        = events[sort_idx]

    log_cumsum = torch.logcumsumexp(log_h, dim=0)
    loss       = -torch.mean((log_h - log_cumsum) * evt)
    return loss


def build_survival_labels(df: pd.DataFrame,
                          catalog: pd.DataFrame,
                          horizon_minutes: int = 60) -> pd.DataFrame:
    """
    For each timestep in df, compute:
      - time_to_next_flare: minutes until next flare start
      - event_in_horizon:   1 if flare occurs within horizon_minutes
    
    Parameters
    ----------
    df      : aligned time series DataFrame
    catalog : nowcast catalog from ThresholdNowcaster
    horizon : forecast horizon in minutes
    """
    labels = pd.DataFrame(index=df.index)
    labels["time_to_next_flare"] = np.inf
    labels["event_in_horizon"]   = 0

    for _, event in catalog.iterrows():
        flare_time = pd.Timestamp(event["start_time"])
        # All timesteps within horizon_minutes before flare start
        mask = ((flare_time - df.index) >= pd.Timedelta(seconds=0)) & \
               ((flare_time - df.index) <= pd.Timedelta(minutes=horizon_minutes))
        time_to = ((flare_time - df.index[mask])
                   .total_seconds() / 60).values
        labels.loc[mask, "time_to_next_flare"] = np.minimum(
            labels.loc[mask, "time_to_next_flare"].values, time_to)
        labels.loc[mask, "event_in_horizon"] = 1

    labels["time_to_next_flare"] = labels["time_to_next_flare"].replace(
        np.inf, horizon_minutes)
    return labels