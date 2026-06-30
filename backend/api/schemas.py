"""Pydantic schemas for all API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class XRayWindow(BaseModel):
    """Input: recent time window of X-ray data."""
    timestamps:   List[float]    = Field(..., description="UNIX timestamps UTC")
    soft_flux:    List[float]    = Field(..., description="SoLEXS soft X-ray flux (W/m²)")
    hard_flux:    List[float]    = Field(..., description="HEL1OS hard X-ray flux (counts/s)")
    soft_counts:  Optional[List[float]] = None
    hard_counts:  Optional[List[float]] = None


class SimilarEvent(BaseModel):
    date:        str
    flare_class: str
    similarity:  float
    lead_time:   float
    description: str


class ForecastResponse(BaseModel):
    """Probabilistic forecast with full uncertainty quantification."""
    model_config = {'protected_namespaces': ()}
    timestamp:         datetime
    flare_probability: float   = Field(..., ge=0, le=1)
    uncertainty:       float   = Field(..., ge=0, description="Std of MC Dropout passes")
    lower_bound:       float   = Field(..., ge=0, le=1, description="5th percentile")
    upper_bound:       float   = Field(..., ge=0, le=1, description="95th percentile")
    predicted_class:   str     = Field(..., description="Most likely flare class A/B/C/M/X")
    class_probs:       dict    = Field(..., description="Probability per class")
    lead_time_minutes: float   = Field(..., description="Estimated minutes to flare peak")
    alert_level:       str     = Field(..., description="GREEN/YELLOW/ORANGE/RED")
    similar_events:    List[SimilarEvent] = []
    physics_reason:    str = ""
    model:             str = "unknown"
    data_timestamp:    str = ""
    attention_weights: Optional[List[float]] = None
    model_version:     str = "1.0.0"


class NowcastResponse(BaseModel):
    """Detected flare event."""
    is_flare:      bool
    flare_class:   Optional[str]
    peak_flux:     Optional[float]
    onset_time:    Optional[datetime]
    peak_time:     Optional[datetime]
    duration_sec:  Optional[float]
    confidence:    str   # HIGH / MEDIUM / LOW
    channel:       str   # DUAL / SOFT_ONLY / HARD_ONLY