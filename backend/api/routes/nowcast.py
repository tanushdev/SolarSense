"""Nowcast endpoint — detect flares in real-time."""

import pandas as pd
from fastapi import APIRouter, Depends
from loguru import logger

from backend.api.schemas import XRayWindow, NowcastResponse
from backend.models.nowcaster.threshold_detector import ThresholdNowcaster

router = APIRouter()

_nowcaster = None


def get_nowcaster():
    global _nowcaster
    if _nowcaster is None:
        _nowcaster = ThresholdNowcaster()
    return _nowcaster


@router.post("/nowcast", response_model=NowcastResponse)
def nowcast(window: XRayWindow, nowcaster=Depends(get_nowcaster)):
    df = pd.DataFrame({
        "soft_flux": window.soft_flux,
        "hard_flux": window.hard_flux,
    }, index=pd.to_datetime(window.timestamps, unit="s", utc=True))

    catalog = nowcaster.detect(df)
    if len(catalog) > 0:
        ev = catalog.iloc[-1]
        start = pd.Timestamp(ev["start_time"])
        peak = pd.Timestamp(ev["peak_time"])
        end = ev["end_time"]
        duration = (pd.Timestamp(end) - start).total_seconds() if pd.notna(end) else 0.0
        return NowcastResponse(
            is_flare=True,
            flare_class=ev["flare_class"],
            peak_flux=float(ev["peak_soft_flux"]),
            onset_time=start.to_pydatetime(),
            peak_time=peak.to_pydatetime(),
            duration_sec=duration,
            confidence="HIGH",
            channel=ev.get("source", "DUAL").upper(),
        )
    return NowcastResponse(
        is_flare=False,
        flare_class=None,
        peak_flux=None,
        onset_time=None,
        peak_time=None,
        duration_sec=None,
        confidence="LOW",
        channel="NONE",
    )
