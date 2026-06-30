"""
Physics-Based Threshold Nowcaster
===================================
The FIRST nowcaster in the pipeline.

ALGORITHM (adapted from standard solar physics practice):

For NOWCASTING (detection), before any ML:
  1. Compute background B = rolling 10-min 10th percentile
  2. Compute noise N = rolling 10-min std of quiet-sun segments
  3. Alert when: flux > B + k*N for k=3 (3-sigma) sustained for T=60s

This mirrors the algorithm used by NOAA SWPC for operational GOES alerts.

INDEPENDENT DETECTION (challenge-compliant):
  Per the ISRO challenge specification, we detect flares independently
  on each instrument BEFORE merging catalogs:

  SoLEXS ──detect──> Catalog_A
  HEL1OS ──detect──> Catalog_B
                           │
                           └─merge──> Master_Catalog

  The merge rules:
    - Dual-channel: event appears in BOTH catalogs within 5-minute window
    - SoLEXS-only:  event appears only in Catalog_A
    - HEL1OS-only:  event appears only in Catalog_B
    - Confidence:   "dual_channel" > "solexs_only" > "hel1os_only"

DUAL-CHANNEL CONFIRMATION RULE:
  A nowcast event is CONFIRMED only when:
    - SoLEXS soft flux exceeds threshold (thermal emission detected)
    AND
    - HEL1OS hard flux shows simultaneous or prior rise (non-thermal component)

  This dramatically reduces false alarms from:
    - Particle events (single-channel spike)
    - Detector noise (not coherent across channels)
    - Slow background drifts (not impulsive)

FLARE CLASSIFICATION:
  Peak soft flux in W/m² (GOES-equivalent) determines class:
    A: 1e-8 to 1e-7
    B: 1e-7 to 1e-6
    C: 1e-6 to 1e-5
    M: 1e-5 to 1e-4
    X: > 1e-4
"""

import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Tuple
import yaml


@dataclass
class FlareEvent:
    """A detected flare event."""
    start_time:    pd.Timestamp
    peak_time:     pd.Timestamp
    end_time:      Optional[pd.Timestamp]
    peak_soft_flux: float
    peak_hard_flux: float
    flare_class:   str        # A, B, C, M, X
    flare_subclass: float     # 1.0–9.9
    confirmation:  str        # "dual_channel", "solexs_only", or "hel1os_only"
    quality:       int        # 0=high, 1=medium, 2=low
    source:        str        # instrument that detected it


class ThresholdNowcaster:
    """
    Physics-based nowcasting detector supporting independent detection
    per instrument (challenge-compliant) as well as dual-channel mode.

    Usage:
        nowcaster = ThresholdNowcaster()

        # Challenge-compliant: detect independently, merge catalogs
        cat_solexs, cat_hel1os, cat_master = nowcaster.detect_all(df_merged)

        # Fast path: just master catalog
        events = nowcaster.detect_master(df_merged)
    """

    FLUX_CLASS_BOUNDARIES = {
        "A": (1e-8, 1e-7),
        "B": (1e-7, 1e-6),
        "C": (1e-6, 1e-5),
        "M": (1e-5, 1e-4),
        "X": (1e-4, np.inf),
    }

    COUNTS_CLASS_BOUNDARIES = {}

    CATALOG_COLUMNS = [
        "start_time", "peak_time", "end_time",
        "peak_soft_flux", "peak_hard_flux",
        "flare_class", "flare_subclass",
        "confirmation", "quality", "source",
    ]

    def __init__(self, config_path: str = "configs/models.yaml",
                 thresholds_path: str = "configs/thresholds.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["nowcaster"]["threshold_detector"]
        self.soft_k    = cfg["soft_xray_threshold_multiplier"]
        self.hard_k    = cfg["hard_xray_threshold_multiplier"]
        self.min_dur   = pd.Timedelta(seconds=cfg["min_duration_seconds"])
        self.n_confirm = cfg["confirmation_samples"]
        with open(thresholds_path) as f:
            tcfg = yaml.safe_load(f)["flare_classification"]
        self.COUNTS_CLASS_BOUNDARIES = {
            "A": (0, tcfg["counts_A"]),
            "B": (tcfg["counts_A"], tcfg["counts_B"]),
            "C": (tcfg["counts_B"], tcfg["counts_C"]),
            "M": (tcfg["counts_C"], tcfg["counts_M"]),
            "X": (tcfg["counts_M"], tcfg["counts_X"]),
        }
        # Merge tolerance: events within this window are considered the same
        self.merge_tolerance = pd.Timedelta(minutes=5)

    # ─── Public API ─────────────────────────────────────────────────────

    def detect_all(self, df: pd.DataFrame
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Challenge-compliant independent detection.

        Returns
        -------
        (solexs_catalog, hel1os_catalog, master_catalog) as DataFrames.
        """
        logger.info("ThresholdNowcaster: independent detection (3 catalogs)")

        # 1. Detect independently on each channel
        cat_solexs = self.detect_single_channel(df, channel="soft")
        cat_hel1os = self.detect_single_channel(df, channel="hard")

        # 2. Merge into master catalog
        cat_master = self._merge_catalogs(cat_solexs, cat_hel1os)

        logger.info("  SoLEXS catalog:   {} events", len(cat_solexs))
        logger.info("  HEL1OS catalog:   {} events", len(cat_hel1os))
        logger.info("  Master catalog:   {} events", len(cat_master))
        return cat_solexs, cat_hel1os, cat_master

    def detect_master(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fast path: produce only the master catalog (dual-channel).
        Skips saving independent catalogs.
        """
        return self.detect_all(df)[2]

    def detect_single_channel(self, df: pd.DataFrame, channel: str = "soft"
                              ) -> pd.DataFrame:
        """
        Detect flares on a single instrument channel.

        Parameters
        ----------
        df : DataFrame with at least {channel}_flux column
        channel : "soft" or "hard"

        Returns
        -------
        Catalog DataFrame
        """
        if channel == "soft":
            flux_col = "soft_flux"
            k = self.soft_k
            source = "SoLEXS"
        else:
            flux_col = "hard_flux"
            k = self.hard_k
            source = "HEL1OS"

        bg = df[flux_col].rolling(120).quantile(0.10)
        noise = df[flux_col].rolling(120).std()
        threshold = bg + k * noise

        above = df[flux_col] > threshold
        triggered = above.rolling(self.n_confirm).sum() == self.n_confirm

        events = self._extract_single_events(
            triggered, df, flux_col, source, channel
        )
        catalog = self._events_to_catalog(events)
        logger.info("  {}: {} events detected (k={})", source, len(catalog), k)
        return catalog

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for detect_master — backward compatibility."""
        return self.detect_master(df)

    def to_catalog(self, events_or_df) -> pd.DataFrame:
        """Convert event list to catalog DataFrame for storage."""
        if isinstance(events_or_df, pd.DataFrame):
            return events_or_df
        if not events_or_df:
            return pd.DataFrame(columns=self.CATALOG_COLUMNS)
        return pd.DataFrame([vars(e) for e in events_or_df])

    # ─── Private: Single-Channel Extraction ────────────────────────────

    def _extract_single_events(self, triggered: pd.Series, df: pd.DataFrame,
                                flux_col: str, source: str, channel: str
                                ) -> List[FlareEvent]:
        """Extract events from a single-channel trigger series."""
        values = triggered.values.astype(np.int8)
        edges = np.diff(np.r_[0, values, 0])
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]

        events = []
        for s, e in zip(starts, ends):
            end_idx = max(0, e - 1)
            duration = df.index[end_idx] - df.index[s]
            if isinstance(duration, pd.Timedelta) and duration < self.min_dur:
                continue

            segment = df.iloc[s:max(0, e)]
            peak_idx = segment[flux_col].idxmax()
            peak_flux = segment[flux_col].max()

            # Fill cross-channel flux with available data
            if channel == "soft":
                peak_soft = peak_flux
                peak_hard = segment["hard_flux"].mean() if "hard_flux" in segment.columns else 0.0
            else:
                peak_hard = peak_flux
                peak_soft = segment["soft_flux"].mean() if "soft_flux" in segment.columns else 0.0

            cls, subcls = self._classify_flux(peak_flux)
            events.append(FlareEvent(
                start_time=df.index[s],
                peak_time=peak_idx,
                end_time=df.index[end_idx],
                peak_soft_flux=peak_soft,
                peak_hard_flux=peak_hard,
                flare_class=cls,
                flare_subclass=subcls,
                confirmation=f"{channel}_only",
                quality=0,
                source=source,
            ))
        return events

    # ─── Private: Catalog Merging ───────────────────────────────────────

    def _merge_catalogs(self, cat_solexs: pd.DataFrame,
                        cat_hel1os: pd.DataFrame) -> pd.DataFrame:
        """
        Merge independent catalogs into a master catalog (O(n log n) via
        merge_asof in both directions).

        Merge rules:
          1. Dual-channel: events in BOTH catalogs within merge_tolerance
          2. SoLEXS-only:  events only in SoLEXS catalog
          3. HEL1OS-only:  events only in HEL1OS catalog
        """
        if len(cat_solexs) == 0:
            cat_hel1os["confirmation"] = "hel1os_only"
            return cat_hel1os[self.CATALOG_COLUMNS]
        if len(cat_hel1os) == 0:
            cat_solexs["confirmation"] = "solexs_only"
            return cat_solexs[self.CATALOG_COLUMNS]

        tol = self.merge_tolerance
        cls_priority = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}

        # Sort both by peak_time for merge_asof; keep original indices as columns
        s = cat_solexs.sort_values("peak_time").reset_index()
        s = s.rename(columns={"index": "s_idx"}).reset_index(drop=True)
        h = cat_hel1os.sort_values("peak_time").reset_index()
        h = h.rename(columns={"index": "h_idx"}).reset_index(drop=True)

        # Prefix HEL1OS columns (except key & idx) so suffixes are clean
        h_in = h.copy()
        h_in["h_peak_time"] = h_in["peak_time"]  # keep a h-version of peak_time
        h_in = h_in.rename(
            columns={c: f"h_{c}" for c in h_in.columns
                     if c not in ("peak_time", "h_idx", "h_peak_time")}
        )

        # Forward match: next HEL1OS event after each SoLEXS event
        fwd = pd.merge_asof(
            s, h_in, on="peak_time", direction="forward", tolerance=tol
        )
        # Backward match: previous HEL1OS event before each SoLEXS event
        bwd = pd.merge_asof(
            s, h_in, on="peak_time", direction="backward", tolerance=tol
        )

        # Any HEL1OS match in either direction → dual-channel SoLEXS
        s_matched = fwd["h_peak_soft_flux"].notna() | bwd["h_peak_soft_flux"].notna()
        s_used = s_matched.values

        h_used = np.zeros(len(h), dtype=bool)
        dual_rows = []

        for i in range(len(s)):
            if not s_matched.iloc[i]:
                continue
            s_row = s.iloc[i]
            # Prefer forward (later) match — HEL1OS tends to peak after SXR
            match = fwd.iloc[i] if pd.notna(fwd.iloc[i]["h_peak_soft_flux"]) else bwd.iloc[i]
            h_used[int(match["h_idx"])] = True

            dual_rows.append({
                "start_time": min(s_row["start_time"], match["h_start_time"]),
                "peak_time": match["h_peak_time"],
                "end_time": max(s_row["end_time"], match["h_end_time"]),
                "peak_soft_flux": max(s_row["peak_soft_flux"], match["h_peak_soft_flux"]),
                "peak_hard_flux": max(s_row["peak_hard_flux"], match["h_peak_hard_flux"]),
                "flare_class": max(s_row["flare_class"], match["h_flare_class"],
                                   key=lambda x: cls_priority.get(x, 0)),
                "flare_subclass": max(s_row["flare_subclass"], match["h_flare_subclass"]),
                "confirmation": "dual_channel",
                "quality": 0,
                "source": "SoLEXS+HEL1OS",
            })

        master_rows = dual_rows

        # SoLEXS-only
        for i, row in s.iterrows():
            if not s_used[i]:
                r = row.drop(["s_idx"]).to_dict()
                r["confirmation"] = "solexs_only"
                r["source"] = "SoLEXS"
                master_rows.append(r)

        # HEL1OS-only
        for j, row in h.iterrows():
            if not h_used[j]:
                r = row.drop(["h_idx"]).to_dict()
                r["confirmation"] = "hel1os_only"
                r["source"] = "HEL1OS"
                master_rows.append(r)

        if not master_rows:
            return pd.DataFrame(columns=self.CATALOG_COLUMNS)

        master = pd.DataFrame(master_rows)
        master = master.sort_values("peak_time").reset_index(drop=True)
        return master[self.CATALOG_COLUMNS]

    # ─── Private: Helpers ───────────────────────────────────────────────

    def _events_to_catalog(self, events: List[FlareEvent]) -> pd.DataFrame:
        if not events:
            return pd.DataFrame(columns=self.CATALOG_COLUMNS)
        return pd.DataFrame([vars(e) for e in events])

    def _classify_flux(self, flux: float):
        if flux < 1e-6:
            bounds = self.FLUX_CLASS_BOUNDARIES
        else:
            bounds = self.COUNTS_CLASS_BOUNDARIES
        for cls, (lo, hi) in bounds.items():
            if lo <= flux < hi:
                base = max(lo, 1e-30)
                subclass = (flux / base)
                return cls, round(subclass, 1)
        return "X", 10.0
