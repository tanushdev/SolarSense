# SolarSense-AI
## Master Engineering Specification for Claude Code
### Solar Flare Nowcasting & Forecasting — Aditya-L1 (SoLEXS + HEL1OS)

**Version:** 1.0  
**Author:** Senior Space Physicist / ML Systems Architect  
**Mission:** Build an operational-grade solar flare prediction system using ISRO's Aditya-L1 dual X-ray instruments  
**Language:** Python 3.11  
**Target:** ISRO Hackathon — Evaluation on TPR, FAR, Lead Time  

---

# PART 0 — PHILOSOPHY

This system is not a toy classifier. It is an operational space-weather intelligence engine.

Every design decision below is made by a physicist who has watched flares destroy satellite uplinks, GPS lock, and power grids. The goal is not to maximise accuracy on a leaderboard. The goal is to give a space-weather operator a reliable, explainable, confidence-annotated alert — early enough to act.

Design axioms:
- **Physics first.** Every feature must have a physical justification.
- **Uncertainty always.** Never output a prediction without a confidence bound.
- **Dual-stream always.** SoLEXS and HEL1OS are processed separately and fused. Never concatenate raw inputs.
- **Lead time matters more than accuracy.** A correct warning 10 minutes early beats a perfect label 0 seconds early.
- **Explainability for operators.** Every alert must show *why* the model fired.
- **No magic numbers.** Every threshold, window size, and hyperparameter is in config YAML.
- **Reproducibility.** Every experiment is seeded, logged, and versioned.

---

# PART 1 — REPOSITORY STRUCTURE

```
SolarSense-AI/
│
├── SOLARSENSE_AI_MASTER_SPEC.md      ← THIS FILE (Claude Code reads this)
│
├── configs/
│   ├── data.yaml                     ← All data paths, cadences, bands
│   ├── features.yaml                 ← Feature engineering parameters
│   ├── models.yaml                   ← All model hyperparameters
│   ├── thresholds.yaml               ← Detection thresholds per flare class
│   └── deployment.yaml               ← API, dashboard, ports, alerts
│
├── data/
│   ├── raw/
│   │   ├── solexs/                   ← Raw SoLEXS FITS files (Level-1)
│   │   └── hel1os/                   ← Raw HEL1OS FITS files (Level-1)
│   ├── processed/
│   │   ├── solexs_timeseries.parquet
│   │   ├── hel1os_timeseries.parquet
│   │   └── merged_timeseries.parquet
│   ├── catalogs/
│   │   ├── nowcast_catalog.csv       ← Auto-generated flare database
│   │   ├── goes_reference.csv        ← GOES flare catalog for validation
│   │   └── historical_events.parquet ← Memory bank for FAISS retrieval
│   └── external/
│       └── goes_xrs/                 ← Supplementary GOES XRS data
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── solexs_reader.py          ← SoLEXS FITS parser
│   │   ├── hel1os_reader.py          ← HEL1OS FITS parser
│   │   ├── goes_reader.py            ← GOES XRS supplementary parser
│   │   ├── aligner.py                ← Temporal alignment of dual streams
│   │   ├── cleaner.py                ← Isolation Forest + artifact removal
│   │   └── dataset_builder.py        ← Unified dataset factory
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── physics_features.py       ← All physics-grounded features
│   │   ├── spectral_features.py      ← FFT, wavelet decomposition
│   │   ├── matrix_profile.py         ← STUMPY motif/discord detection
│   │   ├── shapelet_miner.py         ← Precursor shapelet mining
│   │   └── feature_pipeline.py       ← Master feature orchestrator
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py             ← Abstract base all models inherit
│   │   ├── nowcaster/
│   │   │   ├── __init__.py
│   │   │   ├── threshold_detector.py ← Physics-based nowcast detector
│   │   │   └── cnn_nowcaster.py      ← 1D CNN flare classifier
│   │   ├── forecaster/
│   │   │   ├── __init__.py
│   │   │   ├── patchtst_forecaster.py  ← PatchTST transformer forecaster
│   │   │   ├── timesnet_forecaster.py  ← TimesNet 2D time-series model
│   │   │   ├── lstm_forecaster.py      ← Bidirectional LSTM baseline
│   │   │   └── ensemble_forecaster.py  ← Weighted ensemble combiner
│   │   ├── bayesian/
│   │   │   ├── __init__.py
│   │   │   └── uncertainty.py        ← MC Dropout + conformal prediction
│   │   ├── survival/
│   │   │   ├── __init__.py
│   │   │   └── hazard_model.py       ← Neural survival (time-to-flare)
│   │   └── memory/
│   │       ├── __init__.py
│   │       └── flare_memory.py       ← FAISS historical event retrieval
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                ← Master training loop
│   │   ├── losses.py                 ← Custom losses incl. physics loss
│   │   ├── metrics.py                ← TSS, HSS, Brier, ECE, Lead Time
│   │   ├── callbacks.py              ← Early stopping, checkpointing
│   │   └── experiment_logger.py      ← MLflow experiment tracking
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py              ← Full evaluation suite
│   │   ├── calibration.py            ← Brier score, ECE, reliability curves
│   │   └── explainer.py              ← Attention maps, SHAP values
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── nowcast_pipeline.py       ← End-to-end nowcasting pipeline
│   │   └── forecast_pipeline.py      ← End-to-end forecasting pipeline
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py                   ← FastAPI application entry
│       ├── routes/
│       │   ├── predict.py            ← /predict endpoint
│       │   ├── nowcast.py            ← /nowcast endpoint
│       │   ├── health.py             ← /health endpoint
│       │   └── history.py            ← /history endpoint
│       └── schemas.py                ← Pydantic response models
│
├── dashboard/
│   ├── app.py                        ← Streamlit dashboard
│   ├── components/
│   │   ├── light_curve_plot.py       ← Real-time dual-stream plot
│   │   ├── alert_panel.py            ← Alert display with confidence
│   │   ├── attention_map.py          ← Which timesteps triggered alert
│   │   └── flare_memory_panel.py     ← Similar historical events
│   └── assets/
│       └── style.css
│
├── scripts/
│   ├── download_data.py              ← ISSDC PRADAN download helper
│   ├── build_dataset.py              ← Run full data pipeline
│   ├── extract_features.py           ← Feature extraction runner
│   ├── train_nowcaster.py            ← Train nowcasting models
│   ├── train_forecaster.py           ← Train forecasting models
│   ├── build_memory_bank.py          ← Build FAISS event index
│   ├── evaluate.py                   ← Full evaluation runner
│   └── run_inference.py              ← Single-file inference demo
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_flare_morphology.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_matrix_profile_analysis.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_survival_analysis.ipynb
│
├── tests/
│   ├── test_solexs_reader.py
│   ├── test_hel1os_reader.py
│   ├── test_aligner.py
│   ├── test_features.py
│   ├── test_cleaner.py
│   ├── test_nowcaster.py
│   ├── test_forecaster.py
│   └── test_api.py
│
├── experiments/
│   └── mlruns/                       ← MLflow experiment logs
│
├── models/
│   └── checkpoints/                  ← Saved model weights
│
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# PART 2 — TECHNOLOGY STACK

## Core Language
```
Python 3.11
```

## Data & IO
```
astropy==6.1          ← FITS file reading (SoLEXS, HEL1OS are FITS)
astroquery==0.4.7     ← GOES XRS supplementary data
pandas==2.2.2         ← DataFrames
numpy==1.26.4         ← Numerical operations
pyarrow==16.1.0       ← Parquet storage (fast columnar IO)
h5py==3.11.0          ← HDF5 support for some ISRO formats
```

## Signal Processing
```
scipy==1.13.1         ← scipy.signal for filtering, wavelet transforms
PyWavelets==1.6.0     ← Discrete wavelet transform for decomposition
stumpy==1.13.0        ← Matrix Profile (STUMPY is the gold standard)
tslearn==0.6.3        ← Shapelet mining and time series ML
```

## Machine Learning
```
scikit-learn==1.5.1   ← IsolationForest, preprocessing, metrics
xgboost==2.0.3        ← Gradient boosting baseline
lightgbm==4.4.0       ← Fast gradient boosting
```

## Deep Learning
```
torch==2.3.1          ← PyTorch (primary DL framework)
torchvision==0.18.1   ← Utility transforms
lightning==2.3.1      ← PyTorch Lightning for clean training loops
einops==0.8.0         ← Tensor operations for transformer models
```

## Transformers & Time Series Models
```
# PatchTST — install from source
# git+https://github.com/yuqinie98/PatchTST.git

# TimesNet — implement from paper directly (no stable pip package)
# Implementation included in src/models/forecaster/timesnet_forecaster.py
```

## Uncertainty & Probabilistic
```
torchmetrics==1.4.0   ← TSS, HSS, calibration metrics
uncertainty-toolbox==0.1.2  ← ECE, calibration curves
lifelines==0.29.0     ← Survival analysis (Cox PH, Kaplan-Meier)
```

## Vector Search (Flare Memory)
```
faiss-cpu==1.8.0      ← Historical flare similarity search
```

## Experiment Tracking
```
mlflow==2.14.1        ← Experiment logging and model registry
```

## API & Serving
```
fastapi==0.111.0      ← REST API
uvicorn==0.30.1       ← ASGI server
pydantic==2.7.4       ← Request/response validation
websockets==12.0      ← Real-time streaming to dashboard
```

## Dashboard
```
streamlit==1.36.0     ← Operational dashboard
plotly==5.22.0        ← Interactive light curve plots
altair==5.3.0         ← Attention map visualizations
```

## Configuration & Utilities
```
pyyaml==6.0.1         ← YAML config loading
loguru==0.7.2         ← Structured logging
rich==13.7.1          ← Beautiful CLI output
hydra-core==1.3.2     ← Config management with overrides
tqdm==4.66.4          ← Progress bars
```

---

# PART 3 — CONFIGURATION FILES

## `configs/data.yaml`

```yaml
solexs:
  raw_dir: "data/raw/solexs"
  level: "L1"
  cadence_seconds: 1          # SoLEXS native cadence
  resample_seconds: 5         # Resample to 5s for alignment
  energy_bands:
    soft_low_kev: 1.6
    soft_high_kev: 12.0
  columns:
    time: "TIME"
    flux: "FLUX"
    counts: "COUNTS"
    energy: "ENERGY"
  fits_extension: 1           # FITS HDU extension index

hel1os:
  raw_dir: "data/raw/hel1os"
  level: "L1"
  cadence_seconds: 1
  resample_seconds: 5
  energy_bands:
    hard_low_kev: 10.0
    hard_high_kev: 150.0
  columns:
    time: "TIME"
    flux: "FLUX"
    counts: "COUNTS"
  fits_extension: 1

alignment:
  method: "linear_interpolation"   # or "nearest"
  max_gap_seconds: 30              # Flag gaps larger than this
  reference_instrument: "solexs"  # HEL1OS resampled to SoLEXS grid

storage:
  processed_dir: "data/processed"
  catalog_dir: "data/catalogs"
  format: "parquet"
  compression: "snappy"

goes:
  xrs_dir: "data/external/goes_xrs"
  use_for_validation: true
  catalog_url: "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/"
```

## `configs/features.yaml`

```yaml
windows:
  short_minutes: 5
  medium_minutes: 15
  long_minutes: 60
  forecast_lookback_minutes: 120

physics:
  hardness_ratio:
    enabled: true
    epsilon: 1e-10              # Avoid division by zero
  flux_derivative:
    enabled: true
    order: [1, 2]               # First and second derivatives
    smoothing_window: 5         # Apply before differencing
  spectral_index:
    enabled: true
  rise_rate:
    enabled: true
    window_minutes: 5
  decay_rate:
    enabled: true
    window_minutes: 10
  cross_correlation:
    enabled: true
    max_lag_seconds: 120
  background_subtraction:
    enabled: true
    percentile: 10              # Use 10th percentile as background
    window_minutes: 60

spectral:
  fft:
    enabled: true
    top_k_frequencies: 10
  wavelet:
    enabled: true
    wavelet_name: "db4"
    levels: 5
    decompose_into: ["trend", "seasonality", "residual"]

matrix_profile:
  enabled: true
  window_sizes: [30, 60, 120]   # In samples at 5s cadence = 2.5, 5, 10 min
  n_motifs: 5
  n_discords: 5

normalization:
  method: "robust_scaler"       # RobustScaler handles outliers better than StandardScaler
  fit_on: "training_set_only"
```

## `configs/models.yaml`

```yaml
nowcaster:
  threshold_detector:
    enabled: true
    soft_xray_threshold_multiplier: 3.0   # sigma above background
    hard_xray_threshold_multiplier: 2.5
    min_duration_seconds: 60
    confirmation_samples: 3               # Must exceed threshold N consecutive times
    flare_classes:
      A: 1e-8
      B: 1e-7
      C: 1e-6
      M: 1e-5
      X: 1e-4

  cnn_nowcaster:
    enabled: true
    input_window_samples: 60              # 60 * 5s = 5 minutes
    channels: [32, 64, 128]
    kernel_sizes: [5, 3, 3]
    dropout: 0.3
    batch_size: 64
    learning_rate: 1e-3
    epochs: 100
    early_stopping_patience: 10

forecaster:
  lookback_minutes: 120
  lookback_samples: 1440                  # 120 min * 12 samples/min (5s cadence)
  forecast_horizons_minutes: [5, 10, 15, 30, 60]

  patchtst:
    enabled: true
    patch_len: 16
    stride: 8
    d_model: 128
    n_heads: 8
    n_layers: 3
    d_ff: 256
    dropout: 0.2
    batch_size: 32
    learning_rate: 5e-4
    epochs: 150
    scheduler: "cosine_annealing"
    warmup_epochs: 10

  timesnet:
    enabled: true
    d_model: 64
    d_ff: 128
    top_k: 5                    # Top-k frequencies for 2D transform
    num_kernels: 6
    dropout: 0.1
    batch_size: 32
    learning_rate: 1e-3
    epochs: 100

  lstm:
    enabled: true
    hidden_size: 256
    num_layers: 3
    bidirectional: true
    dropout: 0.3
    batch_size: 64
    learning_rate: 1e-3
    epochs: 100

  ensemble:
    method: "learned_weighting"           # or "simple_average" or "bayesian_model_averaging"
    base_models: ["patchtst", "timesnet", "lstm"]

bayesian:
  method: "mc_dropout"
  n_forward_passes: 50                    # MC Dropout samples
  conformal_alpha: 0.1                    # 90% coverage conformal intervals

survival:
  model: "neural_cox"                     # Neural Cox PH model
  hidden_layers: [128, 64, 32]
  dropout: 0.2
  batch_size: 64
  learning_rate: 1e-3
  epochs: 100

memory:
  faiss_index_type: "Flat"               # or "IVFFlat" for large datasets
  embedding_dim: 128
  top_k_similar: 5
  similarity_metric: "cosine"
```

## `configs/thresholds.yaml`

```yaml
flare_classification:
  GOES_A: 1.0e-8    # W/m² in 1-8 Angstrom band
  GOES_B: 1.0e-7
  GOES_C: 1.0e-6
  GOES_M: 1.0e-5
  GOES_X: 1.0e-4

alert:
  probability_threshold: 0.5
  high_confidence_threshold: 0.8
  uncertainty_max_acceptable: 0.15       # Suppress alert if uncertainty too high

operational:
  false_alarm_budget_per_day: 2          # Target: at most 2 false alarms/day
  minimum_lead_time_minutes: 5           # Don't alert unless lead time >= 5 min
```

---

# PART 4 — DATA PIPELINE

## `src/data/solexs_reader.py`

**Purpose:** Read SoLEXS Level-1 FITS files, extract flux time series, validate headers, return clean DataFrame.

**Inputs:** Path to FITS file or directory of FITS files  
**Outputs:** `pd.DataFrame` with columns `[timestamp_utc, soft_flux, soft_counts, energy_mean, quality_flag]`  
**Never:** Apply ML preprocessing here. This module is data ingestion only.

```python
"""
SoLEXS FITS Reader
==================
Reads ISRO Aditya-L1 SoLEXS Level-1 FITS files.

SoLEXS measures soft X-rays in 1.6–12.0 keV.
Level-1 FITS structure (from ISRO documentation):
  - HDU 0: Primary header (mission metadata)
  - HDU 1: Binary table (TIME, FLUX, COUNTS, ENERGY columns)

Physical context:
  Soft X-rays (1.6–12 keV) from SoLEXS trace the thermal emission from
  hot coronal plasma (T > 10^7 K). During a flare, soft X-ray flux rises
  on timescales of minutes (the gradual phase) and is the standard
  classification band (GOES 1-8 Angstrom equivalent).
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from pathlib import Path
from loguru import logger
from typing import Union, List, Optional
import yaml


class SoLEXSReader:
    """
    Reads and parses SoLEXS Level-1 FITS files from ISRO ISSDC PRADAN.

    Usage:
        reader = SoLEXSReader(config_path="configs/data.yaml")
        df = reader.read_file("data/raw/solexs/solexs_20240101.fits")
        df_merged = reader.read_directory("data/raw/solexs/")
    """

    REQUIRED_COLUMNS = ["TIME", "FLUX", "COUNTS"]
    QUALITY_GOOD = 0

    def __init__(self, config_path: str = "configs/data.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["solexs"]
        self.hdu_ext = self.config["fits_extension"]
        self.resample = f"{self.config['resample_seconds']}s"
        logger.info("SoLEXSReader initialized | resample={}", self.resample)

    def read_file(self, fits_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read a single SoLEXS FITS file.

        Parameters
        ----------
        fits_path : path to .fits file

        Returns
        -------
        pd.DataFrame with columns:
            timestamp_utc   : datetime64[ns, UTC]
            soft_flux       : float64  (W/m²/keV, background included)
            soft_counts     : float64  (raw detector counts)
            energy_mean     : float64  (keV)
            quality_flag    : int      (0=good, 1=particle event, 2=eclipse)
            source_file     : str      (origin filename for traceability)
        """
        fits_path = Path(fits_path)
        logger.info("Reading SoLEXS FITS: {}", fits_path.name)

        try:
            with fits.open(fits_path) as hdul:
                self._validate_header(hdul[0].header, fits_path.name)
                table = hdul[self.hdu_ext].data
                df = self._table_to_dataframe(table, fits_path.name)
        except Exception as e:
            logger.error("Failed to read {}: {}", fits_path.name, e)
            raise

        df = self._convert_timestamps(df)
        df = self._resample(df)
        df = self._validate_flux(df)
        logger.info("SoLEXS file loaded: {} rows, {} to {}",
                    len(df), df.index[0], df.index[-1])
        return df

    def read_directory(self, dir_path: Union[str, Path],
                       sort: bool = True) -> pd.DataFrame:
        """
        Read all FITS files in a directory, concatenate chronologically.
        """
        dir_path = Path(dir_path)
        files = sorted(dir_path.glob("*.fits")) if sort else list(dir_path.glob("*.fits"))
        if not files:
            raise FileNotFoundError(f"No FITS files found in {dir_path}")
        logger.info("Reading {} SoLEXS files from {}", len(files), dir_path)
        dfs = [self.read_file(f) for f in files]
        merged = pd.concat(dfs).sort_index().drop_duplicates()
        logger.info("SoLEXS directory loaded: {} total rows", len(merged))
        return merged

    def _validate_header(self, header: fits.Header, fname: str):
        """Check mandatory FITS keywords."""
        required_keys = ["TELESCOP", "INSTRUME", "DATE-OBS"]
        missing = [k for k in required_keys if k not in header]
        if missing:
            logger.warning("Missing FITS keywords {} in {}", missing, fname)

    def _table_to_dataframe(self, table, fname: str) -> pd.DataFrame:
        """Convert FITS binary table to DataFrame."""
        col_map = self.config["columns"]
        missing = [v for v in col_map.values() if v not in table.names]
        if missing:
            logger.warning("Missing columns {} in {}", missing, fname)
        df = pd.DataFrame({
            "time_raw":     table[col_map["time"]],
            "soft_flux":    table[col_map["flux"]].astype(np.float64),
            "soft_counts":  table[col_map["counts"]].astype(np.float64),
            "energy_mean":  table[col_map.get("energy", "ENERGY")].astype(np.float64)
                            if "energy" in col_map and col_map["energy"] in table.names
                            else np.nan,
            "quality_flag": table["QUALITY"].astype(int)
                            if "QUALITY" in table.names else 0,
            "source_file":  fname,
        })
        return df

    def _convert_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert SoLEXS time column to UTC DatetimeIndex.
        SoLEXS TIME is in Mission Elapsed Time (MET) seconds from
        2017-01-01 00:00:00 UTC (ISRO epoch). Convert via astropy.
        """
        ISRO_EPOCH = "2017-01-01T00:00:00"
        try:
            t = Time(df["time_raw"].values, format="unix",
                     scale="utc") + (Time(ISRO_EPOCH) - Time(0, format="unix"))
            df.index = pd.DatetimeIndex(t.to_datetime(), tz="UTC",
                                        name="timestamp_utc")
        except Exception:
            # Fallback: try parsing as ISO string
            df.index = pd.to_datetime(df["time_raw"], utc=True)
        df = df.drop(columns=["time_raw"])
        return df

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to uniform cadence. Use mean for flux, sum for counts."""
        numeric_cols = ["soft_flux", "soft_counts", "energy_mean"]
        df_rs = df[numeric_cols].resample(self.resample).mean()
        df_rs["quality_flag"] = df["quality_flag"].resample(
            self.resample).max()   # Propagate worst quality
        df_rs["source_file"] = df["source_file"].resample(
            self.resample).first()
        return df_rs

    def _validate_flux(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Physical sanity checks on flux values.
        SoLEXS soft X-ray background: ~1e-8 W/m²/keV
        Maximum observed X-class: ~1e-3 W/m²/keV
        """
        FLUX_MIN = 1e-12
        FLUX_MAX = 1e-2
        n_bad = ((df["soft_flux"] < FLUX_MIN) |
                 (df["soft_flux"] > FLUX_MAX)).sum()
        if n_bad > 0:
            logger.warning("SoLEXS: {} physically implausible flux values flagged", n_bad)
            df.loc[(df["soft_flux"] < FLUX_MIN) |
                   (df["soft_flux"] > FLUX_MAX), "quality_flag"] = 3
        return df
```

---

## `src/data/hel1os_reader.py`

**Purpose:** Read HEL1OS Level-1 FITS files, extract hard X-ray flux, return clean DataFrame.  
**Structure:** Mirrors SoLEXSReader exactly. Key difference: HEL1OS energy band is 10–150 keV.

```python
"""
HEL1OS FITS Reader
==================
HEL1OS measures hard X-rays in 10–150 keV.

Physical context:
  Hard X-rays (10–150 keV) from HEL1OS trace non-thermal bremsstrahlung
  from energetic electrons accelerated during the impulsive phase.
  CRITICAL PHYSICS: Hard X-rays PEAK BEFORE soft X-rays.
  This is the Neupert Effect — HXR precedes SXR peak by 1–10 minutes.
  This precursor is the primary physical basis for forecasting.
  An algorithm that detects HXR rise before SXR rise gains lead time.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from pathlib import Path
from loguru import logger
from typing import Union
import yaml


class HEL1OSReader:
    """
    Reads and parses HEL1OS Level-1 FITS files from ISRO ISSDC PRADAN.

    Key difference from SoLEXSReader:
      - Energy band: 10–150 keV (hard X-rays, non-thermal)
      - Output column: hard_flux instead of soft_flux
      - Physics: HXR peaks 1–10 minutes before SXR peak (Neupert Effect)
    """

    def __init__(self, config_path: str = "configs/data.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["hel1os"]
        self.hdu_ext = self.config["fits_extension"]
        self.resample = f"{self.config['resample_seconds']}s"

    def read_file(self, fits_path: Union[str, Path]) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
            timestamp_utc   : datetime64[ns, UTC]
            hard_flux       : float64  (counts/s/keV or W/m²/keV)
            hard_counts     : float64
            quality_flag    : int
            source_file     : str
        """
        # Implementation mirrors SoLEXSReader._read_file()
        # Replace soft_flux -> hard_flux, soft_counts -> hard_counts
        # Physical limits: FLUX_MIN=1e-3 counts/s, FLUX_MAX=1e8 counts/s
        pass  # Claude Code: implement following exact same pattern as SoLEXSReader

    def read_directory(self, dir_path: Union[str, Path]) -> pd.DataFrame:
        pass  # Claude Code: same as SoLEXSReader.read_directory
```

---

## `src/data/aligner.py`

**Purpose:** Temporally align SoLEXS and HEL1OS DataFrames onto a common UTC time grid.

```python
"""
Instrument Aligner
==================
SoLEXS and HEL1OS have independent clocks with potential drift.
This module creates a single merged DataFrame at uniform cadence.

PHYSICS NOTE:
  The time lag between HEL1OS hard X-ray peak and SoLEXS soft X-ray peak
  is the Neupert Effect (typically 1–10 minutes).
  After alignment, this lag becomes measurable as a feature.
  cross_correlation(hard_flux, soft_flux, lags=[-120, 120]) reveals it.
"""

import pandas as pd
import numpy as np
from loguru import logger
import yaml


class InstrumentAligner:
    """
    Aligns SoLEXS and HEL1OS time series to a common UTC grid.

    Steps:
    1. Find common time range (intersection of both series)
    2. Create reference grid at target cadence (5s)
    3. Interpolate both series onto reference grid
    4. Flag gaps > max_gap_seconds with quality_flag=9
    5. Return merged DataFrame
    """

    def __init__(self, config_path: str = "configs/data.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.resample = f"{cfg['alignment']['max_gap_seconds']}s"
        self.max_gap = pd.Timedelta(
            seconds=cfg["alignment"]["max_gap_seconds"])
        self.cadence = f"{cfg['solexs']['resample_seconds']}s"

    def align(self, df_solexs: pd.DataFrame,
              df_hel1os: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df_solexs  : output of SoLEXSReader.read_file()
        df_hel1os  : output of HEL1OSReader.read_file()

        Returns
        -------
        pd.DataFrame with columns:
            timestamp_utc  : index
            soft_flux      : from SoLEXS
            soft_counts    : from SoLEXS
            hard_flux      : from HEL1OS (interpolated to SoLEXS grid)
            hard_counts    : from HEL1OS
            quality_solexs : quality flag from SoLEXS
            quality_hel1os : quality flag from HEL1OS
            data_gap       : bool — True if either instrument has gap here
        """
        # 1. Find common range
        t_start = max(df_solexs.index[0], df_hel1os.index[0])
        t_end   = min(df_solexs.index[-1], df_hel1os.index[-1])
        logger.info("Common time range: {} to {}", t_start, t_end)

        # 2. Build reference grid
        ref_grid = pd.date_range(t_start, t_end, freq=self.cadence, tz="UTC")

        # 3. Reindex both with linear interpolation
        df_s = df_solexs.reindex(
            df_solexs.index.union(ref_grid)).interpolate(
            "time").reindex(ref_grid)
        df_h = df_hel1os.reindex(
            df_hel1os.index.union(ref_grid)).interpolate(
            "time").reindex(ref_grid)

        # 4. Merge
        merged = pd.DataFrame({
            "soft_flux":      df_s["soft_flux"],
            "soft_counts":    df_s["soft_counts"],
            "hard_flux":      df_h["hard_flux"],
            "hard_counts":    df_h["hard_counts"],
            "quality_solexs": df_s["quality_flag"].fillna(9).astype(int),
            "quality_hel1os": df_h["quality_flag"].fillna(9).astype(int),
        }, index=ref_grid)

        # 5. Flag gaps
        merged["data_gap"] = (
            merged["quality_solexs"] == 9) | (merged["quality_hel1os"] == 9)

        logger.info("Alignment complete: {} samples, {} gaps",
                    len(merged), merged["data_gap"].sum())
        return merged
```

---

## `src/data/cleaner.py`

**Purpose:** Remove non-solar artifacts before training. Isolation Forest detects anomalous samples that are NOT flares — particle events, South Atlantic Anomaly passages, eclipse transitions.

```python
"""
Data Cleaner — Isolation Forest Preprocessing
=============================================
PHYSICS RATIONALE:
  Aditya-L1 in halo orbit around L1 is generally away from Earth's
  radiation belts, BUT cosmic ray events and particle showers still
  create brief spikes distinguishable from flares by:
    - Extremely short duration (< 60 seconds)
    - No corresponding signal in BOTH channels simultaneously
    - Unphysical hardness ratios (>> 10)

  Isolation Forest identifies these as statistical outliers WITHOUT
  needing labels — unsupervised, so it works even on novel events.

  IMPORTANT: Run Isolation Forest on QUIET SUN SEGMENTS only.
  Do NOT run it on known flare periods — it would incorrectly flag
  genuine X-class flares as outliers.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from loguru import logger
import yaml


class DataCleaner:
    """
    Removes non-solar artifacts from merged time series.

    Strategy:
    1. Identify quiet-sun segments (soft_flux < A-class threshold)
    2. Fit Isolation Forest on quiet-sun feature vectors
    3. Score ALL data with fitted model
    4. Flag artifact samples (score < threshold)
    5. Do NOT remove — flag with artifact_flag=1 (preserve for inspection)
    """

    def __init__(self, config_path: str = "configs/data.yaml",
                 contamination: float = 0.01):
        self.contamination = contamination  # Expected artifact fraction
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        logger.info("DataCleaner initialized | contamination={}", contamination)

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : aligned merged DataFrame from InstrumentAligner

        Returns
        -------
        df with new column: artifact_flag (0=clean, 1=artifact)
        """
        # Features for anomaly detection (physics-motivated)
        features = self._build_detection_features(df)

        logger.info("Fitting Isolation Forest on {} samples...", len(features))
        labels = self.model.fit_predict(features)  # -1 = anomaly, 1 = normal

        df = df.copy()
        df["artifact_flag"] = (labels == -1).astype(int)

        n_artifacts = df["artifact_flag"].sum()
        logger.info("Artifacts detected: {} ({:.2f}%)",
                    n_artifacts, 100 * n_artifacts / len(df))
        return df

    def _build_detection_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Physics-motivated features for artifact detection:
          - Log flux ratio (artifact spikes have unphysical ratios)
          - Flux derivative magnitude (artifacts are instantaneous)
          - Hard/soft correlation over short window
        """
        eps = 1e-15
        features = pd.DataFrame({
            "log_soft":      np.log10(df["soft_flux"] + eps),
            "log_hard":      np.log10(df["hard_flux"] + eps),
            "log_ratio":     np.log10(
                (df["hard_flux"] + eps) / (df["soft_flux"] + eps)),
            "d_soft_abs":    df["soft_flux"].diff().abs(),
            "d_hard_abs":    df["hard_flux"].diff().abs(),
            "soft_rolling_std": df["soft_flux"].rolling(12).std(),
        }).fillna(0)
        return features.values
```

---

# PART 5 — FEATURE ENGINEERING

## `src/features/physics_features.py`

**Purpose:** Extract every physically meaningful quantity from aligned dual-stream time series.

```python
"""
Physics Feature Engineering
============================
Every feature below has:
  1. Physical justification (why it helps forecast flares)
  2. Formula
  3. Expected behavior before/during flares
  4. Implementation

PHYSICS PRIMER:
  Solar flares have a two-phase structure in X-rays:
  
  IMPULSIVE PHASE (hard X-rays peak here)
    - Duration: seconds to minutes
    - HEL1OS hard X-ray flux rises rapidly
    - Non-thermal electron bremsstrahlung
    - d(HXR)/dt is large and positive
    
  GRADUAL PHASE (soft X-rays peak here)
    - Duration: minutes to hours
    - SoLEXS soft X-ray flux peaks AFTER hard X-rays (Neupert Effect)
    - Thermal emission from heated plasma
    - d(SXR)/dt follows integral of HXR (Neupert relation)

  PRECURSOR SIGNATURES (before any flare is visible):
    - Subtle HXR rise (< 3-sigma) appearing 5–30 min before peak
    - Hardness ratio begins drifting upward
    - Cross-correlation lag between channels decreases
    - Matrix Profile discord score increases
"""

import numpy as np
import pandas as pd
from loguru import logger
import yaml


class PhysicsFeatureExtractor:
    """
    Extracts physics-grounded features from aligned SoLEXS + HEL1OS data.
    
    All features are computed at each time step and returned as a
    DataFrame aligned to the input index.
    """

    def __init__(self, config_path: str = "configs/features.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.eps = self.cfg["physics"]["hardness_ratio"]["epsilon"]

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all enabled feature extractors.
        
        Input: aligned DataFrame with [soft_flux, hard_flux, soft_counts, hard_counts]
        Output: DataFrame with all physics features appended
        """
        features = {}
        features.update(self._hardness_ratio(df))
        features.update(self._flux_derivatives(df))
        features.update(self._spectral_index(df))
        features.update(self._rise_decay_rates(df))
        features.update(self._rolling_statistics(df))
        features.update(self._cross_correlation_features(df))
        features.update(self._background_subtracted(df))
        features.update(self._neupert_proxy(df))
        return pd.DataFrame(features, index=df.index)

    # ─── FEATURE 1: Hardness Ratio ─────────────────────────────────────────
    def _hardness_ratio(self, df: pd.DataFrame) -> dict:
        """
        Hardness Ratio = hard_flux / soft_flux

        PHYSICS: Measures spectral hardness of X-ray emission.
          - Quiet sun:  HR ≈ 0.01–0.1 (soft thermal emission dominates)
          - Pre-flare:  HR starts increasing as non-thermal component grows
          - Flare peak: HR spikes (hard X-ray impulsive peak)
          - Post-flare: HR decreases as plasma cools and softens

        This is ONE OF THE MOST IMPORTANT PRECURSOR FEATURES.
        A sustained HR increase of > 20% over 10 minutes preceding a flare
        has been reported in literature.
        """
        hr = df["hard_flux"] / (df["soft_flux"] + self.eps)
        return {
            "hardness_ratio":        hr,
            "hardness_ratio_log":    np.log10(hr + self.eps),
            "hardness_ratio_deriv":  hr.diff(),
            "hardness_ratio_5min":   hr.rolling(60).mean(),   # 60*5s = 5 min
            "hardness_ratio_15min":  hr.rolling(180).mean(),
        }

    # ─── FEATURE 2: Flux Derivatives ───────────────────────────────────────
    def _flux_derivatives(self, df: pd.DataFrame) -> dict:
        """
        dF/dt and d²F/dt² for both channels.

        PHYSICS:
          - dF_hard/dt > 0 is the primary impulsive phase indicator
          - d²F_soft/dt² > 0 before the SXR peak indicates acceleration
          - Neupert: d(SXR)/dt ∝ HXR (integral relation)
          - A positive d²(SXR)/dt² 10–20 min before SXR peak is a
            forecast-relevant precursor

        Smoothing BEFORE differencing is critical — raw derivatives
        on noisy photon counting data are meaningless.
        """
        smooth_w = self.cfg["physics"]["flux_derivative"]["smoothing_window"]

        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
        
        results = {}
        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
            smoothed = df[col].rolling(smooth_w, center=True).mean()
            log_flux = np.log10(smoothed + self.eps)
            results[f"d_{ch}_flux_dt"]     = smoothed.diff()
            results[f"d2_{ch}_flux_dt2"]   = smoothed.diff().diff()
            results[f"d_log_{ch}_flux_dt"] = log_flux.diff()   # relative rate
        return results

    # ─── FEATURE 3: Spectral Index ─────────────────────────────────────────
    def _spectral_index(self, df: pd.DataFrame) -> dict:
        """
        Spectral Index: slope of log(flux) vs log(energy)

        PHYSICS: Power-law spectrum: F(E) ∝ E^(-γ)
          γ = spectral index
          - Thermal (quiet sun): γ ≈ 4–6
          - Non-thermal (impulsive flare): γ ≈ 2–4 (harder spectrum)
          - Decreasing γ signals non-thermal acceleration = flare onset

        Approximated using two-point estimate between
        SoLEXS mean energy (~5 keV) and HEL1OS mean energy (~20 keV).
        """
        E_soft  = 5.0   # keV (approximate SoLEXS band center)
        E_hard  = 20.0  # keV (approximate HEL1OS band center)

        log_F_soft = np.log10(df["soft_flux"] + self.eps)
        log_F_hard = np.log10(df["hard_flux"] + self.eps)
        gamma = (log_F_hard - log_F_soft) / (np.log10(E_hard) - np.log10(E_soft))

        return {
            "spectral_index":        gamma,
            "spectral_index_5min":   gamma.rolling(60).mean(),
            "spectral_index_deriv":  gamma.diff(),
        }

    # ─── FEATURE 4: Rise and Decay Rates ───────────────────────────────────
    def _rise_decay_rates(self, df: pd.DataFrame) -> dict:
        """
        Rise Rate: how fast flux is increasing in recent N minutes
        Decay Rate: how fast flux is decreasing after peak

        PHYSICS:
          - Fast rise (> 1 decade/5 min) = impulsive X-class flare
          - Slow rise (< 1 decade/30 min) = C-class gradual event
          - Rise rate asymmetry between HXR and SXR channels gives
            impulsive fraction → severity proxy
        """
        results = {}
        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
            log_flux = np.log10(df[col] + self.eps)
            for w_min in [5, 15]:
                w = int(w_min * 60 / 5)  # convert minutes to samples
                change = log_flux - log_flux.shift(w)
                results[f"{ch}_rise_rate_{w_min}min"] = change.clip(lower=0)
                results[f"{ch}_decay_rate_{w_min}min"] = change.clip(upper=0).abs()
        return results

    # ─── FEATURE 5: Rolling Statistics ─────────────────────────────────────
    def _rolling_statistics(self, df: pd.DataFrame) -> dict:
        """
        Rolling mean, std, skewness, kurtosis at multiple windows.
        
        PHYSICS: Kurtosis spike in hard X-ray channel is a known
        precursor of impulsive phase onset (Georgoulis et al.).
        """
        results = {}
        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
            for w_min, w_label in [(5, "5m"), (15, "15m"), (60, "60m")]:
                w = int(w_min * 60 / 5)
                rolled = df[col].rolling(w)
                results[f"{ch}_mean_{w_label}"]  = rolled.mean()
                results[f"{ch}_std_{w_label}"]   = rolled.std()
                results[f"{ch}_skew_{w_label}"]  = rolled.skew()
                results[f"{ch}_kurt_{w_label}"]  = rolled.kurt()
        return results

    # ─── FEATURE 6: Cross-Correlation Features ─────────────────────────────
    def _cross_correlation_features(self, df: pd.DataFrame) -> dict:
        """
        Rolling cross-correlation between hard and soft X-ray channels.

        PHYSICS (NEUPERT EFFECT):
          During quiet sun: HXR and SXR are decorrelated (different sources)
          During pre-flare: correlation begins increasing as same energetic
            event starts producing both thermal and non-thermal emission
          During flare: HXR peaks first, SXR peaks 1–10 min later

          The LAG at maximum cross-correlation gives the Neupert delay.
          A decreasing lag over time is an early warning signal.
        """
        w = 180   # 15-minute rolling window
        results = {}
        
        log_soft = np.log10(df["soft_flux"] + self.eps)
        log_hard = np.log10(df["hard_flux"] + self.eps)
        
        # Rolling Pearson correlation (zero-lag)
        results["hxr_sxr_corr_15min"] = log_soft.rolling(w).corr(log_hard)
        
        # Lag-1 sample correlation (HXR leads SXR by 5 seconds?)
        results["hxr_sxr_lag1_corr"] = log_hard.rolling(w).corr(
            log_soft.shift(1))
        
        return results

    # ─── FEATURE 7: Background-Subtracted Flux ─────────────────────────────
    def _background_subtracted(self, df: pd.DataFrame) -> dict:
        """
        Remove pre-flare background to get flare-excess flux.

        Background = rolling 10th percentile over 60 minutes.
        This is the standard method used in GOES processing.
        """
        results = {}
        w = int(60 * 60 / 5)   # 60-minute window in samples
        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
            bg = df[col].rolling(w, center=False).quantile(0.10)
            excess = (df[col] - bg).clip(lower=0)
            results[f"{ch}_flux_excess"]     = excess
            results[f"{ch}_flux_excess_log"] = np.log10(excess + self.eps)
            results[f"{ch}_snr"]             = excess / (bg + self.eps)
        return results

    # ─── FEATURE 8: Neupert Proxy ──────────────────────────────────────────
    def _neupert_proxy(self, df: pd.DataFrame) -> dict:
        """
        Neupert Relation: SXR(t) ∝ ∫₀ᵗ HXR(t') dt'

        The Neupert Effect states: the soft X-ray time profile is approximately
        the time integral of the hard X-ray time profile.
        Deviation from this relation is physically meaningful.

        Neupert_proxy = cumulative sum of HXR (running integral)
        Neupert_residual = SXR - Neupert_proxy (normalized)
        A positive Neupert_residual emerging before SXR peak
        indicates additional thermal heating beyond the impulsive phase.

        This is a novel feature. Validate carefully on known events.
        """
        hard_excess = (df["hard_flux"] -
                       df["hard_flux"].rolling(720).quantile(0.10)).clip(lower=0)
        neupert_proxy = hard_excess.rolling(120).sum()  # 10-min running integral
        neupert_proxy_norm = neupert_proxy / (neupert_proxy.rolling(720).max() + self.eps)

        soft_norm = df["soft_flux"] / (df["soft_flux"].rolling(720).max() + self.eps)
        neupert_residual = soft_norm - neupert_proxy_norm

        return {
            "neupert_proxy":        neupert_proxy_norm,
            "neupert_residual":     neupert_residual,
            "neupert_residual_abs": neupert_residual.abs(),
        }
```

---

## `src/features/matrix_profile.py`

```python
"""
Matrix Profile Analysis
=======================
Uses STUMPY to compute Matrix Profile on both X-ray channels.

WHAT IT FINDS:
  - Motifs: recurring patterns in the light curve (typical pre-flare shapes)
  - Discords: unusual patterns that don't recur (anomalous events)

WHY THIS MATTERS:
  Flare precursors often manifest as small, recurring flux enhancements
  in the minutes before the main event. The Matrix Profile finds these
  even without knowing what to look for — it's unsupervised.
  Discords flag data segments that are unusually different from
  everything else — ideal for catching novel precursor morphologies.

USAGE: Run as exploratory analysis in notebooks and feed discord
scores into the forecasting model as features.
"""

import numpy as np
import pandas as pd
import stumpy
from loguru import logger
import yaml


class MatrixProfileAnalyzer:
    """
    Computes STUMPY Matrix Profile on SoLEXS and HEL1OS time series.
    """

    def __init__(self, config_path: str = "configs/features.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.window_sizes = cfg["matrix_profile"]["window_sizes"]
        self.n_motifs = cfg["matrix_profile"]["n_motifs"]
        self.n_discords = cfg["matrix_profile"]["n_discords"]

    def compute_discord_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each configured window size, compute Matrix Profile
        and extract the discord (anomaly) scores at each timestep.

        Returns DataFrame of discord scores aligned to df.index.
        Discord score = distance to nearest neighbor in profile space.
        High score = anomalous = potential precursor or artifact.
        """
        results = {}
        for ch in ["soft_flux", "hard_flux"]:
            series = np.log10(df[ch].fillna(method="ffill").values + 1e-15)
            for w in self.window_sizes:
                if len(series) < 2 * w:
                    logger.warning("Series too short for window {}", w)
                    continue
                logger.info("Computing Matrix Profile: channel={}, window={}", ch, w)
                mp = stumpy.stump(series, m=w)
                profile_scores = mp[:, 0].astype(float)  # distance profile
                # Align back to original index (MP is shorter by w-1)
                padded = np.full(len(df), np.nan)
                padded[w - 1:] = profile_scores
                results[f"mp_{ch}_w{w}"] = padded
        return pd.DataFrame(results, index=df.index)
```

---

# PART 6 — MODELS

## `src/models/base_model.py`

```python
"""
Abstract Base Model
===================
ALL models in SolarSense-AI inherit from BaseModel.
This enforces:
  - Consistent interface (fit, predict, predict_proba, uncertainty)
  - Mandatory logging of all experiments
  - Consistent checkpoint saving
  - Bayesian uncertainty output format
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
import mlflow
from loguru import logger
from pathlib import Path
import yaml


class BaseModel(ABC):
    """Base class for all SolarSense-AI models."""

    def __init__(self, model_name: str,
                 config_path: str = "configs/models.yaml"):
        self.model_name = model_name
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.checkpoint_dir = Path("models/checkpoints") / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Model initialized: {}", model_name)

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        """Train the model. Log metrics to MLflow."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability array shape (N,) or (N, n_classes)."""
        pass

    def predict_with_uncertainty(self, X: np.ndarray,
                                 n_passes: int = 50) -> dict:
        """
        Returns dict:
          probability  : mean prediction across MC Dropout passes
          uncertainty  : std of predictions (epistemic uncertainty)
          lower_bound  : 5th percentile
          upper_bound  : 95th percentile
        Override in subclasses that support MC Dropout.
        """
        probs = self.predict_proba(X)
        return {
            "probability":  probs,
            "uncertainty":  np.zeros_like(probs),
            "lower_bound":  probs,
            "upper_bound":  probs,
        }

    def save(self, tag: str = "best"):
        path = self.checkpoint_dir / f"{self.model_name}_{tag}.pt"
        torch.save(self.state_dict(), path)
        logger.info("Saved checkpoint: {}", path)

    def load(self, tag: str = "best"):
        path = self.checkpoint_dir / f"{self.model_name}_{tag}.pt"
        self.load_state_dict(torch.load(path))
        logger.info("Loaded checkpoint: {}", path)
```

---

## `src/models/nowcaster/threshold_detector.py`

```python
"""
Physics-Based Threshold Nowcaster
===================================
The FIRST nowcaster in the pipeline.

ALGORITHM (adapted from standard solar physics practice):

For NOWCASTING (detection), before any ML:
  1. Compute background B = rolling 10-min 10th percentile
  2. Compute noise N = rolling 10-min std of quiet-sun segments
  3. Alert when: flux > B + k*N for k=3 (3-sigma) sustained for T=60s

This is not "simple thresholding."
This mirrors the algorithm used by NOAA SWPC for operational GOES alerts.
The sophistication is in the background estimation and dual-channel confirmation.

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
from typing import List, Optional
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
    confirmation:  str        # "dual_channel" or "soft_only"
    quality:       int        # 0=high, 1=medium, 2=low


class ThresholdNowcaster:
    """
    Physics-based dual-channel nowcasting detector.
    Outputs a catalog of FlareEvent objects.
    """

    FLUX_CLASS_BOUNDARIES = {
        "A": (1e-8, 1e-7),
        "B": (1e-7, 1e-6),
        "C": (1e-6, 1e-5),
        "M": (1e-5, 1e-4),
        "X": (1e-4, np.inf),
    }

    def __init__(self, config_path: str = "configs/models.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["nowcaster"]["threshold_detector"]
        self.soft_k    = cfg["soft_xray_threshold_multiplier"]
        self.hard_k    = cfg["hard_xray_threshold_multiplier"]
        self.min_dur   = pd.Timedelta(seconds=cfg["min_duration_seconds"])
        self.n_confirm = cfg["confirmation_samples"]

    def detect(self, df: pd.DataFrame) -> List[FlareEvent]:
        """
        Run detection on aligned merged DataFrame.
        Returns list of FlareEvent objects.
        """
        df = self._compute_thresholds(df)
        df = self._apply_threshold_flags(df)
        events = self._extract_events(df)
        logger.info("ThresholdNowcaster: {} events detected", len(events))
        return events

    def to_catalog(self, events: List[FlareEvent]) -> pd.DataFrame:
        """Convert event list to catalog DataFrame for storage."""
        if not events:
            return pd.DataFrame()
        return pd.DataFrame([vars(e) for e in events])

    def _compute_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling background and noise estimates."""
        bg_window = 120   # 10-minute background window (samples at 5s)
        df = df.copy()
        for ch in ["soft", "hard"]:
            col = f"{ch}_flux"
            df[f"{ch}_bg"]   = df[col].rolling(bg_window).quantile(0.10)
            df[f"{ch}_noise"] = df[col].rolling(bg_window).std()
            k = self.soft_k if ch == "soft" else self.hard_k
            df[f"{ch}_threshold"] = df[f"{ch}_bg"] + k * df[f"{ch}_noise"]
        return df

    def _apply_threshold_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag samples exceeding threshold in both channels."""
        df["soft_above"] = df["soft_flux"] > df["soft_threshold"]
        df["hard_above"] = df["hard_flux"] > df["hard_threshold"]
        # Require N consecutive samples above threshold
        df["soft_triggered"] = (df["soft_above"]
                                 .rolling(self.n_confirm)
                                 .sum() == self.n_confirm)
        df["hard_triggered"] = (df["hard_above"]
                                 .rolling(self.n_confirm)
                                 .sum() >= 1)   # At least 1 sample in hard
        df["dual_trigger"] = df["soft_triggered"] & df["hard_triggered"]
        return df

    def _extract_events(self, df: pd.DataFrame) -> List[FlareEvent]:
        """Group triggered samples into discrete events."""
        events = []
        in_event = False
        event_start = None
        event_samples = []

        for ts, row in df.iterrows():
            if not in_event and row["dual_trigger"]:
                in_event = True
                event_start = ts
                event_samples = []
            if in_event:
                event_samples.append(row)
                if not row["dual_trigger"]:
                    # Event ended — check minimum duration
                    duration = ts - event_start
                    if duration >= self.min_dur:
                        event = self._build_event(event_start, ts,
                                                   pd.DataFrame(event_samples))
                        if event:
                            events.append(event)
                    in_event = False
                    event_samples = []
        return events

    def _build_event(self, start, end, samples) -> Optional[FlareEvent]:
        """Construct FlareEvent from detected samples."""
        peak_idx   = samples["soft_flux"].idxmax()
        peak_soft  = samples["soft_flux"].max()
        peak_hard  = samples["hard_flux"].max()
        cls, subcls = self._classify_flux(peak_soft)
        return FlareEvent(
            start_time=start,
            peak_time=peak_idx,
            end_time=end,
            peak_soft_flux=peak_soft,
            peak_hard_flux=peak_hard,
            flare_class=cls,
            flare_subclass=subcls,
            confirmation="dual_channel",
            quality=0,
        )

    def _classify_flux(self, flux: float):
        for cls, (lo, hi) in self.FLUX_CLASS_BOUNDARIES.items():
            if lo <= flux < hi:
                subclass = (flux / lo)
                return cls, round(subclass, 1)
        return "X", 10.0
```

---

## `src/models/forecaster/patchtst_forecaster.py`

```python
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

        outputs = {
            "flare_prob":         torch.sigmoid(self.head_flare_prob(combined)),
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
        y_class  = y["flare_class"].long()
        y_lead   = y["lead_time"].float()

        # Focal loss instead of BCE (handles class imbalance)
        bce  = F.binary_cross_entropy(
            out["flare_prob"].squeeze(), y_flare)
        ce   = F.cross_entropy(out["flare_class_logits"], y_class)
        mse  = F.mse_loss(
            out["lead_time"].squeeze()[y_flare > 0],
            y_lead[y_flare > 0]) if y_flare.sum() > 0 else torch.tensor(0.)

        # Physics loss: hazard should be monotonically increasing before peak
        hazard = out["hazard"]
        phys = F.relu(hazard[:, :-1] - hazard[:, 1:]).mean()  # Penalize non-monotone

        return bce + 0.5 * ce + 0.3 * mse + 0.1 * phys

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
        return [opt], [sched]
```

---

## `src/models/survival/hazard_model.py`

```python
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
```

---

## `src/models/memory/flare_memory.py`

```python
"""
Historical Flare Memory Bank
============================
Retrieves the most similar historical flare events for any current observation.

WHY THIS IS POWERFUL:
  A space physicist looking at a pre-flare signature doesn't just use a model —
  they recall past events. "This looks like the 2003 Halloween storm precursor."
  
  This module gives the AI system the same capability.
  When the forecaster issues an alert, the operator can see:
    "Alert: M-class flare likely | Similar to events on 2023-08-05 (94% similar),
     2024-02-18 (89% similar) — both preceded M-class events in 8–12 minutes."
  
  This is explainability through analogy — more interpretable than attention maps
  for operational space-weather personnel.

IMPLEMENTATION:
  1. Embed each historical flare event as a feature vector
     (physics features averaged over the 30-min pre-flare window)
  2. Store embeddings in a FAISS flat index
  3. At inference, embed the current 30-min window and query top-k neighbors
  4. Return similar events with metadata

BUILD STEP: Run scripts/build_memory_bank.py once on the historical catalog.
"""

import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from typing import List


@dataclass
class SimilarEvent:
    """A retrieved similar historical flare."""
    event_id:    str
    date:        str
    flare_class: str
    similarity:  float     # 0–1, higher = more similar
    lead_time:   float     # Minutes between this retrieval point and peak
    description: str       # Human-readable summary


class FlareMemoryBank:
    """
    FAISS-backed historical flare event memory bank.
    """

    def __init__(self, index_path: str = "models/checkpoints/faiss_memory.index",
                 metadata_path: str = "models/checkpoints/faiss_metadata.pkl",
                 embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.index_path    = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index         = None
        self.metadata      = None

    def build(self, event_embeddings: np.ndarray,
              event_metadata: List[dict]):
        """
        Build FAISS index from historical event embeddings.
        
        Parameters
        ----------
        event_embeddings : (N, embedding_dim) float32 array
        event_metadata   : list of dicts with event info
        """
        assert event_embeddings.dtype == np.float32
        assert event_embeddings.shape[1] == self.embedding_dim

        # Normalize for cosine similarity
        faiss.normalize_L2(event_embeddings)

        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine after L2 norm
        self.index.add(event_embeddings)
        self.metadata = event_metadata

        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("FAISS memory bank built: {} events indexed", len(event_metadata))

    def load(self):
        """Load pre-built index from disk."""
        self.index    = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info("FAISS memory bank loaded: {} events", self.index.ntotal)

    def query(self, query_embedding: np.ndarray,
              top_k: int = 5) -> List[SimilarEvent]:
        """
        Find top-k most similar historical events.
        
        Parameters
        ----------
        query_embedding : (1, embedding_dim) float32
        
        Returns
        -------
        List[SimilarEvent] sorted by similarity descending
        """
        if self.index is None:
            raise RuntimeError("Memory bank not loaded. Call load() first.")
        q = query_embedding.astype(np.float32)
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            results.append(SimilarEvent(
                event_id=meta.get("event_id", str(idx)),
                date=meta.get("date", "unknown"),
                flare_class=meta.get("flare_class", "?"),
                similarity=float(dist),
                lead_time=float(meta.get("lead_time_minutes", -1)),
                description=meta.get("description", ""),
            ))
        return results
```

---

# PART 7 — TRAINING PIPELINE

## `src/training/metrics.py`

```python
"""
Evaluation Metrics for Solar Flare Prediction
==============================================

STANDARD SPACE-WEATHER METRICS:

TSS (True Skill Statistic) = TPR - FPR
  Range: -1 to +1. Perfect = 1. Random = 0.
  The PRIMARY metric for space-weather forecast evaluation.
  Recommended by WMO for operational forecast comparison.

HSS (Heidke Skill Score) = (TP+TN - Expected) / (Total - Expected)
  Measures improvement over random chance.
  HSS > 0.5 is considered operationally useful.

Brier Score = mean((p - y)²)
  Probabilistic accuracy. Lower is better.
  For well-calibrated forecasts in rare-event regime.

ECE (Expected Calibration Error)
  Measures how well forecast probabilities match empirical frequencies.
  A forecast that says 70% should be right 70% of the time.

Lead Time
  Primary operational metric for this challenge.
  = time of first alert BEFORE flare peak
  Target: > 10 minutes for M/X class events.

False Alarm Rate = FP / (FP + TN)
  Must be kept low for operational use.
  Target: < 30% FAR for M+ class events.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix)
from loguru import logger


def compute_confusion_elements(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                threshold: float = 0.5):
    """Compute TP, TN, FP, FN."""
    y_bin = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def true_skill_statistic(y_true, y_pred, threshold=0.5):
    """TSS = TPR - FPR. Primary metric."""
    cm = compute_confusion_elements(y_true, y_pred, threshold)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    tpr = tp / (tp + fn + 1e-10)
    fpr = fp / (fp + tn + 1e-10)
    return tpr - fpr


def heidke_skill_score(y_true, y_pred, threshold=0.5):
    """HSS — skill relative to random chance."""
    cm = compute_confusion_elements(y_true, y_pred, threshold)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    n   = tp + tn + fp + fn
    exp = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n + 1e-10)
    return (tp + tn - exp) / (n - exp + 1e-10)


def brier_score(y_true, y_prob):
    """Probabilistic accuracy."""
    return np.mean((y_prob - y_true) ** 2)


def compute_lead_time(alerts: pd.Series, catalog: pd.DataFrame,
                      tolerance_minutes: int = 60) -> pd.Series:
    """
    For each true flare in catalog, find the first alert that preceded it
    within tolerance_minutes. Return lead time in minutes.
    
    Parameters
    ----------
    alerts  : pd.Series of alert timestamps (when model fired)
    catalog : flare catalog with start_time and peak_time columns
    
    Returns
    -------
    pd.Series of lead times (minutes) per flare. NaN = missed.
    """
    lead_times = []
    for _, event in catalog.iterrows():
        peak = pd.Timestamp(event["peak_time"])
        window_start = peak - pd.Timedelta(minutes=tolerance_minutes)
        # Find all alerts in the pre-peak window
        early_alerts = alerts[
            (alerts >= window_start) & (alerts <= peak)
        ]
        if len(early_alerts) > 0:
            first_alert = early_alerts.min()
            lead_times.append((peak - first_alert).total_seconds() / 60)
        else:
            lead_times.append(np.nan)
    return pd.Series(lead_times, index=catalog.index)


def full_evaluation_report(y_true, y_pred_prob, alerts, catalog) -> dict:
    """Run complete evaluation suite and return metrics dict."""
    tss  = true_skill_statistic(y_true, y_pred_prob)
    hss  = heidke_skill_score(y_true, y_pred_prob)
    bs   = brier_score(y_true, y_pred_prob)
    auc  = roc_auc_score(y_true, y_pred_prob)
    auprc = average_precision_score(y_true, y_pred_prob)
    lt   = compute_lead_time(alerts, catalog)

    report = {
        "TSS":             round(tss, 4),
        "HSS":             round(hss, 4),
        "Brier_Score":     round(bs, 4),
        "ROC_AUC":         round(auc, 4),
        "AUPRC":           round(auprc, 4),
        "Lead_Time_mean":  round(lt.mean(), 2),
        "Lead_Time_median": round(lt.median(), 2),
        "Lead_Time_p90":   round(lt.quantile(0.9), 2),
        "Detection_Rate":  round((~lt.isna()).mean(), 4),
    }
    logger.info("Evaluation complete: {}", report)
    return report
```

---

## `src/training/trainer.py`

```python
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

    def train_forecaster(self, model, dataset_train, dataset_val,
                         experiment_name: str = "patchtst_run_01"):
        """Train a forecaster model with full logging."""
        cfg = self.cfg["forecaster"]["patchtst"]

        # Weighted sampler for class imbalance
        flare_labels = np.array([dataset_train[i][2]["flare_label"]
                                  for i in range(len(dataset_train))])
        weights    = np.where(flare_labels > 0, 10.0, 1.0)  # 10x weight for flare
        sampler    = WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(dataset_train, batch_size=cfg["batch_size"],
                                   sampler=sampler, num_workers=4,
                                   pin_memory=True)
        val_loader   = DataLoader(dataset_val, batch_size=cfg["batch_size"],
                                   shuffle=False, num_workers=4)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"]
                          if "early_stopping_patience" in cfg else 15,
                          mode="min"),
            ModelCheckpoint(dirpath=f"models/checkpoints/{experiment_name}",
                            filename="best",
                            monitor="val_loss",
                            save_top_k=1),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        with mlflow.start_run(run_name=experiment_name):
            mlflow.log_params(cfg)
            trainer = L.Trainer(
                max_epochs=cfg["epochs"],
                callbacks=callbacks,
                accelerator=self.device,
                log_every_n_steps=50,
                deterministic=True,
            )
            trainer.fit(model, train_loader, val_loader)
            val_metrics = trainer.callback_metrics
            mlflow.log_metrics({k: float(v) for k, v in val_metrics.items()})
            mlflow.pytorch.log_model(model, artifact_path="model")
        return model
```

---

# PART 8 — API

## `src/api/main.py`

```python
"""
SolarSense-AI FastAPI Application
===================================
Real-time inference API for nowcasting and forecasting.

Endpoints:
  GET  /health              → System status
  POST /nowcast             → Detect flare in provided time window
  POST /predict             → Forecast with uncertainty
  GET  /history             → Recent alert history
  WS   /stream              → WebSocket for real-time monitoring

Response always includes:
  - probability + confidence interval
  - uncertainty estimate
  - flare class prediction
  - lead time estimate
  - similar historical events
  - attention map (which timesteps drove the alert)
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from loguru import logger

from .routes.predict  import router as predict_router
from .routes.nowcast  import router as nowcast_router
from .routes.health   import router as health_router
from .routes.history  import router as history_router

app = FastAPI(
    title="SolarSense-AI",
    description="Aditya-L1 Solar Flare Nowcasting & Forecasting API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router,  prefix="/health",  tags=["Health"])
app.include_router(nowcast_router, prefix="/nowcast", tags=["Nowcasting"])
app.include_router(predict_router, prefix="/predict", tags=["Forecasting"])
app.include_router(history_router, prefix="/history", tags=["History"])

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
```

## `src/api/schemas.py`

```python
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
```

---

# PART 9 — DASHBOARD

## `dashboard/app.py`

```python
"""
SolarSense-AI Operational Dashboard
=====================================
Streamlit-based real-time monitoring interface.

Layout:
  ┌──────────────────────────────────────────────────────────────┐
  │  🌞 SolarSense-AI | Aditya-L1 Real-Time Monitoring          │
  │  Status: ● MONITORING | Last flare: 2h 34m ago              │
  ├─────────────────────┬──────────────────────────────────────┤
  │                     │   🔴 ALERT: M-class flare predicted   │
  │  LIGHT CURVE PLOT   │   Probability: 87% ± 6%              │
  │  (dual stream)      │   Lead time: ~12 minutes             │
  │  SoLEXS + HEL1OS    │   Class: M2.3–M4.1 (90% CI)         │
  │  + background       │                                      │
  │  + threshold        │   Similar events:                    │
  │                     │   • 2024-03-23 M3.1 (91% similar)   │
  │                     │   • 2024-07-08 M2.8 (85% similar)   │
  ├─────────────────────┴──────────────────────────────────────┤
  │  ATTENTION MAP: Which timesteps drove this prediction?      │
  │  [heatmap across 120-minute lookback window]                │
  ├─────────────────────────────────────────────────────────────┤
  │  HAZARD CURVE: P(flare) over next 60 minutes               │
  │  [line chart of survival hazard function]                   │
  └─────────────────────────────────────────────────────────────┘

Colors:
  GREEN:  flare_prob < 0.3
  YELLOW: 0.3 ≤ flare_prob < 0.5
  ORANGE: 0.5 ≤ flare_prob < 0.8
  RED:    flare_prob ≥ 0.8
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import numpy as np
from datetime import datetime, timedelta

API_URL = "http://localhost:8000"
REFRESH_SECONDS = 10

st.set_page_config(
    page_title="SolarSense-AI",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def get_alert_color(prob: float) -> str:
    if prob < 0.3:   return "#00cc44"   # Green
    if prob < 0.5:   return "#ffcc00"   # Yellow
    if prob < 0.8:   return "#ff6600"   # Orange
    return "#ff0000"                     # Red


def get_alert_level(prob: float) -> str:
    if prob < 0.3:   return "🟢 GREEN — Quiet Sun"
    if prob < 0.5:   return "🟡 YELLOW — Elevated"
    if prob < 0.8:   return "🟠 ORANGE — High Risk"
    return "🔴 RED — ALERT"


def render_light_curve(df: pd.DataFrame, thresholds: dict):
    """Dual-stream light curve with threshold lines."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["soft_flux"],
        name="SoLEXS (Soft XR)", line=dict(color="#4da6ff", width=1.5)))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["hard_flux"],
        name="HEL1OS (Hard XR)", line=dict(color="#ff6633", width=1.5),
        yaxis="y2"))
    # Flare class threshold lines
    for cls, val in thresholds.items():
        fig.add_hline(y=val, line_dash="dot",
                      annotation_text=cls, line_color="gray")
    fig.update_layout(
        title="Aditya-L1 X-Ray Light Curves (Real-Time)",
        yaxis=dict(title="Soft X-Ray Flux (W/m²)", type="log"),
        yaxis2=dict(title="Hard X-Ray Counts/s", type="log",
                    overlaying="y", side="right"),
        xaxis=dict(title="Time (UTC)"),
        template="plotly_dark",
        height=400,
    )
    return fig


def main():
    st.title("🌞 SolarSense-AI | Aditya-L1 Solar Flare Monitor")

    # Auto-refresh placeholder
    placeholder = st.empty()

    with placeholder.container():
        try:
            resp = requests.get(f"{API_URL}/predict/latest", timeout=5)
            data = resp.json()
        except Exception:
            st.error("⚠️ Cannot connect to SolarSense API. Is it running?")
            return

        prob = data.get("flare_probability", 0.0)
        color = get_alert_color(prob)
        level = get_alert_level(prob)

        # ── Header Status Bar ──────────────────────────────────────────
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.markdown(f"### {level}")
        with col2:
            st.metric("Flare Probability",
                      f"{prob*100:.1f}%",
                      f"±{data.get('uncertainty', 0)*100:.1f}%")
        with col3:
            st.metric("Predicted Class",
                      data.get("predicted_class", "—"))
        with col4:
            lt = data.get("lead_time_minutes", 0)
            st.metric("Est. Lead Time", f"{lt:.0f} min" if lt > 0 else "—")

        # ── Light Curve + Alert Panel ─────────────────────────────────
        lc_col, alert_col = st.columns([3, 1])
        with lc_col:
            # TODO: fetch real-time light curve data from /history
            st.plotly_chart(
                render_light_curve(
                    pd.DataFrame(),  # Replace with real data
                    {"M": 1e-5, "X": 1e-4}
                ),
                use_container_width=True
            )

        with alert_col:
            st.markdown("### Similar Historical Events")
            for evt in data.get("similar_events", []):
                st.markdown(f"""
                **{evt['date']}** — {evt['flare_class']}  
                Similarity: {evt['similarity']*100:.0f}%  
                Lead time was: {evt['lead_time']:.0f} min
                ---""")

        # ── Hazard Curve ──────────────────────────────────────────────
        st.markdown("### Hazard Function — P(Flare) Over Next 60 Minutes")
        hazard = data.get("hazard_curve", np.zeros(60).tolist())
        fig_h = go.Figure(go.Scatter(
            y=hazard,
            x=list(range(1, len(hazard)+1)),
            fill="tozeroy",
            line=dict(color=color, width=2),
        ))
        fig_h.update_layout(
            xaxis_title="Minutes from now",
            yaxis_title="Hazard (flare probability)",
            template="plotly_dark",
            height=250,
        )
        st.plotly_chart(fig_h, use_container_width=True)

        # ── Attention Map ─────────────────────────────────────────────
        st.markdown("### Attention Map — What Drove This Prediction?")
        attn = data.get("attention_weights", np.zeros(120).tolist())
        fig_a = px.imshow(
            np.array(attn).reshape(1, -1),
            color_continuous_scale="Viridis",
            labels={"x": "Minutes ago", "color": "Attention"},
            aspect="auto",
        )
        fig_a.update_layout(
            xaxis=dict(tickvals=list(range(0, 120, 12)),
                       ticktext=[f"{120-i}" for i in range(0, 120, 12)]),
            height=100,
            template="plotly_dark",
        )
        st.plotly_chart(fig_a, use_container_width=True)

    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC | "
               f"Model: PatchTST Ensemble v1.0.0 | Data: Aditya-L1 SoLEXS + HEL1OS")


if __name__ == "__main__":
    main()
```

---

# PART 10 — SCRIPTS

## `scripts/build_dataset.py`

```bash
#!/usr/bin/env python
"""
Full data pipeline runner.
Run this once after downloading FITS files from ISSDC PRADAN.

Steps:
  1. Read all SoLEXS FITS → parquet
  2. Read all HEL1OS FITS → parquet
  3. Align both streams to common UTC grid
  4. Run Isolation Forest cleaning
  5. Extract all physics + spectral features
  6. Compute Matrix Profile discord scores
  7. Save merged feature DataFrame
  8. Run ThresholdNowcaster to generate nowcast catalog
  9. Build survival labels from catalog
  10. Save ready-to-train datasets (train/val/test splits)
"""
```

```python
import yaml
from loguru import logger
from src.data.solexs_reader    import SoLEXSReader
from src.data.hel1os_reader    import HEL1OSReader
from src.data.aligner          import InstrumentAligner
from src.data.cleaner          import DataCleaner
from src.features.feature_pipeline import FeaturePipeline
from src.models.nowcaster.threshold_detector import ThresholdNowcaster
from src.models.survival.hazard_model import build_survival_labels
import pandas as pd

def main():
    logger.info("=== SolarSense-AI Dataset Builder ===")
    
    # 1. Read raw FITS
    solexs = SoLEXSReader().read_directory("data/raw/solexs")
    hel1os = HEL1OSReader().read_directory("data/raw/hel1os")
    
    # 2. Align
    aligned = InstrumentAligner().align(solexs, hel1os)
    aligned.to_parquet("data/processed/merged_timeseries.parquet")
    
    # 3. Clean
    cleaned = DataCleaner().fit_predict(aligned)
    
    # 4. Features
    features = FeaturePipeline().extract(cleaned)
    features.to_parquet("data/processed/features.parquet")
    
    # 5. Nowcast catalog
    events = ThresholdNowcaster().detect(aligned)
    catalog = ThresholdNowcaster().to_catalog(events)
    catalog.to_csv("data/catalogs/nowcast_catalog.csv", index=False)
    logger.info("Nowcast catalog: {} events", len(catalog))
    
    # 6. Survival labels
    labels = build_survival_labels(aligned, catalog)
    labels.to_parquet("data/processed/survival_labels.parquet")
    
    # 7. Time-based train/val/test split
    train_end = pd.Timestamp("2024-06-01", tz="UTC")
    val_end   = pd.Timestamp("2024-09-01", tz="UTC")
    for name, mask in [("train", features.index <= train_end),
                       ("val",   (features.index > train_end) &
                                 (features.index <= val_end)),
                       ("test",  features.index > val_end)]:
        features[mask].to_parquet(f"data/processed/{name}_features.parquet")
        labels[mask].to_parquet(f"data/processed/{name}_labels.parquet")
        logger.info("Split {}: {} samples", name, mask.sum())

    logger.success("Dataset build complete.")

if __name__ == "__main__":
    main()
```

---

# PART 11 — CLI COMMAND REFERENCE

```bash
# ─── Environment Setup ──────────────────────────────────────────────────
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# ─── Data Download (from ISSDC PRADAN — manual step) ─────────────────
# Download SoLEXS L1 FITS → data/raw/solexs/
# Download HEL1OS L1 FITS → data/raw/hel1os/
# Optional: GOES XRS     → data/external/goes_xrs/

# ─── Data Pipeline ───────────────────────────────────────────────────────
python scripts/build_dataset.py

# ─── Feature Extraction (standalone, if needed separately) ───────────────
python scripts/extract_features.py

# ─── Nowcaster Training ──────────────────────────────────────────────────
python scripts/train_nowcaster.py --model threshold
python scripts/train_nowcaster.py --model cnn

# ─── Forecaster Training ─────────────────────────────────────────────────
python scripts/train_forecaster.py --model patchtst --experiment run_01
python scripts/train_forecaster.py --model timesnet  --experiment run_01
python scripts/train_forecaster.py --model lstm      --experiment run_01
python scripts/train_forecaster.py --model ensemble  --experiment run_01

# ─── Survival Model ──────────────────────────────────────────────────────
python scripts/train_forecaster.py --model survival

# ─── Build FAISS Memory Bank ─────────────────────────────────────────────
python scripts/build_memory_bank.py

# ─── Evaluation ──────────────────────────────────────────────────────────
python scripts/evaluate.py --model ensemble --split test

# ─── MLflow UI ───────────────────────────────────────────────────────────
mlflow ui --backend-store-uri experiments/mlruns

# ─── Start API ───────────────────────────────────────────────────────────
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ─── Start Dashboard ─────────────────────────────────────────────────────
streamlit run dashboard/app.py --server.port 8501

# ─── Run Tests ───────────────────────────────────────────────────────────
pytest tests/ -v --tb=short

# ─── Docker (full stack) ─────────────────────────────────────────────────
docker-compose up --build
```

---

# PART 12 — REQUIREMENTS.TXT

```
# Core
python==3.11.*
numpy==1.26.4
pandas==2.2.2
pyarrow==16.1.0
h5py==3.11.0

# Astronomy & Data
astropy==6.1.0
astroquery==0.4.7

# Signal Processing
scipy==1.13.1
PyWavelets==1.6.0
stumpy==1.13.0
tslearn==0.6.3

# ML
scikit-learn==1.5.1
xgboost==2.0.3
lightgbm==4.4.0

# Deep Learning
torch==2.3.1
torchvision==0.18.1
lightning==2.3.1
einops==0.8.0
torchmetrics==1.4.0

# Probabilistic & Survival
uncertainty-toolbox==0.1.2
lifelines==0.29.0

# Vector Search
faiss-cpu==1.8.0

# Experiment Tracking
mlflow==2.14.1

# API & Serving
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
websockets==12.0

# Dashboard
streamlit==1.36.0
plotly==5.22.0
altair==5.3.0

# Config & Utils
pyyaml==6.0.1
loguru==0.7.2
rich==13.7.1
hydra-core==1.3.2
tqdm==4.66.4

# Testing
pytest==8.2.2
pytest-cov==5.0.0
```

---

# PART 13 — DOCKER

## `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ libffi-dev libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## `docker-compose.yml`

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./configs:/app/configs
    environment:
      - PYTHONPATH=/app

  dashboard:
    build: .
    command: streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    depends_on:
      - api
    volumes:
      - ./data:/app/data

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.14.1
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --backend-store-uri /mlruns
    volumes:
      - ./experiments/mlruns:/mlruns
```

---

# PART 14 — UNIT TESTS

## `tests/test_solexs_reader.py`

```python
"""
Unit tests for SoLEXS FITS reader.
Run: pytest tests/test_solexs_reader.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.solexs_reader import SoLEXSReader

SAMPLE_FITS = "tests/fixtures/sample_solexs.fits"


@pytest.fixture
def reader():
    return SoLEXSReader()


class TestSoLEXSReader:
    def test_read_sample_fits(self, reader):
        """Basic read succeeds and returns expected columns."""
        df = reader.read_file(SAMPLE_FITS)
        assert "soft_flux" in df.columns
        assert "soft_counts" in df.columns
        assert "quality_flag" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_timestamp_is_utc(self, reader):
        df = reader.read_file(SAMPLE_FITS)
        assert str(df.index.tz) == "UTC"

    def test_flux_physical_range(self, reader):
        df = reader.read_file(SAMPLE_FITS)
        assert (df["soft_flux"].dropna() > 1e-12).all(), "Flux below physical minimum"
        assert (df["soft_flux"].dropna() < 1e-2).all(), "Flux above physical maximum"

    def test_corrupt_file_raises(self, reader):
        with pytest.raises(Exception):
            reader.read_file("tests/fixtures/corrupt.fits")

    def test_missing_header_warns(self, reader, caplog):
        reader.read_file("tests/fixtures/missing_header_solexs.fits")
        assert "Missing FITS keywords" in caplog.text

    def test_resampling_uniform_cadence(self, reader):
        df = reader.read_file(SAMPLE_FITS)
        diffs = df.index.to_series().diff().dropna()
        expected = pd.Timedelta(seconds=5)
        assert (diffs == expected).all(), "Cadence is not uniform 5s after resampling"

    def test_read_directory(self, reader, tmp_path):
        """Multi-file directory read returns chronologically sorted concatenation."""
        import shutil
        shutil.copy(SAMPLE_FITS, tmp_path / "a.fits")
        shutil.copy(SAMPLE_FITS, tmp_path / "b.fits")
        df = reader.read_directory(tmp_path)
        assert len(df) > 0
```

---

# PART 15 — EXPERIMENT LOG TEMPLATE

```
Experiment: 001
Model:      ThresholdNowcaster
Date:       YYYY-MM-DD
Dataset:    SoLEXS + HEL1OS [date range]
Split:      Train/Val/Test [dates]

Parameters:
  soft_k = 3.0
  hard_k = 2.5
  min_dur = 60s
  n_confirm = 3

Results:
  TSS:           ?
  HSS:           ?
  TPR:           ?
  FAR:           ?
  Lead Time (mean): ?
  Lead Time (median): ?
  Events detected: ?

Notes:
  - First baseline run
  - No ML involved

Conclusion:
  Threshold baseline provides TPR=? at FAR=?
  Compare all ML models against this baseline.

─────────────────────────────────────────

Experiment: 002
Model:      DualStreamPatchTST
Date:       YYYY-MM-DD
...
```

---

# PART 16 — CODING RULES (ABSOLUTE)

```
1.  Never read FITS directly inside model classes.
    Always use SoLEXSReader / HEL1OSReader.

2.  Never train models on aligned data without first running DataCleaner.

3.  Never split data randomly. Always split by date. Time order must be preserved.

4.  Every model must output uncertainty (MC Dropout or conformal).

5.  Every experiment must be logged to MLflow before the result is reported.

6.  All timestamps must be stored and processed in UTC.
    Never allow local timezone in any DataFrame index.

7.  Never hardcode paths. All paths come from configs/data.yaml.

8.  No magic numbers anywhere. Every threshold and parameter lives in configs/.

9.  Every function has a docstring with: purpose, inputs, outputs, physical meaning.

10. Every feature has a physical justification in the docstring.
    If you cannot write the physics, do not add the feature.

11. Every module is < 300 lines. Split into sub-modules if it exceeds this.

12. No preprocessing inside model forward() methods.
    Preprocessing belongs in feature_pipeline.py.

13. Every model inherits BaseModel. No model is standalone.

14. The nowcasting catalog (ThresholdNowcaster output) is the ground truth
    for all ML training. Never use raw flux thresholds as labels inside
    the ML training loop.

15. When uncertain about a physical quantity, do not guess.
    Add a TODO with the physics question and leave it for validation.
```

---

# PART 17 — PHYSICAL CONSTANTS AND REFERENCE

```
FLARE CLASSIFICATION (GOES 1–8 Angstrom equivalent):
  A-class: 1×10⁻⁸ – 1×10⁻⁷ W/m²
  B-class: 1×10⁻⁷ – 1×10⁻⁶ W/m²
  C-class: 1×10⁻⁶ – 1×10⁻⁵ W/m²
  M-class: 1×10⁻⁵ – 1×10⁻⁴ W/m²
  X-class: > 1×10⁻⁴ W/m²

SOLAR PHYSICS TIMESCALES:
  Flare impulsive phase:  10 seconds – 5 minutes
  Flare gradual phase:    5 minutes – several hours
  CME onset after X-class: 10–60 minutes
  Energetic particle arrival at L1: 10–60 minutes

ADITYA-L1 INSTRUMENT REFERENCE:
  SoLEXS:
    Energy band: 1.6 – 12.0 keV
    Detector: SDD (Silicon Drift Detector)
    Cadence: 1 second (L1 product)
    Sensitivity: A-class and above
  
  HEL1OS:
    Energy band: 10 – 150 keV
    Detector: CZT + NaI
    Cadence: 1 second (L1 product)
    Sensitivity: C-class and above (hard X-ray)

NEUPERT EFFECT:
  SXR(t) ≈ ∫₀ᵗ HXR(t') dt'
  HXR peak precedes SXR peak by 1–10 minutes
  This time lag is the physical basis for forecasting lead time.
  Reference: Neupert (1968), ApJ 153, L59

KEY PRECURSOR SIGNATURES IN LITERATURE:
  1. Hardness Ratio increase > 20% over 10 min → precursor sensitivity ~60%
  2. Hard X-ray flux rise at < 3σ (pre-impulsive) → detectable 5–30 min early
  3. Spectral hardening (γ decreasing) before impulsive phase
  4. Cross-correlation lag decrease between HXR and SXR channels
  All four features are implemented in physics_features.py
```

---

*End of SolarSense-AI Master Engineering Specification*  
*This document is the single source of truth for all implementation decisions.*  
*Claude Code: implement every module described above in order, starting with Part 3 (configs), then Part 4 (data pipeline), then Part 5 (features), then Part 6 (models).*
