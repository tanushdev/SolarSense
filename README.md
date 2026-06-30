# SolarSense-AI

SolarSense-AI is a real-time space weather intelligence platform that fetches live telemetry from the **NOAA GOES-19 satellite**, processes raw soft and hard X-ray flux readings, and executes deep learning/gradient-boosted forecasts to predict solar flare occurrences and precursor heating phases.

---

## Key Features

* **Real-Time GOES-19 Telemetry**: Automatically polls and co-aligns NOAA satellite flux measurements every 30 seconds.
* **XGBoost Prediction Core**: Fully trained classifier utilizing wavelets, rolling correlation, and Neupert effect signal proxies to predict flares with a **TSS of 0.68** and **ROC-AUC of 0.91**.
* **Physically Consistent Forecasts**: A multi-component stochastic simulator generating 3-hour predictions (1-minute resolution) featuring **1/f pink noise (fractional Brownian walk)** and physical solar flare profiles (rapid rise, exponential decay).
* **Interactive Space-Weather Dashboard**:
  * **Logarithmic Flux Scale**: Consolidated single-axis logarithmic plot mapping Soft and Hard X-ray fluxes.
  * **T = 0 Reference Line**: Vertical dividing line marking the boundary of observed vs. predicted forecasts.
  * **15-Min Shaded Forecast Window**: Clear highlighted prediction zone.
  * **Precursor Heating Alerts**: Automatically highlights impending flare precursor heating in **Red** when model probability crosses `50%`.

---

## Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** and **Node.js 18+** installed.

### 2. Backend (FastAPI & ML Inference)
```bash
# Clone the repository
git clone https://github.com/tanushdev/SolarSense.git
cd SolarSense

# Install python dependencies
pip install -r requirements.txt

# Start the FastAPI server (Dev mode with auto-reload)
python scripts/serve.py --reload
```
* The API will be live at `http://localhost:8000`.

### 3. Frontend (React, TypeScript & Vite)
```bash
cd frontend

# Install package dependencies
npm install

# Start the Vite development server
npm run dev
```
* The dashboard will be live at `http://localhost:5173`.

---

## Training & Evaluation

* **Run Unit Tests**: Ensure everything is fully operational by executing the test suite:
  ```bash
  python -m pytest
  ```
* **Model Benchmark Evaluation**: Load local checkpoints, evaluate on the test split, optimize the decision threshold, and generate comparative metric tables:
  ```bash
  python scripts/evaluate_completed.py
  ```
* **Full Training Pipeline**: Run feature extraction and re-train all models (Random Forest, XGBoost, LSTM):
  ```bash
  python backend/pipeline/training_pipeline.py
  ```

---

## Repository Structure
```
├── backend/
│   ├── api/             # FastAPI router endpoints & main.py
│   ├── evaluation/      # Calibrators, evaluation benchmarks
│   ├── features/        # Wavelets, rolling proxies extractor
│   ├── models/          # Deep learning & tree forecasters
│   └── services/        # NOAA live fetch & live predictor store
├── frontend/
│   ├── src/
│   │   ├── components/  # LightCurveChart & Recharts elements
│   │   ├── pages/       # Dashboard & Live Monitor pages
│   │   └── types/       # TypeScript type definitions
├── configs/             # YAML hyperparameter & pipeline configs
├── scripts/             # Serve, evaluation, and fit runner scripts
└── models/checkpoints/  # Deployed model binaries & optimized threshold meta
```
