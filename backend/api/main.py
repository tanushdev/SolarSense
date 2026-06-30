"""
SolarSense-AI FastAPI Application
===================================
Real-time inference API for nowcasting and forecasting.

Route modules:
  /health   → backend.api.routes.health
  /nowcast  → backend.api.routes.nowcast
  /predict  → backend.api.routes.predict
  /history  → backend.api.routes.history
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from loguru import logger

from backend.api.routes.health import router as health_router
from backend.api.routes.nowcast import router as nowcast_router
from backend.api.routes.predict import router as predict_router
from backend.api.routes.history import router as history_router
from backend.api.routes.validation import router as validation_router
from backend.api.routes.lightcurve import router as lightcurve_router
from backend.api.routes.lightcurve_live import router as lightcurve_live_router
from backend.api.routes.noaa import router as noaa_router
from backend.api.routes.metrics import router as metrics_router
from backend.api.routes.models_info import router as models_router
from backend.api.routes.alerts import router as alerts_router
from backend.api.routes.datasets import router as datasets_router
from backend.api.routes.forecast_timeseries import router as forecast_ts_router

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

app.include_router(health_router)
app.include_router(nowcast_router)
app.include_router(predict_router)
app.include_router(history_router)
app.include_router(validation_router)
app.include_router(lightcurve_router)
app.include_router(noaa_router)
app.include_router(metrics_router)
app.include_router(models_router)
app.include_router(alerts_router)
app.include_router(datasets_router)
app.include_router(forecast_ts_router)
app.include_router(lightcurve_live_router)

logger.info("SolarSense-AI API ready | {} routes loaded",
            len(app.routes))

if __name__ == "__main__":
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=False)
