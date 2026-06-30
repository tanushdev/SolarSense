import threading
import time as time_module
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger
import urllib.request
import urllib.error

XRS_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
FLARES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json"
CACHE_FILE = Path("backend/services/noaa_cache.json")
LOCAL_XRS = Path("xrays-6-hour.json")
LOCAL_FLARES = Path("xray-flares-latest.json")

CACHE_TTL = 60
REQUEST_TIMEOUT = 10


def _fetch_json(url: str, local_path: Optional[Path] = None) -> Optional[list]:
    if local_path and local_path.exists():
        try:
            data = json.loads(local_path.read_text())
            logger.info(f"NOAA: loaded from local file {local_path}")
            return data
        except Exception as e:
            logger.warning(f"NOAA local file error ({local_path}): {e}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SolarSense-AI/1.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as e:
        logger.warning(f"NOAA fetch failed ({url}): {e}")
        return None


def _classify_flux(w_m2: float) -> str:
    if w_m2 >= 1e-4:
        return "X"
    elif w_m2 >= 1e-5:
        return "M"
    elif w_m2 >= 1e-6:
        return "C"
    elif w_m2 >= 1e-7:
        return "B"
    elif w_m2 >= 1e-8:
        return "A"
    return "A"


def _read_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _write_cache(data: dict):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"NOAA cache write failed: {e}")


class NOAAService:
    def __init__(self):
        self._cache = {
            "timestamp": None,
            "flux_w_m2": None,
            "flux_class": "A",
            "flare_class": None,
            "flare_peak_time": None,
            "data_time": None,
            "status": "initializing",
            "error": None,
        }
        self._lock = threading.Lock()
        self._running = False
        self._refresh_thread = None

    def refresh(self):
        xrs_raw = _fetch_json(XRS_URL, local_path=LOCAL_XRS)
        flares_raw = _fetch_json(FLARES_URL, local_path=LOCAL_FLARES)

        now_iso = datetime.now(timezone.utc).isoformat()
        has_data = False

        with self._lock:
            self._cache["timestamp"] = now_iso
            self._cache["status"] = "cached"

            if xrs_raw and len(xrs_raw) > 0:
                try:
                    latest = xrs_raw[-1]
                    w_m2 = float(latest.get("flux", 0))
                    time_tag = latest.get("time_tag", now_iso)
                    self._cache["flux_w_m2"] = w_m2
                    self._cache["flux_class"] = _classify_flux(w_m2)
                    self._cache["data_time"] = time_tag
                    self._cache["error"] = None
                    self._cache["status"] = "live"
                    has_data = True
                except (TypeError, ValueError, KeyError) as e:
                    logger.warning(f"NOAA XRS parse error: {e}")

            if flares_raw and len(flares_raw) > 0:
                try:
                    latest_flare = flares_raw[-1]
                    self._cache["flare_class"] = latest_flare.get("class", None)
                    self._cache["flare_peak_time"] = latest_flare.get("peak_time", None)
                    has_data = True
                except (TypeError, ValueError, KeyError) as e:
                    logger.warning(f"NOAA flares parse error: {e}")

            if has_data:
                _write_cache(self._cache)
            else:
                cached = _read_cache()
                if cached:
                    self._cache["flux_w_m2"] = cached.get("flux_w_m2")
                    self._cache["flux_class"] = cached.get("flux_class", "A")
                    self._cache["flare_class"] = cached.get("flare_class")
                    self._cache["flare_peak_time"] = cached.get("flare_peak_time")
                    self._cache["data_time"] = cached.get("data_time")
                    self._cache["error"] = cached.get("error")
                    self._cache["status"] = "stale"
                    logger.info("NOAA: using cached data (stale)")

            logger.info(
                f"NOAA: flux={self._cache['flux_w_m2']}, "
                f"class={self._cache['flux_class']}, "
                f"flare={self._cache['flare_class']}, "
                f"status={self._cache['status']}"
            )

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._cache)

    def start_auto_refresh(self, interval_seconds: int = CACHE_TTL):
        self._running = True

        def _loop():
            self.refresh()
            while self._running:
                time_module.sleep(interval_seconds)
                try:
                    self.refresh()
                except Exception as e:
                    logger.error(f"NOAA auto-refresh error: {e}")

        self._refresh_thread = threading.Thread(target=_loop, daemon=True)
        self._refresh_thread.start()
        logger.info(f"NOAA: auto-refresh every {interval_seconds}s started")

    def stop(self):
        self._running = False


_noaa_instance = None


def get_noaa() -> NOAAService:
    global _noaa_instance
    if _noaa_instance is None:
        _noaa_instance = NOAAService()
        _noaa_instance.start_auto_refresh(interval_seconds=CACHE_TTL)
    return _noaa_instance
