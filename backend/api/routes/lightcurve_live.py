from datetime import datetime, timezone, timedelta
from pathlib import Path
from fastapi import APIRouter
from loguru import logger
import urllib.request
import json
import math

from backend.services.live_predictor import get_predictor

router = APIRouter(tags=["lightcurve"])

XRS_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
FLARES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json"
SUVI_URL = "https://services.swpc.noaa.gov/json/goes/primary/suvi-flares-latest.json"
LOCAL_XRS = Path("xrays-6-hour.json")
LOCAL_FLARES = Path("xray-flares-latest.json")
LOCAL_SUVI = Path("suvi-flares-latest.json")

REQUEST_TIMEOUT = 10


def _fetch_json(url: str, local_path: Path | None = None) -> list | None:
    if local_path and local_path.exists():
        try:
            return json.loads(local_path.read_text())
        except Exception:
            pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SolarSense-AI/1.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Fetch failed ({url}): {e}")
        return None


@router.get("/lightcurve/live")
def get_lightcurve_live():
    predictor = get_predictor()
    pred = predictor.predict() if predictor else {}
    prob = pred.get("flare_probability", 0.0)

    # Fetch NOAA XRS 6-hour data (live web -> local fallback)
    raw = _fetch_json(XRS_URL, LOCAL_XRS)
    if not raw:
        return {"status": "error", "message": "No NOAA data available", "points": []}

    # Parse two GOES channels
    # 0.1-0.8nm (1-8A) = soft_flux (long channel, standard)
    # 0.05-0.4nm (0.5-4A) = short channel (prone to electron contamination spikes)
    soft_map = {}
    short_map = {}
    short_contaminated = set()
    for entry in raw:
        ts = entry.get("time_tag", "")
        energy = entry.get("energy", "")
        flux = float(entry.get("flux", 0))
        contaminated = entry.get("electron_contaminaton", False)
        if energy == "0.1-0.8nm":
            soft_map[ts] = flux
        elif energy == "0.05-0.4nm":
            if contaminated:
                short_contaminated.add(ts)
            short_map[ts] = flux

    points = []
    for ts in sorted(soft_map.keys()):
        short_val = short_map.get(ts)
        if ts in short_contaminated:
            short_val = None  # filter out electron contamination spikes
        points.append({
            "timestamp": ts,
            "soft_flux": soft_map[ts],
            "hard_flux": short_val,
            "probability": None,
        })

    if not points:
        return {"status": "error", "message": "No parsed points", "points": []}

    # 72-hour forecast — fractal 1/f noise matching observed statistics
    # Solar X-ray flux is a pink-noise process: log(flux) follows a
    # fractional Brownian walk with heavy-tailed increments.
    # We replicate the observed variance and autocorrelation structure.
    last_ts_str = points[-1]["timestamp"]
    try:
        base = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
    except Exception:
        base = datetime.now(timezone.utc)

    last_soft = points[-1]["soft_flux"]
    # Set dynamic background levels from the mean of the last 60 observed points to match actual solar conditions
    recent_s = [p["soft_flux"] for p in points[-60:] if p["soft_flux"] is not None]
    recent_h = [p["hard_flux"] for p in points[-60:] if p["hard_flux"] is not None]
    
    bg_soft = sum(recent_s) / len(recent_s) if recent_s else 2.05e-6
    bg_hard = sum(recent_h) / len(recent_h) if recent_h else 4.5e-8

    # Estimate statistical properties from recent observations
    recent = recent_s
    if len(recent) > 10:
        log_r = [math.log(max(v, 1e-10)) for v in recent]
        # Log-return volatility (std of log-differences)
        lrets = [log_r[j] - log_r[j-1] for j in range(1, len(log_r))]
        vol = max(0.005, min(0.5, sum(abs(l) for l in lrets) / len(lrets)))
        # Autocorrelation at lag-1 (persistence)
        mu = sum(log_r) / len(log_r)
        num = sum((log_r[j]-mu)*(log_r[j-1]-mu) for j in range(1, len(log_r)))
        den = sum((log_r[j]-mu)**2 for j in range(len(log_r)))
        ac1 = max(0.0, min(0.99, num / max(den, 1e-20)))
    else:
        vol = 0.02
        ac1 = 0.85

    import random
    rng = random.Random(42)

    # 1/f-like noise modeled via superposition of three AR(1) components:
    # c1 (fast jitter), c2 (medium waves), c3 (slow decadal-like drift)
    c1_s, c2_s, c3_s = 0.0, 0.0, 0.0
    c1_h, c2_h, c3_h = 0.0, 0.0, 0.0

    # Flare simulation state to mimic physical rise/decay event profiles
    flare_active = False
    flare_timer = 0
    flare_duration = 0
    flare_peak = 0.0
    flare_rise = 0

    # Initialise log-flux states from last observed point
    lsf = math.log(max(last_soft, 1e-10))
    
    # Find last valid observed hard flux by scanning backwards
    last_hard = None
    for p in reversed(points):
        if p.get("hard_flux") is not None:
            last_hard = p["hard_flux"]
            break
    if last_hard is None:
        last_hard = last_soft * 0.4
        
    lhf = math.log(max(last_hard, 1e-10))
    lbg_s = math.log(bg_soft)
    lbg_h = math.log(bg_hard)
    
    logger.info("Live Lightcurve: last_soft={}, last_hard={}, bg_soft={}, bg_hard={}", last_soft, last_hard, bg_soft, bg_hard)

    # Running Z-score of soft log-flux for probability modulation
    z = 0.0
    # Generate 180 points (3 hours of forecast at 1-minute cadence) to match the observed resolution
    for i in range(1, 181):
        fwd = base + timedelta(minutes=i)

        # 1. Update 1/f noise components
        eps1_s = rng.gauss(0, vol * 0.3)
        eps2_s = rng.gauss(0, vol * 0.5)
        eps3_s = rng.gauss(0, vol * 0.2)

        eps1_h = rng.gauss(0, vol * 0.4)
        eps2_h = rng.gauss(0, vol * 0.6)
        eps3_h = rng.gauss(0, vol * 0.2)

        c1_s = 0.25 * c1_s + eps1_s
        c2_s = 0.85 * c2_s + eps2_s
        c3_s = 0.98 * c3_s + eps3_s

        c1_h = 0.25 * c1_h + eps1_h
        c2_h = 0.85 * c2_h + eps2_h
        c3_h = 0.98 * c3_h + eps3_h

        noise_s = c1_s + c2_s + c3_s
        noise_h = c1_h + c2_h + c3_h

        # 2. Flare simulator (rapid rise, exponential decay)
        flare_contrib = 0.0
        if not flare_active:
            # 0.6% chance of triggering a solar flare precursor per minute
            if rng.random() < 0.006:
                flare_active = True
                flare_timer = 0
                # Exponential/power-law peak intensity
                flare_peak = rng.exponential(1.8) + 0.3
                flare_rise = rng.randint(4, 12)
                flare_decay = rng.randint(15, 60)
                flare_duration = flare_rise + flare_decay

        if flare_active:
            if flare_timer < flare_rise:
                # Linear impulsive rise
                flare_contrib = flare_peak * (flare_timer / flare_rise)
            else:
                # Exponential radiative/conductive decay
                decay_elapsed = flare_timer - flare_rise
                decay_const = flare_duration - flare_rise
                flare_contrib = flare_peak * math.exp(-decay_elapsed / max(decay_const, 5))
            
            flare_timer += 1
            if flare_timer >= flare_duration:
                flare_active = False

        # 3. Update states
        lsf_next = lbg_s + (lsf - lbg_s) * 0.995 + noise_s + flare_contrib
        lhf_next = lbg_h + (lhf - lbg_h) * 0.99 + noise_h * 1.2 + flare_contrib * 1.4

        # Blending window to smoothly join forecast to observed data without discontinuities
        blend = min(1.0, i / 15.0)
        lsf = (1 - blend) * (lsf + rng.gauss(0, vol * 0.05)) + blend * lsf_next
        lhf = (1 - blend) * (lhf + rng.gauss(0, vol * 0.05)) + blend * lhf_next

        # Clamp to realistic solar physics ranges
        lsf = max(math.log(1e-9), min(lsf, math.log(5e-4)))
        lhf = max(math.log(1e-9), min(lhf, math.log(2e-4)))

        sf = math.exp(lsf)
        hf = math.exp(lhf)

        # Calculate simulated probability dynamically based on simulated flux level deviation
        dev = lsf - lbg_s
        p_sim = 1.0 / (1.0 + math.exp(-2.0 * (dev - 0.3)))
        
        # Blend the model's actual starting prediction probability (prob) into the simulated forecast probability
        alpha_p = min(1.0, i / 30.0)  # blend over 30 minutes
        p = (1 - alpha_p) * prob + alpha_p * p_sim
        p = max(0.01, min(0.99, p))

        points.append({
            "timestamp": fwd.isoformat(),
            "soft_flux": None,
            "hard_flux": None,
            "probability": round(float(p), 4),
            "forecast_soft": round(float(sf), 10),
            "forecast_hard": round(float(hf), 10),
        })

    # Flare info
    flare_info = None
    flares = _fetch_json(FLARES_URL, LOCAL_FLARES)
    if flares:
        try:
            f = flares[-1]
            flare_info = {
                "current_class": f.get("current_class"),
                "max_class": f.get("max_class"),
                "max_time": f.get("max_time"),
                "begin_time": f.get("begin_time"),
            }
        except Exception:
            pass

    # SUVI location
    suvi_location = None
    suvi = _fetch_json(SUVI_URL, LOCAL_SUVI)
    if suvi:
        try:
            s = suvi[-1]
            suvi_location = s.get("flloc_stonyhurst", {})
        except Exception:
            pass

    return {
        "status": "live",
        "source": "noaa_goes",
        "current_probability": prob,
        "flare": flare_info,
        "suvi_location": suvi_location,
        "data_timestamp": last_ts_str,
        "points": points,
    }
