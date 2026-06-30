"""
Check actual UTC overlap between SoLEXS and HEL1OS FITS files.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from astropy.io import fits
from astropy.time import Time
import numpy as np
import pandas as pd

print("=" * 60)
print("SoLEXS Time Ranges (sample files)")
print("=" * 60)

solexs_dirs = sorted(Path("dataset/raw/solexs").iterdir())
print(f"Total SoLEXS date directories: {len(solexs_dirs)}")
print(f"Date range: {solexs_dirs[0].name} to {solexs_dirs[-1].name}")

# Check 5 SoLEXS files: first, last, and 3 middle
solexs_checks = [0, len(solexs_dirs)//4, len(solexs_dirs)//2, 3*len(solexs_dirs)//4, -1]
for idx in solexs_checks:
    d = solexs_dirs[idx]
    lc_files = list(d.rglob("*.lc.gz"))
    lc_files = [f for f in lc_files if "gti" not in f.name and "pi" not in f.name]
    if lc_files:
        f = lc_files[0]
        with fits.open(str(f)) as hdul:
            data = hdul[1].data
            times = data["TIME"]
            t0 = Time(times[0], format="unix", scale="utc")
            t1 = Time(times[-1], format="unix", scale="utc")
            print(f"\n  {d.name.split('_')[3]}:")
            print(f"    Start: {t0.isot} UTC")
            print(f"    End:   {t1.isot} UTC")
            print(f"    Duration: {(times[-1]-times[0])/3600:.1f}h")
            print(f"    Samples: {len(times)}")
    else:
        print(f"\n  {d.name}: NO .lc.gz files found")

print("\n" + "=" * 60)
print("HEL1OS Time Ranges (sample files)")
print("=" * 60)

hel1os_files = sorted(Path("dataset/raw/hel1os").rglob("lightcurve_cdte*.fits"))
print(f"Total HEL1OS lightcurve files: {len(hel1os_files)}")

# Group by date directory
dates = set()
for f in hel1os_files:
    dates.add(f.parent.parent.parent.name)
print(f"Unique date directories (YYYY/MM/DD): check parent structure")

# Check 5 HEL1OS files spanning the range
hel1os_checks = [0, len(hel1os_files)//4, len(hel1os_files)//2, 3*len(hel1os_files)//4, -1]
for idx in hel1os_checks:
    f = hel1os_files[idx]
    rel = f.relative_to("dataset/raw/hel1os")
    parts = str(rel).replace("\\", "/")
    with fits.open(str(f)) as hdul:
        pri = hdul[0].header
        data = hdul[1].data
        mjd = data["MJD"]
        t0 = Time(mjd[0], format="mjd", scale="utc")
        t1 = Time(mjd[-1], format="mjd", scale="utc")
        print(f"\n  {parts}:")
        print(f"    Start: {t0.isot} UTC")
        print(f"    End:   {t1.isot} UTC")
        print(f"    Duration: {(mjd[-1]-mjd[0])*24*60:.1f} min")
        print(f"    Samples: {len(mjd)}")
        print(f"    ISOSTART: {pri.get('ISOSTART','N/A')}")

# Now check a specific overlapping date
print("\n" + "=" * 60)
print("SPECIFIC OVERLAP CHECK: 2025-04-30")
print("=" * 60)

solexs_date = "2025-04-30"
hel1os_date_path = Path("dataset/raw/hel1os/2025/04/30")

# Find SoLEXS file for this date
for d in solexs_dirs:
    date_str = d.name.split("_")[3]  # AL1_SLX_L1_20250430_v1.0 -> 20250430
    if date_str == "20250430":
        lc_files = list(d.rglob("*.lc.gz"))
        lc_files = [f for f in lc_files if "gti" not in f.name and "pi" not in f.name]
        if lc_files:
            with fits.open(str(lc_files[0])) as hdul:
                data = hdul[1].data
                times = data["TIME"]
                t0 = Time(times[0], format="unix", scale="utc")
                t1 = Time(times[-1], format="unix", scale="utc")
                print(f"\n  SoLEXS ({lc_files[0].name}):")
                print(f"    Start: {t0.isot}")
                print(f"    End:   {t1.isot}")
                print(f"    Duration: {(times[-1]-times[0])/3600:.1f}h")

# Find HEL1OS files for this date
hel_dirs = [d for d in hel1os_date_path.iterdir() if d.is_dir()]
print(f"\n  HEL1OS session dirs: {len(hel_dirs)}")
for hel_dir in sorted(hel_dirs):
    cdte_files = list(hel_dir.rglob("lightcurve_cdte*.fits"))
    for cf in cdte_files:
        with fits.open(str(cf)) as hdul:
            data = hdul[1].data
            mjd = data["MJD"]
            t0 = Time(mjd[0], format="mjd", scale="utc")
            t1 = Time(mjd[-1], format="mjd", scale="utc")
            print(f"\n  HEL1OS ({cf.parent.parent.name}/{cf.name}):")
            print(f"    Start: {t0.isot}")
            print(f"    End:   {t1.isot}")
            print(f"    Duration: {(mjd[-1]-mjd[0])*24*60:.1f} min")

# Compute actual overlap
print("\n" + "=" * 60)
print("COMPUTE OVERLAP FORMATIVE TABLE")
print("=" * 60)

overlap_rows = []
for d in solexs_dirs[:10]:  # first 10 for demo
    date_str = d.name.split("_")[3]
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    
    # SoLEXS time range
    lc_files = list(d.rglob("*.lc.gz"))
    lc_files = [f for f in lc_files if "gti" not in f.name and "pi" not in f.name]
    if not lc_files:
        continue
    
    with fits.open(str(lc_files[0])) as hdul:
        data = hdul[1].data
        times = data["TIME"]
        s_start = Time(times[0], format="unix", scale="utc")
        s_end = Time(times[-1], format="unix", scale="utc")
    
    # HEL1OS time range for same date
    hel_path = Path(f"dataset/raw/hel1os/{year}/{month}/{day}")
    if not hel_path.exists():
        continue
    
    hel_t0 = None
    hel_t1 = None
    for hel_dir in sorted(hel_path.iterdir()):
        if not hel_dir.is_dir():
            continue
        for cf in sorted(hel_dir.rglob("lightcurve_cdte*.fits")):
            with fits.open(str(cf)) as hdul:
                data = hdul[1].data
                mjd = data["MJD"]
                t0 = Time(mjd[0], format="mjd", scale="utc")
                t1 = Time(mjd[-1], format="mjd", scale="utc")
                if hel_t0 is None or t0 < hel_t0:
                    hel_t0 = t0
                if hel_t1 is None or t1 > hel_t1:
                    hel_t1 = t1
    
    if hel_t0 is None:
        continue
    
    # Overlap
    overlap_start = max(s_start, hel_t0)
    overlap_end = min(s_end, hel_t1)
    overlap_sec = max(0, (overlap_end - overlap_start).sec)
    
    overlap_rows.append({
        "date": date_str,
        "solexs_start": s_start.isot,
        "solexs_end": s_end.isot,
        "solexs_dur_h": round((s_end - s_start).sec / 3600, 1),
        "hel1os_start": hel_t0.isot,
        "hel1os_end": hel_t1.isot,
        "hel1os_dur_min": round((hel_t1 - hel_t0).sec / 60, 1),
        "overlap_sec": int(overlap_sec),
        "overlap_min": round(overlap_sec / 60, 1),
    })

df_overlap = pd.DataFrame(overlap_rows)
print(df_overlap.to_string(index=False))

total_overlap_min = df_overlap["overlap_min"].sum()
print(f"\nTotal overlap (first 10 dates): {total_overlap_min:.0f} minutes")
