"""
SolarSense-AI — Run Everything
================================
Single entry point: build dataset, train models, serve API.

Usage:
    python run.py                  # Full pipeline + API
    python run.py --skip-train     # Skip training, only serve
    python run.py --skip-build     # Skip dataset build, train + serve
    python run.py --serve-only     # Only start API (models must exist)
    python run.py --check          # Check prerequisites only
"""

import sys, subprocess, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

def check():
    ok = True
    print(f"[1/5] Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("  ✗ Need Python >= 3.11")
        ok = False
    else:
        print("  ✓ OK")

    try:
        import xgboost; print(f"[2/5] xgboost {xgboost.__version__} ✓")
    except ImportError:
        print("  ✗ xgboost not installed — run: pip install -r requirements.txt"); ok = False

    solexs = list((ROOT / "dataset/raw/solexs").glob("*.fits"))
    hel1os = list((ROOT / "dataset/raw/hel1os").glob("*.fits"))
    print(f"[3/5] SoLEXS FITS: {len(solexs)} {'⚠ empty!' if not solexs else '✓'}")
    print(f"[4/5] HEL1OS FITS: {len(hel1os)} {'⚠ empty!' if not hel1os else '✓'}")

    models = list((ROOT / "models/checkpoints/xgboost").glob("*.pkl"))
    print(f"[5/5] Trained models: {len(models)} {'⚠ none — run training' if not models else '✓'}")
    return ok

def step(label, cmd):
    print(f"\n{'='*60}\n=== {label}\n{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"✗ {label} failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"✓ {label} complete\n")

def main():
    parser = argparse.ArgumentParser(description="SolarSense-AI: build → train → serve")
    parser.add_argument("--skip-build", action="store_true", help="Skip dataset build")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--serve-only", action="store_true", help="Only start the API server")
    parser.add_argument("--check", action="store_true", help="Check prerequisites and exit")
    parser.add_argument("--host", default="0.0.0.0", help="API bind address")
    parser.add_argument("--port", type=int, default=8000, help="API bind port")
    args = parser.parse_args()

    if args.check:
        check()
        sys.exit(0)

    if args.serve_only:
        step("Starting API server", [sys.executable, str(SCRIPTS / "serve.py"), "--host", args.host, "--port", str(args.port)])
        return

    if not args.skip_build:
        step("Building dataset", [sys.executable, str(SCRIPTS / "build_dataset.py")])

    if not args.skip_train:
        step("Training binary XGBoost + benchmarking", [sys.executable, str(SCRIPTS / "train.py")])
        step("Training multi-class XGBoost", [sys.executable, str(SCRIPTS / "train_multiclass.py")])

    step("Starting API server", [sys.executable, str(SCRIPTS / "serve.py"), "--host", args.host, "--port", str(args.port)])

if __name__ == "__main__":
    main()
