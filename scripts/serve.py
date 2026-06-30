#!/usr/bin/env python
"""
Serve the SolarSense-AI API.

Starts a FastAPI server with endpoints:
  - GET  /health         → system status
  - POST /predict        → flare probability + uncertainty
  - POST /nowcast        → detect current events
  - GET  /history        → recent predictions
  - GET  /metrics        → model performance metrics

Usage:
    python scripts/serve.py                    # Dev server
    python scripts/serve.py --host 0.0.0.0 --port 8000  # Production
"""

import sys
from pathlib import Path
import argparse
import uvicorn

sys.path.append(str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Serve SolarSense-AI API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    print(f"Starting SolarSense-AI API on {args.host}:{args.port}")
    uvicorn.run(
        "backend.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
