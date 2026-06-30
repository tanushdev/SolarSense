#!/usr/bin/env python
"""
Build Dataset: end-to-end data processing pipeline.

Reads raw FITS files → aligns → cleans → extracts features → splits.
Output: train/val/test parquet files + nowcast catalog.

Usage:
    python scripts/build_dataset.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.pipeline.dataset_pipeline import DatasetPipeline


def main():
    pipeline = DatasetPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
