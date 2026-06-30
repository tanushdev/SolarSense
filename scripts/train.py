#!/usr/bin/env python
"""
Train all models and select the best performer.

Usage:
    python scripts/train.py
    python scripts/train.py --model rf       # Train only Random Forest
    python scripts/train.py --model xgb      # Train only XGBoost
    python scripts/train.py --model lstm     # Train only LSTM
    python scripts/train.py --model patchtst # Train only PatchTST
    python scripts/train.py --model timesnet # Train only TimesNet
"""

import sys
import warnings
warnings.filterwarnings("ignore", ".*num_workers.*")
warnings.filterwarnings("ignore", ".*Many workers.*")
warnings.filterwarnings("ignore", ".*persistent_workers.*")
import logging
logging.getLogger("lightning.pytorch.utilities.data").setLevel(logging.ERROR)
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.pipeline.training_pipeline import TrainingPipeline


def main():
    pipeline = TrainingPipeline()
    pipeline.run()
    print(f"Best model: {pipeline.winner}")


if __name__ == "__main__":
    main()
