import sys, warnings
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset
import lightning as L

from backend.models.forecaster.timesnet_forecaster import TimesNetFlareModule
from backend.training.trainer import SolarFlareDataset
from backend.evaluation.benchmark import Benchmark

CATALOG_PATH = "dataset/catalogs/nowcast_catalog.csv"
DATA_PATH = "dataset/processed/merged_timeseries.parquet"
LOOKBACK = 120
STEP = 60
BATCH_SIZE = 32
EPOCHS = 10

logger.info("Loading data...")
df = pd.read_parquet(DATA_PATH)
catalog = pd.read_csv(CATALOG_PATH)

labels = np.zeros(len(df), dtype=np.float32)
lead_times = np.zeros(len(df), dtype=np.float32)
for _, event in catalog.iterrows():
    peak = pd.to_datetime(event["peak_time"])
    start = pd.to_datetime(event["start_time"])
    if peak in df.index:
        idx = df.index.get_loc(peak)
        labels[idx] = 1.0
        lead_times[idx] = (peak - start).total_seconds()

soft_cols = [c for c in df.columns if c.startswith("soft_") and c in df.columns]
hard_cols = [c for c in df.columns if c.startswith("hard_") and c in df.columns]
Xs = df[soft_cols].fillna(0).values.astype(np.float32)
Xh = df[hard_cols].fillna(0).values.astype(np.float32)

n = len(df)
idx_val = int(n * 0.60)
idx_test = int(n * 0.80)

ds_train = SolarFlareDataset(Xs[:idx_val], Xh[:idx_val],
                              {"flare_label": labels[:idx_val], "lead_time": lead_times[:idx_val]},
                              lookback=LOOKBACK, step=STEP)
ds_val = SolarFlareDataset(Xs[idx_val:idx_test], Xh[idx_val:idx_test],
                            {"flare_label": labels[idx_val:idx_test], "lead_time": lead_times[idx_val:idx_test]},
                            lookback=LOOKBACK, step=STEP)

n_train = min(3000, len(ds_train))
n_val = min(500, len(ds_val))

train_subset = Subset(ds_train, list(range(n_train)))
val_subset = Subset(ds_val, list(range(n_val)))

flare_labels_sub = np.array([ds_train[i][2]["flare_label"] for i in range(n_train)])
weights = np.where(flare_labels_sub > 0, 10.0, 1.0)
from torch.utils.data import WeightedRandomSampler
sampler = WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

timesnet = TimesNetFlareModule(lookback=LOOKBACK, d_model=32, d_ff=64,
                                soft_channels=len(soft_cols), hard_channels=len(hard_cols))
trainer = L.Trainer(max_epochs=EPOCHS, accelerator="auto", log_every_n_steps=50,
                    logger=False, enable_checkpointing=False, enable_progress_bar=False)
trainer.fit(timesnet, train_loader, val_loader)

ds_test = SolarFlareDataset(Xs[idx_test:], Xh[idx_test:],
                             {"flare_label": labels[idx_test:], "lead_time": lead_times[idx_test:]},
                             lookback=LOOKBACK, step=STEP)
n_test = min(1000, len(ds_test))
timesnet.eval()
probs = []
y_true = []
lt_arr = []
with torch.no_grad():
    for i in range(n_test):
        xs, xh, y = ds_test[i]
        logit = timesnet(xs.unsqueeze(0), xh.unsqueeze(0))
        probs.append(torch.sigmoid(logit).item())
        y_true.append(y["flare_label"].item())
        lt_arr.append(y["lead_time"].item())

bm = Benchmark()
bm.add_model("TimesNet", np.array(y_true), np.array(probs), np.array(lt_arr))
summary = bm.summarize()
print("\nTimesNet Results:")
print(summary.to_string(index=False))

Path("models/checkpoints/timesnet_benchmark").mkdir(parents=True, exist_ok=True)
torch.save(timesnet.state_dict(), "models/checkpoints/timesnet_benchmark/best.ckpt")
logger.info("TimesNet complete")
