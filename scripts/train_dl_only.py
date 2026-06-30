import sys, warnings, logging, yaml, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
from backend.training.trainer import SolarFlareDataset, SolarFlareTrainer
from backend.models.forecaster.lstm_forecaster import LSTMFlareModule
from backend.models.forecaster.patchtst_forecaster import DualStreamPatchTST
from backend.models.forecaster.timesnet_forecaster import TimesNetFlareModule

with open("configs/training.yaml") as f: cfg = yaml.safe_load(f)
with open("configs/models.yaml") as f: models_cfg = yaml.safe_load(f)
suffix = cfg["data"].get("file_suffix","")
processed_dir = Path(cfg["data"]["processed_dir"])
catalog_path = Path(cfg["data"]["catalog_path"])
lookback = cfg["data"]["lookback_samples"]
step = cfg["data"]["window_step"]

train_df = pd.read_parquet(processed_dir / f"train_timeseries{suffix}.parquet")
val_df = pd.read_parquet(processed_dir / f"val_timeseries{suffix}.parquet")
test_df = pd.read_parquet(processed_dir / f"test_timeseries{suffix}.parquet")
catalog = pd.read_csv(catalog_path)

sc = [c for c in train_df.columns if c.startswith("soft_")]
hc = [c for c in train_df.columns if c.startswith("hard_")]
cross = {"soft":["quality_solexs","d_soft_flux_dt","d2_soft_flux_dt2","d_log_soft_flux_dt"],
         "hard":["quality_hel1os","d_hard_flux_dt","d2_hard_flux_dt2","d_log_hard_flux_dt",
                 "event_rate_1h","event_rate_6h","event_rate_24h","flux_above_2sigma"],
         "both":["hardness_ratio","hardness_ratio_log","hardness_ratio_deriv",
                 "hardness_ratio_5min","hardness_ratio_15min","spectral_index",
                 "spectral_index_5min","spectral_index_deriv","hxr_sxr_corr_15min",
                 "hxr_sxr_lag1_corr","neupert_proxy","neupert_residual",
                 "neupert_residual_abs","hour_sin","hour_cos","doy_sin","doy_cos",
                 "data_gap","artifact_flag"]}
sc.extend(cross["soft"]); sc.extend(cross["both"])
hc.extend(cross["hard"]); hc.extend(cross["both"])
sc = [c for c in sc if c in train_df.columns]
hc = [c for c in hc if c in train_df.columns]

horizon_min = cfg["data"].get("forecast_horizons_minutes",[5])[0]
horizon_sec = horizon_min * 60
class_map = {"A":0,"B":1,"C":2,"M":3,"X":4}

def build_labels(df):
    labels = np.zeros(len(df), dtype=np.float32)
    lead_times = np.zeros(len(df), dtype=np.float32)
    flare_class = np.full(len(df), -1, dtype=np.int32)
    t0, t1 = df.index[0], df.index[-1]
    cat = catalog.copy()
    cat["peak_time"] = pd.to_datetime(cat["peak_time"])
    cat = cat[(cat["peak_time"] >= t0 - pd.Timedelta(seconds=horizon_sec)) & (cat["peak_time"] <= t1)]
    if len(cat) == 0:
        return {"flare_label":labels,"lead_time":lead_times,"flare_class":flare_class}
    peak_ns = cat["peak_time"].values.astype(np.int64)
    t_ns = df.index.values.astype(np.int64)
    horizon_ns = horizon_sec * 1_000_000_000
    left = np.searchsorted(t_ns, peak_ns - horizon_ns, side="left")
    right = np.searchsorted(t_ns, peak_ns, side="right")
    for j in range(len(cat)):
        lo, hi = left[j], right[j]
        if lo < hi:
            labels[lo:hi] = 1.0
            cls_str = str(cat.iloc[j].get("flare_class","C"))[0].upper()
            flare_class[lo:hi] = class_map.get(cls_str, 2)
            lt_slice = lead_times[lo:hi]
            new_mask = (lt_slice == 0).nonzero()[0]
            if len(new_mask) > 0:
                idxs = lo + new_mask
                lead_times[idxs] = np.maximum((peak_ns[j] - t_ns[idxs]) / 1_000_000_000, 0)
    return {"flare_label":labels,"lead_time":lead_times,"flare_class":flare_class}

train_labels = build_labels(train_df)
val_labels = build_labels(val_df)
test_labels = build_labels(test_df)

Xs_train = train_df[sc].values.astype(np.float32)
Xh_train = train_df[hc].values.astype(np.float32)
Xs_val = val_df[sc].values.astype(np.float32)
Xh_val = val_df[hc].values.astype(np.float32)
Xs_test = test_df[sc].values.astype(np.float32)
Xh_test = test_df[hc].values.astype(np.float32)

ds_train = SolarFlareDataset(Xs_train, Xh_train, train_labels, lookback, step)
ds_val = SolarFlareDataset(Xs_val, Xh_val, val_labels, lookback, step)
ds_test = SolarFlareDataset(Xs_test, Xh_test, test_labels, lookback, step)
print(f"Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")
print(f"Features: soft={Xs_train.shape[1]}, hard={Xh_train.shape[1]}")

test_idx = np.array(ds_test.indices)
y_test = test_labels["flare_label"][test_idx]
lt_test = test_labels["lead_time"][test_idx]
from backend.evaluation.benchmark import Benchmark
benchmark = Benchmark()

def predict(model, ds, batch_size=256):
    model.eval()
    model.to("cuda")
    preds = []
    for i in range(0, len(ds), batch_size):
        batch = [ds[j] for j in range(i, min(i+batch_size, len(ds)))]
        xs = torch.stack([b[0] for b in batch]).cuda()
        xh = torch.stack([b[1] for b in batch]).cuda()
        with torch.no_grad():
            out = model(xs, xh)
            if isinstance(out, dict):
                p = out["flare_prob"].cpu().numpy().squeeze()
            else:
                p = torch.sigmoid(out).cpu().numpy()
        preds.append(p)
    return np.concatenate(preds)

# LSTM
print("=" * 60)
print("Training LSTM...")
lstm_cfg = models_cfg.get("forecaster",{}).get("lstm",{})
lstm = LSTMFlareModule(
    input_dim=Xs_train.shape[1]+Xh_train.shape[1],
    hidden_dim=lstm_cfg.get("hidden_size",64),
    num_layers=lstm_cfg.get("num_layers",1),
    bidirectional=lstm_cfg.get("bidirectional",False),
    dropout=lstm_cfg.get("dropout",0.2),
    learning_rate=lstm_cfg.get("learning_rate",0.001))
SolarFlareTrainer().train_forecaster(lstm, ds_train, ds_val, experiment_name="lstm_benchmark", model_name="lstm")
lstm_probs = predict(lstm, ds_test)
benchmark.add_model("LSTM", y_test, lstm_probs, lt_test)
r = benchmark.results["LSTM"]
print(f"LSTM: TSS={r.tss:.4f} HSS={r.hss:.4f} Brier={r.brier:.4f}")

# PatchTST
print("=" * 60)
print("Training PatchTST...")
ptst = DualStreamPatchTST(n_soft_features=Xs_train.shape[1], n_hard_features=Xh_train.shape[1])
SolarFlareTrainer().train_forecaster(ptst, ds_train, ds_val, experiment_name="patchtst_benchmark", model_name="patchtst")
ptst_probs = predict(ptst, ds_test)
benchmark.add_model("PatchTST", y_test, ptst_probs, lt_test)
r = benchmark.results["PatchTST"]
print(f"PatchTST: TSS={r.tss:.4f} HSS={r.hss:.4f} Brier={r.brier:.4f}")

# TimesNet
print("=" * 60)
print("Training TimesNet...")
tn = TimesNetFlareModule(soft_channels=Xs_train.shape[1], hard_channels=Xh_train.shape[1])
SolarFlareTrainer().train_forecaster(tn, ds_train, ds_val, experiment_name="timesnet_benchmark", model_name="timesnet")
tn_probs = predict(tn, ds_test)
benchmark.add_model("TimesNet", y_test, tn_probs, lt_test)
r = benchmark.results["TimesNet"]
print(f"TimesNet: TSS={r.tss:.4f} HSS={r.hss:.4f} Brier={r.brier:.4f}")

print("=" * 60)
print("FINAL BENCHMARK")
for name, r in benchmark.results.items():
    print(f"  {name}: TSS={r.tss:.4f} HSS={r.hss:.4f} Brier={r.brier:.4f}")
