"""
Custom Training Callbacks
===========================
Lightning callbacks for the SolarSense-AI training pipeline.

Includes:
  - FlareMetricTracker: logs per-epoch TSS, HSS, Brier
  - LRScheduler: cosine annealing with warmup
  - GradientClipping: prevent loss spikes from rare events
  - ModelCheckpointing: saves best model by TSS
  - LeadTimeMonitor: tracks average lead time during training
"""

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from loguru import logger

from backend.training.metrics import true_skill_statistic, heidke_skill_score, brier_score


class FlareMetricTracker(Callback):
    """
    Compute and log TSS, HSS, Brier score at the end of each validation epoch.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.val_metrics = []

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not hasattr(trainer, "logged_metrics"):
            return
        if "val_loss" not in trainer.callback_metrics:
            return

        # Collect predictions from validation step outputs
        outputs = getattr(trainer, "val_outputs", None)
        if outputs is None or len(outputs) == 0:
            return

        all_preds = []
        all_targets = []
        for batch_out in outputs:
            logits = batch_out.get("logits")
            labels = batch_out.get("labels")
            if logits is not None and labels is not None:
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_preds.append(probs)
                all_targets.append(labels.detach().cpu().numpy())

        if not all_preds:
            return

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        y_bin = (y_pred > self.threshold).astype(int)

        tss = true_skill_statistic(y_true, y_bin)
        hss = heidke_skill_score(y_true, y_bin)
        brier = brier_score(y_true, y_pred)

        self.val_metrics.append({"tss": tss, "hss": hss, "brier": brier})

        trainer.logger.log_metrics({
            "val_tss": tss,
            "val_hss": hss,
            "val_brier": brier,
        }, step=trainer.global_step)

        logger.info("Epoch {} | TSS={:.4f} HSS={:.4f} Brier={:.4f}",
                    trainer.current_epoch, tss, hss, brier)


class LeadTimeMonitor(Callback):
    """
    Tracks average lead time (time from prediction to flare peak).
    Higher lead time = better operational value.
    """

    def __init__(self):
        self.lead_times = []

    def on_validation_batch_end(self, trainer, pl_module,
                                outputs, batch, batch_idx):
        lead_times = outputs.get("lead_times", None)
        if lead_times is not None:
            lt = lead_times.detach().cpu().numpy()
            self.lead_times.extend(lt[lt > 0].tolist())

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.lead_times:
            avg_lt = np.mean(self.lead_times)
            trainer.logger.log_metrics({"val_avg_lead_time": avg_lt},
                                       step=trainer.global_step)
            logger.info("Epoch {} | Avg lead time: {:.1f}s", trainer.current_epoch, avg_lt)
            self.lead_times.clear()


class GradientClippingCallback(Callback):
    """
    More aggressive gradient clipping for solar flare training.
    Flare events are rare spikes — loss landscape is highly uneven,
    and gradient norms can spike 100x on flare samples.
    """

    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        torch.nn.utils.clip_grad_norm_(pl_module.parameters(), self.max_norm)
