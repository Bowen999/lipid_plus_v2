"""
trainer.py — Model-agnostic multi-task training loop.

Handles:
  - Multi-task weighted cross-entropy with epoch-0 normalisation
  - AdamW + CosineAnnealingLR scheduler
  - Early stopping on val key-target accuracy (class + chain-1)
  - Checkpoint save / restore
  - Per-epoch logging to console and training_log.txt
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import TARGETS
from training.losses import MultiTaskLoss

# Targets used as early-stopping proxy (class + chain-1)
_ES_TARGETS = ["class_enc", "num_c_1", "num_db_1", "num_ox_1"]


class Trainer:
    """
    Parameters
    ----------
    model        : any LipidMLP / LipidCNN / LipidTransformer instance
    train_loader : DataLoader for training split
    val_loader   : DataLoader for validation split
    config       : dict from configs/*.json
    device       : torch.device
    out_dir      : Path to outputs/{model_name}/
    model_name   : str for logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        out_dir: Path,
        model_name: str,
        l3_eval_fn=None,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.out_dir      = out_dir
        self.model_name   = model_name

        p = config.get("training", {})
        self.max_epochs = int(p.get("max_epochs", 100))
        self.patience   = int(p.get("patience", 10))
        lr              = float(p.get("lr", 3e-4))
        wd              = float(p.get("weight_decay", 1e-4))
        eta_min         = float(p.get("eta_min", 1e-6))
        scheduler_type  = p.get("scheduler", "cosine")

        self.optimizer  = AdamW(model.parameters(), lr=lr, weight_decay=wd)

        if scheduler_type == "cosine_restarts":
            t0     = int(p.get("T_0", 30))
            t_mult = int(p.get("T_mult", 2))
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min
            )
        else:
            t_max = int(p.get("t_max", 50))
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=eta_min
            )
        self.criterion  = MultiTaskLoss(TARGETS)

        self.l3_eval_fn    = l3_eval_fn   # callable(model, val_loader, device) → float | None
        self.best_val_acc  = 0.0
        self.patience_ctr  = 0
        self.log_lines: list[str] = []

        (out_dir / "models").mkdir(parents=True, exist_ok=True)
        (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_inputs_and_labels(self, batch: dict) -> tuple[dict, dict]:
        """Separate model inputs from labels; move both to device."""
        labels = {t: batch["labels"][t].to(self.device) for t in TARGETS}
        inputs = {k: v.to(self.device) for k, v in batch.items()
                  if k not in ("labels",)}
        return inputs, labels

    def _forward(self, inputs: dict) -> dict[str, torch.Tensor]:
        """Call model forward with the right argument set."""
        keys = getattr(self.model, "INPUT_KEYS", list(inputs.keys()))
        return self.model(**{k: inputs[k] for k in keys})

    # ── Train one epoch ───────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        accum: dict[str, float] = defaultdict(float)
        n_batches = 0

        for batch in self.train_loader:
            inputs, labels = self._get_inputs_and_labels(batch)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self._forward(inputs)
            total_loss, head_losses = self.criterion(logits, labels)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.optimizer.zero_grad()
                continue
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            for t, l in head_losses.items():
                accum[t] += l.item()
            n_batches += 1

        return {t: v / max(n_batches, 1) for t, v in accum.items()}

    # ── Evaluate ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self) -> float:
        """
        Return the validation metric used for early stopping and best-checkpoint
        selection.

        If l3_eval_fn was provided at construction: returns L3 exact-name-match
        accuracy (preferred).  Otherwise falls back to mean per-head accuracy
        over _ES_TARGETS (class + chain-1 c/db/ox).
        """
        if self.l3_eval_fn is not None:
            return self.l3_eval_fn(self.model, self.val_loader, self.device)

        self.model.eval()
        correct = defaultdict(int)
        total   = defaultdict(int)

        for batch in self.val_loader:
            inputs, labels = self._get_inputs_and_labels(batch)
            logits = self._forward(inputs)

            for t in _ES_TARGETS:
                pred = logits[t].argmax(dim=1)
                true = labels[t]
                mask = true != -1
                correct[t] += (pred[mask] == true[mask]).sum().item()
                total[t]   += mask.sum().item()

        accs = [correct[t] / max(total[t], 1) for t in _ES_TARGETS]
        return float(sum(accs) / len(accs))

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_best(self) -> None:
        path = self.out_dir / "models" / "best.pt"
        torch.save(self.model.state_dict(), path)

    def load_best(self) -> None:
        path = self.out_dir / "models" / "best.pt"
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"  Loaded best checkpoint from {path}")

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*62}")
        print(f"Training: {self.model_name}   |  device: {self.device}")
        print(f"Max epochs: {self.max_epochs}  |  patience: {self.patience}")
        print(f"Start: {ts_start}")
        print(f"{'='*62}")

        header = (
            f"{'Epoch':>6}  {'Train loss':>11}  {'Val acc':>8}  "
            f"{'LR':>9}  {'Time':>6}  {'Note'}"
        )
        print(f"\n{header}")
        print("─" * len(header))
        self.log_lines = [
            f"Training log — {self.model_name} — {ts_start}",
            "=" * 70,
            header,
            "─" * 70,
        ]

        wall_start = time.time()

        for epoch in range(self.max_epochs):
            t0 = time.time()
            head_losses = self._train_epoch(epoch)
            val_acc     = self._evaluate()
            self.scheduler.step()

            # After epoch 0: set loss normalisation denominators
            if epoch == 0:
                self.criterion.set_loss0(head_losses)

            train_loss = sum(head_losses.values())
            lr_now     = self.optimizer.param_groups[0]["lr"]
            elapsed    = time.time() - t0
            note       = ""

            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_ctr = 0
                self._save_best()
                note = "✓ saved"
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.patience:
                    note = "STOP"

            row = (
                f"{epoch:>6}  {train_loss:>11.4f}  {val_acc:>8.4f}  "
                f"{lr_now:>9.2e}  {elapsed:>5.1f}s  {note}"
            )
            print(row)
            self.log_lines.append(row)

            if note == "STOP":
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best val acc: {self.best_val_acc:.4f}")
                break

        total_wall = time.time() - wall_start
        summary = (
            f"\nTotal training time: {total_wall/60:.1f} min  "
            f"| Best val acc: {self.best_val_acc:.4f}"
        )
        print(summary)
        self.log_lines.append(summary)

        # Save training log
        log_path = self.out_dir / "evaluation" / "training_log.txt"
        with open(log_path, "w") as fh:
            fh.write("\n".join(self.log_lines) + "\n")
        print(f"Training log → {log_path}")

        # Restore best weights
        self.load_best()
