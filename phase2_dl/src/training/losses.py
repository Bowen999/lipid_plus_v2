"""
losses.py — Multi-task weighted cross-entropy loss.

Weight scheme: each head's loss is normalised by its epoch-0 value so that
all heads contribute roughly equally regardless of the number of classes.
In epoch 0 the loss is computed unweighted; after that each head loss is
divided by its mean epoch-0 value.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import TARGETS


class MultiTaskLoss(nn.Module):
    """
    Weighted cross-entropy across 14 prediction targets.

    Usage
    -----
    loss = MultiTaskLoss(TARGETS)

    # epoch 0:
    total, per_head = loss(logits, labels)   # unweighted
    loss.set_loss0(epoch0_mean_per_head_losses)

    # epoch 1+:
    total, per_head = loss(logits, labels)   # weighted
    """

    def __init__(self, targets: list[str] = TARGETS) -> None:
        super().__init__()
        self.targets = targets
        self._loss0: dict[str, float] | None = None

    def set_loss0(self, epoch0_losses: dict[str, float]) -> None:
        """Called once after epoch 0 to set the normalisation denominators.
        Heads with loss≈0 (all labels ignored in epoch 0) get denominator=1.0
        so their normalised contribution stays at face value later.
        """
        self._loss0 = {t: (v + 1e-4 if v > 1e-6 else 1.0)
                       for t, v in epoch0_losses.items()}

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        logits : {target: (B, K) float}
        labels : {target: (B,) long}  — -100 = ignore

        Returns
        -------
        total_loss : scalar tensor
        head_losses: {target: scalar tensor}  (detached for logging)
        """
        head_losses: dict[str, torch.Tensor] = {}
        for t in self.targets:
            raw = F.cross_entropy(
                logits[t], labels[t], ignore_index=-1, reduction="mean"
            )
            # When all labels are ignored, cross_entropy returns nan; treat as 0
            head_losses[t] = torch.where(torch.isnan(raw), torch.zeros_like(raw), raw)

        if self._loss0 is None:
            # Epoch 0: unweighted sum
            total = sum(head_losses.values())
        else:
            # Subsequent epochs: normalise by epoch-0 values
            total = sum(head_losses[t] / self._loss0[t] for t in self.targets)

        return total, {t: v.detach() for t, v in head_losses.items()}
