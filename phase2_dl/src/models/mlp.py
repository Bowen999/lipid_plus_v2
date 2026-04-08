"""
mlp.py — Multi-task MLP for lipid structure prediction.

Architecture:
  Linear(3102→512) → BN → ReLU → Dropout(0.3)
  → Linear(512→256) → BN → ReLU
  → 14 per-target classification heads
"""
from __future__ import annotations

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import TARGETS, BASE_FEAT_DIM


class LipidMLP(nn.Module):
    """
    Shared MLP encoder + 14 classification heads.

    All heads receive the same 256-dim representation; no cascade at training
    time (teacher forcing not needed — there is no conditioning in this model).
    """

    INPUT_KEYS = ["x"]   # keys the trainer extracts from each batch

    def __init__(self, n_classes: dict[str, int]) -> None:
        """
        Parameters
        ----------
        n_classes : {target_name: number_of_classes}
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(BASE_FEAT_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.heads = nn.ModuleDict({
            t: nn.Linear(256, n_classes[t]) for t in TARGETS
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3102) float32

        Returns
        -------
        {target: (B, K) logits}
        """
        feat = self.encoder(x)          # (B, 256)
        return {t: head(feat) for t, head in self.heads.items()}
