"""
cnn.py — Multi-task 1D-CNN with residual blocks for lipid structure prediction.

Architecture:
  Input (2, 3100) — fragment + NL channels at 0.5 Da resolution
  4 × ResBlock1d  (2→64, k=3, s=2, 3100→1550) → (64→128, k=7) → (128→256, k=15)
                  → (256→512, k=15, s=2, 1550→775)
  GlobalAvgPool + GlobalMaxPool  →  (B, 1024)
  Adduct head:   Linear(1025 → K_adduct)  [pooled(1024) + precursor_mz(1)]
  Neck:          Linear(1033 → 512) → GN → ReLU  [pooled(1024) + pmz(1) + adduct_emb(8)]
  Other 13 heads: Linear(512 → K_target)

During training:  teacher-forced adduct_cond is used in the embedding.
During inference: predicted adduct replaces teacher-forced adduct_cond.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import TARGETS, CNN_N_BINS

_ADDUCT_EMB_DIM = 8
_POOLED_DIM     = 1024  # 512*2 (avg + max from 4-block encoder)
_NECK_IN        = _POOLED_DIM + 1 + _ADDUCT_EMB_DIM  # 1033
_NECK_OUT       = 512
_ADDUCT_HEAD_IN = _POOLED_DIM + 1                     # 1025


class ResBlock1d(nn.Module):
    """Post-activation residual block for 1-D sequences."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size,
                               stride=stride, padding=pad, bias=False)
        self.bn1   = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        # Skip connection: match stride and channels
        self.skip  = (nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False)
                      if (in_ch != out_ch or stride != 1) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class LipidCNN(nn.Module):
    """
    1D-CNN multi-task model with adduct conditioning.

    Parameters
    ----------
    n_classes : {target_name: number_of_classes}
    n_adducts : size of adduct embedding table (number of unique training adducts)
    """

    INPUT_KEYS = ["spectrum", "precursor_mz", "adduct_cond"]

    def __init__(
        self,
        n_classes: dict[str, int],
        n_adducts: int,
    ) -> None:
        super().__init__()

        # Residual blocks:
        #   2→64   (stride=2, 3100→1550)
        #   64→128 (1550)
        #   128→256 (1550)
        #   256→512 (stride=2, 1550→775)
        self.blocks = nn.Sequential(
            ResBlock1d(2,   64,  3,  stride=2),
            ResBlock1d(64,  128, 7),
            ResBlock1d(128, 256, 15),
            ResBlock1d(256, 512, 15, stride=2),
        )

        # Adduct embedding (for conditioning class / chain heads)
        self.adduct_emb = nn.Embedding(n_adducts, _ADDUCT_EMB_DIM)

        # Adduct head: pooled(512) + precursor_mz(1) → K_adduct
        self.adduct_head = nn.Linear(_ADDUCT_HEAD_IN, n_classes["adduct_enc"])

        # Neck: projects conditioned features to 512 for other heads
        self.neck = nn.Sequential(
            nn.Linear(_NECK_IN, _NECK_OUT, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=_NECK_OUT),
            nn.ReLU(inplace=True),
        )

        # Per-target heads for all non-adduct targets
        self.heads = nn.ModuleDict({
            t: nn.Linear(_NECK_OUT, n_classes[t])
            for t in TARGETS if t != "adduct_enc"
        })

    # ── Internal building blocks ──────────────────────────────────────────────

    def _encode(
        self, spectrum: torch.Tensor, precursor_mz: torch.Tensor
    ) -> torch.Tensor:
        """
        Run conv blocks + global pool.

        Returns
        -------
        base : (B, _ADDUCT_HEAD_IN = 1025)  [pooled_1024 || pmz_1]
        """
        feat = self.blocks(spectrum)                    # (B, 512, L)
        avg  = feat.mean(dim=-1)                        # (B, 512)
        mx   = feat.max(dim=-1).values                  # (B, 512)
        pooled = torch.cat([avg, mx], dim=1)            # (B, 1024)
        pmz    = precursor_mz.unsqueeze(1)              # (B, 1)
        return torch.cat([pooled, pmz], dim=1)          # (B, 1025)

    def _neck_forward(
        self, base: torch.Tensor, adduct_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Condition base features on adduct embedding and project through neck.

        Parameters
        ----------
        base       : (B, 1025)
        adduct_idx : (B,) long — remapped adduct index (0..K_adduct-1)

        Returns
        -------
        h : (B, 512)
        """
        emb  = self.adduct_emb(adduct_idx)             # (B, 8)
        cond = torch.cat([base, emb], dim=1)           # (B, 1033)
        return self.neck(cond)                          # (B, 512)

    # ── Forward passes ────────────────────────────────────────────────────────

    def forward(
        self,
        spectrum: torch.Tensor,
        precursor_mz: torch.Tensor,
        adduct_cond: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Training-time forward: uses provided adduct_cond (teacher forcing).

        Parameters
        ----------
        spectrum     : (B, 2, 3100) float32
        precursor_mz : (B,) float32
        adduct_cond  : (B,) long   — remapped true adduct index

        Returns
        -------
        {target: (B, K) logits}
        """
        base = self._encode(spectrum, precursor_mz)     # (B, 1025)
        adduct_logits = self.adduct_head(base)          # (B, K_adduct)
        h = self._neck_forward(base, adduct_cond)       # (B, 512)
        result: dict[str, torch.Tensor] = {"adduct_enc": adduct_logits}
        for t, head in self.heads.items():
            result[t] = head(h)
        return result

    def forward_inference(
        self,
        spectrum: torch.Tensor,
        precursor_mz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Inference-time forward: predicts adduct first, then conditions on it.

        Returns
        -------
        {target: (B, K) logits}
        """
        base          = self._encode(spectrum, precursor_mz)
        adduct_logits = self.adduct_head(base)
        pred_adduct   = adduct_logits.argmax(dim=1)     # (B,) remapped
        h             = self._neck_forward(base, pred_adduct)
        result: dict[str, torch.Tensor] = {"adduct_enc": adduct_logits}
        for t, head in self.heads.items():
            result[t] = head(h)
        return result
