"""
cnn_dataset.py — PyTorch Dataset for the 1D-CNN model.

Input: raw MS2 spectrum → clean → 0.5 Da bins → (2, 3100), pre-computed at init.
Augmentation (bin dropout, intensity scale, Gaussian noise) applied per-item at getitem.
Additional model inputs: precursor_mz (float), adduct_cond (int, teacher-forced).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    TARGETS, CHAIN_MIN, encode_label,
    parse_spectrum, clean_spectrum,
    spectrum_to_cnn_input,
)


class CNNDataset(Dataset):
    """
    Parameters
    ----------
    df_feat      : features DataFrame (for encoded labels + adduct_cond + precursor_mz_norm)
    df_raw       : cleaned parquet DataFrame (for MS2 strings + precursor_mz)
    indices      : row indices for this split
    label_maps   : {target: sorted training labels}
    row_num_chain: per-row num_chain array
    augment      : if True, apply feature-space augmentation in __getitem__ (training set only)
    pmz_stats    : (mean, std) for recovering original precursor_mz from normalised value
    """

    def __init__(
        self,
        df_feat: pd.DataFrame,
        df_raw: pd.DataFrame,
        indices: np.ndarray,
        label_maps: dict[str, np.ndarray],
        row_num_chain: np.ndarray,
        augment: bool = False,
        pmz_stats: tuple[float, float] | None = None,
    ) -> None:
        self.indices      = indices.astype(np.int32)
        self.label_maps   = label_maps
        self.row_num_chain = row_num_chain
        self.augment      = augment

        # Recover original precursor_mz
        if pmz_stats is not None:
            pmz_mean, pmz_std = pmz_stats
            self.precursor_mz = (
                df_feat["precursor_mz_norm"].values * pmz_std + pmz_mean
            ).astype(np.float32)
        else:
            self.precursor_mz = df_raw["precursor_mz"].values.astype(np.float32)

        # Pre-compute all CNN inputs: clean + bin → (N, 2, 3100) float32
        n_total = len(df_raw)
        print(f"  [CNNDataset] Pre-computing {n_total:,} CNN inputs …")
        ms2_strings = df_raw["MS2"].values
        self.data = np.zeros((n_total, 2, 3100), dtype=np.float32)
        for i, s in enumerate(ms2_strings):
            mz, intensity = parse_spectrum(s)
            pmz_i = float(self.precursor_mz[i])
            mz, intensity = clean_spectrum(mz, intensity, pmz_i)
            self.data[i] = spectrum_to_cnn_input(mz, intensity, pmz_i)
        print(f"  [CNNDataset] Pre-computed shape: {self.data.shape}.")

        # Pre-encode labels
        print("  [CNNDataset] Encoding labels …")
        self.labels: dict[str, np.ndarray] = {}
        for t in TARGETS:
            raw = df_feat[t].values.astype(np.int32)
            enc = np.full(len(df_feat), -1, dtype=np.int32)
            min_chain = CHAIN_MIN.get(t, 0)
            valid = (row_num_chain >= min_chain) if min_chain > 0 else np.ones(len(df_feat), dtype=bool)
            lm = label_maps[t]
            for i in np.where(valid)[0]:
                enc[i] = encode_label(int(raw[i]), lm)
            self.labels[t] = enc

        # Pre-encode adduct for conditioning
        raw_adduct = df_feat["adduct_enc"].values.astype(np.int32)
        lm_adduct  = label_maps["adduct_enc"]
        self.adduct_cond = np.array(
            [encode_label(int(v), lm_adduct) for v in raw_adduct], dtype=np.int32
        )

        print(f"  [CNNDataset] {len(indices):,} samples ready.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        row = int(self.indices[idx])
        x   = self.data[row].copy()        # (2, 3100) float32
        pmz = float(self.precursor_mz[row])

        if self.augment:
            rng   = np.random.default_rng()
            scale = rng.uniform(0.85, 1.15)   # shared across both channels
            x     = x * scale
            for ch in range(2):
                # Bin dropout: zero ~10% of non-zero bins
                nz = np.where(x[ch] != 0.0)[0]
                if len(nz) > 0:
                    n_drop = max(1, int(0.10 * len(nz)))
                    x[ch, rng.choice(nz, size=n_drop, replace=False)] = 0.0
                # Gaussian noise on remaining non-zero bins, clipped to 0
                nz2 = np.where(x[ch] != 0.0)[0]
                if len(nz2) > 0:
                    x[ch, nz2] = np.clip(
                        x[ch, nz2] + rng.normal(0.0, 0.01, len(nz2)).astype(np.float32),
                        0.0, None,
                    )

        lbl = {t: torch.tensor(self.labels[t][row], dtype=torch.long)
               for t in TARGETS}

        return {
            "spectrum":     torch.from_numpy(x),
            "precursor_mz": torch.tensor(pmz, dtype=torch.float32),
            "adduct_cond":  torch.tensor(int(self.adduct_cond[row]), dtype=torch.long),
            "labels":       lbl,
            "orig_idx":     torch.tensor(row, dtype=torch.long),
        }
