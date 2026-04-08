"""
transformer_dataset.py — PyTorch Dataset for the Transformer model.

Input: raw MS2 → clean → top-50 peaks as tokens (mz, NL, intensity), pre-computed at init.
Augmentation (token dropout, intensity scale, mz/NL noise) applied per-item at getitem.
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
    TARGETS, CHAIN_MIN, TRANS_TOP_K, encode_label,
    parse_spectrum, clean_spectrum, spectrum_to_tokens,
)


class TransformerDataset(Dataset):
    """
    Parameters
    ----------
    df_feat      : features DataFrame (encoded labels + precursor_mz_norm)
    df_raw       : cleaned parquet DataFrame (MS2 strings + precursor_mz)
    indices      : row indices for this split
    label_maps   : {target: sorted training labels}
    row_num_chain: per-row num_chain array
    augment      : apply feature-space augmentation in __getitem__ (training set only)
    pmz_stats    : (mean, std) to recover original precursor_mz
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
        self.indices       = indices.astype(np.int32)
        self.label_maps    = label_maps
        self.row_num_chain = row_num_chain
        self.augment       = augment

        # Recover original precursor_mz
        if pmz_stats is not None:
            pmz_mean, pmz_std = pmz_stats
            self.precursor_mz = (
                df_feat["precursor_mz_norm"].values * pmz_std + pmz_mean
            ).astype(np.float32)
        else:
            self.precursor_mz = df_raw["precursor_mz"].values.astype(np.float32)

        # Pre-compute all token arrays: clean → tokenise → (N, TRANS_TOP_K, 3) + mask
        n_total = len(df_raw)
        print(f"  [TransformerDataset] Pre-computing {n_total:,} token arrays …")
        ms2_strings  = df_raw["MS2"].values
        self.tokens  = np.zeros((n_total, TRANS_TOP_K, 3), dtype=np.float32)
        self.masks   = np.zeros((n_total, TRANS_TOP_K),    dtype=bool)
        for i, s in enumerate(ms2_strings):
            mz, intensity = parse_spectrum(s)
            pmz_i = float(self.precursor_mz[i])
            mz, intensity = clean_spectrum(mz, intensity, pmz_i)
            tok, mask = spectrum_to_tokens(mz, intensity, pmz_i)
            self.tokens[i] = tok
            self.masks[i]  = mask
        print(f"  [TransformerDataset] Pre-computed tokens shape: {self.tokens.shape}.")

        # Pre-encode labels
        print("  [TransformerDataset] Encoding labels …")
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

        print(f"  [TransformerDataset] {len(indices):,} samples ready.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        row    = int(self.indices[idx])
        tokens = self.tokens[row].copy()   # (TRANS_TOP_K, 3) float32
        mask   = self.masks[row].copy()    # (TRANS_TOP_K,) bool — True = padded

        if self.augment:
            rng       = np.random.default_rng()
            valid_idx = np.where(~mask)[0]

            if len(valid_idx) > 0:
                # 1. Token row dropout: zero out ~4 random non-padded rows
                n_drop = min(4, len(valid_idx))
                drop   = rng.choice(valid_idx, size=n_drop, replace=False)
                tokens[drop] = 0.0
                mask[drop]   = True   # mark as padding so model skips them

                # Re-derive valid positions after dropout
                valid_idx2 = np.where(~mask)[0]
                if len(valid_idx2) > 0:
                    # 2. Intensity channel (col 2) scale: U(0.85, 1.15)
                    tokens[valid_idx2, 2] *= rng.uniform(0.85, 1.15)
                    # 3. mz (col 0) and NL (col 1) noise: N(0, 0.002)
                    n_v = len(valid_idx2)
                    tokens[valid_idx2, 0] += rng.normal(0.0, 0.002, n_v).astype(np.float32)
                    tokens[valid_idx2, 1] += rng.normal(0.0, 0.002, n_v).astype(np.float32)

        # Prepend False for CLS position (model computes its own mask, but return for reference)
        full_mask = np.concatenate([[False], mask])   # (TRANS_TOP_K + 1,)

        lbl = {t: torch.tensor(self.labels[t][row], dtype=torch.long)
               for t in TARGETS}

        return {
            "tokens":        torch.from_numpy(tokens),
            "padding_mask":  torch.from_numpy(full_mask),
            "precursor_mz":  torch.tensor(float(self.precursor_mz[row]), dtype=torch.float32),
            "labels":        lbl,
            "orig_idx":      torch.tensor(row, dtype=torch.long),
        }
