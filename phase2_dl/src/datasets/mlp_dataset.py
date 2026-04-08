"""
mlp_dataset.py — PyTorch Dataset for the MLP model.

Input: 3,102-dim pre-computed binned feature vector (already in features parquet).
No augmentation, no raw spectrum needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import TARGETS, CHAIN_MIN, BASE_FEAT_DIM, encode_label


class MLPDataset(Dataset):
    """
    Wraps the pre-computed 3102-dim feature vectors with remapped labels.

    Parameters
    ----------
    df_feat      : DataFrame with F_*, NL_*, precursor_mz_norm, ion_mode_enc,
                   adduct_enc, class_enc, num_c_1..4, num_db_1..4, num_ox_1..4
    indices      : row indices for this split
    label_maps   : {target: sorted unique training labels} from build_label_maps()
    row_num_chain: per-row num_chain array
    base_feat_cols: ordered list of the 3102 feature column names
    augment      : if True, apply feature-space augmentation in __getitem__
                   (use True for training set only)
    """

    def __init__(
        self,
        df_feat: pd.DataFrame,
        indices: np.ndarray,
        label_maps: dict[str, np.ndarray],
        row_num_chain: np.ndarray,
        base_feat_cols: list[str],
        augment: bool = False,
    ) -> None:
        self.indices      = indices.astype(np.int32)
        self.label_maps   = label_maps
        self.row_num_chain = row_num_chain
        self.augment      = augment

        # Pre-extract feature matrix for fast __getitem__
        print("  [MLPDataset] Extracting feature matrix …")
        self.X = df_feat[base_feat_cols].values.astype(np.float32)  # (N, 3102)
        assert self.X.shape[1] == BASE_FEAT_DIM, (
            f"Expected {BASE_FEAT_DIM} features, got {self.X.shape[1]}"
        )

        # Pre-extract and encode labels for all rows
        print("  [MLPDataset] Encoding labels …")
        self.labels: dict[str, np.ndarray] = {}
        for t in TARGETS:
            raw = df_feat[t].values.astype(np.int32)
            enc = np.full(len(df_feat), -1, dtype=np.int32)  # default = ignored
            min_chain = CHAIN_MIN.get(t, 0)
            valid_rows = (row_num_chain >= min_chain) if min_chain > 0 else np.ones(len(df_feat), dtype=bool)
            lm = label_maps[t]
            for i in np.where(valid_rows)[0]:
                enc[i] = encode_label(int(raw[i]), lm)
            self.labels[t] = enc

        print(f"  [MLPDataset] {len(indices):,} samples ready.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        row = int(self.indices[idx])
        x   = self.X[row].copy()   # (3102,) float32 copy

        if self.augment:
            rng = np.random.default_rng()

            # 1. Bin dropout: zero ~10% of currently non-zero bins
            nz = np.where(x != 0.0)[0]
            if len(nz) > 0:
                n_drop = max(1, int(0.10 * len(nz)))
                drop   = rng.choice(nz, size=n_drop, replace=False)
                x[drop] = 0.0

            # 2. Intensity scale: multiply all bins by Uniform(0.85, 1.15)
            x = x * rng.uniform(0.85, 1.15)

            # 3. Gaussian noise on non-zero bins: N(0, 0.01), clip to 0
            nz2 = np.where(x != 0.0)[0]
            if len(nz2) > 0:
                x[nz2] = np.clip(
                    x[nz2] + rng.normal(0.0, 0.01, len(nz2)).astype(np.float32),
                    0.0, None,
                )

        lbl = {t: torch.tensor(self.labels[t][row], dtype=torch.long)
               for t in TARGETS}
        return {"x": torch.from_numpy(x.astype(np.float32)),
                "labels": lbl,
                "orig_idx": torch.tensor(row, dtype=torch.long)}
