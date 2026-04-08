"""
utils.py — Shared constants, path anchors, and spectrum utilities for phase2_dl.

Reuses parse_spectrum / clean_spectrum from phase1_ml/src/utils.py and adds
0.5 Da binning (for CNN) and tokenisation (for Transformer).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# ── Directory anchors ─────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).resolve().parent       # phase2_dl/src/
PHASE2_ROOT = SRC_DIR.parent                        # phase2_dl/
REPO_ROOT   = PHASE2_ROOT.parent                    # lipid_plus_v2/
PHASE1_ROOT = REPO_ROOT / "phase1_ml"
DATASET_ROOT = REPO_ROOT / "dataset"

SHARED_DIR   = PHASE1_ROOT / "outputs" / "shared"
FEAT_PARQUET = PHASE1_ROOT / "data" / "processed" / "lipid_ms2_features.parquet"
SPLITS_DIR   = PHASE1_ROOT / "data" / "splits"
RAW_PARQUET  = DATASET_ROOT / "processed_ms2_data" / "lipid_ms2_cleaned.parquet"
OUTPUTS_DIR  = PHASE2_ROOT / "outputs"

# ── Import phase1_ml utilities via importlib (avoids circular import) ─────────
import importlib.util as _ilu

_p1_utils_spec = _ilu.spec_from_file_location(
    "_phase1_ml_utils",
    str(PHASE1_ROOT / "src" / "utils.py"),
)
_p1_utils = _ilu.module_from_spec(_p1_utils_spec)
_p1_utils_spec.loader.exec_module(_p1_utils)  # type: ignore[union-attr]

parse_spectrum         = _p1_utils.parse_spectrum
clean_spectrum         = _p1_utils.clean_spectrum
load_class_to_numchain = _p1_utils.load_class_to_numchain
load_backbone_masses   = _p1_utils.load_backbone_masses
reconstruct_name       = _p1_utils.reconstruct_name
ADDUCT_TABLE           = _p1_utils.ADDUCT_TABLE
SUM_COMP_PPM_TOL       = _p1_utils.SUM_COMP_PPM_TOL
adduct_to_neutral      = _p1_utils.adduct_to_neutral
find_sum_comp_candidates = _p1_utils.find_sum_comp_candidates

# Also register in sys.modules so downstream imports (e.g. pipeline/inference.py)
# that do `from utils import ...` after inserting phase1_ml/src find it correctly.
_PHASE1_SRC = str(PHASE1_ROOT / "src")
if _PHASE1_SRC not in sys.path:
    sys.path.insert(0, _PHASE1_SRC)

# ── Target definitions ────────────────────────────────────────────────────────
TARGETS: list[str] = [
    "adduct_enc",
    "class_enc",
    "num_c_1", "num_db_1", "num_ox_1",
    "num_c_2", "num_db_2", "num_ox_2",
    "num_c_3", "num_db_3", "num_ox_3",
    "num_c_4", "num_db_4", "num_ox_4",
]

# Minimum num_chain required for each target (chain targets only)
CHAIN_MIN: dict[str, int] = {
    "num_c_2": 2, "num_db_2": 2, "num_ox_2": 2,
    "num_c_3": 3, "num_db_3": 3, "num_ox_3": 3,
    "num_c_4": 4, "num_db_4": 4, "num_ox_4": 4,
}

# Base feature dimensionality (from phase1_ml)
BASE_FEAT_DIM = 3102    # 1550 F + 1550 NL + precursor_mz_norm + ion_mode_enc

# ── CNN binning parameters ─────────────────────────────────────────────────────
CNN_MZ_MIN    = 50.0
CNN_MZ_MAX    = 1600.0
CNN_BIN_WIDTH = 0.5
CNN_N_BINS    = int((CNN_MZ_MAX - CNN_MZ_MIN) / CNN_BIN_WIDTH)  # 3100

# ── Transformer parameters ────────────────────────────────────────────────────
TRANS_TOP_K   = 100    # number of peaks kept
TRANS_D_TOKEN = 3      # (norm_mz, norm_NL, sqrt_intensity)


# ── 0.5 Da binning for CNN ────────────────────────────────────────────────────
def bin_spectrum_cnn(
    mz: np.ndarray,
    intensity: np.ndarray,
    mz_min: float = CNN_MZ_MIN,
    mz_max: float = CNN_MZ_MAX,
    bin_width: float = CNN_BIN_WIDTH,
) -> np.ndarray:
    """Bin a cleaned spectrum at 0.5 Da resolution → float32 vector (3100,)."""
    n_bins = int((mz_max - mz_min) / bin_width)
    out = np.zeros(n_bins, dtype=np.float32)
    if mz.size == 0:
        return out
    idx = ((mz - mz_min) / bin_width).astype(np.int32)
    mask = (idx >= 0) & (idx < n_bins)
    np.maximum.at(out, idx[mask], intensity[mask])
    return out


def bin_neutral_loss_cnn(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    mz_min: float = CNN_MZ_MIN,
    mz_max: float = CNN_MZ_MAX,
    bin_width: float = CNN_BIN_WIDTH,
) -> np.ndarray:
    """Compute neutral loss and bin at 0.5 Da → float32 vector (3100,)."""
    if mz.size == 0:
        return np.zeros(int((mz_max - mz_min) / bin_width), dtype=np.float32)
    nl = (precursor_mz - mz).astype(np.float32)
    return bin_spectrum_cnn(nl, intensity, mz_min=mz_min, mz_max=mz_max,
                            bin_width=bin_width)


def spectrum_to_cnn_input(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
) -> np.ndarray:
    """Stack fragment + NL bins into (2, CNN_N_BINS) float32 array."""
    frag = bin_spectrum_cnn(mz, intensity)
    nl   = bin_neutral_loss_cnn(mz, intensity, precursor_mz)
    return np.stack([frag, nl], axis=0)   # (2, 3100)


# ── Transformer tokenisation ──────────────────────────────────────────────────
def spectrum_to_tokens(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    top_k: int = TRANS_TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (mz, sqrt_intensity) arrays to (top_k, 3) token matrix and mask.

    Token channels: (norm_mz, norm_NL, sqrt_intensity).
    Padding rows are all-zero; mask[i] = True means padding (ignored by attention).

    Returns
    -------
    tokens : float32 (top_k, 3)
    mask   : bool    (top_k,)   True = padding
    """
    tokens = np.zeros((top_k, TRANS_D_TOKEN), dtype=np.float32)
    mask   = np.ones(top_k, dtype=bool)   # all padding by default

    if mz.size == 0:
        return tokens, mask

    # Keep at most top_k peaks (they are already trimmed by clean_spectrum
    # but the spectrum may have been re-parsed; take top-k by intensity)
    if len(mz) > top_k:
        top_idx = np.argpartition(intensity, -top_k)[-top_k:]
        mz       = mz[top_idx]
        intensity = intensity[top_idx]

    n = len(mz)
    mz_range = CNN_MZ_MAX - CNN_MZ_MIN    # 1550

    norm_mz = np.clip((mz - CNN_MZ_MIN) / mz_range, 0.0, 1.0)
    nl      = precursor_mz - mz
    norm_nl = np.clip((nl  - CNN_MZ_MIN) / mz_range, 0.0, 1.0)

    tokens[:n, 0] = norm_mz.astype(np.float32)
    tokens[:n, 1] = norm_nl.astype(np.float32)
    tokens[:n, 2] = intensity.astype(np.float32)   # already sqrt from clean_spectrum
    mask[:n]      = False  # real peaks — not padding

    return tokens, mask


# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply four data-augmentation transforms to a cleaned (mz, sqrt_intensity) pair.

    1. Gaussian intensity noise    N(0, 0.02)
    2. m/z jitter                  N(0, 0.005 Da)
    3. Peak dropout                zero 15 % of low-intensity peaks
    4. Intensity scale             U(0.85, 1.15)

    Returns new (mz, intensity) arrays (copies; original unchanged).
    """
    if mz.size == 0:
        return mz.copy(), intensity.copy()

    mz_out  = mz.copy().astype(np.float32)
    int_out = intensity.copy().astype(np.float32)

    # 1. Gaussian intensity noise
    int_out = np.clip(int_out + rng.normal(0, 0.02, int_out.shape).astype(np.float32),
                      0.0, None)

    # 2. m/z jitter
    mz_out  = mz_out + rng.normal(0, 0.005, mz_out.shape).astype(np.float32)

    # 3. Peak dropout — zero 15 % of low-intensity peaks
    if len(int_out) > 1:
        n_drop  = max(1, int(0.15 * len(int_out)))
        # lowest-intensity peaks are candidates
        low_idx = np.argpartition(int_out, n_drop)[:n_drop]
        drop    = rng.choice(low_idx, size=n_drop, replace=False)
        int_out[drop] = 0.0

    # 4. Intensity scale
    scale   = rng.uniform(0.85, 1.15)
    int_out = np.clip(int_out * scale, 0.0, None)

    return mz_out, int_out


# ── Label-map utilities ───────────────────────────────────────────────────────
def build_label_maps(
    df_labels,          # DataFrame with TARGETS columns
    train_idx: np.ndarray,
    row_num_chain: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build sorted unique training-set labels for each target.
    Matches the logic in phase1_ml build_class_maps().
    class_map[i] = original_label_for_encoded_index_i
    """
    import pandas as pd

    train_set = set(train_idx.tolist())
    all_mask  = np.ones(len(df_labels), dtype=bool)

    masks = {
        "adduct_enc": all_mask,
        "class_enc":  all_mask,
        "num_c_1": all_mask, "num_db_1": all_mask, "num_ox_1": all_mask,
        "num_c_2": row_num_chain >= 2, "num_db_2": row_num_chain >= 2,
        "num_ox_2": row_num_chain >= 2,
        "num_c_3": row_num_chain >= 3, "num_db_3": row_num_chain >= 3,
        "num_ox_3": row_num_chain >= 3,
        "num_c_4": row_num_chain >= 4, "num_db_4": row_num_chain >= 4,
        "num_ox_4": row_num_chain >= 4,
    }

    label_maps: dict[str, np.ndarray] = {}
    for t in TARGETS:
        mask    = masks[t]
        valid   = np.array([i for i in np.where(mask)[0] if i in train_set])
        y       = df_labels[t].values.astype(np.int32)
        label_maps[t] = np.sort(np.unique(y[valid]))

    return label_maps


# ── Data-starved classes ──────────────────────────────────────────────────────
# Classes with insufficient training data for reliable chain-level prediction.
# Multi-chain samples from these classes are excluded from L1/L2/L3 evaluation.
DATA_STARVED_CLASSES: frozenset[str] = frozenset({
    "CAR", "HBMP", "ADGGA-O", "DGDG-O", "DGGA",
    "NAGlySer", "NAOrn", "PE-Cer", "PG-P", "ST", "VAE",
})

# ── Mass constants for rule-based chain inference ─────────────────────────────
_H_RULE   = 1.00782503207
_C_RULE   = 12.0
_O_RULE   = 15.99491461957
_H2O_RULE = 2 * _H_RULE + _O_RULE
_CH2_RULE = _C_RULE + 2 * _H_RULE    # 14.01565006


def encode_label(value: int, label_map: np.ndarray) -> int:
    """Map original label to 0..K-1; returns -1 for unseen labels
    so CrossEntropyLoss(ignore_index=-1) ignores them rather than
    silently scoring them as class 0.
    """
    idx = np.searchsorted(label_map, value)
    if idx < len(label_map) and label_map[idx] == value:
        return int(idx)
    return -1


def apply_chain_rules(
    pred_chains: dict[str, int],
    adduct_str: str,
    class_str: str,
    precursor_mz: float,
    cgm_table: dict[str, float],
    num_chain: int,
    tol_da: float = 0.01,
) -> dict[str, int]:
    """
    Apply algebraic chain inference at prediction time for 1-chain lipids.

    For 1-chain lipids (num_chain == 1):
      - Enumerate all (nc, ndb, nox) satisfying:
          nc*(C+2H) - 2*ndb*H + (2+nox)*O - H2O == total_chain_mass  ± tol_da
      - Exactly 1 candidate → override model prediction with rule values.
      - Exactly 2 candidates → pick the one closer to model prediction.
      - Otherwise → return model prediction unchanged.

    For multi-chain lipids: returns pred_chains unchanged.

    Parameters
    ----------
    pred_chains  : model chain predictions, keys "nc1","ndb1","nox1",...
    adduct_str   : predicted adduct string, e.g. "[M+H]+"
    class_str    : predicted class string, e.g. "FA"
    precursor_mz : observed precursor m/z (float)
    cgm_table    : {class: CGM float} from class_backbone_masses.json
    num_chain    : number of acyl chains for this class
    tol_da       : mass tolerance in Da (default 0.01)

    Returns
    -------
    Updated pred_chains dict (copy).
    """
    out = dict(pred_chains)

    if num_chain != 1:
        return out

    cgm = cgm_table.get(class_str)
    if cgm is None:
        return out

    exact_mass = adduct_to_neutral(precursor_mz, adduct_str)
    if exact_mass is None:
        return out

    total = exact_mass - cgm
    tol_nc = tol_da / _CH2_RULE

    candidates: list[tuple[int, int, int]] = []
    for nox in range(5):       # 0..4
        for ndb in range(11):  # 0..10
            # Solve nc from: total = nc*CH2 - 2*ndb*H + (2+nox)*O - H2O
            nc_f = (total + 2 * ndb * _H_RULE
                    - (2 + nox) * _O_RULE
                    + _H2O_RULE) / _CH2_RULE
            nc_i = int(round(nc_f))
            if nc_i <= 0 or nc_i > 50:
                continue
            if nc_i < ndb:
                continue
            if abs(nc_f - nc_i) > tol_nc:
                continue
            candidates.append((nc_i, ndb, nox))

    if len(candidates) == 1:
        nc1, ndb1, nox1 = candidates[0]
        out["nc1"] = nc1;  out["ndb1"] = ndb1;  out["nox1"] = nox1
    elif len(candidates) == 2:
        p_nc1  = pred_chains.get("nc1",  0)
        p_ndb1 = pred_chains.get("ndb1", 0)
        p_nox1 = pred_chains.get("nox1", 0)
        d0 = (abs(candidates[0][0] - p_nc1) +
              abs(candidates[0][1] - p_ndb1) +
              abs(candidates[0][2] - p_nox1))
        d1 = (abs(candidates[1][0] - p_nc1) +
              abs(candidates[1][1] - p_ndb1) +
              abs(candidates[1][2] - p_nox1))
        nc1, ndb1, nox1 = candidates[0] if d0 <= d1 else candidates[1]
        out["nc1"] = nc1;  out["ndb1"] = ndb1;  out["nox1"] = nox1

    return out
