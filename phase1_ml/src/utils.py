"""
utils.py — Shared constants, helpers, and spectrum processing functions
for the Phase 1 lipid structure prediction pipeline (phase1_ml).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Directory anchors ─────────────────────────────────────────────────────────
# utils.py lives in phase1_ml/src/  →  parent is phase1_ml/
SRC_DIR      = Path(__file__).resolve().parent   # phase1_ml/src/
PHASE1_ROOT  = SRC_DIR.parent                    # phase1_ml/
OUTPUTS_DIR  = PHASE1_ROOT / "outputs"
SHARED_DIR   = OUTPUTS_DIR / "shared"            # shared encoders + metadata JSON
DATA_DIR     = PHASE1_ROOT / "data" / "processed"  # lipid_ms2_features.parquet
SPLITS_DIR   = PHASE1_ROOT / "data" / "splits"     # split_{train,val,test}.npy

# Legacy aliases kept for backward compat with src/evaluation/metrics.py
MODELS_DIR     = SHARED_DIR
EVALUATION_DIR = OUTPUTS_DIR / "xgboost" / "evaluation"

# ── Constants ─────────────────────────────────────────────────────────────────
MZ_MIN        = 50.0
MZ_MAX        = 1600.0
BIN_WIDTH     = 1.0                               # 1 Da bins → 1550 bins per channel
N_BINS        = int((MZ_MAX - MZ_MIN) / BIN_WIDTH)  # 1550
TOP_K_PEAKS   = 50       # keep top-K peaks by intensity per spectrum
NOISE_FLOOR   = 0.01     # drop peaks < 1 % of base peak


# ── Spectrum parser ───────────────────────────────────────────────────────────
def parse_spectrum(ms2_str: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse "[[mz1, int1], [mz2, int2], ...]" into two float32 arrays.

    Returns (mz_array, intensity_array).
    Returns (empty, empty) on any parse failure or if no peaks remain.
    """
    if not isinstance(ms2_str, str) or not ms2_str.strip():
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    try:
        peaks = ast.literal_eval(ms2_str)
        if not peaks:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        arr = np.array(peaks, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        mz  = arr[:, 0]
        intensity = arr[:, 1]
        # drop non-positive intensities
        mask = intensity > 0
        return mz[mask], intensity[mask]
    except Exception:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)


# ── Spectrum cleaner ──────────────────────────────────────────────────────────
def clean_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean a raw spectrum in five steps:

    1. Remove peaks with mz >= precursor_mz + 2.
    2. Normalise intensities to base-peak = 1.0.
    3. Apply NOISE_FLOOR: drop peaks where normalised intensity < NOISE_FLOOR.
    4. Keep TOP_K_PEAKS by intensity.
    5. sqrt-transform intensities.

    Returns cleaned (mz, sqrt_intensity).
    May return empty arrays if no peaks survive.
    """
    if mz.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Step 1 — remove precursor region
    mask = mz < (precursor_mz + 2.0)
    mz        = mz[mask]
    intensity = intensity[mask]
    if mz.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Step 2 — normalise to base peak
    base = intensity.max()
    if base <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    intensity = intensity / base

    # Step 3 — noise floor
    mask = intensity >= NOISE_FLOOR
    mz        = mz[mask]
    intensity = intensity[mask]
    if mz.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Step 4 — keep top-K by intensity
    if mz.size > TOP_K_PEAKS:
        top_idx   = np.argpartition(intensity, -TOP_K_PEAKS)[-TOP_K_PEAKS:]
        mz        = mz[top_idx]
        intensity = intensity[top_idx]

    # Step 5 — sqrt transform
    intensity = np.sqrt(intensity).astype(np.float32)
    return mz.astype(np.float32), intensity


# ── Binning ───────────────────────────────────────────────────────────────────
def bin_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    mz_min: float = MZ_MIN,
    mz_max: float = MZ_MAX,
    bin_width: float = BIN_WIDTH,
) -> np.ndarray:
    """
    Return a float32 vector of length N_BINS.

    Each bin holds the maximum sqrt-intensity of all peaks that fall in
    [bin_left, bin_left + bin_width).  Peaks outside [mz_min, mz_max)
    are ignored.
    """
    n_bins = int((mz_max - mz_min) / bin_width)
    out = np.zeros(n_bins, dtype=np.float32)
    if mz.size == 0:
        return out
    idx = ((mz - mz_min) / bin_width).astype(np.int32)
    mask = (idx >= 0) & (idx < n_bins)
    mz        = mz[mask]
    intensity = intensity[mask]
    idx       = idx[mask]
    np.maximum.at(out, idx, intensity)
    return out


# ── Neutral-loss binning ──────────────────────────────────────────────────────
def bin_neutral_loss(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    mz_min: float = MZ_MIN,
    mz_max: float = MZ_MAX,
    bin_width: float = BIN_WIDTH,
) -> np.ndarray:
    """
    Compute NL = precursor_mz - mz for each fragment peak, then bin
    the NL values using the same [mz_min, mz_max) grid.

    Returns a float32 vector of length N_BINS.
    """
    if mz.size == 0:
        n_bins = int((mz_max - mz_min) / bin_width)
        return np.zeros(n_bins, dtype=np.float32)
    nl = (precursor_mz - mz).astype(np.float32)
    return bin_spectrum(nl, intensity, mz_min=mz_min, mz_max=mz_max,
                        bin_width=bin_width)


# ── Adduct scalar encoder ─────────────────────────────────────────────────────
def encode_adduct_onehot(adduct_encoded: int, n_adducts: int) -> np.ndarray:
    """One-hot encode an integer-encoded adduct into a float32 vector."""
    vec = np.zeros(n_adducts, dtype=np.float32)
    if 0 <= adduct_encoded < n_adducts:
        vec[adduct_encoded] = 1.0
    return vec


# ── Hierarchical accuracy helpers ─────────────────────────────────────────────
def _ox_suffix(nox: int) -> str:
    """Return the oxidation suffix string for a given nox count."""
    if nox == 0:
        return ""
    if nox == 1:
        return ";O"
    return f";{nox}O"


def reconstruct_name(
    cls: str,
    nc1: int,
    ndb1: int,
    nox1: int,
    nc2: int = 0,
    ndb2: int = 0,
    nox2: int = 0,
    nc3: int = 0,
    ndb3: int = 0,
    nox3: int = 0,
    nc4: int = 0,
    ndb4: int = 0,
    nox4: int = 0,
    class_to_numchain: dict | None = None,
) -> str:
    """
    Build the canonical lipid name from chain components.

    Chain position order is ignored: chains are sorted by
    (nc descending, ndb descending, nox descending) and joined
    with '-' (indicating unordered / position-agnostic composition).

    Oxidation suffix convention:
        nox == 0  →  ""
        nox == 1  →  ";O"
        nox >= 2  →  ";{n}O"

    Only includes chain tokens up to num_chain[cls].

    Example: reconstruct_name("PC", 16, 0, 0, 18, 2, 0) → "PC 18:2-16:0"
    """
    if class_to_numchain is not None:
        n_chains = class_to_numchain.get(cls, 1)
    else:
        # guess from supplied non-zero values
        if nc4 != 0:
            n_chains = 4
        elif nc3 != 0:
            n_chains = 3
        elif nc2 != 0:
            n_chains = 2
        else:
            n_chains = 1

    chains = [
        (nc1, ndb1, nox1),
        (nc2, ndb2, nox2),
        (nc3, ndb3, nox3),
        (nc4, ndb4, nox4),
    ][:n_chains]

    # Sort: longer chain first, more double bonds first, higher oxidation first
    chains_sorted = sorted(chains, key=lambda t: (-t[0], -t[1], -t[2]))

    tokens = [f"{nc}:{ndb}{_ox_suffix(nox)}" for nc, ndb, nox in chains_sorted]
    return f"{cls} {'-'.join(tokens)}"


# ── class_to_numchain loader ──────────────────────────────────────────────────
def load_class_to_numchain() -> dict[str, int]:
    """Load class_to_numchain.json from the shared directory."""
    path = SHARED_DIR / "class_to_numchain.json"
    with open(path) as fh:
        return json.load(fh)


# ── Mass constants ─────────────────────────────────────────────────────────────
H_MASS      = 1.0078250319
O_MASS      = 15.9949146221
CH2_MASS    = 12.0 + 2.0 * H_MASS    # 14.01565006 Da per CH2
H2_MASS     = 2.0 * H_MASS            # 2.01565006 Da per double bond
H2O_MASS    = 2.0 * H_MASS + O_MASS  # 18.01056468
CHAIN_CONST = 2.0 * O_MASS - H2O_MASS  # 13.97926456 Da per acyl chain attachment

SUM_COMP_PPM_TOL = 10.0   # default ppm tolerance for mass-based enumeration
SUM_COMP_ABS_TOL = 0.020  # absolute fallback (Da)

# ── Adduct → neutral mass table ───────────────────────────────────────────────
# Each entry: (charge, ion_mass_Da, n_mol)
#   neutral_mass = (precursor_mz * charge - ion_mass) / n_mol
# n_mol=1 for monomers, n_mol=2 for dimers ([2M±X] adducts).
_H   = 1.00727647    # proton mass
_Na  = 22.98921677
_K   = 38.96315868
_NH4 = 18.03437157
_Li  = 7.01545500
_Cl  = 34.96940210
_Br  = 78.91884000
_H2O = 18.01056468
_CH3CN    = 41.0265491   # C2H3N monoisotopic
_CH3      = 15.0234751   # CH3 monoisotopic
_HCOO     = 44.99765420  # formate
_OAc      = 59.01330500  # acetate / [M+FA-H]-
_C6H10O3  = 130.0629942  # C6H10O3 monoisotopic

ADDUCT_TABLE: dict[str, tuple[int, float, int]] = {
    # ── positive monomers ───────────────────────────────────────
    "[M+H]+"        : (1,  _H,                        1),
    "[M+Na]+"       : (1,  _Na,                       1),
    "[M+K]+"        : (1,  _K,                        1),
    "[M+NH4]+"      : (1,  _NH4,                      1),
    "[M+Li]+"       : (1,  _Li,                       1),
    "[M+2H]2+"      : (2,  2 * _H,                    1),
    "[M+H+Na]2+"    : (2,  _H + _Na,                  1),
    "[M+H-H2O]+"    : (1,  _H - _H2O,                 1),
    "[M-H2O+H]+"    : (1,  _H - _H2O,                 1),   # alias
    "[M+H-2H2O]+"   : (1,  _H - 2 * _H2O,             1),
    "[M-3H2O+H]+"   : (1,  _H - 3 * _H2O,             1),
    "[M+Na-H]+"     : (1,  _Na - _H,                  1),
    "[M-H+2Na]+"    : (1,  2 * _Na - _H,              1),
    "[M+CH3CN+H]+"  : (1,  _CH3CN + _H,               1),
    "[M-C6H10O3+H]+": (1,  _H - _C6H10O3,             1),
    "[M]+"          : (1,  0.0,                        1),   # radical cation
    # ── negative monomers ───────────────────────────────────────
    "[M-H]-"        : (1, -_H,                        1),
    "[M+Cl]-"       : (1,  _Cl,                       1),
    "[M+HCOO]-"     : (1,  _HCOO,                     1),
    "[M+CH3COO]-"   : (1,  _OAc,                      1),
    "[M+Br]-"       : (1,  _Br,                       1),
    "[M-2H]2-"      : (2, -2 * _H,                    1),
    "[M-H-H2O]-"    : (1, -_H - _H2O,                 1),
    "[M-H2O-H]-"    : (1, -_H - _H2O,                 1),   # alias
    "[M-CH3]-"      : (1, -_CH3,                      1),
    "[M+OAc]-"      : (1,  _OAc,                      1),   # alias
    "[M+FA-H]-"     : (1,  _HCOO,                     1),   # alias
    # ── positive dimers ─────────────────────────────────────────
    "[2M+H]+"       : (1,  _H,                        2),
    "[2M+Na]+"      : (1,  _Na,                       2),
    # ── negative dimers ─────────────────────────────────────────
    "[2M-H]-"       : (1, -_H,                        2),
    "[2M-2H+Na]-"   : (1,  _Na - 2 * _H,              2),
}


def adduct_to_neutral(precursor_mz: float, adduct: str) -> float | None:
    """
    Convert precursor_mz + adduct string to neutral monoisotopic mass.

    Supports monomers (n_mol=1) and dimers (n_mol=2).
    Returns None if the adduct string is not recognised.
    """
    entry = ADDUCT_TABLE.get(adduct.strip())
    if entry is None:
        return None
    charge, ion_mass, n_mol = entry
    return (precursor_mz * charge - ion_mass) / n_mol


def find_sum_comp_candidates(
    exact_mass: float,
    backbone_mass: float,
    num_chain: int,
    ppm_tol: float = SUM_COMP_PPM_TOL,
    abs_tol: float = SUM_COMP_ABS_TOL,
) -> list[tuple[int, int, int, float]]:
    """
    Enumerate (total_nc, total_db, total_ox, mass_residual_Da) tuples whose
    implied exact mass is within ppm_tol of exact_mass.

    For each (nc, ndb) pair, total_ox is solved algebraically — no 3-D grid.
    Results are sorted by mass_residual ascending (best match first).
    """
    nc_min, nc_max   = 2, 80
    ndb_max          = 14
    nox_min, nox_max = 0, 6

    tol_da = max(abs_tol, ppm_tol * exact_mass * 1e-6)
    target = exact_mass - backbone_mass - num_chain * CHAIN_CONST

    out: list[tuple[int, int, int, float]] = []
    for nc in range(nc_min, nc_max + 1):
        for ndb in range(0, min(ndb_max, nc) + 1):
            nox_exact   = (target - nc * CH2_MASS + ndb * H2_MASS) / O_MASS
            nox_rounded = round(nox_exact)
            if nox_rounded < nox_min or nox_rounded > nox_max:
                continue
            residual = abs(nox_exact - nox_rounded) * O_MASS
            if residual <= tol_da:
                out.append((nc, ndb, nox_rounded, residual))

    out.sort(key=lambda x: x[3])
    return out


def load_backbone_masses() -> dict[str, float]:
    """Load class_backbone_masses.json from the shared directory."""
    path = SHARED_DIR / "class_backbone_masses.json"
    with open(path) as fh:
        return json.load(fh)


# ── Feature-matrix builder ────────────────────────────────────────────────────
def get_base_feat_cols(df: pd.DataFrame) -> list[str]:
    """
    Base feature columns: spectral bins + precursor_mz_norm + ion_mode_enc.
    adduct_enc is intentionally excluded — it is the first prediction target.
    """
    f_cols  = sorted(c for c in df.columns if c.startswith("F_"))
    nl_cols = sorted(c for c in df.columns if c.startswith("NL_"))
    return f_cols + nl_cols + ["precursor_mz_norm", "ion_mode_enc"]


class FeatureSet:
    """
    Lazy builder for augmented feature matrices.

    Matrices are built on first access and cached.  'base' is never copied —
    it returns the original X_base array directly.

    Keys
    ----
    "base"   : 3102  spectral only (no adduct)                      — adduct model
    "cls"    : 3103  base + adduct                                   — class model
    "chain1" : 3107  base + adduct + class + total (tc,tdb,tox)     — chain-1 models
    "ch2"    : 3110  chain1 + chain-1 labels                        — chain-2 models
    "ch3"    : 3113  ch2   + chain-2 labels                         — chain-3 models
    "ch4"    : 3116  ch3   + chain-3 labels                         — chain-4 models

    Teacher forcing: during training the true class_enc and rule-derived
    (total_c, total_db, total_ox) are used as conditioning features for all
    chain models. At inference time predicted values are used instead.
    """

    def __init__(self, df: pd.DataFrame, base_feat_cols: list[str]) -> None:
        print(f"  Building base feature matrix ({len(base_feat_cols)} cols) …")
        self._base   = df[base_feat_cols].values.astype(np.float32)
        self._adduct = df["adduct_enc"].values.reshape(-1, 1).astype(np.float32)

        # True class label (teacher forcing for chain models).
        self._class  = df["class_enc"].values.reshape(-1, 1).astype(np.float32)

        # True sum-composition totals (teacher forcing for chain models).
        tc  = (df["num_c_1"]  + df["num_c_2"]  + df["num_c_3"]  + df["num_c_4"]).values
        tdb = (df["num_db_1"] + df["num_db_2"] + df["num_db_3"] + df["num_db_4"]).values
        tox = (df["num_ox_1"] + df["num_ox_2"] + df["num_ox_3"] + df["num_ox_4"]).values
        self._total  = np.stack([tc, tdb, tox], axis=1).astype(np.float32)  # (N, 3)

        self._c1     = df[["num_c_1", "num_db_1", "num_ox_1"]].values.astype(np.float32)
        self._c2     = df[["num_c_2", "num_db_2", "num_ox_2"]].values.astype(np.float32)
        self._c3     = df[["num_c_3", "num_db_3", "num_ox_3"]].values.astype(np.float32)
        self._cache: dict[str, np.ndarray] = {}
        print(f"    X_base shape: {self._base.shape}  "
              f"({self._base.nbytes/1e9:.2f} GB)")

    def get(self, key: str) -> np.ndarray:
        if key == "base":
            return self._base
        if key not in self._cache:
            if key == "cls":
                mat = np.concatenate([self._base, self._adduct], axis=1)
            elif key == "chain1":
                mat = np.concatenate([self._base, self._adduct,
                                      self._class, self._total], axis=1)
            elif key == "ch2":
                mat = np.concatenate([self._base, self._adduct,
                                      self._class, self._total, self._c1], axis=1)
            elif key == "ch3":
                mat = np.concatenate([self._base, self._adduct,
                                      self._class, self._total,
                                      self._c1, self._c2], axis=1)
            elif key == "ch4":
                mat = np.concatenate([self._base, self._adduct,
                                      self._class, self._total,
                                      self._c1, self._c2, self._c3], axis=1)
            else:
                raise ValueError(f"Unknown feature key: {key!r}")
            print(f"    Built X[{key}]: shape={mat.shape}  "
                  f"({mat.nbytes/1e9:.2f} GB)")
            self._cache[key] = mat
        return self._cache[key]
