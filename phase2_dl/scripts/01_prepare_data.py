"""
01_prepare_data.py — Verify data files exist, check hardware, print summary.

Checks:
  - phase1_ml features parquet (lipid_ms2_features.parquet)
  - raw MS2 cleaned parquet   (lipid_ms2_cleaned.parquet)
  - split .npy files          (split_{train,val,test}.npy)
  - shared encoder files      (class_encoder.joblib, adduct_encoder.joblib, etc.)
  - MPS / CUDA availability
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── sys.path ──────────────────────────────────────────────────────────────────
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd
import torch

from utils import (
    FEAT_PARQUET, RAW_PARQUET, SPLITS_DIR, SHARED_DIR,
    TARGETS, BASE_FEAT_DIM,
)

REQUIRED_SHARED = [
    "class_encoder.joblib",
    "adduct_encoder.joblib",
    "class_to_numchain.json",
    "class_backbone_masses.json",
    "precursor_mz_stats.npy",
]


def check_file(path: Path, label: str) -> bool:
    ok = path.exists() and path.stat().st_size > 0
    status = "OK" if ok else "MISSING"
    print(f"  [{status}]  {label}  ({path})")
    return ok


def main() -> None:
    print("=" * 65)
    print("phase2_dl — Data & Hardware Check")
    print("=" * 65)

    all_ok = True

    # ── Feature parquet ───────────────────────────────────────────────────────
    print("\n[1] Feature parquet (phase1_ml)")
    ok = check_file(FEAT_PARQUET, "lipid_ms2_features.parquet")
    all_ok &= ok
    if ok:
        df = pd.read_parquet(FEAT_PARQUET)
        print(f"       rows={len(df):,}  cols={len(df.columns):,}")
        missing_tgts = [t for t in TARGETS if t not in df.columns]
        if missing_tgts:
            print(f"  [WARN] Missing target columns: {missing_tgts}")
        feat_cols = [c for c in df.columns if c.startswith("F_") or c.startswith("NL_")]
        print(f"       feature cols found: {len(feat_cols):,}  (expected ~{BASE_FEAT_DIM})")

    # ── Raw MS2 parquet ───────────────────────────────────────────────────────
    print("\n[2] Raw MS2 parquet (cleaned)")
    ok = check_file(RAW_PARQUET, "lipid_ms2_cleaned.parquet")
    all_ok &= ok
    if ok:
        df_raw = pd.read_parquet(RAW_PARQUET, columns=["MS2", "precursor_mz"])
        print(f"       rows={len(df_raw):,}")

    # ── Split indices ─────────────────────────────────────────────────────────
    print("\n[3] Split indices")
    for split in ("train", "val", "test"):
        path = SPLITS_DIR / f"split_{split}.npy"
        ok = check_file(path, f"split_{split}.npy")
        all_ok &= ok
        if ok:
            idx = np.load(path)
            print(f"       {split}: {len(idx):,} rows")

    # ── Shared encoder files ──────────────────────────────────────────────────
    print("\n[4] Shared encoder / metadata files")
    for fname in REQUIRED_SHARED:
        ok = check_file(SHARED_DIR / fname, fname)
        all_ok &= ok

    # ── Hardware ──────────────────────────────────────────────────────────────
    print("\n[5] Hardware")
    if torch.backends.mps.is_available():
        print("  [OK]  MPS (Apple Silicon GPU) available")
        device_str = "mps"
    elif torch.cuda.is_available():
        print(f"  [OK]  CUDA available  ({torch.cuda.get_device_name(0)})")
        device_str = "cuda"
    else:
        print("  [WARN] No GPU detected — training will use CPU (slow)")
        device_str = "cpu"

    print(f"       selected device: {device_str}")
    print(f"       torch version:   {torch.__version__}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_ok:
        print("All checks passed.  Ready to train.")
    else:
        print("Some files are MISSING.  Run phase1_ml pipeline first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
