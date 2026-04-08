"""
00_clean_spectra.py — Validate spectra and derive class_to_numchain lookup.

Steps:
1. Load the source parquet.
2. Parse every MS2 string; flag rows where parsing yields zero peaks.
3. Print a summary (total rows, rows without parseable spectra).
4. Derive and save class_to_numchain.json.
5. Save a copy of the dataframe with has_spectrum flag appended.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Ensure sibling imports work when run from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DATA_PATH,
    DATA_DIR,
    MODELS_DIR,
    parse_spectrum,
)


def main() -> None:
    """Run spectrum validation and class_to_numchain derivation."""
    print("=" * 60)
    print("00_clean_spectra.py — Spectrum Validation")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print(f"\nLoading data from:\n  {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # ── 2. Parse spectra and flag rows ────────────────────────────
    print("\nParsing MS2 spectra …")
    has_spectrum: list[bool] = []
    for ms2 in tqdm(df["MS2"], desc="Parsing", unit="row"):
        mz, _ = parse_spectrum(ms2)
        has_spectrum.append(mz.size > 0)

    df["has_spectrum"] = has_spectrum

    # ── 3. Summary ────────────────────────────────────────────────
    n_total    = len(df)
    n_with     = sum(has_spectrum)
    n_without  = n_total - n_with
    print(f"\n{'─'*40}")
    print(f"Total rows              : {n_total:>10,}")
    print(f"Rows WITH spectrum      : {n_with:>10,}  ({100*n_with/n_total:.2f} %)")
    print(f"Rows WITHOUT spectrum   : {n_without:>10,}  ({100*n_without/n_total:.2f} %)")
    print(f"{'─'*40}")
    print("NOTE: rows without a parseable spectrum are retained.")
    print("      Their spectral feature vectors will be all-zeros.")

    # ── 4. Derive and save class_to_numchain ──────────────────────
    class_to_numchain: dict[str, int] = (
        df.groupby("class")["num_chain"]
        .first()
        .astype(int)
        .to_dict()
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = MODELS_DIR / "class_to_numchain.json"
    with open(out_json, "w") as fh:
        json.dump(class_to_numchain, fh, indent=2, sort_keys=True)
    print(f"\nSaved class_to_numchain.json → {out_json}")

    # Print table
    print(f"\n{'Class':<20} {'num_chain':>9}")
    print("─" * 32)
    for cls, nc in sorted(class_to_numchain.items(), key=lambda x: (x[1], x[0])):
        print(f"{cls:<20} {nc:>9}")

    # Chain-count distribution
    from collections import Counter
    chain_dist = Counter(class_to_numchain.values())
    print(f"\nChain-count distribution:")
    for n, cnt in sorted(chain_dist.items()):
        print(f"  num_chain={n} : {cnt} classes")

    # ── 5. Save validated parquet ─────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = DATA_DIR / "lipid_ms2_source_validated.parquet"
    df.to_parquet(out_parquet, engine="pyarrow", compression="snappy", index=False)
    print(f"\nSaved source-validated parquet → {out_parquet}")
    print(f"Shape: {df.shape}")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[DONE] {ts}")


if __name__ == "__main__":
    main()
