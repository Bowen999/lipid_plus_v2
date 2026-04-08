"""
01_feature_engineering.py — Build the full feature matrix and label matrix.

Feature layout (3,103 features total):
    F_0 … F_1549        : binned fragment spectrum    (1550 float32)
    NL_0 … NL_1549      : binned neutral-loss spectrum (1550 float32)
    precursor_mz_norm   : z-score normalised precursor m/z (float32)
    adduct_enc          : LabelEncoder integer for adduct (int16)
    ion_mode_enc        : 0=negative, 1=positive (int8)

Label columns (int16):
    class_enc, num_c_1..4, num_db_1..4, num_ox_1..4
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DATA_DIR,
    MODELS_DIR,
    N_BINS,
    bin_neutral_loss,
    bin_spectrum,
    clean_spectrum,
    parse_spectrum,
)


def main() -> None:
    """Build and save the feature matrix + labels parquet."""
    print("=" * 60)
    print("01_feature_engineering.py — Feature Engineering")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    src = DATA_DIR / "lipid_ms2_source_validated.parquet"
    print(f"\nLoading validated source from:\n  {src}")
    df = pd.read_parquet(src, engine="pyarrow")
    print(f"Loaded {len(df):,} rows")

    # Load class_to_numchain (for reference; not used in feature building)
    with open(MODELS_DIR / "class_to_numchain.json") as fh:
        _class_to_numchain = json.load(fh)

    # ── 2. Fit and save LabelEncoders ────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFitting class LabelEncoder …")
    class_le = LabelEncoder()
    class_le.fit(df["class"].astype(str))
    joblib.dump(class_le, MODELS_DIR / "class_encoder.joblib", compress=3)
    print(f"  {len(class_le.classes_)} classes: {list(class_le.classes_)[:5]} …")

    print("Fitting adduct LabelEncoder …")
    adduct_le = LabelEncoder()
    adduct_le.fit(df["adduct"].astype(str))
    joblib.dump(adduct_le, MODELS_DIR / "adduct_encoder.joblib", compress=3)
    print(f"  {len(adduct_le.classes_)} adducts")

    # ── 3. Precursor m/z stats ────────────────────────────────────
    pmz_values = df["precursor_mz"].astype(np.float32).values
    pmz_mean   = float(pmz_values.mean())
    pmz_std    = float(pmz_values.std())
    pmz_stats  = np.array([pmz_mean, pmz_std], dtype=np.float64)
    np.save(MODELS_DIR / "precursor_mz_stats.npy", pmz_stats)
    print(f"\nprecursor_mz: mean={pmz_mean:.4f}, std={pmz_std:.4f}")

    # ── 4. Encode label columns ───────────────────────────────────
    class_enc    = class_le.transform(df["class"].astype(str)).astype(np.int16)
    adduct_enc   = adduct_le.transform(df["adduct"].astype(str)).astype(np.int16)
    ion_mode_enc = (df["ion_mode"].str.lower().str.strip() == "positive"
                    ).astype(np.int8).values
    pmz_norm     = ((pmz_values - pmz_mean) / (pmz_std + 1e-8)).astype(np.float32)

    # ── 5. Build spectral feature arrays ─────────────────────────
    n = len(df)
    F_arr  = np.zeros((n, N_BINS), dtype=np.float32)
    NL_arr = np.zeros((n, N_BINS), dtype=np.float32)

    ms2_list     = df["MS2"].tolist()
    precmz_list  = df["precursor_mz"].tolist()

    print(f"\nBuilding spectral features for {n:,} rows …")
    for i in tqdm(range(n), desc="Spectrum processing", unit="row"):
        mz, intensity = parse_spectrum(ms2_list[i])
        if mz.size > 0:
            mz_c, int_c = clean_spectrum(mz, intensity, float(precmz_list[i]))
            F_arr[i]  = bin_spectrum(mz_c, int_c)
            NL_arr[i] = bin_neutral_loss(mz_c, int_c, float(precmz_list[i]))

    # ── 6. Assemble DataFrame ─────────────────────────────────────
    print("\nAssembling feature DataFrame …")
    f_cols  = [f"F_{i}"  for i in range(N_BINS)]
    nl_cols = [f"NL_{i}" for i in range(N_BINS)]

    feature_df = pd.DataFrame(F_arr,  columns=f_cols,  dtype=np.float32)

    # Append NL columns
    for j, col in enumerate(nl_cols):
        feature_df[col] = NL_arr[:, j]

    feature_df["precursor_mz_norm"] = pmz_norm
    feature_df["adduct_enc"]        = adduct_enc
    feature_df["ion_mode_enc"]      = ion_mode_enc

    # Label columns
    feature_df["class_enc"] = class_enc

    label_cols = [
        "num_c_1", "num_db_1", "num_ox_1",
        "num_c_2", "num_db_2", "num_ox_2",
        "num_c_3", "num_db_3", "num_ox_3",
        "num_c_4", "num_db_4", "num_ox_4",
    ]
    for col in label_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int16)
        else:
            vals = pd.array([0] * n, dtype="int16")
        feature_df[col] = vals.values

    # ── 7. Save ───────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "lipid_ms2_features.parquet"
    print(f"\nSaving feature matrix to:\n  {out}")
    feature_df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)

    # ── 8. Summary ────────────────────────────────────────────────
    print(f"\nFeature matrix shape : {feature_df.shape}")
    print(f"Total features       : {3 + 2*N_BINS} (fragment + NL + precmz_norm + adduct_enc + ion_mode_enc)")
    print(f"Memory usage         : {feature_df.memory_usage(deep=True).sum() / 1e9:.3f} GB")
    dtypes = feature_df.dtypes.value_counts()
    print("\nDtype breakdown:")
    for dtype, cnt in dtypes.items():
        print(f"  {str(dtype):<12} : {cnt:>6} columns")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[DONE] {ts}")


if __name__ == "__main__":
    main()
