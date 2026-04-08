"""
02_split_data.py — Stratified 70 / 15 / 15 train/val/test split.

Split is stratified on class_enc to ensure each class is proportionally
represented in every subset.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, MODELS_DIR


def main() -> None:
    """Perform stratified split and save index arrays."""
    print("=" * 60)
    print("02_split_data.py — Stratified Train / Val / Test Split")
    print("=" * 60)

    # ── Load feature parquet (only need class_enc column) ─────────
    feat_path = DATA_DIR / "lipid_ms2_features.parquet"
    print(f"\nLoading class labels from:\n  {feat_path}")
    class_enc = pd.read_parquet(feat_path, engine="pyarrow",
                                columns=["class_enc"])["class_enc"].values
    n = len(class_enc)
    indices = np.arange(n, dtype=np.int32)
    print(f"Total samples: {n:,}")

    # ── Load class encoder for human-readable reporting ───────────
    import joblib
    class_le = joblib.load(MODELS_DIR / "class_encoder.joblib")

    # ── Separate singleton / doubleton classes (can't stratify on them) ──
    # Classes with < 3 members can't be split into train/val/test with stratify.
    # Assign them directly to train so they are at least seen during training.
    class_counts = np.bincount(class_enc, minlength=int(class_enc.max()) + 1)
    rare_classes = set(np.where(class_counts < 3)[0].tolist())
    if rare_classes:
        rare_names = [class_le.inverse_transform([c])[0] for c in sorted(rare_classes)]
        print(f"\nRare classes (< 3 members, assigned directly to train): "
              f"{rare_names}")
    rare_mask    = np.array([c in rare_classes for c in class_enc], dtype=bool)
    rare_idx     = indices[rare_mask]
    common_idx   = indices[~rare_mask]
    common_class = class_enc[common_idx]

    # ── Split 1: 70 % train, 30 % temp (stratified on common classes) ──
    train_common, temp_idx = train_test_split(
        common_idx, test_size=0.30, stratify=common_class, random_state=42
    )
    # Merge rare singletons into train
    train_idx = np.concatenate([train_common, rare_idx]).astype(np.int32)

    # ── Split 2: 50 % of temp → val, 50 % → test (= 15 / 15) ────
    # After the 70/30 split some classes may land only 1 sample in temp
    # (e.g. a class with 3 members: 2 → train, 1 → temp).  Extract those
    # before the second stratified split and assign them directly to val.
    temp_class_counts = Counter(class_enc[temp_idx].tolist())
    rare_in_temp = {c for c, cnt in temp_class_counts.items() if cnt < 2}
    if rare_in_temp:
        rare_names2 = [class_le.inverse_transform([c])[0] for c in sorted(rare_in_temp)]
        print(f"Rare-in-temp classes (< 2 members in temp, assigned to val): "
              f"{rare_names2}")
    rare_temp_mask  = np.array([class_enc[i] in rare_in_temp for i in temp_idx],
                                dtype=bool)
    rare_temp_idx   = temp_idx[rare_temp_mask]
    common_temp_idx = temp_idx[~rare_temp_mask]

    val_common, test_idx = train_test_split(
        common_temp_idx,
        test_size=0.50,
        stratify=class_enc[common_temp_idx],
        random_state=42,
    )
    val_idx = np.concatenate([val_common, rare_temp_idx]).astype(np.int32)

    # ── Save index arrays ─────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "split_train.npy", train_idx.astype(np.int32))
    np.save(DATA_DIR / "split_val.npy",   val_idx.astype(np.int32))
    np.save(DATA_DIR / "split_test.npy",  test_idx.astype(np.int32))

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'Split':<10} {'Size':>10}  {'%':>6}")
    print("─" * 30)
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        pct = 100 * len(idx) / n
        print(f"{name:<10} {len(idx):>10,}  {pct:>5.1f} %")

    # ── Per-class counts ──────────────────────────────────────────
    print("\nPer-class distribution (first 20 classes by total count):")
    train_ctr = Counter(class_enc[train_idx].tolist())
    val_ctr   = Counter(class_enc[val_idx].tolist())
    test_ctr  = Counter(class_enc[test_idx].tolist())

    class_totals = Counter(class_enc.tolist())
    top_classes  = [c for c, _ in class_totals.most_common(20)]

    print(f"\n{'Class':<22} {'Total':>8} {'Train':>8} {'Val':>6} {'Test':>6}")
    print("─" * 56)
    for c in top_classes:
        cls_name = class_le.inverse_transform([c])[0]
        print(
            f"{cls_name:<22} {class_totals[c]:>8,}"
            f" {train_ctr[c]:>8,} {val_ctr[c]:>6,} {test_ctr[c]:>6,}"
        )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[DONE] {ts}")


if __name__ == "__main__":
    main()
