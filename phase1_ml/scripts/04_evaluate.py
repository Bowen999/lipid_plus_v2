"""
04_evaluate.py — Evaluate a trained model family on val/test splits.

Usage:
    python scripts/04_evaluate.py --model lightgbm
    python scripts/04_evaluate.py --model random_forest

Outputs (written to outputs/{model}/evaluation/ and outputs/{model}/predictions/):
    val_predictions.csv
    test_predictions.csv
    val_metrics.json
    test_metrics.json
    val_class_confusion.csv
    test_class_confusion.csv
    evaluation_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
PHASE1_ROOT = SCRIPTS_DIR.parent
SRC_DIR     = PHASE1_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import (          # noqa: E402
    DATA_DIR, SPLITS_DIR, SHARED_DIR, OUTPUTS_DIR,
    get_base_feat_cols,
    load_class_to_numchain, load_backbone_masses,
)
from evaluation.metrics import (   # noqa: E402
    build_class_maps,
    predict_split,
    compute_metrics,
    write_confusion_matrix,
    write_report,
)
from pipeline.inference import InferencePipeline   # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        help="Model family name (e.g. lightgbm, random_forest, xgboost)",
    )
    args = parser.parse_args()
    model_name = args.model

    print("=" * 62)
    print(f"04_evaluate.py — Evaluation ({model_name})")
    print("=" * 62)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Output paths ──────────────────────────────────────────────
    eval_dir  = OUTPUTS_DIR / model_name / "evaluation"
    preds_dir = OUTPUTS_DIR / model_name / "predictions"
    eval_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # ── Load feature matrix ───────────────────────────────────────
    feat_path = DATA_DIR / "lipid_ms2_features.parquet"
    print(f"\nLoading features from:\n  {feat_path}")
    df = pd.read_parquet(feat_path, engine="pyarrow")
    print(f"Feature matrix: {df.shape}")

    # ── Load split indices ────────────────────────────────────────
    train_idx = np.load(SPLITS_DIR / "split_train.npy")
    val_idx   = np.load(SPLITS_DIR / "split_val.npy")
    test_idx  = np.load(SPLITS_DIR / "split_test.npy")
    print(f"Splits: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")

    # ── Load encoders + metadata ──────────────────────────────────
    class_le          = joblib.load(SHARED_DIR / "class_encoder.joblib")
    adduct_le         = joblib.load(SHARED_DIR / "adduct_encoder.joblib")
    class_to_numchain = load_class_to_numchain()
    backbone_masses   = load_backbone_masses()

    # ── Row num_chain ─────────────────────────────────────────────
    class_names   = class_le.inverse_transform(df["class_enc"].values)
    row_num_chain = np.array(
        [class_to_numchain.get(c, 1) for c in class_names], dtype=np.int8
    )

    # ── Build base feature matrix (all rows) ─────────────────────
    base_feat_cols = get_base_feat_cols(df)
    print(f"\nBase feature columns: {len(base_feat_cols)}  (no adduct_enc)")
    X_base_all = df[base_feat_cols].values.astype(np.float32)

    # ── Recover original precursor_mz from normalised value ──────
    pmz_stats  = np.load(SHARED_DIR / "precursor_mz_stats.npy")
    pmz_mean, pmz_std = float(pmz_stats[0]), float(pmz_stats[1])
    precmz_all = df["precursor_mz_norm"].values * pmz_std + pmz_mean

    # ── Load models via InferencePipeline ─────────────────────────
    print(f"\nLoading {model_name} models …")
    pipeline = InferencePipeline(model_name)

    # ── Build class maps ──────────────────────────────────────────
    class_maps = build_class_maps(df, train_idx, row_num_chain)

    # ── Run predictions ───────────────────────────────────────────
    common_kwargs = dict(
        X_base_all        = X_base_all,
        precmz_all        = precmz_all,
        df                = df,
        class_maps        = class_maps,
        class_le          = class_le,
        adduct_le         = adduct_le,
        class_to_numchain = class_to_numchain,
        row_num_chain     = row_num_chain,
        backbone_masses   = backbone_masses,
    )

    val_pred_df  = pipeline.run("val",  val_idx,  **common_kwargs)
    test_pred_df = pipeline.run("test", test_idx, **common_kwargs)

    # ── Save predictions ──────────────────────────────────────────
    val_pred_df.to_csv( preds_dir / "val_predictions.csv",  index=False)
    test_pred_df.to_csv(preds_dir / "test_predictions.csv", index=False)
    print(f"\nSaved val_predictions.csv  ({len(val_pred_df):,} rows)")
    print(f"Saved test_predictions.csv ({len(test_pred_df):,} rows)")

    # ── Compute metrics ───────────────────────────────────────────
    print("\nComputing metrics …")
    val_metrics  = compute_metrics(val_pred_df,  class_to_numchain)
    test_metrics = compute_metrics(test_pred_df, class_to_numchain)

    with open(eval_dir / "val_metrics.json",  "w") as fh:
        json.dump(val_metrics,  fh, indent=2)
    with open(eval_dir / "test_metrics.json", "w") as fh:
        json.dump(test_metrics, fh, indent=2)

    # ── Print summary ─────────────────────────────────────────────
    scalar_keys = [
        "adduct_accuracy",
        "level0_class_accuracy",
        "level1_sum_composition_accuracy",
        "level2_full_chain_accuracy",
        "level3_name_exact_match",
    ]
    print(f"\n{'Metric':<44} {'Val':>8} {'Test':>8}")
    print("─" * 62)
    for k in scalar_keys:
        vv = val_metrics.get(k, float("nan"))
        tv = test_metrics.get(k, float("nan"))
        print(f"{k:<44} {vv:>8.4f} {tv:>8.4f}")

    print("\nSum comp status (val):")
    for s, cnt in sorted(val_metrics["sum_comp_status_counts"].items()):
        pct = 100 * cnt / len(val_pred_df)
        print(f"  {s:<14}: {cnt:>6,}  ({pct:.1f}%)")

    # ── Confusion matrices ────────────────────────────────────────
    print("\nWriting confusion matrices …")
    write_confusion_matrix(
        val_pred_df,  eval_dir / "val_class_confusion.csv",  class_le
    )
    write_confusion_matrix(
        test_pred_df, eval_dir / "test_class_confusion.csv", class_le
    )

    # ── Evaluation report ─────────────────────────────────────────
    write_report(
        val_pred_df, test_pred_df,
        val_metrics, test_metrics,
        class_to_numchain,
        train_size = len(train_idx),
        val_size   = len(val_idx),
        test_size  = len(test_idx),
        out_path   = eval_dir / "evaluation_report.md",
    )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[DONE] {ts}")


if __name__ == "__main__":
    main()
