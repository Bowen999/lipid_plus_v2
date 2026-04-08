"""
04_evaluate.py — Run inference and compute metrics for a trained DL model.

Usage:
  python scripts/04_evaluate.py --model mlp
  python scripts/04_evaluate.py --model cnn
  python scripts/04_evaluate.py --model transformer
  python scripts/04_evaluate.py --model mlp --splits val test

Outputs to:
  phase2_dl/outputs/{model_name}/evaluation/{split}_metrics.json
  phase2_dl/outputs/{model_name}/predictions/{split}_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

# ── sys.path ──────────────────────────────────────────────────────────────────
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib.util as _ilu

import joblib
import numpy as np
import pandas as pd
import torch

from utils import (
    FEAT_PARQUET, RAW_PARQUET, SPLITS_DIR, SHARED_DIR, OUTPUTS_DIR,
    TARGETS,
    load_class_to_numchain, load_backbone_masses,
    build_label_maps,
)
from datasets.mlp_dataset         import MLPDataset
from datasets.cnn_dataset         import CNNDataset
from datasets.transformer_dataset import TransformerDataset
from models.mlp          import LipidMLP
from models.cnn          import LipidCNN
from models.transformer  import LipidTransformer
from pipeline.inference  import predict_split_dl

# phase1_ml evaluation — load via importlib to avoid utils name collision
_REPO_ROOT  = Path(__file__).resolve().parents[2]
_P1_SRC     = _REPO_ROOT / "phase1_ml" / "src"
_P1_EVAL    = _P1_SRC / "evaluation"

_p1u_spec = _ilu.spec_from_file_location("_p1_utils", str(_P1_SRC / "utils.py"))
_p1u_mod  = _ilu.module_from_spec(_p1u_spec)
sys.modules["_p1_utils"] = _p1u_mod
_p1u_spec.loader.exec_module(_p1u_mod)   # type: ignore[union-attr]

_prev_utils = sys.modules.get("utils")
sys.modules["utils"] = _p1u_mod
_m_spec = _ilu.spec_from_file_location("_p1_eval_metrics", str(_P1_EVAL / "metrics.py"))
_p1_metrics = _ilu.module_from_spec(_m_spec)
_m_spec.loader.exec_module(_p1_metrics)  # type: ignore[union-attr]
if _prev_utils is None:
    sys.modules.pop("utils", None)
else:
    sys.modules["utils"] = _prev_utils

compute_metrics = _p1_metrics.compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["mlp", "cnn", "transformer"])
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size from config")
    return p.parse_args()


def get_base_feat_cols(df: pd.DataFrame) -> list[str]:
    f_cols  = sorted([c for c in df.columns if c.startswith("F_")],
                     key=lambda x: int(x.split("_")[1]))
    nl_cols = sorted([c for c in df.columns if c.startswith("NL_")],
                     key=lambda x: int(x.split("_")[1]))
    return f_cols + nl_cols + ["precursor_mz_norm", "ion_mode_enc"]


def main() -> None:
    args       = parse_args()
    model_name = args.model
    out_dir    = OUTPUTS_DIR / model_name

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load train metadata ────────────────────────────────────────────────────
    meta_path = out_dir / "models" / "train_meta.pkl"
    if not meta_path.exists():
        print(f"[ERROR] Train metadata not found: {meta_path}")
        print("  Run 02_train.py first.")
        sys.exit(1)

    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)

    label_maps    = meta["label_maps"]
    n_classes     = meta["n_classes"]
    pmz_stats     = meta["pmz_stats"]
    row_num_chain = meta["row_num_chain"]

    # ── Load shared encoders / metadata ───────────────────────────────────────
    class_le  = joblib.load(SHARED_DIR / "class_encoder.joblib")
    adduct_le = joblib.load(SHARED_DIR / "adduct_encoder.joblib")

    class_to_numchain = load_class_to_numchain()
    backbone_masses   = load_backbone_masses()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data …")
    df_feat = pd.read_parquet(FEAT_PARQUET)
    df_raw  = pd.read_parquet(RAW_PARQUET)

    precmz_all = df_raw["precursor_mz"].values.astype(np.float32)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {model_name} model …")
    if model_name == "mlp":
        model = LipidMLP(n_classes)
    elif model_name == "cnn":
        model = LipidCNN(n_classes, n_adducts=n_classes["adduct_enc"])
    else:
        model = LipidTransformer(n_classes)

    ckpt_path = out_dir / "models" / "best.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")

    # ── Config batch size ─────────────────────────────────────────────────────
    config_path = Path(__file__).resolve().parent.parent / "configs" / f"{model_name}.json"
    with open(config_path) as fh:
        config = json.load(fh)
    batch_size = args.batch_size or int(config["training"].get("batch_size", 256))

    # ── Per-split evaluation ───────────────────────────────────────────────────
    (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    base_feat_cols = get_base_feat_cols(df_feat) if model_name == "mlp" else None

    for split_name in args.splits:
        split_idx = np.load(SPLITS_DIR / f"split_{split_name}.npy")

        # Build dataset (no augmentation for eval)
        if model_name == "mlp":
            dataset = MLPDataset(df_feat, split_idx, label_maps, row_num_chain,
                                 base_feat_cols)
        elif model_name == "cnn":
            dataset = CNNDataset(df_feat, df_raw, split_idx, label_maps, row_num_chain,
                                 augment=False, pmz_stats=pmz_stats)
        else:
            dataset = TransformerDataset(df_feat, df_raw, split_idx, label_maps, row_num_chain,
                                         augment=False, pmz_stats=pmz_stats)

        # Run inference cascade
        pred_df = predict_split_dl(
            split_name=split_name,
            dataset=dataset,
            model=model,
            model_type=model_name,
            label_maps=label_maps,
            class_le=class_le,
            adduct_le=adduct_le,
            class_to_numchain=class_to_numchain,
            row_num_chain=row_num_chain,
            backbone_masses=backbone_masses,
            precmz_all=precmz_all,
            device=device,
            batch_size=batch_size,
        )

        # ── Separate flagged (data-starved, multi-chain) samples ──────────────
        n_total    = len(pred_df)
        low_flag   = pred_df.get("low_data_flag", pd.Series(False, index=pred_df.index))
        n_flagged  = int(low_flag.sum())
        eval_df    = pred_df[~low_flag].copy() if n_flagged > 0 else pred_df
        if n_flagged > 0:
            print(f"  Excluded {n_flagged:,} insufficient_data samples "
                  f"({100*n_flagged/n_total:.1f}%) from L1/L2/L3 metrics.")

        # Compute metrics (reuse phase1_ml compute_metrics) on non-flagged rows
        metrics = compute_metrics(eval_df, class_to_numchain)
        metrics["insufficient_data_count"] = n_flagged
        metrics["insufficient_data_pct"]   = round(100 * n_flagged / max(n_total, 1), 2)

        # Save metrics JSON
        metrics_path = out_dir / "evaluation" / f"{split_name}_metrics.json"
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"  Metrics → {metrics_path}")

        # Save predictions CSV
        pred_path = out_dir / "predictions" / f"{split_name}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"  Predictions → {pred_path}")

        # Print key metrics to console
        l0 = metrics.get("adduct_accuracy",                 0.0)
        l1 = metrics.get("level0_class_accuracy",           0.0)
        l2 = metrics.get("level1_sum_composition_accuracy", 0.0)
        l3 = metrics.get("level3_name_exact_match",         0.0)
        print(f"  {split_name.upper():5s} adduct={l0:.4f}  class={l1:.4f}  "
              f"sum_comp={l2:.4f}  name={l3:.4f}  "
              f"(excl. {n_flagged} insufficient_data)")

    print(f"\nEvaluation complete for {model_name}.")


if __name__ == "__main__":
    main()
