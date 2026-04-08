"""
02_train.py — Train a single DL model (mlp, cnn, or transformer).

Usage:
  python scripts/02_train.py --model mlp
  python scripts/02_train.py --model cnn
  python scripts/02_train.py --model transformer
  python scripts/02_train.py --model mlp --quick     # 5-epoch smoke-test

Outputs to:
  phase2_dl/outputs/{model_name}/models/best.pt
  phase2_dl/outputs/{model_name}/evaluation/training_log.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── sys.path ──────────────────────────────────────────────────────────────────
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import (
    FEAT_PARQUET, RAW_PARQUET, SPLITS_DIR, SHARED_DIR, OUTPUTS_DIR,
    TARGETS, BASE_FEAT_DIM,
    load_class_to_numchain,
    build_label_maps,
)
from datasets.mlp_dataset        import MLPDataset
from datasets.cnn_dataset        import CNNDataset
from datasets.transformer_dataset import TransformerDataset
from models.mlp         import LipidMLP
from models.cnn         import LipidCNN
from models.transformer import LipidTransformer
from training.trainer   import Trainer
from training.l3_eval   import make_l3_eval_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["mlp", "cnn", "transformer"])
    p.add_argument("--quick", action="store_true",
                   help="5-epoch smoke-test with reduced data")
    return p.parse_args()


def get_base_feat_cols(df: pd.DataFrame) -> list[str]:
    """Return the 3102 feature column names in order."""
    f_cols  = sorted([c for c in df.columns if c.startswith("F_")],
                     key=lambda x: int(x.split("_")[1]))
    nl_cols = sorted([c for c in df.columns if c.startswith("NL_")],
                     key=lambda x: int(x.split("_")[1]))
    return f_cols + nl_cols + ["precursor_mz_norm", "ion_mode_enc"]


def main() -> None:
    args = parse_args()
    model_name = args.model

    # ── Config ────────────────────────────────────────────────────────────────
    config_path = Path(__file__).resolve().parent.parent / "configs" / f"{model_name}.json"
    with open(config_path) as fh:
        config = json.load(fh)

    if args.quick:
        config["training"]["max_epochs"] = 5
        config["training"]["patience"]   = 5
        print("[quick mode] max_epochs=5, patience=5")

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data …")
    df_feat = pd.read_parquet(FEAT_PARQUET)
    df_raw  = pd.read_parquet(RAW_PARQUET)
    assert len(df_feat) == len(df_raw), (
        f"Row count mismatch: features={len(df_feat)}, raw={len(df_raw)}"
    )

    train_idx = np.load(SPLITS_DIR / "split_train.npy")
    val_idx   = np.load(SPLITS_DIR / "split_val.npy")

    if args.quick:
        rng = np.random.default_rng(42)
        train_idx = rng.choice(train_idx, size=min(2000, len(train_idx)),  replace=False)
        val_idx   = rng.choice(val_idx,   size=min(500,  len(val_idx)),    replace=False)
        print(f"[quick] train={len(train_idx):,}  val={len(val_idx):,}")

    # ── class_to_numchain → row_num_chain ─────────────────────────────────────
    class_to_numchain = load_class_to_numchain()
    class_enc_col = df_feat["class_enc"].values.astype(np.int32)

    class_le = joblib.load(SHARED_DIR / "class_encoder.joblib")
    adduct_le = joblib.load(SHARED_DIR / "adduct_encoder.joblib")

    class_names     = class_le.inverse_transform(class_enc_col)
    row_num_chain   = np.array(
        [class_to_numchain.get(c, 1) for c in class_names], dtype=np.int32
    )

    # ── precursor_mz stats ────────────────────────────────────────────────────
    pmz_stats_arr = np.load(SHARED_DIR / "precursor_mz_stats.npy")
    pmz_stats = (float(pmz_stats_arr[0]), float(pmz_stats_arr[1]))   # (mean, std)
    pmz_mean, pmz_std = pmz_stats
    precmz_all = (df_feat["precursor_mz_norm"].values * pmz_std + pmz_mean
                  ).astype(np.float32)

    # ── Label maps ────────────────────────────────────────────────────────────
    label_maps = build_label_maps(df_feat, train_idx, row_num_chain)
    n_classes  = {t: len(label_maps[t]) for t in TARGETS}
    print(f"  n_classes: adduct={n_classes['adduct_enc']}  class={n_classes['class_enc']}")

    batch_size = int(config["training"].get("batch_size", 256))

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nBuilding datasets …")
    if model_name == "mlp":
        base_feat_cols = get_base_feat_cols(df_feat)
        train_ds = MLPDataset(df_feat, train_idx, label_maps, row_num_chain, base_feat_cols,
                              augment=True)
        val_ds   = MLPDataset(df_feat, val_idx,   label_maps, row_num_chain, base_feat_cols,
                              augment=False)
    elif model_name == "cnn":
        train_ds = CNNDataset(df_feat, df_raw, train_idx, label_maps, row_num_chain,
                              augment=True, pmz_stats=pmz_stats)
        val_ds   = CNNDataset(df_feat, df_raw, val_idx,   label_maps, row_num_chain,
                              augment=False, pmz_stats=pmz_stats)
    else:  # transformer
        train_ds = TransformerDataset(df_feat, df_raw, train_idx, label_maps, row_num_chain,
                                      augment=True, pmz_stats=pmz_stats)
        val_ds   = TransformerDataset(df_feat, df_raw, val_idx,   label_maps, row_num_chain,
                                      augment=False, pmz_stats=pmz_stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=False,
                              multiprocessing_context="fork",
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=False,
                              multiprocessing_context="fork",
                              persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model …")
    if model_name == "mlp":
        model = LipidMLP(n_classes)
    elif model_name == "cnn":
        model = LipidCNN(n_classes, n_adducts=n_classes["adduct_enc"])
    else:
        model = LipidTransformer(n_classes)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # ── Output dirs ───────────────────────────────────────────────────────────
    out_dir = OUTPUTS_DIR / model_name
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)

    # ── L3 early-stopping function ────────────────────────────────────────────
    print("\nBuilding L3 evaluation function …")
    from utils import reconstruct_name as _rn
    import importlib.util as _ilu, sys as _sys
    _p1_src  = Path(_SRC).parent.parent / "phase1_ml" / "src"
    _p1_eval = _p1_src / "evaluation"
    # Load phase1_ml utils fresh so metrics.py can import it as "utils"
    _u_spec = _ilu.spec_from_file_location("_02t_p1u", str(_p1_src / "utils.py"))
    _p1u    = _ilu.module_from_spec(_u_spec)
    _u_spec.loader.exec_module(_p1u)
    _prev = _sys.modules.get("utils")
    _sys.modules["utils"] = _p1u
    _m_spec = _ilu.spec_from_file_location("_02t_metrics", str(_p1_eval / "metrics.py"))
    _p1m    = _ilu.module_from_spec(_m_spec)
    _m_spec.loader.exec_module(_p1m)
    if _prev is None:
        _sys.modules.pop("utils", None)
    else:
        _sys.modules["utils"] = _prev

    # Pre-compute true canonical names for all rows
    _class_names_all = class_le.inverse_transform(df_feat["class_enc"].values)
    _true_names_all  = np.array([
        _rn(
            _class_names_all[i],
            int(df_feat["num_c_1"].iat[i]),  int(df_feat["num_db_1"].iat[i]),  int(df_feat["num_ox_1"].iat[i]),
            int(df_feat["num_c_2"].iat[i]),  int(df_feat["num_db_2"].iat[i]),  int(df_feat["num_ox_2"].iat[i]),
            int(df_feat["num_c_3"].iat[i]),  int(df_feat["num_db_3"].iat[i]),  int(df_feat["num_ox_3"].iat[i]),
            int(df_feat["num_c_4"].iat[i]),  int(df_feat["num_db_4"].iat[i]),  int(df_feat["num_ox_4"].iat[i]),
            class_to_numchain=class_to_numchain,
        )
        for i in range(len(df_feat))
    ])
    print(f"  Pre-computed {len(_true_names_all):,} true names.")

    from utils import load_backbone_masses
    l3_eval_fn = make_l3_eval_fn(
        label_maps            = label_maps,
        class_le              = class_le,
        adduct_le             = adduct_le,
        backbone_masses       = load_backbone_masses(),
        class_to_numchain     = class_to_numchain,
        precmz_all            = precmz_all,
        row_num_chain         = row_num_chain,
        true_names_all        = _true_names_all,
        run_sum_comp_rules_fn = _p1m.run_sum_comp_rules,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        out_dir=out_dir,
        model_name=model_name,
        l3_eval_fn=l3_eval_fn,
    )
    trainer.train()

    # ── Save label maps and metadata for evaluation ───────────────────────────
    import pickle
    meta = {
        "label_maps":      label_maps,
        "n_classes":       n_classes,
        "pmz_stats":       pmz_stats,
        "row_num_chain":   row_num_chain,
    }
    meta_path = out_dir / "models" / "train_meta.pkl"
    with open(meta_path, "wb") as fh:
        pickle.dump(meta, fh, protocol=4)
    print(f"\nTrain metadata → {meta_path}")

    print(f"\nDone.  Model saved to {out_dir / 'models' / 'best.pt'}")


if __name__ == "__main__":
    main()
