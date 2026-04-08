"""
02_train.py — Train all 14 models for a single model family.

Usage:
    python scripts/02_train.py --model lightgbm
    python scripts/02_train.py --model random_forest --quick
    python scripts/02_train.py --model xgboost        # retrain XGBoost

Model training order and feature sets
--------------------------------------
    adduct           : adduct type  (base spectral features only)
    class            : lipid class  (base + true adduct — teacher forced)
    nc1/ndb1/nox1    : chain-1      (base + adduct + class + sum-comp totals)
    nc2/ndb2/nox2    : chain-2      (above + chain-1 true labels)
    nc3/ndb3/nox3    : chain-3      (above + chain-2 true labels)
    nc4/ndb4/nox4    : chain-4      (above + chain-3 true labels)

Models are saved to:
    outputs/{model_name}/models/{prefix}_{target}.joblib

Training log is saved to:
    outputs/{model_name}/evaluation/training_log.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
PHASE1_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PHASE1_ROOT / "src"))

from utils import (          # noqa: E402
    DATA_DIR, SPLITS_DIR, SHARED_DIR, OUTPUTS_DIR,
    FeatureSet, get_base_feat_cols,
    load_class_to_numchain,
)

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_PREFIXES: dict[str, str] = {
    "xgboost":         "xgb",
    "lightgbm":        "lgb",
    "random_forest":   "rf",
    "decision_tree":   "dt",
    "random_baseline": "baseline",
}

# Models that use early stopping (need X_val during fit)
EARLY_STOPPING_MODELS = {"xgboost", "lightgbm"}


def make_model(model_name: str, target_name: str, config: dict):
    """Instantiate the appropriate BaseLipidModel subclass."""
    if model_name == "xgboost":
        from models.xgboost import XGBoostModel
        return XGBoostModel(target_name, config)
    elif model_name == "lightgbm":
        from models.lightgbm import LightGBMModel
        return LightGBMModel(target_name, config)
    elif model_name == "random_forest":
        from models.random_forest import RandomForestModel
        return RandomForestModel(target_name, config)
    elif model_name == "decision_tree":
        from models.decision_tree import DecisionTreeModel
        return DecisionTreeModel(target_name, config)
    elif model_name == "random_baseline":
        from models.random_baseline import RandomBaselineModel
        return RandomBaselineModel(target_name, config)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


# ── Notifications ─────────────────────────────────────────────────────────────
NTFY_TOPIC = "lipid-plus"


def notify(title: str, message: str) -> None:
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    try:
        req = urllib.request.Request(
            url, data=message.encode("utf-8"),
            headers={"Title": title, "Priority": "default", "Tags": "tada"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
        print(f"  [ntfy] Notification sent → {url}")
    except Exception as exc:
        print(f"  [ntfy] Could not send notification: {exc}")


# ── Split-index helper ────────────────────────────────────────────────────────
_split_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def get_split_indices(
    mask: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    key = id(mask)
    if key not in _split_cache:
        all_idx = np.where(mask)[0].astype(np.int32)
        tr = all_idx[np.isin(all_idx, train_idx)]
        vl = all_idx[np.isin(all_idx, val_idx)]
        _split_cache[key] = (tr, vl)
    return _split_cache[key]


# ── Per-model trainer ─────────────────────────────────────────────────────────
MIN_TRAIN_SAMPLES = 50


def train_one(
    model_name: str,
    model_key: str,
    target: str,
    X_model: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    mask: np.ndarray,
    config: dict,
    out_dir: Path,
    prefix: str,
    log_lines: list[str],
    model_no: int,
    total_models: int,
) -> None:
    bar = f"[{model_no:>2}/{total_models}]"

    tr_idx, vl_idx = get_split_indices(mask, train_idx, val_idx)

    if len(tr_idx) < MIN_TRAIN_SAMPLES:
        msg = f"{bar} [SKIP] {model_key}: only {len(tr_idx)} training samples."
        print(msg);  log_lines.append(msg)
        return

    X_tr = X_model[tr_idx]
    X_vl = X_model[vl_idx]
    y_all = df[target].values.astype(np.int32)
    y_tr  = y_all[tr_idx]
    y_vl  = y_all[vl_idx]

    n_cls = len(np.unique(y_tr))
    print(f"\n{'═'*62}")
    print(f"{bar} Training : {model_name}/{model_key}")
    print(f"    target        : {target}")
    print(f"    train / val   : {len(X_tr):,} / {len(X_vl):,}")
    print(f"    unique classes: {n_cls}")
    print(f"    features      : {X_tr.shape[1]}")
    print(f"    started at    : {datetime.now().strftime('%H:%M:%S')}")

    sample_weight = compute_sample_weight("balanced", y_tr)

    model = make_model(model_name, target, config)

    t0 = time.time()
    if model_name in EARLY_STOPPING_MODELS:
        meta = model.fit(X_tr, y_tr, X_vl, y_vl, sample_weight=sample_weight)
    else:
        meta = model.fit(X_tr, y_tr, sample_weight=sample_weight)
    elapsed = time.time() - t0

    out_path = out_dir / f"{prefix}_{model_key}.joblib"
    model.save(out_path)
    print(f"  → Saved: {out_path.name}  ({elapsed/60:.1f} min)")

    best_iter = meta.get("best_iteration", "-")
    log_lines.append(
        f"{model_key:<20}  target={target:<12}  "
        f"train={len(X_tr):>7,}  val={len(X_vl):>6,}  "
        f"feats={X_tr.shape[1]:>4}  best_iter={str(best_iter):>5}  "
        f"time={elapsed:.0f}s"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        choices=list(MODEL_PREFIXES),
        help="Model family to train",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke-test: subsample ~5 k rows, minimal hyperparams",
    )
    args = parser.parse_args()

    model_name = args.model
    prefix     = MODEL_PREFIXES[model_name]

    print("=" * 62)
    print(f"02_train.py — {model_name} Training")
    if args.quick:
        print("  *** QUICK / SMOKE-TEST MODE ***")
    print("=" * 62)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Load config ───────────────────────────────────────────────
    config_path = PHASE1_ROOT / "configs" / f"{model_name}.json"
    with open(config_path) as fh:
        config = json.load(fh)

    # Apply quick-mode overrides
    if args.quick:
        p = config.setdefault("params", {})
        if model_name in EARLY_STOPPING_MODELS:
            p["n_estimators"]          = 50
            p["early_stopping_rounds"] = 10
        print(f"[QUICK] config overrides: {p}")

    # ── Load feature matrix ───────────────────────────────────────
    feat_path = DATA_DIR / "lipid_ms2_features.parquet"
    print(f"\nLoading features from:\n  {feat_path}")
    df = pd.read_parquet(feat_path, engine="pyarrow")
    print(f"Feature matrix: {df.shape}")

    # ── Load split indices ────────────────────────────────────────
    train_idx = np.load(SPLITS_DIR / "split_train.npy")
    val_idx   = np.load(SPLITS_DIR / "split_val.npy")

    if args.quick:
        rng       = np.random.default_rng(42)
        train_idx = rng.choice(train_idx, size=min(3_500, len(train_idx)), replace=False)
        val_idx   = rng.choice(val_idx,   size=min(1_500, len(val_idx)),   replace=False)
        print(f"[QUICK] train: {len(train_idx):,}  val: {len(val_idx):,}")
    else:
        print(f"Split: train={len(train_idx):,}  val={len(val_idx):,}")

    # ── Load class_to_numchain ────────────────────────────────────
    class_to_numchain = load_class_to_numchain()
    class_le          = joblib.load(SHARED_DIR / "class_encoder.joblib")
    class_names       = class_le.inverse_transform(df["class_enc"].values)
    row_num_chain     = np.array(
        [class_to_numchain.get(c, 1) for c in class_names], dtype=np.int8
    )

    # ── Build feature matrices ────────────────────────────────────
    print("\nBuilding feature matrices …")
    base_feat_cols = get_base_feat_cols(df)
    print(f"  Base feature count : {len(base_feat_cols)}  (no adduct_enc)")
    fs = FeatureSet(df, base_feat_cols)

    # ── Row masks ─────────────────────────────────────────────────
    all_mask    = np.ones(len(df), dtype=bool)
    mask_chain2 = row_num_chain >= 2
    mask_chain3 = row_num_chain >= 3
    mask_chain4 = row_num_chain >= 4

    print(f"\nRow masks:")
    print(f"  all     : {all_mask.sum():,}")
    print(f"  chain≥2 : {mask_chain2.sum():,}")
    print(f"  chain≥3 : {mask_chain3.sum():,}")
    print(f"  chain≥4 : {mask_chain4.sum():,}")

    # ── Model schedule ─────────────────────────────────────────────
    # (model_key, target_col, row_mask, feature_key)
    schedule = [
        ("adduct", "adduct_enc", all_mask,    "base"),
        ("class",  "class_enc",  all_mask,    "cls"),
        ("nc1",    "num_c_1",    all_mask,    "chain1"),
        ("ndb1",   "num_db_1",   all_mask,    "chain1"),
        ("nox1",   "num_ox_1",   all_mask,    "chain1"),
        ("nc2",    "num_c_2",    mask_chain2, "ch2"),
        ("ndb2",   "num_db_2",   mask_chain2, "ch2"),
        ("nox2",   "num_ox_2",   mask_chain2, "ch2"),
        ("nc3",    "num_c_3",    mask_chain3, "ch3"),
        ("ndb3",   "num_db_3",   mask_chain3, "ch3"),
        ("nox3",   "num_ox_3",   mask_chain3, "ch3"),
        ("nc4",    "num_c_4",    mask_chain4, "ch4"),
        ("ndb4",   "num_db_4",   mask_chain4, "ch4"),
        ("nox4",   "num_ox_4",   mask_chain4, "ch4"),
    ]
    total_models = len(schedule)

    # ── Output directories ────────────────────────────────────────
    out_dir  = OUTPUTS_DIR / model_name / "models"
    eval_dir = OUTPUTS_DIR / model_name / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = [
        f"Training log — {model_name} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Quick mode: {args.quick}",
        "=" * 90,
        f"{'Model':<20}  {'target':<12}  {'train':>7}  {'val':>6}  {'feats':>5}"
        f"  {'best_iter':>9}  {'time':>8}",
        "─" * 90,
    ]

    # ── Training loop ─────────────────────────────────────────────
    wall_start   = time.time()
    model_times: list[float] = []
    completed    = 0

    for model_no, (model_key, target, mask, feat_key) in enumerate(schedule, start=1):
        X_model = fs.get(feat_key)
        t_start = time.time()
        train_one(
            model_name, model_key, target,
            X_model, df, train_idx, val_idx, mask,
            config, out_dir, prefix, log_lines,
            model_no, total_models,
        )
        model_times.append(time.time() - t_start)
        completed += 1

        avg_t     = sum(model_times) / len(model_times)
        remaining = total_models - completed
        eta_secs  = avg_t * remaining
        eta_str   = f"{eta_secs/60:.0f} min" if eta_secs >= 60 else f"{eta_secs:.0f} s"
        wall_so_far = (time.time() - wall_start) / 60
        print(f"\n  Progress: {completed}/{total_models} done  |  "
              f"ETA ~{eta_str}  |  Wall: {wall_so_far:.1f} min")

    # ── Save training log ─────────────────────────────────────────
    log_path = eval_dir / "training_log.txt"
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(f"\nTraining log saved → {log_path}")

    total_wall = time.time() - wall_start
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = (
        f"All {total_models} {model_name} models trained in "
        f"{total_wall/60:.1f} min."
    )
    print(f"\n{summary}")
    print(f"\n[DONE] {ts}")

    if not args.quick:
        notify(
            title   = f"lipid-plus {model_name} training complete",
            message = f"{ts}\n{summary}",
        )


if __name__ == "__main__":
    main()
