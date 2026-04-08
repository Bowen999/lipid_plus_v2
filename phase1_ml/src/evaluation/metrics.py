"""
04_evaluate.py — Generate predictions on val/test sets; compute all metrics;
write evaluation files.

Inference pipeline
------------------
1. Predict adduct   from base spectral features (no adduct input).
2. Predict class    from base features + predicted adduct.
3. Sum composition  computed algebraically from backbone_masses + predicted exact mass;
                    rows with no mass match within SUM_COMP_PPM_TOL are labelled
                    "no_match" and their chain predictions are zeroed out.
4. Predict chain-1  from base + predicted adduct + rule totals (tc, tdb, tox).
5. Predict chain-2  from above + predicted chain-1 (chain conditioning).
6. Predict chain-3  from above + predicted chain-2.
7. Predict chain-4  from above + predicted chain-3.

Outputs
-------
    evaluation/val_predictions.csv
    evaluation/test_predictions.csv
    evaluation/val_metrics.json
    evaluation/test_metrics.json
    evaluation/val_class_confusion.csv
    evaluation/test_class_confusion.csv
    evaluation/evaluation_report.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DATA_DIR,
    EVALUATION_DIR,
    MODELS_DIR,
    ADDUCT_TABLE,
    SUM_COMP_PPM_TOL,
    adduct_to_neutral,
    find_sum_comp_candidates,
    load_backbone_masses,
    load_class_to_numchain,
    reconstruct_name,
)

# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_FILES = {
    "adduct": "xgb_adduct.joblib",
    "class":  "xgb_class.joblib",
    "nc1":    "xgb_nc1.joblib",
    "ndb1":   "xgb_ndb1.joblib",
    "nox1":   "xgb_nox1.joblib",
    "nc2":    "xgb_nc2.joblib",
    "ndb2":   "xgb_ndb2.joblib",
    "nox2":   "xgb_nox2.joblib",
    "nc3":    "xgb_nc3.joblib",
    "ndb3":   "xgb_ndb3.joblib",
    "nox3":   "xgb_nox3.joblib",
    "nc4":    "xgb_nc4.joblib",
    "ndb4":   "xgb_ndb4.joblib",
    "nox4":   "xgb_nox4.joblib",
}


def load_model(key: str):
    path = MODELS_DIR / MODEL_FILES[key]
    if not path.exists():
        print(f"  [WARN] Model not found: {path.name} — will use 0 predictions")
        return None
    return joblib.load(path)


def predict_with_model(
    model,
    X: np.ndarray,
    class_map: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run model.predict(X).  Remap encoded 0..K-1 labels back to original label
    space if class_map is provided (class_map[i] = original_label_for_index_i).

    Returns zeros if the model is None or has a stale feature count that does
    not match X (prevents crashes when a model was trained under a different
    feature schema).
    """
    if model is None:
        return np.zeros(len(X), dtype=np.int32)
    expected = getattr(model, "n_features_in_", X.shape[1])
    if X.shape[1] != expected:
        print(f"  [WARN] Feature mismatch: model expects {expected}, got {X.shape[1]}"
              " — returning zeros (model needs retraining)")
        return np.zeros(len(X), dtype=np.int32)
    preds = model.predict(X).astype(np.int32)
    if class_map is not None and len(class_map) > 0:
        preds = np.array([class_map[p] if p < len(class_map) else 0
                          for p in preds], dtype=np.int32)
    return preds


# ── Feature column helpers ─────────────────────────────────────────────────────
def get_base_feat_cols(df: pd.DataFrame) -> list[str]:
    """Base features: spectral bins + precursor_mz_norm + ion_mode_enc (no adduct)."""
    f_cols  = sorted(c for c in df.columns if c.startswith("F_"))
    nl_cols = sorted(c for c in df.columns if c.startswith("NL_"))
    return f_cols + nl_cols + ["precursor_mz_norm", "ion_mode_enc"]


# ── Class-map reconstruction ──────────────────────────────────────────────────
def build_class_maps(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    row_num_chain: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Reconstruct the label remapping arrays (sorted unique training labels) for
    every model, exactly matching what 03_train_xgboost.py built during training.
    class_map[i] = original_label_for_encoded_index_i
    """
    train_set = set(train_idx.tolist())
    all_mask  = np.ones(len(df), dtype=bool)
    mask2     = row_num_chain >= 2
    mask3     = row_num_chain >= 3
    mask4     = row_num_chain >= 4

    def sub_train(mask: np.ndarray) -> np.ndarray:
        return np.array([i for i in np.where(mask)[0] if i in train_set],
                        dtype=np.int32)

    maps: dict[str, np.ndarray] = {}

    # adduct model
    y = df["adduct_enc"].values.astype(np.int32)
    maps["adduct"] = np.sort(np.unique(y[sub_train(all_mask)]))

    # class model
    y = df["class_enc"].values.astype(np.int32)
    maps["class"] = np.sort(np.unique(y[sub_train(all_mask)]))

    for key, target, mask in [
        ("nc1",  "num_c_1",  all_mask),
        ("ndb1", "num_db_1", all_mask),
        ("nox1", "num_ox_1", all_mask),
        ("nc2",  "num_c_2",  mask2),
        ("ndb2", "num_db_2", mask2),
        ("nox2", "num_ox_2", mask2),
        ("nc3",  "num_c_3",  mask3),
        ("ndb3", "num_db_3", mask3),
        ("nox3", "num_ox_3", mask3),
        ("nc4",  "num_c_4",  mask4),
        ("ndb4", "num_db_4", mask4),
        ("nox4", "num_ox_4", mask4),
    ]:
        y = df[target].values.astype(np.int32)
        maps[key] = np.sort(np.unique(y[sub_train(mask)]))

    return maps


# ── Sum-composition rule engine ────────────────────────────────────────────────
def run_sum_comp_rules(
    pred_class_str: np.ndarray,
    pred_adduct_str: np.ndarray,
    precmz_arr: np.ndarray,       # precursor_mz values, aligned to split indices
    backbone_masses: dict[str, float],
    class_to_numchain: dict[str, int],
    ppm_tol: float = SUM_COMP_PPM_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each row determine (rule_nc, rule_ndb, rule_nox, sum_comp_status).

    status values:
      "matched"      — unique or best candidate found within ppm_tol
      "no_match"     — no candidate within ppm_tol
      "no_backbone"  — predicted class has no backbone mass
      "no_adduct"    — predicted adduct not in ADDUCT_TABLE
      "multi"        — multiple candidates (best match reported)

    Returns four arrays of length n_rows.
    """
    n = len(pred_class_str)
    rule_nc   = np.zeros(n, dtype=np.int16)
    rule_ndb  = np.zeros(n, dtype=np.int16)
    rule_nox  = np.zeros(n, dtype=np.int16)
    status    = np.empty(n, dtype=object)

    for i in range(n):
        cls      = str(pred_class_str[i])
        adduct   = str(pred_adduct_str[i])
        pmz      = float(precmz_arr[i])

        bm       = backbone_masses.get(cls)
        if bm is None:
            status[i] = "no_backbone"
            continue

        exact_m  = adduct_to_neutral(pmz, adduct)
        if exact_m is None:
            status[i] = "no_adduct"
            continue

        n_chain  = class_to_numchain.get(cls, 1)
        cands    = find_sum_comp_candidates(exact_m, bm, n_chain, ppm_tol=ppm_tol)

        if not cands:
            status[i] = "no_match"
        elif len(cands) == 1:
            status[i] = "matched"
        else:
            status[i] = "multi"

        if cands:
            rule_nc[i]  = cands[0][0]
            rule_ndb[i] = cands[0][1]
            rule_nox[i] = cands[0][2]

    return rule_nc, rule_ndb, rule_nox, status


# ── Main prediction pipeline ──────────────────────────────────────────────────
def predict_split(
    split_name: str,
    indices: np.ndarray,
    X_base_all: np.ndarray,       # base features, all rows
    precmz_all: np.ndarray,       # original precursor_mz, all rows
    df: pd.DataFrame,
    models: dict,
    class_maps: dict[str, np.ndarray],
    class_le,
    adduct_le,
    class_to_numchain: dict[str, int],
    row_num_chain: np.ndarray,
    backbone_masses: dict[str, float],
) -> pd.DataFrame:
    """
    Run the full hierarchical inference pipeline on `indices`.

    Feature matrices are built on the fly for this split's rows only,
    using predicted (not true) adduct/chain values at each stage.
    Rows where sum-composition fails are marked; chain predictions are
    zeroed out for "no_match" and "no_adduct" rows.
    """
    print(f"\nPredicting on {split_name} ({len(indices):,} rows) …")
    n = len(indices)

    # Extract split-local feature matrix (base only, no adduct)
    X_base = X_base_all[indices]                 # (n, 3102)
    precmz = precmz_all[indices]                 # (n,)

    # ── Step 1: Predict adduct ────────────────────────────────────
    print("  Step 1: adduct prediction …")
    pred_adduct_enc = predict_with_model(models["adduct"], X_base,
                                         class_maps["adduct"])
    pred_adduct_str = adduct_le.inverse_transform(pred_adduct_enc)

    true_adduct_enc = df["adduct_enc"].values[indices].astype(np.int32)
    true_adduct_str = adduct_le.inverse_transform(true_adduct_enc)

    # ── Step 2: Predict class (base + pred_adduct) ────────────────
    print("  Step 2: class prediction …")
    adduct_col  = pred_adduct_enc.reshape(-1, 1).astype(np.float32)
    X_cls       = np.concatenate([X_base, adduct_col], axis=1)  # (n, 3103)

    pred_class_enc = predict_with_model(models["class"], X_cls,
                                         class_maps["class"])
    pred_class_str = class_le.inverse_transform(pred_class_enc)

    true_class_enc = df["class_enc"].values[indices].astype(np.int32)
    true_class_str = class_le.inverse_transform(true_class_enc)

    # ── Step 3: Rule-based sum composition ───────────────────────
    print("  Step 3: rule-based sum composition …")
    rule_nc, rule_ndb, rule_nox, sc_status = run_sum_comp_rules(
        pred_class_str, pred_adduct_str, precmz,
        backbone_masses, class_to_numchain,
    )

    # Mask for rows where chain models should run
    chain_ok_mask = np.isin(sc_status, ["matched", "multi"])
    print(f"    matched / multi : {chain_ok_mask.sum():,} / {n:,}  "
          f"({100*chain_ok_mask.mean():.1f}%)")

    # ── Step 4: Predict chain-1 (X_cls + class + rule totals) ───
    # Predicted class_enc and rule-derived (total_c, total_db, total_ox) are
    # appended as conditioning features for ALL chain models — mirrors the
    # teacher-forcing used during training.
    # For no_match / no_adduct / no_backbone rows the totals are 0; those rows
    # are zeroed out at the end of this function anyway.
    print("  Step 4: chain-1 prediction …")
    class_col   = pred_class_enc.reshape(-1, 1).astype(np.float32)
    total_block = np.stack([rule_nc, rule_ndb, rule_nox], axis=1).astype(np.float32)
    X_ch1       = np.concatenate([X_cls, class_col, total_block], axis=1)  # (n, 3107)
    pred_nc1    = predict_with_model(models["nc1"],  X_ch1, class_maps["nc1"])
    pred_ndb1   = predict_with_model(models["ndb1"], X_ch1, class_maps["ndb1"])
    pred_nox1   = predict_with_model(models["nox1"], X_ch1, class_maps["nox1"])

    # ── Step 5: Predict chain-2 (X_ch1 + chain-1) ────────────────
    print("  Step 5: chain-2 prediction …")
    c1_block  = np.stack([pred_nc1, pred_ndb1, pred_nox1], axis=1).astype(np.float32)
    X_ch2     = np.concatenate([X_ch1, c1_block], axis=1)  # (n, 3110)
    pred_nc2  = predict_with_model(models["nc2"],  X_ch2, class_maps["nc2"])
    pred_ndb2 = predict_with_model(models["ndb2"], X_ch2, class_maps["ndb2"])
    pred_nox2 = predict_with_model(models["nox2"], X_ch2, class_maps["nox2"])

    # ── Step 6: Predict chain-3 (X_ch2 + chain-2) ────────────────
    print("  Step 6: chain-3 prediction …")
    c2_block  = np.stack([pred_nc2, pred_ndb2, pred_nox2], axis=1).astype(np.float32)
    X_ch3     = np.concatenate([X_ch2, c2_block], axis=1)  # (n, 3113)
    pred_nc3  = predict_with_model(models["nc3"],  X_ch3, class_maps["nc3"])
    pred_ndb3 = predict_with_model(models["ndb3"], X_ch3, class_maps["ndb3"])
    pred_nox3 = predict_with_model(models["nox3"], X_ch3, class_maps["nox3"])

    # ── Step 7: Predict chain-4 (X_ch3 + chain-3) ────────────────
    print("  Step 7: chain-4 prediction …")
    c3_block  = np.stack([pred_nc3, pred_ndb3, pred_nox3], axis=1).astype(np.float32)
    X_ch4     = np.concatenate([X_ch3, c3_block], axis=1)  # (n, 3116)
    pred_nc4  = predict_with_model(models["nc4"],  X_ch4, class_maps["nc4"])
    pred_ndb4 = predict_with_model(models["ndb4"], X_ch4, class_maps["ndb4"])
    pred_nox4 = predict_with_model(models["nox4"], X_ch4, class_maps["nox4"])

    # ── Zero out chain predictions for no-match rows ──────────────
    no_chain  = ~chain_ok_mask
    for arr in (pred_nc1, pred_ndb1, pred_nox1,
                pred_nc2, pred_ndb2, pred_nox2,
                pred_nc3, pred_ndb3, pred_nox3,
                pred_nc4, pred_ndb4, pred_nox4):
        arr[no_chain] = 0

    # Apply predicted num_chain: zero out chains beyond predicted class's chain count
    pred_n_chain = np.array([class_to_numchain.get(c, 1)
                              for c in pred_class_str], dtype=np.int8)
    for arr in (pred_nc2, pred_ndb2, pred_nox2):
        arr[pred_n_chain < 2] = 0
    for arr in (pred_nc3, pred_ndb3, pred_nox3):
        arr[pred_n_chain < 3] = 0
    for arr in (pred_nc4, pred_ndb4, pred_nox4):
        arr[pred_n_chain < 4] = 0

    # ── True chain descriptors ────────────────────────────────────
    get = lambda col: df[col].values[indices].astype(np.int32)
    true_nc1  = get("num_c_1");  true_ndb1 = get("num_db_1"); true_nox1 = get("num_ox_1")
    true_nc2  = get("num_c_2");  true_ndb2 = get("num_db_2"); true_nox2 = get("num_ox_2")
    true_nc3  = get("num_c_3");  true_ndb3 = get("num_db_3"); true_nox3 = get("num_ox_3")
    true_nc4  = get("num_c_4");  true_ndb4 = get("num_db_4"); true_nox4 = get("num_ox_4")

    true_total_c   = true_nc1  + true_nc2  + true_nc3  + true_nc4
    true_total_db  = true_ndb1 + true_ndb2 + true_ndb3 + true_ndb4
    true_total_ox  = true_nox1 + true_nox2 + true_nox3 + true_nox4
    pred_total_c   = pred_nc1  + pred_nc2  + pred_nc3  + pred_nc4
    pred_total_db  = pred_ndb1 + pred_ndb2 + pred_ndb3 + pred_ndb4
    pred_total_ox  = pred_nox1 + pred_nox2 + pred_nox3 + pred_nox4

    # ── Reconstruct canonical names ───────────────────────────────
    print("  Reconstructing names …")
    pred_names, true_names = [], []
    for i in range(n):
        pred_names.append(reconstruct_name(
            pred_class_str[i],
            int(pred_nc1[i]),  int(pred_ndb1[i]),  int(pred_nox1[i]),
            int(pred_nc2[i]),  int(pred_ndb2[i]),  int(pred_nox2[i]),
            int(pred_nc3[i]),  int(pred_ndb3[i]),  int(pred_nox3[i]),
            int(pred_nc4[i]),  int(pred_ndb4[i]),  int(pred_nox4[i]),
            class_to_numchain=class_to_numchain,
        ))
        true_names.append(reconstruct_name(
            true_class_str[i],
            int(true_nc1[i]),  int(true_ndb1[i]),  int(true_nox1[i]),
            int(true_nc2[i]),  int(true_ndb2[i]),  int(true_nox2[i]),
            int(true_nc3[i]),  int(true_ndb3[i]),  int(true_nox3[i]),
            int(true_nc4[i]),  int(true_ndb4[i]),  int(true_nox4[i]),
            class_to_numchain=class_to_numchain,
        ))

    # ── Assemble output DataFrame ─────────────────────────────────
    out = pd.DataFrame({
        "row_index":         indices,
        "true_name":         true_names,
        "pred_name":         pred_names,
        "true_adduct":       true_adduct_str,
        "pred_adduct":       pred_adduct_str,
        "true_class":        true_class_str,
        "pred_class":        pred_class_str,
        "sum_comp_status":   sc_status,
        "rule_nc":           rule_nc,
        "rule_ndb":          rule_ndb,
        "rule_nox":          rule_nox,
        "true_nc1":   true_nc1,  "pred_nc1":  pred_nc1,
        "true_ndb1":  true_ndb1, "pred_ndb1": pred_ndb1,
        "true_nox1":  true_nox1, "pred_nox1": pred_nox1,
        "true_nc2":   true_nc2,  "pred_nc2":  pred_nc2,
        "true_ndb2":  true_ndb2, "pred_ndb2": pred_ndb2,
        "true_nox2":  true_nox2, "pred_nox2": pred_nox2,
        "true_nc3":   true_nc3,  "pred_nc3":  pred_nc3,
        "true_ndb3":  true_ndb3, "pred_ndb3": pred_ndb3,
        "true_nox3":  true_nox3, "pred_nox3": pred_nox3,
        "true_nc4":   true_nc4,  "pred_nc4":  pred_nc4,
        "true_ndb4":  true_ndb4, "pred_ndb4": pred_ndb4,
        "true_nox4":  true_nox4, "pred_nox4": pred_nox4,
        "true_total_c":  true_total_c,  "pred_total_c":  pred_total_c,
        "true_total_db": true_total_db, "pred_total_db": pred_total_db,
        "true_total_ox": true_total_ox, "pred_total_ox": pred_total_ox,
    })
    return out


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(
    pred_df: pd.DataFrame,
    class_to_numchain: dict[str, int],
) -> dict:
    n  = len(pred_df)
    tc = pred_df["true_class"].values
    pc = pred_df["pred_class"].values

    class_ok = (tc == pc)
    level0   = float(class_ok.mean())

    adduct_ok = (pred_df["true_adduct"].values == pred_df["pred_adduct"].values)
    adduct_acc = float(adduct_ok.mean())

    # Level 1: class + sum composition
    sum_ok = (
        (pred_df["true_total_c"]  == pred_df["pred_total_c"]) &
        (pred_df["true_total_db"] == pred_df["pred_total_db"]) &
        (pred_df["true_total_ox"] == pred_df["pred_total_ox"])
    ).values
    level1 = float((class_ok & sum_ok).mean())

    # Level 2: class + full per-chain multiset (position-agnostic)
    def _chain_multiset(row, prefix: str) -> frozenset:
        tuples = []
        for i in range(1, 5):
            nc  = row[f"{prefix}_nc{i}"]
            ndb = row[f"{prefix}_ndb{i}"]
            nox = row[f"{prefix}_nox{i}"]
            if nc != 0:
                tuples.append((nc, ndb, nox))
        return frozenset(tuples)

    chain_ok = np.array([
        _chain_multiset(pred_df.iloc[i], "true") ==
        _chain_multiset(pred_df.iloc[i], "pred")
        for i in range(n)
    ])
    level2 = float((class_ok & chain_ok).mean())

    # Level 3: exact name match
    level3 = float((pred_df["true_name"].values == pred_df["pred_name"].values).mean())

    # Sum-composition rule stats
    sc_counts = pred_df["sum_comp_status"].value_counts().to_dict()

    # MAE for chain descriptors
    c1_nc_mae  = float(np.abs(pred_df["true_nc1"]  - pred_df["pred_nc1"]).mean())
    c1_ndb_mae = float(np.abs(pred_df["true_ndb1"] - pred_df["pred_ndb1"]).mean())
    c1_nox_mae = float(np.abs(pred_df["true_nox1"] - pred_df["pred_nox1"]).mean())

    def chain_mae(col: str, mask: np.ndarray) -> float:
        sub = pred_df[mask]
        return float(np.abs(sub[f"true_{col}"] - sub[f"pred_{col}"]).mean()) \
            if mask.any() else float("nan")

    mask2 = pred_df["true_class"].map(lambda c: class_to_numchain.get(c, 1) >= 2).values
    mask3 = pred_df["true_class"].map(lambda c: class_to_numchain.get(c, 1) >= 3).values

    # Per-class Level 0 / Level 1
    per_class_l0: dict[str, float] = {}
    per_class_l1: dict[str, float] = {}
    for cls in np.unique(tc):
        m = tc == cls
        per_class_l0[cls] = round(float(class_ok[m].mean()), 4)
        per_class_l1[cls] = round(float((class_ok & sum_ok)[m].mean()), 4)

    def _r(v):
        return round(v, 4) if not np.isnan(v) else None

    return {
        "n_samples":                       n,
        "adduct_accuracy":                 round(adduct_acc, 4),
        "level0_class_accuracy":           round(level0,  4),
        "level1_sum_composition_accuracy": round(level1,  4),
        "level2_full_chain_accuracy":      round(level2,  4),
        "level3_name_exact_match":         round(level3,  4),
        "sum_comp_status_counts":          sc_counts,
        "chain1_nc_mae":   _r(c1_nc_mae),
        "chain1_ndb_mae":  _r(c1_ndb_mae),
        "chain1_nox_mae":  _r(c1_nox_mae),
        "chain2_nc_mae":   _r(chain_mae("nc2",  mask2)),
        "chain2_ndb_mae":  _r(chain_mae("ndb2", mask2)),
        "chain2_nox_mae":  _r(chain_mae("nox2", mask2)),
        "chain3_nc_mae":   _r(chain_mae("nc3",  mask3)),
        "chain3_ndb_mae":  _r(chain_mae("ndb3", mask3)),
        "chain3_nox_mae":  _r(chain_mae("nox3", mask3)),
        "per_class_level0_accuracy": per_class_l0,
        "per_class_level1_accuracy": per_class_l1,
    }


# ── Confusion matrix ───────────────────────────────────────────────────────────
def write_confusion_matrix(pred_df: pd.DataFrame, out_path: Path, class_le) -> None:
    all_classes = list(class_le.classes_)
    cm = confusion_matrix(pred_df["true_class"].values,
                          pred_df["pred_class"].values,
                          labels=all_classes)
    pd.DataFrame(cm, index=all_classes, columns=all_classes
                 ).rename_axis("true_class").to_csv(out_path)
    print(f"  Confusion matrix saved → {out_path}")


# ── Markdown report ────────────────────────────────────────────────────────────
def write_report(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_m: dict,
    test_m: dict,
    class_to_numchain: dict[str, int],
    train_size: int,
    val_size: int,
    test_size: int,
    out_path: Path,
) -> None:
    from collections import Counter
    val_counts  = Counter(val_df["true_class"].tolist())
    test_counts = Counter(test_df["true_class"].tolist())
    top_classes = sorted(val_counts, key=lambda c: -(val_counts[c] + test_counts.get(c, 0)))[:20]

    scalar_keys = [
        ("Adduct accuracy",          "adduct_accuracy"),
        ("Level 0 — Class",          "level0_class_accuracy"),
        ("Level 1 — Sum composition","level1_sum_composition_accuracy"),
        ("Level 2 — Full chain",     "level2_full_chain_accuracy"),
        ("Level 3 — Exact name",     "level3_name_exact_match"),
    ]

    lines = [
        "# Phase 1 XGBoost Evaluation Report",
        "",
        "## Dataset Summary",
        "",
        "| Split | Rows |",
        "|-------|------|",
        f"| Train | {train_size:,} |",
        f"| Val   | {val_size:,} |",
        f"| Test  | {test_size:,} |",
        f"| Total | {train_size + val_size + test_size:,} |",
        "",
        "| Detail | Value |",
        "|--------|-------|",
        f"| Base features    | 3,102 (1550 F + 1550 NL + precursor_mz_norm + ion_mode_enc) |",
        f"| Models           | 14 (adduct + class + 12 chain) |",
        f"| Lipid classes    | 78 |",
        f"| Sum comp PPM tol | {SUM_COMP_PPM_TOL} |",
        "",
        "## Pipeline",
        "",
        "1. **Adduct** predicted from base spectral features.",
        "2. **Class** predicted from base + predicted adduct.",
        "3. **Sum composition** (total_nc, total_db, total_ox) derived algebraically",
        "   from backbone masses + precursor mass within the PPM tolerance.",
        "   Rows without a mass match are labelled `no_match` / `no_adduct` /",
        "   `no_backbone`; their chain predictions are zeroed out.",
        "4. **Chain-1** predicted from base + predicted adduct + rule totals (total_c, total_db, total_ox).",
        "5. **Chain-2** predicted from above + predicted chain-1 (chain conditioning).",
        "6. **Chain-3** predicted from above + predicted chain-2.",
        "7. **Chain-4** predicted from above + predicted chain-3.",
        "",
        "## Top-Level Metrics",
        "",
        "| Metric | Val | Test |",
        "|--------|-----|------|",
    ]
    for label, k in scalar_keys:
        vv = val_m.get(k, "N/A")
        tv = test_m.get(k, "N/A")
        lines.append(f"| {label} | {vv} | {tv} |")

    # Sum comp status breakdown
    lines += ["", "## Sum Composition Status", ""]
    all_statuses = sorted(
        set(val_m["sum_comp_status_counts"]) | set(test_m["sum_comp_status_counts"])
    )
    lines += ["| Status | Val | Test |", "|--------|-----|------|"]
    for s in all_statuses:
        vv = val_m["sum_comp_status_counts"].get(s, 0)
        tv = test_m["sum_comp_status_counts"].get(s, 0)
        lines.append(f"| {s} | {vv:,} | {tv:,} |")

    lines += [
        "",
        "## Per-Class Breakdown (top 20)",
        "",
        "| Class | #Val | #Test | Val L0 | Val L1 | Test L0 | Test L1 |",
        "|-------|------|-------|--------|--------|---------|---------|",
    ]
    for cls in top_classes:
        nv  = val_counts.get(cls, 0)
        nt  = test_counts.get(cls, 0)
        vl0 = val_m["per_class_level0_accuracy"].get(cls, 0.0)
        vl1 = val_m["per_class_level1_accuracy"].get(cls, 0.0)
        tl0 = test_m["per_class_level0_accuracy"].get(cls, 0.0)
        tl1 = test_m["per_class_level1_accuracy"].get(cls, 0.0)
        lines.append(f"| {cls} | {nv} | {nt} | {vl0:.4f} | {vl1:.4f} | {tl0:.4f} | {tl1:.4f} |")

    lines += [
        "",
        "## Chain Descriptor MAE",
        "",
        "| Chain | Target | Val MAE | Test MAE |",
        "|-------|--------|---------|---------|",
    ]
    for chain_label, nc_k, ndb_k, nox_k in [
        ("Chain 1", "chain1_nc_mae",  "chain1_ndb_mae",  "chain1_nox_mae"),
        ("Chain 2", "chain2_nc_mae",  "chain2_ndb_mae",  "chain2_nox_mae"),
        ("Chain 3", "chain3_nc_mae",  "chain3_ndb_mae",  "chain3_nox_mae"),
    ]:
        for tgt, key in [("nc", nc_k), ("ndb", ndb_k), ("nox", nox_k)]:
            vv = val_m.get(key)
            tv = test_m.get(key)
            lines.append(f"| {chain_label} | {tgt} | "
                         f"{'N/A' if vv is None else f'{vv:.4f}'} | "
                         f"{'N/A' if tv is None else f'{tv:.4f}'} |")

    lines += [
        "",
        f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Evaluation report saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 62)
    print("04_evaluate.py — Evaluation")
    print("=" * 62)

    # ── Load feature matrix ───────────────────────────────────────
    feat_path = DATA_DIR / "lipid_ms2_features.parquet"
    print(f"\nLoading features from:\n  {feat_path}")
    df = pd.read_parquet(feat_path, engine="pyarrow")

    # ── Load split indices ────────────────────────────────────────
    val_idx   = np.load(DATA_DIR / "split_val.npy")
    test_idx  = np.load(DATA_DIR / "split_test.npy")
    train_idx = np.load(DATA_DIR / "split_train.npy")

    # ── Load encoders + metadata ──────────────────────────────────
    class_le          = joblib.load(MODELS_DIR / "class_encoder.joblib")
    adduct_le         = joblib.load(MODELS_DIR / "adduct_encoder.joblib")
    class_to_numchain = load_class_to_numchain()
    backbone_masses   = load_backbone_masses()

    # ── Row num_chain ─────────────────────────────────────────────
    class_names   = class_le.inverse_transform(df["class_enc"].values)
    row_num_chain = np.array([class_to_numchain.get(c, 1) for c in class_names],
                              dtype=np.int8)

    # ── Build base feature matrix (all rows) ─────────────────────
    base_feat_cols = get_base_feat_cols(df)
    print(f"\nBase feature columns: {len(base_feat_cols)}  (no adduct_enc)")
    X_base_all = df[base_feat_cols].values.astype(np.float32)

    # ── Recover original precursor_mz from normalised value ──────
    pmz_stats = np.load(MODELS_DIR / "precursor_mz_stats.npy")
    pmz_mean, pmz_std = float(pmz_stats[0]), float(pmz_stats[1])
    precmz_all = df["precursor_mz_norm"].values * pmz_std + pmz_mean

    # ── Load models ───────────────────────────────────────────────
    print("\nLoading models …")
    models = {k: load_model(k) for k in MODEL_FILES}

    # ── Build class maps ──────────────────────────────────────────
    class_maps = build_class_maps(df, train_idx, row_num_chain)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run prediction for each split ────────────────────────────
    val_pred_df  = predict_split(
        "val",  val_idx,  X_base_all, precmz_all, df, models, class_maps,
        class_le, adduct_le, class_to_numchain, row_num_chain, backbone_masses,
    )
    test_pred_df = predict_split(
        "test", test_idx, X_base_all, precmz_all, df, models, class_maps,
        class_le, adduct_le, class_to_numchain, row_num_chain, backbone_masses,
    )

    # ── Save predictions ──────────────────────────────────────────
    val_pred_df.to_csv(EVALUATION_DIR  / "val_predictions.csv",  index=False)
    test_pred_df.to_csv(EVALUATION_DIR / "test_predictions.csv", index=False)
    print(f"\nSaved val_predictions.csv  ({len(val_pred_df):,} rows)")
    print(f"Saved test_predictions.csv ({len(test_pred_df):,} rows)")

    # ── Compute metrics ───────────────────────────────────────────
    print("\nComputing metrics …")
    val_metrics  = compute_metrics(val_pred_df,  class_to_numchain)
    test_metrics = compute_metrics(test_pred_df, class_to_numchain)

    with open(EVALUATION_DIR / "val_metrics.json",  "w") as fh:
        json.dump(val_metrics,  fh, indent=2)
    with open(EVALUATION_DIR / "test_metrics.json", "w") as fh:
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
    write_confusion_matrix(val_pred_df,  EVALUATION_DIR / "val_class_confusion.csv",  class_le)
    write_confusion_matrix(test_pred_df, EVALUATION_DIR / "test_class_confusion.csv", class_le)

    # ── Evaluation report ─────────────────────────────────────────
    write_report(
        val_pred_df, test_pred_df,
        val_metrics, test_metrics,
        class_to_numchain,
        train_size=len(train_idx),
        val_size=len(val_idx),
        test_size=len(test_idx),
        out_path=EVALUATION_DIR / "evaluation_report.md",
    )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[DONE] {ts}")


if __name__ == "__main__":
    main()
