"""
inference.py — DL inference pipeline producing the same DataFrame format
as phase1_ml's predict_split() so compute_metrics() can be reused.

Cascade at inference (mirrors phase1_ml):
  Step 1  Predict adduct
  Step 2  Predict class
  Step 3  Rule-based sum composition
  Step 4  Predict chain-1
  Step 5  Predict chain-2
  Step 6  Predict chain-3
  Step 7  Predict chain-4

For MLP / Transformer: single forward pass (no internal conditioning).
For CNN: adduct prediction → condition via adduct embedding → other heads.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils import (
    TARGETS, CHAIN_MIN,
    reconstruct_name,
    DATA_STARVED_CLASSES,
    apply_chain_rules,
)

# Re-use rule engine from phase1_ml
# Load via importlib to avoid sys.modules collision between phase2_dl/src/utils.py
# and phase1_ml/src/utils.py (both named "utils").
import importlib.util as _ilu

_REPO_ROOT  = Path(__file__).resolve().parents[3]
_P1_SRC     = _REPO_ROOT / "phase1_ml" / "src"
_P1_EVAL    = _P1_SRC / "evaluation"

# Step 1: load phase1_ml utils with a private name
_p1u_spec = _ilu.spec_from_file_location("_p1_utils", str(_P1_SRC / "utils.py"))
_p1u_mod  = _ilu.module_from_spec(_p1u_spec)
sys.modules["_p1_utils"] = _p1u_mod
_p1u_spec.loader.exec_module(_p1u_mod)   # type: ignore[union-attr]

# Step 2: temporarily expose it as "utils" so metrics.py can import it
_prev_utils = sys.modules.get("utils")
sys.modules["utils"] = _p1u_mod

# Step 3: load metrics.py
_m_spec = _ilu.spec_from_file_location("_p1_eval_metrics", str(_P1_EVAL / "metrics.py"))
_p1_metrics = _ilu.module_from_spec(_m_spec)
_m_spec.loader.exec_module(_p1_metrics)  # type: ignore[union-attr]

# Step 4: restore original "utils" entry
if _prev_utils is None:
    sys.modules.pop("utils", None)
else:
    sys.modules["utils"] = _prev_utils

run_sum_comp_rules = _p1_metrics.run_sum_comp_rules  # noqa: E402

CHAIN_TARGETS = [
    "num_c_1", "num_db_1", "num_ox_1",
    "num_c_2", "num_db_2", "num_ox_2",
    "num_c_3", "num_db_3", "num_ox_3",
    "num_c_4", "num_db_4", "num_ox_4",
]


@torch.no_grad()
def predict_split_dl(
    split_name: str,
    dataset,                           # MLPDataset / CNNDataset / TransformerDataset
    model: nn.Module,
    model_type: str,                   # "mlp", "cnn", "transformer"
    label_maps: dict[str, np.ndarray],
    class_le,
    adduct_le,
    class_to_numchain: dict[str, int],
    row_num_chain: np.ndarray,
    backbone_masses: dict[str, float],
    precmz_all: np.ndarray,            # original precursor_mz for ALL rows in dataset
    device: torch.device,
    batch_size: int = 256,
    chain_rule_tol_da: float = 0.01,
) -> pd.DataFrame:
    """
    Run the full hierarchical inference cascade on `dataset`.
    Each sample in `dataset` includes `orig_idx` identifying its global row.

    Returns a DataFrame compatible with phase1_ml's compute_metrics().
    """
    n_total = len(dataset)
    print(f"\nPredicting on {split_name} ({n_total:,} rows) …")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    model.eval()
    all_probs: dict[str, list[np.ndarray]] = {t: [] for t in TARGETS}
    all_orig_idx: list[int] = []

    # ── Batch inference ───────────────────────────────────────────────────────
    for batch in loader:
        orig_idx = batch["orig_idx"].numpy()
        all_orig_idx.extend(orig_idx.tolist())

        inputs = {k: v.to(device) for k, v in batch.items()
                  if k not in ("labels", "orig_idx")}

        if model_type == "cnn":
            logits = model.forward_inference(
                inputs["spectrum"], inputs["precursor_mz"]
            )
        else:
            keys   = getattr(model, "INPUT_KEYS", list(inputs.keys()))
            logits = model(**{k: inputs[k] for k in keys if k in inputs})

        for t in TARGETS:
            all_probs[t].append(logits[t].softmax(dim=1).cpu().numpy())

    probs: dict[str, np.ndarray] = {
        t: np.concatenate(all_probs[t], axis=0) for t in TARGETS
    }
    orig_idx_arr = np.array(all_orig_idx, dtype=np.int32)
    n = len(orig_idx_arr)

    # ── Step 1: adduct ────────────────────────────────────────────────────────
    print("  Step 1: adduct prediction …")
    lm_adduct      = label_maps["adduct_enc"]
    pred_adduct_enc = lm_adduct[probs["adduct_enc"].argmax(1)]
    pred_adduct_str = adduct_le.inverse_transform(pred_adduct_enc)

    # ── Step 2: class ─────────────────────────────────────────────────────────
    print("  Step 2: class prediction …")
    lm_class      = label_maps["class_enc"]
    pred_class_enc = lm_class[probs["class_enc"].argmax(1)]
    pred_class_str = class_le.inverse_transform(pred_class_enc)

    # ── Step 3: rule-based sum composition ────────────────────────────────────
    print("  Step 3: rule-based sum composition …")
    precmz_split = precmz_all[orig_idx_arr]
    rule_nc, rule_ndb, rule_nox, sc_status = run_sum_comp_rules(
        pred_class_str, pred_adduct_str, precmz_split,
        backbone_masses, class_to_numchain,
    )
    n_ok = int(((sc_status == "matched") | (sc_status == "multi")).sum())
    print(f"    matched / multi : {n_ok:,} / {n:,}  ({100*n_ok/n:.1f}%)")

    no_chain = np.isin(sc_status, ["no_match", "no_adduct", "no_backbone"])
    row_nc   = row_num_chain[orig_idx_arr]

    # ── Steps 4-7: chain predictions ──────────────────────────────────────────
    for step_no, ct in enumerate(CHAIN_TARGETS, start=4):
        if step_no <= 7:
            print(f"  Step {step_no}: chain prediction …")
    pred_chains: dict[str, np.ndarray] = {}
    for ct in CHAIN_TARGETS:
        lm   = label_maps[ct]
        pred = lm[probs[ct].argmax(1)].copy()
        pred[no_chain] = 0
        min_nc = CHAIN_MIN.get(ct, 0)
        if min_nc > 0:
            pred[row_nc < min_nc] = 0
        pred_chains[ct] = pred

    # ── Rule-based chain inference (1-chain lipids) ───────────────────────────
    print("  Rule-based chain refinement for 1-chain lipids …")
    n_rule_applied = 0
    for i in range(n):
        num_ch = int(row_nc[i])
        if num_ch != 1:
            continue
        pd_in = {
            "nc1":  int(pred_chains["num_c_1"][i]),
            "ndb1": int(pred_chains["num_db_1"][i]),
            "nox1": int(pred_chains["num_ox_1"][i]),
        }
        pd_out = apply_chain_rules(
            pd_in,
            adduct_str   = pred_adduct_str[i],
            class_str    = pred_class_str[i],
            precursor_mz = float(precmz_split[i]),
            cgm_table    = backbone_masses,
            num_chain    = 1,
            tol_da       = chain_rule_tol_da,
        )
        if pd_out != pd_in:
            pred_chains["num_c_1"][i]  = pd_out["nc1"]
            pred_chains["num_db_1"][i] = pd_out["ndb1"]
            pred_chains["num_ox_1"][i] = pd_out["nox1"]
            n_rule_applied += 1
    print(f"    Rule overrode model predictions for {n_rule_applied:,} samples")

    # ── Project per-chain predictions to rule-decided totals ─────────────────
    # For matched/multi samples: adjust individual chain values so they sum
    # exactly to rule_nc / rule_ndb / rule_nox (mirrors XGBoost's conditioning).
    print("  Projecting chain sums to rule totals …")

    def _rescale_chains_to_sum(chain_vals: list[int], target: int) -> list[int]:
        """Adjust integer chain values to sum exactly to target (>=0)."""
        n = len(chain_vals)
        if n == 0:
            return chain_vals
        target = max(0, target)
        current = sum(chain_vals)
        if current == target:
            return list(chain_vals)
        if current == 0:
            # All zero but need non-zero total: assign entirely to chain 1
            result = list(chain_vals)
            result[0] = target
            return result
        # Scale proportionally, floor, then distribute rounding remainder
        scaled   = [v * target / current for v in chain_vals]
        floored  = [int(s) for s in scaled]
        remainder = target - sum(floored)
        order = sorted(range(n), key=lambda i: scaled[i] - floored[i], reverse=True)
        for k in range(abs(remainder)):
            idx = order[k % n]
            floored[idx] += 1 if remainder > 0 else -1
            floored[idx] = max(0, floored[idx])
        return floored

    valid_rule = (sc_status == "matched") | (sc_status == "multi")
    n_projected = 0
    for i in range(n):
        if not valid_rule[i]:
            continue
        nc = int(row_nc[i])
        if nc < 1:
            continue
        c_vals  = [int(pred_chains[f"num_c_{j}"][i])  for j in range(1, nc + 1)]
        db_vals = [int(pred_chains[f"num_db_{j}"][i]) for j in range(1, nc + 1)]
        ox_vals = [int(pred_chains[f"num_ox_{j}"][i]) for j in range(1, nc + 1)]
        c_new  = _rescale_chains_to_sum(c_vals,  int(rule_nc[i]))
        db_new = _rescale_chains_to_sum(db_vals, int(rule_ndb[i]))
        ox_new = _rescale_chains_to_sum(ox_vals, int(rule_nox[i]))
        if c_new != c_vals or db_new != db_vals or ox_new != ox_vals:
            n_projected += 1
        for j in range(nc):
            pred_chains[f"num_c_{j+1}"][i]  = c_new[j]
            pred_chains[f"num_db_{j+1}"][i] = db_new[j]
            pred_chains[f"num_ox_{j+1}"][i] = ox_new[j]
    print(f"    Adjusted chain sums for {n_projected:,} samples")

    # ── Low-data flag (true class in DATA_STARVED_CLASSES, multi-chain) ──────
    # These samples are excluded from L1/L2/L3 metrics; reported separately.
    # We need true class — derive it after building true_class_str below, so
    # defer flag computation until after _unremap block.

    # ── True labels (un-remap from dataset's encoded labels) ──────────────────
    print("  Reconstructing names …")

    def _unremap(target: str, enc_arr: np.ndarray) -> np.ndarray:
        lm = label_maps[target]
        return np.where(enc_arr < 0, -1,
                        np.array([lm[v] if 0 <= v < len(lm) else -1
                                  for v in enc_arr]))

    true_adduct_enc_r = np.array(
        [int(dataset.labels["adduct_enc"][r]) for r in orig_idx_arr], dtype=np.int32)
    true_class_enc_r  = np.array(
        [int(dataset.labels["class_enc"][r])  for r in orig_idx_arr], dtype=np.int32)

    true_adduct_enc = np.clip(_unremap("adduct_enc", true_adduct_enc_r), 0, None)
    true_class_enc  = np.clip(_unremap("class_enc",  true_class_enc_r),  0, None)

    true_class_str = class_le.inverse_transform(true_class_enc)

    true_chains: dict[str, np.ndarray] = {}
    for ct in CHAIN_TARGETS:
        enc_arr = np.array(
            [int(dataset.labels[ct][r]) for r in orig_idx_arr], dtype=np.int32)
        unr = _unremap(ct, enc_arr)
        true_chains[ct] = np.where(unr < 0, 0, unr).astype(np.int32)

    # ── Low-data flag ─────────────────────────────────────────────────────────
    low_data_flag = np.array([
        (true_class_str[i] in DATA_STARVED_CLASSES) and (int(row_nc[i]) > 1)
        for i in range(n)
    ], dtype=bool)

    # ── Build name strings ─────────────────────────────────────────────────────
    pred_names = np.array([
        reconstruct_name(
            pred_class_str[i],
            pred_chains["num_c_1"][i], pred_chains["num_db_1"][i], pred_chains["num_ox_1"][i],
            pred_chains["num_c_2"][i], pred_chains["num_db_2"][i], pred_chains["num_ox_2"][i],
            pred_chains["num_c_3"][i], pred_chains["num_db_3"][i], pred_chains["num_ox_3"][i],
            pred_chains["num_c_4"][i], pred_chains["num_db_4"][i], pred_chains["num_ox_4"][i],
            class_to_numchain,
        ) for i in range(n)
    ])
    true_names = np.array([
        reconstruct_name(
            true_class_str[i],
            true_chains["num_c_1"][i], true_chains["num_db_1"][i], true_chains["num_ox_1"][i],
            true_chains["num_c_2"][i], true_chains["num_db_2"][i], true_chains["num_ox_2"][i],
            true_chains["num_c_3"][i], true_chains["num_db_3"][i], true_chains["num_ox_3"][i],
            true_chains["num_c_4"][i], true_chains["num_db_4"][i], true_chains["num_ox_4"][i],
            class_to_numchain,
        ) for i in range(n)
    ])

    # ── True adduct string ────────────────────────────────────────────────────
    true_adduct_str = adduct_le.inverse_transform(
        np.clip(_unremap("adduct_enc", true_adduct_enc_r), 0, None)
    )

    # ── Compute total chain descriptors ───────────────────────────────────────
    def _col(prefix: str, base: str, n: int) -> np.ndarray:
        # base: "c", "db", or "ox"; n: chain index 1-4
        return pred_chains[f"num_{base}_{n}"] if prefix == "pred" \
               else true_chains[f"num_{base}_{n}"]

    pred_total_c  = sum(_col("pred", "c",  i) for i in range(1, 5))
    pred_total_db = sum(_col("pred", "db", i) for i in range(1, 5))
    pred_total_ox = sum(_col("pred", "ox", i) for i in range(1, 5))

    # For samples with multiple valid sum-comp candidates, override the model's
    # chain-sum total with the lowest-PPM rule candidate (run_sum_comp_rules
    # already returns cands[0] = lowest residual as rule_nc/ndb/nox).
    multi_mask = (sc_status == "multi")
    n_multi = int(multi_mask.sum())
    if n_multi > 0:
        pred_total_c  = pred_total_c.copy().astype(np.int32)
        pred_total_db = pred_total_db.copy().astype(np.int32)
        pred_total_ox = pred_total_ox.copy().astype(np.int32)
        pred_total_c[multi_mask]  = rule_nc[multi_mask].astype(np.int32)
        pred_total_db[multi_mask] = rule_ndb[multi_mask].astype(np.int32)
        pred_total_ox[multi_mask] = rule_nox[multi_mask].astype(np.int32)
        print(f"    Overrode {n_multi:,} multi-candidate sum-comp totals with lowest-PPM rule")
    true_total_c  = sum(_col("true", "c",  i) for i in range(1, 5))
    true_total_db = sum(_col("true", "db", i) for i in range(1, 5))
    true_total_ox = sum(_col("true", "ox", i) for i in range(1, 5))

    # ── Assemble DataFrame (column names match phase1_ml predict_split) ───────
    # Chain column naming: num_c_1 → nc1, num_db_1 → ndb1, num_ox_1 → nox1
    def _chain_col(ct: str) -> str:
        base = ct.replace("num_", "")          # "c_1", "db_1", "ox_1"
        parts = base.split("_")                # ["c","1"] / ["db","1"] / ["ox","1"]
        return "n" + parts[0] + parts[1]       # "nc1" / "ndb1" / "nox1"

    rows: dict[str, np.ndarray] = {
        "orig_idx":        orig_idx_arr,
        "low_data_flag":   low_data_flag,
        "pred_adduct":     pred_adduct_str,
        "true_adduct":     true_adduct_str,
        "pred_class":      pred_class_str,
        "true_class":      true_class_str,
        "sum_comp_status": sc_status,
        "rule_nc":         rule_nc,
        "rule_ndb":        rule_ndb,
        "rule_nox":        rule_nox,
        "pred_name":       pred_names,
        "true_name":       true_names,
        "pred_total_c":    pred_total_c,
        "true_total_c":    true_total_c,
        "pred_total_db":   pred_total_db,
        "true_total_db":   true_total_db,
        "pred_total_ox":   pred_total_ox,
        "true_total_ox":   true_total_ox,
    }
    for ct in CHAIN_TARGETS:
        col = _chain_col(ct)
        rows[f"pred_{col}"] = pred_chains[ct]
        rows[f"true_{col}"] = true_chains[ct]

    return pd.DataFrame(rows)
