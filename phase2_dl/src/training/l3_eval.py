"""
l3_eval.py — Inline L3 (exact name match) evaluation for use during training.

Provides make_l3_eval_fn(), which returns a callable suitable for passing
to Trainer(l3_eval_fn=...).  At each validation step the trainer calls:

    l3 = l3_eval_fn(model, val_loader, device)

and uses the returned float for early stopping and best-checkpoint selection.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils import (
    TARGETS, CHAIN_MIN,
    reconstruct_name, apply_chain_rules,
)

# CHAIN_TARGETS = all targets except adduct and class
_CHAIN_TARGETS = [t for t in TARGETS if t not in ("adduct_enc", "class_enc")]


def _rescale_to_sum(vals: list[int], target: int) -> list[int]:
    n = len(vals)
    if n == 0:
        return vals
    target = max(0, target)
    current = sum(vals)
    if current == target:
        return list(vals)
    if current == 0:
        result = list(vals)
        result[0] = target
        return result
    scaled  = [v * target / current for v in vals]
    floored = [int(s) for s in scaled]
    rem     = target - sum(floored)
    order   = sorted(range(n), key=lambda i: scaled[i] - floored[i], reverse=True)
    for k in range(abs(rem)):
        idx = order[k % n]
        floored[idx] += 1 if rem > 0 else -1
        floored[idx] = max(0, floored[idx])
    return floored


@torch.no_grad()
def _compute_l3(
    model,
    val_loader,
    device,
    label_maps: dict[str, np.ndarray],
    class_le,
    adduct_le,
    backbone_masses: dict[str, float],
    class_to_numchain: dict[str, int],
    precmz_all: np.ndarray,
    row_num_chain: np.ndarray,
    true_names_all: np.ndarray,
    run_sum_comp_rules_fn,
) -> float:
    model.eval()

    # ── Collect logits + orig_idx ─────────────────────────────────────────────
    all_logits: dict[str, list] = {t: [] for t in TARGETS}
    all_orig_idx: list[int] = []

    for batch in val_loader:
        inputs = {k: v.to(device) for k, v in batch.items()
                  if k not in ("labels", "orig_idx")}
        # CNN uses teacher-forced adduct_cond during training forward(), but
        # inference uses forward_inference() which predicts adduct first.
        # Always use forward_inference() here so early stopping reflects true
        # inference performance, not inflated teacher-forced accuracy.
        if hasattr(model, "forward_inference"):
            out = model.forward_inference(inputs["spectrum"], inputs["precursor_mz"])
        else:
            keys = getattr(model, "INPUT_KEYS", list(inputs.keys()))
            out  = model(**{k: inputs[k] for k in keys if k in inputs})
        for t in TARGETS:
            all_logits[t].append(out[t].cpu().numpy())
        all_orig_idx.extend(batch["orig_idx"].numpy().tolist())

    n        = len(all_orig_idx)
    orig_idx = np.array(all_orig_idx, dtype=np.int32)
    logits   = {t: np.concatenate(all_logits[t], axis=0) for t in TARGETS}

    # ── Decode adduct + class ─────────────────────────────────────────────────
    pred_adduct_enc = label_maps["adduct_enc"][logits["adduct_enc"].argmax(1)]
    pred_adduct_str = adduct_le.inverse_transform(pred_adduct_enc)
    pred_class_enc  = label_maps["class_enc"][logits["class_enc"].argmax(1)]
    pred_class_str  = class_le.inverse_transform(pred_class_enc)

    # ── Sum-comp rules ────────────────────────────────────────────────────────
    precmz_split = precmz_all[orig_idx]
    rule_nc, rule_ndb, rule_nox, sc_status = run_sum_comp_rules_fn(
        pred_class_str, pred_adduct_str, precmz_split,
        backbone_masses, class_to_numchain,
    )

    no_chain = np.isin(sc_status, ["no_match", "no_adduct", "no_backbone"])
    row_nc   = row_num_chain[orig_idx]

    # ── Decode chains ─────────────────────────────────────────────────────────
    pred_chains: dict[str, np.ndarray] = {}
    for ct in _CHAIN_TARGETS:
        lm   = label_maps[ct]
        pred = lm[logits[ct].argmax(1)].copy().astype(np.int32)
        pred[no_chain] = 0
        min_nc = CHAIN_MIN.get(ct, 0)
        if min_nc > 0:
            pred[row_nc < min_nc] = 0
        pred_chains[ct] = pred

    # ── 1-chain rule refinement ───────────────────────────────────────────────
    for i in range(n):
        if int(row_nc[i]) != 1:
            continue
        pd_in  = {"nc1":  int(pred_chains["num_c_1"][i]),
                  "ndb1": int(pred_chains["num_db_1"][i]),
                  "nox1": int(pred_chains["num_ox_1"][i])}
        pd_out = apply_chain_rules(
            pd_in,
            adduct_str   = pred_adduct_str[i],
            class_str    = pred_class_str[i],
            precursor_mz = float(precmz_split[i]),
            cgm_table    = backbone_masses,
            num_chain    = 1,
        )
        if pd_out != pd_in:
            pred_chains["num_c_1"][i]  = pd_out["nc1"]
            pred_chains["num_db_1"][i] = pd_out["ndb1"]
            pred_chains["num_ox_1"][i] = pd_out["nox1"]

    # ── Chain-sum projection ──────────────────────────────────────────────────
    valid_rule = (sc_status == "matched") | (sc_status == "multi")
    for i in range(n):
        if not valid_rule[i]:
            continue
        nc = int(row_nc[i])
        if nc < 1:
            continue
        c_v  = [int(pred_chains[f"num_c_{j}"][i])  for j in range(1, nc + 1)]
        db_v = [int(pred_chains[f"num_db_{j}"][i]) for j in range(1, nc + 1)]
        ox_v = [int(pred_chains[f"num_ox_{j}"][i]) for j in range(1, nc + 1)]
        for j, v in enumerate(_rescale_to_sum(c_v,  int(rule_nc[i]))):
            pred_chains[f"num_c_{j+1}"][i] = v
        for j, v in enumerate(_rescale_to_sum(db_v, int(rule_ndb[i]))):
            pred_chains[f"num_db_{j+1}"][i] = v
        for j, v in enumerate(_rescale_to_sum(ox_v, int(rule_nox[i]))):
            pred_chains[f"num_ox_{j+1}"][i] = v

    # ── L3 ────────────────────────────────────────────────────────────────────
    matches = 0
    for i in range(n):
        pred_name = reconstruct_name(
            pred_class_str[i],
            int(pred_chains["num_c_1"][i]),  int(pred_chains["num_db_1"][i]),  int(pred_chains["num_ox_1"][i]),
            int(pred_chains["num_c_2"][i]),  int(pred_chains["num_db_2"][i]),  int(pred_chains["num_ox_2"][i]),
            int(pred_chains["num_c_3"][i]),  int(pred_chains["num_db_3"][i]),  int(pred_chains["num_ox_3"][i]),
            int(pred_chains["num_c_4"][i]),  int(pred_chains["num_db_4"][i]),  int(pred_chains["num_ox_4"][i]),
            class_to_numchain=class_to_numchain,
        )
        if pred_name == true_names_all[orig_idx[i]]:
            matches += 1
    return matches / max(n, 1)


def make_l3_eval_fn(
    label_maps: dict[str, np.ndarray],
    class_le,
    adduct_le,
    backbone_masses: dict[str, float],
    class_to_numchain: dict[str, int],
    precmz_all: np.ndarray,
    row_num_chain: np.ndarray,
    true_names_all: np.ndarray,
    run_sum_comp_rules_fn,
):
    """
    Return a callable(model, val_loader, device) → float that computes L3.

    Parameters
    ----------
    true_names_all      : np.ndarray of shape (N_total,), indexed by orig_idx,
                          containing the canonical true name for every dataset row.
    run_sum_comp_rules_fn : the run_sum_comp_rules function from phase1_ml metrics.
    """
    def _fn(model, val_loader, device) -> float:
        return _compute_l3(
            model, val_loader, device,
            label_maps, class_le, adduct_le,
            backbone_masses, class_to_numchain,
            precmz_all, row_num_chain, true_names_all,
            run_sum_comp_rules_fn,
        )
    return _fn
