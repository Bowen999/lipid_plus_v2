"""
sum_comp_test.py — Mass-based sum composition verification.

Principle
---------
For a lipid of class C with num_chain acyl tails and sum composition
(total_nc, total_db, total_ox), the exact neutral mass follows:

    exact_mass = backbone_mass[C]
               + total_nc  * CH2          (14.01565006 Da per carbon)
               - total_db  * H2           ( 2.01565006 Da per double bond)
               + total_ox  * O_MASS       (15.99491462 Da per oxidation)
               + num_chain * CHAIN_CONST  (13.97926456 Da per chain)

backbone_mass[C] is a class-specific constant derived empirically from the
training set (encodes the head group + backbone after factoring out acyl tails).

For each candidate (total_nc, total_db) pair, total_ox is solved exactly:

    total_ox_exact = (exact_mass - backbone_mass
                      - num_chain*CHAIN_CONST
                      - total_nc*CH2 + total_db*H2) / O_MASS

A candidate is accepted when the implied mass residual is within 10 ppm.

Evaluation
----------
Two oracle conditions per row:
    - true-class  : backbone_mass keyed by the TRUE class name string
    - pred-class  : backbone_mass keyed by the PREDICTED class name string

Outputs (evaluation/):
    sum_comp_candidates_val.parquet
    sum_comp_candidates_test.parquet
    sum_comp_metrics.json
    sum_comp_report.md

Usage:
    python sum_comp_test.py              # val + test
    python sum_comp_test.py --split val  # only val
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, DATA_PATH, EVALUATION_DIR, MODELS_DIR, load_class_to_numchain

# ── Monoisotopic constants ────────────────────────────────────────────────────
H_MASS      = 1.0078250319
O_MASS      = 15.9949146221
CH2         = 12.0 + 2.0 * H_MASS    # 14.01565006
H2          = 2.0 * H_MASS            # 2.01565006
H2O         = 2.0 * H_MASS + O_MASS  # 18.01056468
CHAIN_CONST = 2.0 * O_MASS - H2O     # 13.97926456

PPM_TOL  = 10.0   # mass tolerance in ppm
ABS_TOL  = 0.020  # absolute fallback (Da) for very low-mass lipids

NC_MIN, NC_MAX   = 2, 80
NDB_MIN, NDB_MAX = 0, 14
NOX_MIN, NOX_MAX = 0,  6


# ── Adduct → neutral mass ─────────────────────────────────────────────────────
_ADDUCT_TABLE: dict[str, tuple[int, float]] = {
    "[M+H]+"      : (1,  1.00727647),
    "[M+Na]+"     : (1, 22.98921677),
    "[M+NH4]+"    : (1, 18.03437157),
    "[M+K]+"      : (1, 38.96315868),
    "[M+Li]+"     : (1,  7.01545500),
    "[M+2H]2+"    : (2,  2 * 1.00727647),
    "[M+H-H2O]+"  : (1,  1.00727647 - 18.01056468),
    "[M+H+Na]2+"  : (2,  1.00727647 + 22.98921677),
    "[M-H]-"      : (1, -1.00727647),
    "[M+Cl]-"     : (1, 34.96940210),
    "[M+HCOO]-"   : (1, 44.99765420),
    "[M+CH3COO]-" : (1, 59.01330500),
    "[M+Br]-"     : (1, 78.91884000),
    "[M-2H]2-"    : (2, -2 * 1.00727647),
    "[M-H-H2O]-"  : (1, -1.00727647 - 18.01056468),
    "[M+OAc]-"    : (1, 59.01330500),
    "[M+FA-H]-"   : (1, 44.99765420),
}


def adduct_to_neutral(precursor_mz: float, adduct: str) -> float | None:
    """Convert precursor_mz + adduct string to neutral monoisotopic mass."""
    entry = _ADDUCT_TABLE.get(adduct.strip())
    if entry is None:
        return None
    charge, ion_mass = entry
    return precursor_mz * charge - ion_mass


# ── Acyl-tail mass ────────────────────────────────────────────────────────────
def acyl_sum_mass(total_nc: int, total_db: int, total_ox: int,
                  num_chain: int) -> float:
    """
    Mass contribution of acyl tails to the intact lipid.

        acyl_sum = total_nc*CH2 - total_db*H2 + total_ox*O + num_chain*CHAIN_CONST
    """
    return (total_nc * CH2
            - total_db * H2
            + total_ox * O_MASS
            + num_chain * CHAIN_CONST)


# ── Backbone mass derivation ──────────────────────────────────────────────────
def derive_backbone_masses(df: pd.DataFrame,
                           class_to_numchain: dict[str, int]) -> dict[str, float]:
    """
    Derive per-class backbone mass as the median over training rows.

    backbone_mass[cls] = exact_mass - acyl_sum_mass(total_nc, total_db,
                                                     total_ox, num_chain)

    Returns {class_name_string: median_backbone_mass}.
    """
    accum: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        cls_name  = str(row["class"])          # always a string class name
        n_chain   = class_to_numchain.get(cls_name, 1)
        nc        = int(row["total_c"])
        ndb       = int(row["total_db"])
        nox       = int(row["total_ox"])
        exact     = float(row["exact_mass"])
        backbone  = exact - acyl_sum_mass(nc, ndb, nox, n_chain)
        accum.setdefault(cls_name, []).append(backbone)

    result: dict[str, float] = {}
    print(f"\n{'Class':<22} {'n_rows':>7} {'median_backbone':>16} {'std':>10}")
    print("─" * 58)
    for cls_name in sorted(accum):
        vals = np.array(accum[cls_name])
        med  = float(np.median(vals))
        std  = float(np.std(vals))
        result[cls_name] = med
        flag = "  *** HIGH STD" if std > 0.05 else ""
        print(f"{cls_name:<22} {len(vals):>7,} {med:>16.6f} {std:>10.6f}{flag}")

    return result


# ── Candidate enumeration ─────────────────────────────────────────────────────
def find_candidates(exact_mass: float,
                    backbone_mass: float,
                    num_chain: int,
                    ppm_tol: float = PPM_TOL,
                    abs_tol: float = ABS_TOL,
                    ) -> list[tuple[int, int, int, float]]:
    """
    Return (total_nc, total_db, total_ox, mass_residual_Da) tuples whose
    implied exact mass is within tolerance of the target.

    For each (nc, ndb) pair, total_ox is solved algebraically — no 3D grid.
    Results are sorted by mass_residual ascending (best match first).
    """
    tol_da  = max(abs_tol, ppm_tol * exact_mass * 1e-6)
    target  = exact_mass - backbone_mass - num_chain * CHAIN_CONST

    out: list[tuple[int, int, int, float]] = []
    for nc in range(NC_MIN, NC_MAX + 1):
        for ndb in range(NDB_MIN, min(NDB_MAX, nc) + 1):
            nox_exact   = (target - nc * CH2 + ndb * H2) / O_MASS
            nox_rounded = round(nox_exact)
            if nox_rounded < NOX_MIN or nox_rounded > NOX_MAX:
                continue
            residual = abs(nox_exact - nox_rounded) * O_MASS
            if residual <= tol_da:
                out.append((nc, ndb, nox_rounded, residual))

    out.sort(key=lambda x: x[3])
    return out


# ── Evaluation loop ───────────────────────────────────────────────────────────
def evaluate_split(
    split_name: str,
    indices: np.ndarray,
    src_df: pd.DataFrame,
    backbone_masses: dict[str, float],
    pred_class_names: np.ndarray,        # string class names, aligned to src_df rows
    class_to_numchain: dict[str, int],
) -> tuple[pd.DataFrame, dict]:
    """
    For each row in indices, enumerate candidates under two conditions:

    - true-class  : backbone_mass and num_chain from the TRUE class name string
    - pred-class  : backbone_mass and num_chain from the PREDICTED class name string

    pred_class_names must contain actual class name strings (e.g. "PC", "TG"),
    NOT label-encoder integer indices.
    """
    print(f"\nEvaluating {split_name} ({len(indices):,} rows) …")
    print(f"  src_df rows: {len(src_df):,}   max index: {int(indices.max())}")

    # Use to_numpy() to guarantee plain numpy arrays — avoids ArrowExtensionArray
    # indexing issues when parquet was read with arrow-backed dtypes.
    true_nc_arr  = src_df["total_c"].to_numpy(dtype=int,   na_value=0)
    true_ndb_arr = src_df["total_db"].to_numpy(dtype=int,  na_value=0)
    true_nox_arr = src_df["total_ox"].to_numpy(dtype=int,  na_value=0)
    exact_arr    = src_df["exact_mass"].to_numpy(dtype=float, na_value=0.0)
    true_cls_arr = src_df["class"].astype(str).to_numpy()

    recall_true, recall_pred = [], []
    ncand_true,  ncand_pred  = [], []
    top1_true,   top1_pred   = [], []
    records: list[dict] = []

    for i, row_i in enumerate(indices):
        row_i = int(row_i)
        true_cls  = true_cls_arr[row_i]
        # Use string class name for predictions — never an integer index
        pred_cls  = str(pred_class_names[row_i])
        exact_m   = exact_arr[row_i]
        true_tuple = (int(true_nc_arr[row_i]),
                      int(true_ndb_arr[row_i]),
                      int(true_nox_arr[row_i]))

        row_record: dict = {
            "row_index":  row_i,
            "true_class": true_cls,
            "pred_class": pred_cls,
            "exact_mass": exact_m,
            "true_nc":    true_tuple[0],
            "true_ndb":   true_tuple[1],
            "true_nox":   true_tuple[2],
        }

        for oracle, cls_name in [("true", true_cls), ("pred", pred_cls)]:
            bm      = backbone_masses.get(cls_name)
            n_chain = class_to_numchain.get(cls_name, 1)

            if bm is None:
                in_set  = False
                is_top1 = False
                n_cands = 0
                cands_str = "[]"
            else:
                cands = find_candidates(exact_m, bm, n_chain)
                cands_tuples = [(c[0], c[1], c[2]) for c in cands]
                in_set   = true_tuple in cands_tuples
                is_top1  = bool(cands_tuples) and cands_tuples[0] == true_tuple
                n_cands  = len(cands)
                cands_str = str(cands_tuples[:5])

            if oracle == "true":
                recall_true.append(in_set)
                ncand_true.append(n_cands)
                top1_true.append(is_top1)
                row_record["n_cand_true"]     = n_cands
                row_record["in_cand_true"]    = in_set
                row_record["top1_true"]       = is_top1
                row_record["top5_cands_true"] = cands_str
            else:
                recall_pred.append(in_set)
                ncand_pred.append(n_cands)
                top1_pred.append(is_top1)
                row_record["n_cand_pred"]     = n_cands
                row_record["in_cand_pred"]    = in_set
                row_record["top1_pred"]       = is_top1
                row_record["top5_cands_pred"] = cands_str

        records.append(row_record)

        if (i + 1) % 5000 == 0:
            print(f"  … {i+1:,} / {len(indices):,}")

    cand_df = pd.DataFrame(records)

    # ── Diagnostics: break down failures by cause ─────────────────
    n = len(indices)
    no_bm_true  = int(cand_df["n_cand_true"].eq(0).sum())
    no_bm_pred  = int(cand_df["n_cand_pred"].eq(0).sum())
    cls_correct = int((cand_df["true_class"] == cand_df["pred_class"]).sum())
    cls_wrong   = n - cls_correct

    # Among rows where class IS correct, how many fail the mass search?
    mask_correct = (cand_df["true_class"] == cand_df["pred_class"]).values
    recall_pred_correct = float(cand_df.loc[mask_correct, "in_cand_pred"].mean()) \
        if mask_correct.any() else float("nan")
    recall_pred_wrong   = float(cand_df.loc[~mask_correct, "in_cand_pred"].mean()) \
        if (~mask_correct).any() else float("nan")

    print(f"\n  Diagnostics ({split_name}):")
    print(f"    Class correct       : {cls_correct:,} / {n:,}  ({100*cls_correct/n:.2f}%)")
    print(f"    Class wrong         : {cls_wrong:,}   ({100*cls_wrong/n:.2f}%)")
    print(f"    No backbone (true)  : {no_bm_true:,}  (class not in training backbone table)")
    print(f"    No backbone (pred)  : {no_bm_pred:,}")
    print(f"    Recall | cls correct: {recall_pred_correct:.4f}")
    print(f"    Recall | cls wrong  : {recall_pred_wrong:.4f}")
    expected = cls_correct/n * recall_pred_correct + cls_wrong/n * recall_pred_wrong
    print(f"    Expected total recall (from above): {expected:.4f}")
    print(f"    Actual pred recall                : {float(np.mean(recall_pred)):.4f}")

    metrics: dict = {
        "n_samples": len(indices),
        "ppm_tolerance": PPM_TOL,
        "true_class_oracle": {
            "recall_in_candidate_set": round(float(np.mean(recall_true)), 4),
            "top1_accuracy":           round(float(np.mean(top1_true)),   4),
            "mean_candidates":         round(float(np.mean(ncand_true)),  2),
            "median_candidates":       float(np.median(ncand_true)),
        },
        "pred_class_oracle": {
            "recall_in_candidate_set":        round(float(np.mean(recall_pred)),        4),
            "top1_accuracy":                  round(float(np.mean(top1_pred)),           4),
            "mean_candidates":                round(float(np.mean(ncand_pred)),          2),
            "median_candidates":              float(np.median(ncand_pred)),
            "recall_given_correct_class":     round(recall_pred_correct,                4),
            "recall_given_wrong_class":       round(recall_pred_wrong,                  4),
            "pct_class_correct":              round(100 * cls_correct / n,              2),
            "rows_no_backbone_for_pred_class": no_bm_pred,
        },
    }
    return cand_df, metrics


# ── Report writer ─────────────────────────────────────────────────────────────
def write_report(val_m: dict, test_m: dict, ppm: float, out_path: Path) -> None:
    """Write a short markdown summary report."""
    keys = [
        ("Recall in candidate set", "recall_in_candidate_set"),
        ("Top-1 accuracy",          "top1_accuracy"),
        ("Mean # candidates",       "mean_candidates"),
        ("Median # candidates",     "median_candidates"),
    ]
    lines = [
        "# Sum Composition Mass-Constraint Test",
        "",
        "## Method",
        "",
        "Given lipid class and exact neutral mass, candidate sum compositions",
        f"(total_nc, total_db, total_ox) are enumerated within **{ppm} ppm** tolerance.",
        "For each (nc, ndb) pair, total_ox is solved algebraically — no 3D grid search.",
        "",
        "Two oracle conditions:",
        "- **True-class**: backbone mass from the true class name (upper bound).",
        "- **Pred-class**: backbone mass from the XGBoost-predicted class name (realistic).",
        "  Class names are always used as string keys — never integer label-encoder indices.",
        "",
        "## Results",
        "",
        "| Metric | Val (true) | Val (pred) | Test (true) | Test (pred) |",
        "|--------|-----------|-----------|------------|------------|",
    ]
    for label, k in keys:
        vt = val_m["true_class_oracle"][k]
        vp = val_m["pred_class_oracle"][k]
        tt = test_m["true_class_oracle"][k]
        tp = test_m["pred_class_oracle"][k]
        lines.append(f"| {label} | {vt} | {vp} | {tt} | {tp} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Recall** — fraction of rows where the true sum composition appears in",
        "  the candidate set. High recall means the mass constraint is informative.",
        "- **Top-1 accuracy** — fraction where the closest-mass candidate equals",
        "  the true composition (mass alone is sufficient for disambiguation).",
        "- **Mean candidates** — average candidate set size; smaller = more",
        "  discriminative power.",
        "- A gap between true-class and pred-class oracles shows how much class",
        "  misclassification degrades sum-composition recovery.",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Report saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Run the mass-based sum composition evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()

    print("=" * 60)
    print("sum_comp_test.py — Mass-Based Sum Composition Test")
    print(f"  PPM tolerance : {PPM_TOL}")
    print("=" * 60)

    # ── Load source data ──────────────────────────────────────────
    # Load from the ORIGINAL cleaned parquet (DATA_PATH), not the validated
    # copy.  This guarantees the row count (119,108) matches the feature matrix
    # and split index arrays, preventing out-of-bounds access in evaluate_split.
    print(f"\nLoading source data:\n  {DATA_PATH}")
    src_df = pd.read_parquet(
        DATA_PATH, engine="pyarrow",
        columns=["class", "exact_mass", "precursor_mz", "adduct",
                 "total_c", "total_db", "total_ox"],
    )
    print(f"Rows: {len(src_df):,}")

    class_to_numchain = load_class_to_numchain()
    train_idx = np.load(DATA_DIR / "split_train.npy")
    val_idx   = np.load(DATA_DIR / "split_val.npy")
    test_idx  = np.load(DATA_DIR / "split_test.npy")

    # ── Derive backbone masses from training rows ─────────────────
    print("\nDeriving backbone masses from training rows …")
    backbone_masses = derive_backbone_masses(src_df.iloc[train_idx], class_to_numchain)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bm_path = MODELS_DIR / "class_backbone_masses.json"
    with open(bm_path, "w") as fh:
        json.dump({k: round(v, 8) for k, v in backbone_masses.items()},
                  fh, indent=2, sort_keys=True)
    print(f"Backbone masses saved → {bm_path}")

    # ── Sanity check: adduct-derived mass vs exact_mass ──────────
    print("\nAdduct-based neutral mass check (5 training rows):")
    for ri in train_idx[:5]:
        row    = src_df.iloc[int(ri)]
        m_calc = adduct_to_neutral(float(row["precursor_mz"]), str(row["adduct"]))
        m_data = float(row["exact_mass"])
        delta  = (m_calc - m_data) if m_calc is not None else float("nan")
        calc_str = f"{m_calc:.4f}" if m_calc is not None else "N/A"
        print(f"  {row['adduct']!s:<16}  exact={m_data:.4f}"
              f"  calc={calc_str}  Δ={delta:+.5f}")

    # ── Build predicted-class name arrays (string, not integer) ──
    # Load from predictions CSV when available; fall back to true class.
    # IMPORTANT: always store/use class NAME strings, never label-encoder ints.
    # Convert class column to a plain Python-string numpy array.
    # Using .tolist() avoids returning an ArrowExtensionArray (which does not
    # support integer-index assignment) when the parquet was read with arrow dtypes.
    src_class_list = src_df["class"].tolist()

    # Size the lookup array to cover every row index in any split, not just
    # the length of src_df (which may be smaller than the max split index).
    n_total = int(max(train_idx.max(), val_idx.max(), test_idx.max())) + 1
    true_class_names = np.empty(n_total, dtype=object)
    true_class_names[:len(src_class_list)] = [str(c) for c in src_class_list]
    if len(src_class_list) < n_total:
        true_class_names[len(src_class_list):] = ""

    def load_pred_class_names(split: str) -> np.ndarray:
        """
        Return a plain numpy object array of predicted class NAME strings,
        indexed by the original dataset row index (size = n_total).

        Reads pred_class strings from the predictions CSV written by
        04_evaluate.py.  Falls back to the true class name for any row not
        present in the CSV.  Never stores integer label-encoder indices.
        """
        pred_names = true_class_names.copy()   # start from true class names
        csv_path   = EVALUATION_DIR / f"{split}_predictions.csv"
        if csv_path.exists():
            pred_df = pd.read_csv(csv_path, usecols=["row_index", "pred_class"])
            row_indices  = pred_df["row_index"].astype(int).values
            class_strs   = pred_df["pred_class"].astype(str).values
            valid        = row_indices < n_total
            pred_names[row_indices[valid]] = class_strs[valid]
            print(f"  [{split}] Loaded {valid.sum():,} predicted class names from CSV.")
        else:
            print(f"  [{split}] No predictions CSV — using true class names.")
        return pred_names

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict] = {}

    splits_to_run = []
    if args.split in ("val",  "both"):
        splits_to_run.append(("val",  val_idx))
    if args.split in ("test", "both"):
        splits_to_run.append(("test", test_idx))

    for split_name, split_idx in splits_to_run:
        print(f"\n{'─'*60}")
        pred_names = load_pred_class_names(split_name)

        cand_df, metrics = evaluate_split(
            split_name        = split_name,
            indices           = split_idx,
            src_df            = src_df,
            backbone_masses   = backbone_masses,
            pred_class_names  = pred_names,
            class_to_numchain = class_to_numchain,
        )

        out_path = EVALUATION_DIR / f"sum_comp_candidates_{split_name}.parquet"
        cand_df.to_parquet(out_path, engine="pyarrow", compression="snappy",
                           index=False)
        print(f"Saved → {out_path}  ({len(cand_df):,} rows)")

        all_metrics[split_name] = metrics
        print(f"\n  [{split_name.upper()}]")
        for oracle in ("true_class_oracle", "pred_class_oracle"):
            m = metrics[oracle]
            print(f"    {oracle}: recall={m['recall_in_candidate_set']:.4f}"
                  f"  top1={m['top1_accuracy']:.4f}"
                  f"  mean_cands={m['mean_candidates']:.1f}"
                  f"  median_cands={m['median_candidates']:.0f}")

    metrics_path = EVALUATION_DIR / "sum_comp_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    if "val" in all_metrics and "test" in all_metrics:
        write_report(all_metrics["val"], all_metrics["test"],
                     PPM_TOL, EVALUATION_DIR / "sum_comp_report.md")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[DONE] {ts}")


if __name__ == "__main__":
    main()
