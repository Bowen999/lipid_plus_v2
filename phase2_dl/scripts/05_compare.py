"""
05_compare.py — Unified ML vs DL comparison report.

Loads evaluation JSONs from phase1_ml AND phase2_dl outputs, then produces
a combined markdown table at:
  phase2_dl/outputs/comparison/dl_vs_ml_report.md

Usage:
  python scripts/05_compare.py
  python scripts/05_compare.py --splits val test
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

from utils import OUTPUTS_DIR, PHASE1_ROOT

PHASE1_OUTPUTS = PHASE1_ROOT / "outputs"

# All model families to include (phase1_ml + phase2_dl)
PHASE1_MODELS = ["xgboost", "lightgbm", "random_forest", "decision_tree", "random_baseline"]
PHASE2_MODELS = ["mlp", "cnn", "transformer"]

METRIC_COLS = [
    ("adduct_accuracy",                  "Adduct"),
    ("level0_class_accuracy",            "L0-Class"),
    ("level1_sum_composition_accuracy",  "L1-SumComp"),
    ("level2_full_chain_accuracy",       "L2-Chain"),
    ("level3_name_exact_match",          "L3-Name"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   choices=["train", "val", "test"])
    return p.parse_args()


def load_metrics(
    model_name: str,
    metrics_dir: Path,
    splits: list[str],
) -> dict[str, dict]:
    """Load {split: metrics_dict} for a given model, skipping missing files."""
    result: dict[str, dict] = {}
    for split in splits:
        path = metrics_dir / f"{split}_metrics.json"
        if path.exists():
            with open(path) as fh:
                result[split] = json.load(fh)
        else:
            result[split] = {}
    return result


def format_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{100*v:.2f}%"


def build_table(
    all_metrics: dict[str, dict[str, dict]],  # {model: {split: metrics}}
    splits: list[str],
) -> str:
    """Build a markdown comparison table."""
    # Header
    header_parts = ["Model", "Type"]
    for split in splits:
        for _, col_label in METRIC_COLS:
            header_parts.append(f"{split.capitalize()}-{col_label}")
    header = "| " + " | ".join(header_parts) + " |"
    sep    = "| " + " | ".join(["---"] * len(header_parts)) + " |"

    rows = [header, sep]

    def model_type(name: str) -> str:
        return "DL" if name in PHASE2_MODELS else "ML"

    for model_name, split_metrics in all_metrics.items():
        cells = [model_name, model_type(model_name)]
        for split in splits:
            m = split_metrics.get(split, {})
            for metric_key, _ in METRIC_COLS:
                v = m.get(metric_key)
                cells.append(format_pct(v))
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def find_best(
    all_metrics: dict[str, dict[str, dict]],
    split: str = "val",
    metric: str = "level3_name_exact_match",
) -> str | None:
    best_name, best_val = None, -1.0
    for model_name, split_metrics in all_metrics.items():
        v = split_metrics.get(split, {}).get(metric, None)
        if v is not None and v > best_val:
            best_val = v
            best_name = model_name
    return best_name, best_val


def main() -> None:
    args   = parse_args()
    splits = args.splits

    # ── Collect metrics ────────────────────────────────────────────────────────
    all_metrics: dict[str, dict[str, dict]] = {}

    # phase1_ml models
    for m in PHASE1_MODELS:
        metrics_dir = PHASE1_OUTPUTS / m / "evaluation"
        if metrics_dir.exists():
            data = load_metrics(m, metrics_dir, splits)
            if any(data[s] for s in splits):
                all_metrics[m] = data
            else:
                print(f"  [skip] {m}: no metrics files found in {metrics_dir}")
        else:
            print(f"  [skip] {m}: outputs directory does not exist")

    # phase2_dl models
    for m in PHASE2_MODELS:
        metrics_dir = OUTPUTS_DIR / m / "evaluation"
        if metrics_dir.exists():
            data = load_metrics(m, metrics_dir, splits)
            if any(data[s] for s in splits):
                all_metrics[m] = data
            else:
                print(f"  [skip] {m}: no metrics files found in {metrics_dir}")
        else:
            print(f"  [skip] {m}: outputs directory does not exist")

    if not all_metrics:
        print("No model metrics found.  Run 02_train.py + 04_evaluate.py first.")
        sys.exit(1)

    print(f"\nLoaded metrics for {len(all_metrics)} model(s): {list(all_metrics)}")

    # ── Build table ───────────────────────────────────────────────────────────
    table = build_table(all_metrics, splits)

    # ── Best model (by val L3) ─────────────────────────────────────────────────
    best_name, best_val = find_best(all_metrics, split="val",
                                    metric="level3_name_exact_match")
    if best_name:
        print(f"\nBest model (val L3): {best_name}  ({100*best_val:.2f}%)")

    # ── Rankings per metric on val ─────────────────────────────────────────────
    print("\n--- Val Rankings ---")
    for metric_key, col_label in METRIC_COLS:
        ranked = sorted(
            [(m, all_metrics[m].get("val", {}).get(metric_key, 0.0) or 0.0)
             for m in all_metrics],
            key=lambda x: x[1], reverse=True
        )
        top = ranked[0]
        print(f"  {col_label:20s}: {top[0]}  ({100*top[1]:.2f}%)")

    # ── Write report ──────────────────────────────────────────────────────────
    cmp_dir = OUTPUTS_DIR / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    report_path = cmp_dir / "dl_vs_ml_report.md"

    lines = [
        "# DL vs ML Comparison Report",
        "",
        f"Models compared: {', '.join(all_metrics.keys())}",
        f"Splits: {', '.join(splits)}",
        "",
        "## Metrics Table",
        "",
        table,
        "",
    ]

    if best_name:
        lines += [
            "## Best Model",
            "",
            f"**{best_name}** achieves the highest val L3 accuracy: "
            f"**{100*best_val:.2f}%**",
            "",
        ]

    lines += [
        "## Per-Metric Rankings (val set)",
        "",
    ]
    for metric_key, col_label in METRIC_COLS:
        ranked = sorted(
            [(m, all_metrics[m].get("val", {}).get(metric_key, 0.0) or 0.0)
             for m in all_metrics],
            key=lambda x: x[1], reverse=True
        )
        lines.append(f"**{col_label}**: " +
                     ", ".join(f"{n} ({100*v:.2f}%)" for n, v in ranked))
        lines.append("")

    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nReport → {report_path}")


if __name__ == "__main__":
    main()
