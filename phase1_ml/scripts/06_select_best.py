"""
06_select_best.py — Compare all model families and select the best.

Loads val/test metrics from all trained + evaluated models,
generates a comparison report, and reports the best model family
per pipeline level.

Usage:
    python scripts/06_select_best.py
    python scripts/06_select_best.py --models lightgbm random_forest xgboost

Outputs:
    outputs/comparison/comparison_report.md
    outputs/comparison/selection_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
PHASE1_ROOT = SCRIPTS_DIR.parent
SRC_DIR     = PHASE1_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import OUTPUTS_DIR   # noqa: E402
from pipeline.selection import find_best_combination     # noqa: E402
from evaluation.reporting import generate_comparison_report   # noqa: E402

ALL_MODELS = ["xgboost", "lightgbm", "random_forest", "decision_tree", "random_baseline"]


def load_metrics(model_name: str) -> dict[str, dict] | None:
    """Load val + test metrics for a model. Returns None if not available."""
    eval_dir = OUTPUTS_DIR / model_name / "evaluation"
    val_path  = eval_dir / "val_metrics.json"
    test_path = eval_dir / "test_metrics.json"
    if not val_path.exists() or not test_path.exists():
        return None
    with open(val_path)  as fh: val_m  = json.load(fh)
    with open(test_path) as fh: test_m = json.load(fh)
    return {"val": val_m, "test": test_m}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=ALL_MODELS,
        help="Which models to compare (default: auto-detect from outputs/)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("06_select_best.py — Model Comparison & Selection")
    print("=" * 62)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Load available metrics ────────────────────────────────────
    candidates = args.models if args.models is not None else ALL_MODELS
    all_metrics: dict[str, dict[str, dict]] = {}

    for model_name in candidates:
        m = load_metrics(model_name)
        if m is None:
            print(f"  [SKIP] {model_name}: metrics not found — run 05_evaluate_all.py first")
        else:
            all_metrics[model_name] = m
            print(f"  [OK]   {model_name}")

    if not all_metrics:
        print("\n[ERROR] No evaluated models found. Run 04_evaluate.py first.")
        sys.exit(1)

    # ── Find best combination ─────────────────────────────────────
    all_val_metrics = {name: m["val"] for name, m in all_metrics.items()}
    result = find_best_combination(all_val_metrics)

    print("\n── Comparison Table (Val) ──────────────────────────────")
    for line in result.get("summary_table", []):
        print(line)

    print("\n── Per-Level Winner ─────────────────────────────────────")
    for level, winner in result.get("per_level_winner", {}).items():
        scores = dict(result["rankings"].get(level, []))
        score  = scores.get(winner, float("nan"))
        print(f"  {level:<28}: {winner}  ({score:.4f})")

    # ── Generate comparison report ────────────────────────────────
    comp_dir = OUTPUTS_DIR / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    generate_comparison_report(all_metrics, comp_dir / "comparison_report.md")

    # ── Save selection results ────────────────────────────────────
    # Convert rankings to JSON-serialisable format
    serialisable = {
        "best_overall":     result.get("best_overall", ""),
        "per_level_winner": result.get("per_level_winner", {}),
        "rankings": {
            label: [[name, float(score)] for name, score in scores]
            for label, scores in result.get("rankings", {}).items()
        },
        "timestamp": result.get("timestamp", ""),
    }
    sel_path = comp_dir / "selection_results.json"
    with open(sel_path, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    print(f"\nSelection results saved → {sel_path}")

    # ── Test set metrics for the best model ───────────────────────
    best = result.get("best_overall", "")
    if best and best in all_metrics:
        test_m = all_metrics[best]["test"]
        print(f"\n── Best Model ({best}) Test Metrics ───────────────────")
        for k in ["adduct_accuracy", "level0_class_accuracy",
                  "level1_sum_composition_accuracy",
                  "level2_full_chain_accuracy", "level3_name_exact_match"]:
            v = test_m.get(k, float("nan"))
            print(f"  {k:<44}: {v:.4f}")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[DONE] {ts}")


if __name__ == "__main__":
    main()
