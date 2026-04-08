"""
05_evaluate_all.py — Evaluate all model families in sequence.

Usage:
    python scripts/05_evaluate_all.py
    python scripts/05_evaluate_all.py --models lightgbm random_forest xgboost
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PHASE1_ROOT = SCRIPTS_DIR.parent

DEFAULT_MODELS = ["xgboost", "lightgbm", "random_forest", "decision_tree", "random_baseline"]


def run_evaluation(model_name: str) -> int:
    """Run 04_evaluate.py for a single model. Returns process return code."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "04_evaluate.py"), "--model", model_name]
    print(f"\n{'='*62}")
    print(f"  Evaluating: {model_name}  at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*62}")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=DEFAULT_MODELS,
        help="Which models to evaluate (default: all with trained outputs)",
    )
    args = parser.parse_args()

    # Auto-detect available models if not specified
    if args.models is not None:
        models = args.models
    else:
        models = []
        for name in DEFAULT_MODELS:
            models_dir = PHASE1_ROOT / "outputs" / name / "models"
            if models_dir.exists() and any(models_dir.glob("*.joblib")):
                models.append(name)
        if not models:
            print("[WARN] No trained models found in outputs/. Run 03_train_all.py first.")
            sys.exit(1)

    print("=" * 62)
    print("05_evaluate_all.py — Evaluate All Model Families")
    print("=" * 62)
    print(f"Models: {models}")
    print(f"Start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    wall_start = time.time()
    results: list[tuple[str, int]] = []

    for model_name in models:
        t0 = time.time()
        rc = run_evaluation(model_name)
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        results.append((model_name, rc))
        print(f"\n  [{status}] {model_name}  ({elapsed/60:.1f} min)")

    # ── Summary ───────────────────────────────────────────────────
    total_wall = time.time() - wall_start
    print(f"\n{'='*62}")
    print("Evaluation Summary")
    print(f"{'='*62}")
    for name, rc in results:
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {name:<20}: {status}")
    print(f"\nTotal wall time: {total_wall/60:.1f} min")
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if any(rc != 0 for _, rc in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
