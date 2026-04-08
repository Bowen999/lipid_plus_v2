"""
03_train_all.py — Train all model families in sequence.

Usage:
    python scripts/03_train_all.py              # train lightgbm, rf, dt, baseline
    python scripts/03_train_all.py --quick      # smoke-test all
    python scripts/03_train_all.py --models lightgbm random_forest
    python scripts/03_train_all.py --include-xgboost   # also retrain XGBoost
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

DEFAULT_MODELS = ["lightgbm", "random_forest", "decision_tree", "random_baseline"]
ALL_MODELS     = ["xgboost"] + DEFAULT_MODELS


def run_training(model_name: str, quick: bool) -> int:
    """Run 02_train.py for a single model. Returns process return code."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "02_train.py"), "--model", model_name]
    if quick:
        cmd.append("--quick")
    print(f"\n{'='*62}")
    print(f"  Starting: {model_name}  at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*62}")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=None,
        help="Which models to train (default: all except xgboost)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke-test mode: passed through to 02_train.py",
    )
    parser.add_argument(
        "--include-xgboost", action="store_true",
        help="Also retrain XGBoost (skipped by default — already frozen)",
    )
    args = parser.parse_args()

    if args.models is not None:
        models = args.models
    elif args.include_xgboost:
        models = ALL_MODELS
    else:
        models = DEFAULT_MODELS

    print("=" * 62)
    print("03_train_all.py — Train All Model Families")
    print("=" * 62)
    print(f"Models: {models}")
    if args.quick:
        print("Mode  : QUICK / SMOKE-TEST")
    print(f"Start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    wall_start = time.time()
    results: list[tuple[str, int]] = []

    for model_name in models:
        t0 = time.time()
        rc = run_training(model_name, args.quick)
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        results.append((model_name, rc))
        print(f"\n  [{status}] {model_name}  ({elapsed/60:.1f} min)")

    # ── Summary ───────────────────────────────────────────────────
    total_wall = time.time() - wall_start
    print(f"\n{'='*62}")
    print("Training Summary")
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
