"""
03_train_all.py — Train all three DL models sequentially.

Usage:
  python scripts/03_train_all.py
  python scripts/03_train_all.py --quick
  python scripts/03_train_all.py --skip cnn transformer
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ALL_MODELS = ["mlp", "cnn", "transformer"]

_SCRIPTS = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Pass --quick to each training run (5-epoch smoke-test)")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=ALL_MODELS,
                   help="Models to skip")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    skip  = set(args.skip)
    todo  = [m for m in ALL_MODELS if m not in skip]

    print(f"Training {len(todo)} model(s): {todo}")
    if args.quick:
        print("  [quick mode enabled]")

    failed = []
    for model_name in todo:
        cmd = [sys.executable, str(_SCRIPTS / "02_train.py"), "--model", model_name]
        if args.quick:
            cmd.append("--quick")

        print(f"\n{'='*62}")
        print(f"Starting: {model_name}")
        print(f"{'='*62}")

        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[ERROR] {model_name} training failed (exit code {rc})")
            failed.append(model_name)

    print(f"\n{'='*62}")
    if failed:
        print(f"Completed with errors.  Failed: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(todo)} model(s) trained successfully.")


if __name__ == "__main__":
    main()
