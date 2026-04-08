"""
inference.py — Model-agnostic inference pipeline.

Loads the 14 trained models for a given model family from
outputs/{model_name}/models/ and wraps the predict_split() call
from src/evaluation/metrics.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Ensure src/ is on the path so utils and evaluation.metrics are importable
_SRC = Path(__file__).resolve().parent.parent   # phase1_ml/src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import OUTPUTS_DIR   # noqa: E402  (after sys.path insert)

# ── Model naming conventions ───────────────────────────────────────────────────
MODEL_PREFIXES: dict[str, str] = {
    "xgboost":        "xgb",
    "lightgbm":       "lgb",
    "random_forest":  "rf",
    "decision_tree":  "dt",
    "random_baseline": "baseline",
}

TARGET_KEYS: list[str] = [
    "adduct", "class",
    "nc1", "ndb1", "nox1",
    "nc2", "ndb2", "nox2",
    "nc3", "ndb3", "nox3",
    "nc4", "ndb4", "nox4",
]


class InferencePipeline:
    """
    Load 14 models for a model family and run the full 7-step
    hierarchical inference cascade.

    Parameters
    ----------
    model_name : str
        One of the keys in MODEL_PREFIXES, e.g. "lightgbm".
    """

    def __init__(self, model_name: str) -> None:
        if model_name not in MODEL_PREFIXES:
            raise ValueError(
                f"Unknown model_name {model_name!r}. "
                f"Choose from {list(MODEL_PREFIXES)}"
            )
        self.model_name = model_name
        prefix          = MODEL_PREFIXES[model_name]
        models_dir      = OUTPUTS_DIR / model_name / "models"

        print(f"  Loading {model_name} models from {models_dir} …")
        self.models: dict = {}
        for key in TARGET_KEYS:
            path = models_dir / f"{prefix}_{key}.joblib"
            if path.exists():
                self.models[key] = joblib.load(path)
                print(f"    loaded {path.name}")
            else:
                print(f"    [WARN] {path.name} not found — will use 0 predictions")
                self.models[key] = None

    def run(
        self,
        split_name: str,
        indices: np.ndarray,
        X_base_all: np.ndarray,
        precmz_all: np.ndarray,
        df: pd.DataFrame,
        class_maps: dict,
        class_le,
        adduct_le,
        class_to_numchain: dict,
        row_num_chain: np.ndarray,
        backbone_masses: dict,
    ) -> pd.DataFrame:
        """
        Run predict_split() from evaluation/metrics.py with the loaded models.

        Returns a DataFrame with predicted and true values for the split.
        """
        from evaluation.metrics import predict_split   # lazy import to avoid circular deps
        return predict_split(
            split_name, indices, X_base_all, precmz_all,
            df, self.models, class_maps,
            class_le, adduct_le,
            class_to_numchain, row_num_chain, backbone_masses,
        )
