"""LightGBM model wrapper implementing BaseLipidModel."""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier

from .base import BaseLipidModel


class LightGBMModel(BaseLipidModel):
    """
    LGBMClassifier wrapper.

    LGBMClassifier handles arbitrary integer labels natively — no explicit
    label remapping needed.  Early stopping is passed via the callbacks API.

    save() serialises only self.model (the raw LGBMClassifier) so that
    predict_with_model() in metrics.py works unchanged.
    """

    @property
    def name(self) -> str:
        return "lightgbm"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        # Remap labels to contiguous 0..K-1 so the val set never has
        # labels unseen in training (can happen with subsampled quick mode).
        unique = np.sort(np.unique(y_train))
        class_map = {int(v): i for i, v in enumerate(unique.tolist())}
        y_tr_enc  = np.array([class_map[int(v)] for v in y_train], dtype=np.int32)
        y_vl_enc  = np.array([class_map.get(int(v), 0) for v in y_val], dtype=np.int32)

        params = dict(self.config["params"])
        early_stop = params.pop("early_stopping_rounds", 30)

        self.model = LGBMClassifier(**params)
        self.model.fit(
            X_train, y_tr_enc,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_vl_enc)],
            callbacks=[
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(50),
            ],
        )
        return {"best_iteration": self.model.best_iteration_}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        """Persist the inner LGBMClassifier (not the wrapper)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path, compress=3)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
