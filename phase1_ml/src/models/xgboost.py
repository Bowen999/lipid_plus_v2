"""XGBoost model wrapper implementing BaseLipidModel."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

from .base import BaseLipidModel


class XGBoostModel(BaseLipidModel):
    """
    XGBClassifier wrapper.

    XGBoost requires contiguous 0..K-1 labels, so fit() remaps y_train
    and y_val internally before training.  The inner model stores the
    remapped labels; predict() returns them as-is (remapping back to
    original label space is done by build_class_maps() in the evaluator).

    save() serialises only self.model (the raw XGBClassifier), so that
    predict_with_model() in metrics.py can load and use it without
    knowing about this wrapper.
    """

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        unique = np.sort(np.unique(y_train))
        class_map = {int(v): i for i, v in enumerate(unique.tolist())}
        n_cls = len(unique)

        y_tr_enc = np.array([class_map[int(v)] for v in y_train], dtype=np.int32)
        y_vl_enc = np.array([class_map.get(int(v), 0) for v in y_val], dtype=np.int32)

        params = dict(self.config["params"], num_class=n_cls)
        self.model = XGBClassifier(objective="multi:softmax", **params)
        self.model.fit(
            X_train, y_tr_enc,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_vl_enc)],
            verbose=50,
        )
        return {"best_iteration": self.model.best_iteration}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        """Persist the inner XGBClassifier (not the wrapper)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path, compress=3)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
