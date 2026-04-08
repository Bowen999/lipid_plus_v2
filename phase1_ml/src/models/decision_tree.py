"""Decision Tree model wrapper implementing BaseLipidModel."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .base import BaseLipidModel


class DecisionTreeModel(BaseLipidModel):
    """
    DecisionTreeClassifier wrapper.

    Uses class_weight="balanced" to handle class imbalance.
    No early stopping — the validation set is ignored during training.

    save() serialises only self.model (the raw DecisionTreeClassifier).
    """

    @property
    def name(self) -> str:
        return "decision_tree"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        params = dict(self.config["params"])
        self.model = DecisionTreeClassifier(**params)
        self.model.fit(X_train, y_train)
        return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path, compress=3)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
