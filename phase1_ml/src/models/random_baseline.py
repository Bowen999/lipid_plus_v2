"""Random baseline model implementing BaseLipidModel."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .base import BaseLipidModel


class _FrequencyClassifier:
    """
    Sklearn-compatible frequency-based classifier.

    Stores training label frequencies and on predict() samples from them
    using a seeded RNG.  Exposes n_features_in_ and classes_ so that
    predict_with_model() in metrics.py treats it like any other model.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state    = random_state
        self.classes_        = None
        self._probs          = None
        self.n_features_in_  = 0        # set to actual feature count in fit()

    def fit(self, y: np.ndarray, n_features: int = 0) -> "_FrequencyClassifier":
        classes, counts      = np.unique(y, return_counts=True)
        self.classes_        = classes.astype(np.int32)
        self._probs          = counts.astype(np.float64) / counts.sum()
        self.n_features_in_  = n_features
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        return rng.choice(self.classes_, size=len(X), p=self._probs).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_cls = len(self.classes_)
        return np.tile(self._probs.astype(np.float32), (len(X), 1)).reshape(-1, n_cls)


class RandomBaselineModel(BaseLipidModel):
    """
    Frequency-based random baseline.

    Predictions are sampled from the training-set label distribution.
    The validation set is not used.  The inner _FrequencyClassifier is
    saved as a joblib file so metrics.py's predict_with_model() works.
    """

    @property
    def name(self) -> str:
        return "random_baseline"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        random_state = self.config.get("params", {}).get("random_state", 42)
        self.model = _FrequencyClassifier(random_state=random_state)
        self.model.fit(y_train, n_features=X_train.shape[1])
        return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path, compress=3)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
