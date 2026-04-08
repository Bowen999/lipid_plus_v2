"""Abstract base class for all lipid MS/MS prediction models."""
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd


class BaseLipidModel(ABC):
    """
    Contract that every model (XGBoost, LightGBM, RF, DT, Random) must satisfy.

    Each model predicts ONE target column (e.g. 'class_enc', 'nc1', 'ndb2').
    The training script composes 14 instances (adduct + class + 12 chain targets)
    into a full prediction pipeline.
    """

    def __init__(self, target_name: str, config: dict):
        self.target_name = target_name
        self.config = config
        self.model = None

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            sample_weight: np.ndarray | None = None) -> dict:
        """Train the model. Return a dict of training metadata (best_iter, etc.)."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (int array)."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix (n_samples, n_classes)."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'xgboost', 'lightgbm'."""
        ...
