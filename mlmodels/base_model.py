"""Regularizers module.
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Base model."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model according to the provided training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model."""
        pass
