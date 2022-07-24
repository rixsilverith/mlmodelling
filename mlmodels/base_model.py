"""Base model module """

from __future__ import annotations
from typing import Dict, Any
from abc import ABC, abstractmethod
from terminaltables import AsciiTable
import numpy as np

from mlmodels.utils import stringify_config

class BaseModel(ABC):
    """Base model. """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model according to the provided training data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model."""

    @property
    def name(self: Self) -> str:
        """Name of the model. """

        return type(self).__name__

    @abstractmethod
    def get_config(self: Self) -> Dict[str, Any]:
        """Get the configuration of the model. """

    def summary(self: Self) -> None:
        """Print a summary containing model information. """

        print(AsciiTable([[f'{self.name}']]).table)
        print(stringify_config(self.get_config()))
