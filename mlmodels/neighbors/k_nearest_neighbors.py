"""k-Nearest neighbors module.

This module implements the k-nearest neighbors algorithm for classification. 
"""

from __future__ import annotations
from typing import Dict, Any
from abc import ABC
import numpy as np

from mlmodels import BaseModel

class KNearestNeighbors(BaseModel, ABC):
    """k-nearest neighbors algorithm.

    This model can be used for both classification (`KNeighborsClassifier`) and for
    regression (`KNeighborsRegressor`).
    """

    def __init__(self: Self, k_neighbors: int) -> Self:
        """Initialize a `KNearestNeighbors` class instance.

        Args:
            k_neighbors (int): number of neighbors to be used in the algorithm.
        """
    
        if k_neighbors < 1:
            raise ValueError(f'k_neighbors must be greater than 0. Got: {k_neighbors}')
        self.k_neighbors = k_neighbors

    def get_config(self: Self) -> Dict[str, Any]:
        """Get the configuration of the k-nearest neighbors model. """

        return { 'k_neighbors': self.k_neighbors, 'total_neighbors': len(self.neighbors) }

    def fit(self: Self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the model according to the provided training data. """

        self.data = X
        self.neighbors = y
        return self

    def predict(self: Self, X: np.ndarray) -> float:
        y_pred = np.empty(X.shape[0])

        for i, sample in enumerate(X):
            idx = np.argsort([np.linalg.norm(sample - x) for x in self.data])[:self.k_neighbors]
            kn_neighbors = np.array([self.neighbors[i] for i in idx])
            y_pred[i] = self.neighbor_operation(kn_neighbors)

        return y_pred

class KNeighborsClassifier(KNearestNeighbors):
    """k-Nearest neighbors classifier. """

    def _most_common_class(self: Self, y: np.ndarray) -> float:
        """Compute the most common class in the response vector `y`. """

        return max(list(y), key = list(y).count)

    def predict(self: Self, X: np.ndarray) -> float:
        self.neighbor_operation = self._most_common_class
        return super().predict(X)

class KNeighborsRegressor(KNearestNeighbors):
    """k-Nearest neighbors regresssor. """

    def predict(self: Self, X: np.ndarray) -> float:
        self.neighbor_operation = lambda x: np.mean(x)
        return super().predict(X)
