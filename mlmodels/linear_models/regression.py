"""Regression linear models module.

This module implements the linear regression model and variants, such as lasso and
Ridge regression.
"""

from itertools import combinations_with_replacement
from typing import Dict, Any
import numpy as np

from mlmodels.linear_models import LinearModel
from mlmodels.optimizers import GradientBasedOptimizer, StochasticGradientDescent
from mlmodels.activations import Identity
from mlmodels.losses import SquareLoss
from mlmodels.regularizers import L2Ridge
from mlmodels.utils import normalize

class LinearRegressor(LinearModel):
    """A linear regressor without regularization. """

    def __init__(self, optimizer: GradientBasedOptimizer = StochasticGradientDescent()):
        super().__init__(phi = Identity(), optimizer = optimizer, loss = SquareLoss(),
            regularizer = L2Ridge(alpha = 0.))

class PolynomialRegressor(LinearModel):
    """Polynomial regression model. """

    def __init__(self, degree: int = 2, optimizer: GradientBasedOptimizer = StochasticGradientDescent()):
        self.degree = degree

        super().__init__(phi = Identity(), optimizer = optimizer, loss = SquareLoss(),
            regularizer = L2Ridge(alpha = 0.3))

    '''
    def _transform_polynomial_features(self, X: np.ndarray, degree: int) -> np.ndarray:
        X_transform = np.ones((X.shape[1], 1)) # ()

        for i in range(1, degree + 1):
            X_transform = np.append(X_transform, np.power(X, i + 1).reshape(-1, 1), axis = 1)

        return X_transform

    '''
    def _transform_polynomial_features(self, X: np.ndarray, degree: int) -> np.ndarray:
        n_samples, n_features = np.shape(X)

        def index_combinations():
            combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
            flat_combs = [item for sublist in combs for item in sublist]
            return flat_combs
    
        combinations = index_combinations()
        n_output_features = len(combinations)
        X_new = np.empty((n_samples, n_output_features))
    
        for i, index_combs in enumerate(combinations):  
            X_new[:, i] = np.prod(X[:, index_combs], axis=1)

        return X_new

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 3000):
        print('X_train after polynomial features:', self._transform_polynomial_features(X, self.degree))
        X = normalize(self._transform_polynomial_features(X, self.degree))

        print('X_train after normalization and polyfeatures:', X)
        #print('trans X', X)
        #print('trans X shape', X.shape)
        return super().fit(X, y, epochs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = normalize(self._transform_polynomial_features(X, self.degree))
        return super().predict(X)

    def get_config(self) -> Dict[str, Any]:
        return { 'degree': int(self.degree) } | super().get_config()
