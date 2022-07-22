"""Regression linear models module

This module implements the linear regression model and variants, such as Lasso and
Ridge regression.
"""

from itertools import combinations_with_replacement
from typing import Dict, Any
import numpy as np

from mlmodels.linear_models import LinearModel
from mlmodels.optimizers import GradientBasedOptimizer, StochasticGradientDescent
from mlmodels.activations import Identity
from mlmodels.losses import SquaredLoss
from mlmodels.regularizers import L2Ridge, L1Lasso
from mlmodels.utils import normalize

class LinearRegressor(LinearModel):
    """Linear regression model with no regularization. 

    For a regularized version of this model see the `RidgeRegressor` and `LassoRegressor`
    models for L2 and L1 regularization, respectively, which also support polynomial 
    features out of the box.
    """

    def __init__(self, optimizer: GradientBasedOptimizer = StochasticGradientDescent()):
        super().__init__(phi = Identity(), optimizer = optimizer, loss = SquaredLoss(),
            regularizer = L2Ridge(alpha = 0.))

class PolynomialRegressor(LinearModel):
    """Polynomial regression model. """

    def __init__(self, degree: int = 2, optimizer: GradientBasedOptimizer = StochasticGradientDescent()):
        self.degree = degree

        super().__init__(phi = Identity(), optimizer = optimizer, loss = SquaredLoss(),
            regularizer = L2Ridge(alpha = 0.3))

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
        X = normalize(self._transform_polynomial_features(X, self.degree))
        return super().fit(X, y, epochs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = normalize(self._transform_polynomial_features(X, self.degree))
        return super().predict(X)

    def get_config(self) -> Dict[str, Any]:
        return { 'degree': int(self.degree) } | super().get_config()

class RidgeRegressor(LinearModel):
    """Ridge regression model. 

    This regressor performs OLS linear regression with L2 (Ridge) regularization. It also
    support polynomial features.
    """

    def __init__(self, degree: int = 1, reg_factor: float = 0.01,
        optimizer: GradientBasedOptimizer = StochasticGradientDescent()):
        """Initialize a `RidgeRegressor` class instance. """

        if degree <= 0:
            raise ValueError(f'degree must be > 0. Got {degree}')

        if reg_factor < 0:
            raise ValueError(f'reg_factor must be >= 0. Got {reg_factor}')

        self.degree = degree
        self.reg_factor = reg_factor

        super().__init__(phi = Identity(), optimizer = optimizer, loss = SquaredLoss(),
            regularizer = L2Ridge(alpha = self.reg_factor))

class LassoRegressor(LinearModel):
    """Lasso regression model. 

    This regressor performs OLS linear regression with L1 (Lasso) regularization. It also
    support polynomial features.
    """

    def __init__(self, degree: int = 1, reg_factor: float = 0.01,
        optimizer: GradientBasedOptimizer = StochasticGradientDescent()):
        """Initialize a `LassoRegressor` class instance. """

        if degree <= 0:
            raise ValueError(f'degree must be > 0. Got {degree}')

        if reg_factor < 0:
            raise ValueError(f'reg_factor must be >= 0. Got {reg_factor}')

        self.degree = degree
        self.reg_factor = reg_factor

        super().__init__(phi = Identity(), optimizer = optimizer, loss = SquaredLoss(),
            regularizer = L1Lasso(alpha = self.reg_factor))
