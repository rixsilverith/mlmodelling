"""Base linear model module

This module implements an abstraction for a generalized linear model from which inherits
the linear regression model and its variants, as well as logistic regression.
"""

from __future__ import annotations
from typing import Dict, Any
from abc import ABC
import math
import numpy as np

from mlmodels import BaseModel
from mlmodels.optimizers import GradientBasedOptimizer
from mlmodels.losses import Loss
from mlmodels.activations import Activation
from mlmodels.regularizers import Regularizer

class LinearModel(BaseModel, ABC):
    """Base class to implement generalized linear models.

    Attributes:
        phi (Activation): base function of the generalized linear model.

    References:
        Glorot, X. & Bengio, Y. (2010). "Understanding the difficulty of training deep 
            feedforward neural networks". In Y. W. Teh & M. Titterington (eds.), Proceedings 
            of the Thirteenth International Conference on Artificial Intelligence and 
            Statistics: pp. 249-256.
    """

    def __init__(self: Self, phi: Activation, optimizer: GradientBasedOptimizer, loss: Loss,
        regularizer: Regularizer) -> Self:
        """Initialize a `LinearModel` instance.

        Args:
            phi (Activation): base function of the generalized linear model. For instance, in
                linear regression `phi = Identity()` and for logistic regression `phi = Sigmoid()`.
            optimizer (GradientBasedOptimizer): optimization algorithm used to fit the model.
            loss (LossFunction): loss function used to compute prediction errors.
            regularizer (Regularizer): regularization function used to reduce model complexity
                and prevent overfitting.
        """

        self.coefficients = None
        self.phi = phi
        self.optimizer = optimizer
        self.loss = loss
        self.regularizer = regularizer

    def fit(self: Self, X: np.ndarray, y: np.ndarray, epochs: int = 3000) -> Self:
        """Fit the linear model according to the provided training data. """

        X = np.insert(X, 0, 1, axis=1)
        y = y.reshape(-1, 1)

        n_instances, n_features = X.shape
        # use Xavier initialization. See Glorot (2010).
        bound = 1 / math.sqrt(n_features)
        self.coefficients = np.random.uniform(-bound, bound, (n_features, 1))

        for i in range(epochs):
            y_pred = self.phi(X.dot(self.coefficients))

            loss = np.mean(self.loss(y_pred, y) + self.regularizer(self.coefficients) / n_instances)
            grad_loss = X.T.dot(y_pred - y) + self.regularizer.gradient(self.coefficients).reshape(-1, 1) / n_instances
            self.coefficients = self.optimizer.update(self.coefficients, grad_loss)

            if i % 100 == 99:
                print(f'Epoch {i + 1}/{epochs} - loss {float(loss):.2f}')

        return self

    def predict(self: Self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model. """

        X = np.insert(X, 0, 1, axis=1)
        return self.phi(X.dot(self.coefficients))

    def get_config(self: Self) -> Dict[str, Any]:
        """Get the configuration of the linear model. """

        return { 'phi (activation)': self.phi.name,
            'optimizer': { 'name': self.optimizer.name } | self.optimizer.get_config(),
            'loss': self.loss.name,
            'regularizer': { 'name': self.regularizer.name } | self.regularizer.get_config() }
