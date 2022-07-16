"""Base linear model module.

This module implements an abstraction for a generalized linear model from which inherits
the linear regression model and its variants, as well as logistic regression.
"""

import math
from abc import ABC
import numpy as np
from terminaltables import AsciiTable

from mlmodels import BaseModel
from mlmodels.optimizers import GradientBasedOptimizer
from mlmodels.losses import Loss
from mlmodels.activations import Activation
from mlmodels.regularizers import Regularizer

class LinearModel(ABC):
    """Base class to implement generalized linear models.

    Attributes:
        phi (Activation): base function of the generalized linear model.
    """
    
    def __init__(self, phi: Activation, optimizer: GradientBasedOptimizer, loss: Loss, 
        regularizer=Regularizer):
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

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int=4000):
        """Fit the linear model according to the provided training data.
        """

        X = np.insert(X, 0, 1, axis=1)
        n_features = X.shape[1]
        bound = 1 / math.sqrt(n_features)
        self.coefficients = np.random.uniform(-bound, bound, (n_features,))

        for i in range(epochs):
            y_pred = self.phi(X.dot(self.coefficients))
            loss = self.loss(y_pred, y)
            grad_loss = X.T.dot(y_pred - y)
            self.coefficients = self.optimizer.update(self.coefficients, grad_loss)

            if i % math.ceil(epochs / 10) == 0 or i == (epochs - 1):
                print(f"Epoch {i + 1}/{epochs} - loss {float(loss):.2f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model."""

        X = np.insert(X, 0, 1, axis=1)
        return self.phi(X.dot(self.coefficients))

    @property
    def name(self):
        return type(self).__name__

    def summary(self):
        print(AsciiTable([[f'{self.name}']]).table)
        print('phi (activation):', self.phi.name)
        print('optimizer:', self.optimizer.name)
        print(' └──', self.optimizer.config)
        print('loss:', self.loss.name)
        print('regularizer:', self.regularizer.name)
