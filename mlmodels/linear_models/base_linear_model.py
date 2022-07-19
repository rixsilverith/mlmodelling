"""Base linear model module

This module implements an abstraction for a generalized linear model from which inherits
the linear regression model and its variants, as well as logistic regression.
"""

import math

import numpy as np
from terminaltables import AsciiTable

from mlmodels import BaseModel
from mlmodels.optimizers import GradientBasedOptimizer
from mlmodels.losses import Loss
from mlmodels.activations import Activation
from mlmodels.regularizers import Regularizer
from mlmodels.utils import stringify_config

class LinearModel(BaseModel):
    """Base class to implement generalized linear models.

    Attributes:
        phi (Activation): base function of the generalized linear model.
    """

    def __init__(self, phi: Activation, optimizer: GradientBasedOptimizer, loss: Loss,
        regularizer: Regularizer):
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

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 3000):
        """Fit the linear model according to the provided training data.
        """

        X = np.insert(X, 0, 1, axis=1)
        y = y.reshape(-1, 1)

        print("X (in fit) shape", X.shape)
        print("X used to fit:", X)

        print("y (in fit) shape", y.shape)

        n_instances, n_features = X.shape
        bound = 1 / math.sqrt(n_features)
        self.coefficients = np.random.uniform(-bound, bound, (n_features, 1))
        #print('initial coefficients:', self.coefficients)
        print("coefficients shape:", self.coefficients.shape)

        for i in range(epochs):
            y_pred = self.phi(X.dot(self.coefficients).reshape(-1, 1))

            #print('linear model predicted with shape', y_pred.shape)
            #print(y_pred)
            #print('linear model actual values for given input with shape', y.shape)
            #print(y)

            #print('y_pred shape', y_pred.shape)
            #print('y shape', y.shape)

            #print(f'epoch {i + 1}, coefficients: {self.coefficients}')
            
            #print('loss shape', self.loss(y_pred, y).shape)
            #print('regularization term:', float(self.regularizer(self.coefficients)))

            #print('loss vector', self.loss(y_pred, y))

            loss = np.mean(self.loss(y_pred, y) + self.regularizer(self.coefficients) / n_instances)
            print(f"epoch {i + 1} loss {float(loss)} regularization {self.regularizer(self.coefficients) / n_instances}")
            #print('X.T shape', X.T.shape)
            #print('X0 - y vector shape', (y_pred - y).shape)
            grad_loss = X.T.dot(y_pred - y) + self.regularizer.gradient(self.coefficients).reshape(-1, 1) / n_instances
            #print("grad loss shape", grad_loss.shape)
            #print("regularized grad loss shape", self.regularizer.gradient(self.coefficients).reshape(-1, 1).shape)
            #print('coefficients before update:', self.coefficients)
            self.coefficients = self.optimizer.update(self.coefficients, grad_loss)
            #print('coefficients after update:', self.coefficients)

            if i % 100 == 99:
                print(f'Epoch {i + 1}/{epochs} - loss {float(loss):.2f}')

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model."""

        X = np.insert(X, 0, 1, axis=1)
        return self.phi(X.dot(self.coefficients))

    @property
    def name(self) -> str:
        """Name of linear model. """

        return type(self).__name__

    def summary(self):
        """Print a summary containing model information. """

        print(AsciiTable([[f'{self.name}']]).table)
        print('phi (activation):', self.phi.name)
        print('optimizer:', self.optimizer.name)
        print(' └──', stringify_config(self.optimizer.get_config()))
        print('loss:', self.loss.name)
        print('regularizer:', self.regularizer.name)
        print(' └──', stringify_config(self.regularizer.get_config()))
