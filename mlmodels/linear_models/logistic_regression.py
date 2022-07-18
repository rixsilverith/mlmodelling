"""Logistic regression model module

This module implements the logistic regression model for binary classification.
"""

import numpy as np

from mlmodels.linear_models import LinearModel
from mlmodels.losses import BinaryCrossEntropy
from mlmodels.activations import Sigmoid
from mlmodels.optimizers import GradientBasedOptimizer, StochasticGradientDescent
from mlmodels.regularizers import Regularizer, L2Ridge

class LogisticRegressionClassifier(LinearModel):
    """A logistic regression classifier.

    Args:
        learning_rate (float): step length when using gradient descent for optimization.
    """
    def __init__(self, optimizer: GradientBasedOptimizer = StochasticGradientDescent(),
        regularizer: Regularizer = L2Ridge()):
        """Constructor for `LogisticRegressionClassifier` class.

        Args:
            optimizer (GradientBasedOptimizer): optimization algorithm used to fit the model.
        """

        super().__init__(phi = Sigmoid(), optimizer = optimizer, loss = BinaryCrossEntropy(),
            regularizer = regularizer)

    def predict_prob(self, X):
        """Predict the probabilities of the feature vectors in `X` belonging to a class. """
        X = np.insert(X, 0, 1, axis=1)
        return self.phi(X.dot(self.coefficients))

    def predict(self, X):
        """Predict a response vector given feature vectors in `X`. """
        return np.round(self.predict_prob(X))
