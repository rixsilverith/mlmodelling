"""Logistic regression model module

This module implements the logistic and softmax regression models for binary and
multi-class classification, respectively.
"""

import numpy as np

from mlmodels.linear_models import LinearModel
from mlmodels.losses import BinaryCrossEntropy
from mlmodels.activations import Sigmoid
from mlmodels.optimizers import GradientBasedOptimizer, StochasticGradientDescent
from mlmodels.regularizers import Regularizer, L2Ridge

class LogisticRegressionClassifier(LinearModel):
    """A logistic regression classifier. """

    def __init__(self, optimizer: GradientBasedOptimizer = StochasticGradientDescent(),
        regularizer: Regularizer = L2Ridge()):
        """Initialize a `LogisticRegressionClassifier` class instance.

        Args:
            optimizer (GradientBasedOptimizer, default = StochasticGradientDescent): optimization method
                used to fit the model to the data.
            regularizer (Regularizer, default = L2Ridge): regularization term to be used.
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

class SoftmaxClassifier(LinearModel):
    """Softmax regression classifier. Also known as multinomial logistic regression.

    This is a generalization of the `LogisticRegressionClassifier` to multi-class classification
    problems. 
    """

    def __init__(self, optimizer: GradientBasedOptimizer = StochasticGradientDescent(),
        regularizer: Regularizer = L2Ridge()):
        """Initialize a `SoftmaxClassifier` class instance. """

        super().__init__(phi = Softmax(), optimizer = optimizer, loss = CategorialCrossEntropy(),
            regularizer = regularizer)
