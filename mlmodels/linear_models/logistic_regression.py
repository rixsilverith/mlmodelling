"""Logistic regression model module.

This module implements the logistic regression model for binary classification.
"""

import math
import numpy as np

from mlmodels.linear_models import LinearModel
from mlmodels.losses import BinaryCrossentropy
from mlmodels.activations import Sigmoid
from mlmodels.optimizers import GradientBasedOptimizer, StochasticGradientDescent
from mlmodels.regularizers import Regularizer

class LogisticRegressionClassifier(LinearModel):
    """A logistic regression classifier.

    Args:
        learning_rate (float): step length when using gradient descent for optimization.
    """
    def __init__(self, optimizer: GradientBasedOptimizer=StochasticGradientDescent(), 
        regularizer: Regularizer=None):
        """Constructor for `LogisticRegressionClassifier` class.

        Args:
            optimizer (GradientBasedOptimizer): optimization algorithm used to fit the model.
        """

        if regularizer is None:
            regularizer =  lambda x: 0
            regularizer.gradient = lambda x: 0
            regularizer.name = 'None'

        super(LogisticRegressionClassifier, self).__init__( 
            phi=Sigmoid(), optimizer=optimizer, loss=BinaryCrossentropy(), regularizer=regularizer)

    def predict_prob(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return self.phi(X.dot(self.coefficients))

    def predict(self, X):
        return np.round(self.predict_prob(X))

    def summary(self):
        print('LogisticRegressionClasifier model')
        print('phi (activation):', self.phi.name)
        print('optimizer:', self.optimizer.name)
        print('loss:', self.loss.name)
        print('regularizer', self.regularizer.name)

class OldLogisticRegressionClassifier():
    """A logistic regression classifier.

    Args:
        learning_rate (float): step length when using gradient descent for optimization.
    """
    def __init__(self, optimizer: GradientBasedOptimizer=StochasticGradientDescent()):
        """Constructor for `LogisticRegressionClassifier` class.

        Args:
            optimizer (GradientBasedOptimizer): optimization algorithm used to fit the model.
        """
        self.coefficients = None
        self.optimizer = optimizer
        self.cost_function = BinaryCrossEntropy()

    def fit(self, X, y, epochs=3000):
        """Fit the logistic regression model to the given training data.

        Args:
            X (ndarray (m,n)): feature matrix containing m feature vectors with n features each.
            y (ndarray (m,)): label vector containing the m labels corresponding to the
                feature vectors.
            epochs (int, optional): number of iterations used to fit the model.
        """
        X = np.insert(X, 0, 1, axis=1)
        n_features = X.shape[1]
        bound = 1 / math.sqrt(n_features)
        self.coefficients = np.random.uniform(-bound, bound, (n_features,))

        for i in range(epochs):
            y_pred = Sigmoid()(X.dot(self.coefficients))
            cost = self.cost_function(y_pred, y)

            grad_cost = X.T.dot(y_pred - y)
            self.coefficients = self.optimizer.update(self.coefficients, grad_cost)

            if i % math.ceil(epochs / 10) == 0 or i == (epochs - 1):
                print(f"Epoch {i}: cross-entropy error (cost): {float(cost):.2f}")

        return self

    def predict_prob(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return Sigmoid()(X.dot(self.coefficients))

    def predict(self, X):
        return np.round(self.predict_prob(X))
