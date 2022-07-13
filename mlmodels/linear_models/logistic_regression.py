import numpy as np
import math

from mlmodels.loss_functions import BinaryCrossEntropy
from mlmodels.activation_functions import Sigmoid
from mlmodels.optimizers import GradientBasedOptimizer, StochasticGradientDescent

class LogisticRegressionClassifier(object):
    """A logistic regression classifier.

    Args:
        learning_rate (float): step length when using gradient descent for optimization.
    """
    def __init__(self, optimizer: GradientBasedOptimizer=StochasticGradientDescent()):
        self.coefficients = None
        self.optimizer = optimizer
        self.cost_function = BinaryCrossEntropy()

    def fit(self, X, y, epochs=3000):
        X = np.insert(X, 0, 1, axis=1)
        m_samples, n_features = X.shape
        bound = 1 / math.sqrt(n_features)
        self.coefficients = np.random.uniform(-bound, bound, (n_features,))

        for i in range(epochs):
            y_pred = Sigmoid(X.dot(self.coefficients))
            cost = self.cost_function(y_pred, y)

            grad_cost = X.T.dot(y_pred - y)
            self.coefficients = self.optimizer.update(self.coefficients, grad_cost)

            if i % math.ceil(epochs / 10) == 0 or i == (epochs - 1):
                print(f"Epoch {i}: cross-entropy error (cost): {float(cost):.2f}")

        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return Sigmoid(X.dot(self.coefficients))

class GDLogisticRegressionClassifier(object):
    """A logistic regression classifier.

    Args:
        learning_rate (float): step length when using gradient descent for optimization.
    """
    def __init__(self, learning_rate=.1):
        self.coefficients = None
        self.learning_rate = learning_rate
        self.cost_function = BinaryCrossEntropy()

    def fit(self, X, y, epochs=3000):
        X = np.insert(X, 0, 1, axis=1)
        m_samples, n_features = X.shape
        bound = 1 / math.sqrt(n_features)
        self.coefficients = np.random.uniform(-bound, bound, (n_features,))

        for i in range(epochs):
            y_pred = Sigmoid(X.dot(self.coefficients))
            cost = self.cost_function(y_pred, y)
            self.coefficients -= (self.learning_rate / m_samples) * X.T.dot(y_pred - y)

            if i % math.ceil(epochs / 10) == 0 or i == (epochs - 1):
                print(f"Epoch {i}: cross-entropy error (cost): {float(cost):.2f}")

        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return Sigmoid(X.dot(self.coefficients))

