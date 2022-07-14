"""Optimizers module

This module contains the implementation of several gradient-based optimization
algorithms used to train the models.
"""

import numpy as np
from abc import ABC, abstractmethod

class GradientBasedOptimizer(ABC):
    def __init__(self, learning_rate):
        pass

class StandardGradientDescent(GradientBasedOptimizer):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, w, grad_w):
        return w - self.learning_rate * grad_w

class StochasticGradientDescent(GradientBasedOptimizer):
    """Stochastic Gradient Descent (SGD) algorithm with momentum.

    This SGD implementation uses a decay factor (momentum) of 0 by default, making it
    work just like the standard batch Gradient Descent algorithm.
    """

    def __init__(self, learning_rate=0.01, momentum=0):
        """Constructor for StochasticGradientDescent class.

        Initializes an instance of this class with the given parameters.

        Args:
            learning_rate (float, default=0.01): learning rate (step size) to be used at 
                each iteration of the algorithm.
            momentum (float, default=0): decay factor by which the correction term for the 
                negative gradient direction is applied.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.delta_w = None

    def update(self, w, grad_w):
        """Update rule used to optimize the given parameter vector.

        Args:
            w (ndarray (1,n)): numpy array containing the n model parameters to be updated.
            grad_w (ndarray (1,n)): numpy array containing the n gradients of the cost
                function for the model evaluated at each model parameter.

        Returns:
            numpy array of shape (1,n) containing the n updated model parameters.
        """
        if self.delta_w is None:
            self.delta_w = np.zeros(np.shape(w))

        self.delta_w = self.momentum * self.delta_w - self.learning_rate * grad_w
        return w + self.delta_w
        
