"""Optimizers module

This module contains the implementation of several gradient-based optimization
algorithms used to train the models.

References:
    Ruder, S. "An overview of gradient descent optimisation algorithms".
        (2016): arXiv preprint arXiv:1609.04747.
"""

from abc import ABC, abstractmethod
import numpy as np

class GradientBasedOptimizer(ABC):
    """Base class for an optimization algorithm based on gradient descent."""

    @abstractmethod
    def update(self, w: np.ndarray, grad_w: np.ndarray) -> np.ndarray:
        pass

    @property
    def name(self):
        return type(self).__name__

class StochasticGradientDescent(GradientBasedOptimizer):
    """Stochastic Gradient Descent (SGD) algorithm with momentum.

    This SGD implementation uses a decay factor (momentum) of 0 by default, making it
    work just like the standard batch Gradient Descent algorithm.
    """

    def __init__(self, learning_rate: float=0.01, momentum:float=0.0, nesterov:bool=False):
        """Constructor for StochasticGradientDescent class.

        Initializes an instance of this class with the given parameters.

        Args:
            learning_rate (float, default=0.01): learning rate (step size) to be used at
                each iteration of the algorithm.
            momentum (float, default=0.0): decay factor by which the correction term for the
                negative gradient direction is applied.
            nesterov (boolean, default=False): whether to apply Nesterov momentum. See
                Sutskever et al., 2013.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.delta_w = None

    def update(self, w: np.ndarray, grad_w: np.ndarray) -> np.ndarray:
        """Update rule used to optimize the given parameter vector.

        Args:
            w (np.ndarray (1,n)): numpy array containing the n model parameters to be updated.
            grad_w (np.ndarray (1,n)): numpy array containing the n gradients of the cost
                function for the model evaluated at each model parameter.

        Returns:
            numpy array of shape (1,n) containing the n updated model parameters.

        References:
            Sutskever, I. "Training Recurrent neural Networks". PhD Thesis. (2013).
        """
        if self.delta_w is None:
            self.delta_w = np.zeros(np.shape(w))

        self.delta_w = self.momentum * self.delta_w - self.learning_rate * grad_w

        if self.nesterov:
            return w + self.momentum * self.delta_w - self.learning_rate * grad_w
        return w + self.delta_w
