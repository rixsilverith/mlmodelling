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
    def update(self, theta: np.ndarray, grad_theta: np.ndarray) -> np.ndarray:
        pass

    @property
    def name(self):
        return type(self).__name__

class StochasticGradientDescent(GradientBasedOptimizer):
    """Stochastic Gradient Descent (SGD) algorithm with momentum.

    This SGD implementation uses a decay factor (momentum) of 0 by default, making it
    work just like the standard batch Gradient Descent algorithm.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = False):
        """Initialize a `StochasticGradientDescent` class instance.

        Args:
            learning_rate (float, default=0.01): learning rate (step size) to be used at
                each iteration of the algorithm.
            momentum (float, default=0.0): decay factor by which the correction term for the
                negative gradient direction is applied.
            nesterov (bool, default=False): whether to apply Nesterov momentum. See
                Sutskever et al., 2013.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.delta_theta = None

    def update(self, theta: np.ndarray, grad_theta: np.ndarray) -> np.ndarray:
        """Update rule used to optimize the given parameter vector.

        Args:
            theta (np.ndarray (1,n)): numpy array containing the n model parameters to be updated.
            grad_theta (np.ndarray (1,n)): numpy array containing the n gradients of the cost
                function for the model evaluated at each model parameter.

        Returns:
            numpy array of shape (1,n) containing the n updated model parameters.

        References:
            Sutskever, I. et al. "On the importance of initialization and momentum in deep learning".
                ICML-13. Vol 28. (2013): pp. 1139-1147.
            Sutskever, I. "Training Recurrent neural Networks". PhD Thesis. (2013).
        """

        if self.delta_theta is None:
            self.delta_theta = np.zeros(np.shape(theta))

        self.delta_theta = self.momentum * self.delta_theta - self.learning_rate * grad_theta

        if self.nesterov:
            return theta + self.momentum * self.delta_theta - self.learning_rate * grad_theta
        return theta + self.delta_theta

    @property
    def config(self) -> str:
        return f"learning_rate: {self.learning_rate}, momentum: {self.momentum}, nesterov: {self.nesterov}"
