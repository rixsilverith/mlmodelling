"""Activations module

This module contains the implementation of several activation functions widely
used in neural networks and in some generalized linear models.
"""

from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    """Base class for an activation function."""

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    @property
    def name(self):
        return type(self).__name__

class Identity(Activation):
    """Identity function."""

    def __call__(self, x):
        return x

    def gradient(self, x):
        return x

class Sigmoid(Activation):
    """Sigmoid (or logistic) activation function."""

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        pass

class Softmax(Activation):
    """Softmax activation function.

    Softmax converts a vector of values into a probability distribution.
    """

    def __call__(self, x):
        """Compute the softmax activation of the given vector.

        Args:
            x (np.ndarray (n,)): numpy array with n features.

        Returns:
            np.ndarray (n,): softmax of x
        """

        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def gradient(self, x):
        pass

class ReLU(Activation):
    """Rectified Linear Unit (ReLU) activation function. """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Compute the ReLU activation for the given vector.

        Args:
            x (np.ndarray (n,)): input vector to which the ReLU activation is applied.

        Returns:
            np.ndarray (n,): ReLU activation of the input vector `x`.
        """

        return np.where(x >= 0, x, 0)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the ReLU activation for the given vector.

        Args:
            x (np.ndarray (n,)): input vector to which the ReLU activation is applied.

        Returns:
            np.ndarray (n,): gradients of the ReLU activation of the input vector `x`.
        """

        return np.where(x >= 0, 1, 0)

