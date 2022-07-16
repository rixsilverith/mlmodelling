"""Activations module."""

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

class Softmax(Activation):
    """Softmax activation function. 

    Softmax converts a vector of values to a probability distribution.
    """

    def __call__(self, x):
        """Convert a 

        Args:
            x (np.ndarray (n,)): numpy array with n features.

        Returns:
            np.ndarray (n,): softmax of x
        """

        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def gradient(self, x):
        pass

class Sigmoid(Activation):
    """Sigmoid (or logistic) activation function."""

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        pass

class Identity(Activation):
    """Identity function."""

    def __call__(self, x):
        return x

    def gradient(self, x):
        return x
