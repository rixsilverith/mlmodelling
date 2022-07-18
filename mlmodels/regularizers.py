"""Regularizers module

This module implements several regularizers used to reduce model complexity and prevent
overfitting during model training.
"""

from abc import ABC, abstractmethod
from typings import Self, Dict
import numpy as np

class Regularizer(ABC):
    """Base class for a regularizer. 

    This regularization term is used to decrease model complexity and prevent the model
    coefficients from becoming too large by adding a penalty to this complexity to the
    empirical loss.

    Attributes:
        alpha (float): regularization factor; i.e. factor that multiplies the regularization term.
        name (str): name of the regularizer.
    """

    def __init__(self, alpha: float) -> Self:
        """Initialize a `Regularizer` class instance.

        Args:
            alpha (float): regularization factor (0 < alpha < 1).
        """

        if alpha < 0:
            raise ValueError(f'Regularization factor alpha must be nonnegative. Got alpha = {alpha}')

        self.alpha = alpha

    @abstractmethod
    def __call__(self, theta: np.ndarray) -> float:
        """Compute the regularization term of the given parameter vector.

        Args:
            theta (np.numpy (n,)): vector containing n model parameters.

        Returns:
            float: regularization term of `theta`.
        """
        pass

    @abstractmethod
    def gradient(self, theta: np.ndarray) -> np.ndarray:
        pass

    @property
    def name(self) -> str:
        return type(self).__name__

    def get_config(self) -> Dict[str, Any]:
        return { 'alpha': float(self.alpha) }

class L1LassoRegularizer(Regularizer):
    """l1 norm (Lasso) regularization term. Lasso stands for Least Absolute Shrinkage 
    and Selection Operator.

    Attributes:
        alpha (float): regularization factor; i.e. factor that multiplies the regularization term.
    """

    def __init__(self, alpha: float = 0.01):
        """Initialize a `L1LassoRegularizer` class instance.

        Args:
            alpha (float, default=0.01): regularization factor (0 < alpha < 1).
        """

        super(L1LassoRegularizer, self).__init__(alpha = alpha)

    def __call__(self, theta: np.ndarray) -> float:
        return self.alpha * np.sum(np.abs(theta))

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        return self.alpha * np.sign(theta)

class L2RidgeRegularizer(Regularizer):
    """l2 norm (Ridge) regularization term, also known as Tikhonov regularization.

    Attributes:
        alpha (float): regularization factor; i.e. factor that multiplies the regularization term.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, theta: np.ndarray) -> float:
        return self.alpha * 0.5 * theta.T.dot(theta)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        return self.alpha * theta
