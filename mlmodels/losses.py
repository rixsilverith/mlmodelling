"""Loss functions module

This module contains the implementation of several loss/cost functions used
to compute errors during model training.
"""

from abc import ABC, abstractmethod
import numpy as np

from mlmodels.activations import Softmax

class Loss(ABC):
    """Base class for a loss function."""

    @abstractmethod
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    @property
    def name(self) -> str:
        return type(self).__name__

class SquareLoss(Loss):
    """Square loss function."""

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the square loss between the predicted response vector `y_pred` and the true
        response vector `y_true`.

        Args:
            y_pred (np.ndarray (m,)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m,)): response vector containg the corresponding m true values.

        Returns:
            np.ndarray (m,): vector containing the square loss between each `y_pred` and `y_true`.
        """

        return 0.5 * np.power((y_pred - y_true), 2)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of the square loss between the predicted response vector `y_pred`
        and the true response vector `y_true`.

        Args:
            y_pred (np.ndarray (m,)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m,)): response vector containg the corresponding m true values.

        Returns:
            np.ndarray (m,): vector containing the gradients of the  square loss between each
            `y_pred` and `y_true`.
        """

        return - (y_pred - y_true)

class BinaryCrossEntropy(Loss):
    """Binary cross-entropy loss function.

    The binary cross-entropy loss function is mainly used in binary classification problems.
    It can be derived via maximum likelihood estimation.
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the binary cross-entropy loss between `y_pred` and `y_true`.

        Clips each value in the given prediction vector to avoid division by 0.

        Args:
            y_pred (np.ndarray (m,)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m,)): response vector containg the corresponding m true values.

        Returns:
            np.ndarray (m,): binary cross-entropy loss between each `y_pred` and `y_true`.
        """

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of the binary cross-entropy loss between `y_pred` and `y_true`.

        Clips each value in the given prediction vector to avoid division by 0.

        Args:
            y_pred (np.ndarray (m,)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m,)): response vector containg the corresponding m true values.

        Returns:
            np.ndarray (m,): vector containing the gradients of the binary cross-entropy loss
            between each `y_pred` and `y_true`.
        """

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)

class CategorialCrossEntropy(Loss):
    """Categorical cross-entropy loss function.

    The categorical cross-entropy loss function is mainly used in multi-class classification
    problems.
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray, from_logits: bool = False):
        """Compute the categorical cross-entropy between the predicted response vector `y_pred`
        and the true response vector `y_true`.

        Args:
            y_pred (np.ndarray (m,)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m,)): response vector containg the corresponding m true values.
            from_logits (bool, default=False): whether to interpret the predicted response vector
                as logits. If true, a softmax activation is applied to `y_pred` to obtain a
                probability distribution.
        """

        if from_logits:
            Softmax()(y_pred)

        return - np.sum(y_true * np.log(y_pred))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of the categorical cross-entropy loss between the predicted response
        vector `y_pred` and the true response vector `y_true`.

        Args:
            y_pred (np.ndarray (m,)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m,)): response vector containg the corresponding m true values.

        Returns:
            np.ndarray (m,): vector containing the gradients of the categorical cross-entropy loss
            between each `y_pred` and `y_true`.
        """
        pass
