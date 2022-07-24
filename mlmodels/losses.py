"""Loss functions module

This module contains the implementation of several loss/cost functions used
to compute errors during model training.
"""

from abc import ABC, abstractmethod
import numpy as np

from mlmodels.activations import Softmax

class Loss(ABC):
    """Base class for a loss function. """

    @abstractmethod
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    @property
    def name(self) -> str:
        return type(self).__name__

class SquaredLoss(Loss):
    """Squared loss function. If a (m, 1) vector is given, the loss is computed element-wise. """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the square loss between the predicted response vector `y_pred` and the true
        response vector `y_true`.

        Args:
            y_pred (np.ndarray (m, 1)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m, 1)): response vector containg the corresponding m true values.

        Returns:
            np.ndarray (m, 1): vector containing the square loss between each `y_pred` and `y_true`.
        """

        return 0.5 * np.square(y_pred - y_true)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of the square loss between the predicted response vector `y_pred`
        and the true response vector `y_true`.

        Args:
            y_pred (np.ndarray (m, 1)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m, 1)): response vector containg the corresponding m true values.

        Returns:
            np.ndarray (m, 1): vector containing the gradients of the  square loss between each
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
            y_pred (np.ndarray (m, 1)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m, 1)): response vector containg the corresponding m true values.

        Returns:
            np.ndarray (m,): binary cross-entropy loss between each `y_pred` and `y_true`.
        """

        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient of the binary cross-entropy loss between `y_pred` and `y_true`.

        Clips each value in the given prediction vector to avoid division by 0.

        Args:
            y_pred (np.ndarray (m, 1)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m, 1)): response vector containg the corresponding m true values.

        Returns:
            np.ndarray (m,): vector containing the gradients of the binary cross-entropy loss
            between each `y_pred` and `y_true`.
        """

        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)

class CategorialCrossEntropy(Loss):
    """Categorical cross-entropy loss function. Sometimes referred to as the log-loss function.

    The categorical cross-entropy loss function is mainly used in multi-class classification
    problems.
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray, from_logits: bool = False):
        """Compute the categorical cross-entropy between the predicted response vector `y_pred`
        and the true response vector `y_true`.

        Args:
            y_pred (np.ndarray (m, 1)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m, 1)): response vector containg the corresponding m true values.
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
            y_pred (np.ndarray (m, 1)): response vector containg m predicted values by a model.
            y_true (np.ndarray (m, 1)): response vector containg the corresponding m true values.

        Returns:
            np.ndarray (m,): vector containing the gradients of the categorical cross-entropy loss
            between each `y_pred` and `y_true`.
        """
        pass
