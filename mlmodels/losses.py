"""Loss functions module

This module contains the implementation of several loss/cost functions used
to compute errors during model training.
"""

from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    """Base class for a loss function."""

    @abstractmethod
    def __call__(self, y_pred, y_true):
        pass

    @abstractmethod
    def gradient(self, y_pred, y_true):
        pass

class CategorialCrossentropy(Loss):
    """Categorical cross-entropy loss function.

    Mainly used as loss function in classification problems.
    """

    def __init__(self):
        self.name = 'CategoricalCrossentropy'

    def __call__(self, y_pred, y_true, from_logits=False):
        """Compute the categorical cross-entropy between the predicted label vector `y_pred`
        and the true label vector `y_true`.
        """
    
        if from_logits:
            Softmax()(y_pred)

        return - np.sum(y_true * np.log(y_pred))

class BinaryCrossentropy():
    """Binary cross-entropy loss function.

    It is used in the logistic regression model. It can be derived using the maximum
    likelihood estimation method.
    """

    def __init__(self):
        self.name = 'BinaryCrossentropy'

    def __call__(self, y_pred, y_true):
        """Compute the binary cross-entropy loss between `y_pred` and `y_true`.

        Clips each value in the given prediction vector to avoid division by 0.

        Args:
            y_pred (ndarray (m,)): numpy array containing m predicted values by a model.
            y_true (ndarray (m,)): numpy array containg m actual values from the dataset.

        Returns:
            float: categorical binary cross-entropy cost (i.e. mean of the binary cross-entropy
                losses for each predicted value in `y_pred`).
        """

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return np.mean(- y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

class BinaryCrossEntropy():
    """Categorical binary cross-entropy loss function.

    It is used in the logistic regression model. It can be derived using the maximum
    likelihood estimation method.
    """

    def __call__(self, y_pred, y_true):
        """Call method for BinaryCrossEntropy class.

        Clips each value in the given prediction vector to avoid division by 0.

        Args:
            y_pred (ndarray (m,)): numpy array containing m predicted values by a model.
            y_true (ndarray (m,)): numpy array containg m actual values from the dataset.

        Returns:
            float: categorical binary cross-entropy cost (i.e. mean of the binary cross-entropy
                losses for each predicted value in `y_pred`).
        """

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return np.mean(- y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
