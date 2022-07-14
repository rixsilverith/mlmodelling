"""Loss functions module

This module contains the implementation of several loss/cost functions used
to compute errors during model training.
"""

import numpy as np

class BinaryCrossEntropy():
    """Categorical binary cross-entropy loss function.

    It is used in the logistic regression model. It can be derived using the maximum
    likelihood estimation method.
    """
    def __call__(self, y_pred, y):
        """Call method for BinaryCrossEntropy class.

        Clips each value in the given prediction vector to avoid division by 0.
        
        Args:
            y_pred (ndarray (m,)): numpy array containing m predicted values by a model.
            y (ndarray (m,)): numpy array containg m actual values from the dataset.

        Returns:
            float: categorical binary cross-entropy cost (i.e. mean of the binary cross-entropy
                losses for each predicted value in `y_pred`).
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return np.mean(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))
