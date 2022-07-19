"""Utils module

This module implements general utility functions used through the library.
"""

from typing import Dict, Any
import numpy as np

def stringify_config(config: Dict[str, Any]) -> str:
    """Convert a config object to a string. """

    config_str: str = ""
    for key in config:
        config_str += f'{key}: {config[key]}, '

    return config_str[:len(config_str) - 2] # remove colon in the last config attribute

def accuracy_score(y_pred, y_true):
    return np.mean(y_pred == y_true.reshape(-1, 1), axis = 0)

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def standardize(X):
    """ Standardize the dataset X """
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std
