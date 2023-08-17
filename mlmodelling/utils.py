"""Utils module

This module implements general utility functions used through the library.
"""

from typing import Dict, Any
import numpy as np

def stringify_config(config: Dict[str, Any]) -> str:
    """Convert a model configuration dictionary into a string. """

    config_str: str = ""
    for key in config:
        if isinstance(config[key], dict):
            subconfig = config[key]
            if 'name' in subconfig:
                itemname = subconfig['name']
                del subconfig['name']
            config_str += f'{key}: {itemname}\n └── '
            for subkey in subconfig:
                config_str += f'{subkey}: {subconfig[subkey]}, '
            config_str = config_str[:len(config_str) - 2] + "\n"
            continue
        config_str += f'{key}: {config[key]}\n'

    return config_str

def accuracy_score(y_pred, y_true):
    return np.mean(y_pred == y_true.reshape(-1, 1))

def entropy(y: np.ndarray) -> float:
    """Compute the entropy of the input vector `y`. """

    entropy = 0
    for cls in np.unique(y):
        cls_proportion = len(y[y == cls]) / len(y)
        entropy += - cls_proportion * np.log2(cls_proportion)
    return entropy

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X. """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def standardize(X):
    """ Standardize the dataset X. """
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std
