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

def accuracy_score(y_pred, y):
    return np.mean(y_pred == y)
