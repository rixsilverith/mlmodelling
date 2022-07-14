import numpy as np

def accuracy_score(y_pred, y):
    return np.mean(y_pred == y)
