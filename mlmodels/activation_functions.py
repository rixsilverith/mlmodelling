import numpy as np

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))
