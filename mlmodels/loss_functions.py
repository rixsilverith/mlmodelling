import numpy as np

class BinaryCrossEntropy():
    def __call__(self, y_pred, y):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) # avoid division by 0
        return np.mean(self.loss(y_pred, y))

    def loss(self, y_pred, y):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) # avoid division by 0
        return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
