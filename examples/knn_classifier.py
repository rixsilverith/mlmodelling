"""K-nearest neighbors classifier example. """

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np

import mlmodelling

def main():
    X, y = make_blobs(n_samples = 300, centers = 2, n_features = 2, cluster_std = 6, random_state = 11)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    model = mlmodelling.neighbors.KNeighborsClassifier(k_neighbors = 151).fit(X_train, y_train)
    model.summary()

    y_pred = model.predict(X_test)
    print(f'acc score: {float(np.mean(y_pred == y_test)):.2f}')

if __name__ == '__main__':
    main()
