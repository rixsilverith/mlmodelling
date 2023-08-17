"""Random forest classifier example. """

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

import mlmodelling

def main():
    X, y = datasets.make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=3, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    model = mlmodelling.decision_trees.RandomForestClassifier(criterion = 'entropy',
        n_estimators = 250, min_samples_split = 2, max_depth = 3)

    model.fit(X_train, y_train)
    model.summary()

    y_pred = model.predict(X_test)
    print(f'acc score: {np.mean(y_pred == y_test):.2f}')

if __name__ == "__main__":
    main()

