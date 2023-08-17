import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from mlmodelling.linear_models import LogisticRegressionClassifier
from mlmodelling.optimizers import StochasticGradientDescent
from mlmodelling.utils import accuracy_score

def main():
    X, y = datasets.make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=3, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    optim = StochasticGradientDescent(learning_rate=0.01, momentum=0.9, nesterov=True)
    model = LogisticRegressionClassifier(optimizer=optim).fit(X_train, y_train, epochs = 6000)
    model.summary()

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_pred, y_test)

    viridis = plt.get_cmap('viridis')

    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color=viridis(0.5), s=20, label='Class 0')
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color=viridis(0.9), s=20, label='Class 1')

    dboundary_x = np.array([min(X[:, 0]), max(X[:, 0])])
    dboundary_y = (-1. / model.coefficients[2]) * (model.coefficients[1] * dboundary_x + model.coefficients[0])
    plt.plot(dboundary_x, dboundary_y, color='black', linewidth=2, label='Decision boundary')

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    plt.suptitle('LogisticRegressionClassifier', fontsize=15)
    plt.title(f'Accuracy score: {float(acc):.2f}', fontsize=12)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()

