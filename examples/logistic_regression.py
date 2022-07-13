import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from mlmodels.linear_models import LogisticRegressionClassifier
from mlmodels.loss_functions import BinaryCrossEntropy

def main():
    X, y = make_classification(n_classes=2, n_features=2, n_informative=2, 
                               n_redundant=0, n_samples=300, random_state=66)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegressionClassifier(learning_rate=0.01).fit(X_train, y_train)
    pred_v = model.predict(X_test)
    bce = BinaryCrossEntropy()(y_test, pred_v)

    y_predicted = model.predict(X)

    x1 = X_train[:, 0] # feature 1
    x2 = X_train[:, 1] # feature 2
    yp = np.array(y_train).astype(int)

    viridis = plt.get_cmap('viridis')

    plt.scatter(x1[yp == 0], x2[yp == 0], color=viridis(0.9), s=20, label='Class 0')
    plt.scatter(x1[yp == 1], x2[yp == 1], color=viridis(0.5), s=20, label='Class 1')

    plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
    plot_y = (-1. / model.coefficients[2]) * (model.coefficients[1] * plot_x + model.coefficients[0])

    plt.plot(plot_x, plot_y, color='black', linewidth=2, label='Decision boundary')
    plt.ylim([-4, 3])

    plt.suptitle('LogisticRegressionClassifier', fontsize=15)
    plt.title('Binary Cross-Entropy error: %.2f' % bce, fontsize=10)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()

