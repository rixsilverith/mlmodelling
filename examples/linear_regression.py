import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

import mlmodels

def main():
    X, y = make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

    n_samples, n_features = np.shape(X)
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    optim = mlmodels.optimizers.StochasticGradientDescent(learning_rate=0.01)
    model = mlmodels.linear_models.LinearRegressor(optimizer=optim).fit(X_train, y_train, epochs=100)

    model.summary()

    y_pred = model.predict(X_test)
    mse = np.mean(mlmodels.losses.SquaredLoss()(y_pred, y_test))
    print('Mean squared error: %s' % (mse))

    y_pred_line = model.predict(X)

    viridis = plt.get_cmap('viridis')

    plt.scatter(X_train, y_train, color=viridis(0.5), s=20, label='Training data')
    plt.scatter(X_test, y_test, color=viridis(0.9), s=20, label='Test data')
    plt.plot(X, y_pred_line, color='black', linewidth=2, label='Model prediction')
    plt.suptitle('LinearRegressor', fontsize=15)
    plt.title('MSE: %.2f' % mse, fontsize=12)
    plt.xlabel('Feature 1')
    plt.ylabel('Response value')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
