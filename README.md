[![License](https://img.shields.io/github/license/rixsilverith/machine-learning-models)](https://mit-license.org/)

# Machine learning models (mlmodels)

*mlmodels* is a Python library that implements several machine learning models and algorithms from scratch using the [NumPy](https://numpy.org/)
package for efficient numerical computing and [matplotlib](https://matplotlib.org/) for data visualization.

Despite being fully functional, these implementations are not meant be used in production machine learning systems, but as a way of understanding
the internals and working principles of the algorithms available in well known frameworks such as
[scikit-learn](https://scikit-learn.org/stable/index.html), [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/).

---

## Installation

The *mlmodels* package can be installed by running the following sequence of commands.

```bash
$ git clone https://github.com/rixsilverith/machine-learning-models
$ cd machine-learning-models
$ python3 setup.py install
```

> **Note** You may need to run the `setup.py` file with root priviledges.

### Requirements

*mlmodels* depends on the following packages:

- [numpy](https://numpy.org/) - Efficient numerical computing library for Python.
- [matplotlib](https://matplotlib.org/) - Plotting library for Python

---

## Some examples

### Logistic Regression model

```bash
$ python3 examples/logistic_regression.py
```
This example generates a synthetic dataset suited for binary classification, fits a logistic regression model
using the Stochastic Gradient Descent optimizer according to the generated data and plots the estimated decision 
boundary for the problem.

```
+------------------------------+
| LogisticRegressionClassifier |
+------------------------------+
phi (activation): Sigmoid
optimizer: StochasticGradientDescent
 └── learning_rate: 0.01, momentum: 0.0, nesterov: False
loss: BinaryCrossEntropy
regularizer: L2Ridge
 └── alpha: 0.01
```

<p><img src="imgs/logistic_regression_example.png" width="440"\></p>

---

## List of implemented models

The following is a list of the currently implemented models.

### Linear models

**Model** | **Implementation** | **Used for**
--- | --- | --- 
Logistic Regression | [`LogisticRegressionClassifier`](mlmodels/linear_models/logistic_regression.py) | Binary classification
Linear Regression | [`LinearRegressor`](mlmodels/linear_models/regression.py) | Regression
Polynomial Regression | [`PolynomialRegressor`](mlmodels/linear_models/regression.py) | Regression
Lasso (L1) Regression | [`LassoRegressor`](mlmodels/linear_models/regression.py) | Regression
Ridge (L2) Regression | [`RidgeRegressor`](mlmodels/linear_models/regression.py) | Regression

### Tree-based models

**Model** | **Implementation** | **Used for**
--- | --- | --- 
Classification Decision Tree | [`DecisionTreeClassifier`](mlmodels/decision_trees/decision_tree.py) | Classification
Regression Decision Tree | [`DecisionTreeRegressor`](mlmodels/decision_trees/decision_tree.py) | Regression
Random Forest Classifier | [`RandomForestClassifier`](mlmodels/decision_trees/random_forest.py) | Classification
Random Forest Regressor | [`RandomForestRegressor`](mlmodels/decision_trees/random_forest.py) | Regression

### Neighbor-based models

**Model** | **Implementation** | **Used for**
--- | --- | --- 
K-Nearest Neighbors Classifier | [`KNeighborsClassifier`](mlmodels/neighbors/k_nearest_neighbors.py) | Classification
K-Nearest Neighbors Regressor | [`KNeighborsRegressor`](mlmodels/neighbors/k_nearest_neighbors.py) | Regression

---

## License

*mlmodels* is licensed under the MIT License. See [LICENSE](LICENSE) for more information. 
A copy of the license can be found along with the code.

---

## References

- Deisenroth, M. P., Faisal, A. A.,, Ong, C. S. (2020). *Mathematics for Machine Learning*. Cambridge University Press.
- Hastie, T., Tibshirani, R.,, Friedman, J. (2009). *The elements of statistical learning: data mining, inference and prediction*. Springer.
- Ruder, S. (2016). *An overview of gradient descent optimisation algorithms*. [arxiv.org/pdf/1609.04747.pdf](https://arxiv.org/pdf/1609.04747.pdf).
- Sutskever, I., Martens, J., Dahl, G. E. & Hinton, G. E. (2013). *On the importance of initialization and momentum in deep learning*. ICML-13. Vol 28. (2013): pp. 1139-1147. [[pdf]](https://www.cs.utoronto.ca/~ilya/pubs/2013/1051_2.pdf)
- Sutskever, I. (2013). *Training Recurrent neural Networks*. PhD Thesis. University of Toronto. [[pdf]](https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
- Glorot, X. & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks*. In Y. W. Teh & M. Titterington (eds.), Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics: pp. 249-256.
- Breiman, L. (2001). *Random Forests*. Machine learning, 45, 5-32. doi: 10.1023/A:1010933404324
