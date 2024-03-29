![Python version](https://img.shields.io/badge/python-3.7%2B-g)
[![License](https://img.shields.io/github/license/rixsilverith/mlmodelling?color=g)](https://github.com/rixsilverith/mlmodelling/blob/main/LICENSE)

# mlmodelling: Experimental machine learning library in NumPy

*mlmodelling* is a Python package that provides readable yet efficient
implementations of fundamental models used in machine learning, written using the [NumPy](https://numpy.org/)
scientific computing package.

Despite being fully usable, the models implemented in this library are not
meant for production environments, but as a way of understanding the internals
and working principles of the implementations that can be found in widely used
machine learning libraries, such as [scikit-learn](https://scikit-learn.org/stable/index.html)
and [PyTorch](https://pytorch.org/).

---

## Installation

Although not strictly necessary, it is highly recommended to perform the
installation locally in a virtual environment, which can be initialized as
follows using the `venv` package.

```bash
git clone https://github.com/rixsilverith/mlmodelling
cd mlmodelling

# optional: initialize a virtual environment using venv
python3 -m venv .venv
source .venv/bin/activate

# installation using pip
pip3 install .
```

### Installation for development

Installation for development can be done by adding the `-e` flag to `pip` as

```bash
pip3 install -e .
```

instead of running the usual `pip install` command.

### Requirements

*mlmodelling* depends on the following packages:

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
Logistic Regression | [`LogisticRegressionClassifier`](mlmodelling/linear_models/logistic_regression.py) | Binary classification
Linear Regression | [`LinearRegressor`](mlmodelling/linear_models/regression.py) | Regression
Polynomial Regression | [`PolynomialRegressor`](mlmodelling/linear_models/regression.py) | Regression
Lasso (L1) Regression | [`LassoRegressor`](mlmodelling/linear_models/regression.py) | Regression
Ridge (L2) Regression | [`RidgeRegressor`](mlmodelling/linear_models/regression.py) | Regression

### Tree-based models

**Model** | **Implementation** | **Used for**
--- | --- | ---
Classification Decision Tree | [`DecisionTreeClassifier`](mlmodelling/decision_trees/decision_tree.py) | Classification
Regression Decision Tree | [`DecisionTreeRegressor`](mlmodelling/decision_trees/decision_tree.py) | Regression
Random Forest Classifier | [`RandomForestClassifier`](mlmodelling/decision_trees/random_forest.py) | Classification
Random Forest Regressor | [`RandomForestRegressor`](mlmodelling/decision_trees/random_forest.py) | Regression

### Neighbor-based models

**Model** | **Implementation** | **Used for**
--- | --- | ---
K-Nearest Neighbors Classifier | [`KNeighborsClassifier`](mlmodelling/neighbors/k_nearest_neighbors.py) | Classification
K-Nearest Neighbors Regressor | [`KNeighborsRegressor`](mlmodelling/neighbors/k_nearest_neighbors.py) | Regression

---

## License

*mlmodelling* is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
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

