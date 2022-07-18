[![License](https://img.shields.io/github/license/rixsilverith/machine-learning-models)](https://mit-license.org/)

# Machine learning models

Implementations of several machine learning models and algorithms from scratch in Python using the [NumPy](https://numpy.org/)
library for efficient numerical computations and [matplotlib](https://matplotlib.org/) for data visualization.

These implementations are not meant be used in production machine learning systems, but as a way of understanding
the internals and working principles of the algorithms available in well known frameworks such as
[TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/).

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

<p><img src="imgs/logistic_regression_example.png" width="540"\></p>

---

## List of implemented models

The following is a list of the currently implemented models.

### Linear models

**Model** | **Implementation** | **Used for**
--- | --- | --- 
Logistic Regression | [`LogisticRegressionClassifier`](mlmodels/linear_models/logistic_regression.py) | Binary classification
Linear Regression | [`LinearRegressor`](mlmodels/linear_models/regression.py) | Regression
Lasso (L1) Regression | [`LassoRegressor`](mlmodels/linear_models/regression.py) | Regression
Ridge (L2) Regression | [`RidgeRegressor`](mlmodels/linear_models/regression.py) | Regression

---

## License

*mlmodels* is licensed under the MIT License. See [LICENSE](LICENSE) for more information. 
A copy of the license can be found along with the code.

---

## References

- Ruder, S. "An overview of gradient descent optimisation algorithms". (2016). [arxiv.org/pdf/1609.04747.pdf](https://arxiv.org/pdf/1609.04747.pdf).
- Sutskever, I. et al. "On the importance of initialization and momentum in deep learning". ICML-13. Vol 28. (2013): pp. 1139-1147. 
[[pdf]](https://www.cs.utoronto.ca/~ilya/pubs/2013/1051_2.pdf)
- Sutskever, I. "Training Recurrent neural Networks". PhD Thesis. (2013). [[pdf]](https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
