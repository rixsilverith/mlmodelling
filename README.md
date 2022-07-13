[![License](https://img.shields.io/github/license/rixsilverith/machine-learning-models)](https://mit-license.org/)

# Machine learning models

Implementations of several machine learning models and algorithms from scratch in Python using the [NumPy](https://numpy.org/)
library for efficient numerical computations and [matplotlib](https://matplotlib.org/) for data visualization.

These implementations are not meant be used in production machine learning systems, but as a way of understanding
the internals and working principles of the algorithms available in well known frameworks such as
[TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/).

---

## Installation

The `mlmodels` package can be installed by running the following sequence of commands.

```bash
$ git clone https://github.com/rixsilverith/machine-learning-models
$ cd machine-learning-models
$ python3 setup.py install
```

> **Note** You may need to run the `setup.py` file with root priviledges.

## List of implemented models

The following is a list of the currently implemented models.

### Linear models

**Model name** | **Implementation** | **Used for**
--- | --- | --- 
Logistic Regression | [`LogisticRegressionClassifier`](mlmodels/linear_models/logistic_regression.py) | Binary classification

---

## License

The MIT License. See [LICENSE](LICENSE) for more information.

