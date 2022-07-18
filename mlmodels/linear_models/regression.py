"""Regression linear models module.

This module implements the linear regression model and variants, such as lasso and
Ridge regression.
"""

import math
from abc import ABC, abstractmethod
import numpy as np

from mlmodels.linear_models import LinearModel
from mlmodels.optimizers import GradientBasedOptimizer, StochasticGradientDescent
from mlmodels.activations import Identity
from mlmodels.losses import SquareLoss
from mlmodels.regularizers import L2RidgeRegularizer

class LinearRegressor(LinearModel):
    """A linear regressor without regularization. """

    def __init__(self, optimizer: GradientBasedOptimizer = StochasticGradientDescent()):
        super(LinearRegressor, self).__init__(
            phi = Identity(), optimizer = optimizer, loss = SquareLoss(), 
            regularizer = L2RidgeRegularizer(alpha = 0.))

