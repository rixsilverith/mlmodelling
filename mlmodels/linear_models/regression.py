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
from mlmodels.losses import BinaryCrossentropy

class LinearRegressor(LinearModel):
    """A linear regressor without regularization."""

    def __init__(self, optimizer: GradientBasedOptimizer=StochasticGradientDescent()):
        regularization = lambda x: 0
        regularization.gradient = lambda x: 0
        regularization.name = 'None'

        super(LinearRegressor, self).__init__( 
            phi=Identity(), optimizer=optimizer, loss=BinaryCrossentropy(), regularizer=regularization)

