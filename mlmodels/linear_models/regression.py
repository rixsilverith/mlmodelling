"""Regression linear models module.

This module implements the linear regression model and variants, such as lasso and
Ridge regression.
"""

from mlmodels.linear_models import LinearModel
from mlmodels.optimizers import GradientBasedOptimizer, StochasticGradientDescent
from mlmodels.activations import Identity
from mlmodels.losses import SquareLoss
from mlmodels.regularizers import L2Ridge

class LinearRegressor(LinearModel):
    """A linear regressor without regularization. """

    def __init__(self, optimizer: GradientBasedOptimizer = StochasticGradientDescent()):
        super().__init__(phi = Identity(), optimizer = optimizer, loss = SquareLoss(),
            regularizer = L2Ridge(alpha = 0.))
