"""Kernel module."""

import numpy as np
from abc import ABC

from typing import Any, Dict


class BaseKernel(ABC):
    """Base class for a kernel."""

    @property
    def name(self) -> str:
        """Name of the kernel function."""

        return type(self).__name__

    def get_config(self) -> Dict[str, Any]:
        """Get kernel configuration dictionary."""

        return {}


class LinearKernel(BaseKernel):
    """Linear kernel."""

    def __call__(self, x1, x2):
        return x1 @ x2


class PolynomialKernel(BaseKernel):
    """Polynomial kernel."""

    degree_ = None
    coef0_ = None
    gamma = None

    def __init__(self, degree: int = 3, gamma=1, coef0=1):
        self.degree_ = degree
        self.coef0_ = coef0
        self.gamma_ = gamma

    def __call__(self, x, y, degree: int = 3, coef0 = 1):
        return np.power(self.gamma_ * (x @ y) + self.coef0_, self.degree_)

    def get_config(self) -> Dict[str, Any]:
        return {'degree': self.degree_, 'coef0': self.coef0_, 'gamma': self.gamma_}


class RBFKernel(BaseKernel):
    """Radial Basis Function (RBF) kernel."""

    gamma_ = None

    def __init__(self, gamma):
        self.gamma_ = gamma

    def __call__(self, x, y):
        return np.exp(- self.gamma_ * np.linalg.norm(x - y) ** 2)

    def get_config(self) -> Dict[str, Any]:
        return {'gamma': self.gamma_}

