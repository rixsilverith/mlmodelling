"""Regularizers module."""

from abc import ABC, abstractmethod

class Regularizer(ABC):
    """Base regularizer."""

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod 
    def gradient(self, x):
        pass
