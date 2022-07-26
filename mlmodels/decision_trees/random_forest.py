"""Random forest module

This module implements an abstract `RandomForest` model, from which inherit the 
concrete `RandomForestClassifier` and `RandomForestRegressor` models.
"""

from __future__ import annotations
from typing import Dict, List, Any
from abc import ABC
import math
import numpy as np

import progressbar

from mlmodels import BaseModel
from mlmodels.decision_trees import DecisionTree, DecisionTreeClassifier, DecisionTreeRegressor

progressbar_widgets = [ 'Fitting model: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='-', 
    left='[', right=']'), ' ', progressbar.ETA() ]

def random_subsets(X: np.ndarray, y: np.ndarray, n_subsets, replace = True):
    """Random subsets with replacement of the data. """

    n_samples = X.shape[0]
    Xy = np.concatenate((X, y.reshape(-1, 1)), axis = 1)
    subsets = []

    subsample_size = n_samples // 2
    if replace:
        subsample_size = n_samples

    for _ in range(n_subsets):
        sample_index = np.random.choice(range(n_samples), size = subsample_size, replace = replace)
        X = Xy[sample_index][:, :-1]
        y = Xy[sample_index][:, -1]
        subsets.append([X, y])
    return subsets

class RandomForest(BaseModel, ABC):
    """Random forest model. 

    References:
        Hastie, T., Tibshirani, R.,, Friedman, J. (2009). 
            "The elements of statistical learning: data mining, inference and prediction". Springer.
        Breiman, L. (2001). "Random Forests". Machine learning, 45, 5--32. doi: 10.1023/A:1010933404324
    """

    def __init__(self: Self, n_estimators: int = 100, max_features: int = None,
        min_samples_split: int = 2, max_depth: int = None, min_impurity_decrease: float = .0) -> Self:
        """Initialize a `RandomForest` class instance. """

        self.trees: List[DecisionTree] = []
        self.n_estimators: int = n_estimators
        self.max_features: int = max_features
        self.min_samples_split: int = min_samples_split
        self.max_depth: int = max_depth
        self.min_impurity_decrease: float = min_impurity_decrease

        self.progress = progressbar.ProgressBar(widgets = progressbar_widgets)

    def get_config(self: Self) -> Dict[str, Any]:
        """Get the configuration dictionary of the random forest. """

        return { 'n_estimators': self.n_estimators, 'max_features': self.max_features, 
            'min_samples_split': self.min_samples_split, 'max_depth': self.max_depth, 
            'min_impurity_decrease': float(self.min_impurity_decrease)}

    def fit(self: Self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the random forest according to the provided training data. """

        n_features = X.shape[1]
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features)) # See Breiman (2001).

        data_subsets = random_subsets(X, y, self.n_estimators)

        for i in self.progress(range(self.n_estimators)):
            X_subset, y_subset = data_subsets[i]
            idx = np.random.choice(range(n_features), size = self.max_features, replace = True)
            self.trees[i].feature_indices = idx
            X_subset = X_subset[:, idx]
            self.trees[i].fit(X_subset, y_subset)

    def predict(self: Self, X: np.ndarray) -> np.ndarray:
        """Predict using the random forest. """

        y_preds = np.empty((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            y_preds[:, i] = tree.predict(X[:, tree.feature_indices])

        y_pred = []
        for prediction in y_preds:
            # use most common class in classification and mean in regression
            y_pred.append(self.trees[0].leaf_value_criterion(prediction))
        return y_pred

class RandomForestClassifier(RandomForest):
    """Random forest classifier. """

    def __init__(self: Self, criterion: str = 'entropy', n_estimators: int = 100, max_features: int = None,
        min_samples_split: int = 2, max_depth: int = None, min_impurity_decrease: float = .0) -> Self:
        """Initialize a `RandomForestClassifier` class instance. """

        super().__init__(n_estimators = n_estimators, max_features = max_features, 
            min_samples_split = min_samples_split, max_depth = max_depth, 
            min_impurity_decrease = min_impurity_decrease)

        for _ in range(self.n_estimators):
            self.trees.append(
                DecisionTreeClassifier(criterion = criterion,
                    min_samples_split = self.min_samples_split, max_depth = self.max_depth,
                    min_impurity_decrease = self.min_impurity_decrease))

class RandomForestRegressor(RandomForest):
    """Random forest regressor. """

    def __init__(self: Self, n_estimators: int = 100, max_features: int = None,
        min_samples_split: int = 2, max_depth: int = None, min_impurity_decrease: float = .0) -> Self:
        """Initialize a `RandomForestRegressor` class instance. """

        super().__init__(n_estimators = n_estimators, max_features = max_features, 
            min_samples_split = min_samples_split, max_depth = max_depth, 
            min_impurity_decrease = min_impurity_decrease)

        for _ in range(self.n_estimators):
            self.trees.append(
                DecisionTreeRegressor(min_samples_split = self.min_samples_split, max_depth = self.max_depth,
                    min_impurity_decrease = self.min_impurity_decrease))
