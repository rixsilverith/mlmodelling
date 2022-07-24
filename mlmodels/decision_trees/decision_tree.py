"""Decision tree module

This module implements the CART (Classification and Regression Tree) model for
decision trees. `DecisionTree` is an abstraction from which inherit the concrete
`DecisionTreeClassifier` and `DecisionTreeRegressor` models.
"""

from __future__ import annotations
import math
from abc import ABC
import numpy as np
from terminaltables import AsciiTable

from mlmodels import BaseModel
from mlmodels.utils import entropy

class DecisionTreeNode():
    """Node of a decision tree. 

    An internal decision node contains the index of the feature on which the condition is 
    evaluated and the corresponding threshold for its value. Also, an internal node contains
    a reference to their true and false branches, i.e. the branch containg samples in the 
    dataset that fulfills the condition and the branch containg samples that don't.

    A leaf decision node doesn't contain a condition, but a value that may be an integer
    when the decision tree is used for classification (in that case the value corresponds to 
    a class/category) or a float when the decision tree is used for regression (case in which
    it corresponds to the predicted value for a given sample).
    """

    def __init__(self: Self, feature_index: int = None, threshold: float = None, value: float = None, 
        true_branch = None, false_branch = None) -> Self:
        """Initialize a `DecisionTreeNode` class instance.
    
        Args:
            feature_index (int): index in a feature vector of the feature on which condition
                is evaluated.
            threshold (float): corresponding value of the feature at `feature_index` on which
                the condition is evaluated.
            value (float): class/category (classification) or response value (regression) of 
                the sample that arrives at this node takes. 
            true_branch (DecisionTreeNode): subtree containing the elements in the dataset that
                fulfills the condition in the node.
            false_branch (DecisionTreeNode): subtree containing the elements in the dataset 
                that don't fulfill the condition in the node.
        """

        self.feature_index: int = feature_index
        self.threshold: float = threshold
        self.value: float = value
        self.true_branch: Self = true_branch
        self.false_branch: Self = false_branch

class DecisionTree(BaseModel, ABC):
    """Decision tree abstract model.

    See the `DecisionTreeClassifier` and `DecisionTreeRegressor` concrete classes for decision 
    trees used for classification and regression, respectively.

    Decision trees and tree-based models, such as random forests and XGBoost, currently outperform
    deep learning on tabular data. See Grinsztajn, L. (2022) (https://arxiv.org/abs/2207.08815).
    """

    def __init__(self, min_samples_split: int = 2, max_depth: int = None, min_impurity_decrease: float = .0):
        """Initialize a `DecisionTree` class instance.

        Args:
            min_samples_split (int): minimum number of samples required to split an internal node.
            max_depth (int): maximum depth of the decision tree. If None, nodes are expanded either
                until all leaves are pure or until all leaves contain less than min_samples_split 
                samples.
            min_impurity_decrease (float):
        """

        self.root: DecisionTreeNode = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.depth = 0
        self.min_impurity_decrease = min_impurity_decrease

        self.impurity_criterion = None
        self.leaf_value_criterion = None

    @property
    def name(self) -> str:
        return type(self).__name__

    def summary(self):
        """Print a summary containing model information. """

        print(AsciiTable([[f'{self.name}']]).table)
        print('min_samples_split:', self.min_samples_split)
        print('max_depth:', self.max_depth)
        print('min_impurity_decrease:', self.min_impurity_decrease)
        print('depth:', self.depth + 1)
        print('impurity_criterion:', self.impurity_criterion.__name__)
        print('leaf_value_criterion:', self.leaf_value_criterion.__name__)
        print()
        self.print_tree()

    def fit(self, X, y):
        """Fit the decision tree according to the provided training data. """

        self.root = self._build_tree(X, y)

    def _compute_entropy(self, y):
        proportions = np.bincount(y) / len(y)
        return - np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _split_on_feature(self, X, feature_i, threshold):
        """Split the dataset based on whether X[:, feature_i] is larger than the fixed threshold. """
    
        X_true = np.array([sample for sample in X if sample[feature_i] >= threshold])
        X_false = np.array([sample for sample in X if sample[feature_i] < threshold])

        return X_true, X_false

    def _compute_best_split(self, X, y):
        """Compute the best possible dataset split; i.e. the split that minimizes impurity. """

        Xy = np.concatenate((X, y.reshape(-1, 1)), axis = 1)
        n_samples, n_features = X.shape

        max_impurity = 0
        best_split = {}

        for feature_index in range(n_features):
            feature_values = X[:, feature_index].reshape(-1, 1)

            for threshold in np.unique(feature_values):
                Xy_true, Xy_false = self._split_on_feature(Xy, feature_index, threshold)

                if len(Xy_true) > 0 and len(Xy_false) > 0:
                    y_true, y_false = Xy_true[:, -1], Xy_false[:, -1]
                    impurity = self.impurity_criterion(y, y_true, y_false)

                    if impurity > max_impurity:
                        max_impurity = impurity
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['X_true'] = Xy_true[:, :n_features]
                        best_split['y_true'] = y_true
                        best_split['X_false'] = Xy_false[:, :n_features]
                        best_split['y_false'] = y_false
                        best_split['impurity'] = impurity

        return best_split


    def _build_tree(self, X, y, current_depth=0):
        """Recursively build the decision tree and split the feature matrix X and the response 
        vector `y` on the feature of `X` which best separates the dataset. """

        n_samples = X.shape[0]
        self.depth = current_depth

        max_impurity = 0
        best_split = None
        
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            best_split = self._compute_best_split(X, y)
            if best_split:
                imp = best_split['impurity']
                if imp > max_impurity:
                    max_impurity = imp

        if max_impurity > self.min_impurity_decrease:
            true_branch = self._build_tree(best_split['X_true'], best_split['y_true'], current_depth + 1)
            false_branch = self._build_tree(best_split['X_false'], best_split['y_false'], current_depth + 1)

            return DecisionTreeNode(feature_index = best_split['feature_index'], threshold = best_split['threshold'],
                true_branch = true_branch, false_branch = false_branch)

        leaf_value = self.leaf_value_criterion(y)
        return DecisionTreeNode(value = leaf_value)

    def predict_label(self, x, node = None):
        """Classify a single sample. """

        if node.value is not None: # leaf aka decision
            return node.value

        branch = node.false_branch
        if x[node.feature_index] >= node.threshold:
            branch = node.true_branch

        return self.predict_label(x, node = branch)

    def predict(self, X):
        """Classify element-wise the samples in the feature matrix using the classification tree. """

        return np.array([self.predict_label(sample, node = self.root) for sample in X])

    def print_tree(self, tree = None, indent = "-"):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("X_%s <= %.3f" % (tree.feature_index, tree.threshold))
            print("%s T -> " % (indent), end = "")
            self.print_tree(tree.true_branch, indent + indent)
            print("%s F -> " % (indent), end = "")
            self.print_tree(tree.false_branch, indent + indent)

class DecisionTreeRegressor(DecisionTree):
    """Decision tree regressor model. """

    def variance_reduction(self, y_node, y_true, y_false):
        """Compute the variance reduction of the split of `y_node` into `y_true` and `y_false`. """

        total_variance = np.var(y_node)
        y_true_variance = np.var(y_true)
        y_false_variance = np.var(y_false)
        y_true_prop, y_false_prop = len(y_true) / len(y), len(y_false) / len(y)
        
        true_variance_factor = y_true_prop * y_true_variance
        false_variance_factor = y_false_prop * y_false_variance

        variance_reduction = total_variance - (true_variance_factor + false_variance_factor)
        return np.sum(variance_reduction)

    def mean_y(self, y):
        """Computes the mean of `y`. Used as the leaf value computation criterion. """

        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self: Self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the decision tree classifier according to the provided training data. """

        self.impurity_criterion = self.variance_reduction
        self.leaf_value_criterion = self.mean_y
        super().fit(X, y)
    
class DecisionTreeClassifier(DecisionTree):
    """Decision tree classifier model. """

    def __init__(self: Self, criterion: str = 'gini', min_samples_split: int = 2, max_depth: int = None, 
        min_impurity_decrease: float = .0) -> Self:
        """Initialize a `DecisionTreeClassifier` class instance.
        """

        if criterion not in ['gini', 'entropy']:
            raise ValueError(f'criterion must be either "gini" or "entropy". Got: {criterion}')

        self.criterion = criterion

        super().__init__(min_samples_split = min_samples_split, max_depth = max_depth, 
            min_impurity_decrease = min_impurity_decrease)

    def gini_index(self: Self, y: np.ndarray) -> float:
        """Compute the Gini index of the given vector `y`. """

        gini = 0
        for cls in np.unique(y):
            cls_proportion = len(y[y == cls]) / len(y)
            gini += cls_proportion ** 2
        return 1 - gini

    def information_gain(self: Self, y_node: np.ndarray, y_true: np.ndarray, y_false: np.ndarray) -> float:
        """Compute the information gain of the split of `y_node` into `y_true` and `y_false`. """

        true_p = len(y_true) / len(y_node)
        false_p = len(y_false) / len(y_node)
        if self.criterion == 'gini': 
            impurity_measure = self.gini_index
        else: impurity_measure = entropy

        return impurity_measure(y_node) - true_p * impurity_measure(y_true) \
            - (1 - true_p) * impurity_measure(y_false)

    def most_common_class(self: Self, y: np.ndarray) -> float:
        """Compute the most common class in the response vector `y`. """

        y = list(y) 
        return max(y, key = y.count)

    def fit(self: Self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the decision tree classifier according to the provided training data. """

        self.impurity_criterion = self.information_gain
        self.leaf_value_criterion = self.most_common_class
        super().fit(X, y)
