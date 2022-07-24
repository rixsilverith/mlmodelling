"""Decision tree module

This module implements the decision tree model.
"""

import math
from abc import ABC
import numpy as np
from terminaltables import AsciiTable

from mlmodels import BaseModel

def entropy(y):
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy

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

    def __init__(self, feature_index: int = None, threshold: float = None, value: float = None, 
        true_branch = None, false_branch = None):
        """Initialize a `DecisionTreeNode` class instance.
    
        Args:
            feature_index (int):
            threshold (float):
            value (float):
            true_branch (DecisionTreeNode):
            false_branch (DecisionTreeNode):
        """

        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

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

        return [X_true, X_false]

        '''
        left_index = np.argwhere(X <= threshold).flatten()
        right_index = np.argwhere(X > threshold).flatten()
        return left_index, right_index
        '''

    def _compute_information_gain(self, X, y, threshold):
        """Compute the information gain of splitting the node on a given feature. """

        '''
        parent_loss = self._compute_entropy(y)
        left_index, right_index = self._split_on_feature(X, threshold)
        n, n_left, n_right = len(y), len(left_index), len(right_index)

        if n_left == 0 or n_right == 0:
            return 0

        child_loss = (n_left / n) * self._compute_entropy(y[left_index]) + (n_right / n) * self._compute_entropy(y[right_index])
        return parent_loss - child_loss
        '''

    '''
    def _best_split(self, X, y, features):
        split = {'score': -1, 'feat': None, 'threshold': None}
        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._compute_information_gain(X_feat, y, thresh)
                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['threshold'] = thresh

        return split['feat'], split['threshold']
    '''

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

        #print(best_split)
        return best_split


    def _build_tree(self, X, y, current_depth=0):
        """Recursively build the decision tree and split the feature matrix X and the response 
        vector `y` on the feature of `X` which best separates the dataset. """

        Xy = np.concatenate((X, y.reshape(-1, 1)), axis=1) # concat X and y to easily perform data split
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.depth = current_depth

        max_impurity = 0
        best_split = None
        
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            best_split = self._compute_best_split(X, y)
            if best_split:
                imp = best_split['impurity']
                if imp > max_impurity:
                    max_impurity = imp

            '''
            # Compute the impurity (Gini index or information gain) for each feature
            for feature_i in range(n_features):
                feature_values = X[:, feature_i].reshape(-1, 1)

                for threshold in np.unique(feature_values):
                    # For each feature_i split the datsset depending if the feature value of X at
                    # index feature_i meets the threshold
                    Xy1, Xy2 = self._split_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y_left = Xy1[:, n_features:]
                        y_right = Xy2[:, n_features:]

                        impurity = self.impurity_criterion(y, y_left, y_right)
                        if impurity > max_impurity:
                            max_impurity = impurity
                            best_criteria = { 'feature_i': feature_i, 'threshold': threshold }
                            best_sets = {
                                'leftX': Xy1[:, :n_features],
                                'lefty': Xy1[:, n_features:],
                                'rightX': Xy2[:, :n_features],
                                'righty': Xy2[:, n_features:]
                            }
            '''

        if max_impurity > self.min_impurity_decrease:
            true_branch = self._build_tree(best_split['X_true'], best_split['y_true'], current_depth + 1)
            false_branch = self._build_tree(best_split['X_false'], best_split['y_false'], current_depth + 1)

            return DecisionTreeNode(feature_index = best_split['feature_index'], threshold = best_split['threshold'],
                true_branch = true_branch, false_branch = false_branch)

        leaf_value = self.leaf_value_criterion(y)
        return DecisionTreeNode(value = leaf_value)

        """
            random_feats = np.random.choice(n_features, n_features, replace=False)
            best_feat, best_thresh = self._bset_split(X, y, random_feats)

            left_index, right_index = self._split_on_feature(X[:, best_feat], best_thresh)
            left_child = self._build_tree(X[left_index, :], y[left_index], current_depth + 1)
            right_child = self._build_tree(X[right_index, :], y[right_index], current_depth + 1)

        most_common_label = np.argmax(np.bincount(y))
        return DecisionTreeNode(value=most_common_label)
        """

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
    
class DecisionTreeClassifier(DecisionTree):
    """Decision tree classifier model. """

    def __init__(self, criterion: str = 'gini', min_samples_split: int = 2, max_depth: int = None, min_impurity_decrease: float = .0):
        """Initialize a `DecisionTreeClassifier` class instance.
        """

        if criterion not in ['gini', 'entropy']:
            raise ValueError(f'criterion must be either "gini" or "entropy". Got: {criterion}')

        self.criterion = criterion

        super().__init__(min_samples_split = min_samples_split, max_depth = max_depth, min_impurity_decrease = min_impurity_decrease)

    def information_gain(self, y_node, y_true, y_false):
        """Compute the information gain of the split of `y_node` into `y_true` and `y_false`. """

        node_entropy = entropy(y_node)
        true_p = len(y_true) / len(y_node)
        info_gain = node_entropy - true_p * entropy(y_true) - (1 - true_p) * entropy(y_false)

        return info_gain

    def most_common_class(self, y: np.ndarray):
        """Compute the most common class in the response vector `y`. """

        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        """Fit the decision tree classifier according to the provided training data. """

        self.impurity_criterion = self.information_gain
        self.leaf_value_criterion = self.most_common_class

        return super().fit(X, y)

