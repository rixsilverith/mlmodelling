from matplotlib.pyplot import show
import numpy as np
import cvxopt
import itertools
from typing import Dict, Any, List
from numpy._typing import NDArray
from abc import ABC
from tqdm import trange

from mlmodelling import BaseModel
from mlmodelling.kernels import BaseKernel, LinearKernel, RBFKernel


class BaseSVM(BaseModel, ABC):
    """Base model for a support vector machine (SVM) model."""

    C_ = None
    lagrange_mults_ = None  # (n_support_vecs,)
    support_vectors_ = None  # (n_support_vecs, n_features)
    support_vector_labels_ = None  # (n_support_vecs,)
    intercept_ = None
    _kernel = None

    theta_ = None  # linear kernel parametrization (n_features,)

    def __init__(self, C, kernel):
        self.C_ = C
        self._kernel = kernel

    def get_config(self) -> Dict[str, Any]:
        return {'solver': 'cxvopt', 'kernel': {'name': self._kernel.name} | self._kernel.get_config() }


class BinarySVMClassifier(BaseSVM):
    """Binary Support Vector Machine classifier."""

    def __init__(self, C=1, kernel: BaseKernel = RBFKernel(gamma=0.5)):
        super().__init__(C=C, kernel=kernel)

    def _compute_kernel_matrix(self, X):
        """Compute the kernel matrix K of X."""

        n_instances = X.shape[0]
        kernel_matrix = np.zeros((n_instances, n_instances))
        for i, j in itertools.product(range(n_instances), repeat=2):
            kernel_matrix[i, j] = self._kernel(X[i], X[j])

        return kernel_matrix

    def _solve_quadratic(self, X, kernel_matrix, y):
        """Solve the SVM quadratic optimization problem."""

        n_instances = X.shape[0]

        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_instances) * -1)
        A = cvxopt.matrix(y, (1, n_instances), tc='d')
        b = cvxopt.matrix(0, tc='d')

        G_max = np.identity(n_instances) * -1
        G_min = np.identity(n_instances)
        G = cvxopt.matrix(np.vstack((G_max, G_min)))
        h_max = cvxopt.matrix(np.zeros(n_instances))
        h_min = cvxopt.matrix(np.ones(n_instances) * self.C_)
        h = cvxopt.matrix(np.vstack((h_max, h_min)))

        qp_sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        lagr_mults = np.ravel(qp_sol['x'])
        return lagr_mults

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the binary SVM model to data X, y."""

        n_instances, n_features = np.shape(X)
        n_classes = np.unique(y)
        if len(n_classes) != 2:
            raise RuntimeError(
                f'BinarySVMClassifier only supports datasets with 2 labels. Got: {n_classes}')

        kernel_matrix = self._compute_kernel_matrix(X)
        lagr_mults = self._solve_quadratic(X, kernel_matrix, y)

        # support vectors have nonzero Lagrange multipliers
        support_vecs = lagr_mults > 1e-7
        self.lagrange_mults_ = lagr_mults[support_vecs]
        self.support_vectors_ = X[support_vecs]
        self.support_vector_labels_ = y[support_vecs]

        inds = np.arange(len(lagr_mults))[support_vecs]

        self.intercept_ = 0
        for i in range(len(self.lagrange_mults_)):
            self.intercept_ += self.support_vector_labels_[i]
            self.intercept_ -= np.sum(self.lagrange_mults_ * self.support_vector_labels_ * kernel_matrix[inds[i], support_vecs])
        self.intercept_ /= len(self.lagrange_mults_)

        # theta parameter is only needed when using a linear kernel (linear SVM)
        if self._kernel == LinearKernel:
            self.theta_ = np.zeros(n_features)

            # linear kernel parametrization is computed using representer's theorem
            for lagrange_mult, sv_label, sv in zip(self.lagrange_mults_,
                                                   self.support_vector_labels_,
                                                   self.support_vectors_):
                # note that here (lagrange_mult * sv_label) is broadcasted across sv
                self.theta_ += lagrange_mult * sv_label * sv

        return self

    def decision_values(self, X: NDArray[np.float64], progress: bool = True) -> NDArray[np.float64]:
        """Compute the binary SVM model decision values for labels given data X."""

        should_show_progress = not progress
        instance_iter = trange(len(X), disable=should_show_progress)

        if self.theta_ is not None:
            return X @ self.theta_ + self.intercept_

        y_pred = np.zeros(len(X))
        for i in instance_iter:
            s = 0

            for a, sv_y, sv in zip(self.lagrange_mults_, self.support_vector_labels_, self.support_vectors_):
                s += a * sv_y * self._kernel(X[i], sv)
            y_pred[i] = s

        return y_pred + self.intercept_

    def predict(self, X: NDArray[np.float64], progress: bool = True) -> NDArray[np.float64]:
        """Predict labels using the binary SVM model given data X."""

        return np.sign(self.decision_values(X, progress=progress))

