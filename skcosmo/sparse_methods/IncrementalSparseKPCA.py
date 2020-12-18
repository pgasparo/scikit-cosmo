#!/usr/bin/env python

import numpy as np

# from regression import IncrementalSparseKRR

from sklearn.utils import gen_batches
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels

from ..utils import eig_solver
from ..selection.FPS import SampleFPS
from ..preprocessing.flexible_scaler import SparseKernelCenterer


class IncrementalSparseKPCA(TransformerMixin, BaseEstimator):
    """
    Performs sparsified principal component analysis
    using batches.

    Parameters
    ----------

    Parameters
    ----------
    n_components : int, default=n_active
        Number of components.
    n_active : int
        Number of active samples to use within the sparse kernel.
    batch_size : int, default=None
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.
    kernel : {'linear', 'poly', \
            'rbf', 'sigmoid', 'cosine', 'precomputed'}, default='linear'
        Kernel used for PCA.
    gamma : float, default=None
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels. If ``gamma`` is ``None``, then it is set to ``1/n_features``.
    degree : int, default=3
        Degree for poly kernels. Ignored by other kernels.
    coef0 : float, default=1
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : dict, default=None
        Parameters (keyword arguments) and
        values for kernel passed as callable object.
        Ignored by other kernels.
    alpha : float, default=1.0
        Hyperparameter of the ridge regression that learns the
        inverse transform (when fit_inverse_transform=True).
    fit_inverse_transform : bool, default=False
        Learn the inverse transform for non-precomputed kernels.
        (i.e. learn to find the pre-image of a point)
    tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.
    copy_X : bool, default=True
        If True, input X is copied and stored by the model in the `X_fit_`
        attribute. If no further changes will be done to X, setting
        `copy_X=False` saves memory by storing a reference.
    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.


    ---References---
    1.  https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
    2.  M. E. Tipping 'Sparse Kernel Principal Component Analysis',
        Advances in Neural Information Processing Systems 13, 633-639, 2001
    3.  C. Williams, M. Seeger, 'Using the Nystrom Method to Speed Up Kernel Machines',
        Avnaces in Neural Information Processing Systems 13, 682-688, 2001
    4.  K. Zhang, I. W. Tsang, J. T. Kwok, 'Improved Nystrom Low-Rank Approximation
        and Error Analysis', Proceedings of the 25th International Conference
        on Machine Learning, 1232-1239, 2008
    5.  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(
        self,
        n_components=None,
        n_active=None,
        batch_size=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        selector=SampleFPS,
        center=True,
        alpha=1.0,
        fit_inverse_transform=False,
        tol=1e-12,
        copy_X=True,
        n_jobs=None,
    ):
        if fit_inverse_transform and kernel == "precomputed":
            raise ValueError("Cannot fit_inverse_transform with a precomputed kernel.")

        self.n_active = n_active

        if n_components is not None:
            self.n_components = n_components
        else:
            self.n_components = self.n_active

        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.copy_X = copy_X

        self.center = center
        self.batch_size = batch_size
        self.tol = tol

        self.pkt_ = None
        self.X_sparse_ = None
        self.K_sparse_ = None

        self._selector = selector
        self._centerer = SparseKernelCenterer(rcond=self.tol)

    def _get_kernel(self, X, Y=None):

        if self.kernel == "precomputed":
            if X.shape[-1] != self.n_active:
                raise ValueError("The supplied kernel does not match n_active.")
            return X

        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    def fit(self, X, X_sparse=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features. This may also be the
            precomputed kernel of shape (n_samples, n_samples)
            in the case that self.kernel == 'precomputed'
        X_sparse : {array-like} of shape (n_active, n_features)
            Active set of samples, where n_features is the number of features.
            This may also be the precomputed active kernel of shape
            (n_active, n_active) in the case that self.kernel == 'precomputed'

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        # check the dimensions and suitability of X
        X = check_array(X, copy=self.copy_X)
        if X_sparse is None:

            i_active = self._selector(X).select(self.n_active)

            X_sparse = X[i_active]

            # if X_sparse is pulling samples from the full kernel X, then
            # we want to select both the rows _and_ columns
            if self.kernel == "precomputed":
                X_sparse = X_sparse[:, i_active]

        self.X_sparse_ = X_sparse
        self.K_sparse_ = self._get_kernel(X_sparse)
        self.n_active_ = X_sparse.shape[0]

        if self.batch_size is None:
            self.batch_size_ = 5 * self.n_active_
        else:
            self.batch_size_ = self.batch_size

        self.C = np.zeros((self.n_active, self.n_active))
        self.T_mean = np.zeros(self.n_active)

        self.K_fit_mean_ = np.zeros(self.n_active)
        self.K_fit_trace_ = 0

        self.n_samples = 0

        self.Vm, self.Um = np.linalg.eig(self.K_sparse_)
        self.Vm_isqrt = np.linalg.pinv(np.diagflat(np.sqrt(self.Vm)))

        for batch in gen_batches(
            X.shape[0], self.batch_size_, min_batch_size=self.n_components or 0
        ):
            X_batch = X[batch]
            Knmi = self._get_kernel(X_batch, X_sparse)
            self._partial_fit(Knmi, check_input=False)

        if self.n_samples < 1:
            raise NotFittedError(
                "Error: must fit at least one batch" "before finalizing the fit"
            )
            return

        self.K_fit_mean_ /= self.n_samples

        if self.center:
            self._centerer.scale_ = np.sqrt(self.K_fit_trace_ / self.n_samples)
            self._centerer.n_active_ = self.n_active
            self._centerer.K_fit_rows_ = self.K_fit_mean_
            self.Vm_isqrt *= np.sqrt(np.sqrt(self.K_fit_trace_ / self.n_samples))
            self.Vm /= np.sqrt(self.K_fit_trace_ / self.n_samples)
            self.C /= np.sqrt(self.K_fit_trace_ / self.n_samples)
        else:
            self._centerer.scale_ = 1
            self._centerer.n_active_ = self.n_active
            self._centerer.K_fit_rows_ = np.zeros(self.n_active)

        self.Vc, self.Uc = eig_solver(
            self.C, n_components=self.n_components, tol=self.tol, add_null=True
        )
        self.T_mean = self.T_mean @ self.Uc

        self.pkt_ = self.Um @ self.Vm_isqrt @ self.Uc

        self.pkt_ = self.pkt_[:, 0 : self.n_components]
        self.T_mean = self.T_mean[0 : self.n_components]

    def _partial_fit(self, Knm, y=None, check_input=True):
        """
        Fits a batch for the sparse KPCA

        ---Arguments---
        Knm: kernel between all training points and the active set
        """

        check_is_fitted(self, ["Um", "Vm"])

        # Reshape 1D arrays
        if Knm.ndim < 2:
            Knm = np.reshape(Knm, (1, -1))

        # Don't need to do auxiliary centering of T or KNM
        # since the covariance matrix will be centered once
        # we are finished building it

        # TODO: also scale T?
        T = Knm @ self.Um @ self.Vm_isqrt

        # Single iteration of one-pass covariance
        old_mean = self.T_mean.copy()

        self.n_samples += Knm.shape[0]
        self.K_fit_mean_ += Knm.sum(axis=0)
        self.K_fit_trace_ += np.trace(
            Knm @ np.linalg.pinv(self.K_sparse_, rcond=self.tol) @ Knm.T
        )

        self.T_mean += np.sum(T - old_mean, axis=0) / self.n_samples

        self.C += np.matmul((T - self.T_mean).T, T - old_mean)

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features). This may also be the
            precomputed kernel of shape (n_samples, n_samples)
            in the case that self.kernel == 'precomputed'

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """

        check_is_fitted(self, ["pkt_", "T_mean"])

        transformed = np.zeros((X.shape[0], self.n_components))

        for batch in gen_batches(
            X.shape[0], self.batch_size_, min_batch_size=self.n_components or 0
        ):
            X_batch = X[batch]
            Knmi = self._get_kernel(X_batch, self.X_sparse_)
            transformed[batch] = self._partial_transform(Knmi)

        return transformed

    def _partial_transform(self, Knm):
        """Transform a block of the training kernel.

        Parameters
        ----------
        Knm : kernel between all training points and the active set

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """
        return (self._centerer.transform(Knm) @ self.pkt_) - self.T_mean

    def fit_transform(self, X, X_sparse=None):
        self.fit(X, X_sparse)
        return self.transform(X)

    # def initialize_inverse_transform(self, KMM, x_dim=1, sigma=1.0,
    #         regularization=1.0E-12, regularization_type='scalar', rcond=None):
    #     """
    #         Initialize the sparse KPCA inverse transform
    #
    #         ---Arguments---
    #         KMM: centered kernel between the transformed representative points
    #         x_dim: dimension of X data
    #         sigma: regulariztion parameter
    #         regularization: additional regularization for the Sparse KRR solution
    #             for the inverse transform
    #         rcond: cutoff ratio for small singular values in the least squares
    #             solution to determine the inverse transform
    #     """
    #
    #     # (can also do LR here)
    #     self.iskrr = IncrementalSparseKRR(sigma=sigma, regularization=regularization,
    #             regularization_type=regularization_type, rcond=rcond)
    #     self.iskrr.initialize_fit(KMM, y_dim=x_dim)
    #
    # def fit_inverse_transform_batch(self, KTM, X):
    #     """
    #         Fit a batch for the inverse KPCA transform
    #
    #         ---Arguments---
    #         KTM: centered kernel between the KPCA transformed training data
    #             and the transformed representative points
    #         X: the centered original input data
    #     """
    #
    #     self.iskrr.fit_batch(KTM, X)
    #
    # def finalize_inverse_transform(self):
    #     """
    #         Finalize the fitting of the inverse KPCA transform
    #     """
    #
    #     self.iskrr.finalize_fit()
    #
    # def inverse_transform(self, KXM):
    #     """
    #         Computes the reconstruction of X
    #
    #         ---Arguments---
    #         KXM: centered kernel between the transformed data and the
    #             representative transformed data
    #
    #         ---Returns---
    #         Xr: reconstructed centered input data
    #     """
    #
    #     # Compute the reconstruction
    #     W = self.iskrr.W
    #     Xr = np.matmul(KXM, W)
    #
    #     return Xr
