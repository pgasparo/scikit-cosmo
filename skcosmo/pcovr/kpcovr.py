import numpy as np
from functools import partial
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils import check_array
from sklearn.decomposition._base import _BasePCA
from sklearn.linear_model._base import LinearModel
from sklearn.metrics.pairwise import pairwise_kernels

from skcosmo.utils import eig_solver
from skcosmo.pcovr import pcovr_kernel
from skcosmo.preprocessing import KernelFlexibleCenterer


class KPCovR(_BasePCA, LinearModel):
    """
    Performs Kernel Principal Covariates Regression, as described in
    `[Helfrecht, et al., 2020]
    <https://iopscience.iop.org/article/10.1088/2632-2153/aba9ef>`_.

    Parameters
    ----------
    mixing: float, defaults to 1
        mixing parameter, as described in PCovR as :math:`{\\alpha}`

    n_components: int, float or str, default=None
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

    kernel: "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel. Default="linear".

    gamma: float, default=1/n_features
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels.

    degree: int, default=3
        Degree for poly kernels. Ignored by other kernels.

    coef0: float, default=1
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params: mapping of string to any, default=None
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    regularization: float, default=1E-6
            Regularization parameter to use in all regression operations.

    fit_inverse_transform: bool, default=False
        Learn the inverse transform for non-precomputed kernels.
        (i.e. learn to find the pre-image of a point)

    tol: float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    n_jobs: int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.


    Attributes
    ----------

    mixing_: float, defaults to 1
        mixing parameter, as described in PCovR as :math:`{\\alpha}`

    regularization_: float, default=1E-6
            Regularization parameter to use in all regression operations.

    tol: float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    n_components_: int
        The estimated number of components, which equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.

    pt__: ndarray of size :math:`({n_{components}, n_{components}})`
           pseudo-inverse of the latent-space projection, which
           can be used to contruct projectors from latent-space

    pkt_: ndarray of size :math:`({n_{samples}, n_{components}})`
           the projector, or weights, from the input kernel :math:`\\mathbf{K}`
           to the latent-space projection :math:`\\mathbf{T}`

    pky_: ndarray of size :math:`({n_{samples}, n_{properties}})`
           the projector, or weights, from the input kernel :math:`\\mathbf{K}`
           to the properties :math:`\\mathbf{Y}`

    pty_: ndarray of size :math:`({n_{components}, n_{properties}})`
          the projector, or weights, from the latent-space projection
          :math:`\\mathbf{T}` to the properties :math:`\\mathbf{Y}`

    ptx_: ndarray of size :math:`({n_{components}, n_{features}})`
         the projector, or weights, from the latent-space projection
         :math:`\\mathbf{T}` to the feature matrix :math:`\\mathbf{X}`

    X_fit_: ndarray of shape (n_samples, n_features)
        The data used to fit the model. This attribute is used to build kernels
        from new data.

    References
    ----------
        1.  B. A. Helfrecht, R. K. Cersonsky, G. Fraux, and M. Ceriotti,
            'Structure-property maps with Kernel principal covariates regression',
            Machine Learning: Science and Technology 1(4):045021, 2020
        2.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        3.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015

    Examples
    --------
    >>> import numpy as np
    >>> from skcosmo.pcovr import KPCovR
    >>> from skcosmo.preprocessing import StandardFlexibleScaler as SFS
    >>>
    >>> X = np.array([[-1, 1, -3, 1], [1, -2, 1, 2], [-2, 0, -2, -2], [1, 0, 2, -1]])
    >>> X = SFS().fit_transform(X)
    >>> Y = np.array([[ 0, -5], [-1, 1], [1, -5], [-3, 2]])
    >>> Y = SFS(column_wise=True).fit_transform(Y)
    >>>
    >>> kpcovr = KPCovR(mixing=0.1, n_components=2, kernel='rbf', gamma=2)
    >>> kpcovr.fit(X, Y)
        KPCovR(coef0=1, degree=3, fit_inverse_transform=False, gamma=0.01, kernel='rbf',
           kernel_params=None, mixing=None, n_components=2, n_jobs=None,
           regularization=None, tol=1e-12)
    >>> T = kpcovr.transform(X)
        [[ 1.01199065, -0.35439061],
         [-0.68099591,  0.48912275],
         [ 1.4677616 ,  0.13757037],
         [-1.79874193, -0.27232032]]
    >>> Yp = kpcovr.predict(X)
        [[-0.01044648, -0.84443158],
         [-0.1758848 ,  0.16224503],
         [ 0.1573037 , -0.84211944],
         [-0.51133139,  0.32552881]]
    >>> kpcovr.score(X, Y)
        (0.5312320029915978, 0.06254540655698511)
    """

    def __init__(
        self,
        mixing=0.0,
        n_components=None,
        *,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        regularization=1e-6,
        kernel_params=None,
        fit_inverse_transform=False,
        tol=1e-12,
        n_jobs=None,
    ):

        self.mixing_ = mixing
        self.n_components = n_components
        self.regularization_ = regularization
        self.tol = tol

        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_jobs = n_jobs

        self.fit_inverse_transform = fit_inverse_transform

        self._eig_solver = partial(
            eig_solver, n_components=self.n_components, tol=self.tol, add_null=True
        )

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    def _fit(self, K, Yhat, W):
        """
        Fit the model with the computed kernel and approximated properties.
        """

        K = self._centerer.fit_transform(K)
        K_tilde = pcovr_kernel(mixing=self.mixing_, X=K, Y=Yhat, kernel="precomputed")

        v, U = eig_solver(
            K_tilde, tol=self.tol, n_components=self.n_components
        )

        P = (self.mixing_ * np.eye(K.shape[0])) + (1.0 - self.mixing_) * (W @ Yhat.T)

        v_inv = np.linalg.pinv(np.diagflat(v))

        self.pkt_ = P @ U @ np.sqrt(v_inv)

        T = K @ self.pkt_

        self.pt__ = np.linalg.lstsq(T, np.eye(T.shape[0]), rcond=self.regularization_)[0]

    def fit(self, X, Y, Yhat=None, W=None):
        """

        Fit the model with X and Y.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

            It is suggested that :math:`\\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be scaled
            to have unit variance, otherwise :math:`\\mathbf{X}` should be
            scaled so that each feature has a variance of 1 / n_features.

        Y: array-like, shape (n_samples, n_properties)
            Training data, where n_samples is the number of samples and
            n_properties is the number of properties

            It is suggested that :math:`\\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be scaled
            to have unit variance, otherwise :math:`\\mathbf{Y}` should be
            scaled so that each feature has a variance of 1 / n_features.

        Yhat: array-like, shape (n_samples, n_properties), optional
            Regressed training data, where n_samples is the number of samples and
            n_properties is the number of properties. If not supplied, computed
            by ridge regression.

        Returns
        -------
        self: object
            Returns the instance itself.

        """

        X, Y = check_X_y(X, Y, y_numeric=True, multi_output=True)
        self.X_fit_ = X.copy()

        if self.n_components is None:
            self.n_components = min(X.shape)

        self._centerer = KernelFlexibleCenterer()
        K = self._get_kernel(X)

        if W is None:
            if Yhat is None:
                W = np.linalg.lstsq(K, Y, rcond=self.regularization_)[0]
            else:
                W = np.linalg.lstsq(K, Yhat, rcond=self.regularization_)[0]

        if Yhat is None:
            Yhat = K @ W

        self._fit(K, Yhat, W)

        self.ptk_ = self.pt__ @ K
        self.pty_ = self.pt__ @ Y

        if self.fit_inverse_transform:
            self.ptx_ = self.pt__ @ X

        self.pky_ = self.pkt_ @ self.pty_

        self.components_ = self.pkt_.T  # for sklearn compatibility
        return self

    def predict(self, X=None, T=None):
        """Predicts the property values"""

        check_is_fitted(self, ["pky_", "pty_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = check_array(X)
            K = self._get_kernel(X)
            return K @ self.pky_
        else:
            T = check_array(T)
            return T @ self.pty_

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components as determined by the
        modified Kernel PCovR distances.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        """

        check_is_fitted(self, ["pkt_", "X_fit_"])

        X = check_array(X)
        K = self._get_kernel(X, self.X_fit_)
        return K @ self.pkt_

    def inverse_transform(self, T):
        """Transform input data back to its original space.

        .. math::

            \\mathbf{\\hat{X}} = \\mathbf{T} \\mathbf{P}_{TX}
                              = \\mathbf{K} \\mathbf{P}_{KT} \\mathbf{P}_{TX}


        Similar to KPCA, the original features are not always recoverable,
        as the projection is computed from the kernel features, not the original
        features, and the mapping between the original and kernel features
        is not one-to-one.

        Parameters
        ----------
        T: array-like, shape (n_samples, n_components)
            Projected data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)
        """

        return T @ self.ptx_

    def score(self, X, Y):
        """
        Computes the loss values for KPCovR on the given predictor and
        response variables. The loss in :math:`\\mathbf{K}` does not directly
        correspond to the loss minimized in KPCovR, as explained in `[Helfrecht, et al., 2020]
        <https://iopscience.iop.org/article/10.1088/2632-2153/aba9ef>`_.

        Arguments
        ---------
        X:              independent (predictor) variable
        Y:              dependent (response) variable

        Returns
        -------
        Lk:             KPCA loss, determined by the reconstruction of the kernel
        Ly:             KR loss

        """

        check_is_fitted(self, ["pkt_", "X_fit_"])

        X = check_array(X)
        K = self._get_kernel(X, self.X_fit_)

        k = K @ self.pkt_ @ self.ptk_
        y = K @ self.pky_

        Lkpca = np.linalg.norm(K - k) ** 2 / np.linalg.norm(K) ** 2
        Lkrr = np.linalg.norm(Y - y) ** 2 / np.linalg.norm(Y) ** 2

        return Lkpca, Lkrr
