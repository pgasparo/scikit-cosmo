import unittest
from skcosmo.pcovr import KPCovR, PCovR
from sklearn.datasets import load_boston
import numpy as np
from sklearn import exceptions
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_X_y
from skcosmo.preprocessing import StandardFlexibleScaler as SFS


class KPCovRBaseTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.error_tol = 1e-6

        self.X, self.Y = load_boston(return_X_y=True)

        # artificial second property
        self.Y = np.array(
            [self.Y, self.X @ np.random.randint(-2, 2, (self.X.shape[-1],))]
        ).T
        self.Y = self.Y.reshape(self.X.shape[0], -1)

        self.X = SFS().fit_transform(self.X)
        self.Y = SFS(column_wise=True).fit_transform(self.Y)

    def setUp(self):
        pass


class KPCovRErrorTest(KPCovRBaseTest):
    def test_lr_with_x_errors(self):
        """
        This test checks that KPCovR returns a non-null property prediction
        and that the prediction error increases with `mixing`
        """
        prev_error = -1.0

        for i, mixing in enumerate(np.linspace(0, 1, 11)):

            pcovr = KPCovR(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            _, error = pcovr.score(self.X, self.Y)

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertGreaterEqual(error, prev_error - self.error_tol)

            prev_error = error

    def test_reconstruction_errors(self):
        """
        This test checks that KPCovR returns a non-null reconstructed X
        and that the reconstruction error decreases with `mixing`
        """

        prev_error = 1.0

        for i, mixing in enumerate(np.linspace(0, 1, 11)):
            pcovr = KPCovR(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            error, _ = pcovr.score(self.X, self.Y)

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertLessEqual(error, prev_error + self.error_tol)

            prev_error = error


class KPCovRInfrastructureTest(KPCovRBaseTest):
    def test_nonfitted_failure(self):
        """
        This test checks that KPCovR will raise a `NonFittedError` if
        `transform` is called before the model is fitted
        """
        model = KPCovR(mixing=0.5, n_components=2, tol=1e-12)
        with self.assertRaises(exceptions.NotFittedError):
            _ = model.transform(self.X)

    def test_no_arg_predict(self):
        """
        This test checks that KPCovR will raise a `ValueError` if
        `predict` is called without arguments
        """
        model = KPCovR(mixing=0.5, n_components=2, tol=1e-12)
        model.fit(self.X, self.Y)
        with self.assertRaises(ValueError):
            _ = model.predict()

    def test_T_shape(self):
        """
        This test checks that KPCovR returns a latent space projection
        consistent with the shape of the input matrix
        """
        n_components = 5
        pcovr = KPCovR(mixing=0.5, n_components=n_components, tol=1e-12)
        pcovr.fit(self.X, self.Y)
        T = pcovr.transform(self.X)
        self.assertTrue(check_X_y(self.X, T, multi_output=True))
        self.assertTrue(T.shape[-1] == n_components)


class KernelTests(KPCovRBaseTest):
    def test_kernel_types(self):
        """
        This test checks that KPCovR can handle all kernels passable to
        sklearn kernel classes, including callable kernels
        """

        def _linear_kernel(X, Y):
            return X @ Y.T

        kernel_params = {
            "poly": {"degree": 2},
            "rbf": {"gamma": 3.0},
            "sigmoid": {"gamma": 3.0, "coef0": 0.5},
        }
        for kernel in ["linear", "poly", "rbf", "sigmoid", "cosine", _linear_kernel]:
            with self.subTest(kernel=kernel):
                model = KPCovR(
                    mixing=0.5,
                    n_components=2,
                    kernel=kernel,
                    **kernel_params.get(kernel, {})
                )
                model.fit(self.X, self.Y)

    def test_linear_matches_pcovr(self):
        """
        This test checks that KPCovR returns the same results as PCovR when
        using a linear kernel
        """

        # making a common Yhat so that the models are working off the same values
        ridge = RidgeCV(fit_intercept=False, alphas=np.logspace(-8, 2))
        Yhat = ridge.fit(self.X, self.Y).predict(self.X)

        # common instantiation parameters for the two models
        hypers = dict(
            mixing=0.5,
            n_components=4,
            regularization=1e-8,
        )

        # computing projection and predicton loss with linear KPCovR
        model = KPCovR(kernel="linear", fit_inverse_transform=True, **hypers)
        model.fit(self.X, self.Y, Yhat=Yhat)
        lk, ly = model.score(self.X, self.Y)

        # computing projection and predicton loss with PCovR
        ref_model = PCovR(**hypers)
        ref_model.fit(self.X, self.Y, Yhat=Yhat)
        t_ref = ref_model.transform(self.X)
        _, ly_ref = ref_model.score(self.X, self.Y)

        # computing the loss in K similar to that in kpcovr
        K = model._get_kernel(self.X, model.X_fit_)
        k_ref = t_ref @ np.linalg.lstsq(t_ref, K, rcond=1e-8)[0]
        lk_ref = np.linalg.norm(K - k_ref) ** 2.0 / np.linalg.norm(K) ** 2.0

        rounding = 3
        self.assertEqual(
            round(ly, rounding),
            round(ly_ref, rounding),
        )

        self.assertEqual(
            round(lk, rounding),
            round(lk_ref, rounding),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
