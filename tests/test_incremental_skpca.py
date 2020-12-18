import unittest
from skcosmo.sparse_methods import IncrementalSparseKPCA as iSKPCA
from skcosmo.sparse_methods import SparseKPCA as SKPCA
import sklearn
import numpy as np


class IncrementalSKPCATests(unittest.TestCase):
    def test_check_equivalent_with_x(self):
        # tests that incremental sparse KPCA outputs the same result as sparse KPCA
        X = np.random.uniform(-3, 3, size=(1000, 100))
        X -= np.mean(X, axis=0)
        X /= np.linalg.norm(X) / np.sqrt(X.shape[0])

        i_sparse = np.random.choice(X.shape[0], 3)
        X_sparse = X.copy()[i_sparse]

        iskpca = iSKPCA(
            n_active=len(i_sparse),
            n_components=2,
            batch_size=len(i_sparse),
            kernel="linear",
            tol=1e-12,
            center=True,
        )

        skpca = SKPCA(
            n_active=len(i_sparse),
            n_components=2,
            kernel="linear",
            tol=1e-12,
            center=True,
        )

        T_iskpca = iskpca.fit_transform(X.copy(), X_sparse=X_sparse.copy())
        T_skpca = skpca.fit_transform(X.copy(), X_sparse=X_sparse.copy())

        self.assertLessEqual(
            np.max(
                np.abs(
                    T_iskpca - T_skpca,
                )
            ),
            1e-8,
        )

    def test_check_equivalent_with_K(self):
        # tests that incremental sparse KPCA outputs the same result as sparse KPCA
        X = np.random.uniform(-3, 3, size=(8, 100))
        X -= np.mean(X, axis=0)
        X /= np.linalg.norm(X) / np.sqrt(X.shape[0])

        i_sparse = np.random.choice(X.shape[0], 2)
        X_sparse = X.copy()[i_sparse]

        Knm = X @ X_sparse.T
        Kmm = X_sparse @ X_sparse.T

        iskpca = iSKPCA(
            n_active=len(i_sparse),
            n_components=2,
            batch_size=len(i_sparse),
            kernel="precomputed",
            tol=1e-12,
            center=False,
        )

        skpca = SKPCA(
            n_active=len(i_sparse),
            n_components=2,
            kernel="precomputed",
            tol=1e-12,
            center=False,
        )

        T_iskpca = iskpca.fit_transform(Knm, X_sparse=Kmm)
        T_skpca = skpca.fit_transform(Knm, X_sparse=Kmm)

        self.assertLessEqual(
            np.max(
                np.abs(
                    T_iskpca - T_skpca,
                )
            ),
            1e-8,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
