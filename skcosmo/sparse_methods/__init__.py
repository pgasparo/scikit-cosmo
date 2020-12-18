"""
The :mod:`skcosmo.sparse_methods` module includes SparseKPCA, SparseKRR
and SparseKPCovR methods
"""

from .SparseKPCA import SparseKPCA
from .IncrementalSparseKPCA import IncrementalSparseKPCA

# from .SparseKPCovR import SparseKPCovR
# from .SparseKRR import SparseKRR

__all__ = ["SparseKPCA", "IncrementalSparseKPCA"]
