# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

if TYPE_CHECKING:
    import cupyx.scipy.sparse
    import scipy.sparse

Array = Union[np.ndarray, "cp.ndarray"]
SparseCSR = Union["scipy.sparse.csr_array", "scipy.sparse.csr_matrix",
                   "cupyx.scipy.sparse.csr_matrix"]


class BitMatrixCSR:
    """CSR representation of a binary (GF(2)) matrix.

    All nonzero entries are implicitly 1.  Storage-agnostic: holds
    numpy or cupy arrays depending on how it was constructed.
    """

    __slots__ = ("row_offsets", "col_indices", "nnz", "shape")

    row_offsets: Array
    """Row pointer array, shape ``(num_rows + 1,)``, dtype int64."""

    col_indices: Array
    """Column index array, shape ``(nnz,)``, dtype int64."""

    nnz: int
    """Number of nonzero entries."""

    shape: Tuple[int, int]
    """``(num_rows, num_cols)`` of the logical matrix."""

    def __init__(
        self,
        row_offsets: Array,
        col_indices: Array,
        nnz: int,
        shape: Tuple[int, int],
    ):
        self.row_offsets = row_offsets
        self.col_indices = col_indices
        self.nnz = int(nnz)
        self.shape = (int(shape[0]), int(shape[1]))

    @classmethod
    def from_sparse(cls, mat: SparseCSR) -> BitMatrixCSR:
        """Create from a ``scipy.sparse`` or ``cupyx.scipy.sparse`` CSR matrix.

        Accepts ``csr_array``, ``csr_matrix`` (scipy), or ``csr_matrix`` (cupyx).
        The ``data`` array is ignored -- all stored entries are treated as 1.
        """
        if not hasattr(mat, "indptr") or not hasattr(mat, "indices"):
            raise TypeError("Expected a scipy or cupyx CSR matrix with .indptr and .indices")
        ro = mat.indptr
        ci = mat.indices
        if isinstance(ro, np.ndarray):
            ro = ro.astype(np.int64, copy=False)
            ci = ci.astype(np.int64, copy=False)
        elif cp is not None and isinstance(ro, cp.ndarray):
            ro = ro.astype(cp.int64, copy=False)
            ci = ci.astype(cp.int64, copy=False)
        return cls(row_offsets=ro, col_indices=ci, nnz=int(mat.nnz), shape=mat.shape)

    def to_scipy_sparse(self) -> "scipy.sparse.csr_array":  # noqa: F821
        """Convert to ``scipy.sparse.csr_array``.  Transfers to CPU if needed."""
        import scipy.sparse

        ro = self.row_offsets
        ci = self.col_indices[:self.nnz]
        if hasattr(ro, "get"):
            ro = ro.get()
            ci = ci.get()
        data = np.ones(self.nnz, dtype=np.uint8)
        return scipy.sparse.csr_array(
            (data, ci.astype(np.int64, copy=False), ro.astype(np.int64, copy=False)),
            shape=self.shape,
        )

    def to_cupyx_sparse(self) -> "cupyx.scipy.sparse.csr_matrix":  # noqa: F821
        """Convert to ``cupyx.scipy.sparse.csr_matrix``.  Transfers to GPU if needed."""
        if cp is None:
            raise ImportError("CuPy is required for to_cupyx_sparse()")
        import cupyx.scipy.sparse

        ro = self.row_offsets
        ci = self.col_indices[:self.nnz]
        if isinstance(ro, np.ndarray):
            ro = cp.asarray(ro)
            ci = cp.asarray(ci)
        data = cp.ones(self.nnz, dtype=cp.uint8)
        return cupyx.scipy.sparse.csr_matrix(
            (data, ci.astype(cp.int64, copy=False), ro.astype(cp.int64, copy=False)),
            shape=self.shape,
        )

    def __repr__(self) -> str:
        return f"BitMatrixCSR(shape={self.shape}, nnz={self.nnz})"
