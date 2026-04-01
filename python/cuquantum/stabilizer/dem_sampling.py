# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
except Exception as e:  # pragma: no cover
    raise RuntimeError("cupy is required for cuStabilizer dem sampling") from e

import cuda.bindings.runtime as cudart
from cuquantum.bindings import custabilizer as custab
from nvmath import memory
from nvmath.internal import utils as nvmath_utils

from ._options import Options, _ManagedOptions
from .bit_matrix import BitMatrixCSR, SparseCSR
from .utils import _pack_arrays, _unpack_arrays, _ptr_as_cupy, _get_memptr

Array = Union[np.ndarray, "cp.ndarray"]  # noqa: F821
Stream = Union[int, "cp.cuda.Stream", None]  # noqa: F821


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _as_cupy(x: Array, *, dtype=None) -> "cp.ndarray":  # noqa: F821
    if isinstance(x, cp.ndarray):
        if dtype is None:
            return x
        return x.astype(dtype, copy=False)
    return cp.asarray(x, dtype=dtype)


@dataclass(frozen=True)
class _MatrixSpec:
    n_errors: int
    n_outcomes: int
    n_outcomes_padded: int
    outcome_words: int


def _reinterpret_packed(b: "cp.ndarray", words: int) -> "cp.ndarray":  # noqa: F821
    """Reinterpret an already bit-packed matrix as uint32 words."""
    if b.dtype == cp.uint32:
        if int(b.shape[1]) != words:
            raise ValueError(
                f"packed uint32 matrix has shape[1]={b.shape[1]}, expected {words}")
        return b
    b_u8 = b.astype(cp.uint8, copy=False)
    if int(b_u8.shape[1]) != words * 4:
        raise ValueError(
            f"packed uint8 matrix has shape[1]={b_u8.shape[1]}, expected {words * 4}")
    if (b_u8.nbytes & 3) != 0:
        raise ValueError("packed matrix must have nbytes multiple of 4")
    return b_u8.view(cp.uint32).reshape(b.shape[0], words)


def _prepare_bit_matrix(
    matrix: Array,
    probs: Array,
    num_outcomes: Optional[int],
    bit_packed: bool,
) -> Tuple[_MatrixSpec, "cp.ndarray", "cp.ndarray"]:  # noqa: F821
    """Validate inputs and convert to GPU-packed representation.

    Returns ``(_MatrixSpec, probs_d, b_u32)`` ready for the C API.
    """
    probs_d = _as_cupy(probs, dtype=cp.float64)
    if probs_d.ndim != 1:
        raise ValueError("probs must be a 1D array")
    n_errors = int(probs_d.shape[0])

    b = _as_cupy(matrix)
    if b.ndim != 2:
        raise ValueError("matrix must be a 2D array")
    if b.shape[0] != n_errors:
        raise ValueError(f"matrix.shape[0]={b.shape[0]} must match len(probs)={n_errors}")

    if bit_packed:
        if num_outcomes is None:
            raise ValueError("num_outcomes must be provided when bit_packed=True")
        n_outcomes = int(num_outcomes)
    else:
        n_outcomes = int(b.shape[1]) if num_outcomes is None else int(num_outcomes)
        if n_outcomes != int(b.shape[1]):
            raise ValueError("num_outcomes must match matrix.shape[1] when dense matrix is provided")

    n_outcomes_padded = ((n_outcomes + 31) // 32) * 32
    words = n_outcomes_padded // 32
    spec = _MatrixSpec(n_errors=n_errors, n_outcomes=n_outcomes,
                       n_outcomes_padded=n_outcomes_padded, outcome_words=words)

    if bit_packed:
        b_u32 = _reinterpret_packed(b, words)
    else:
        (packed_u8,) = _pack_arrays(n_outcomes, b.astype(cp.uint8, copy=False))
        b_u32 = packed_u8.view(cp.uint32).reshape(n_errors, words)

    return spec, probs_d, b_u32


def _unpack_outcome(
    c_u32: "cp.ndarray",  # noqa: F821
    num_shots: int,
    spec: _MatrixSpec,
    package: str,
    bit_packed: bool,
) -> Array:
    """Convert packed ``c_u32`` outcomes to the requested output format."""
    c_u8 = c_u32[:num_shots].view(cp.uint8).reshape(num_shots, spec.n_outcomes_padded // 8)
    if bit_packed:
        return cp.asnumpy(c_u8) if package == "numpy" else c_u8
    (unpacked,) = _unpack_arrays(spec.n_outcomes_padded, c_u8)
    dense = unpacked[:, :spec.n_outcomes].astype(cp.uint8, copy=False)
    return cp.asnumpy(dense) if package == "numpy" else dense


def _extract_from_dem(dem: "stim.DetectorErrorModel") -> Tuple[int, np.ndarray, "BitMatrixCSR"]:
    """Extract error-to-detector CSR matrix and probabilities from a DEM.

    Returns ``(n_detectors, probs, BitMatrixCSR)`` where the CSR matrix has
    shape ``(n_errors, n_detectors)`` with sorted column indices per row.
    """
    dem = dem.flattened()
    n_det = int(dem.num_detectors)
    n_err = int(dem.num_errors)

    probs_h = np.empty((n_err,), dtype=np.float64)
    row_offsets = np.empty((n_err + 1,), dtype=np.int64)
    col_indices_list: list = []

    col = 0
    for inst in dem:
        if inst.type != "error":
            continue
        args = inst.args_copy()
        probs_h[col] = float(args[0]) if args else 0.0
        row_offsets[col] = len(col_indices_list)
        row_dets: list = []
        for t in inst.targets_copy():
            is_sep = t.is_separator() if callable(getattr(t, "is_separator", None)) else bool(t.is_separator)
            if is_sep:
                continue
            is_det = (
                t.is_relative_detector_id()
                if callable(getattr(t, "is_relative_detector_id", None))
                else bool(t.is_relative_detector_id)
            )
            if not is_det:
                continue
            det = int(t.val)
            if det < 0 or det >= n_det:
                import warnings
                warnings.warn(f"DEM references detector {det} but num_detectors={n_det}; skipping", stacklevel=2)
                continue
            row_dets.append(det)
        row_dets.sort()
        col_indices_list.extend(row_dets)
        col += 1

    if col != n_err:
        raise RuntimeError("DEM error count mismatch")

    row_offsets[n_err] = len(col_indices_list)
    col_indices = np.array(col_indices_list, dtype=np.int64)
    b_csr = BitMatrixCSR(row_offsets=row_offsets, col_indices=col_indices,
                         nnz=len(col_indices_list), shape=(n_err, n_det))
    return n_det, probs_h, b_csr


# ---------------------------------------------------------------------------
# Internal result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SparseSampleResult:
    num_shots: int
    c_ptr: Any
    c_nbytes: int
    row_offsets_ptr: Any
    row_offsets_nbytes: int
    col_indices_ptr: Any
    col_indices_nbytes: int
    nnz: int


# ---------------------------------------------------------------------------
# BitMatrixSparseSampler
# ---------------------------------------------------------------------------

class BitMatrixSparseSampler:
    """Sparse Bernoulli sampler with GF(2) matrix multiply.

    Samples independent Bernoulli errors into a sparse intermediate
    representation, then multiplies by a binary outcome matrix to
    produce detector outcomes.

    Best when error probabilities are low and ``n_errors`` is large.
    """

    _opts: _ManagedOptions
    _rng: np.random.Generator
    _spec: _MatrixSpec
    _max_shots: int

    _probs_d: "cp.ndarray"                            # (n_errors,) float64, device
    _b_mode: Literal["sparse", "dense"]
    _b_row_offsets_d: "cp.ndarray"                     # sparse B: CSR row offsets, device
    _b_col_indices_d: "cp.ndarray"                     # sparse B: CSR col indices, device
    _b_nnz: int                                        # sparse B: number of non-zeros
    _b_u32: "cp.ndarray"                               # dense B: bit-packed uint32, device

    _ws_size: int
    _ws_ptr: memory.MemoryPointer
    _capacity: int                                     # current columnIndices capacity (may grow on retry)
    _row_offsets_ptr: memory.MemoryPointer
    _row_offsets_nbytes: int
    _col_indices_ptr: memory.MemoryPointer             # may be reallocated on retry
    _col_indices_nbytes: int
    _c_ptr: memory.MemoryPointer
    _c_nbytes: int

    _last: Optional[_SparseSampleResult]               # result of last sample() call

    def __init__(
        self,
        matrix: Union[Array, BitMatrixCSR, SparseCSR],
        probs: Array,
        max_shots: int,
        *,
        num_outcomes: Optional[int] = None,
        bit_packed: bool = False,
        package: Literal["numpy", "cupy"] = "numpy",
        seed: Optional[int] = None,
        options: Optional[Options] = None,
        stream: Stream = None,
    ):
        if hasattr(matrix, "indptr") and not isinstance(matrix, BitMatrixCSR):
            matrix = BitMatrixCSR.from_sparse(matrix)

        if options is None:
            options = Options()
        self._opts = _ManagedOptions(options, package)
        self._rng = np.random.default_rng(seed)
        stream_holder = self._opts.get_or_create_stream(stream)

        with nvmath_utils.device_ctx(self._opts.device_id), cp.cuda.ExternalStream(stream_holder.ptr):
            if isinstance(matrix, BitMatrixCSR):
                n_errors = int(matrix.shape[0])
                n_outcomes = int(matrix.shape[1])
                if num_outcomes is not None and int(num_outcomes) != n_outcomes:
                    raise ValueError(
                        f"num_outcomes={num_outcomes} conflicts with matrix.shape[1]={n_outcomes}")
                n_outcomes_padded = ((n_outcomes + 31) // 32) * 32
                outcome_words = n_outcomes_padded // 32
                self._spec = _MatrixSpec(
                    n_errors=n_errors, n_outcomes=n_outcomes,
                    n_outcomes_padded=n_outcomes_padded, outcome_words=outcome_words)
                self._probs_d = _as_cupy(probs, dtype=cp.float64)
                if self._probs_d.ndim != 1 or int(self._probs_d.shape[0]) != n_errors:
                    raise ValueError("len(probs) must match matrix.shape[0]")
                ro = matrix.row_offsets
                ci = matrix.col_indices[:matrix.nnz]
                self._b_row_offsets_d = _as_cupy(ro)
                self._b_col_indices_d = _as_cupy(ci)
                self._b_nnz = int(matrix.nnz)
                self._b_mode = "sparse"
            else:
                self._spec, self._probs_d, self._b_u32 = _prepare_bit_matrix(
                    matrix, probs, num_outcomes, bit_packed,
                )
                self._b_mode = "dense"

            self._max_shots = int(max_shots)
            self._ws_size = custab.sample_prob_array_sparse_prepare(
                self._opts.handle, self._max_shots, self._spec.n_errors)
            self._ws_ptr = self._opts.allocate_memory(self._ws_size, stream)
            self._pre_allocate(stream)

        self._last: Optional[_SparseSampleResult] = None

    def _pre_allocate(self, stream: Stream = None):
        ms = self._max_shots
        spec = self._spec
        probs_sum = float(cp.asnumpy(cp.sum(self._probs_d)))
        expected = float(ms) * probs_sum
        capacity = int(expected * 1.25 + 1024.0)
        if capacity < ms:
            capacity = ms
        max_cap = ms * spec.n_errors
        if capacity > max_cap:
            capacity = max_cap
        self._capacity = capacity
        self._row_offsets_nbytes = (ms + 1) * 8
        self._row_offsets_ptr = self._opts.allocate_memory(self._row_offsets_nbytes, stream)
        self._col_indices_nbytes = capacity * 8
        self._col_indices_ptr = self._opts.allocate_memory(self._col_indices_nbytes, stream)
        self._c_nbytes = ms * spec.outcome_words * 4
        self._c_ptr = self._opts.allocate_memory(self._c_nbytes, stream)

    @property
    def n_errors(self) -> int:
        return self._spec.n_errors

    @property
    def n_outcomes(self) -> int:
        return self._spec.n_outcomes

    @property
    def operands_package(self) -> str:
        return self._opts.package if self._opts.package != "cuda" else "numpy"

    def sample(self, num_shots: int, *, seed: Optional[int] = None, stream: Stream = None) -> None:
        """Sample errors and outcomes.

        Results are stored internally; retrieve via :meth:`get_outcomes`
        and :meth:`get_errors`.
        """
        num_shots = int(num_shots)
        if num_shots <= 0:
            raise ValueError("num_shots must be > 0")
        if num_shots > self._max_shots:
            raise ValueError(f"num_shots ({num_shots}) exceeds max_shots ({self._max_shots})")

        seed_ = int(seed if seed is not None else self._rng.integers(0, 2**31))
        stream_holder = self._opts.get_or_create_stream(stream)

        with nvmath_utils.device_ctx(self._opts.device_id):
            ro_ptr = self._row_offsets_ptr
            ci_ptr = self._col_indices_ptr
            capacity = self._capacity
            ci_nbytes = self._col_indices_nbytes

            nnz_h = np.asarray([np.uint64(capacity)], dtype=np.uint64)

            try:
                custab.sample_prob_array_sparse_compute(
                    self._opts.handle,
                    num_shots,
                    self._spec.n_errors,
                    int(self._probs_d.data.ptr),
                    seed_,
                    int(nnz_h.ctypes.data),
                    _get_memptr(ci_ptr),
                    _get_memptr(ro_ptr),
                    _get_memptr(self._ws_ptr),
                    self._ws_size,
                    stream_holder.ptr,
                )
            except custab.cuStabilizerError as e:
                if custab.Status(e.status) != custab.Status.INSUFFICIENT_SPARSE_STORAGE:
                    raise
                required = int(nnz_h[0])
                ci_nbytes = required * 8
                ci_ptr = self._opts.allocate_memory(ci_nbytes)
                self._col_indices_ptr = ci_ptr
                self._col_indices_nbytes = ci_nbytes
                self._capacity = required
                nnz_h[0] = np.uint64(required)
                custab.sample_prob_array_sparse_compute(
                    self._opts.handle,
                    num_shots,
                    self._spec.n_errors,
                    int(self._probs_d.data.ptr),
                    seed_,
                    int(nnz_h.ctypes.data),
                    _get_memptr(ci_ptr),
                    _get_memptr(ro_ptr),
                    _get_memptr(self._ws_ptr),
                    self._ws_size,
                    stream_holder.ptr,
                )

            nnz_used = int(nnz_h[0])

            if self._b_mode == "sparse":
                custab.gf2_sparse_sparse_matrix_multiply(
                    self._opts.handle,
                    num_shots,
                    self._spec.n_outcomes_padded,
                    self._spec.n_errors,
                    _get_memptr(ci_ptr),
                    _get_memptr(ro_ptr),
                    self._b_nnz,
                    int(self._b_col_indices_d.data.ptr),
                    int(self._b_row_offsets_d.data.ptr),
                    0,
                    _get_memptr(self._c_ptr),
                    stream_holder.ptr,
                )
            else:
                custab.gf2_sparse_dense_matrix_multiply(
                    self._opts.handle,
                    num_shots,
                    self._spec.n_outcomes_padded,
                    self._spec.n_errors,
                    nnz_used,
                    _get_memptr(ci_ptr),
                    _get_memptr(ro_ptr),
                    int(self._b_u32.data.ptr),
                    0,
                    _get_memptr(self._c_ptr),
                    stream_holder.ptr,
                )

            self._last = _SparseSampleResult(
                num_shots=num_shots,
                c_ptr=self._c_ptr,
                c_nbytes=self._c_nbytes,
                row_offsets_ptr=ro_ptr,
                row_offsets_nbytes=self._row_offsets_nbytes,
                col_indices_ptr=ci_ptr,
                col_indices_nbytes=ci_nbytes,
                nnz=nnz_used,
            )

    def _require_last(self) -> _SparseSampleResult:
        if self._last is None:
            raise RuntimeError("sample() has not been called")
        return self._last

    def get_outcomes(self, bit_packed: bool = True) -> Array:
        """Retrieve outcomes from the last :meth:`sample` call."""
        last = self._require_last()
        spec = self._spec
        c_u32 = _ptr_as_cupy(
            last.c_ptr, last.c_nbytes,
            shape=(self._max_shots, spec.outcome_words), dtype=cp.uint32,
        )
        pkg = "numpy" if self.operands_package == "numpy" else "cupy"
        return _unpack_outcome(c_u32, last.num_shots, spec, pkg, bit_packed)

    def get_errors(self) -> BitMatrixCSR:
        """Retrieve errors as :class:`BitMatrixCSR` from the last :meth:`sample` call."""
        last = self._require_last()
        ro_all = _ptr_as_cupy(
            last.row_offsets_ptr, last.row_offsets_nbytes,
            shape=(last.row_offsets_nbytes // 8,), dtype=cp.uint64,
        )
        ci_all = _ptr_as_cupy(
            last.col_indices_ptr, last.col_indices_nbytes,
            shape=(last.col_indices_nbytes // 8,), dtype=cp.uint64,
        )
        ro = ro_all[:last.num_shots + 1].view(cp.int64)
        ci = ci_all[:last.nnz].view(cp.int64)
        if self.operands_package == "numpy":
            ro = cp.asnumpy(ro)
            ci = cp.asnumpy(ci)
        return BitMatrixCSR(
            row_offsets=ro,
            col_indices=ci,
            nnz=last.nnz,
            shape=(last.num_shots, self._spec.n_errors),
        )


# Alias
BitMatrixSampler = BitMatrixSparseSampler


# ---------------------------------------------------------------------------
# DEMSampler
# ---------------------------------------------------------------------------

class DEMSampler:
    """High-level sampler that takes a ``stim.DetectorErrorModel`` directly.

    Parses the DEM into a binary matrix and probability array, then
    delegates to :class:`BitMatrixSparseSampler`.
    """

    def __init__(
        self,
        dem: Any,
        max_shots: int,
        options: Optional[Options] = None,
        stream: Stream = None,
        *,
        package: Literal["numpy", "cupy"] = "cupy",
        seed: Optional[int] = None,
    ):
        try:
            import stim  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError("stim is required for DEMSampler") from e

        if options is None:
            options = Options()

        n_det, probs_h, b_csr = _extract_from_dem(dem)
        self._sampler = BitMatrixSparseSampler(
            b_csr,
            cp.asarray(probs_h, dtype=cp.float64),
            max_shots,
            package=package,
            seed=seed,
            options=options,
            stream=stream,
        )

    def sample(self, num_shots: int, *, seed: Optional[int] = None, stream: Stream = None) -> None:
        """Sample; delegates to the underlying matrix sampler."""
        self._sampler.sample(num_shots, seed=seed, stream=stream)

    def get_outcomes(self, bit_packed: bool = True) -> Array:
        return self._sampler.get_outcomes(bit_packed=bit_packed)

    get_detector_samples = get_outcomes

    def get_errors(self, **kwargs):
        return self._sampler.get_errors(**kwargs)
