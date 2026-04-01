# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 25.11.0 to 26.02.0, generator version 0.3.1.dev1380+g6ceff55cb.d20260311. Do not modify it directly.

cimport cython  # NOQA
cimport cpython
from libcpp.vector cimport vector

from enum import IntEnum as _IntEnum
from ._utils cimport get_resource_ptr, get_resource_ptrs, nullable_unique_ptr, get_buffer_pointer

import numpy as _numpy


###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """
    Status codes returned by the cuStabilizer library.

    See `custabilizerStatus_t`.
    """
    SUCCESS = CUSTABILIZER_STATUS_SUCCESS
    ERROR = CUSTABILIZER_STATUS_ERROR
    NOT_INITIALIZED = CUSTABILIZER_STATUS_NOT_INITIALIZED
    INVALID_VALUE = CUSTABILIZER_STATUS_INVALID_VALUE
    NOT_SUPPORTED = CUSTABILIZER_STATUS_NOT_SUPPORTED
    ALLOC_FAILED = CUSTABILIZER_STATUS_ALLOC_FAILED
    INTERNAL_ERROR = CUSTABILIZER_STATUS_INTERNAL_ERROR
    INSUFFICIENT_WORKSPACE = CUSTABILIZER_STATUS_INSUFFICIENT_WORKSPACE
    CUDA_ERROR = CUSTABILIZER_STATUS_CUDA_ERROR
    INSUFFICIENT_SPARSE_STORAGE = CUSTABILIZER_STATUS_INSUFFICIENT_SPARSE_STORAGE


###############################################################################
# Error handling
###############################################################################

class cuStabilizerError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value}): {get_error_string(status)}"
        super(cuStabilizerError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuStabilizerError(status)


###############################################################################
# Special dtypes
###############################################################################



###############################################################################
# Wrapper functions
###############################################################################

cpdef int get_version() except? 0:
    """Returns the semantic version number of the cuStabilizer library.

    .. seealso:: `custabilizerGetVersion`
    """
    return custabilizerGetVersion()


cpdef str get_error_string(int status):
    """Get the description string for a given cuStabilizer status code.

    Args:
        status (Status): The status code.

    .. seealso:: `custabilizerGetErrorString`
    """
    cdef bytes _output_
    _output_ = custabilizerGetErrorString(<_Status>status)
    return _output_.decode()


cpdef intptr_t create() except? 0:
    """Create and initialize the library context.

    Returns:
        intptr_t: Library handle.

    .. seealso:: `custabilizerCreate`
    """
    cdef Handle handle
    with nogil:
        __status__ = custabilizerCreate(&handle)
    check_status(__status__)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """Destroy the library context.

    Args:
        handle (intptr_t): Library handle.

    .. seealso:: `custabilizerDestroy`
    """
    with nogil:
        __status__ = custabilizerDestroy(<Handle>handle)
    check_status(__status__)


cpdef int64_t circuit_size_from_string(intptr_t handle, circuit_string) except? 0:
    """Returns the size of the device buffer required for a circuit.

    Args:
        handle (intptr_t): Library handle.
        circuit_string (str): String representation of the circuit.

    Returns:
        int64_t: Size of the buffer in bytes.

    .. seealso:: `custabilizerCircuitSizeFromString`
    """
    if not isinstance(circuit_string, str):
        raise TypeError("circuit_string must be a Python str")
    cdef bytes _temp_circuit_string_ = (<str>circuit_string).encode()
    cdef char* _circuit_string_ = _temp_circuit_string_
    cdef int64_t buffer_size
    with nogil:
        __status__ = custabilizerCircuitSizeFromString(<const Handle>handle, <const char*>_circuit_string_, &buffer_size)
    check_status(__status__)
    return buffer_size


cpdef intptr_t create_circuit_from_string(intptr_t handle, circuit_string, buffer_device, int64_t buffer_size) except? 0:
    """Create a new circuit from a string representation.

    Args:
        handle (intptr_t): Library handle.
        circuit_string (str): String representation of the circuit.
        buffer_device (bytes): Device buffer to store the circuit.
        buffer_size (int64_t): Size of the device buffer in bytes.

    Returns:
        intptr_t: Pointer to the created circuit.

    .. seealso:: `custabilizerCreateCircuitFromString`
    """
    if not isinstance(circuit_string, str):
        raise TypeError("circuit_string must be a Python str")
    cdef bytes _temp_circuit_string_ = (<str>circuit_string).encode()
    cdef char* _circuit_string_ = _temp_circuit_string_
    cdef void* _buffer_device_ = get_buffer_pointer(buffer_device, buffer_size, readonly=False)
    cdef Circuit circuit
    with nogil:
        __status__ = custabilizerCreateCircuitFromString(<const Handle>handle, <const char*>_circuit_string_, <void*>_buffer_device_, buffer_size, &circuit)
    check_status(__status__)
    return <intptr_t>circuit


cpdef destroy_circuit(intptr_t circuit):
    """Destroy a circuit.

    Args:
        circuit (intptr_t): Circuit to destroy.

    .. seealso:: `custabilizerDestroyCircuit`
    """
    with nogil:
        __status__ = custabilizerDestroyCircuit(<Circuit>circuit)
    check_status(__status__)


cpdef intptr_t create_frame_simulator(intptr_t handle, int64_t num_qubits, int64_t num_shots, int64_t num_measurements, int64_t table_stride_major) except? 0:
    """Create a FrameSimulator.

    Args:
        handle (intptr_t): Library handle.
        num_qubits (int64_t): Number of qubits in the Pauli frame.
        num_shots (int64_t): Number of samples to simulate.
        num_measurements (int64_t): Number of measurements in the measurement table.
        table_stride_major (int64_t): Stride over the major axis for all input bit tables. Specified in bytes and must be a multiple of 4.

    Returns:
        intptr_t: Pointer to the created frame simulator.

    .. seealso:: `custabilizerCreateFrameSimulator`
    """
    cdef FrameSimulator frame_simulator
    with nogil:
        __status__ = custabilizerCreateFrameSimulator(<const Handle>handle, num_qubits, num_shots, num_measurements, table_stride_major, &frame_simulator)
    check_status(__status__)
    return <intptr_t>frame_simulator


cpdef destroy_frame_simulator(intptr_t frame_simulator):
    """Destroy the FrameSimulator.

    Args:
        frame_simulator (intptr_t): Frame simulator to destroy.

    .. seealso:: `custabilizerDestroyFrameSimulator`
    """
    with nogil:
        __status__ = custabilizerDestroyFrameSimulator(<FrameSimulator>frame_simulator)
    check_status(__status__)


cpdef frame_simulator_apply_circuit(intptr_t handle, intptr_t frame_simulator, intptr_t circuit, int randomize_frame_after_measurement, uint64_t seed, intptr_t x_table_device, intptr_t z_table_device, intptr_t m_table_device, intptr_t stream):
    """Run Pauli frame simulation using the circuit.

    Args:
        handle (intptr_t): Library handle.
        frame_simulator (intptr_t): An instance of FrameSimulator with parameters consistent with the bit tables.
        circuit (intptr_t): A circuit that acts on at most ``numQubits`` and contains at most ``numMeasurements`` measurements.
        randomize_frame_after_measurement (int): Disabling the randomization is helpful in some cases to focus on the error frame propagation.
        seed (uint64_t): Random seed.
        x_table_device (intptr_t): Device buffer of the X bit table in qubit-major order. Must be of size at least ``numQubits`` * ``tableStrideMajor``.
        z_table_device (intptr_t): Device buffer of the Z bit table in qubit-major order. Must be of size at least ``numQubits`` * ``tableStrideMajor``.
        m_table_device (intptr_t): Device buffer of the measurement bit table in measurement-major order. Must be of size at least ``numMeasurements`` * ``tableStrideMajor``.
        stream (intptr_t): CUDA stream.

    .. seealso:: `custabilizerFrameSimulatorApplyCircuit`
    """
    with nogil:
        __status__ = custabilizerFrameSimulatorApplyCircuit(<const Handle>handle, <FrameSimulator>frame_simulator, <const Circuit>circuit, randomize_frame_after_measurement, seed, <custabilizerBitInt_t*>x_table_device, <custabilizerBitInt_t*>z_table_device, <custabilizerBitInt_t*>m_table_device, <Stream>stream)
    check_status(__status__)


cpdef sample_prob_array(intptr_t handle, int64_t num_samples, int64_t num_probs, intptr_t probs, uint64_t seed, intptr_t samples, intptr_t stream):
    """Sample Bernoulli random bits from a probability array.

    Args:
        handle (intptr_t): Library handle.
        num_samples (int64_t): Number of samples (minor dimension). Must be a multiple of 32.
        num_probs (int64_t): Number of probabilities (major dimension).
        probs (intptr_t): Probability array of length ``num_probs`` (device-accessible pointer). Values should be in [0, 1]; out-of-range values are clamped, NaN is treated as 0.
        seed (uint64_t): Random seed.
        samples (intptr_t): Output buffer for bit-packed samples (device-accessible pointer). Must have at least (num_probs * num_samples / 32) words.
        stream (intptr_t): CUDA stream.

    .. seealso:: `custabilizerSampleProbArray`
    """
    with nogil:
        __status__ = custabilizerSampleProbArray(<Handle>handle, num_samples, num_probs, <const double*>probs, seed, <custabilizerBitInt_t*>samples, <Stream>stream)
    check_status(__status__)


cpdef size_t sample_prob_array_sparse_prepare(intptr_t handle, int64_t num_samples, int64_t num_probs) except? 0:
    """Query the device workspace size required for sparse Bernoulli sampling.

    Args:
        handle (intptr_t): Library handle.
        num_samples (int64_t): Number of samples.
        num_probs (int64_t): Number of probabilities.

    Returns:
        size_t: Required device workspace size in bytes.

    .. seealso:: `custabilizerSampleProbArraySparsePrepare`
    """
    cdef size_t workspace_size
    with nogil:
        __status__ = custabilizerSampleProbArraySparsePrepare(<Handle>handle, num_samples, num_probs, &workspace_size)
    check_status(__status__)
    return workspace_size


cpdef sample_prob_array_sparse_compute(intptr_t handle, int64_t num_samples, int64_t num_probs, intptr_t probs, uint64_t seed, intptr_t nnz, intptr_t column_indices, intptr_t row_offsets, intptr_t workspace, size_t workspace_size, intptr_t stream):
    """Sample Bernoulli random bits from a probability array and return a CSR matrix.

    Args:
        handle (intptr_t): Library handle.
        num_samples (int64_t): Number of samples (rows / shots).
        num_probs (int64_t): Number of probabilities (columns).
        probs (intptr_t): Probability array of length ``num_probs`` (device-accessible pointer). Values should be in [0, 1]; out-of-range values are clamped, NaN is treated as 0.
        seed (uint64_t): Random seed.
        nnz (intptr_t): On input, capacity of ``column_indices``. On output, number of non-zeros used (or required).
        column_indices (intptr_t): Output CSR column indices (device-accessible pointer), length at least ``*nnz`` on input.
        row_offsets (intptr_t): Output CSR row offsets (device-accessible pointer), length ``num_samples+1``.
        workspace (intptr_t): Device-accessible workspace of at least the size returned by Prepare.
        workspace_size (size_t): Size of workspace in bytes.
        stream (intptr_t): CUDA stream.

    .. seealso:: `custabilizerSampleProbArraySparseCompute`
    """
    with nogil:
        __status__ = custabilizerSampleProbArraySparseCompute(<Handle>handle, num_samples, num_probs, <const double*>probs, seed, <uint64_t*>nnz, <uint64_t*>column_indices, <uint64_t*>row_offsets, <void*>workspace, workspace_size, <Stream>stream)
    check_status(__status__)


cpdef gf2_sparse_dense_matrix_multiply(intptr_t handle, uint64_t m, uint64_t n, uint64_t k, uint64_t nnz, intptr_t column_indices, intptr_t row_offsets, intptr_t b, int32_t beta, intptr_t c, intptr_t stream):
    """compute GF(2) sparse-dense matrix multiplication ``c`` = A @ b.

    Args:
        handle (intptr_t): Library handle.
        m (uint64_t): Number of rows of ``A`` and ``c``.
        n (uint64_t): Number of columns of ``b`` and ``c`` (must be a multiple of 32).
        k (uint64_t): Number of columns of ``A`` and rows of ``b``.
        nnz (uint64_t): Number of non-zeros in ``A`` and length of ``column_indices``.
        column_indices (intptr_t): cSR column indices of ``A`` (device-accessible pointer), length ``nnz``.
        row_offsets (intptr_t): cSR row offsets of ``A`` (device-accessible pointer), length ``m+1``.
        b (intptr_t): bit-packed dense input matrix ``b`` (device-accessible pointer).
        beta (int32_t): 0 for assign (c not read), 1 for XOR-accumulate.
        c (intptr_t): bit-packed dense output matrix ``c`` (device-accessible pointer).
        stream (intptr_t): cUDA stream.

    .. seealso:: `custabilizerGF2SparseDenseMatrixMultiply`
    """
    with nogil:
        __status__ = custabilizerGF2SparseDenseMatrixMultiply(<Handle>handle, m, n, k, nnz, <const uint64_t*>column_indices, <const uint64_t*>row_offsets, <const custabilizerBitInt_t*>b, beta, <custabilizerBitInt_t*>c, <Stream>stream)
    check_status(__status__)


cpdef gf2_sparse_sparse_matrix_multiply(intptr_t handle, uint64_t m, uint64_t n, uint64_t k, intptr_t a_column_indices, intptr_t a_row_offsets, uint64_t b_nnz, intptr_t b_column_indices, intptr_t b_row_offsets, int32_t beta, intptr_t c, intptr_t stream):
    """compute GF(2) sparse-sparse matrix multiplication ``c`` = A @ B.

    Args:
        handle (intptr_t): Library handle.
        m (uint64_t): Number of rows of ``A`` and ``c``.
        n (uint64_t): Number of columns of ``B`` and ``c`` (must be a multiple of 32).
        k (uint64_t): Number of columns of ``A`` and rows of ``B``.
        a_column_indices (intptr_t): cSR column indices of ``A`` (device-accessible pointer).
        a_row_offsets (intptr_t): cSR row offsets of ``A`` (device-accessible pointer), length ``m+1``.
        b_nnz (uint64_t): Number of non-zeros in ``B``.
        b_column_indices (intptr_t): cSR column indices of ``B`` (device-accessible pointer), length ``b_nnz``. column indices within each row must be sorted in ascending order.
        b_row_offsets (intptr_t): cSR row offsets of ``B`` (device-accessible pointer), length ``k+1``.
        beta (int32_t): 0 for assign (c not read), 1 for XOR-accumulate.
        c (intptr_t): Bit-packed dense output matrix ``c`` (device-accessible pointer).
        stream (intptr_t): cUDA stream.

    .. seealso:: `custabilizerGF2SparseSparseMatrixMultiply`
    """
    with nogil:
        __status__ = custabilizerGF2SparseSparseMatrixMultiply(<Handle>handle, m, n, k, <const uint64_t*>a_column_indices, <const uint64_t*>a_row_offsets, b_nnz, <const uint64_t*>b_column_indices, <const uint64_t*>b_row_offsets, beta, <custabilizerBitInt_t*>c, <Stream>stream)
    check_status(__status__)


