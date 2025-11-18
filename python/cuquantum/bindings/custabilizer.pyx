# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.11.0. Do not modify it directly.

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
    """See `custabilizerStatus_t`."""
    SUCCESS = CUSTABILIZER_STATUS_SUCCESS
    ERROR = CUSTABILIZER_STATUS_ERROR
    NOT_INITIALIZED = CUSTABILIZER_STATUS_NOT_INITIALIZED
    INVALID_VALUE = CUSTABILIZER_STATUS_INVALID_VALUE
    NOT_SUPPORTED = CUSTABILIZER_STATUS_NOT_SUPPORTED
    ALLOC_FAILED = CUSTABILIZER_STATUS_ALLOC_FAILED
    INTERNAL_ERROR = CUSTABILIZER_STATUS_INTERNAL_ERROR
    INSUFFICIENT_WORKSPACE = CUSTABILIZER_STATUS_INSUFFICIENT_WORKSPACE
    CUDA_ERROR = CUSTABILIZER_STATUS_CUDA_ERROR


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


