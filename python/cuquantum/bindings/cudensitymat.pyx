# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.03.0. Do not modify it directly.

cimport cython
from cpython.memoryview cimport PyMemoryView_FromMemory
from cpython.buffer cimport PyBUF_READ, PyBUF_WRITE

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

from ._utils cimport (get_resource_ptr, get_nested_resource_ptr, nested_resource, nullable_unique_ptr,
                      get_buffer_pointer, get_resource_ptrs, DeviceAllocType, DeviceFreeType,
                      cuqnt_alloc_wrapper, cuqnt_free_wrapper, logger_callback_with_data)

from enum import IntEnum as _IntEnum
from collections import defaultdict
from typing import Callable
import traceback
import warnings as _warnings

import numpy as np
import cupy as cp


###############################################################################
# Callback wrappers
###############################################################################

_callback_holders = defaultdict(lambda: set())


cdef cuda_to_numpy_data_type(cudaDataType_t data_type):
    """Convert cudaDataType_t to NumPy data type."""
    if data_type == CUDA_R_32F:
        return np.dtype("float32")
    elif data_type == CUDA_R_64F:
        return np.dtype("float64")
    elif data_type == CUDA_C_32F:
        return np.dtype("complex64")
    elif data_type == CUDA_C_64F:
        return np.dtype("complex128")
    else:
        raise ValueError(
            "Data types other than single/double-precision real or single/double-precision " 
            "complex are not supported."
        )


cdef class WrappedScalarCallback:

    def __cinit__(self, callback: Callable, device: CallbackDevice):
        self.callback = callback
        self.device = device

        self.c_struct = <cudensitymatWrappedScalarCallback_t*>malloc(sizeof(cudensitymatWrappedScalarCallback_t))
        if self.c_struct == NULL:
            raise MemoryError("Failed to allocate memory for WrappedScalarCallback.")

        self.c_struct.callback = <cudensitymatScalarCallback_t>(<void*>callback)
        self.c_struct.device = device
        if device == CallbackDevice.CPU:
            self.c_struct.wrapper = <void*>cpu_scalar_callback_wrapper
        else:
            self.c_struct.wrapper = <void*>gpu_scalar_callback_wrapper

    def __dealloc__(self):
        if self.c_struct != NULL:
            free(self.c_struct)


cdef class WrappedTensorCallback:

    def __cinit__(self, callback: Callable, device: CallbackDevice):
        self.callback = callback
        self.device = device

        self.c_struct = <cudensitymatWrappedTensorCallback_t*>malloc(sizeof(cudensitymatWrappedTensorCallback_t))
        if self.c_struct == NULL:
            raise MemoryError("Failed to allocate memory for WrappedTensorCallback.")

        self.c_struct.callback = <cudensitymatTensorCallback_t>(<void*>callback)
        self.c_struct.device = device
        if device == CallbackDevice.CPU:
            self.c_struct.wrapper = <void*>cpu_tensor_callback_wrapper
        else:
            self.c_struct.wrapper = <void*>gpu_tensor_callback_wrapper

    def __dealloc__(self):
        if self.c_struct != NULL:
            free(self.c_struct)


cdef int32_t cpu_scalar_callback_wrapper(cudensitymatScalarCallback_t _callback_,
                                         double time,
                                         int64_t batch_size,
                                         int32_t num_params,
                                         const double * _params_,
                                         cudaDataType_t _data_type_,
                                         void * _storage_,
                                         cudaStream_t _stream_) with gil:
    """CPU scalar callback wrapper."""
    try:
        # Reconstruct CUDA stream from stream pointer
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)

        with stream:
            # Reconstruct callback from pointer
            callback = <object>(<void *>_callback_)

            # Reconstruct params NumPy array from pointer and size
            params_size = sizeof(double) * num_params * batch_size
            params_buffer = PyMemoryView_FromMemory(<char *>_params_, params_size, PyBUF_READ)
            params = np.ndarray((num_params, batch_size), dtype=np.float64, buffer=params_buffer, order='F')
            
            # Reconstruct storage NumPy array from pointer and size
            data_type = cuda_to_numpy_data_type(_data_type_)
            storage_size = data_type.itemsize * batch_size
            storage_buffer = PyMemoryView_FromMemory(<char *>_storage_, storage_size, PyBUF_WRITE)
            storage = np.ndarray((batch_size,), dtype=data_type, buffer=storage_buffer)

            # Python function call
            callback(time, params, storage)

    except Exception:
        traceback.print_exc()
        return -1

    return 0


cdef int32_t gpu_scalar_callback_wrapper(cudensitymatScalarCallback_t _callback_,
                                         double time,
                                         int64_t batch_size,
                                         int32_t num_params,
                                         const double * _params_,
                                         cudaDataType_t _data_type_,
                                         void * _storage_,
                                         cudaStream_t _stream_) with gil:
    """GPU scalar callback wrapper."""
    try:
        # Reconstruct CUDA stream from stream pointer
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)

        with stream:
            # Reconstruct callback from pointer
            callback = <object>(<void *>_callback_)

            # Reconstruct params CuPy array from pointer and size
            params_size = sizeof(double) * num_params * batch_size        
            params_memory = cp.cuda.UnownedMemory(<intptr_t>_params_, params_size, None)
            params_memory_ptr = cp.cuda.MemoryPointer(params_memory, 0)
            params = cp.ndarray((num_params, batch_size), dtype=cp.float64, memptr=params_memory_ptr, order='F')

            # Reconstruct CuPy array from data storage
            data_type = cuda_to_numpy_data_type(_data_type_)
            storage_size = data_type.itemsize * batch_size
            storage_memory = cp.cuda.UnownedMemory(<intptr_t>_storage_, storage_size, None)
            storage_memory_ptr = cp.cuda.MemoryPointer(storage_memory, 0)
            storage = cp.ndarray((batch_size,), dtype=data_type, memptr=storage_memory_ptr)
            
            # Python function call
            callback(time, params, storage)

    except Exception:
        traceback.print_exc()
        return -1

    return 0


cdef int32_t cpu_tensor_callback_wrapper(cudensitymatTensorCallback_t _callback_,
                                         cudensitymatElementaryOperatorSparsity_t sparsity,
                                         int32_t num_modes,
                                         const int64_t * _mode_extents_,
                                         const int32_t * _diagonal_offsets_,
                                         double time,
                                         int64_t batch_size,
                                         int32_t num_params,
                                         const double * _params_,
                                         cudaDataType_t _data_type_,
                                         void * _storage_,
                                         cudaStream_t _stream_) with gil:
    """CPU tensor callback wrapper."""
    try:
        # Reconstruct CUDA stream from stream pointer
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)
        
        with stream:
            # Reconstruct callback and mode extents from pointers
            callback = <object>(<void *>_callback_)
            mode_extents = tuple(_mode_extents_[i] for i in range(num_modes))

            # Reconstruct params NumPy array from pointer and size
            params_size = sizeof(double) * num_params * batch_size
            params_buffer = PyMemoryView_FromMemory(<char *>_params_, params_size, PyBUF_READ)
            params = np.ndarray((num_params, batch_size), dtype=np.float64, buffer=params_buffer, order='F')

            # Reconstruct NumPy array from data storage
            data_type = cuda_to_numpy_data_type(_data_type_)
            storage_size = data_type.itemsize * np.prod(mode_extents) * batch_size
            storage_buffer = PyMemoryView_FromMemory(<char *>_storage_, storage_size, PyBUF_WRITE)
            storage = np.ndarray((*mode_extents, batch_size), dtype=data_type, buffer=storage_buffer, order='F')

            # Python function call
            callback(time, params, storage)

    except Exception:
        traceback.print_exc()
        return -1

    return 0


cdef int32_t gpu_tensor_callback_wrapper(cudensitymatTensorCallback_t _callback_,
                                         cudensitymatElementaryOperatorSparsity_t sparsity,
                                         int32_t num_modes,
                                         const int64_t * _mode_extents_,
                                         const int32_t * _diagonal_offsets_,
                                         double time,
                                         int64_t batch_size,
                                         int32_t num_params,
                                         const double * _params_,
                                         cudaDataType_t _data_type_,
                                         void * _storage_,
                                         cudaStream_t _stream_) with gil:
    """GPU tensor callback wrapper."""
    try:
        # Reconstruct CUDA stream from stream pointer
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)

        with stream:
            # Reconstruct callback and mode extents from pointers
            callback = <object>(<void *>_callback_)
            mode_extents = tuple(_mode_extents_[i] for i in range(num_modes))

            # Reconstruct params CuPy array from pointer and size
            params_size = sizeof(double) * num_params * batch_size        
            params_memory = cp.cuda.UnownedMemory(<intptr_t>_params_, params_size, None)
            params_memory_ptr = cp.cuda.MemoryPointer(params_memory, 0)
            params = cp.ndarray((num_params, batch_size), dtype=cp.float64, memptr=params_memory_ptr, order='F')

            # Construct CuPy array from data storage
            data_type = cuda_to_numpy_data_type(_data_type_)
            storage_size = data_type.itemsize * np.prod(mode_extents) * batch_size
            storage_memory = cp.cuda.UnownedMemory(<intptr_t>_storage_, storage_size, None)
            storage_memory_ptr = cp.cuda.MemoryPointer(storage_memory, 0)
            storage = cp.ndarray((*mode_extents, batch_size), dtype=data_type, memptr=storage_memory_ptr, order='F')

            # Python function call
            callback(time, params, storage)

    except Exception:
        traceback.print_exc()
        return -1

    return 0


###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """See `cudensitymatStatus_t`."""
    SUCCESS = CUDENSITYMAT_STATUS_SUCCESS
    NOT_INITIALIZED = CUDENSITYMAT_STATUS_NOT_INITIALIZED
    ALLOC_FAILED = CUDENSITYMAT_STATUS_ALLOC_FAILED
    INVALID_VALUE = CUDENSITYMAT_STATUS_INVALID_VALUE
    ARCH_MISMATCH = CUDENSITYMAT_STATUS_ARCH_MISMATCH
    EXECUTION_FAILED = CUDENSITYMAT_STATUS_EXECUTION_FAILED
    INTERNAL_ERROR = CUDENSITYMAT_STATUS_INTERNAL_ERROR
    NOT_SUPPORTED = CUDENSITYMAT_STATUS_NOT_SUPPORTED
    CALLBACK_ERROR = CUDENSITYMAT_STATUS_CALLBACK_ERROR
    CUBLAS_ERROR = CUDENSITYMAT_STATUS_CUBLAS_ERROR
    CUDA_ERROR = CUDENSITYMAT_STATUS_CUDA_ERROR
    INSUFFICIENT_WORKSPACE = CUDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE
    INSUFFICIENT_DRIVER = CUDENSITYMAT_STATUS_INSUFFICIENT_DRIVER
    IO_ERROR = CUDENSITYMAT_STATUS_IO_ERROR
    CUTENSOR_VERSION_MISMATCH = CUDENSITYMAT_STATUS_CUTENSOR_VERSION_MISMATCH
    NO_DEVICE_ALLOCATOR = CUDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR
    CUTENSOR_ERROR = CUDENSITYMAT_STATUS_CUTENSOR_ERROR
    CUSOLVER_ERROR = CUDENSITYMAT_STATUS_CUSOLVER_ERROR
    DEVICE_ALLOCATOR_ERROR = CUDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR
    DISTRIBUTED_FAILURE = CUDENSITYMAT_STATUS_DISTRIBUTED_FAILURE
    INTERRUPTED = CUDENSITYMAT_STATUS_INTERRUPTED
    CUTENSORNET_ERROR = CUDENSITYMAT_STATUS_CUTENSORNET_ERROR

class ComputeType(_IntEnum):
    """See `cudensitymatComputeType_t`."""
    COMPUTE_32F = CUDENSITYMAT_COMPUTE_32F
    COMPUTE_64F = CUDENSITYMAT_COMPUTE_64F

class DistributedProvider(_IntEnum):
    """See `cudensitymatDistributedProvider_t`."""
    NONE = CUDENSITYMAT_DISTRIBUTED_PROVIDER_NONE
    MPI = CUDENSITYMAT_DISTRIBUTED_PROVIDER_MPI

class CallbackDevice(_IntEnum):
    """See `cudensitymatCallbackDevice_t`."""
    CPU = CUDENSITYMAT_CALLBACK_DEVICE_CPU
    GPU = CUDENSITYMAT_CALLBACK_DEVICE_GPU

class StatePurity(_IntEnum):
    """See `cudensitymatStatePurity_t`."""
    PURE = CUDENSITYMAT_STATE_PURITY_PURE
    MIXED = CUDENSITYMAT_STATE_PURITY_MIXED

class ElementaryOperatorSparsity(_IntEnum):
    """See `cudensitymatElementaryOperatorSparsity_t`."""
    OPERATOR_SPARSITY_NONE = CUDENSITYMAT_OPERATOR_SPARSITY_NONE
    OPERATOR_SPARSITY_MULTIDIAGONAL = CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL

class Memspace(_IntEnum):
    """See `cudensitymatMemspace_t`."""
    DEVICE = CUDENSITYMAT_MEMSPACE_DEVICE
    HOST = CUDENSITYMAT_MEMSPACE_HOST

class WorkspaceKind(_IntEnum):
    """See `cudensitymatWorkspaceKind_t`."""
    WORKSPACE_SCRATCH = CUDENSITYMAT_WORKSPACE_SCRATCH


###############################################################################
# Error handling
###############################################################################

cpdef str get_error_string(int error):
    """Returns the description string for an error code.

    Args:
        error (Status): Error code to convert to string.

    .. seealso:: `cudensitymatGetErrorString`
    """
    return ""


class cuDensityMatError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value}): {get_error_string(status)}"
        super(cuDensityMatError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuDensityMatError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef get_version():
    """Returns the semantic version number of the cuDensityMat library.

    .. seealso:: `cudensitymatGetVersion`
    """
    with nogil:
        status = cudensitymatGetVersion()
    check_status(status)


cpdef intptr_t create() except? 0:
    """Creates and initializes the library context.

    Returns:
        intptr_t: Library handle.

    .. seealso:: `cudensitymatCreate`
    """
    cdef Handle handle
    with nogil:
        status = cudensitymatCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """Destroys the library context.

    Args:
        handle (intptr_t): Library handle.

    .. seealso:: `cudensitymatDestroy`
    """
    with nogil:
        status = cudensitymatDestroy(<Handle>handle)
    check_status(status)


cpdef reset_distributed_configuration(intptr_t handle, int provider, intptr_t comm_ptr, size_t comm_size):
    """Resets the current distributed execution configuration associated with the given library context by importing a user-provided inter-process communicator (e.g., MPI_Comm).

    Args:
        handle (intptr_t): Library handle.
        provider (DistributedProvider): Communication service provider.
        comm_ptr (intptr_t): Pointer to the communicator in a type-erased form.
        comm_size (size_t): Size of the communicator in bytes.

    .. seealso:: `cudensitymatResetDistributedConfiguration`
    """
    with nogil:
        status = cudensitymatResetDistributedConfiguration(<Handle>handle, <_DistributedProvider>provider, <const void*>comm_ptr, comm_size)
    check_status(status)


cpdef int32_t get_num_ranks(intptr_t handle) except? -1:
    """Returns the total number of distributed processes associated with the given library context in its current distributed execution configuration.

    Args:
        handle (intptr_t): Library handle.

    Returns:
        int32_t: Number of distributed processes.

    .. seealso:: `cudensitymatGetNumRanks`
    """
    cdef int32_t num_ranks
    with nogil:
        status = cudensitymatGetNumRanks(<const Handle>handle, &num_ranks)
    check_status(status)
    return num_ranks


cpdef int32_t get_proc_rank(intptr_t handle) except? -1:
    """Returns the rank of the current process in the distributed execution configuration associated with the given library context.

    Args:
        handle (intptr_t): Library handle.

    Returns:
        int32_t: Rank of the current distributed process.

    .. seealso:: `cudensitymatGetProcRank`
    """
    cdef int32_t proc_rank
    with nogil:
        status = cudensitymatGetProcRank(<const Handle>handle, &proc_rank)
    check_status(status)
    return proc_rank


cpdef reset_random_seed(intptr_t handle, int32_t random_seed):
    """Resets the context-level random seed used by the random number generator inside the library context.

    Args:
        handle (intptr_t): Library handle.
        random_seed (int32_t): Random seed value.

    .. seealso:: `cudensitymatResetRandomSeed`
    """
    with nogil:
        status = cudensitymatResetRandomSeed(<Handle>handle, random_seed)
    check_status(status)


cpdef intptr_t create_state(intptr_t handle, int purity, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int data_type) except? 0:
    """Defines an empty quantum state of a given purity and shape, or a batch of such quantum states.

    Args:
        handle (intptr_t): Library handle.
        purity (StatePurity): Desired quantum state purity.
        num_space_modes (int32_t): Number of space modes (number of quantum degrees of freedom).
        space_mode_extents (object): Extents of the space modes (dimensions of the quantum degrees of freedom). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        batch_size (int64_t): Batch size (number of equally-shaped quantum states in the batch). Note that setting the batch size to zero is the same as setting it to 1 (no batching).
        data_type (int): Numerical representation data type (type of tensor elements).

    Returns:
        intptr_t: Empty quantum state (or a batch of quantum states).

    .. seealso:: `cudensitymatCreateState`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef State state
    with nogil:
        status = cudensitymatCreateState(<const Handle>handle, <_StatePurity>purity, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), batch_size, <DataType>data_type, &state)
    check_status(status)
    return <intptr_t>state


cpdef destroy_state(intptr_t state):
    """Destroys the quantum state.

    Args:
        state (intptr_t): Quantum state (or a batch of quantum states).

    .. seealso:: `cudensitymatDestroyState`
    """
    with nogil:
        status = cudensitymatDestroyState(<State>state)
    check_status(status)


cpdef int32_t state_get_num_components(intptr_t handle, intptr_t state) except? -1:
    """Queries the number of components (tensors) constituting the chosen quantum state representation (on the current process in multi-process runs).

    Args:
        handle (intptr_t): Library handle.
        state (intptr_t): Quantum state (or a batch of quantum states).

    Returns:
        int32_t: Number of components (tensors) in the quantum state representation (on the current process).

    .. seealso:: `cudensitymatStateGetNumComponents`
    """
    cdef int32_t num_state_components
    with nogil:
        status = cudensitymatStateGetNumComponents(<const Handle>handle, <const State>state, &num_state_components)
    check_status(status)
    return num_state_components


cpdef state_attach_component_storage(intptr_t handle, intptr_t state, int32_t num_state_components, component_buffer, component_buffer_size):
    """Attaches a user-owned GPU-accessible storage buffer for each component (tensor) constituting the quantum state representation (on the current process in multi-process runs).

    Args:
        handle (intptr_t): Library handle.
        state (intptr_t): Quantum state (or a batch of quantum states).
        num_state_components (int32_t): Number of components (tensors) in the quantum state representation (on the current process). The number of components can be retrived by calling the API function ``cudensitymatStateGetNumComponents``.
        component_buffer (object): Pointers to user-owned GPU-accessible storage buffers for all components (tensors) constituting the quantum state representation (on the current process). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        component_buffer_size (object): Sizes of the provded storage buffers for all components (tensors) constituting the quantum state representation (on the current process). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``size_t``.


    .. seealso:: `cudensitymatStateAttachComponentStorage`
    """
    cdef nullable_unique_ptr[ vector[void*] ] _component_buffer_
    get_resource_ptrs[void](_component_buffer_, component_buffer, <void*>NULL)
    cdef nullable_unique_ptr[ vector[size_t] ] _component_buffer_size_
    get_resource_ptr[size_t](_component_buffer_size_, component_buffer_size, <size_t*>NULL)
    with nogil:
        status = cudensitymatStateAttachComponentStorage(<const Handle>handle, <State>state, num_state_components, <void**>(_component_buffer_.data()), <const size_t*>(_component_buffer_size_.data()))
    check_status(status)


cpdef state_get_component_num_modes(intptr_t handle, intptr_t state, int32_t state_component_local_id, intptr_t state_component_global_id, intptr_t state_component_num_modes, intptr_t batch_mode_location):
    """Queries the number of modes in a local component tensor (on the current process in multi-process runs).

    Args:
        handle (intptr_t): Library handle.
        state (intptr_t): Quantum state (or a batch of quantum states).
        state_component_local_id (int32_t): Component local id (on the current parallel process).
        state_component_global_id (intptr_t): Component global id (across all parallel processes).
        state_component_num_modes (intptr_t): Number of modes in the queried component tensor.
        batch_mode_location (intptr_t): Location of the batch mode (or -1 if the batch mode is absent).

    .. seealso:: `cudensitymatStateGetComponentNumModes`
    """
    with nogil:
        status = cudensitymatStateGetComponentNumModes(<const Handle>handle, <State>state, state_component_local_id, <int32_t*>state_component_global_id, <int32_t*>state_component_num_modes, <int32_t*>batch_mode_location)
    check_status(status)


cpdef state_get_component_info(intptr_t handle, intptr_t state, int32_t state_component_local_id, intptr_t state_component_global_id, intptr_t state_component_num_modes, intptr_t state_component_mode_extents, intptr_t state_component_mode_offsets):
    """Queries information for a locally stored component tensor which represents either the full component or its slice (on the current process in multi-process runs).

    Args:
        handle (intptr_t): Library handle.
        state (intptr_t): Quantum state (or a batch of quantum states).
        state_component_local_id (int32_t): Component local id (on the current parallel process).
        state_component_global_id (intptr_t): Component global id (across all parallel processes).
        state_component_num_modes (intptr_t): Number of modes in the queried component tensor.
        state_component_mode_extents (intptr_t): Component tensor mode extents (the size of the array must be sufficient, see ``cudensitymatStateGetComponentNumModes``).
        state_component_mode_offsets (intptr_t): Component tensor mode offsets defining the locally stored slice (the size of the array must be sufficient, see ``cudensitymatStateGetComponentNumModes``).

    .. seealso:: `cudensitymatStateGetComponentInfo`
    """
    with nogil:
        status = cudensitymatStateGetComponentInfo(<const Handle>handle, <State>state, state_component_local_id, <int32_t*>state_component_global_id, <int32_t*>state_component_num_modes, <int64_t*>state_component_mode_extents, <int64_t*>state_component_mode_offsets)
    check_status(status)


cpdef state_initialize_zero(intptr_t handle, intptr_t state, intptr_t stream):
    """Initializes the quantum state to zero (null state).

    Args:
        handle (intptr_t): Library handle.
        state (intptr_t): Quantum state (or a batch of quantum states).
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatStateInitializeZero`
    """
    with nogil:
        status = cudensitymatStateInitializeZero(<const Handle>handle, <State>state, <Stream>stream)
    check_status(status)


cpdef state_compute_scaling(intptr_t handle, intptr_t state, intptr_t scaling_factors, intptr_t stream):
    """Multiplies the quantum state(s) by a scalar factor(s).

    Args:
        handle (intptr_t): Library handle.
        state (intptr_t): Quantum state (or a batch of quantum states).
        scaling_factors (intptr_t): Array of scaling factor(s) of dimension equal to the batch size in the GPU-accessible RAM (same data type as used by the quantum state).
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatStateComputeScaling`
    """
    with nogil:
        status = cudensitymatStateComputeScaling(<const Handle>handle, <State>state, <const void*>scaling_factors, <Stream>stream)
    check_status(status)


cpdef state_compute_norm(intptr_t handle, intptr_t state, intptr_t norm, intptr_t stream):
    """Computes the squared Frobenius norm(s) of the quantum state(s).

    Args:
        handle (intptr_t): Library handle.
        state (intptr_t): Quantum state (or a batch of quantum states).
        norm (intptr_t): Pointer to the squared Frobenius norm(s) vector storage in the GPU-accessible RAM (float or double real data type).
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatStateComputeNorm`
    """
    with nogil:
        status = cudensitymatStateComputeNorm(<const Handle>handle, <const State>state, <void*>norm, <Stream>stream)
    check_status(status)


cpdef state_compute_trace(intptr_t handle, intptr_t state, intptr_t trace, intptr_t stream):
    """Computes the trace(s) of the quantum state(s).

    Args:
        handle (intptr_t): Library handle.
        state (intptr_t): Quantum state (or a batch of quantum states).
        trace (intptr_t): Pointer to the trace(s) vector storage in the GPU-accessible RAM (same data type as used by the quantum state).
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatStateComputeTrace`
    """
    with nogil:
        status = cudensitymatStateComputeTrace(<const Handle>handle, <const State>state, <void*>trace, <Stream>stream)
    check_status(status)


cpdef state_compute_accumulation(intptr_t handle, intptr_t state_in, intptr_t state_out, intptr_t scaling_factors, intptr_t stream):
    """Computes accumulation of a quantum state(s) into another quantum state(s) of compatible shape.

    Args:
        handle (intptr_t): Library handle.
        state_in (intptr_t): Accumulated quantum state (or a batch of quantum states).
        state_out (intptr_t): Accumulating quantum state (or a batch of quantum states).
        scaling_factors (intptr_t): Array of scaling factor(s) of dimension equal to the batch size in the GPU-accessible RAM (same data type as used by the quantum state).
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatStateComputeAccumulation`
    """
    with nogil:
        status = cudensitymatStateComputeAccumulation(<const Handle>handle, <const State>state_in, <State>state_out, <const void*>scaling_factors, <Stream>stream)
    check_status(status)


cpdef state_compute_inner_product(intptr_t handle, intptr_t state_left, intptr_t state_right, intptr_t inner_product, intptr_t stream):
    """Computes the inner product(s) between the left quantum state(s) and the right quantum state(s): < state(s)Left | state(s)Right >.

    Args:
        handle (intptr_t): Library handle.
        state_left (intptr_t): Left quantum state (or a batch of quantum states).
        state_right (intptr_t): Right quantum state (or a batch of quantum states).
        inner_product (intptr_t): Pointer to the inner product(s) vector storage in the GPU-accessible RAM (same data type as the one used by the quantum states).
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatStateComputeInnerProduct`
    """
    with nogil:
        status = cudensitymatStateComputeInnerProduct(<const Handle>handle, <const State>state_left, <const State>state_right, <void*>inner_product, <Stream>stream)
    check_status(status)


cpdef intptr_t create_operator_term(intptr_t handle, int32_t num_space_modes, space_mode_extents) except? 0:
    """Creates an empty operator term which is going to be a sum of products of either elementary tensor operators or full matrix operators. Each individual elementary tensor operator within a product acts on a subset of space modes, either from the left or from the right (for each mode). Each full matrix operator within a product acts on all space modes, either from the left or from the right (for all modes).

    Args:
        handle (intptr_t): Library handle.
        num_space_modes (int32_t): Number of modes (quantum degrees of freedom) defining the primary/dual tensor product space in which the operator term will act.
        space_mode_extents (object): Extents of the modes (quantum degrees of freedom) defining the primary/dual tensor product space in which the operator term will act. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    Returns:
        intptr_t: Operator term.

    .. seealso:: `cudensitymatCreateOperatorTerm`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef OperatorTerm operator_term
    with nogil:
        status = cudensitymatCreateOperatorTerm(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), &operator_term)
    check_status(status)
    return <intptr_t>operator_term


cpdef intptr_t create_operator(intptr_t handle, int32_t num_space_modes, space_mode_extents) except? 0:
    """Creates an empty operator which is going to be a collection of operator terms with some coefficients.

    Args:
        handle (intptr_t): Library handle.
        num_space_modes (int32_t): Number of modes (degrees of freedom) defining the primary/dual tensor product space in which the operator term will act.
        space_mode_extents (object): Extents of the modes (degrees of freedom) defining the primary/dual tensor product space in which the operator term will act. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    Returns:
        intptr_t: Operator.

    .. seealso:: `cudensitymatCreateOperator`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef Operator superoperator
    with nogil:
        status = cudensitymatCreateOperator(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), &superoperator)
    check_status(status)
    return <intptr_t>superoperator


cpdef operator_prepare_action(intptr_t handle, intptr_t superoperator, intptr_t state_in, intptr_t state_out, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream):
    """Prepares the operator for an action on a quantum state.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        state_in (intptr_t): Representative input quantum state on which the operator is supposed to act. The actual state acted on during computation may be different, but it has to be of the same shape, kind, and factorization structure (topology, bond dimensions, etc).
        state_out (intptr_t): Representative output quantum state produced by the action of the operator on the input quantum state. The actual state acted on during computation may be different, but it has to be of the same shape, kind, and factorization structure (topology, bond dimensions, etc).
        compute_type (ComputeType): Desired compute type.
        workspace_size_limit (size_t): Workspace buffer size limit (bytes).
        workspace (intptr_t): Empty workspace descriptor on entrance. The workspace size required for the computation will be set on exit.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatOperatorPrepareAction`
    """
    with nogil:
        status = cudensitymatOperatorPrepareAction(<const Handle>handle, <const Operator>superoperator, <const State>state_in, <const State>state_out, <_ComputeType>compute_type, workspace_size_limit, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef operator_compute_action(intptr_t handle, intptr_t superoperator, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state_in, intptr_t state_out, intptr_t workspace, intptr_t stream):
    """Computes the action of the operator on a given input quantum state, accumulating the result in the output quantum state (accumulative action).

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        time (double): Time value.
        batch_size (int64_t): Batch size (>=1).
        num_params (int32_t): Number of variable parameters defined by the user.
        params (intptr_t): F-order 2d-array of user-defined real parameter values: params[num_params, batch_size].
        state_in (intptr_t): Input quantum state (or a batch of input quantum states).
        state_out (intptr_t): Updated resulting quantum state which accumulates operator action on the input quantum state.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatOperatorComputeAction`
    """
    with nogil:
        status = cudensitymatOperatorComputeAction(<const Handle>handle, <const Operator>superoperator, time, batch_size, num_params, <const double*>params, <const State>state_in, <State>state_out, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef intptr_t create_operator_action(intptr_t handle, int32_t num_operators, operators) except? 0:
    """Creates an action descriptor for one or more operators, thus defining an aggregate action of the operator(s) on a set of input quantum states compliant with the operator domains, where all input quantum states can also be batched.

    Args:
        handle (intptr_t): Library handle.
        num_operators (int32_t): Number of operators involved (number of operator-state products).
        operators (object): Constituting operator(s) with the same domain of action. Some of the operators may be set to NULL to represent zero action on a specific input quantum state. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).


    Returns:
        intptr_t: Operator action.

    .. seealso:: `cudensitymatCreateOperatorAction`
    """
    cdef nullable_unique_ptr[ vector[Operator*] ] _operators_
    get_resource_ptrs[Operator](_operators_, operators, <Operator*>NULL)
    cdef OperatorAction operator_action
    with nogil:
        status = cudensitymatCreateOperatorAction(<const Handle>handle, num_operators, <Operator*>(_operators_.data()), &operator_action)
    check_status(status)
    return <intptr_t>operator_action


cpdef destroy_operator_action(intptr_t operator_action):
    """Destroys the operator action descriptor.

    Args:
        operator_action (intptr_t): Operator action.

    .. seealso:: `cudensitymatDestroyOperatorAction`
    """
    with nogil:
        status = cudensitymatDestroyOperatorAction(<OperatorAction>operator_action)
    check_status(status)


cpdef operator_action_prepare(intptr_t handle, intptr_t operator_action, state_in, intptr_t state_out, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream):
    """Prepares the (aggregate) operator(s) action for computation.

    Args:
        handle (intptr_t): Library handle.
        operator_action (intptr_t): Operator(s) action specification.
        state_in (object): Input quantum state(s) for all operator(s) defining the current Operator Action. Each input quantum state can be a batch of quantum states itself (with the same batch size). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        state_out (intptr_t): Updated output quantum state (or a batch) which accumulates the (aggregate) operator(s) action on all input quantum state(s).
        compute_type (ComputeType): Desired compute type.
        workspace_size_limit (size_t): Workspace buffer size limit (bytes).
        workspace (intptr_t): Empty workspace descriptor on entrance. The workspace size required for the computation will be set on exit.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatOperatorActionPrepare`
    """
    cdef nullable_unique_ptr[ vector[State*] ] _state_in_
    get_resource_ptrs[State](_state_in_, state_in, <State*>NULL)
    with nogil:
        status = cudensitymatOperatorActionPrepare(<const Handle>handle, <OperatorAction>operator_action, <const State*>(_state_in_.data()), <const State>state_out, <_ComputeType>compute_type, workspace_size_limit, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef operator_action_compute(intptr_t handle, intptr_t operator_action, double time, int64_t batch_size, int32_t num_params, intptr_t params, state_in, intptr_t state_out, intptr_t workspace, intptr_t stream):
    """Executes the action of one or more operators constituting the aggreggate operator(s) action on the same number of input quantum states, accumulating the results into a single output quantum state.

    Args:
        handle (intptr_t): Library handle.
        operator_action (intptr_t): Operator(s) action.
        time (double): Time value.
        batch_size (int64_t): Batch size (>=1).
        num_params (int32_t): Number of variable parameters defined by the user.
        params (intptr_t): F-order 2d-array of user-defined real parameter values: params[num_params, batch_size].
        state_in (object): Input quantum state(s). Each input quantum state can be a batch of quantum states, in general. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        state_out (intptr_t): Updated output quantum state which accumulates operator action(s) on all input quantum state(s).
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatOperatorActionCompute`
    """
    cdef nullable_unique_ptr[ vector[State*] ] _state_in_
    get_resource_ptrs[State](_state_in_, state_in, <State*>NULL)
    with nogil:
        status = cudensitymatOperatorActionCompute(<const Handle>handle, <OperatorAction>operator_action, time, batch_size, num_params, <const double*>params, <const State*>(_state_in_.data()), <State>state_out, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef intptr_t create_expectation(intptr_t handle, intptr_t superoperator) except? 0:
    """Creates the operator expectation value computation object.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.

    Returns:
        intptr_t: Expectation value object.

    .. seealso:: `cudensitymatCreateExpectation`
    """
    cdef Expectation expectation
    with nogil:
        status = cudensitymatCreateExpectation(<const Handle>handle, <Operator>superoperator, &expectation)
    check_status(status)
    return <intptr_t>expectation


cpdef destroy_expectation(intptr_t expectation):
    """Destroys an expectation value object.

    Args:
        expectation (intptr_t): Expectation value object.

    .. seealso:: `cudensitymatDestroyExpectation`
    """
    with nogil:
        status = cudensitymatDestroyExpectation(<Expectation>expectation)
    check_status(status)


cpdef expectation_prepare(intptr_t handle, intptr_t expectation, intptr_t state, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream):
    """Prepares the expectation value object for computation.

    Args:
        handle (intptr_t): Library handle.
        expectation (intptr_t): Expectation value object.
        state (intptr_t): Quantum state (or a batch of quantum states).
        compute_type (ComputeType): Desired compute type.
        workspace_size_limit (size_t): Workspace buffer size limit (bytes).
        workspace (intptr_t): Empty workspace descriptor on entrance. The workspace size required for the computation will be set on exit.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatExpectationPrepare`
    """
    with nogil:
        status = cudensitymatExpectationPrepare(<const Handle>handle, <Expectation>expectation, <const State>state, <_ComputeType>compute_type, workspace_size_limit, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef expectation_compute(intptr_t handle, intptr_t expectation, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state, intptr_t expectation_value, intptr_t workspace, intptr_t stream):
    """Computes the operator expectation value(s) with respect to the given quantum state(s).

    Args:
        handle (intptr_t): Library handle.
        expectation (intptr_t): Expectation value object.
        time (double): Specified time.
        batch_size (int64_t): Batch size (>=1).
        num_params (int32_t): Number of variable parameters defined by the user.
        params (intptr_t): F-order 2d-array of user-defined real parameter values: params[num_params, batch_size].
        state (intptr_t): Quantum state (or a batch of quantum states).
        expectation_value (intptr_t): Pointer to the expectation value(s) vector storage in GPU-accessible RAM of the same data type as used by the state and operator.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatExpectationCompute`
    """
    with nogil:
        status = cudensitymatExpectationCompute(<const Handle>handle, <Expectation>expectation, time, batch_size, num_params, <const double*>params, <const State>state, <void*>expectation_value, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef intptr_t create_workspace(intptr_t handle) except? 0:
    """Creates a workspace descriptor.

    Args:
        handle (intptr_t): Library handle.

    Returns:
        intptr_t: Workspace descriptor.

    .. seealso:: `cudensitymatCreateWorkspace`
    """
    cdef WorkspaceDescriptor workspace_descr
    with nogil:
        status = cudensitymatCreateWorkspace(<const Handle>handle, &workspace_descr)
    check_status(status)
    return <intptr_t>workspace_descr


cpdef destroy_workspace(intptr_t workspace_descr):
    """Destroys a workspace descriptor.

    Args:
        workspace_descr (intptr_t): Workspace descriptor.

    .. seealso:: `cudensitymatDestroyWorkspace`
    """
    with nogil:
        status = cudensitymatDestroyWorkspace(<WorkspaceDescriptor>workspace_descr)
    check_status(status)


cpdef size_t workspace_get_memory_size(intptr_t handle, intptr_t workspace_descr, int mem_space, int workspace_kind) except? -1:
    """Queries the required workspace buffer size.

    Args:
        handle (intptr_t): Library handle.
        workspace_descr (intptr_t): Workspace descriptor.
        mem_space (Memspace): Memory space.
        workspace_kind (WorkspaceKind): Workspace kind.

    Returns:
        size_t: Required workspace buffer size in bytes.

    .. seealso:: `cudensitymatWorkspaceGetMemorySize`
    """
    cdef size_t memory_buffer_size
    with nogil:
        status = cudensitymatWorkspaceGetMemorySize(<const Handle>handle, <const WorkspaceDescriptor>workspace_descr, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, &memory_buffer_size)
    check_status(status)
    return memory_buffer_size


cpdef workspace_set_memory(intptr_t handle, intptr_t workspace_descr, int mem_space, int workspace_kind, intptr_t memory_buffer, size_t memory_buffer_size):
    """Attaches memory to a workspace buffer.

    Args:
        handle (intptr_t): Library handle.
        workspace_descr (intptr_t): Workspace descriptor.
        mem_space (Memspace): Memory space.
        workspace_kind (WorkspaceKind): Workspace kind.
        memory_buffer (intptr_t): Pointer to a user-owned memory buffer to be used by the specified workspace.
        memory_buffer_size (size_t): Size of the provided memory buffer in bytes.

    .. seealso:: `cudensitymatWorkspaceSetMemory`
    """
    with nogil:
        status = cudensitymatWorkspaceSetMemory(<const Handle>handle, <WorkspaceDescriptor>workspace_descr, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, <void*>memory_buffer, memory_buffer_size)
    check_status(status)


cpdef tuple workspace_get_memory(intptr_t handle, intptr_t workspace_descr, int mem_space, int workspace_kind):
    """Retrieves a workspace buffer.

    Args:
        handle (intptr_t): Library handle.
        workspace_descr (intptr_t): Workspace descriptor.
        mem_space (Memspace): Memory space.
        workspace_kind (WorkspaceKind): Workspace kind.

    Returns:
        A 2-tuple containing:

        - intptr_t: Pointer to a user-owned memory buffer used by the specified workspace.
        - size_t: Size of the memory buffer in bytes.

    .. seealso:: `cudensitymatWorkspaceGetMemory`
    """
    cdef void* memory_buffer
    cdef size_t memory_buffer_size
    with nogil:
        status = cudensitymatWorkspaceGetMemory(<const Handle>handle, <const WorkspaceDescriptor>workspace_descr, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, &memory_buffer, &memory_buffer_size)
    check_status(status)
    return (<intptr_t>memory_buffer, memory_buffer_size)

###############################################################################
# Handwritten functions
###############################################################################

cpdef tuple state_get_component_storage_size(intptr_t handle, intptr_t state, int32_t num_state_components):
    """Queries the storage size (in bytes) for each component (tensor) constituting the quantum state representation (on the current process in multi-process runs).

    Args:
        handle (intptr_t): Library handle.
        state (intptr_t): Quantum state (or a batch of quantum states).
        num_state_components (int32_t): Number of components (tensors) in the quantum state representation (on the current process).

    Returns:
        object: Storage size (bytes) for each component (tensor) consituting the quantum state representation (on the current process).

    .. seealso:: `cudensitymatStateGetComponentStorageSize`
    """
    cdef vector[size_t] _component_buffer_size_
    _component_buffer_size_.resize(num_state_components)
    with nogil:
        status = cudensitymatStateGetComponentStorageSize(<const Handle>handle, <const State>state, num_state_components, _component_buffer_size_.data())
    check_status(status)

    # NOTE: The syntax tuple(_component_buffer_size_[i] for i in range(num_state_components)) did
    # not work, so had to create a list first and then convert to tuple
    component_buffer_size = []
    for i in range(num_state_components):
        component_buffer_size.append(_component_buffer_size_[i])
    return tuple(component_buffer_size)


cpdef intptr_t create_elementary_operator(intptr_t handle, int32_t num_space_modes, space_mode_extents, int sparsity, int32_t num_diagonals, diagonal_offsets, int data_type, intptr_t tensor_data, wrapped_tensor_callback) except? 0:
    """Creates an elementary tensor operator acting on a given number of quantum state modes (aka space modes).

    Args:
        handle (intptr_t): Library handle.
        num_space_modes (int32_t): Number of the (state) space modes acted on.
        space_mode_extents (object): Extents of the (state) space modes acted on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        sparsity (ElementaryOperatorSparsity): Tensor operator sparsity defining the storage scheme.
        num_diagonals (int32_t): For multi-diagonal tensor operator matrices, specifies the total number of non-zero diagonals.
        diagonal_offsets (object): Offsets of the non-zero diagonals (for example, the main diagonal has offset 0, the diagonal right above the main diagonal has offset +1, the diagonal right below the main diagonal has offset -1, and so on). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        data_type (int): Tensor operator data type.
        tensor_data (intptr_t): GPU-accessible pointer to the tensor operator elements storage.
        tensor_callback (object): Optional user-defined tensor callback function which can be called later to fill in the tensor elements in the provided storage, or NULL.

    Returns:
        intptr_t: Elementary tensor operator.

    .. seealso:: `cudensitymatCreateElementaryOperator`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _diagonal_offsets_
    get_resource_ptr[int32_t](_diagonal_offsets_, diagonal_offsets, <int32_t*>NULL)
    cdef ElementaryOperator elem_operator

    cdef cudensitymatWrappedTensorCallback_t _wrapped_tensor_callback_
    if wrapped_tensor_callback is not None:
        _wrapped_tensor_callback_ = (<WrappedTensorCallback>wrapped_tensor_callback).c_struct[0]
    else:
        _wrapped_tensor_callback_ = cudensitymatTensorCallbackNone

    with nogil:
        status = cudensitymatCreateElementaryOperator(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), <_ElementaryOperatorSparsity>sparsity, num_diagonals, <const int32_t*>(_diagonal_offsets_.data()), <DataType>data_type, <void*>tensor_data, _wrapped_tensor_callback_, &elem_operator)
    check_status(status)

    if wrapped_tensor_callback is not None:
        _callback_holders[<intptr_t>elem_operator].add((<WrappedTensorCallback>wrapped_tensor_callback).callback)
    return <intptr_t>elem_operator


cpdef intptr_t create_elementary_operator_batch(intptr_t handle, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int sparsity, int32_t num_diagonals, diagonal_offsets, int data_type, intptr_t tensor_data, wrapped_tensor_callback) except? 0:
    # TODO: Docstring should be updated.
    """Creates an elementary tensor operator acting on a given number of quantum state modes (aka space modes).

    Args:
        handle (intptr_t): Library handle.
        num_space_modes (int32_t): Number of the (state) space modes acted on.
        space_mode_extents (object): Extents of the (state) space modes acted on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        sparsity (ElementaryOperatorSparsity): Tensor operator sparsity defining the storage scheme.
        num_diagonals (int32_t): For multi-diagonal tensor operator matrices, specifies the total number of non-zero diagonals.
        diagonal_offsets (object): Offsets of the non-zero diagonals (for example, the main diagonal has offset 0, the diagonal right above the main diagonal has offset +1, the diagonal right below the main diagonal has offset -1, and so on). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        data_type (int): Tensor operator data type.
        tensor_data (intptr_t): GPU-accessible pointer to the tensor operator elements storage.
        tensor_callback (object): Optional user-defined tensor callback function which can be called later to fill in the tensor elements in the provided storage, or NULL.

    Returns:
        intptr_t: Elementary tensor operator.

    .. seealso:: `cudensitymatCreateElementaryOperator`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _diagonal_offsets_
    get_resource_ptr[int32_t](_diagonal_offsets_, diagonal_offsets, <int32_t*>NULL)
    cdef ElementaryOperator elem_operator

    cdef cudensitymatWrappedTensorCallback_t _wrapped_tensor_callback_
    if wrapped_tensor_callback is not None:
        _wrapped_tensor_callback_ = (<WrappedTensorCallback>wrapped_tensor_callback).c_struct[0]
    else:
        _wrapped_tensor_callback_ = cudensitymatTensorCallbackNone

    with nogil:
        status = cudensitymatCreateElementaryOperatorBatch(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), batch_size, <_ElementaryOperatorSparsity>sparsity, num_diagonals, <const int32_t*>(_diagonal_offsets_.data()), <DataType>data_type, <void*>tensor_data, _wrapped_tensor_callback_, &elem_operator)
    check_status(status)

    if wrapped_tensor_callback is not None:
        _callback_holders[<intptr_t>elem_operator].add((<WrappedTensorCallback>wrapped_tensor_callback).callback)
    return <intptr_t>elem_operator


cpdef intptr_t create_matrix_operator_dense_local(intptr_t handle, int32_t num_space_modes, space_mode_extents, int data_type, intptr_t matrix_data, wrapped_matrix_callback) except? 0:
    # TODO: Docstring is missing.
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef MatrixOperator matrix_operator

    cdef cudensitymatWrappedTensorCallback_t _wrapped_matrix_callback_
    if wrapped_matrix_callback is not None:
        _wrapped_matrix_callback_ = (<WrappedTensorCallback>wrapped_matrix_callback).c_struct[0]
    else:
        _wrapped_matrix_callback_ = cudensitymatTensorCallbackNone

    with nogil:
        status = cudensitymatCreateMatrixOperatorDenseLocal(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), <DataType>data_type, <void*>matrix_data, _wrapped_matrix_callback_, &matrix_operator)
    check_status(status)

    if wrapped_matrix_callback is not None:
        _callback_holders[<intptr_t>matrix_operator].add((<WrappedTensorCallback>wrapped_matrix_callback).callback)
    return <intptr_t>matrix_operator


cpdef intptr_t create_matrix_operator_dense_local_batch(intptr_t handle, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int data_type, intptr_t matrix_data, wrapped_matrix_callback) except? 0:
    # TODO: Docstring is missing.
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef MatrixOperator matrix_operator

    cdef cudensitymatWrappedTensorCallback_t _wrapped_matrix_callback_
    if wrapped_matrix_callback is not None:
        _wrapped_matrix_callback_ = (<WrappedTensorCallback>wrapped_matrix_callback).c_struct[0]
    else:
        _wrapped_matrix_callback_ = cudensitymatTensorCallbackNone

    with nogil:
        status = cudensitymatCreateMatrixOperatorDenseLocalBatch(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), batch_size, <DataType>data_type, <void*>matrix_data, _wrapped_matrix_callback_, &matrix_operator)
    check_status(status)

    if wrapped_matrix_callback is not None:
        _callback_holders[<intptr_t>matrix_operator].add((<WrappedTensorCallback>wrapped_matrix_callback).callback)
    return <intptr_t>matrix_operator


cpdef operator_term_append_elementary_product(intptr_t handle, intptr_t operator_term, int32_t num_elem_operators, elem_operators, state_modes_acted_on, mode_action_duality, coefficient, wrapped_coefficient_callback):
    """Appends a product of elementary tensor operators acting on quantum state modes to the operator term.

    Args:
        handle (intptr_t): Library handle.
        operator_term (intptr_t): Operator term.
        num_elem_operators (int32_t): Number of elementary tensor operators in the tensor operator product.
        elem_operators (object): Elementary tensor operators constituting the tensor operator product. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``ElementaryOperator``.

        state_modes_acted_on (object): State modes acted on by the tensor operator product. This is a concatenated list of the state modes acted on by all constituting elementary tensor operators in the same order how they appear in the elem_operators argument. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mode_action_duality (object): Duality status of each mode action, that is, whether the action applies to a ket mode of the quantum state (value 0) or a bra mode of the quantum state (value 1 or other non-zero). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        coefficient (complex): Constant complex scalar coefficient associated with the tensor operator product.
        coefficient_callback (object): User-defined complex scalar callback function which can be called later to update the scalar coefficient associated with the tensor operator product, or NULL. The total coefficient associated with the tensor operator product is a product of the constant coefficient and the result of the scalar callback function, if defined.

    .. seealso:: `cudensitymatOperatorTermAppendElementaryProduct`
    """
    cdef nullable_unique_ptr[ vector[ElementaryOperator*] ] _elem_operators_
    get_resource_ptrs[ElementaryOperator](_elem_operators_, elem_operators, <ElementaryOperator*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_modes_acted_on_
    get_resource_ptr[int32_t](_state_modes_acted_on_, state_modes_acted_on, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_action_duality_
    get_resource_ptr[int32_t](_mode_action_duality_, mode_action_duality, <int32_t*>NULL)

    cdef cuDoubleComplex _coefficient_
    _coefficient_.x = coefficient.real
    _coefficient_.y = coefficient.imag

    cdef cudensitymatWrappedScalarCallback_t _wrapped_coefficient_callback_
    if wrapped_coefficient_callback is not None:
        _wrapped_coefficient_callback_ = (<WrappedScalarCallback>wrapped_coefficient_callback).c_struct[0]
    else:
        _wrapped_coefficient_callback_ = cudensitymatScalarCallbackNone
    
    with nogil:
        status = cudensitymatOperatorTermAppendElementaryProduct(<const Handle>handle, <OperatorTerm>operator_term, num_elem_operators, <const ElementaryOperator*>(_elem_operators_.data()), <const int32_t*>(_state_modes_acted_on_.data()), <const int32_t*>(_mode_action_duality_.data()), _coefficient_, _wrapped_coefficient_callback_)
    check_status(status)

    if wrapped_coefficient_callback is not None:
        _callback_holders[operator_term].add((<WrappedScalarCallback>wrapped_coefficient_callback).callback)


cpdef operator_term_append_elementary_product_batch(intptr_t handle, intptr_t operator_term, int32_t num_elem_operators, elem_operators, state_modes_acted_on, mode_action_duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, wrapped_coefficient_callback):
    cdef nullable_unique_ptr[ vector[ElementaryOperator*] ] _elem_operators_
    get_resource_ptrs[ElementaryOperator](_elem_operators_, elem_operators, <ElementaryOperator*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_modes_acted_on_
    get_resource_ptr[int32_t](_state_modes_acted_on_, state_modes_acted_on, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_action_duality_
    get_resource_ptr[int32_t](_mode_action_duality_, mode_action_duality, <int32_t*>NULL)

    cdef cudensitymatWrappedScalarCallback_t _wrapped_coefficient_callback_
    if wrapped_coefficient_callback is not None:
        _wrapped_coefficient_callback_ = (<WrappedScalarCallback>wrapped_coefficient_callback).c_struct[0]
    else:
        _wrapped_coefficient_callback_ = cudensitymatScalarCallbackNone

    with nogil:
        status = cudensitymatOperatorTermAppendElementaryProductBatch(<const Handle>handle, <OperatorTerm>operator_term, num_elem_operators, <const ElementaryOperator*>(_elem_operators_.data()), <const int32_t*>(_state_modes_acted_on_.data()), <const int32_t*>(_mode_action_duality_.data()), batch_size, <const cuDoubleComplex *>static_coefficients, <cuDoubleComplex *>total_coefficients, _wrapped_coefficient_callback_)
    check_status(status)

    if wrapped_coefficient_callback is not None:
        _callback_holders[operator_term].add((<WrappedScalarCallback>wrapped_coefficient_callback).callback)


cpdef operator_term_append_general_product(intptr_t handle, intptr_t operator_term, int32_t num_elem_operators, num_operator_modes, operator_mode_extents, operator_mode_strides, state_modes_acted_on, mode_action_duality, int data_type, tensor_data, wrapped_tensor_callbacks, coefficient, wrapped_coefficient_callback):
    """Appends a product of generic dense tensor operators acting on different quantum state modes to the operator term.

    Args:
        handle (intptr_t): Library handle.
        operator_term (intptr_t): Operator term.
        num_elem_operators (int32_t): Number of dense tensor operators in the given tensor operator product.
        num_operator_modes (object): Number of modes in each tensor operator (twice the number of state modes it acts on). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        operator_mode_extents (object): Mode extents for each dense tensor operator. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence, or
            - a nested Python sequence of ``int64_t``.

        operator_mode_strides (object): Mode strides for each dense tensor operator. If a specific element is set to NULL, the corresponding dense tensor operator will assume the default generalized column-wise storage strides. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence, or
            - a nested Python sequence of ``int64_t``.

        state_modes_acted_on (object): State modes acted on by the tensor operator product. This is a concatenated list of the state modes acted on by all constituting dense tensor operators in the same order how they appear in the above arguments. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mode_action_duality (object): Duality status of each mode action, whether the action applies to a ket mode of the quantum state (value 0) or a bra mode of the quantum state (value 1 or other non-zero). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        data_type (int): Data type (for all dense tensor operators).
        tensor_data (object): GPU-accessible pointers to the elements of each dense tensor operator constituting the tensor operator product. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``intptr_t``.

        tensor_callbacks (object): User-defined tensor callback functions which can be called later to update the elements of each dense tensor operator (any of the callbacks can be NULL). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cudensitymatWrappedTensorCallback_t``.

        coefficient (complex): Constant complex scalar coefficient associated with the tensor operator product.
        coefficient_callback (object): User-defined complex scalar callback function which can be called later to update the scalar coefficient associated with the tensor operator product, or NULL. The total coefficient associated with the tensor operator product is a product of the constant coefficient and the result of the scalar callback function, if defined.

    .. seealso:: `cudensitymatOperatorTermAppendGeneralProduct`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _num_operator_modes_
    get_resource_ptr[int32_t](_num_operator_modes_, num_operator_modes, <int32_t*>NULL)
    cdef nested_resource[ int64_t ] _operator_mode_extents_
    get_nested_resource_ptr[int64_t](_operator_mode_extents_, operator_mode_extents, <int64_t*>NULL)
    cdef nested_resource[ int64_t ] _operator_mode_strides_
    get_nested_resource_ptr[int64_t](_operator_mode_strides_, operator_mode_strides, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_modes_acted_on_
    get_resource_ptr[int32_t](_state_modes_acted_on_, state_modes_acted_on, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_action_duality_
    get_resource_ptr[int32_t](_mode_action_duality_, mode_action_duality, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[intptr_t] ] _tensor_data_
    get_resource_ptr[intptr_t](_tensor_data_, tensor_data, <intptr_t*>NULL)
    
    cdef vector[cudensitymatWrappedTensorCallback_t] _wrapped_tensor_callbacks_
    cdef wrapped_tensor_callback
    
    for i in range(num_elem_operators):
        if wrapped_tensor_callbacks[i] is not None:
            _wrapped_tensor_callbacks_.push_back((<WrappedTensorCallback>wrapped_tensor_callbacks[i]).c_struct[0])
        else:
            _wrapped_tensor_callbacks_.push_back(cudensitymatTensorCallbackNone)

    cdef cuDoubleComplex _coefficient_
    _coefficient_.x = coefficient.real
    _coefficient_.y = coefficient.imag

    cdef cudensitymatWrappedScalarCallback_t _wrapped_coefficient_callback_
    if wrapped_coefficient_callback is not None:
        _wrapped_coefficient_callback_ = (<WrappedScalarCallback>wrapped_coefficient_callback).c_struct[0]
    else:
        _wrapped_coefficient_callback_ = cudensitymatScalarCallbackNone
    
    with nogil:
        status = cudensitymatOperatorTermAppendGeneralProduct(<const Handle>handle, <OperatorTerm>operator_term, num_elem_operators, <const int32_t*>(_num_operator_modes_.data()), <const int64_t**>(_operator_mode_extents_.ptrs.data()), 
        <const int64_t**>(_operator_mode_strides_.ptrs.data()), 
        # NULL,
        <const int32_t*>(_state_modes_acted_on_.data()), <const int32_t*>(_mode_action_duality_.data()), <DataType>data_type, <void**>(_tensor_data_.data()), <cudensitymatWrappedTensorCallback_t*>(_wrapped_tensor_callbacks_.data()), _coefficient_, _wrapped_coefficient_callback_)
    check_status(status)

    if wrapped_coefficient_callback is not None:
        _callback_holders[operator_term].add((<WrappedScalarCallback>wrapped_coefficient_callback).callback)


cpdef operator_term_append_matrix_product(intptr_t handle, intptr_t operator_term, int32_t num_matrix_operators, matrix_operators, matrix_conjugation, action_duality, coefficient, wrapped_coefficient_callback):
    cdef nullable_unique_ptr[ vector[MatrixOperator*] ] _matrix_operators_
    get_resource_ptrs[MatrixOperator](_matrix_operators_, matrix_operators, <MatrixOperator*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _matrix_conjugation_
    get_resource_ptr[int32_t](_matrix_conjugation_, matrix_conjugation, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _action_duality_
    get_resource_ptr[int32_t](_action_duality_, action_duality, <int32_t*>NULL)

    cdef cuDoubleComplex _coefficient_
    _coefficient_.x = coefficient.real
    _coefficient_.y = coefficient.imag

    cdef cudensitymatWrappedScalarCallback_t _wrapped_coefficient_callback_
    if wrapped_coefficient_callback is not None:
        _wrapped_coefficient_callback_ = (<WrappedScalarCallback>wrapped_coefficient_callback).c_struct[0]
    else:
        _wrapped_coefficient_callback_ = cudensitymatScalarCallbackNone
    
    with nogil:
        status = cudensitymatOperatorTermAppendMatrixProduct(<const Handle>handle, <OperatorTerm>operator_term, num_matrix_operators, <const MatrixOperator*>(_matrix_operators_.data()), <const int32_t*>(_matrix_conjugation_.data()), <const int32_t*>(_action_duality_.data()), _coefficient_, _wrapped_coefficient_callback_)
    check_status(status)

    if wrapped_coefficient_callback is not None:
        _callback_holders[operator_term].add((<WrappedScalarCallback>wrapped_coefficient_callback).callback)


cpdef operator_term_append_matrix_product_batch(intptr_t handle, intptr_t operator_term, int32_t num_matrix_operators, matrix_operators, matrix_conjugation, action_duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, wrapped_coefficient_callback):

    cdef nullable_unique_ptr[ vector[MatrixOperator*] ] _matrix_operators_
    get_resource_ptrs[MatrixOperator](_matrix_operators_, matrix_operators, <MatrixOperator*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _matrix_conjugation_
    get_resource_ptr[int32_t](_matrix_conjugation_, matrix_conjugation, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _action_duality_
    get_resource_ptr[int32_t](_action_duality_, action_duality, <int32_t*>NULL)

    cdef cudensitymatWrappedScalarCallback_t _wrapped_coefficient_callback_
    if wrapped_coefficient_callback is not None:
        _wrapped_coefficient_callback_ = (<WrappedScalarCallback>wrapped_coefficient_callback).c_struct[0]
    else:
        _wrapped_coefficient_callback_ = cudensitymatScalarCallbackNone
    
    with nogil:
        status = cudensitymatOperatorTermAppendMatrixProductBatch(<const Handle>handle, <OperatorTerm>operator_term, num_matrix_operators, <const MatrixOperator*>(_matrix_operators_.data()), <const int32_t*>(_matrix_conjugation_.data()), <const int32_t*>(_action_duality_.data()), batch_size, <const cuDoubleComplex *>static_coefficients, <cuDoubleComplex *>total_coefficients, _wrapped_coefficient_callback_)
    check_status(status)

    if wrapped_coefficient_callback is not None:
        _callback_holders[operator_term].add((<WrappedScalarCallback>wrapped_coefficient_callback).callback)


cpdef operator_append_term(intptr_t handle, intptr_t superoperator, intptr_t operator_term, int32_t duality, coefficient, wrapped_coefficient_callback):
    """Appends an operator term to the operator.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        operator_term (intptr_t): Operator term.
        duality (int32_t): Duality status of the operator term action as a whole. If not zero, the duality status of each mode action inside the operator term will be flipped, that is, action from the left will be replaced by action from the right, and vice versa.
        coefficient (complex): Constant complex scalar coefficient associated with the operator term.
        coefficient_callback (object): User-defined complex scalar callback function which can be called later to update the scalar coefficient associated with the operator term, or NULL. The total coefficient associated with the operator term is a product of the constant coefficient and the result of the scalar callback function, if defined.

    .. seealso:: `cudensitymatOperatorAppendTerm`
    """
    cdef cuDoubleComplex _coefficient_
    _coefficient_.x = coefficient.real
    _coefficient_.y = coefficient.imag

    cdef cudensitymatWrappedScalarCallback_t _wrapped_coefficient_callback_
    if wrapped_coefficient_callback is not None:
        _wrapped_coefficient_callback_ = (<WrappedScalarCallback>wrapped_coefficient_callback).c_struct[0]
    else:
        _wrapped_coefficient_callback_ = cudensitymatScalarCallbackNone

    with nogil:
        status = cudensitymatOperatorAppendTerm(<const Handle>handle, <Operator>superoperator, <OperatorTerm>operator_term, duality, _coefficient_, _wrapped_coefficient_callback_)
    check_status(status)

    if wrapped_coefficient_callback is not None:
        _callback_holders[operator_term].add((<WrappedScalarCallback>wrapped_coefficient_callback).callback)


cpdef operator_append_term_batch(intptr_t handle, intptr_t superoperator, intptr_t operator_term, int32_t duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, wrapped_coefficient_callback):
    """Appends an operator term to the operator.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        operator_term (intptr_t): Operator term.
        duality (int32_t): Duality status of the operator term action as a whole. If not zero, the duality status of each mode action inside the operator term will be flipped, that is, action from the left will be replaced by action from the right, and vice versa.
        coefficient (complex): Constant complex scalar coefficient associated with the operator term.
        coefficient_callback (object): User-defined complex scalar callback function which can be called later to update the scalar coefficient associated with the operator term, or NULL. The total coefficient associated with the operator term is a product of the constant coefficient and the result of the scalar callback function, if defined.

    .. seealso:: `cudensitymatOperatorAppendTerm`
    """
    cdef cudensitymatWrappedScalarCallback_t _wrapped_coefficient_callback_
    if wrapped_coefficient_callback is not None:
        _wrapped_coefficient_callback_ = (<WrappedScalarCallback>wrapped_coefficient_callback).c_struct[0]
    else:
        _wrapped_coefficient_callback_ = cudensitymatScalarCallbackNone

    with nogil:
        status = cudensitymatOperatorAppendTermBatch(<const Handle>handle, <Operator>superoperator, <OperatorTerm>operator_term, duality, batch_size, <const cuDoubleComplex *>static_coefficients, <cuDoubleComplex *>total_coefficients, _wrapped_coefficient_callback_)
    check_status(status)

    if wrapped_coefficient_callback is not None:
        _callback_holders[superoperator].add((<WrappedScalarCallback>wrapped_coefficient_callback).callback)


cpdef destroy_elementary_operator(intptr_t elem_operator):
    """Destroys an elementary tensor operator (simple or batched).

    Args:
        elem_operator (intptr_t): Elementary tensor operator.

    .. seealso:: `cudensitymatDestroyElementaryOperator`
    """
    with nogil:
        status = cudensitymatDestroyElementaryOperator(<ElementaryOperator>elem_operator)
    check_status(status)
    _callback_holders.pop(elem_operator, None)


cpdef destroy_matrix_operator(intptr_t matrix_operator):
    """Destroys a full matrix operator (simple or batched).

    Args:
        matrix_operator (intptr_t): Full matrix operator.

    .. seealso:: `cudensitymatDestroyMatrixOperator`
    """
    with nogil:
        status = cudensitymatDestroyMatrixOperator(<MatrixOperator>matrix_operator)
    check_status(status)
    _callback_holders.pop(matrix_operator, None)


cpdef destroy_operator_term(intptr_t operator_term):
    """Destroys an operator term.

    Args:
        operator_term (intptr_t): Operator term.

    .. seealso:: `cudensitymatDestroyOperatorTerm`
    """
    with nogil:
        status = cudensitymatDestroyOperatorTerm(<OperatorTerm>operator_term)
    check_status(status)
    _callback_holders.pop(operator_term, None)


cpdef destroy_operator(intptr_t superoperator):
    """Destroys an operator.

    Args:
        superoperator (intptr_t): Operator.

    .. seealso:: `cudensitymatDestroyOperator`
    """
    with nogil:
        status = cudensitymatDestroyOperator(<Operator>superoperator)
    check_status(status)
    _callback_holders.pop(superoperator, None)
