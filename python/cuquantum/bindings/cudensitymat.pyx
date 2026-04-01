# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 26.03.0, generator version 0.3.1.dev1332+g03874867a.d20260311. Do not modify it directly.

cimport cython
cimport cpython
cimport cpython.buffer
cimport cpython.memoryview
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

from collections import defaultdict
from enum import IntEnum as _IntEnum
import traceback
from typing import Callable
import warnings as _warnings

from ._utils cimport get_resource_ptr, get_resource_ptrs, nullable_unique_ptr

import numpy as _numpy

include "cudensitymat.pxi"


###############################################################################
# POD
###############################################################################




###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """
    Return status of the library API functions.All library API functions
    return a status which can take one of the following values.

    See `cudensitymatStatus_t`.
    """
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
    """
    Supported compute types.

    See `cudensitymatComputeType_t`.
    """
    COMPUTE_32F = CUDENSITYMAT_COMPUTE_32F
    COMPUTE_64F = CUDENSITYMAT_COMPUTE_64F

class DistributedProvider(_IntEnum):
    """
    Supported providers of the distributed communication service.

    See `cudensitymatDistributedProvider_t`.
    """
    NONE = CUDENSITYMAT_DISTRIBUTED_PROVIDER_NONE
    MPI = CUDENSITYMAT_DISTRIBUTED_PROVIDER_MPI
    NCCL = CUDENSITYMAT_DISTRIBUTED_PROVIDER_NCCL

class CallbackDevice(_IntEnum):
    """
    Supported target devices for user-defined callbacks.

    See `cudensitymatCallbackDevice_t`.
    """
    CPU = CUDENSITYMAT_CALLBACK_DEVICE_CPU
    GPU = CUDENSITYMAT_CALLBACK_DEVICE_GPU

class DifferentiationDir(_IntEnum):
    """
    Supported differentiation directions.

    See `cudensitymatDifferentiationDir_t`.
    """
    BACKWARD = CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD

class StatePurity(_IntEnum):
    """
    Quantum state purity (pure or mixed state).

    See `cudensitymatStatePurity_t`.
    """
    PURE = CUDENSITYMAT_STATE_PURITY_PURE
    MIXED = CUDENSITYMAT_STATE_PURITY_MIXED

class ElementaryOperatorSparsity(_IntEnum):
    """
    Elementary operator sparsity kind.

    See `cudensitymatElementaryOperatorSparsity_t`.
    """
    OPERATOR_SPARSITY_NONE = CUDENSITYMAT_OPERATOR_SPARSITY_NONE
    OPERATOR_SPARSITY_MULTIDIAGONAL = CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL

class OperatorSpectrumKind(_IntEnum):
    """
    Kinds of the operator extreme eigen-spectrum computation.

    See `cudensitymatOperatorSpectrumKind_t`.
    """
    OPERATOR_SPECTRUM_LARGEST = CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST
    OPERATOR_SPECTRUM_SMALLEST = CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST
    OPERATOR_SPECTRUM_LARGEST_REAL = CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL
    OPERATOR_SPECTRUM_SMALLEST_REAL = CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST_REAL

class OperatorSpectrumConfig(_IntEnum):
    """
    Configuration options for the operator extreme eigen-spectrum
    computation.

    See `cudensitymatOperatorSpectrumConfig_t`.
    """
    MAX_EXPANSION = CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_EXPANSION
    MAX_RESTARTS = CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_RESTARTS
    MIN_BLOCK_SIZE = CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MIN_BLOCK_SIZE

class BoundaryCondition(_IntEnum):
    """
    This enum lists supported boundary conditions for supported state
    factorizations.

    See `cudensitymatBoundaryCondition_t`.
    """
    OPEN = CUDENSITYMAT_BOUNDARY_CONDITION_OPEN

class TimePropagationScopeKind(_IntEnum):
    """
    Time propagation scope (full vs split evolution).

    See `cudensitymatTimePropagationScopeKind_t`.
    """
    PROPAGATION_SCOPE_SPLIT = CUDENSITYMAT_PROPAGATION_SCOPE_SPLIT

class TimePropagationScopeSplitKind(_IntEnum):
    """
    Split kind for split-scope propagation.

    See `cudensitymatTimePropagationScopeSplitKind_t`.
    """
    PROPAGATION_SCOPE_SPLIT_TDVP = CUDENSITYMAT_PROPAGATION_SCOPE_SPLIT_TDVP

class TimePropagationApproachKind(_IntEnum):
    """
    Time propagation approach (time integration / exponentiation method).

    See `cudensitymatTimePropagationApproachKind_t`.
    """
    PROPAGATION_APPROACH_KRYLOV = CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV

class TimePropagationAttribute(_IntEnum):
    """
    Time propagation configuration attributes.

    See `cudensitymatTimePropagationAttribute_t`.
    """
    PROPAGATION_SPLIT_SCOPE_KIND = CUDENSITYMAT_PROPAGATION_SPLIT_SCOPE_KIND
    PROPAGATION_SPLIT_SCOPE_TDVP_CONFIG = CUDENSITYMAT_PROPAGATION_SPLIT_SCOPE_TDVP_CONFIG
    PROPAGATION_APPROACH_KRYLOV_CONFIG = CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_CONFIG

class TimePropagationApproachKrylovConfigAttribute(_IntEnum):
    """
    Configuration attributes for Krylov-subspace method.

    See `cudensitymatTimePropagationApproachKrylovConfigAttribute_t`.
    """
    PROPAGATION_APPROACH_KRYLOV_TOLERANCE = CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_TOLERANCE
    PROPAGATION_APPROACH_KRYLOV_MAX_DIM = CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_MAX_DIM
    PROPAGATION_APPROACH_KRYLOV_MIN_BETA = CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_MIN_BETA
    PROPAGATION_APPROACH_KRYLOV_ADAPTIVE_STEP_SIZE = CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_ADAPTIVE_STEP_SIZE

class TimePropagationScopeSplitTDVPConfigAttribute(_IntEnum):
    """
    Configuration attributes for TDVP time propagation method.

    See `cudensitymatTimePropagationScopeSplitTDVPConfigAttribute_t`.
    """
    PROPAGATION_SPLIT_SCOPE_TDVP_ORDER = CUDENSITYMAT_PROPAGATION_SPLIT_SCOPE_TDVP_ORDER

class Memspace(_IntEnum):
    """
    Memory spaces for workspace buffer allocation.

    See `cudensitymatMemspace_t`.
    """
    DEVICE = CUDENSITYMAT_MEMSPACE_DEVICE
    HOST = CUDENSITYMAT_MEMSPACE_HOST

class WorkspaceKind(_IntEnum):
    """
    Kinds of workspace memory buffers.

    See `cudensitymatWorkspaceKind_t`.
    """
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
        __status__ = cudensitymatGetVersion()
    check_status(__status__)


cpdef intptr_t create() except? 0:
    """Creates and initializes the library context.

    Returns:
        intptr_t: Library handle.

    .. seealso:: `cudensitymatCreate`
    """
    cdef Handle handle
    with nogil:
        __status__ = cudensitymatCreate(&handle)
    check_status(__status__)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """Destroys the library context.

    Args:
        handle (intptr_t): Library handle.

    .. seealso:: `cudensitymatDestroy`
    """
    with nogil:
        __status__ = cudensitymatDestroy(<Handle>handle)
    check_status(__status__)


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
        __status__ = cudensitymatResetDistributedConfiguration(<Handle>handle, <_DistributedProvider>provider, <const void*>comm_ptr, comm_size)
    check_status(__status__)


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
        __status__ = cudensitymatGetNumRanks(<const Handle>handle, &num_ranks)
    check_status(__status__)
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
        __status__ = cudensitymatGetProcRank(<const Handle>handle, &proc_rank)
    check_status(__status__)
    return proc_rank


cpdef reset_random_seed(intptr_t handle, int32_t random_seed):
    """Resets the context-level random seed used by the random number generator inside the library context.

    Args:
        handle (intptr_t): Library handle.
        random_seed (int32_t): Random seed value.

    .. seealso:: `cudensitymatResetRandomSeed`
    """
    with nogil:
        __status__ = cudensitymatResetRandomSeed(<Handle>handle, random_seed)
    check_status(__status__)


cpdef intptr_t create_state(intptr_t handle, int purity, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int data_type) except? 0:
    """Defines an empty dense quantum state of a given purity and shape, or a batch of such dense quantum states.

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
        intptr_t: Empty dense quantum state (or a batch of such quantum states).

    .. seealso:: `cudensitymatCreateState`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef State state
    with nogil:
        __status__ = cudensitymatCreateState(<const Handle>handle, <_StatePurity>purity, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), batch_size, <DataType>data_type, &state)
    check_status(__status__)
    return <intptr_t>state


cpdef intptr_t create_state_mps(intptr_t handle, int purity, int32_t num_space_modes, space_mode_extents, int boundary_condition, bond_extents, int data_type, int64_t batch_size) except? 0:
    """Defines an empty quantum state of a given purity and shape in the Matrix Product State (MPS) factorized form, or a batch of such MPS factorized quantum states.

    Args:
        handle (intptr_t): Library handle.
        purity (StatePurity): Desired quantum state purity.
        num_space_modes (int32_t): Number of space modes (number of quantum degrees of freedom).
        space_mode_extents (object): Extents of the space modes (dimensions of the quantum degrees of freedom). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        boundary_condition (BoundaryCondition): Boundary condition.
        bond_extents (object): Extents of the bond modes. For open boundary condition, the length of the array must be equal to the number of space modes minus one, where ``bond_extents[i]`` is the bond dimension between site ``i`` and site ``i+1``. For periodic boundary condition, the length of the array must be equal to the number of space modes. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        data_type (int): Quantum state data type.
        batch_size (int64_t): Batch size (number of equally-shaped quantum states in the batch).

    Returns:
        intptr_t: Empty quantum state (or a batch of quantum states) in the MPS factorized form.

    .. seealso:: `cudensitymatCreateStateMPS`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _bond_extents_
    get_resource_ptr[int64_t](_bond_extents_, bond_extents, <int64_t*>NULL)
    cdef State state
    with nogil:
        __status__ = cudensitymatCreateStateMPS(<const Handle>handle, <_StatePurity>purity, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), <_BoundaryCondition>boundary_condition, <const int64_t*>(_bond_extents_.data()), <DataType>data_type, batch_size, &state)
    check_status(__status__)
    return <intptr_t>state


cpdef destroy_state(intptr_t state):
    """Destroys the quantum state.

    Args:
        state (intptr_t): Quantum state (or a batch of quantum states).

    .. seealso:: `cudensitymatDestroyState`
    """
    with nogil:
        __status__ = cudensitymatDestroyState(<State>state)
    check_status(__status__)


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
        __status__ = cudensitymatStateGetNumComponents(<const Handle>handle, <const State>state, &num_state_components)
    check_status(__status__)
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
        __status__ = cudensitymatStateAttachComponentStorage(<const Handle>handle, <State>state, num_state_components, <void**>(_component_buffer_.data()), <const size_t*>(_component_buffer_size_.data()))
    check_status(__status__)


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
        __status__ = cudensitymatStateGetComponentNumModes(<const Handle>handle, <State>state, state_component_local_id, <int32_t*>state_component_global_id, <int32_t*>state_component_num_modes, <int32_t*>batch_mode_location)
    check_status(__status__)


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
        __status__ = cudensitymatStateGetComponentInfo(<const Handle>handle, <State>state, state_component_local_id, <int32_t*>state_component_global_id, <int32_t*>state_component_num_modes, <int64_t*>state_component_mode_extents, <int64_t*>state_component_mode_offsets)
    check_status(__status__)


cpdef state_initialize_zero(intptr_t handle, intptr_t state, intptr_t stream):
    """Initializes the quantum state to zero (null state).

    Args:
        handle (intptr_t): Library handle.
        state (intptr_t): Quantum state (or a batch of quantum states).
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatStateInitializeZero`
    """
    with nogil:
        __status__ = cudensitymatStateInitializeZero(<const Handle>handle, <State>state, <Stream>stream)
    check_status(__status__)


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
        __status__ = cudensitymatStateComputeScaling(<const Handle>handle, <State>state, <const void*>scaling_factors, <Stream>stream)
    check_status(__status__)


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
        __status__ = cudensitymatStateComputeNorm(<const Handle>handle, <const State>state, <void*>norm, <Stream>stream)
    check_status(__status__)


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
        __status__ = cudensitymatStateComputeTrace(<const Handle>handle, <const State>state, <void*>trace, <Stream>stream)
    check_status(__status__)


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
        __status__ = cudensitymatStateComputeAccumulation(<const Handle>handle, <const State>state_in, <State>state_out, <const void*>scaling_factors, <Stream>stream)
    check_status(__status__)


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
        __status__ = cudensitymatStateComputeInnerProduct(<const Handle>handle, <const State>state_left, <const State>state_right, <void*>inner_product, <Stream>stream)
    check_status(__status__)


cpdef intptr_t create_elementary_operator(intptr_t handle, int32_t num_space_modes, space_mode_extents, int sparsity, int32_t num_diagonals, diagonal_offsets, int data_type, intptr_t tensor_data, tensor_callback, tensor_gradient_callback) except? 0:
    """Creates an elementary tensor operator acting on a given number of quantum state modes (aka space modes).

    Args:
        handle (intptr_t): Library handle.
        num_space_modes (int32_t): Number of the (state) space modes acted on.
        space_mode_extents (object): Extents of the (state) space modes acted on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        sparsity (ElementaryOperatorSparsity): Tensor operator sparsity defining the storage scheme.
        num_diagonals (int32_t): For multi-diagonal tensor operator matrices, specifies the total number of non-zero diagonals (>= 1), otherwise ignored.
        diagonal_offsets (object): For multi-diagonal tensor operator matrices, these are the offsets of the non-zero diagonals (for example, the main diagonal has offset 0, the diagonal right above the main diagonal has offset +1, the diagonal right below the main diagonal has offset -1, and so on). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        data_type (int): Tensor operator data type.
        tensor_data (intptr_t): GPU-accessible pointer to the tensor operator elements storage.
        tensor_callback (object): Optional user-defined tensor callback function which can be called later to fill in the tensor elements in the provided storage, or NULL.
        tensor_gradient_callback (object): Optional user-defined tensor gradient callback function which can be called later to compute the Vector-Jacobian Product (VJP) for the tensor operator, to produce gradients with respect to the user-defined real parameters, or NULL.

    Returns:
        intptr_t: Elementary tensor operator.

    .. seealso:: `cudensitymatCreateElementaryOperator`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _diagonal_offsets_
    get_resource_ptr[int32_t](_diagonal_offsets_, diagonal_offsets, <int32_t*>NULL)
    cdef _WrappedTensorCallback _tensor_callback_ = _convert_tensor_callback(tensor_callback)
    cdef _WrappedTensorGradientCallback _tensor_gradient_callback_ = _convert_tensor_gradient_callback(tensor_gradient_callback)
    cdef ElementaryOperator elem_operator
    with nogil:
        __status__ = cudensitymatCreateElementaryOperator(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), <_ElementaryOperatorSparsity>sparsity, num_diagonals, <const int32_t*>(_diagonal_offsets_.data()), <DataType>data_type, <void*>tensor_data, _tensor_callback_, _tensor_gradient_callback_, &elem_operator)
    check_status(__status__)
    _hold_tensor_callback_reference(<intptr_t>elem_operator, tensor_callback)
    _hold_tensor_gradient_callback_reference(<intptr_t>elem_operator, tensor_gradient_callback)
    return <intptr_t>elem_operator


cpdef intptr_t create_elementary_operator_batch(intptr_t handle, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int sparsity, int32_t num_diagonals, diagonal_offsets, int data_type, intptr_t tensor_data, tensor_callback, tensor_gradient_callback) except? 0:
    """Creates a batch of elementary tensor operators acting on a given number of quantum state modes (aka space modes). This is a batched version of the ``cudensitymatCreateElementaryOperator`` API function.

    Args:
        handle (intptr_t): Library handle.
        num_space_modes (int32_t): Number of the (state) space modes acted on.
        space_mode_extents (object): Extents of the (state) space modes acted on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        batch_size (int64_t): Batch size (>= 1).
        sparsity (ElementaryOperatorSparsity): Tensor operator sparsity defining the storage scheme.
        num_diagonals (int32_t): For multi-diagonal tensor operator matrices, specifies the total number of non-zero diagonals (>= 1).
        diagonal_offsets (object): Offsets of the non-zero diagonals (for example, the main diagonal has offset 0, the diagonal right above the main diagonal has offset +1, the diagonal right below the main diagonal has offset -1, and so on). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        data_type (int): Tensor operator data type.
        tensor_data (intptr_t): GPU-accessible pointer to the tensor operator elements storage, where all elementary tensor operators within the batch are stored contiguously in memory.
        tensor_callback (object): Optional user-defined batched tensor callback function which can be called later to fill in the tensor elements in the provided batched storage, or NULL. Note that the provided batched tensor callback function is expected to fill in all tensor instances within the batch in one call.
        tensor_gradient_callback (object): Optional user-defined batched tensor gradient callback function which can be called later to compute the Vector-Jacobian Product (VJP) for the batched tensor operator, to produce gradients with respect to the batched user-defined real parameters, or NULL.

    Returns:
        intptr_t: Batched elementary tensor operator (a batch of individual elementary tensor operators stored contiguously in memory).

    .. seealso:: `cudensitymatCreateElementaryOperatorBatch`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _diagonal_offsets_
    get_resource_ptr[int32_t](_diagonal_offsets_, diagonal_offsets, <int32_t*>NULL)
    cdef _WrappedTensorCallback _tensor_callback_ = _convert_tensor_callback(tensor_callback)
    cdef _WrappedTensorGradientCallback _tensor_gradient_callback_ = _convert_tensor_gradient_callback(tensor_gradient_callback)
    cdef ElementaryOperator elem_operator
    with nogil:
        __status__ = cudensitymatCreateElementaryOperatorBatch(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), batch_size, <_ElementaryOperatorSparsity>sparsity, num_diagonals, <const int32_t*>(_diagonal_offsets_.data()), <DataType>data_type, <void*>tensor_data, _tensor_callback_, _tensor_gradient_callback_, &elem_operator)
    check_status(__status__)
    _hold_tensor_callback_reference(<intptr_t>elem_operator, tensor_callback)
    _hold_tensor_gradient_callback_reference(<intptr_t>elem_operator, tensor_gradient_callback)
    return <intptr_t>elem_operator


cpdef destroy_elementary_operator(intptr_t elem_operator):
    """Destroys an elementary tensor operator (simple or batched).

    Args:
        elem_operator (intptr_t): Elementary tensor operator.

    .. seealso:: `cudensitymatDestroyElementaryOperator`
    """
    with nogil:
        __status__ = cudensitymatDestroyElementaryOperator(<ElementaryOperator>elem_operator)
    check_status(__status__)
    _callback_holders.pop(elem_operator, None)


cpdef intptr_t create_matrix_operator_dense_local(intptr_t handle, int32_t num_space_modes, space_mode_extents, int data_type, intptr_t matrix_data, matrix_callback, matrix_gradient_callback) except? 0:
    """Creates a full matrix operator acting on all quantum state modes (aka space modes) from a dense matrix stored (replicated) locally on each process.

    Args:
        handle (intptr_t): Library handle.
        num_space_modes (int32_t): Number of the (state) space modes acted on. It must coincide with the total number of space modes in the Hilbert space.
        space_mode_extents (object): Extents of the (state) space modes acted on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        data_type (int): Matrix operator data type.
        matrix_data (intptr_t): GPU-accessible pointer to the matrix operator elements storage.
        matrix_callback (object): Optional user-defined tensor callback function which can be called later to fill in the matrix elements in the provided storage, or NULL.
        matrix_gradient_callback (object): Optional user-defined tensor gradient callback function which can be called later to compute the Vector-Jacobian Product (VJP) for the matrix operator, to produce gradients with respect to the user-defined real parameters, or NULL.

    Returns:
        intptr_t: Full matrix operator.

    .. seealso:: `cudensitymatCreateMatrixOperatorDenseLocal`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef _WrappedTensorCallback _matrix_callback_ = _convert_tensor_callback(matrix_callback)
    cdef _WrappedTensorGradientCallback _matrix_gradient_callback_ = _convert_tensor_gradient_callback(matrix_gradient_callback)
    cdef MatrixOperator matrix_operator
    with nogil:
        __status__ = cudensitymatCreateMatrixOperatorDenseLocal(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), <DataType>data_type, <void*>matrix_data, _matrix_callback_, _matrix_gradient_callback_, &matrix_operator)
    check_status(__status__)
    _hold_tensor_callback_reference(<intptr_t>matrix_operator, matrix_callback)
    _hold_tensor_gradient_callback_reference(<intptr_t>matrix_operator, matrix_gradient_callback)
    return <intptr_t>matrix_operator


cpdef intptr_t create_matrix_operator_dense_local_batch(intptr_t handle, int32_t num_space_modes, space_mode_extents, int64_t batch_size, int data_type, intptr_t matrix_data, matrix_callback, matrix_gradient_callback) except? 0:
    """Creates a batch of full matrix operators acting on all quantum state modes (aka space modes) from an array of dense matrices stored (replicated) locally on each process. This is a batched version of the ``cudensitymatCreateMatrixOperatorDenseLocal`` API function.

    Args:
        handle (intptr_t): Library handle.
        num_space_modes (int32_t): Number of the (state) space modes acted on. It must coincide with the total number of space modes in the Hilbert space.
        space_mode_extents (object): Extents of the (state) space modes acted on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        batch_size (int64_t): Batch size (>= 1).
        data_type (int): Matrix operator data type.
        matrix_data (intptr_t): GPU-accessible pointer to the matrix operator elements storage where all matrix operator instances within the batch are stored contiguously in memory.
        matrix_callback (object): Optional user-defined batched tensor callback function which can be called later to fill in the matrix elements in the provided batched storage, or NULL. Note that the provided batched tensor callback function is expected to fill in all matrix instances within the batch in one call.
        matrix_gradient_callback (object): Optional user-defined batched tensor gradient callback function which can be called later to compute the Vector-Jacobian Product (VJP) for the batched matrix operator, to produce gradients with respect to the batched user-defined real parameters, or NULL.

    Returns:
        intptr_t: Batched full matrix operator (a batch of full matrix operators).

    .. seealso:: `cudensitymatCreateMatrixOperatorDenseLocalBatch`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef _WrappedTensorCallback _matrix_callback_ = _convert_tensor_callback(matrix_callback)
    cdef _WrappedTensorGradientCallback _matrix_gradient_callback_ = _convert_tensor_gradient_callback(matrix_gradient_callback)
    cdef MatrixOperator matrix_operator
    with nogil:
        __status__ = cudensitymatCreateMatrixOperatorDenseLocalBatch(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), batch_size, <DataType>data_type, <void*>matrix_data, _matrix_callback_, _matrix_gradient_callback_, &matrix_operator)
    check_status(__status__)
    _hold_tensor_callback_reference(<intptr_t>matrix_operator, matrix_callback)
    _hold_tensor_gradient_callback_reference(<intptr_t>matrix_operator, matrix_gradient_callback)
    return <intptr_t>matrix_operator


cpdef destroy_matrix_operator(intptr_t matrix_operator):
    """Destroys a full matrix operator (simple or batched).

    Args:
        matrix_operator (intptr_t): Full matrix operator.

    .. seealso:: `cudensitymatDestroyMatrixOperator`
    """
    with nogil:
        __status__ = cudensitymatDestroyMatrixOperator(<MatrixOperator>matrix_operator)
    check_status(__status__)
    _callback_holders.pop(matrix_operator, None)


cpdef intptr_t create_matrix_product_operator(intptr_t handle, int32_t num_space_modes, space_mode_extents, int boundary_condition, bond_extents, int data_type, tensor_data, tensor_callbacks, tensor_gradient_callbacks) except? 0:
    """Creates a matrix product operator (MPO) acting on a subset of quantum state modes (aka space modes).

    Args:
        handle (intptr_t): Library handle.
        num_space_modes (int32_t): Number of the (state) space modes acted on.
        space_mode_extents (object): Extents of the (state) space modes acted on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        boundary_condition (BoundaryCondition): Boundary condition.
        bond_extents (object): Extents of the bond modes. For open boundary condition, the length of the array must be equal to the number of space modes minus one, where ``bond_extents[i]`` is the bond dimension between site ``i`` and site ``i+1``. For periodic boundary condition, the length of the array must be equal to the number of space modes. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        data_type (int): Matrix product operator data type.
        tensor_data (object): GPU-accessible pointers to the elements of each site tensor constituting the matrix product operator (``tensor_data[i]`` points to site ``i``). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        tensor_callbacks (object): Optional user-defined tensor callback functions (for each MPO tensor) which can be called later to fill in the matrix product operator elements in the provided storage, or NULL.
        tensor_gradient_callbacks (object): Optional user-defined tensor gradient callback functions (for each MPO tensor) which can be called later to compute the Vector-Jacobian Product (VJP) for the matrix product operator, to produce gradients with respect to the user-defined real parameters, or NULL.

    Returns:
        intptr_t: Matrix product operator.

    .. seealso:: `cudensitymatCreateMatrixProductOperator`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _space_mode_extents_
    get_resource_ptr[int64_t](_space_mode_extents_, space_mode_extents, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _bond_extents_
    get_resource_ptr[int64_t](_bond_extents_, bond_extents, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _tensor_data_
    get_resource_ptrs[void](_tensor_data_, tensor_data, <void*>NULL)
    cdef vector[_WrappedTensorCallback] _tensor_callbacks_vec_
    cdef _WrappedTensorCallback * _tensor_callbacks_ = NULL
    if tensor_callbacks is not None:
        for _cb_ in tensor_callbacks:
            _tensor_callbacks_vec_.push_back(_convert_tensor_callback(_cb_))
        _tensor_callbacks_ = _tensor_callbacks_vec_.data()
    cdef vector[_WrappedTensorGradientCallback] _tensor_gradient_callbacks_vec_
    cdef _WrappedTensorGradientCallback * _tensor_gradient_callbacks_ = NULL
    if tensor_gradient_callbacks is not None:
        for _cb_ in tensor_gradient_callbacks:
            _tensor_gradient_callbacks_vec_.push_back(_convert_tensor_gradient_callback(_cb_))
        _tensor_gradient_callbacks_ = _tensor_gradient_callbacks_vec_.data()
    cdef MatrixProductOperator matrix_product_operator
    with nogil:
        __status__ = cudensitymatCreateMatrixProductOperator(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), <_BoundaryCondition>boundary_condition, <const int64_t*>(_bond_extents_.data()), <DataType>data_type, <void**>(_tensor_data_.data()), <cudensitymatWrappedTensorCallback_t*>_tensor_callbacks_, <cudensitymatWrappedTensorGradientCallback_t*>_tensor_gradient_callbacks_, &matrix_product_operator)
    check_status(__status__)
    if tensor_callbacks is not None:
        for _cb_ in tensor_callbacks:
            _hold_tensor_callback_reference(<intptr_t>matrix_product_operator, _cb_)
    if tensor_gradient_callbacks is not None:
        for _cb_ in tensor_gradient_callbacks:
            _hold_tensor_gradient_callback_reference(<intptr_t>matrix_product_operator, _cb_)
    return <intptr_t>matrix_product_operator


cpdef destroy_matrix_product_operator(intptr_t matrix_product_operator):
    """Destroys a matrix product operator (MPO).

    Args:
        matrix_product_operator (intptr_t): Matrix product operator.

    .. seealso:: `cudensitymatDestroyMatrixProductOperator`
    """
    with nogil:
        __status__ = cudensitymatDestroyMatrixProductOperator(<MatrixProductOperator>matrix_product_operator)
    check_status(__status__)
    _callback_holders.pop(matrix_product_operator, None)


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
        __status__ = cudensitymatCreateOperatorTerm(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), &operator_term)
    check_status(__status__)
    return <intptr_t>operator_term


cpdef destroy_operator_term(intptr_t operator_term):
    """Destroys an operator term.

    Args:
        operator_term (intptr_t): Operator term.

    .. seealso:: `cudensitymatDestroyOperatorTerm`
    """
    with nogil:
        __status__ = cudensitymatDestroyOperatorTerm(<OperatorTerm>operator_term)
    check_status(__status__)
    _callback_holders.pop(operator_term, None)


cpdef operator_term_append_elementary_product(intptr_t handle, intptr_t operator_term, int32_t num_elem_operators, elem_operators, state_modes_acted_on, mode_action_duality, complex coefficient, coefficient_callback, coefficient_gradient_callback):
    """Appends a product of elementary tensor operators acting on quantum state modes to the operator term.

    Args:
        handle (intptr_t): Library handle.
        operator_term (intptr_t): Operator term.
        num_elem_operators (int32_t): Number of elementary tensor operators in the tensor operator product.
        elem_operators (object): Elementary tensor operators constituting the tensor operator product. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        state_modes_acted_on (object): State modes acted on by the tensor operator product. This is a concatenated list of the state modes acted on by all constituting elementary tensor operators in the same order how they appear in the elem_operators argument. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mode_action_duality (object): Duality status of each mode action, that is, whether the action applies to a ket mode of the quantum state (value zero) or a bra mode of the quantum state (positive value). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        coefficient (complex): Constant (static) complex scalar coefficient associated with the appended tensor operator product.
        coefficient_callback (object): Optional user-defined complex scalar callback function which can be called later to update the scalar coefficient associated with the tensor operator product, or NULL. The total coefficient associated with the tensor operator product is a product of the constant coefficient and the result of the scalar callback function, if defined.
        coefficient_gradient_callback (object): Optional user-defined scalar gradient callback function which can be called later to compute the gradients of the complex scalar coefficient with respect to the user-defined real parameters, or NULL.

    .. seealso:: `cudensitymatOperatorTermAppendElementaryProduct`
    """
    cdef nullable_unique_ptr[ vector[ElementaryOperator*] ] _elem_operators_
    get_resource_ptrs[ElementaryOperator](_elem_operators_, elem_operators, <ElementaryOperator*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_modes_acted_on_
    get_resource_ptr[int32_t](_state_modes_acted_on_, state_modes_acted_on, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_action_duality_
    get_resource_ptr[int32_t](_mode_action_duality_, mode_action_duality, <int32_t*>NULL)
    cdef cuDoubleComplex _coefficient_ = cuDoubleComplex(coefficient.real, coefficient.imag)
    cdef _WrappedScalarCallback _coefficient_callback_ = _convert_scalar_callback(coefficient_callback)
    cdef _WrappedScalarGradientCallback _coefficient_gradient_callback_ = _convert_scalar_gradient_callback(coefficient_gradient_callback)
    with nogil:
        __status__ = cudensitymatOperatorTermAppendElementaryProduct(<const Handle>handle, <OperatorTerm>operator_term, num_elem_operators, <const ElementaryOperator*>(_elem_operators_.data()), <const int32_t*>(_state_modes_acted_on_.data()), <const int32_t*>(_mode_action_duality_.data()), <cuDoubleComplex>_coefficient_, _coefficient_callback_, _coefficient_gradient_callback_)
    check_status(__status__)
    _hold_scalar_callback_reference(<intptr_t>operator_term, coefficient_callback)
    _hold_scalar_gradient_callback_reference(<intptr_t>operator_term, coefficient_gradient_callback)


cpdef operator_term_append_elementary_product_batch(intptr_t handle, intptr_t operator_term, int32_t num_elem_operators, elem_operators, state_modes_acted_on, mode_action_duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, coefficient_callback, coefficient_gradient_callback):
    """Appends a batch of elementary tensor operator products acting on quantum state modes to the operator term.

    Args:
        handle (intptr_t): Library handle.
        operator_term (intptr_t): Operator term.
        num_elem_operators (int32_t): Number of elementary tensor operators in the tensor operator product.
        elem_operators (object): Elementary tensor operators constituting the tensor operator product (each elementary tensor operator may or may not be batched). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        state_modes_acted_on (object): State modes acted on by the tensor operator product. This is a concatenated list of the state modes acted on by all constituting elementary tensor operators in the same order how they appear in the elem_operators argument. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mode_action_duality (object): Duality status of each mode action, that is, whether the action applies to a ket mode of the quantum state (value zero) or a bra mode of the quantum state (positive value). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        batch_size (int64_t): Batch size (>= 1).
        static_coefficients (intptr_t): GPU-accessible array of constant (static) complex scalar coefficients associated with the appended batch of elementary tensor operator products (of length ``batch_size``).
        total_coefficients (intptr_t): GPU-accessible storage for the array of total complex scalar coefficients associated with the appended batch of elementary tensor operator products (of length ``batch_size``). Each coefficient will be a product of a static coefficient and a dynamic coefficient generated by the provided scalar callback during the computation phase. If the scalar callback is not supplied here (NULL), this argument can also be set to NULL.
        coefficient_callback (object): Optional user-defined batched complex scalar callback function which can be called later to update the array of dynamic scalar coefficients associated with the defined batch of elementary tensor operator products, or NULL. The total coefficient associated with an elementary tensor operator product is a product of the constant (static) coefficient and the result of the scalar callback function, if defined.
        coefficient_gradient_callback (object): Optional user-defined batched scalar gradient callback function which can be called later to compute the gradients of the batched complex scalar coefficients with respect to the batched user-defined real parameters, or NULL.

    .. seealso:: `cudensitymatOperatorTermAppendElementaryProductBatch`
    """
    cdef nullable_unique_ptr[ vector[ElementaryOperator*] ] _elem_operators_
    get_resource_ptrs[ElementaryOperator](_elem_operators_, elem_operators, <ElementaryOperator*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_modes_acted_on_
    get_resource_ptr[int32_t](_state_modes_acted_on_, state_modes_acted_on, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_action_duality_
    get_resource_ptr[int32_t](_mode_action_duality_, mode_action_duality, <int32_t*>NULL)
    cdef _WrappedScalarCallback _coefficient_callback_ = _convert_scalar_callback(coefficient_callback)
    cdef _WrappedScalarGradientCallback _coefficient_gradient_callback_ = _convert_scalar_gradient_callback(coefficient_gradient_callback)
    with nogil:
        __status__ = cudensitymatOperatorTermAppendElementaryProductBatch(<const Handle>handle, <OperatorTerm>operator_term, num_elem_operators, <const ElementaryOperator*>(_elem_operators_.data()), <const int32_t*>(_state_modes_acted_on_.data()), <const int32_t*>(_mode_action_duality_.data()), batch_size, <const cuDoubleComplex*>static_coefficients, <cuDoubleComplex*>total_coefficients, _coefficient_callback_, _coefficient_gradient_callback_)
    check_status(__status__)
    _hold_scalar_callback_reference(<intptr_t>operator_term, coefficient_callback)
    _hold_scalar_gradient_callback_reference(<intptr_t>operator_term, coefficient_gradient_callback)


cpdef operator_term_append_matrix_product(intptr_t handle, intptr_t operator_term, int32_t num_matrix_operators, matrix_operators, matrix_conjugation, action_duality, complex coefficient, coefficient_callback, coefficient_gradient_callback):
    """Appends a product of full matrix operators, each acting on all quantum state modes, to the operator term.

    Args:
        handle (intptr_t): Library handle.
        operator_term (intptr_t): Operator term.
        num_matrix_operators (int32_t): Number of full matrix operators in the matrix operator product.
        matrix_operators (object): Full matrix operators constituting the matrix operator product. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        matrix_conjugation (object): Hermitean conjugation status of each matrix in the matrix operator product (zero means normal, positive integer means conjugate-transposed). For real matrices, hermitean conjugation reduces to a mere matrix transpose since there is no complex conjugation involved. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        action_duality (object): Duality status of each matrix operator action, that is, whether it acts on all ket modes of the quantum state (value zero) or on all bra modes of the quantum state (positive integer value). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        coefficient (complex): Constant (static) complex scalar coefficient associated with the matrix operator product.
        coefficient_callback (object): Optional user-defined complex scalar callback function which can be called later to update the scalar coefficient associated with the matrix operator product, or NULL. The total coefficient associated with the matrix operator product is a product of the constant coefficient and the result of the scalar callback function, if defined.
        coefficient_gradient_callback (object): Optional user-defined scalar gradient callback function which can be called later to compute the gradients of the complex scalar coefficient with respect to the user-defined real parameters, or NULL.

    .. seealso:: `cudensitymatOperatorTermAppendMatrixProduct`
    """
    cdef nullable_unique_ptr[ vector[MatrixOperator*] ] _matrix_operators_
    get_resource_ptrs[MatrixOperator](_matrix_operators_, matrix_operators, <MatrixOperator*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _matrix_conjugation_
    get_resource_ptr[int32_t](_matrix_conjugation_, matrix_conjugation, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _action_duality_
    get_resource_ptr[int32_t](_action_duality_, action_duality, <int32_t*>NULL)
    cdef cuDoubleComplex _coefficient_ = cuDoubleComplex(coefficient.real, coefficient.imag)
    cdef _WrappedScalarCallback _coefficient_callback_ = _convert_scalar_callback(coefficient_callback)
    cdef _WrappedScalarGradientCallback _coefficient_gradient_callback_ = _convert_scalar_gradient_callback(coefficient_gradient_callback)
    with nogil:
        __status__ = cudensitymatOperatorTermAppendMatrixProduct(<const Handle>handle, <OperatorTerm>operator_term, num_matrix_operators, <const MatrixOperator*>(_matrix_operators_.data()), <const int32_t*>(_matrix_conjugation_.data()), <const int32_t*>(_action_duality_.data()), <cuDoubleComplex>_coefficient_, _coefficient_callback_, _coefficient_gradient_callback_)
    check_status(__status__)
    _hold_scalar_callback_reference(<intptr_t>operator_term, coefficient_callback)
    _hold_scalar_gradient_callback_reference(<intptr_t>operator_term, coefficient_gradient_callback)


cpdef operator_term_append_matrix_product_batch(intptr_t handle, intptr_t operator_term, int32_t num_matrix_operators, matrix_operators, matrix_conjugation, action_duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, coefficient_callback, coefficient_gradient_callback):
    """Appends a batch of full matrix operators to the operator term, each full matrix operator acting on all quantum state modes. This is a batched version of the ``cudensitymatOperatorTermAppendMatrixProduct`` API function.

    Args:
        handle (intptr_t): Library handle.
        operator_term (intptr_t): Operator term.
        num_matrix_operators (int32_t): Number of full matrix operators in the matrix operator product.
        matrix_operators (object): Full matrix operators constituting the matrix operator product (each full matrix operator may or may not be batched). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        matrix_conjugation (object): Hermitean conjugation status of each matrix in the matrix operator product (zero means normal, positive integer means conjugate-transposed). For real matrices, hermitean conjugation reduces to a mere matrix transpose since there is no complex conjugation involved. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        action_duality (object): Duality status of each matrix operator action, that is, whether it acts on all ket modes of the quantum state (value zero) or on all bra modes of the quantum state (positive integer value). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        batch_size (int64_t): Batch size (>= 1).
        static_coefficients (intptr_t): GPU-accessible array of constant (static) complex scalar coefficients associated with the appended batch of full matrix operator products (of length ``batch_size``).
        total_coefficients (intptr_t): GPU-accessible storage for the array of total complex scalar coefficients associated with the appended batch of full matrix operator products (of length ``batch_size``). Each coefficient will be a product of a static coefficient and a dynamic coefficient generated by the provided scalar callback during the computation phase. If the scalar callback is not supplied here (NULL), this argument can also be set to NULL.
        coefficient_callback (object): Optional user-defined batched complex scalar callback function which can be called later to update the array of dynamic scalar coefficients associated with the defined batch of full matrix operator products, or NULL. The total coefficient associated with an elementary tensor operator product is a product of the constant (static) coefficient and the result of the scalar callback function, if defined.
        coefficient_gradient_callback (object): Optional user-defined batched scalar gradient callback function which can be called later to compute the gradients of the batched complex scalar coefficients with respect to the batched user-defined real parameters, or NULL.

    .. seealso:: `cudensitymatOperatorTermAppendMatrixProductBatch`
    """
    cdef nullable_unique_ptr[ vector[MatrixOperator*] ] _matrix_operators_
    get_resource_ptrs[MatrixOperator](_matrix_operators_, matrix_operators, <MatrixOperator*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _matrix_conjugation_
    get_resource_ptr[int32_t](_matrix_conjugation_, matrix_conjugation, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _action_duality_
    get_resource_ptr[int32_t](_action_duality_, action_duality, <int32_t*>NULL)
    cdef _WrappedScalarCallback _coefficient_callback_ = _convert_scalar_callback(coefficient_callback)
    cdef _WrappedScalarGradientCallback _coefficient_gradient_callback_ = _convert_scalar_gradient_callback(coefficient_gradient_callback)
    with nogil:
        __status__ = cudensitymatOperatorTermAppendMatrixProductBatch(<const Handle>handle, <OperatorTerm>operator_term, num_matrix_operators, <const MatrixOperator*>(_matrix_operators_.data()), <const int32_t*>(_matrix_conjugation_.data()), <const int32_t*>(_action_duality_.data()), batch_size, <const cuDoubleComplex*>static_coefficients, <cuDoubleComplex*>total_coefficients, _coefficient_callback_, _coefficient_gradient_callback_)
    check_status(__status__)
    _hold_scalar_callback_reference(<intptr_t>operator_term, coefficient_callback)
    _hold_scalar_gradient_callback_reference(<intptr_t>operator_term, coefficient_gradient_callback)


cpdef operator_term_append_mpo_product(intptr_t handle, intptr_t operator_term, int32_t num_mpo_operators, mpo_operators, mpo_conjugation, state_modes_acted_on, mode_action_duality, complex coefficient, coefficient_callback, coefficient_gradient_callback):
    """Appends a product of matrix product operators (MPOs) to the operator term.

    Args:
        handle (intptr_t): Library handle.
        operator_term (intptr_t): Operator term.
        num_mpo_operators (int32_t): Number of MPO operators in the MPO product.
        mpo_operators (object): MPO operators constituting the MPO product. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        mpo_conjugation (object): Hermitean conjugation status of each MPO in the MPO product (zero means normal, positive integer means conjugate-transposed). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        state_modes_acted_on (object): State modes acted on by the product of matrix product operators. This is a concatenated list of the state modes acted on by all constituting MPO operators in the same order how they appear in the mpo_operators argument. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mode_action_duality (object): Duality status of each mode action, that is, whether the action applies to a ket mode of the quantum state (value zero) or a bra mode of the quantum state (positive value). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        coefficient (complex): Constant (static) complex scalar coefficient associated with the MPO product.
        coefficient_callback (object): Optional user-defined complex scalar callback function which can be called later to update the scalar coefficient associated with the MPO product, or NULL. The total coefficient associated with the MPO product is a product of the constant coefficient and the result of the scalar callback function, if defined.
        coefficient_gradient_callback (object): Optional user-defined scalar gradient callback function function which can be called later to compute the gradients of the complex scalar coefficient with respect to the user-defined real parameters, or NULL.

    .. seealso:: `cudensitymatOperatorTermAppendMPOProduct`
    """
    cdef nullable_unique_ptr[ vector[MatrixProductOperator*] ] _mpo_operators_
    get_resource_ptrs[MatrixProductOperator](_mpo_operators_, mpo_operators, <MatrixProductOperator*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mpo_conjugation_
    get_resource_ptr[int32_t](_mpo_conjugation_, mpo_conjugation, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_modes_acted_on_
    get_resource_ptr[int32_t](_state_modes_acted_on_, state_modes_acted_on, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_action_duality_
    get_resource_ptr[int32_t](_mode_action_duality_, mode_action_duality, <int32_t*>NULL)
    cdef cuDoubleComplex _coefficient_ = cuDoubleComplex(coefficient.real, coefficient.imag)
    cdef _WrappedScalarCallback _coefficient_callback_ = _convert_scalar_callback(coefficient_callback)
    cdef _WrappedScalarGradientCallback _coefficient_gradient_callback_ = _convert_scalar_gradient_callback(coefficient_gradient_callback)
    with nogil:
        __status__ = cudensitymatOperatorTermAppendMPOProduct(<const Handle>handle, <OperatorTerm>operator_term, num_mpo_operators, <const MatrixProductOperator*>(_mpo_operators_.data()), <const int32_t*>(_mpo_conjugation_.data()), <const int32_t*>(_state_modes_acted_on_.data()), <const int32_t*>(_mode_action_duality_.data()), <cuDoubleComplex>_coefficient_, _coefficient_callback_, _coefficient_gradient_callback_)
    check_status(__status__)
    _hold_scalar_callback_reference(<intptr_t>operator_term, coefficient_callback)
    _hold_scalar_gradient_callback_reference(<intptr_t>operator_term, coefficient_gradient_callback)


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
        __status__ = cudensitymatCreateOperator(<const Handle>handle, num_space_modes, <const int64_t*>(_space_mode_extents_.data()), &superoperator)
    check_status(__status__)
    return <intptr_t>superoperator


cpdef destroy_operator(intptr_t superoperator):
    """Destroys an operator.

    Args:
        superoperator (intptr_t): Operator.

    .. seealso:: `cudensitymatDestroyOperator`
    """
    with nogil:
        __status__ = cudensitymatDestroyOperator(<Operator>superoperator)
    check_status(__status__)
    _callback_holders.pop(superoperator, None)


cpdef operator_append_term(intptr_t handle, intptr_t superoperator, intptr_t operator_term, int32_t duality, complex coefficient, coefficient_callback, coefficient_gradient_callback):
    """Appends an operator term to the operator.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        operator_term (intptr_t): Operator term.
        duality (int32_t): Duality status of the operator term action as a whole. If not zero, the duality status of each mode action inside the operator term will be flipped, that is, action from the left will be replaced by action from the right, and vice versa.
        coefficient (complex): Constant (static) complex scalar coefficient associated with the operator term.
        coefficient_callback (object): Optional user-defined complex scalar callback function which can be called later to update the scalar coefficient associated with the operator term, or NULL. The total coefficient associated with the operator term is a product of the constant coefficient and the result of the scalar callback function, if defined.
        coefficient_gradient_callback (object): Optional user-defined scalar gradient callback function which can be called later to compute the gradients of the complex scalar coefficient with respect to the user-defined real parameters, or NULL.

    .. seealso:: `cudensitymatOperatorAppendTerm`
    """
    cdef cuDoubleComplex _coefficient_ = cuDoubleComplex(coefficient.real, coefficient.imag)
    cdef _WrappedScalarCallback _coefficient_callback_ = _convert_scalar_callback(coefficient_callback)
    cdef _WrappedScalarGradientCallback _coefficient_gradient_callback_ = _convert_scalar_gradient_callback(coefficient_gradient_callback)
    with nogil:
        __status__ = cudensitymatOperatorAppendTerm(<const Handle>handle, <Operator>superoperator, <OperatorTerm>operator_term, duality, <cuDoubleComplex>_coefficient_, _coefficient_callback_, _coefficient_gradient_callback_)
    check_status(__status__)
    _hold_scalar_callback_reference(<intptr_t>superoperator, coefficient_callback)
    _hold_scalar_gradient_callback_reference(<intptr_t>superoperator, coefficient_gradient_callback)


cpdef operator_append_term_batch(intptr_t handle, intptr_t superoperator, intptr_t operator_term, int32_t duality, int64_t batch_size, intptr_t static_coefficients, intptr_t total_coefficients, coefficient_callback, coefficient_gradient_callback):
    """Appends a batch of operator terms to the operator.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        operator_term (intptr_t): Operator term.
        duality (int32_t): Duality status of the operator term action as a whole. If not zero, the duality status of each mode action inside the operator term will be flipped, that is, action from the left will be replaced by action from the right, and vice versa.
        batch_size (int64_t): Batch size (>= 1).
        static_coefficients (intptr_t): GPU-accessible array of constant (static) complex scalar coefficients associated with the appended batch of operator terms (of length ``batch_size``).
        total_coefficients (intptr_t): GPU-accessible storage for the array of total complex scalar coefficients associated with the appended batch of operator terms (of length ``batch_size``). Each coefficient will be a product of a static coefficient and a dynamic coefficient generated by the coefficient callback during the computation phase. If the scalar callback is not supplied here (NULL), this argument can also be set to NULL.
        coefficient_callback (object): Optional user-defined batched complex scalar callback function which can be called later to update the array of scalar coefficients associated with the defined batch of operator terms, or NULL. The total coefficient associated with an operator term is a product of the constant (static) coefficient and the result of the scalar callback function, if defined.
        coefficient_gradient_callback (object): Optional user-defined batched scalar gradient callback function which can be called later to compute the gradients of the batched complex scalar coefficients with respect to the batched user-defined real parameters, or NULL.

    .. seealso:: `cudensitymatOperatorAppendTermBatch`
    """
    cdef _WrappedScalarCallback _coefficient_callback_ = _convert_scalar_callback(coefficient_callback)
    cdef _WrappedScalarGradientCallback _coefficient_gradient_callback_ = _convert_scalar_gradient_callback(coefficient_gradient_callback)
    with nogil:
        __status__ = cudensitymatOperatorAppendTermBatch(<const Handle>handle, <Operator>superoperator, <OperatorTerm>operator_term, duality, batch_size, <const cuDoubleComplex*>static_coefficients, <cuDoubleComplex*>total_coefficients, _coefficient_callback_, _coefficient_gradient_callback_)
    check_status(__status__)
    _hold_scalar_callback_reference(<intptr_t>superoperator, coefficient_callback)
    _hold_scalar_gradient_callback_reference(<intptr_t>superoperator, coefficient_gradient_callback)


cpdef attach_batched_coefficients(intptr_t handle, intptr_t superoperator, int32_t num_operator_term_batched_coeffs, operator_term_batched_coeffs_tmp, operator_term_batched_coeffs, int32_t num_operator_product_batched_coeffs, operator_product_batched_coeffs_tmp, operator_product_batched_coeffs):
    """Attaches batched coefficients to the operator's term and product coefficients.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        num_operator_term_batched_coeffs (int32_t): Number of batched coefficients in the operator term.
        operator_term_batched_coeffs_tmp (object): Temporary buffer for the batched coefficients in the operator term. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        operator_term_batched_coeffs (object): Actual buffer for the batched coefficients in the operator term. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        num_operator_product_batched_coeffs (int32_t): Number of batched coefficients in the operator product.
        operator_product_batched_coeffs_tmp (object): Temporary buffer for the batched coefficients in the operator product. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        operator_product_batched_coeffs (object): Actual buffer for the batched coefficients in the operator product. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).


    .. seealso:: `cudensitymatAttachBatchedCoefficients`
    """
    cdef nullable_unique_ptr[ vector[void*] ] _operator_term_batched_coeffs_tmp_
    get_resource_ptrs[void](_operator_term_batched_coeffs_tmp_, operator_term_batched_coeffs_tmp, <void*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _operator_term_batched_coeffs_
    get_resource_ptrs[void](_operator_term_batched_coeffs_, operator_term_batched_coeffs, <void*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _operator_product_batched_coeffs_tmp_
    get_resource_ptrs[void](_operator_product_batched_coeffs_tmp_, operator_product_batched_coeffs_tmp, <void*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _operator_product_batched_coeffs_
    get_resource_ptrs[void](_operator_product_batched_coeffs_, operator_product_batched_coeffs, <void*>NULL)
    with nogil:
        __status__ = cudensitymatAttachBatchedCoefficients(<const Handle>handle, <Operator>superoperator, num_operator_term_batched_coeffs, <void**>(_operator_term_batched_coeffs_tmp_.data()), <void**>(_operator_term_batched_coeffs_.data()), num_operator_product_batched_coeffs, <void**>(_operator_product_batched_coeffs_tmp_.data()), <void**>(_operator_product_batched_coeffs_.data()))
    check_status(__status__)


cpdef operator_configure_action(intptr_t handle, intptr_t superoperator, intptr_t state_in, intptr_t state_out, intptr_t attribute_value, size_t attribute_size):
    """Configures the operator action on a quantum state.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        state_in (intptr_t): Representative input quantum state on which the operator is supposed to act. The actual quantum state acted on during computation may be different, but it has to be of the same shape, kind, and factorization structure (topology, bond dimensions, etc).
        state_out (intptr_t): Representative output quantum state produced by the action of the operator on the input quantum state. The actual quantum state acted on during computation may be different, but it has to be of the same shape, kind, and factorization structure (topology, bond dimensions, etc).
        attribute_value (intptr_t): Configuration attribute.
        attribute_size (size_t): Pointer to the configuration attribute value (type-erased).

    .. seealso:: `cudensitymatOperatorConfigureAction`
    """
    with nogil:
        __status__ = cudensitymatOperatorConfigureAction(<const Handle>handle, <Operator>superoperator, <const State>state_in, <const State>state_out, <const void*>attribute_value, attribute_size)
    check_status(__status__)


cpdef operator_prepare_action(intptr_t handle, intptr_t superoperator, intptr_t state_in, intptr_t state_out, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream):
    """Prepares the operator for an action on a quantum state.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        state_in (intptr_t): Representative input quantum state on which the operator is supposed to act. The actual quantum state acted on during computation may be different, but it has to be of the same shape, kind, and factorization structure (topology, bond dimensions, etc).
        state_out (intptr_t): Representative output quantum state produced by the action of the operator on the input quantum state. The actual quantum state acted on during computation may be different, but it has to be of the same shape, kind, and factorization structure (topology, bond dimensions, etc).
        compute_type (ComputeType): Desired compute type.
        workspace_size_limit (size_t): Workspace buffer size limit (bytes).
        workspace (intptr_t): Empty workspace descriptor on entrance. The workspace size required for the computation will be set on exit.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatOperatorPrepareAction`
    """
    with nogil:
        __status__ = cudensitymatOperatorPrepareAction(<const Handle>handle, <Operator>superoperator, <const State>state_in, <const State>state_out, <_ComputeType>compute_type, workspace_size_limit, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef operator_compute_action(intptr_t handle, intptr_t superoperator, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state_in, intptr_t state_out, intptr_t workspace, intptr_t stream):
    """Computes the action of the operator on a given input quantum state, accumulating the result in the output quantum state (accumulative action).

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        time (double): Time value.
        batch_size (int64_t): Batch size (>=1).
        num_params (int32_t): Number of variable parameters defined by the user.
        params (intptr_t): GPU-accessible pointer to an F-order 2d-array of user-defined real parameter values: params[num_params, batch_size].
        state_in (intptr_t): Input quantum state (or a batch of input quantum states).
        state_out (intptr_t): Updated resulting quantum state which accumulates operator action on the input quantum state.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatOperatorComputeAction`
    """
    with nogil:
        __status__ = cudensitymatOperatorComputeAction(<const Handle>handle, <Operator>superoperator, time, batch_size, num_params, <const double*>params, <const State>state_in, <State>state_out, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef operator_prepare_action_backward_diff(intptr_t handle, intptr_t superoperator, intptr_t state_in, intptr_t state_out_adj, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream):
    """Prepares backward differentiation of the operator action on a quantum state.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        state_in (intptr_t): Representative input quantum state on which the operator is supposed to act. The actual quantum state acted on during computation may be different, but it has to be of the same shape, kind, and factorization structure (topology, bond dimensions, etc).
        state_out_adj (intptr_t): Representative adjoint of the output quantum state produced by the action of the operator on the input quantum state. The actual output quantum state acted on during computation may be different, but it has to be of the same shape, kind, and factorization structure (topology, bond dimensions, etc).
        compute_type (ComputeType): Desired compute type.
        workspace_size_limit (size_t): Workspace buffer size limit (bytes).
        workspace (intptr_t): Empty workspace descriptor on entrance. The workspace size required for the computation will be set on exit.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatOperatorPrepareActionBackwardDiff`
    """
    with nogil:
        __status__ = cudensitymatOperatorPrepareActionBackwardDiff(<const Handle>handle, <Operator>superoperator, <const State>state_in, <const State>state_out_adj, <_ComputeType>compute_type, workspace_size_limit, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef operator_compute_action_backward_diff(intptr_t handle, intptr_t superoperator, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state_in, intptr_t state_out_adj, intptr_t state_in_adj, intptr_t params_grad, intptr_t workspace, intptr_t stream):
    """Computes backward differentiation of the operator action on a given quantum state.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        time (double): Time value.
        batch_size (int64_t): Batch size (>=1).
        num_params (int32_t): Number of variable real parameters defined by the user.
        params (intptr_t): GPU-accessible pointer to an F-order 2d-array of user-defined real parameter values: params[num_params, batch_size].
        state_in (intptr_t): Input quantum state (or a batch).
        state_out_adj (intptr_t): Adjoint of the output quantum state (or a batch).
        state_in_adj (intptr_t): Adjoint of the input quantum state (or a batch). Note that this array will not be zeroed out on entrance, it will be accumulated into.
        params_grad (intptr_t): GPU-accessible pointer where the partial derivatives with respect to the user-defined real parameters will be accumulated (same shape as params). Note that this array will not be zeroed out on entrance, it will be accumulated into.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatOperatorComputeActionBackwardDiff`
    """
    with nogil:
        __status__ = cudensitymatOperatorComputeActionBackwardDiff(<const Handle>handle, <Operator>superoperator, time, batch_size, num_params, <const double*>params, <const State>state_in, <const State>state_out_adj, <State>state_in_adj, <double*>params_grad, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


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
        __status__ = cudensitymatCreateOperatorAction(<const Handle>handle, num_operators, <Operator*>(_operators_.data()), &operator_action)
    check_status(__status__)
    return <intptr_t>operator_action


cpdef destroy_operator_action(intptr_t operator_action):
    """Destroys the operator action descriptor.

    Args:
        operator_action (intptr_t): Operator action.

    .. seealso:: `cudensitymatDestroyOperatorAction`
    """
    with nogil:
        __status__ = cudensitymatDestroyOperatorAction(<OperatorAction>operator_action)
    check_status(__status__)


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
        __status__ = cudensitymatOperatorActionPrepare(<const Handle>handle, <OperatorAction>operator_action, <const State*>(_state_in_.data()), <const State>state_out, <_ComputeType>compute_type, workspace_size_limit, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef operator_action_compute(intptr_t handle, intptr_t operator_action, double time, int64_t batch_size, int32_t num_params, intptr_t params, state_in, intptr_t state_out, intptr_t workspace, intptr_t stream):
    """Executes the action of one or more operators constituting the aggreggate operator(s) action on the same number of input quantum states, accumulating the results into a single output quantum state.

    Args:
        handle (intptr_t): Library handle.
        operator_action (intptr_t): Operator(s) action.
        time (double): Time value.
        batch_size (int64_t): Batch size (>=1).
        num_params (int32_t): Number of variable parameters defined by the user.
        params (intptr_t): GPU-accessible pointer to an F-order 2d-array of user-defined real parameter values: params[num_params, batch_size].
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
        __status__ = cudensitymatOperatorActionCompute(<const Handle>handle, <OperatorAction>operator_action, time, batch_size, num_params, <const double*>params, <const State*>(_state_in_.data()), <State>state_out, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef intptr_t create_expectation(intptr_t handle, intptr_t superoperator) except? 0:
    """Creates the operator expectation value computation object.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.

    Returns:
        intptr_t: Expectation value computation object.

    .. seealso:: `cudensitymatCreateExpectation`
    """
    cdef Expectation expectation
    with nogil:
        __status__ = cudensitymatCreateExpectation(<const Handle>handle, <Operator>superoperator, &expectation)
    check_status(__status__)
    return <intptr_t>expectation


cpdef destroy_expectation(intptr_t expectation):
    """Destroys an expectation value computation object.

    Args:
        expectation (intptr_t): Expectation value computation object.

    .. seealso:: `cudensitymatDestroyExpectation`
    """
    with nogil:
        __status__ = cudensitymatDestroyExpectation(<Expectation>expectation)
    check_status(__status__)


cpdef expectation_prepare(intptr_t handle, intptr_t expectation, intptr_t state, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream):
    """Prepares the expectation value object for computation.

    Args:
        handle (intptr_t): Library handle.
        expectation (intptr_t): Expectation value object.
        state (intptr_t): Representative quantum state (or a batch of quantum states).
        compute_type (ComputeType): Desired compute type.
        workspace_size_limit (size_t): Workspace buffer size limit (bytes).
        workspace (intptr_t): Empty workspace descriptor on entrance. The workspace size required for the computation will be set on exit.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatExpectationPrepare`
    """
    with nogil:
        __status__ = cudensitymatExpectationPrepare(<const Handle>handle, <Expectation>expectation, <const State>state, <_ComputeType>compute_type, workspace_size_limit, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef expectation_compute(intptr_t handle, intptr_t expectation, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state, intptr_t expectation_value, intptr_t workspace, intptr_t stream):
    """Computes the operator expectation value(s) with respect to the given quantum state(s).

    Args:
        handle (intptr_t): Library handle.
        expectation (intptr_t): Expectation value object.
        time (double): Specified time.
        batch_size (int64_t): Batch size (>=1).
        num_params (int32_t): Number of variable parameters defined by the user.
        params (intptr_t): GPU-accessible pointer to an F-order 2d-array of user-defined real parameter values: params[num_params, batch_size].
        state (intptr_t): Quantum state (or a batch of quantum states).
        expectation_value (intptr_t): Pointer to the expectation value(s) vector storage in GPU-accessible RAM of the same data type as used by the state and operator.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatExpectationCompute`
    """
    with nogil:
        __status__ = cudensitymatExpectationCompute(<const Handle>handle, <Expectation>expectation, time, batch_size, num_params, <const double*>params, <const State>state, <void*>expectation_value, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef intptr_t create_operator_spectrum(intptr_t handle, intptr_t superoperator, int32_t is_hermitian, int spectrum_kind) except? 0:
    """Creates the eigen-spectrum computation object for a given operator.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator (cannot be batched).
        is_hermitian (int32_t): Specifies whether the operator is Hermitian (!=0) or not (0).
        spectrum_kind (OperatorSpectrumKind): Requested kind of the eigen-spectrum computation.

    Returns:
        intptr_t: Eigen-spectrum computation object.

    .. seealso:: `cudensitymatCreateOperatorSpectrum`
    """
    cdef OperatorSpectrum spectrum
    with nogil:
        __status__ = cudensitymatCreateOperatorSpectrum(<const Handle>handle, <const Operator>superoperator, is_hermitian, <_OperatorSpectrumKind>spectrum_kind, &spectrum)
    check_status(__status__)
    return <intptr_t>spectrum


cpdef destroy_operator_spectrum(intptr_t spectrum):
    """Destroys an eigen-spectrum computation object.

    Args:
        spectrum (intptr_t): Eigen-spectrum computation object.

    .. seealso:: `cudensitymatDestroyOperatorSpectrum`
    """
    with nogil:
        __status__ = cudensitymatDestroyOperatorSpectrum(<OperatorSpectrum>spectrum)
    check_status(__status__)


######################### Python specific utility #########################

cdef dict operator_spectrum_config_sizes = {
    CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_EXPANSION: _numpy.int32,
    CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_RESTARTS: _numpy.int32,
    CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MIN_BLOCK_SIZE: _numpy.int32,
}

cpdef get_operator_spectrum_config_dtype(int attr):
    """Get the Python data type of the corresponding OperatorSpectrumConfig attribute.

    Args:
        attr (OperatorSpectrumConfig): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`operator_spectrum_configure`.
    """
    return operator_spectrum_config_sizes[attr]

###########################################################################


cpdef operator_spectrum_configure(intptr_t handle, intptr_t spectrum, int attribute, intptr_t attribute_value, size_t attribute_value_size):
    """Configures the eigen-spectrum computation object.

    Args:
        handle (intptr_t): Library handle.
        spectrum (intptr_t): Eigen-spectrum computation object.
        attribute (OperatorSpectrumConfig): Attribute to configure.
        attribute_value (intptr_t): CPU-accessible pointer to the attribute value.
        attribute_value_size (size_t): Size of the attribute value in bytes.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_operator_spectrum_config_dtype`.

    .. seealso:: `cudensitymatOperatorSpectrumConfigure`
    """
    with nogil:
        __status__ = cudensitymatOperatorSpectrumConfigure(<const Handle>handle, <OperatorSpectrum>spectrum, <_OperatorSpectrumConfig>attribute, <const void*>attribute_value, attribute_value_size)
    check_status(__status__)


cpdef operator_spectrum_prepare(intptr_t handle, intptr_t spectrum, int32_t max_eigen_states, intptr_t state, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream):
    """Prepares the eigen-spectrum object for computation.

    Args:
        handle (intptr_t): Library handle.
        spectrum (intptr_t): Eigen-spectrum computation object.
        max_eigen_states (int32_t): Maximum number of eigen-pairs to compute.
        state (intptr_t): Representative quantum state (cannot be batched).
        compute_type (ComputeType): Desired compute type.
        workspace_size_limit (size_t): Workspace buffer size limit (bytes).
        workspace (intptr_t): Empty workspace descriptor on entrance. The workspace buffer sizes required for the computation will be set on return.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatOperatorSpectrumPrepare`
    """
    with nogil:
        __status__ = cudensitymatOperatorSpectrumPrepare(<const Handle>handle, <OperatorSpectrum>spectrum, max_eigen_states, <const State>state, <_ComputeType>compute_type, workspace_size_limit, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef operator_spectrum_compute(intptr_t handle, intptr_t spectrum, double time, int64_t batch_size, int32_t num_params, intptr_t params, int32_t num_eigen_states, eigenstates, intptr_t eigenvalues, intptr_t tolerances, intptr_t workspace, intptr_t stream):
    """Computes the eigen-spectrum of an operator.

    Args:
        handle (intptr_t): Library handle.
        spectrum (intptr_t): Eigen-spectrum computation object.
        time (double): Specified time.
        batch_size (int64_t): Batch size (==1).
        num_params (int32_t): Number of variable parameters defined by the user.
        params (intptr_t): GPU-accessible pointer to an F-order 2d-array of user-defined real parameter values: params[num_params, batch_size].
        num_eigen_states (int32_t): Actual number of eigenstates to compute, which must not exceed the value of the ``maxEigenStates`` parameter provided during the preparation of the eigen-spectrum computation object.
        eigenstates (object): Quantum eigenstates (cannot be batched). The initial values of the provided quantum states will be used as the initial guesses for the first Krylov subspace block (if the block size is smaller than the number of requested eigenstates, only the leading quantum states will be used). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        eigenvalues (intptr_t): Pointer to the eigenvalues storage (F-order array of shape [num_eigen_states, batch_size]) in GPU-accessible RAM (same data type as used by the quantum state and operator).
        tolerances (intptr_t): Pointer to an F-order array of shape [num_eigen_states, batch_size] in CPU-accessible RAM. The initial values represent the desirable convergence tolerances for all eigen-states. The returned values represent the actually achieved residual norms for all eigen-states.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatOperatorSpectrumCompute`
    """
    cdef nullable_unique_ptr[ vector[State*] ] _eigenstates_
    get_resource_ptrs[State](_eigenstates_, eigenstates, <State*>NULL)
    with nogil:
        __status__ = cudensitymatOperatorSpectrumCompute(<const Handle>handle, <OperatorSpectrum>spectrum, time, batch_size, num_params, <const double*>params, num_eigen_states, <State*>(_eigenstates_.data()), <void*>eigenvalues, <double*>tolerances, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef intptr_t create_time_propagation_scope_split_tdvp_config(intptr_t handle) except? 0:
    """Creates a TDVP configuration object with default settings.

    Args:
        handle (intptr_t): Library handle.

    Returns:
        intptr_t: TDVP configuration object.

    .. seealso:: `cudensitymatCreateTimePropagationScopeSplitTDVPConfig`
    """
    cdef TimePropagationScopeSplitTDVPConfig config
    with nogil:
        __status__ = cudensitymatCreateTimePropagationScopeSplitTDVPConfig(<const Handle>handle, &config)
    check_status(__status__)
    return <intptr_t>config


cpdef destroy_time_propagation_scope_split_tdvp_config(intptr_t config):
    """Destroys a TDVP configuration object.

    Args:
        config (intptr_t): TDVP configuration object.

    .. seealso:: `cudensitymatDestroyTimePropagationScopeSplitTDVPConfig`
    """
    with nogil:
        __status__ = cudensitymatDestroyTimePropagationScopeSplitTDVPConfig(<TimePropagationScopeSplitTDVPConfig>config)
    check_status(__status__)


######################### Python specific utility #########################

cdef dict time_propagation_scope_split_tdvp_config_attribute_sizes = {
    CUDENSITYMAT_PROPAGATION_SPLIT_SCOPE_TDVP_ORDER: _numpy.int32,
}

cpdef get_time_propagation_scope_split_tdvp_config_attribute_dtype(int attr):
    """Get the Python data type of the corresponding TimePropagationScopeSplitTDVPConfigAttribute attribute.

    Args:
        attr (TimePropagationScopeSplitTDVPConfigAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`time_propagation_scope_split_tdvp_config_get_attribute`, :func:`time_propagation_scope_split_tdvp_config_set_attribute`.
    """
    return time_propagation_scope_split_tdvp_config_attribute_sizes[attr]

###########################################################################


cpdef time_propagation_scope_split_tdvp_config_set_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Sets an attribute of the TDVP configuration.

    Args:
        handle (intptr_t): Library handle.
        config (intptr_t): TDVP configuration object.
        attribute (TimePropagationScopeSplitTDVPConfigAttribute): Attribute to set.
        attribute_value (intptr_t): Pointer to the attribute value.
        attribute_size (size_t): Size of the attribute value in bytes.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_time_propagation_scope_split_tdvp_config_attribute_dtype`.

    .. seealso:: `cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute`
    """
    with nogil:
        __status__ = cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute(<const Handle>handle, <TimePropagationScopeSplitTDVPConfig>config, <_TimePropagationScopeSplitTDVPConfigAttribute>attribute, <const void*>attribute_value, attribute_size)
    check_status(__status__)


cpdef time_propagation_scope_split_tdvp_config_get_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Gets an attribute of the TDVP configuration.

    Args:
        handle (intptr_t): Library handle.
        config (intptr_t): TDVP configuration object.
        attribute (TimePropagationScopeSplitTDVPConfigAttribute): Attribute to get.
        attribute_value (intptr_t): Pointer to store the attribute value.
        attribute_size (size_t): Size of the buffer in bytes.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_time_propagation_scope_split_tdvp_config_attribute_dtype`.

    .. seealso:: `cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute`
    """
    with nogil:
        __status__ = cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute(<const Handle>handle, <const TimePropagationScopeSplitTDVPConfig>config, <_TimePropagationScopeSplitTDVPConfigAttribute>attribute, <void*>attribute_value, attribute_size)
    check_status(__status__)


cpdef intptr_t create_time_propagation_approach_krylov_config(intptr_t handle) except? 0:
    """Creates a Krylov configuration object with default settings.

    Args:
        handle (intptr_t): Library handle.

    Returns:
        intptr_t: Krylov configuration object.

    .. seealso:: `cudensitymatCreateTimePropagationApproachKrylovConfig`
    """
    cdef TimePropagationApproachKrylovConfig config
    with nogil:
        __status__ = cudensitymatCreateTimePropagationApproachKrylovConfig(<const Handle>handle, &config)
    check_status(__status__)
    return <intptr_t>config


cpdef destroy_time_propagation_approach_krylov_config(intptr_t config):
    """Destroys a Krylov configuration object.

    Args:
        config (intptr_t): Krylov configuration object.

    .. seealso:: `cudensitymatDestroyTimePropagationApproachKrylovConfig`
    """
    with nogil:
        __status__ = cudensitymatDestroyTimePropagationApproachKrylovConfig(<TimePropagationApproachKrylovConfig>config)
    check_status(__status__)


######################### Python specific utility #########################

cdef dict time_propagation_approach_krylov_config_attribute_sizes = {
    CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_TOLERANCE: _numpy.float64,
    CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_MAX_DIM: _numpy.int32,
    CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_MIN_BETA: _numpy.float64,
    CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_ADAPTIVE_STEP_SIZE: _numpy.int32,
}

cpdef get_time_propagation_approach_krylov_config_attribute_dtype(int attr):
    """Get the Python data type of the corresponding TimePropagationApproachKrylovConfigAttribute attribute.

    Args:
        attr (TimePropagationApproachKrylovConfigAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`time_propagation_approach_krylov_config_get_attribute`, :func:`time_propagation_approach_krylov_config_set_attribute`.
    """
    return time_propagation_approach_krylov_config_attribute_sizes[attr]

###########################################################################


cpdef time_propagation_approach_krylov_config_set_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Sets an attribute of the Krylov configuration.

    Args:
        handle (intptr_t): Library handle.
        config (intptr_t): Krylov configuration object.
        attribute (TimePropagationApproachKrylovConfigAttribute): Attribute to set.
        attribute_value (intptr_t): Pointer to the attribute value.
        attribute_size (size_t): Size of the attribute value in bytes.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_time_propagation_approach_krylov_config_attribute_dtype`.

    .. seealso:: `cudensitymatTimePropagationApproachKrylovConfigSetAttribute`
    """
    with nogil:
        __status__ = cudensitymatTimePropagationApproachKrylovConfigSetAttribute(<const Handle>handle, <TimePropagationApproachKrylovConfig>config, <_TimePropagationApproachKrylovConfigAttribute>attribute, <const void*>attribute_value, attribute_size)
    check_status(__status__)


cpdef time_propagation_approach_krylov_config_get_attribute(intptr_t handle, intptr_t config, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Gets an attribute of the Krylov configuration.

    Args:
        handle (intptr_t): Library handle.
        config (intptr_t): Krylov configuration object.
        attribute (TimePropagationApproachKrylovConfigAttribute): Attribute to get.
        attribute_value (intptr_t): Pointer to store the attribute value.
        attribute_size (size_t): Size of the buffer in bytes.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_time_propagation_approach_krylov_config_attribute_dtype`.

    .. seealso:: `cudensitymatTimePropagationApproachKrylovConfigGetAttribute`
    """
    with nogil:
        __status__ = cudensitymatTimePropagationApproachKrylovConfigGetAttribute(<const Handle>handle, <const TimePropagationApproachKrylovConfig>config, <_TimePropagationApproachKrylovConfigAttribute>attribute, <void*>attribute_value, attribute_size)
    check_status(__status__)


cpdef intptr_t create_time_propagation(intptr_t handle, intptr_t superoperator, int32_t is_hermitian, int scope_kind, int approach_kind) except? 0:
    """Creates a time propagation object for a given operator.

    Args:
        handle (intptr_t): Library handle.
        superoperator (intptr_t): Operator.
        is_hermitian (int32_t): Specifies whether the operator is Hermitian (!=0) or not (0).
        scope_kind (TimePropagationScopeKind): Requested propagation scope.
        approach_kind (TimePropagationApproachKind): Requested propagation approach.

    Returns:
        intptr_t: Time propagation object.

    .. seealso:: `cudensitymatCreateTimePropagation`
    """
    cdef TimePropagation time_propagation
    with nogil:
        __status__ = cudensitymatCreateTimePropagation(<const Handle>handle, <Operator>superoperator, is_hermitian, <_TimePropagationScopeKind>scope_kind, <_TimePropagationApproachKind>approach_kind, &time_propagation)
    check_status(__status__)
    return <intptr_t>time_propagation


cpdef destroy_time_propagation(intptr_t time_propagation):
    """Destroys a time propagation object.

    Args:
        time_propagation (intptr_t): Time propagation object.

    .. seealso:: `cudensitymatDestroyTimePropagation`
    """
    with nogil:
        __status__ = cudensitymatDestroyTimePropagation(<TimePropagation>time_propagation)
    check_status(__status__)


######################### Python specific utility #########################

cdef dict time_propagation_attribute_sizes = {
    CUDENSITYMAT_PROPAGATION_SPLIT_SCOPE_KIND: _numpy.int32,
    CUDENSITYMAT_PROPAGATION_SPLIT_SCOPE_TDVP_CONFIG: _numpy.intp,
    CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_CONFIG: _numpy.intp,
}

cpdef get_time_propagation_attribute_dtype(int attr):
    """Get the Python data type of the corresponding TimePropagationAttribute attribute.

    Args:
        attr (TimePropagationAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`time_propagation_configure`.
    """
    return time_propagation_attribute_sizes[attr]

###########################################################################


cpdef time_propagation_configure(intptr_t handle, intptr_t time_propagation, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Configures the time propagation object with a configuration attribute.

    Args:
        handle (intptr_t): Library handle.
        time_propagation (intptr_t): Time propagation object.
        attribute (TimePropagationAttribute): Attribute to set.
        attribute_value (intptr_t): Pointer to the attribute value.
        attribute_size (size_t): Size of the attribute value in bytes.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_time_propagation_attribute_dtype`.

    .. seealso:: `cudensitymatTimePropagationConfigure`
    """
    with nogil:
        __status__ = cudensitymatTimePropagationConfigure(<const Handle>handle, <TimePropagation>time_propagation, <_TimePropagationAttribute>attribute, <const void*>attribute_value, attribute_size)
    check_status(__status__)


cpdef time_propagation_prepare(intptr_t handle, intptr_t time_propagation, intptr_t state_in, intptr_t state_out, int compute_type, size_t workspace_size_limit, intptr_t workspace, intptr_t stream):
    """Prepares the time propagation object for computation.

    Args:
        handle (intptr_t): Library handle.
        time_propagation (intptr_t): Time propagation object.
        state_in (intptr_t): Representative input quantum state for the time propagation.
        state_out (intptr_t): Representative output quantum state for the time propagation.
        compute_type (ComputeType): Desired compute type.
        workspace_size_limit (size_t): Workspace buffer size limit (bytes).
        workspace (intptr_t): Empty workspace descriptor on entrance. The workspace size required for the computation will be set on exit.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatTimePropagationPrepare`
    """
    with nogil:
        __status__ = cudensitymatTimePropagationPrepare(<const Handle>handle, <TimePropagation>time_propagation, <const State>state_in, <const State>state_out, <_ComputeType>compute_type, workspace_size_limit, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef time_propagation_compute(intptr_t handle, intptr_t time_propagation, double time_step_real, double time_step_imag, double time, int64_t batch_size, int32_t num_params, intptr_t params, intptr_t state_in, intptr_t state_out, intptr_t workspace, intptr_t stream):
    """Computes the time propagation of a quantum state under the action of the operator.

    Args:
        handle (intptr_t): Library handle.
        time_propagation (intptr_t): Time propagation object.
        time_step_real (double): Real part of time step for propagation.
        time_step_imag (double): Imaginary part of time step for propagation.
        time (double): Time value.
        batch_size (int64_t): Batch size (>=1).
        num_params (int32_t): Number of variable parameters defined by the user.
        params (intptr_t): GPU-accessible pointer to an F-order 2d-array of user-defined real parameter values: params[num_params, batch_size].
        state_in (intptr_t): Input quantum state (can be batched).
        state_out (intptr_t): Time propagated output quantum state (can be batched).
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream.

    .. seealso:: `cudensitymatTimePropagationCompute`
    """
    with nogil:
        __status__ = cudensitymatTimePropagationCompute(<const Handle>handle, <TimePropagation>time_propagation, time_step_real, time_step_imag, time, batch_size, num_params, <const double*>params, <const State>state_in, <State>state_out, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


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
        __status__ = cudensitymatCreateWorkspace(<const Handle>handle, &workspace_descr)
    check_status(__status__)
    return <intptr_t>workspace_descr


cpdef destroy_workspace(intptr_t workspace_descr):
    """Destroys a workspace descriptor.

    Args:
        workspace_descr (intptr_t): Workspace descriptor.

    .. seealso:: `cudensitymatDestroyWorkspace`
    """
    with nogil:
        __status__ = cudensitymatDestroyWorkspace(<WorkspaceDescriptor>workspace_descr)
    check_status(__status__)


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
        __status__ = cudensitymatWorkspaceGetMemorySize(<const Handle>handle, <const WorkspaceDescriptor>workspace_descr, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, &memory_buffer_size)
    check_status(__status__)
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
        __status__ = cudensitymatWorkspaceSetMemory(<const Handle>handle, <WorkspaceDescriptor>workspace_descr, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, <void*>memory_buffer, memory_buffer_size)
    check_status(__status__)


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
        __status__ = cudensitymatWorkspaceGetMemory(<const Handle>handle, <const WorkspaceDescriptor>workspace_descr, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, &memory_buffer, &memory_buffer_size)
    check_status(__status__)
    return (<intptr_t>memory_buffer, memory_buffer_size)


cpdef elementary_operator_attach_buffer(intptr_t handle, intptr_t elem_operator, intptr_t buffer, size_t buffer_size):
    """Attaches a buffer to the elementary tensor operator (either batched or non-batched).

    Args:
        handle (intptr_t): Library handle.
        elem_operator (intptr_t): Elementary tensor operator (either batched or non-batched).
        buffer (intptr_t): GPU-accessible pointer to the tensor operator elements storage.
        buffer_size (size_t): Size of the memory buffer in bytes.

    .. seealso:: `cudensitymatElementaryOperatorAttachBuffer`
    """
    with nogil:
        __status__ = cudensitymatElementaryOperatorAttachBuffer(<const Handle>handle, <ElementaryOperator>elem_operator, <void*>buffer, buffer_size)
    check_status(__status__)


cpdef matrix_operator_dense_local_attach_buffer(intptr_t handle, intptr_t matrix_operator, intptr_t buffer, size_t buffer_size):
    """Attaches a buffer to the full dense local matrix operator (either batched or non-batched).

    Args:
        handle (intptr_t): Library handle.
        matrix_operator (intptr_t): Full dense local matrix operator (either batched or non-batched).
        buffer (intptr_t): GPU-accessible pointer to the matrix operator elements storage.
        buffer_size (size_t): Size of the memory buffer in bytes.

    .. seealso:: `cudensitymatMatrixOperatorDenseLocalAttachBuffer`
    """
    with nogil:
        __status__ = cudensitymatMatrixOperatorDenseLocalAttachBuffer(<const Handle>handle, <MatrixOperator>matrix_operator, <void*>buffer, buffer_size)
    check_status(__status__)

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


###############################################################################
# Callback wrappers
###############################################################################

ArrayType = np.ndarray | cp.ndarray
ScalarCallbackType = Callable[[float, ArrayType, ArrayType], None]
TensorCallbackType = Callable[[float, ArrayType, ArrayType], None]
ScalarGradientCallbackType = Callable[[float, ArrayType, ArrayType, ArrayType], None]
TensorGradientCallbackType = Callable[[float, ArrayType, ArrayType, ArrayType], None]

_CUDA_TO_NUMPY_DATA_TYPE = {
    CUDA_R_32F: np.dtype("float32"),
    CUDA_R_64F: np.dtype("float64"),
    CUDA_C_32F: np.dtype("complex64"),
    CUDA_C_64F: np.dtype("complex128"),
}

_callback_holders = defaultdict(lambda: set())


cdef class WrappedScalarCallback:

    """
    Wrapped scalar callback Python extension type.
    """

    def __init__(self, callback: ScalarCallbackType, device: CallbackDevice):
        """
        __init__(callback: ScalarCallbackType, device: CallbackDevice)

        Create a wrapped scalar callback.

        Args:
            callback: The scalar callback function.
            device: The device on which the callback is invoked.
        """
        self.callback = callback
        self.device = device

        self._struct.callback = <cudensitymatScalarCallback_t>(<void*>callback)
        self._struct.device = device
        if device == CallbackDevice.CPU:
            self._struct.wrapper = <void*>cpu_scalar_callback_wrapper
        else:
            self._struct.wrapper = <void*>gpu_scalar_callback_wrapper


cdef class WrappedTensorCallback:

    """
    Wrapped tensor callback Python extension type.
    """

    def __init__(self, callback: TensorCallbackType, device: CallbackDevice):
        """
        __init__(callback: TensorCallbackType, device: CallbackDevice)

        Create a wrapped tensor callback.

        Args:
            callback: The tensor callback function.
            device: The device on which the callback is invoked.
        """
        self.callback = callback
        self.device = device

        self._struct.callback = <cudensitymatTensorCallback_t>(<void*>callback)
        self._struct.device = device
        if device == CallbackDevice.CPU:
            self._struct.wrapper = <void*>cpu_tensor_callback_wrapper
        else:
            self._struct.wrapper = <void*>gpu_tensor_callback_wrapper


cdef class WrappedScalarGradientCallback:

    """
    Wrapped scalar gradient callback Python extension type.
    """

    def __init__(self,
                 callback: ScalarGradientCallbackType,
                 device: CallbackDevice,
                 direction: DifferentiationDir = DifferentiationDir.BACKWARD):
        """
        __init__(callback: ScalarGradientCallbackType, device: CallbackDevice, direction: DifferentiationDir = DifferentiationDir.BACKWARD)

        Create a wrapped scalar gradient callback.

        Args:
            callback: The scalar gradient callback function.
            device: The device on which the callback is invoked.
            direction: The direction of differentiation.
        """
        if direction != DifferentiationDir.BACKWARD:
            raise NotImplementedError("Only backward differentiation is supported currently.")

        self.callback = callback
        self.device = device

        self._struct.callback = <cudensitymatScalarGradientCallback_t>(<void*>callback)
        self._struct.device = device
        self._struct.direction = direction
        if device == CallbackDevice.CPU:
            self._struct.wrapper = <void*>cpu_scalar_gradient_callback_wrapper
        else:
            self._struct.wrapper = <void*>gpu_scalar_gradient_callback_wrapper


cdef class WrappedTensorGradientCallback:

    """
    Wrapped tensor gradient callback Python extension type.
    """

    def __init__(self,
                 callback: TensorGradientCallbackType,
                 device: CallbackDevice,
                 direction: DifferentiationDir = DifferentiationDir.BACKWARD):
        """
        __init__(callback: TensorGradientCallbackType, device: CallbackDevice, direction: DifferentiationDir = DifferentiationDir.BACKWARD)

        Create a wrapped tensor gradient callback.

        Args:
            callback: The tensor gradient callback function.
            device: The device on which the callback is invoked.
            direction: The direction of differentiation.
        """
        if direction != DifferentiationDir.BACKWARD:
            raise NotImplementedError("Only backward differentiation is supported currently.")

        self.callback = callback
        self.device = device

        self._struct.callback = <cudensitymatTensorGradientCallback_t>(<void*>callback)
        self._struct.device = device
        self._struct.direction = direction
        if device == CallbackDevice.CPU:
            self._struct.wrapper = <void*>cpu_tensor_gradient_callback_wrapper
        else:
            self._struct.wrapper = <void*>gpu_tensor_gradient_callback_wrapper


cdef int32_t cpu_scalar_callback_wrapper(cudensitymatScalarCallback_t _callback_,
                                         double time,
                                         int64_t batch_size,
                                         int32_t num_params,
                                         const double * _params_,
                                         cudaDataType_t _data_type_,
                                         void * _storage_,
                                         cudaStream_t _stream_) with gil:
    """
    CPU scalar callback wrapper.
    """
    try:
        # Reconstruct CUDA stream from stream pointer.
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)

        with stream:
            # Reconstruct all arguments.
            callback = <object>(<void *>_callback_)
            params = _reconstruct_cpu_params(batch_size, num_params, _params_, PyBUF_READ)
            storage = _reconstruct_cpu_storage(
                [], batch_size, _CUDA_TO_NUMPY_DATA_TYPE[_data_type_], _storage_)

            # Invoke the Python callback.
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
    """
    GPU scalar callback wrapper.
    """
    try:
        # Reconstruct CUDA stream from stream pointer.
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)

        with stream:
            # Reconstruct all arguments.
            callback = <object>(<void *>_callback_)
            params = _reconstruct_gpu_params(batch_size, num_params, _params_)
            storage = _reconstruct_gpu_storage(
                [], batch_size, _CUDA_TO_NUMPY_DATA_TYPE[_data_type_], _storage_)

            # Invoke the Python callback.
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
    """
    CPU tensor callback wrapper.
    """
    try:
        # Reconstruct CUDA stream from stream pointer.
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)
        
        with stream:
            # Reconstruct all arguments.
            callback = <object>(<void *>_callback_)
            params = _reconstruct_cpu_params(batch_size, num_params, _params_, PyBUF_READ)
            storage = _reconstruct_cpu_storage(
                tuple(_mode_extents_[i] for i in range(num_modes)),
                batch_size,
                _CUDA_TO_NUMPY_DATA_TYPE[_data_type_],
                _storage_)

            # Invoke the Python callback.
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
    """
    GPU tensor callback wrapper.
    """
    try:
        # Reconstruct CUDA stream from stream pointer.
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)

        with stream:
            # Reconstruct all arguments.
            callback = <object>(<void *>_callback_)
            params = _reconstruct_gpu_params(batch_size, num_params, _params_)
            storage = _reconstruct_gpu_storage(
                tuple(_mode_extents_[i] for i in range(num_modes)),
                batch_size,
                _CUDA_TO_NUMPY_DATA_TYPE[_data_type_],
                _storage_)

            # Invoke the Python callback.
            callback(time, params, storage)

    except Exception:
        traceback.print_exc()
        return -1

    return 0


cdef int32_t cpu_scalar_gradient_callback_wrapper(cudensitymatScalarGradientCallback_t _callback_,
                                                  double time,
                                                  int64_t batch_size,
                                                  int32_t num_params,
                                                  const double * _params_,
                                                  cudaDataType_t _data_type_,
                                                  void * _scalar_grad_,
                                                  double * _params_grad_,
                                                  cudaStream_t _stream_) with gil:
    """
    CPU scalar gradient callback wrapper.
    """
    try:
        # Reconstruct CUDA stream from stream pointer.
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)

        with stream:
            # Reconstruct all arguments.
            callback = <object>(<void *>_callback_)
            params = _reconstruct_cpu_params(batch_size, num_params, _params_, PyBUF_READ)
            scalar_grad = _reconstruct_cpu_storage(
                [], batch_size, _CUDA_TO_NUMPY_DATA_TYPE[_data_type_], _scalar_grad_)
            params_grad = _reconstruct_cpu_params(batch_size, num_params, _params_grad_, PyBUF_WRITE)

            # Invoke the Python callback.
            callback(time, params, scalar_grad, params_grad)

    except Exception:
        traceback.print_exc()
        return -1

    return 0


cdef int32_t gpu_scalar_gradient_callback_wrapper(cudensitymatScalarGradientCallback_t _callback_,
                                                  double time,
                                                  int64_t batch_size,
                                                  int32_t num_params,
                                                  const double * _params_,
                                                  cudaDataType_t _data_type_,
                                                  void * _scalar_grad_,
                                                  double * _params_grad_,
                                                  cudaStream_t _stream_) with gil:
    """
    GPU scalar gradient callback wrapper.
    """
    try:
        # Reconstruct CUDA stream from stream pointer.
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)

        with stream:
            # Reconstruct all arguments.
            callback = <object>(<void *>_callback_)
            params = _reconstruct_gpu_params(batch_size, num_params, _params_)
            scalar_grad = _reconstruct_gpu_storage(
                [], batch_size, _CUDA_TO_NUMPY_DATA_TYPE[_data_type_], _scalar_grad_)
            params_grad = _reconstruct_gpu_params(batch_size, num_params, _params_grad_)

            # Invoke the Python callback.
            callback(time, params, scalar_grad, params_grad)

    except Exception:
        traceback.print_exc()
        return -1

    return 0


cdef int32_t cpu_tensor_gradient_callback_wrapper(cudensitymatTensorGradientCallback_t _callback_,
                                                  cudensitymatElementaryOperatorSparsity_t sparsity,
                                                  int32_t num_modes,
                                                  const int64_t * _mode_extents_,
                                                  const int32_t * _diagonal_offsets_,
                                                  double time,
                                                  int64_t batch_size,
                                                  int32_t num_params,
                                                  const double * _params_,
                                                  cudaDataType_t _data_type_,
                                                  void * _tensor_grad_,
                                                  double * _params_grad_,
                                                  cudaStream_t _stream_) with gil:
    """
    CPU tensor gradient callback wrapper.
    """
    try:
        # Reconstruct CUDA stream from stream pointer.
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)

        with stream:
            # Reconstruct all arguments.
            callback = <object>(<void *>_callback_)
            params = _reconstruct_cpu_params(batch_size, num_params, _params_, PyBUF_READ)
            tensor_grad = _reconstruct_cpu_storage(
                tuple(_mode_extents_[i] for i in range(num_modes)),
                batch_size,
                _CUDA_TO_NUMPY_DATA_TYPE[_data_type_],
                _tensor_grad_)
            params_grad = _reconstruct_cpu_params(batch_size, num_params, _params_grad_, PyBUF_WRITE)

            # Invoke the Python callback.
            callback(time, params, tensor_grad, params_grad)

    except Exception:
        traceback.print_exc()
        return -1

    return 0


cdef int32_t gpu_tensor_gradient_callback_wrapper(cudensitymatTensorGradientCallback_t _callback_,
                                                  cudensitymatElementaryOperatorSparsity_t sparsity,
                                                  int32_t num_modes,
                                                  const int64_t * _mode_extents_,
                                                  const int32_t * _diagonal_offsets_,
                                                  double time,
                                                  int64_t batch_size,
                                                  int32_t num_params,
                                                  const double * _params_,
                                                  cudaDataType_t _data_type_,
                                                  void * _tensor_grad_,
                                                  double * _params_grad_,
                                                  cudaStream_t _stream_) with gil:
    """
    GPU tensor gradient callback wrapper.
    """
    try:
        # Reconstruct CUDA stream from stream pointer.
        stream = cp.cuda.ExternalStream(<intptr_t>_stream_)

        with stream:
            # Reconstruct all arguments.
            callback = <object>(<void *>_callback_)
            params = _reconstruct_gpu_params(batch_size, num_params, _params_)
            tensor_grad = _reconstruct_gpu_storage(
                tuple(_mode_extents_[i] for i in range(num_modes)),
                batch_size,
                _CUDA_TO_NUMPY_DATA_TYPE[_data_type_],
                _tensor_grad_)
            params_grad = _reconstruct_gpu_params(batch_size, num_params, _params_grad_)

            # Invoke the Python callback.
            callback(time, params, tensor_grad, params_grad)

    except Exception:
        traceback.print_exc()
        return -1

    return 0
