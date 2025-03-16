# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 25.03.0. Do not modify it directly.

cimport cython
cimport cpython
from libcpp.vector cimport vector

from ._utils cimport (get_resource_ptr, get_nested_resource_ptr, nested_resource, nullable_unique_ptr,
                      get_buffer_pointer, get_resource_ptrs, DeviceAllocType, DeviceFreeType,
                      cuqnt_alloc_wrapper, cuqnt_free_wrapper, logger_callback_with_data)

from enum import IntEnum as _IntEnum
import warnings as _warnings

import numpy as _numpy


###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """See `cutensornetStatus_t`."""
    SUCCESS = CUTENSORNET_STATUS_SUCCESS
    NOT_INITIALIZED = CUTENSORNET_STATUS_NOT_INITIALIZED
    ALLOC_FAILED = CUTENSORNET_STATUS_ALLOC_FAILED
    INVALID_VALUE = CUTENSORNET_STATUS_INVALID_VALUE
    ARCH_MISMATCH = CUTENSORNET_STATUS_ARCH_MISMATCH
    MAPPING_ERROR = CUTENSORNET_STATUS_MAPPING_ERROR
    EXECUTION_FAILED = CUTENSORNET_STATUS_EXECUTION_FAILED
    INTERNAL_ERROR = CUTENSORNET_STATUS_INTERNAL_ERROR
    NOT_SUPPORTED = CUTENSORNET_STATUS_NOT_SUPPORTED
    LICENSE_ERROR = CUTENSORNET_STATUS_LICENSE_ERROR
    CUBLAS_ERROR = CUTENSORNET_STATUS_CUBLAS_ERROR
    CUDA_ERROR = CUTENSORNET_STATUS_CUDA_ERROR
    INSUFFICIENT_WORKSPACE = CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE
    INSUFFICIENT_DRIVER = CUTENSORNET_STATUS_INSUFFICIENT_DRIVER
    IO_ERROR = CUTENSORNET_STATUS_IO_ERROR
    CUTENSOR_VERSION_MISMATCH = CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH
    NO_DEVICE_ALLOCATOR = CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR
    ALL_HYPER_SAMPLES_FAILED = CUTENSORNET_STATUS_ALL_HYPER_SAMPLES_FAILED
    CUSOLVER_ERROR = CUTENSORNET_STATUS_CUSOLVER_ERROR
    DEVICE_ALLOCATOR_ERROR = CUTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR
    DISTRIBUTED_FAILURE = CUTENSORNET_STATUS_DISTRIBUTED_FAILURE
    INTERRUPTED = CUTENSORNET_STATUS_INTERRUPTED

class ComputeType(_IntEnum):
    """See `cutensornetComputeType_t`."""
    COMPUTE_16F = CUTENSORNET_COMPUTE_16F
    COMPUTE_16BF = CUTENSORNET_COMPUTE_16BF
    COMPUTE_TF32 = CUTENSORNET_COMPUTE_TF32
    COMPUTE_3XTF32 = CUTENSORNET_COMPUTE_3XTF32
    COMPUTE_32F = CUTENSORNET_COMPUTE_32F
    COMPUTE_64F = CUTENSORNET_COMPUTE_64F
    COMPUTE_8U = CUTENSORNET_COMPUTE_8U
    COMPUTE_8I = CUTENSORNET_COMPUTE_8I
    COMPUTE_32U = CUTENSORNET_COMPUTE_32U
    COMPUTE_32I = CUTENSORNET_COMPUTE_32I

class GraphAlgo(_IntEnum):
    """See `cutensornetGraphAlgo_t`."""
    RB = CUTENSORNET_GRAPH_ALGO_RB
    KWAY = CUTENSORNET_GRAPH_ALGO_KWAY

class MemoryModel(_IntEnum):
    """See `cutensornetMemoryModel_t`."""
    HEURISTIC = CUTENSORNET_MEMORY_MODEL_HEURISTIC
    CUTENSOR = CUTENSORNET_MEMORY_MODEL_CUTENSOR

class OptimizerCost(_IntEnum):
    """See `cutensornetOptimizerCost_t`."""
    FLOPS = CUTENSORNET_OPTIMIZER_COST_FLOPS
    TIME = CUTENSORNET_OPTIMIZER_COST_TIME
    TIME_TUNED = CUTENSORNET_OPTIMIZER_COST_TIME_TUNED

class ContractionOptimizerConfigAttribute(_IntEnum):
    """See `cutensornetContractionOptimizerConfigAttributes_t`."""
    GRAPH_NUM_PARTITIONS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_PARTITIONS
    GRAPH_CUTOFF_SIZE = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_CUTOFF_SIZE
    GRAPH_ALGORITHM = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM
    GRAPH_IMBALANCE_FACTOR = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR
    GRAPH_NUM_ITERATIONS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_ITERATIONS
    GRAPH_NUM_CUTS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS
    RECONFIG_NUM_ITERATIONS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS
    RECONFIG_NUM_LEAVES = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES
    SLICER_DISABLE_SLICING = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING
    SLICER_MEMORY_MODEL = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL
    SLICER_MEMORY_FACTOR = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR
    SLICER_MIN_SLICES = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES
    SLICER_SLICE_FACTOR = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_SLICE_FACTOR
    HYPER_NUM_SAMPLES = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES
    HYPER_NUM_THREADS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS
    SIMPLIFICATION_DISABLE_DR = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR
    SEED = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED
    COST_FUNCTION_OBJECTIVE = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE
    CACHE_REUSE_NRUNS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS
    SMART_OPTION = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION

class ContractionOptimizerInfoAttribute(_IntEnum):
    """See `cutensornetContractionOptimizerInfoAttributes_t`."""
    PATH = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH
    NUM_SLICES = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES
    NUM_SLICED_MODES = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES
    SLICED_MODE = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE
    SLICED_EXTENT = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT
    SLICING_CONFIG = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG
    SLICING_OVERHEAD = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_OVERHEAD
    PHASE1_FLOP_COUNT = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PHASE1_FLOP_COUNT
    FLOP_COUNT = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT
    EFFECTIVE_FLOPS_EST = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS_EST
    RUNTIME_EST = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_RUNTIME_EST
    LARGEST_TENSOR = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR
    INTERMEDIATE_MODES = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES
    NUM_INTERMEDIATE_MODES = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_INTERMEDIATE_MODES

class ContractionAutotunePreferenceAttribute(_IntEnum):
    """See `cutensornetContractionAutotunePreferenceAttributes_t`."""
    MAX_ITERATIONS = CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS
    INTERMEDIATE_MODES = CUTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES

class WorksizePref(_IntEnum):
    """See `cutensornetWorksizePref_t`."""
    MIN = CUTENSORNET_WORKSIZE_PREF_MIN
    RECOMMENDED = CUTENSORNET_WORKSIZE_PREF_RECOMMENDED
    MAX = CUTENSORNET_WORKSIZE_PREF_MAX

class Memspace(_IntEnum):
    """See `cutensornetMemspace_t`."""
    DEVICE = CUTENSORNET_MEMSPACE_DEVICE
    HOST = CUTENSORNET_MEMSPACE_HOST

class WorkspaceKind(_IntEnum):
    """See `cutensornetWorkspaceKind_t`."""
    SCRATCH = CUTENSORNET_WORKSPACE_SCRATCH
    CACHE = CUTENSORNET_WORKSPACE_CACHE

class TensorSVDConfigAttribute(_IntEnum):
    """See `cutensornetTensorSVDConfigAttributes_t`."""
    ABS_CUTOFF = CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF
    REL_CUTOFF = CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF
    S_NORMALIZATION = CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION
    S_PARTITION = CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION
    ALGO = CUTENSORNET_TENSOR_SVD_CONFIG_ALGO
    ALGO_PARAMS = CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS
    DISCARDED_WEIGHT_CUTOFF = CUTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF

class TensorSVDPartition(_IntEnum):
    """See `cutensornetTensorSVDPartition_t`."""
    NONE = CUTENSORNET_TENSOR_SVD_PARTITION_NONE
    US = CUTENSORNET_TENSOR_SVD_PARTITION_US
    SV = CUTENSORNET_TENSOR_SVD_PARTITION_SV
    UV_EQUAL = CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL

class TensorSVDNormalization(_IntEnum):
    """See `cutensornetTensorSVDNormalization_t`."""
    NONE = CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE
    L1 = CUTENSORNET_TENSOR_SVD_NORMALIZATION_L1
    L2 = CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2
    LINF = CUTENSORNET_TENSOR_SVD_NORMALIZATION_LINF

class TensorSVDInfoAttribute(_IntEnum):
    """See `cutensornetTensorSVDInfoAttributes_t`."""
    FULL_EXTENT = CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT
    REDUCED_EXTENT = CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT
    DISCARDED_WEIGHT = CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT
    ALGO = CUTENSORNET_TENSOR_SVD_INFO_ALGO
    ALGO_STATUS = CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS

class GateSplitAlgo(_IntEnum):
    """See `cutensornetGateSplitAlgo_t`."""
    DIRECT = CUTENSORNET_GATE_SPLIT_ALGO_DIRECT
    REDUCED = CUTENSORNET_GATE_SPLIT_ALGO_REDUCED

class NetworkAttribute(_IntEnum):
    """See `cutensornetNetworkAttributes_t`."""
    INPUT_TENSORS_NUM_CONSTANT = CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONSTANT
    INPUT_TENSORS_CONSTANT = CUTENSORNET_NETWORK_INPUT_TENSORS_CONSTANT
    INPUT_TENSORS_NUM_CONJUGATED = CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONJUGATED
    INPUT_TENSORS_CONJUGATED = CUTENSORNET_NETWORK_INPUT_TENSORS_CONJUGATED
    INPUT_TENSORS_NUM_REQUIRE_GRAD = CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_REQUIRE_GRAD
    INPUT_TENSORS_REQUIRE_GRAD = CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD

class SmartOption(_IntEnum):
    """See `cutensornetSmartOption_t`."""
    DISABLED = CUTENSORNET_SMART_OPTION_DISABLED
    ENABLED = CUTENSORNET_SMART_OPTION_ENABLED

class TensorSVDAlgo(_IntEnum):
    """See `cutensornetTensorSVDAlgo_t`."""
    GESVD = CUTENSORNET_TENSOR_SVD_ALGO_GESVD
    GESVDJ = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ
    GESVDP = CUTENSORNET_TENSOR_SVD_ALGO_GESVDP
    GESVDR = CUTENSORNET_TENSOR_SVD_ALGO_GESVDR

class StatePurity(_IntEnum):
    """See `cutensornetStatePurity_t`."""
    PURE = CUTENSORNET_STATE_PURITY_PURE

class MarginalAttribute(_IntEnum):
    """See `cutensornetMarginalAttributes_t`."""
    OPT_NUM_HYPER_SAMPLES = CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES
    CONFIG_NUM_HYPER_SAMPLES = CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES
    INFO_FLOPS = CUTENSORNET_MARGINAL_INFO_FLOPS

class SamplerAttribute(_IntEnum):
    """See `cutensornetSamplerAttributes_t`."""
    OPT_NUM_HYPER_SAMPLES = CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES
    CONFIG_NUM_HYPER_SAMPLES = CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES
    CONFIG_DETERMINISTIC = CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC
    INFO_FLOPS = CUTENSORNET_SAMPLER_INFO_FLOPS

class AccessorAttribute(_IntEnum):
    """See `cutensornetAccessorAttributes_t`."""
    OPT_NUM_HYPER_SAMPLES = CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES
    CONFIG_NUM_HYPER_SAMPLES = CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES
    INFO_FLOPS = CUTENSORNET_ACCESSOR_INFO_FLOPS

class ExpectationAttribute(_IntEnum):
    """See `cutensornetExpectationAttributes_t`."""
    OPT_NUM_HYPER_SAMPLES = CUTENSORNET_EXPECTATION_OPT_NUM_HYPER_SAMPLES
    CONFIG_NUM_HYPER_SAMPLES = CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES
    INFO_FLOPS = CUTENSORNET_EXPECTATION_INFO_FLOPS

class BoundaryCondition(_IntEnum):
    """See `cutensornetBoundaryCondition_t`."""
    OPEN = CUTENSORNET_BOUNDARY_CONDITION_OPEN

class StateAttribute(_IntEnum):
    """See `cutensornetStateAttributes_t`."""
    MPS_CANONICAL_CENTER = CUTENSORNET_STATE_MPS_CANONICAL_CENTER
    MPS_SVD_CONFIG_ABS_CUTOFF = CUTENSORNET_STATE_MPS_SVD_CONFIG_ABS_CUTOFF
    MPS_SVD_CONFIG_REL_CUTOFF = CUTENSORNET_STATE_MPS_SVD_CONFIG_REL_CUTOFF
    MPS_SVD_CONFIG_S_NORMALIZATION = CUTENSORNET_STATE_MPS_SVD_CONFIG_S_NORMALIZATION
    MPS_SVD_CONFIG_ALGO = CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO
    MPS_SVD_CONFIG_ALGO_PARAMS = CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO_PARAMS
    MPS_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF = CUTENSORNET_STATE_MPS_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF
    NUM_HYPER_SAMPLES = CUTENSORNET_STATE_NUM_HYPER_SAMPLES
    CONFIG_MPS_CANONICAL_CENTER = CUTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER
    CONFIG_MPS_SVD_ABS_CUTOFF = CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF
    CONFIG_MPS_SVD_REL_CUTOFF = CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF
    CONFIG_MPS_SVD_S_NORMALIZATION = CUTENSORNET_STATE_CONFIG_MPS_SVD_S_NORMALIZATION
    CONFIG_MPS_SVD_ALGO = CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO
    CONFIG_MPS_SVD_ALGO_PARAMS = CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO_PARAMS
    CONFIG_MPS_SVD_DISCARDED_WEIGHT_CUTOFF = CUTENSORNET_STATE_CONFIG_MPS_SVD_DISCARDED_WEIGHT_CUTOFF
    CONFIG_MPS_MPO_APPLICATION = CUTENSORNET_STATE_CONFIG_MPS_MPO_APPLICATION
    CONFIG_MPS_GAUGE_OPTION = CUTENSORNET_STATE_CONFIG_MPS_GAUGE_OPTION
    CONFIG_NUM_HYPER_SAMPLES = CUTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES
    INFO_FLOPS = CUTENSORNET_STATE_INFO_FLOPS

class StateMPOApplication(_IntEnum):
    """See `cutensornetStateMPOApplication_t`."""
    INEXACT = CUTENSORNET_STATE_MPO_APPLICATION_INEXACT
    EXACT = CUTENSORNET_STATE_MPO_APPLICATION_EXACT

class StateMPSGaugeOption(_IntEnum):
    """See `cutensornetStateMPSGaugeOption_t`."""
    STATE_MPS_GAUGE_FREE = CUTENSORNET_STATE_MPS_GAUGE_FREE
    STATE_MPS_GAUGE_SIMPLE = CUTENSORNET_STATE_MPS_GAUGE_SIMPLE


###############################################################################
# Error handling
###############################################################################

class cuTensorNetError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value}): {get_error_string(status)}"
        super(cuTensorNetError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuTensorNetError(status)


###############################################################################
# Special dtypes
###############################################################################

tensor_id_list_dtype = _numpy.dtype(
    {'names':['num_tensors','data'],
     'formats': (_numpy.int32, _numpy.intp),
     'itemsize': sizeof(TensorIDList),
    }, align=True
)

contraction_path_dtype = _numpy.dtype(
    {'names':['num_contractions','data'],
     'formats': (_numpy.uint32, _numpy.intp),
     'itemsize': sizeof(ContractionPath),
    }, align=True
)

# We need this dtype because its members are not of the same type...
slice_info_pair_dtype = _numpy.dtype(
    {'names': ('sliced_mode','sliced_extent'),
     'formats': (_numpy.int32, _numpy.int64),
     'itemsize': sizeof(SliceInfoPair),
    }, align=True
)

slicing_config_dtype = _numpy.dtype(
    {'names': ('num_sliced_modes','data'),
     'formats': (_numpy.uint32, _numpy.intp),
     'itemsize': sizeof(SlicingConfig),
    }, align=True
)

gesvdj_params_dtype = _numpy.dtype(
    {'names': ('tol','max_sweeps'),
     'formats': (_numpy.float64, _numpy.int32),
     'itemsize': sizeof(GesvdjParams),
    }, align=True
)

gesvdr_params_dtype = _numpy.dtype(
    {'names': ('oversampling','niters'),
     'formats': (_numpy.int64, _numpy.int64),
     'itemsize': sizeof(GesvdrParams),
    }, align=True
)

gesvdj_status_dtype = _numpy.dtype(
    {'names': ('residual', 'sweeps'),
     'formats': (_numpy.float64, _numpy.int32),
     'itemsize': sizeof(GesvdjStatus),
    }, align=True
)

gesvdp_status_dtype = _numpy.dtype(
    {'names': ('err_sigma', ),
     'formats': (_numpy.float64, ),
     'itemsize': sizeof(GesvdpStatus),
    }, align=True
)

tensor_qualifiers_dtype = _numpy.dtype(
    {'names':('is_conjugate', 'is_constant', 'requires_gradient'),
     'formats': (_numpy.int32, _numpy.int32, _numpy.int32, ),
     'itemsize': sizeof(TensorQualifiers),
    }, align=True
)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t create() except? 0:
    """Initializes the cuTensorNet library.

    Returns:
        intptr_t: Pointer to ``cutensornetHandle_t``.

    .. seealso:: `cutensornetCreate`
    """
    cdef Handle handle
    with nogil:
        status = cutensornetCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """Destroys the cuTensorNet library handle.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.

    .. seealso:: `cutensornetDestroy`
    """
    with nogil:
        status = cutensornetDestroy(<Handle>handle)
    check_status(status)


cpdef intptr_t create_network_descriptor(intptr_t handle, int32_t num_inputs, num_modes_in, extents_in, strides_in, modes_in, qualifiers_in, int32_t num_modes_out, extents_out, strides_out, modes_out, int data_type, int compute_type) except? 0:
    """Initializes a ``cutensornetNetworkDescriptor_t``, describing the connectivity (i.e., network topology) between the tensors.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        num_inputs (int32_t): Number of input tensors.
        num_modes_in (object): Array of size ``num_inputs``; ``num_modes_in[i]`` denotes the number of modes available in the i-th tensor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        extents_in (object): Array of size ``num_inputs``; ``extents_in[i]`` has ``num_modes_in[i]`` many entries with ``extents_in[i][j]`` (``j`` < ``num_modes_in[i]``) corresponding to the extent of the j-th mode of tensor ``i``. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int64_t', or
            - a nested Python sequence of ``int64_t``.

        strides_in (object): Array of size ``num_inputs``; ``strides_in[i]`` has ``num_modes_in[i]`` many entries with ``strides_in[i][j]`` (``j`` < ``num_modes_in[i]``) corresponding to the linearized offset -- in physical memory -- between two logically-neighboring elements w.r.t the j-th mode of tensor ``i``. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int64_t', or
            - a nested Python sequence of ``int64_t``.

        modes_in (object): Array of size ``num_inputs``; ``modes_in[i]`` has ``num_modes_in[i]`` many entries -- each entry corresponds to a mode. Each mode that does not appear in the input tensor is implicitly contracted. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int32_t', or
            - a nested Python sequence of ``int32_t``.

        qualifiers_in (object): Array of size ``num_inputs``; ``qualifiers_in[i]`` denotes the qualifiers of i-th input tensor. Refer to ``cutensornetTensorQualifiers_t``. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cutensornetTensorQualifiers_t``.

        num_modes_out (int32_t): number of modes of the output tensor. On entry, if this value is ``-1`` and the output modes are not provided, the network will infer the output modes. If this value is ``0``, the network is force reduced.
        extents_out (object): Array of size ``num_modes_out``; ``extents_out[j]`` (``j`` < ``num_modes_out``) corresponding to the extent of the j-th mode of the output tensor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        strides_out (object): Array of size ``num_modes_out``; ``strides_out[j]`` (``j`` < ``num_modes_out``) corresponding to the linearized offset -- in physical memory -- between two logically-neighboring elements w.r.t the j-th mode of the output tensor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        modes_out (object): Array of size ``num_modes_out``; ``modes_out[j]`` denotes the j-th mode of the output tensor. output tensor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        data_type (int): Denotes the data type for all input an output tensors.
        compute_type (ComputeType): Denotes the compute type used throughout the computation.

    Returns:
        intptr_t: Pointer to a ``cutensornetNetworkDescriptor_t``.

    .. seealso:: `cutensornetCreateNetworkDescriptor`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _num_modes_in_
    get_resource_ptr[int32_t](_num_modes_in_, num_modes_in, <int32_t*>NULL)
    cdef nested_resource[ int64_t ] _extents_in_
    get_nested_resource_ptr[int64_t](_extents_in_, extents_in, <int64_t*>NULL)
    cdef nested_resource[ int64_t ] _strides_in_
    get_nested_resource_ptr[int64_t](_strides_in_, strides_in, <int64_t*>NULL)
    cdef nested_resource[ int32_t ] _modes_in_
    get_nested_resource_ptr[int32_t](_modes_in_, modes_in, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[cutensornetTensorQualifiers_t] ] _qualifiers_in_
    get_resource_ptr[cutensornetTensorQualifiers_t](_qualifiers_in_, qualifiers_in, <cutensornetTensorQualifiers_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _extents_out_
    get_resource_ptr[int64_t](_extents_out_, extents_out, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _strides_out_
    get_resource_ptr[int64_t](_strides_out_, strides_out, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _modes_out_
    get_resource_ptr[int32_t](_modes_out_, modes_out, <int32_t*>NULL)
    cdef NetworkDescriptor desc_net
    with nogil:
        status = cutensornetCreateNetworkDescriptor(<const Handle>handle, num_inputs, <const int32_t*>(_num_modes_in_.data()), <const int64_t* const*>(_extents_in_.ptrs.data()), <const int64_t* const*>(_strides_in_.ptrs.data()), <const int32_t* const*>(_modes_in_.ptrs.data()), <const cutensornetTensorQualifiers_t*>(_qualifiers_in_.data()), num_modes_out, <const int64_t*>(_extents_out_.data()), <const int64_t*>(_strides_out_.data()), <const int32_t*>(_modes_out_.data()), <DataType>data_type, <_ComputeType>compute_type, &desc_net)
    check_status(status)
    return <intptr_t>desc_net


cpdef destroy_network_descriptor(intptr_t desc):
    """Frees all the memory associated with the network descriptor.

    Args:
        desc (intptr_t): Opaque handle to a tensor network descriptor.

    .. seealso:: `cutensornetDestroyNetworkDescriptor`
    """
    with nogil:
        status = cutensornetDestroyNetworkDescriptor(<NetworkDescriptor>desc)
    check_status(status)


cpdef intptr_t get_output_tensor_descriptor(intptr_t handle, intptr_t desc_net) except? 0:
    """Creates a ``cutensornetTensorDescriptor_t`` representing the output tensor of the network.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_net (intptr_t): Pointer to a ``cutensornetNetworkDescriptor_t``.

    Returns:
        intptr_t: an opaque ``cutensornetTensorDescriptor_t`` struct. Cannot be null. On return, a new ``cutensornetTensorDescriptor_t`` holds the meta-data of the ``desc_net`` output tensor.

    .. seealso:: `cutensornetGetOutputTensorDescriptor`
    """
    cdef TensorDescriptor output_tensor_desc
    with nogil:
        status = cutensornetGetOutputTensorDescriptor(<const Handle>handle, <const NetworkDescriptor>desc_net, &output_tensor_desc)
    check_status(status)
    return <intptr_t>output_tensor_desc


cpdef intptr_t create_workspace_descriptor(intptr_t handle) except? 0:
    """Creates a workspace descriptor that holds information about the user provided memory buffer.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.

    Returns:
        intptr_t: Pointer to the opaque workspace descriptor.

    .. seealso:: `cutensornetCreateWorkspaceDescriptor`
    """
    cdef WorkspaceDescriptor work_desc
    with nogil:
        status = cutensornetCreateWorkspaceDescriptor(<const Handle>handle, &work_desc)
    check_status(status)
    return <intptr_t>work_desc


cpdef workspace_compute_contraction_sizes(intptr_t handle, intptr_t desc_net, intptr_t optimizer_info, intptr_t work_desc):
    """Computes the workspace size needed to contract the input tensor network using the provided contraction path.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_net (intptr_t): Describes the tensor network (i.e., its tensors and their connectivity).
        optimizer_info (intptr_t): Opaque structure.
        work_desc (intptr_t): The workspace descriptor in which the information is collected.

    .. seealso:: `cutensornetWorkspaceComputeContractionSizes`
    """
    with nogil:
        status = cutensornetWorkspaceComputeContractionSizes(<const Handle>handle, <const NetworkDescriptor>desc_net, <const ContractionOptimizerInfo>optimizer_info, <WorkspaceDescriptor>work_desc)
    check_status(status)


cpdef int64_t workspace_get_memory_size(intptr_t handle, intptr_t work_desc, int work_pref, int mem_space, int work_kind) except? -1:
    """Retrieves the needed workspace size for the given workspace preference, memory space, workspace kind.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        work_desc (intptr_t): Opaque structure describing the workspace.
        work_pref (WorksizePref): Preference of workspace for planning.
        mem_space (Memspace): The memory space where the workspace is allocated.
        work_kind (WorkspaceKind): The kind of workspace.

    Returns:
        int64_t: Needed workspace size.

    .. seealso:: `cutensornetWorkspaceGetMemorySize`
    """
    cdef int64_t memory_size
    with nogil:
        status = cutensornetWorkspaceGetMemorySize(<const Handle>handle, <const WorkspaceDescriptor>work_desc, <_WorksizePref>work_pref, <_Memspace>mem_space, <_WorkspaceKind>work_kind, &memory_size)
    check_status(status)
    return memory_size


cpdef workspace_set_memory(intptr_t handle, intptr_t work_desc, int mem_space, int work_kind, intptr_t memory_ptr, int64_t memory_size):
    """Sets the memory address and workspace size of the workspace provided by user.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        work_desc (intptr_t): Opaque structure describing the workspace.
        mem_space (Memspace): The memory space where the workspace is allocated.
        work_kind (WorkspaceKind): The kind of workspace.
        memory_ptr (intptr_t): Workspace memory pointer, may be null.
        memory_size (int64_t): Workspace size.

    .. seealso:: `cutensornetWorkspaceSetMemory`
    """
    with nogil:
        status = cutensornetWorkspaceSetMemory(<const Handle>handle, <WorkspaceDescriptor>work_desc, <_Memspace>mem_space, <_WorkspaceKind>work_kind, <void* const>memory_ptr, memory_size)
    check_status(status)


cpdef tuple workspace_get_memory(intptr_t handle, intptr_t work_desc, int mem_space, int work_kind):
    """Retrieves the memory address and workspace size of workspace hosted in the workspace descriptor.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        work_desc (intptr_t): Opaque structure describing the workspace.
        mem_space (Memspace): The memory space where the workspace is allocated.
        work_kind (WorkspaceKind): The kind of workspace.

    Returns:
        A 2-tuple containing:

        - intptr_t: Workspace memory pointer.
        - int64_t: Workspace size.

    .. seealso:: `cutensornetWorkspaceGetMemory`
    """
    cdef void* memory_ptr
    cdef int64_t memory_size
    with nogil:
        status = cutensornetWorkspaceGetMemory(<const Handle>handle, <const WorkspaceDescriptor>work_desc, <_Memspace>mem_space, <_WorkspaceKind>work_kind, &memory_ptr, &memory_size)
    check_status(status)
    return (<intptr_t>memory_ptr, memory_size)


cpdef destroy_workspace_descriptor(intptr_t desc):
    """Frees the workspace descriptor.

    Args:
        desc (intptr_t): Opaque structure.

    .. seealso:: `cutensornetDestroyWorkspaceDescriptor`
    """
    with nogil:
        status = cutensornetDestroyWorkspaceDescriptor(<WorkspaceDescriptor>desc)
    check_status(status)


cpdef intptr_t create_contraction_optimizer_config(intptr_t handle) except? 0:
    """Sets up the required hyper-optimization parameters for the contraction order solver (see :func:`contraction_optimize`).

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.

    Returns:
        intptr_t: This data structure holds all information about the user-requested hyper-optimization parameters.

    .. seealso:: `cutensornetCreateContractionOptimizerConfig`
    """
    cdef ContractionOptimizerConfig optimizer_config
    with nogil:
        status = cutensornetCreateContractionOptimizerConfig(<const Handle>handle, &optimizer_config)
    check_status(status)
    return <intptr_t>optimizer_config


cpdef destroy_contraction_optimizer_config(intptr_t optimizer_config):
    """Frees all the memory associated with ``optimizer_config``.

    Args:
        optimizer_config (intptr_t): Opaque structure.

    .. seealso:: `cutensornetDestroyContractionOptimizerConfig`
    """
    with nogil:
        status = cutensornetDestroyContractionOptimizerConfig(<ContractionOptimizerConfig>optimizer_config)
    check_status(status)


######################### Python specific utility #########################

cdef dict contraction_optimizer_config_attribute_sizes = {
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_PARTITIONS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_CUTOFF_SIZE: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_ITERATIONS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_SLICE_FACTOR: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION: _numpy.int32,
}

cpdef get_contraction_optimizer_config_attribute_dtype(int attr):
    """Get the Python data type of the corresponding ContractionOptimizerConfigAttribute attribute.

    Args:
        attr (ContractionOptimizerConfigAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`contraction_optimizer_config_get_attribute`, :func:`contraction_optimizer_config_set_attribute`.
    """
    return contraction_optimizer_config_attribute_sizes[attr]

###########################################################################


cpdef contraction_optimizer_config_get_attribute(intptr_t handle, intptr_t optimizer_config, int attr, intptr_t buf, size_t size_in_bytes):
    """Gets attributes of ``optimizer_config``.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        optimizer_config (intptr_t): Opaque structure that is accessed.
        attr (ContractionOptimizerConfigAttribute): Specifies the attribute that is requested.
        buf (intptr_t): On return, this buffer (of size ``size_in_bytes``) holds the value that corresponds to ``attr`` within ``optimizer_config``.
        size_in_bytes (size_t): Size of ``buf`` (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_contraction_optimizer_config_attribute_dtype`.

    .. seealso:: `cutensornetContractionOptimizerConfigGetAttribute`
    """
    with nogil:
        status = cutensornetContractionOptimizerConfigGetAttribute(<const Handle>handle, <const ContractionOptimizerConfig>optimizer_config, <_ContractionOptimizerConfigAttribute>attr, <void*>buf, size_in_bytes)
    check_status(status)


cpdef contraction_optimizer_config_set_attribute(intptr_t handle, intptr_t optimizer_config, int attr, intptr_t buf, size_t size_in_bytes):
    """Sets attributes of ``optimizer_config``.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        optimizer_config (intptr_t): Opaque structure that is accessed.
        attr (ContractionOptimizerConfigAttribute): Specifies the attribute that is requested.
        buf (intptr_t): This buffer (of size ``size_in_bytes``) determines the value to which ``attr`` will be set.
        size_in_bytes (size_t): Size of ``buf`` (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_contraction_optimizer_config_attribute_dtype`.

    .. seealso:: `cutensornetContractionOptimizerConfigSetAttribute`
    """
    with nogil:
        status = cutensornetContractionOptimizerConfigSetAttribute(<const Handle>handle, <ContractionOptimizerConfig>optimizer_config, <_ContractionOptimizerConfigAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(status)


cpdef destroy_contraction_optimizer_info(intptr_t optimizer_info):
    """Frees all the memory associated with ``optimizer_info``.

    Args:
        optimizer_info (intptr_t): Opaque structure.

    .. seealso:: `cutensornetDestroyContractionOptimizerInfo`
    """
    with nogil:
        status = cutensornetDestroyContractionOptimizerInfo(<ContractionOptimizerInfo>optimizer_info)
    check_status(status)


cpdef intptr_t create_contraction_optimizer_info(intptr_t handle, intptr_t desc_net) except? 0:
    """Allocates resources for ``optimizerInfo``.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_net (intptr_t): Describes the tensor network (i.e., its tensors and their connectivity) for which ``optimizerInfo`` is created.

    Returns:
        intptr_t: Pointer to ``cutensornetContractionOptimizerInfo_t``.

    .. seealso:: `cutensornetCreateContractionOptimizerInfo`
    """
    cdef ContractionOptimizerInfo optimizer_info
    with nogil:
        status = cutensornetCreateContractionOptimizerInfo(<const Handle>handle, <const NetworkDescriptor>desc_net, &optimizer_info)
    check_status(status)
    return <intptr_t>optimizer_info


cpdef contraction_optimize(intptr_t handle, intptr_t desc_net, intptr_t optimizer_config, uint64_t workspace_size_constraint, intptr_t optimizer_info):
    """Computes an "optimized" contraction order as well as slicing info (for more information see Overview section) for a given tensor network such that the total time to solution is minimized while adhering to the user-provided memory constraint.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_net (intptr_t): Describes the topology of the tensor network (i.e., all tensors, their connectivity and modes).
        optimizer_config (intptr_t): Holds all hyper-optimization parameters that govern the search for an "optimal" contraction order.
        workspace_size_constraint (uint64_t): Maximal device memory that will be provided by the user (i.e., cuTensorNet has to find a viable path/slicing solution within this user-defined constraint).
        optimizer_info (intptr_t): On return, this object will hold all necessary information about the optimized path and the related slicing information. ``optimizer_info`` will hold information including (see ``cutensornetContractionOptimizerInfoAttributes_t``):.

    .. seealso:: `cutensornetContractionOptimize`
    """
    with nogil:
        status = cutensornetContractionOptimize(<const Handle>handle, <const NetworkDescriptor>desc_net, <const ContractionOptimizerConfig>optimizer_config, workspace_size_constraint, <ContractionOptimizerInfo>optimizer_info)
    check_status(status)


######################### Python specific utility #########################

cdef dict contraction_optimizer_info_attribute_sizes = {
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH: contraction_path_dtype,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES: _numpy.int64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT: _numpy.int64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG: slicing_config_dtype,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_OVERHEAD: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PHASE1_FLOP_COUNT: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS_EST: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_RUNTIME_EST: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_INTERMEDIATE_MODES: _numpy.int32,
}

cpdef get_contraction_optimizer_info_attribute_dtype(int attr):
    """Get the Python data type of the corresponding ContractionOptimizerInfoAttribute attribute.

    Args:
        attr (ContractionOptimizerInfoAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`contraction_optimizer_info_get_attribute`, :func:`contraction_optimizer_info_set_attribute`.
    """
    return contraction_optimizer_info_attribute_sizes[attr]

###########################################################################


cpdef contraction_optimizer_info_get_attribute(intptr_t handle, intptr_t optimizer_info, int attr, intptr_t buf, size_t size_in_bytes):
    """Gets attributes of ``optimizer_info``.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        optimizer_info (intptr_t): Opaque structure that is accessed.
        attr (ContractionOptimizerInfoAttribute): Specifies the attribute that is requested.
        buf (intptr_t): On return, this buffer (of size ``size_in_bytes``) holds the value that corresponds to ``attr`` within ``optimizeInfo``.
        size_in_bytes (size_t): Size of ``buf`` (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_contraction_optimizer_info_attribute_dtype`.

    .. seealso:: `cutensornetContractionOptimizerInfoGetAttribute`
    """
    with nogil:
        status = cutensornetContractionOptimizerInfoGetAttribute(<const Handle>handle, <const ContractionOptimizerInfo>optimizer_info, <_ContractionOptimizerInfoAttribute>attr, <void*>buf, size_in_bytes)
    check_status(status)


cpdef contraction_optimizer_info_set_attribute(intptr_t handle, intptr_t optimizer_info, int attr, intptr_t buf, size_t size_in_bytes):
    """Sets attributes of optimizer_info.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        optimizer_info (intptr_t): Opaque structure that is accessed.
        attr (ContractionOptimizerInfoAttribute): Specifies the attribute that is requested.
        buf (intptr_t): This buffer (of size ``size_in_bytes``) determines the value to which ``attr`` will be set.
        size_in_bytes (size_t): Size of ``buf`` (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_contraction_optimizer_info_attribute_dtype`.

    .. seealso:: `cutensornetContractionOptimizerInfoSetAttribute`
    """
    with nogil:
        status = cutensornetContractionOptimizerInfoSetAttribute(<const Handle>handle, <ContractionOptimizerInfo>optimizer_info, <_ContractionOptimizerInfoAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(status)


cpdef size_t contraction_optimizer_info_get_packed_size(intptr_t handle, intptr_t optimizer_info) except? 0:
    """Gets the packed size of the ``optimizer_info`` object.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        optimizer_info (intptr_t): Opaque structure of type cutensornetContractionOptimizerInfo_t.

    Returns:
        size_t: The packed size (in bytes).

    .. seealso:: `cutensornetContractionOptimizerInfoGetPackedSize`
    """
    cdef size_t size_in_bytes
    with nogil:
        status = cutensornetContractionOptimizerInfoGetPackedSize(<const Handle>handle, <const ContractionOptimizerInfo>optimizer_info, &size_in_bytes)
    check_status(status)
    return size_in_bytes


cpdef contraction_optimizer_info_pack_data(intptr_t handle, intptr_t optimizer_info, buffer, size_t size_in_bytes):
    """Packs the ``optimizer_info`` object into the provided buffer.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        optimizer_info (intptr_t): Opaque structure of type cutensornetContractionOptimizerInfo_t.
        buffer (bytes): On return, this buffer holds the contents of optimizer_info in packed form.
        size_in_bytes (size_t): The size of the buffer (in bytes).

    .. seealso:: `cutensornetContractionOptimizerInfoPackData`
    """
    cdef void* _buffer_ = get_buffer_pointer(buffer, size_in_bytes, readonly=False)
    with nogil:
        status = cutensornetContractionOptimizerInfoPackData(<const Handle>handle, <const ContractionOptimizerInfo>optimizer_info, <void*>_buffer_, size_in_bytes)
    check_status(status)


cpdef intptr_t create_contraction_optimizer_info_from_packed_data(intptr_t handle, intptr_t desc_net, buffer, size_t size_in_bytes) except? 0:
    """Create an optimizerInfo object from the provided buffer.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_net (intptr_t): Describes the tensor network (i.e., its tensors and their connectivity) for which ``optimizerInfo`` is created.
        buffer (bytes): A buffer with the contents of optimizerInfo in packed form.
        size_in_bytes (size_t): The size of the buffer (in bytes).

    Returns:
        intptr_t: Pointer to ``cutensornetContractionOptimizerInfo_t``.

    .. seealso:: `cutensornetCreateContractionOptimizerInfoFromPackedData`
    """
    cdef void* _buffer_ = get_buffer_pointer(buffer, size_in_bytes, readonly=True)
    cdef ContractionOptimizerInfo optimizer_info
    with nogil:
        status = cutensornetCreateContractionOptimizerInfoFromPackedData(<const Handle>handle, <const NetworkDescriptor>desc_net, <const void*>_buffer_, size_in_bytes, &optimizer_info)
    check_status(status)
    return <intptr_t>optimizer_info


cpdef update_contraction_optimizer_info_from_packed_data(intptr_t handle, buffer, size_t size_in_bytes, intptr_t optimizer_info):
    """Update the provided ``optimizer_info`` object from the provided buffer.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        buffer (bytes): A buffer with the contents of optimizer_info in packed form.
        size_in_bytes (size_t): The size of the buffer (in bytes).
        optimizer_info (intptr_t): Opaque object of type ``cutensornetContractionOptimizerInfo_t`` that will be updated.

    .. seealso:: `cutensornetUpdateContractionOptimizerInfoFromPackedData`
    """
    cdef void* _buffer_ = get_buffer_pointer(buffer, size_in_bytes, readonly=True)
    with nogil:
        status = cutensornetUpdateContractionOptimizerInfoFromPackedData(<const Handle>handle, <const void*>_buffer_, size_in_bytes, <ContractionOptimizerInfo>optimizer_info)
    check_status(status)


cpdef intptr_t create_contraction_plan(intptr_t handle, intptr_t desc_net, intptr_t optimizer_info, intptr_t work_desc) except? 0:
    """Initializes a ``cutensornetContractionPlan_t``.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_net (intptr_t): Describes the tensor network (i.e., its tensors and their connectivity).
        optimizer_info (intptr_t): Opaque structure.
        work_desc (intptr_t): Opaque structure describing the workspace. At the creation of the contraction plan, only the workspace size is needed; the pointer to the workspace memory may be left null. If a device memory handler is set, ``work_desc`` can be set either to null (in which case the "recommended" workspace size is inferred, see ``CUTENSORNET_WORKSIZE_PREF_RECOMMENDED``) or to a valid ``cutensornetWorkspaceDescriptor_t`` with the desired workspace size set and a null workspace pointer, see Memory Management API section.

    Returns:
        intptr_t: cuTensorNet's contraction plan holds all the information required to perform the tensor contractions; to be precise, it initializes a ``cutensorContractionPlan_t`` for each tensor contraction that is required to contract the entire tensor network.

    .. seealso:: `cutensornetCreateContractionPlan`
    """
    cdef ContractionPlan plan
    with nogil:
        status = cutensornetCreateContractionPlan(<const Handle>handle, <const NetworkDescriptor>desc_net, <const ContractionOptimizerInfo>optimizer_info, <const WorkspaceDescriptor>work_desc, &plan)
    check_status(status)
    return <intptr_t>plan


cpdef destroy_contraction_plan(intptr_t plan):
    """Frees all resources owned by ``plan``.

    Args:
        plan (intptr_t): Opaque structure.

    .. seealso:: `cutensornetDestroyContractionPlan`
    """
    with nogil:
        status = cutensornetDestroyContractionPlan(<ContractionPlan>plan)
    check_status(status)


cpdef contraction_autotune(intptr_t handle, intptr_t plan, raw_data_in, intptr_t raw_data_out, intptr_t work_desc, intptr_t pref, intptr_t stream):
    """Auto-tunes the contraction plan to find the best ``cutensorContractionPlan_t`` for each pair-wise contraction.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        plan (intptr_t): The plan must already be created (see :func:`create_contraction_plan`); the individual contraction plans will be fine-tuned.
        raw_data_in (object): Array of N pointers (N being the number of input tensors specified :func:`create_network_descriptor`); ``raw_data_in[i]`` points to the data associated with the i-th input tensor (in device memory). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        raw_data_out (intptr_t): Points to the raw data of the output tensor (in device memory).
        work_desc (intptr_t): Opaque structure describing the workspace. The provided workspace must be ``valid`` (the workspace size must be the same as or larger than both the minimum needed and the value provided at plan creation). See :func:`create_contraction_plan`, :func:`workspace_get_memory_size` & :func:`workspace_set_memory`. If a device memory handler is set, the ``work_desc`` can be set to null, or the workspace pointer in ``work_desc`` can be set to null, and the workspace size can be set either to 0 (in which case the "recommended" size is used, see ``CUTENSORNET_WORKSIZE_PREF_RECOMMENDED``) or to a ``valid`` size. A workspace of the specified size will be drawn from the user's mempool and released back once done.
        pref (intptr_t): Controls the auto-tuning process and gives the user control over how much time is spent in this routine.
        stream (intptr_t): The CUDA stream on which the computation is performed.

    .. seealso:: `cutensornetContractionAutotune`
    """
    cdef nullable_unique_ptr[ vector[void*] ] _raw_data_in_
    get_resource_ptrs[void](_raw_data_in_, raw_data_in, <void*>NULL)
    with nogil:
        status = cutensornetContractionAutotune(<const Handle>handle, <ContractionPlan>plan, <const void* const*>(_raw_data_in_.data()), <void*>raw_data_out, <WorkspaceDescriptor>work_desc, <const ContractionAutotunePreference>pref, <Stream>stream)
    check_status(status)


cpdef intptr_t create_contraction_autotune_preference(intptr_t handle) except? 0:
    """Sets up the required auto-tune parameters for the contraction plan.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.

    Returns:
        intptr_t: This data structure holds all information about the user-requested auto-tune parameters.

    .. seealso:: `cutensornetCreateContractionAutotunePreference`
    """
    cdef ContractionAutotunePreference autotune_preference
    with nogil:
        status = cutensornetCreateContractionAutotunePreference(<const Handle>handle, &autotune_preference)
    check_status(status)
    return <intptr_t>autotune_preference


######################### Python specific utility #########################

cdef dict contraction_autotune_preference_attribute_sizes = {
    CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS: _numpy.int32,
    CUTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES: _numpy.int32,
}

cpdef get_contraction_autotune_preference_attribute_dtype(int attr):
    """Get the Python data type of the corresponding ContractionAutotunePreferenceAttribute attribute.

    Args:
        attr (ContractionAutotunePreferenceAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`contraction_autotune_preference_get_attribute`, :func:`contraction_autotune_preference_set_attribute`.
    """
    return contraction_autotune_preference_attribute_sizes[attr]

###########################################################################


cpdef contraction_autotune_preference_get_attribute(intptr_t handle, intptr_t autotune_preference, int attr, intptr_t buf, size_t size_in_bytes):
    """Gets attributes of ``autotune_preference``.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        autotune_preference (intptr_t): Opaque structure that is accessed.
        attr (ContractionAutotunePreferenceAttribute): Specifies the attribute that is requested.
        buf (intptr_t): On return, this buffer (of size ``size_in_bytes``) holds the value that corresponds to ``attr`` within ``autotune_preference``.
        size_in_bytes (size_t): Size of ``buf`` (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_contraction_autotune_preference_attribute_dtype`.

    .. seealso:: `cutensornetContractionAutotunePreferenceGetAttribute`
    """
    with nogil:
        status = cutensornetContractionAutotunePreferenceGetAttribute(<const Handle>handle, <const ContractionAutotunePreference>autotune_preference, <_ContractionAutotunePreferenceAttribute>attr, <void*>buf, size_in_bytes)
    check_status(status)


cpdef contraction_autotune_preference_set_attribute(intptr_t handle, intptr_t autotune_preference, int attr, intptr_t buf, size_t size_in_bytes):
    """Sets attributes of ``autotune_preference``.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        autotune_preference (intptr_t): Opaque structure that is accessed.
        attr (ContractionAutotunePreferenceAttribute): Specifies the attribute that is requested.
        buf (intptr_t): This buffer (of size ``size_in_bytes``) determines the value to which ``attr`` will be set.
        size_in_bytes (size_t): Size of ``buf`` (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_contraction_autotune_preference_attribute_dtype`.

    .. seealso:: `cutensornetContractionAutotunePreferenceSetAttribute`
    """
    with nogil:
        status = cutensornetContractionAutotunePreferenceSetAttribute(<const Handle>handle, <ContractionAutotunePreference>autotune_preference, <_ContractionAutotunePreferenceAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(status)


cpdef destroy_contraction_autotune_preference(intptr_t autotune_preference):
    """Frees all the memory associated with ``autotune_preference``.

    Args:
        autotune_preference (intptr_t): Opaque structure.

    .. seealso:: `cutensornetDestroyContractionAutotunePreference`
    """
    with nogil:
        status = cutensornetDestroyContractionAutotunePreference(<ContractionAutotunePreference>autotune_preference)
    check_status(status)


cpdef intptr_t create_slice_group_from_id_range(intptr_t handle, int64_t slice_id_start, int64_t slice_id_stop, int64_t slice_id_step) except? 0:
    """Create a ``cutensornetSliceGroup_t`` object from a range, which produces a sequence of slice IDs from the specified start (inclusive) to the specified stop (exclusive) values with the specified step. The sequence can be increasing or decreasing depending on the start and stop values.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        slice_id_start (int64_t): The start slice ID.
        slice_id_stop (int64_t): The final slice ID is the largest (smallest) integer that excludes this value and all those above (below) for an increasing (decreasing) sequence.
        slice_id_step (int64_t): The step size between two successive slice IDs. A negative step size should be specified for a decreasing sequence.

    Returns:
        intptr_t: Opaque object specifying the slice IDs.

    .. seealso:: `cutensornetCreateSliceGroupFromIDRange`
    """
    cdef SliceGroup slice_group
    with nogil:
        status = cutensornetCreateSliceGroupFromIDRange(<const Handle>handle, slice_id_start, slice_id_stop, slice_id_step, &slice_group)
    check_status(status)
    return <intptr_t>slice_group


cpdef destroy_slice_group(intptr_t slice_group):
    """Releases the resources associated with a ``cutensornetSliceGroup_t`` object and sets its value to null.

    Args:
        slice_group (intptr_t): Opaque object specifying the slices to be contracted (see :func:`create_slice_group_from_id_range` and ``cutensornetCreateSliceGroupFromIDs()``).

    .. seealso:: `cutensornetDestroySliceGroup`
    """
    with nogil:
        status = cutensornetDestroySliceGroup(<SliceGroup>slice_group)
    check_status(status)


cpdef contract_slices(intptr_t handle, intptr_t plan, raw_data_in, intptr_t raw_data_out, int32_t accumulate_output, intptr_t work_desc, intptr_t slice_group, intptr_t stream):
    """Performs the actual contraction of the tensor network.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        plan (intptr_t): Encodes the execution of a tensor network contraction (see :func:`create_contraction_plan` and :func:`contraction_autotune`). Some internal meta-data may be updated upon contraction.
        raw_data_in (object): Array of N pointers (N being the number of input tensors specified in :func:`create_network_descriptor`): ``raw_data_in[i]`` points to the data associated with the i-th input tensor (in device memory). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        raw_data_out (intptr_t): Points to the raw data of the output tensor (in device memory).
        accumulate_output (int32_t): If 0, write the contraction result into raw_data_out; otherwise accumulate the result into raw_data_out.
        work_desc (intptr_t): Opaque structure describing the workspace. The provided ``CUTENSORNET_WORKSPACE_SCRATCH`` workspace must be ``valid`` (the workspace pointer must be device accessible, see ``cutensornetMemspace_t``, and the workspace size must be the same as or larger than both the minimum needed and the value provided at plan creation). See :func:`create_contraction_plan`, :func:`workspace_get_memory_size` & :func:`workspace_set_memory`. The provided ``CUTENSORNET_WORKSPACE_CACHE`` workspace must be device accessible, see ``cutensornetMemspace_t``; it can be of any size, the larger the better, up to the size that can be queried with :func:`workspace_get_memory_size`. If a device memory handler is set, then ``work_desc`` can be set to null, or the memory pointer in ``work_desc`` of either the workspace kinds can be set to null, and the workspace size can be set either to a negative value (in which case the "recommended" size is used, see ``CUTENSORNET_WORKSIZE_PREF_RECOMMENDED``) or to a ``valid`` size. For a workspace of kind ``CUTENSORNET_WORKSPACE_SCRATCH``, a memory buffer with the specified size will be drawn from the user's mempool and released back once done. For a workspace of kind ``CUTENSORNET_WORKSPACE_CACHE``, a memory buffer with the specified size will be drawn from the user's mempool and released back once the ``work_desc`` is destroyed, if ``work_desc`` != NULL, otherwise, once the ``plan`` is destroyed, or an alternative ``work_desc`` with a different memory address/size is provided in a subsequent :func:`contract_slices` call.
        slice_group (intptr_t): Opaque object specifying the slices to be contracted (see :func:`create_slice_group_from_id_range` and ``cutensornetCreateSliceGroupFromIDs()``). ``If set to null, all slices will be contracted.``.
        stream (intptr_t): The CUDA stream on which the computation is performed.

    .. seealso:: `cutensornetContractSlices`
    """
    cdef nullable_unique_ptr[ vector[void*] ] _raw_data_in_
    get_resource_ptrs[void](_raw_data_in_, raw_data_in, <void*>NULL)
    with nogil:
        status = cutensornetContractSlices(<const Handle>handle, <ContractionPlan>plan, <const void* const*>(_raw_data_in_.data()), <void*>raw_data_out, accumulate_output, <WorkspaceDescriptor>work_desc, <const SliceGroup>slice_group, <Stream>stream)
    check_status(status)


cpdef intptr_t create_tensor_descriptor(intptr_t handle, int32_t num_modes, extents, strides, modes, int data_type) except? 0:
    """Initializes a ``cutensornetTensorDescriptor_t``, describing the information of a tensor.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        num_modes (int32_t): The number of modes of the tensor.
        extents (object): Array of size ``num_modes``; ``extents[j]`` corresponding to the extent of the j-th mode of the tensor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        strides (object): Array of size ``num_modes``; ``strides[j]`` corresponding to the linearized offset -- in physical memory -- between two logically-neighboring elements w.r.t the j-th mode of the tensor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        modes (object): Array of size ``num_modes``; ``modes[j]`` denotes the j-th mode of the tensor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        data_type (int): Denotes the data type for the tensor.

    Returns:
        intptr_t: Pointer to a ``cutensornetTensorDescriptor_t``.

    .. seealso:: `cutensornetCreateTensorDescriptor`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _extents_
    get_resource_ptr[int64_t](_extents_, extents, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _strides_
    get_resource_ptr[int64_t](_strides_, strides, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _modes_
    get_resource_ptr[int32_t](_modes_, modes, <int32_t*>NULL)
    cdef TensorDescriptor desc_tensor
    with nogil:
        status = cutensornetCreateTensorDescriptor(<const Handle>handle, num_modes, <const int64_t*>(_extents_.data()), <const int64_t*>(_strides_.data()), <const int32_t*>(_modes_.data()), <DataType>data_type, &desc_tensor)
    check_status(status)
    return <intptr_t>desc_tensor


cpdef destroy_tensor_descriptor(intptr_t desc_tensor):
    """Frees all the memory associated with the tensor descriptor.

    Args:
        desc_tensor (intptr_t): Opaque handle to a tensor descriptor.

    .. seealso:: `cutensornetDestroyTensorDescriptor`
    """
    with nogil:
        status = cutensornetDestroyTensorDescriptor(<TensorDescriptor>desc_tensor)
    check_status(status)


cpdef intptr_t create_tensor_svd_config(intptr_t handle) except? 0:
    """Sets up the options for singular value decomposition and truncation.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.

    Returns:
        intptr_t: This data structure holds the user-requested svd parameters.

    .. seealso:: `cutensornetCreateTensorSVDConfig`
    """
    cdef TensorSVDConfig svd_config
    with nogil:
        status = cutensornetCreateTensorSVDConfig(<const Handle>handle, &svd_config)
    check_status(status)
    return <intptr_t>svd_config


cpdef destroy_tensor_svd_config(intptr_t svd_config):
    """Frees all the memory associated with the tensor svd configuration.

    Args:
        svd_config (intptr_t): Opaque handle to a tensor svd configuration.

    .. seealso:: `cutensornetDestroyTensorSVDConfig`
    """
    with nogil:
        status = cutensornetDestroyTensorSVDConfig(<TensorSVDConfig>svd_config)
    check_status(status)


######################### Python specific utility #########################

cdef dict tensor_svd_config_attribute_sizes = {
    CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF: _numpy.float64,
    CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF: _numpy.float64,
    CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION: _numpy.int32,
    CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION: _numpy.int32,
    CUTENSORNET_TENSOR_SVD_CONFIG_ALGO: _numpy.int32,
    CUTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF: _numpy.float64,
}

cpdef get_tensor_svd_config_attribute_dtype(int attr):
    """Get the Python data type of the corresponding TensorSVDConfigAttribute attribute.

    Args:
        attr (TensorSVDConfigAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`tensor_svd_config_get_attribute`, :func:`tensor_svd_config_set_attribute`.
    """
    if attr == CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS:
        raise ValueError('use tensor_svd_algo_params_get_dtype to get the dtype')
    return tensor_svd_config_attribute_sizes[attr]

###########################################################################


cpdef tensor_svd_config_get_attribute(intptr_t handle, intptr_t svd_config, int attr, intptr_t buf, size_t size_in_bytes):
    """Gets attributes of ``svd_config``.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        svd_config (intptr_t): Opaque structure that is accessed.
        attr (TensorSVDConfigAttribute): Specifies the attribute that is requested.
        buf (intptr_t): On return, this buffer (of size ``size_in_bytes``) holds the value that corresponds to ``attr`` within ``svd_config``.
        size_in_bytes (size_t): Size of ``buf`` (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_tensor_svd_config_attribute_dtype`.

    .. seealso:: `cutensornetTensorSVDConfigGetAttribute`
    """
    with nogil:
        status = cutensornetTensorSVDConfigGetAttribute(<const Handle>handle, <const TensorSVDConfig>svd_config, <_TensorSVDConfigAttribute>attr, <void*>buf, size_in_bytes)
    check_status(status)


cpdef tensor_svd_config_set_attribute(intptr_t handle, intptr_t svd_config, int attr, intptr_t buf, size_t size_in_bytes):
    """Sets attributes of ``svd_config``.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        svd_config (intptr_t): Opaque structure that is accessed.
        attr (TensorSVDConfigAttribute): Specifies the attribute that is requested.
        buf (intptr_t): This buffer (of size ``size_in_bytes``) determines the value to which ``attr`` will be set.
        size_in_bytes (size_t): Size of ``buf`` (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_tensor_svd_config_attribute_dtype`.

    .. seealso:: `cutensornetTensorSVDConfigSetAttribute`
    """
    with nogil:
        status = cutensornetTensorSVDConfigSetAttribute(<const Handle>handle, <TensorSVDConfig>svd_config, <_TensorSVDConfigAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(status)


cpdef workspace_compute_svd_sizes(intptr_t handle, intptr_t desc_tensor_in, intptr_t desc_tensor_u, intptr_t desc_tensor_v, intptr_t svd_config, intptr_t work_desc):
    """Computes the workspace size needed to perform the tensor SVD operation.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_tensor_in (intptr_t): Describes the modes, extents and other metadata information for a tensor.
        desc_tensor_u (intptr_t): Describes the modes, extents and other metadata information for the output tensor U.
        desc_tensor_v (intptr_t): Describes the modes, extents and other metadata information for the output tensor V.
        svd_config (intptr_t): This data structure holds the user-requested svd parameters.
        work_desc (intptr_t): The workspace descriptor in which the information is collected.

    .. seealso:: `cutensornetWorkspaceComputeSVDSizes`
    """
    with nogil:
        status = cutensornetWorkspaceComputeSVDSizes(<const Handle>handle, <const TensorDescriptor>desc_tensor_in, <const TensorDescriptor>desc_tensor_u, <const TensorDescriptor>desc_tensor_v, <const TensorSVDConfig>svd_config, <WorkspaceDescriptor>work_desc)
    check_status(status)


cpdef workspace_compute_qr_sizes(intptr_t handle, intptr_t desc_tensor_in, intptr_t desc_tensor_q, intptr_t desc_tensor_r, intptr_t work_desc):
    """Computes the workspace size needed to perform the tensor QR operation.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_tensor_in (intptr_t): Describes the modes, extents and other metadata information for a tensor.
        desc_tensor_q (intptr_t): Describes the modes, extents and other metadata information for the output tensor Q.
        desc_tensor_r (intptr_t): Describes the modes, extents and other metadata information for the output tensor R.
        work_desc (intptr_t): The workspace descriptor in which the information is collected.

    .. seealso:: `cutensornetWorkspaceComputeQRSizes`
    """
    with nogil:
        status = cutensornetWorkspaceComputeQRSizes(<const Handle>handle, <const TensorDescriptor>desc_tensor_in, <const TensorDescriptor>desc_tensor_q, <const TensorDescriptor>desc_tensor_r, <WorkspaceDescriptor>work_desc)
    check_status(status)


cpdef intptr_t create_tensor_svd_info(intptr_t handle) except? 0:
    """Sets up the information for singular value decomposition.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.

    Returns:
        intptr_t: This data structure holds all information about the trucation at runtime.

    .. seealso:: `cutensornetCreateTensorSVDInfo`
    """
    cdef TensorSVDInfo svd_info
    with nogil:
        status = cutensornetCreateTensorSVDInfo(<const Handle>handle, &svd_info)
    check_status(status)
    return <intptr_t>svd_info


######################### Python specific utility #########################

cdef dict tensor_svd_info_attribute_sizes = {
    CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT: _numpy.int64,
    CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT: _numpy.int64,
    CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT: _numpy.float64,
    CUTENSORNET_TENSOR_SVD_INFO_ALGO: _numpy.int32,
}

cpdef get_tensor_svd_info_attribute_dtype(int attr):
    """Get the Python data type of the corresponding TensorSVDInfoAttribute attribute.

    Args:
        attr (TensorSVDInfoAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`tensor_svd_info_get_attribute`.
    """
    if attr == CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS:
        raise ValueError('use tensor_svd_algo_status_get_dtype to get the dtype')
    return tensor_svd_info_attribute_sizes[attr]

###########################################################################


cpdef tensor_svd_info_get_attribute(intptr_t handle, intptr_t svd_info, int attr, intptr_t buf, size_t size_in_bytes):
    """Gets attributes of ``svd_info``.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        svd_info (intptr_t): Opaque structure that is accessed.
        attr (TensorSVDInfoAttribute): Specifies the attribute that is requested.
        buf (intptr_t): On return, this buffer (of size ``size_in_bytes``) holds the value that corresponds to ``attr`` within ``svdConfig``.
        size_in_bytes (size_t): Size of ``buf`` (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_tensor_svd_info_attribute_dtype`.

    .. seealso:: `cutensornetTensorSVDInfoGetAttribute`
    """
    with nogil:
        status = cutensornetTensorSVDInfoGetAttribute(<const Handle>handle, <const TensorSVDInfo>svd_info, <_TensorSVDInfoAttribute>attr, <void*>buf, size_in_bytes)
    check_status(status)


cpdef destroy_tensor_svd_info(intptr_t svd_info):
    """Frees all the memory associated with the TensorSVDInfo object.

    Args:
        svd_info (intptr_t): Opaque handle to a TensorSVDInfo object.

    .. seealso:: `cutensornetDestroyTensorSVDInfo`
    """
    with nogil:
        status = cutensornetDestroyTensorSVDInfo(<TensorSVDInfo>svd_info)
    check_status(status)


cpdef tensor_svd(intptr_t handle, intptr_t desc_tensor_in, intptr_t raw_data_in, intptr_t desc_tensor_u, intptr_t u, intptr_t s, intptr_t desc_tensor_v, intptr_t v, intptr_t svd_config, intptr_t svd_info, intptr_t work_desc, intptr_t stream):
    """Performs SVD decomposition of a tensor.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_tensor_in (intptr_t): Describes the modes, extents, and other metadata information of a tensor.
        raw_data_in (intptr_t): Pointer to the raw data of the input tensor (in device memory).
        desc_tensor_u (intptr_t): Describes the modes, extents, and other metadata information of the output tensor U. The extents for uncontracted modes are expected to be consistent with ``desc_tensor_in``.
        u (intptr_t): Pointer to the output tensor data U (in device memory).
        s (intptr_t): Pointer to the output tensor data S (in device memory). Can be ``NULL`` when the ``CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION`` attribute of ``svd_config`` is not set to default (::CUTENSORNET_TENSOR_SVD_PARTITION_NONE).
        desc_tensor_v (intptr_t): Describes the modes, extents, and other metadata information of the output tensor V.
        v (intptr_t): Pointer to the output tensor data V (in device memory).
        svd_config (intptr_t): This data structure holds the user-requested SVD parameters. Can be ``NULL`` if users do not need to perform value-based truncation or singular value partitioning.
        svd_info (intptr_t): Opaque structure holding all information about the trucation at runtime. Can be ``NULL`` if runtime information on singular value truncation is not needed.
        work_desc (intptr_t): Opaque structure describing the workspace. The provided workspace must be ``valid`` (the workspace size must be the same as or larger than the minimum needed). See :func:`workspace_get_memory_size` & :func:`workspace_set_memory`.
        stream (intptr_t): The CUDA stream on which the computation is performed.

    .. seealso:: `cutensornetTensorSVD`
    """
    with nogil:
        status = cutensornetTensorSVD(<const Handle>handle, <const TensorDescriptor>desc_tensor_in, <const void* const>raw_data_in, <TensorDescriptor>desc_tensor_u, <void*>u, <void*>s, <TensorDescriptor>desc_tensor_v, <void*>v, <const TensorSVDConfig>svd_config, <TensorSVDInfo>svd_info, <const WorkspaceDescriptor>work_desc, <Stream>stream)
    check_status(status)


cpdef tensor_qr(intptr_t handle, intptr_t desc_tensor_in, intptr_t raw_data_in, intptr_t desc_tensor_q, intptr_t q, intptr_t desc_tensor_r, intptr_t r, intptr_t work_desc, intptr_t stream):
    """Performs QR decomposition of a tensor.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_tensor_in (intptr_t): Describes the modes, extents, and other metadata information of a tensor.
        raw_data_in (intptr_t): Pointer to the raw data of the input tensor (in device memory).
        desc_tensor_q (intptr_t): Describes the modes, extents, and other metadata information of the output tensor Q.
        q (intptr_t): Pointer to the output tensor data Q (in device memory).
        desc_tensor_r (intptr_t): Describes the modes, extents, and other metadata information of the output tensor R.
        r (intptr_t): Pointer to the output tensor data R (in device memory).
        work_desc (intptr_t): Opaque structure describing the workspace. The provided workspace must be ``valid`` (the workspace size must be the same as or larger than the minimum needed). See :func:`workspace_get_memory_size` & :func:`workspace_set_memory`.
        stream (intptr_t): The CUDA stream on which the computation is performed.

    .. seealso:: `cutensornetTensorQR`
    """
    with nogil:
        status = cutensornetTensorQR(<const Handle>handle, <const TensorDescriptor>desc_tensor_in, <const void* const>raw_data_in, <const TensorDescriptor>desc_tensor_q, <void*>q, <const TensorDescriptor>desc_tensor_r, <void*>r, <const WorkspaceDescriptor>work_desc, <Stream>stream)
    check_status(status)


cpdef workspace_compute_gate_split_sizes(intptr_t handle, intptr_t desc_tensor_in_a, intptr_t desc_tensor_in_b, intptr_t desc_tensor_in_g, intptr_t desc_tensor_u, intptr_t desc_tensor_v, int gate_algo, intptr_t svd_config, int compute_type, intptr_t work_desc):
    """Computes the workspace size needed to perform the gating operation.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_tensor_in_a (intptr_t): Describes the modes, extents, and other metadata information of the input tensor A.
        desc_tensor_in_b (intptr_t): Describes the modes, extents, and other metadata information of the input tensor B.
        desc_tensor_in_g (intptr_t): Describes the modes, extents, and other metadata information of the input gate tensor.
        desc_tensor_u (intptr_t): Describes the modes, extents, and other metadata information of the output U tensor. The extents of uncontracted modes are expected to be consistent with ``desc_tensor_in_a`` and ``desc_tensor_in_g``.
        desc_tensor_v (intptr_t): Describes the modes, extents, and other metadata information of the output V tensor. The extents of uncontracted modes are expected to be consistent with ``desc_tensor_in_b`` and ``desc_tensor_in_g``.
        gate_algo (int): The algorithm to use for splitting the gate tensor onto tensor A and B.
        svd_config (intptr_t): Opaque structure holding the user-requested SVD parameters.
        compute_type (ComputeType): Denotes the compute type used throughout the computation.
        work_desc (intptr_t): Opaque structure describing the workspace.

    .. seealso:: `cutensornetWorkspaceComputeGateSplitSizes`
    """
    with nogil:
        status = cutensornetWorkspaceComputeGateSplitSizes(<const Handle>handle, <const TensorDescriptor>desc_tensor_in_a, <const TensorDescriptor>desc_tensor_in_b, <const TensorDescriptor>desc_tensor_in_g, <const TensorDescriptor>desc_tensor_u, <const TensorDescriptor>desc_tensor_v, <const _GateSplitAlgo>gate_algo, <const TensorSVDConfig>svd_config, <_ComputeType>compute_type, <WorkspaceDescriptor>work_desc)
    check_status(status)


cpdef gate_split(intptr_t handle, intptr_t desc_tensor_in_a, intptr_t raw_data_in_a, intptr_t desc_tensor_in_b, intptr_t raw_data_in_b, intptr_t desc_tensor_in_g, intptr_t raw_data_in_g, intptr_t desc_tensor_u, intptr_t u, intptr_t s, intptr_t desc_tensor_v, intptr_t v, int gate_algo, intptr_t svd_config, int compute_type, intptr_t svd_info, intptr_t work_desc, intptr_t stream):
    """Performs gate split operation.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        desc_tensor_in_a (intptr_t): Describes the modes, extents, and other metadata information of the input tensor A.
        raw_data_in_a (intptr_t): Pointer to the raw data of the input tensor A (in device memory).
        desc_tensor_in_b (intptr_t): Describes the modes, extents, and other metadata information of the input tensor B.
        raw_data_in_b (intptr_t): Pointer to the raw data of the input tensor B (in device memory).
        desc_tensor_in_g (intptr_t): Describes the modes, extents, and other metadata information of the input gate tensor.
        raw_data_in_g (intptr_t): Pointer to the raw data of the input gate tensor G (in device memory).
        desc_tensor_u (intptr_t): Describes the modes, extents, and other metadata information of the output U tensor. The extents of uncontracted modes are expected to be consistent with ``desc_tensor_in_a`` and ``desc_tensor_in_g``.
        u (intptr_t): Pointer to the output tensor data U (in device memory).
        s (intptr_t): Pointer to the output tensor data S (in device memory). Can be ``NULL`` when the ``CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION`` attribute of ``svd_config`` is not set to default (::CUTENSORNET_TENSOR_SVD_PARTITION_NONE).
        desc_tensor_v (intptr_t): Describes the modes, extents, and other metadata information of the output V tensor. The extents of uncontracted modes are expected to be consistent with ``desc_tensor_in_b`` and ``desc_tensor_in_g``.
        v (intptr_t): Pointer to the output tensor data V (in device memory).
        gate_algo (int): The algorithm to use for splitting the gate tensor into tensor A and B.
        svd_config (intptr_t): Opaque structure holding the user-requested SVD parameters.
        compute_type (ComputeType): Denotes the compute type used throughout the computation.
        svd_info (intptr_t): Opaque structure holding all information about the truncation at runtime.
        work_desc (intptr_t): Opaque structure describing the workspace.
        stream (intptr_t): The CUDA stream on which the computation is performed.

    .. seealso:: `cutensornetGateSplit`
    """
    with nogil:
        status = cutensornetGateSplit(<const Handle>handle, <const TensorDescriptor>desc_tensor_in_a, <const void*>raw_data_in_a, <const TensorDescriptor>desc_tensor_in_b, <const void*>raw_data_in_b, <const TensorDescriptor>desc_tensor_in_g, <const void*>raw_data_in_g, <TensorDescriptor>desc_tensor_u, <void*>u, <void*>s, <TensorDescriptor>desc_tensor_v, <void*>v, <const _GateSplitAlgo>gate_algo, <const TensorSVDConfig>svd_config, <_ComputeType>compute_type, <TensorSVDInfo>svd_info, <const WorkspaceDescriptor>work_desc, <Stream>stream)
    check_status(status)


cpdef logger_set_file(intptr_t file):
    """This function sets the logging output file.

    Args:
        file (intptr_t): An open file with write permission.

    .. seealso:: `cutensornetLoggerSetFile`
    """
    with nogil:
        status = cutensornetLoggerSetFile(<FILE*>file)
    check_status(status)


cpdef logger_open_file(log_file):
    """This function opens a logging output file in the given path.

    Args:
        log_file (str): Path to the logging output file.

    .. seealso:: `cutensornetLoggerOpenFile`
    """
    if not isinstance(log_file, str):
        raise TypeError("log_file must be a Python str")
    cdef bytes _temp_log_file_ = (<str>log_file).encode()
    cdef char* _log_file_ = _temp_log_file_
    with nogil:
        status = cutensornetLoggerOpenFile(<const char*>_log_file_)
    check_status(status)


cpdef logger_set_level(int32_t level):
    """This function sets the value of the logging level.

    Args:
        level (int32_t): Log level, should be one of the following:.

    .. seealso:: `cutensornetLoggerSetLevel`
    """
    with nogil:
        status = cutensornetLoggerSetLevel(level)
    check_status(status)


cpdef logger_set_mask(int32_t mask):
    """This function sets the value of the log mask.

    Args:
        mask (int32_t): Value of the logging mask. Masks are defined as a combination (bitwise OR) of the following masks:.

    .. seealso:: `cutensornetLoggerSetMask`
    """
    with nogil:
        status = cutensornetLoggerSetMask(mask)
    check_status(status)


cpdef logger_force_disable():
    """This function disables logging for the entire run.

    .. seealso:: `cutensornetLoggerForceDisable`
    """
    with nogil:
        status = cutensornetLoggerForceDisable()
    check_status(status)


cpdef size_t get_version() except? 0:
    """Returns Version number of the cuTensorNet library.

    .. seealso:: `cutensornetGetVersion`
    """
    return cutensornetGetVersion()


cpdef size_t get_cudart_version() except? 0:
    """Returns version number of the CUDA runtime that cuTensorNet was compiled against.

    .. seealso:: `cutensornetGetCudartVersion`
    """
    return cutensornetGetCudartVersion()


cpdef str get_error_string(int error):
    """Returns the description string for an error code.

    Args:
        error (Status): Error code to convert to string.

    .. seealso:: `cutensornetGetErrorString`
    """
    cdef bytes _output_
    _output_ = cutensornetGetErrorString(<_Status>error)
    return _output_.decode()


cpdef distributed_reset_configuration(intptr_t handle, intptr_t comm_ptr, size_t comm_size):
    """Resets the distributed MPI parallelization configuration.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        comm_ptr (intptr_t): A pointer to the provided MPI communicator created by MPI_Comm_dup.
        comm_size (size_t): The size of the provided MPI communicator: sizeof(MPI_Comm).

    .. seealso:: `cutensornetDistributedResetConfiguration`
    """
    with nogil:
        status = cutensornetDistributedResetConfiguration(<Handle>handle, <const void*>comm_ptr, comm_size)
    check_status(status)


cpdef int32_t distributed_get_num_ranks(intptr_t handle) except? -1:
    """Queries the number of MPI ranks in the current distributed MPI configuration.

    Args:
        handle (intptr_t): cuTensorNet library handle.

    Returns:
        int32_t: Number of MPI ranks in the current distributed MPI configuration.

    .. seealso:: `cutensornetDistributedGetNumRanks`
    """
    cdef int32_t num_ranks
    with nogil:
        status = cutensornetDistributedGetNumRanks(<const Handle>handle, &num_ranks)
    check_status(status)
    return num_ranks


cpdef int32_t distributed_get_proc_rank(intptr_t handle) except? -1:
    """Queries the rank of the current MPI process in the current distributed MPI configuration.

    Args:
        handle (intptr_t): cuTensorNet library handle.

    Returns:
        int32_t: Rank of the current MPI process in the current distributed MPI configuration.

    .. seealso:: `cutensornetDistributedGetProcRank`
    """
    cdef int32_t proc_rank
    with nogil:
        status = cutensornetDistributedGetProcRank(<const Handle>handle, &proc_rank)
    check_status(status)
    return proc_rank


cpdef distributed_synchronize(intptr_t handle):
    """Globally synchronizes all MPI processes in the current distributed MPI configuration, ensuring that all preceding cutensornet API calls have completed across all MPI processes.

    Args:
        handle (intptr_t): cuTensorNet library handle.

    .. seealso:: `cutensornetDistributedSynchronize`
    """
    with nogil:
        status = cutensornetDistributedSynchronize(<const Handle>handle)
    check_status(status)


######################### Python specific utility #########################

cdef dict network_attribute_sizes = {
    CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONSTANT: _numpy.int32,
    CUTENSORNET_NETWORK_INPUT_TENSORS_CONSTANT: tensor_id_list_dtype,
    CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONJUGATED: _numpy.int32,
    CUTENSORNET_NETWORK_INPUT_TENSORS_CONJUGATED: tensor_id_list_dtype,
    CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_REQUIRE_GRAD: _numpy.int32,
    CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD: tensor_id_list_dtype,
}

cpdef get_network_attribute_dtype(int attr):
    """Get the Python data type of the corresponding NetworkAttribute attribute.

    Args:
        attr (NetworkAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`network_get_attribute`, :func:`network_set_attribute`.
    """
    return network_attribute_sizes[attr]

###########################################################################


cpdef network_get_attribute(intptr_t handle, intptr_t network_desc, int attr, intptr_t buf, size_t size_in_bytes):
    """Gets attributes of network_descriptor.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        network_desc (intptr_t): Opaque structure that is accessed.
        attr (NetworkAttribute): Specifies the attribute that is requested.
        buf (intptr_t): On return, this buffer (of size ``size_in_bytes``) holds the value that corresponds to ``attr`` within ``network_desc``.
        size_in_bytes (size_t): Size of ``buf`` (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_network_attribute_dtype`.

    .. seealso:: `cutensornetNetworkGetAttribute`
    """
    with nogil:
        status = cutensornetNetworkGetAttribute(<const Handle>handle, <const NetworkDescriptor>network_desc, <_NetworkAttribute>attr, <void*>buf, size_in_bytes)
    check_status(status)


cpdef network_set_attribute(intptr_t handle, intptr_t network_desc, int attr, intptr_t buf, size_t size_in_bytes):
    """Sets attributes of network_descriptor.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        network_desc (intptr_t): Opaque structure that is accessed.
        attr (NetworkAttribute): Specifies the attribute that is requested.
        buf (intptr_t): This buffer (of size ``size_in_bytes``) determines the value to which ``attr`` will be set.
        size_in_bytes (size_t): Size of ``buf`` (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_network_attribute_dtype`.

    .. seealso:: `cutensornetNetworkSetAttribute`
    """
    with nogil:
        status = cutensornetNetworkSetAttribute(<const Handle>handle, <NetworkDescriptor>network_desc, <_NetworkAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(status)


cpdef workspace_purge_cache(intptr_t handle, intptr_t work_desc, int mem_space):
    """Purges the cached data in the specified memory space.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        work_desc (intptr_t): Opaque structure describing the workspace.
        mem_space (Memspace): The memory space where the workspace is allocated.

    .. seealso:: `cutensornetWorkspacePurgeCache`
    """
    with nogil:
        status = cutensornetWorkspacePurgeCache(<const Handle>handle, <WorkspaceDescriptor>work_desc, <_Memspace>mem_space)
    check_status(status)


cpdef compute_gradients_backward(intptr_t handle, intptr_t plan, raw_data_in, intptr_t output_gradient, gradients, int32_t accumulate_output, intptr_t work_desc, intptr_t stream):
    """Computes the gradients of the network w.r.t. the input tensors whose gradients are required. The network must have been contracted and loaded in the ``work_desc`` CACHE. Operates only on networks with single slice and no singleton modes.

    Args:
        handle (intptr_t): Opaque handle holding cuTensorNet's library context.
        plan (intptr_t): Encodes the execution of a tensor network contraction (see :func:`create_contraction_plan` and :func:`contraction_autotune`). Some internal meta-data may be updated upon contraction.
        raw_data_in (object): Array of N pointers (N being the number of input tensors specified in :func:`create_network_descriptor`): ``raw_data_in[i]`` points to the data associated with the i-th input tensor (in device memory). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        output_gradient (intptr_t): Gradient of the output tensor (in device memory). Must have the same memory layout (strides) as the output tensor of the tensor network.
        gradients (object): Array of N pointers: ``gradients[i]`` points to the gradient data associated with the i-th input tensor in device memory. Setting ``gradients[i]`` to null would skip computing the gradient of the i-th input tensor. Generated gradient data has the same memory layout (strides) as their corresponding input tensors. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        accumulate_output (int32_t): If 0, write the gradient results into ``gradients``; otherwise accumulates the results into ``gradients``.
        work_desc (intptr_t): Opaque structure describing the workspace. The provided ``CUTENSORNET_WORKSPACE_SCRATCH`` workspace must be ``valid`` (the workspace pointer must be device accessible, see ``cutensornetMemspace_t``, and the workspace size must be the same as or larger than the minimum needed). See :func:`workspace_compute_contraction_sizes`, :func:`workspace_get_memory_size` & :func:`workspace_set_memory`. The provided ``CUTENSORNET_WORKSPACE_CACHE`` workspace must be ``valid`` (the workspace pointer must be device accessible, see ``cutensornetMemspace_t``), and contains the cached intermediate tensors from the corresponding :func:`contract_slices` call. If a device memory handler is set, and ``work_desc`` is set to null, or the memory pointer in ``work_desc`` of either the workspace kinds is set to null, for both calls to :func:`contract_slices` and :func:`compute_gradients_backward`, memory will be drawn from the memory pool. See :func:`contract_slices` for details.
        stream (intptr_t): The CUDA stream on which the computation is performed.

    .. seealso:: `cutensornetComputeGradientsBackward`
    """
    cdef nullable_unique_ptr[ vector[void*] ] _raw_data_in_
    get_resource_ptrs[void](_raw_data_in_, raw_data_in, <void*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _gradients_
    get_resource_ptrs[void](_gradients_, gradients, <void*>NULL)
    with nogil:
        status = cutensornetComputeGradientsBackward(<const Handle>handle, <ContractionPlan>plan, <const void* const*>(_raw_data_in_.data()), <const void*>output_gradient, <void* const*>(_gradients_.data()), accumulate_output, <WorkspaceDescriptor>work_desc, <Stream>stream)
    check_status(status)


cpdef intptr_t create_state(intptr_t handle, int purity, int32_t num_state_modes, state_mode_extents, int data_type) except? 0:
    """Creates an empty tensor network state of a given shape defined by the number of primary tensor modes and their extents.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        purity (StatePurity): Desired purity of the tensor network state (pure or mixed).
        num_state_modes (int32_t): Number of the defining state modes, irrespective of state purity. Note that both pure and mixed tensor network states are defined solely by the modes of the primary direct-product space.
        state_mode_extents (object): Pointer to the extents of the defining state modes (dimensions of the vector spaces constituting the primary direct-product space). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        data_type (int): Data type of the state tensor.

    Returns:
        intptr_t: Tensor network state (empty at this point, aka vacuum).

    .. seealso:: `cutensornetCreateState`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _state_mode_extents_
    get_resource_ptr[int64_t](_state_mode_extents_, state_mode_extents, <int64_t*>NULL)
    cdef State tensor_network_state
    with nogil:
        status = cutensornetCreateState(<const Handle>handle, <_StatePurity>purity, num_state_modes, <const int64_t*>(_state_mode_extents_.data()), <DataType>data_type, &tensor_network_state)
    check_status(status)
    return <intptr_t>tensor_network_state


cpdef int64_t state_apply_tensor(intptr_t handle, intptr_t tensor_network_state, int32_t num_state_modes, state_modes, intptr_t tensor_data, tensor_mode_strides, int32_t immutable, int32_t adjoint, int32_t unitary) except? -1:
    """DEPRECATED: Applies a tensor operator to the tensor network state.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        num_state_modes (int32_t): Number of state modes the tensor operator acts on.
        state_modes (object): Pointer to the state modes the tensor operator acts on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        tensor_data (intptr_t): Elements of the tensor operator (must be of the same data type as the elements of the state tensor).
        tensor_mode_strides (object): Strides of the tensor operator data layout (note that the tensor operator has twice more modes than the number of state modes it acts on). Passing NULL will assume the default generalized columnwise layout. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        immutable (int32_t): Whether or not the tensor operator data may change during the lifetime of the tensor network state. Any data change must be registered via a call to ``cutensornetStateUpdateTensorOperator``.
        adjoint (int32_t): Whether or not the tensor operator is applied as an adjoint (ket and bra modes reversed, with all tensor elements complex conjugated).
        unitary (int32_t): Whether or not the tensor operator is unitary with respect to the first and second halves of its modes.

    Returns:
        int64_t: Unique integer id (for later identification of the tensor operator).

    .. seealso:: `cutensornetStateApplyTensor`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_modes_
    get_resource_ptr[int32_t](_state_modes_, state_modes, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _tensor_mode_strides_
    get_resource_ptr[int64_t](_tensor_mode_strides_, tensor_mode_strides, <int64_t*>NULL)
    cdef int64_t tensor_id
    with nogil:
        status = cutensornetStateApplyTensor(<const Handle>handle, <State>tensor_network_state, num_state_modes, <const int32_t*>(_state_modes_.data()), <void*>tensor_data, <const int64_t*>(_tensor_mode_strides_.data()), <const int32_t>immutable, <const int32_t>adjoint, <const int32_t>unitary, &tensor_id)
    check_status(status)
    return tensor_id


cpdef state_update_tensor(intptr_t handle, intptr_t tensor_network_state, int64_t tensor_id, intptr_t tensor_data, int32_t unitary):
    """Registers an external update of the elements of the specified tensor operator that was previously applied to the tensor network state.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        tensor_id (int64_t): Tensor id assigned during the ``cutensornetStateApplyTensorOperator`` call.
        tensor_data (intptr_t): Pointer to the updated elements of the tensor operator (tensor operator elements must be of the same type as the state tensor).
        unitary (int32_t): Whether or not the tensor operator is unitary with respect to the first and second halves of its modes. This parameter is not applicable to the tensors that are part of a matrix product operator (MPO).

    .. seealso:: `cutensornetStateUpdateTensor`
    """
    with nogil:
        status = cutensornetStateUpdateTensor(<const Handle>handle, <State>tensor_network_state, tensor_id, <void*>tensor_data, unitary)
    check_status(status)


cpdef destroy_state(intptr_t tensor_network_state):
    """Frees all resources owned by the tensor network state.

    Args:
        tensor_network_state (intptr_t): Tensor network state.

    .. seealso:: `cutensornetDestroyState`
    """
    with nogil:
        status = cutensornetDestroyState(<State>tensor_network_state)
    check_status(status)


cpdef intptr_t create_marginal(intptr_t handle, intptr_t tensor_network_state, int32_t num_marginal_modes, marginal_modes, int32_t num_projected_modes, projected_modes, marginal_tensor_strides) except? 0:
    """Creates a representation of the specified marginal tensor for a given tensor network state.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        num_marginal_modes (int32_t): Number of open state modes defining the marginal tensor.
        marginal_modes (object): Pointer to the open state modes defining the marginal tensor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        num_projected_modes (int32_t): Number of projected state modes.
        projected_modes (object): Pointer to the projected state modes. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        marginal_tensor_strides (object): Storage strides for the marginal tensor (number of tensor modes is twice the number of the defining open modes). If NULL, the defaul generalized column-major strides will be assumed. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    Returns:
        intptr_t: Tensor network state marginal.

    .. seealso:: `cutensornetCreateMarginal`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _marginal_modes_
    get_resource_ptr[int32_t](_marginal_modes_, marginal_modes, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _projected_modes_
    get_resource_ptr[int32_t](_projected_modes_, projected_modes, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _marginal_tensor_strides_
    get_resource_ptr[int64_t](_marginal_tensor_strides_, marginal_tensor_strides, <int64_t*>NULL)
    cdef StateMarginal tensor_network_marginal
    with nogil:
        status = cutensornetCreateMarginal(<const Handle>handle, <State>tensor_network_state, num_marginal_modes, <const int32_t*>(_marginal_modes_.data()), num_projected_modes, <const int32_t*>(_projected_modes_.data()), <const int64_t*>(_marginal_tensor_strides_.data()), &tensor_network_marginal)
    check_status(status)
    return <intptr_t>tensor_network_marginal


######################### Python specific utility #########################

cdef dict marginal_attribute_sizes = {
    CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES: _numpy.int32,
    CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES: _numpy.int32,
    CUTENSORNET_MARGINAL_INFO_FLOPS: _numpy.float64,
}

cpdef get_marginal_attribute_dtype(int attr):
    """Get the Python data type of the corresponding MarginalAttribute attribute.

    Args:
        attr (MarginalAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`marginal_configure`, :func:`marginal_get_info`.
    """
    return marginal_attribute_sizes[attr]

###########################################################################


cpdef marginal_configure(intptr_t handle, intptr_t tensor_network_marginal, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Configures computation of the requested tensor network state marginal tensor.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_marginal (intptr_t): Tensor network state marginal representation.
        attribute (MarginalAttribute): Configuration attribute.
        attribute_value (intptr_t): Pointer to the configuration attribute value (type-erased).
        attribute_size (size_t): The size of the configuration attribute value.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_marginal_attribute_dtype`.

    .. seealso:: `cutensornetMarginalConfigure`
    """
    with nogil:
        status = cutensornetMarginalConfigure(<const Handle>handle, <StateMarginal>tensor_network_marginal, <_MarginalAttribute>attribute, <const void*>attribute_value, attribute_size)
    check_status(status)


cpdef marginal_prepare(intptr_t handle, intptr_t tensor_network_marginal, size_t max_workspace_size_device, intptr_t work_desc, intptr_t cuda_stream):
    """Prepares computation of the requested tensor network state marginal tensor.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_marginal (intptr_t): Tensor network state marginal representation.
        max_workspace_size_device (size_t): Upper limit on the amount of available GPU scratch memory (bytes).
        work_desc (intptr_t): Workspace descriptor (the required scratch/cache memory sizes will be set).
        cuda_stream (intptr_t): CUDA stream.

    .. seealso:: `cutensornetMarginalPrepare`
    """
    with nogil:
        status = cutensornetMarginalPrepare(<const Handle>handle, <StateMarginal>tensor_network_marginal, max_workspace_size_device, <WorkspaceDescriptor>work_desc, <Stream>cuda_stream)
    check_status(status)


cpdef marginal_compute(intptr_t handle, intptr_t tensor_network_marginal, projected_mode_values, intptr_t work_desc, intptr_t marginal_tensor, intptr_t cuda_stream):
    """Computes the requested tensor network state marginal tensor.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_marginal (intptr_t): Tensor network state marginal representation.
        projected_mode_values (object): Pointer to the values of the projected modes. Each integer value corresponds to a basis state of the given (projected) state mode. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        work_desc (intptr_t): Workspace descriptor (the required scratch/cache memory buffers must be set by the user).
        marginal_tensor (intptr_t): Pointer to the GPU storage of the marginal tensor which will be computed in this call.
        cuda_stream (intptr_t): CUDA stream.

    .. seealso:: `cutensornetMarginalCompute`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _projected_mode_values_
    get_resource_ptr[int64_t](_projected_mode_values_, projected_mode_values, <int64_t*>NULL)
    with nogil:
        status = cutensornetMarginalCompute(<const Handle>handle, <StateMarginal>tensor_network_marginal, <const int64_t*>(_projected_mode_values_.data()), <WorkspaceDescriptor>work_desc, <void*>marginal_tensor, <Stream>cuda_stream)
    check_status(status)


cpdef destroy_marginal(intptr_t tensor_network_marginal):
    """Destroys the tensor network state marginal.

    Args:
        tensor_network_marginal (intptr_t): Tensor network state marginal representation.

    .. seealso:: `cutensornetDestroyMarginal`
    """
    with nogil:
        status = cutensornetDestroyMarginal(<StateMarginal>tensor_network_marginal)
    check_status(status)


cpdef intptr_t create_sampler(intptr_t handle, intptr_t tensor_network_state, int32_t num_modes_to_sample, modes_to_sample) except? 0:
    """Creates a tensor network state sampler.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        num_modes_to_sample (int32_t): Number of the tensor network state modes to sample from.
        modes_to_sample (object): Pointer to the state modes to sample from (can be NULL when all modes are requested). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.


    Returns:
        intptr_t: Tensor network sampler.

    .. seealso:: `cutensornetCreateSampler`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _modes_to_sample_
    get_resource_ptr[int32_t](_modes_to_sample_, modes_to_sample, <int32_t*>NULL)
    cdef StateSampler tensor_network_sampler
    with nogil:
        status = cutensornetCreateSampler(<const Handle>handle, <State>tensor_network_state, num_modes_to_sample, <const int32_t*>(_modes_to_sample_.data()), &tensor_network_sampler)
    check_status(status)
    return <intptr_t>tensor_network_sampler


######################### Python specific utility #########################

cdef dict sampler_attribute_sizes = {
    CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES: _numpy.int32,
    CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES: _numpy.int32,
    CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC: _numpy.int32,
    CUTENSORNET_SAMPLER_INFO_FLOPS: _numpy.float64,
}

cpdef get_sampler_attribute_dtype(int attr):
    """Get the Python data type of the corresponding SamplerAttribute attribute.

    Args:
        attr (SamplerAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`sampler_configure`, :func:`sampler_get_info`.
    """
    return sampler_attribute_sizes[attr]

###########################################################################


cpdef sampler_configure(intptr_t handle, intptr_t tensor_network_sampler, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Configures the tensor network state sampler.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_sampler (intptr_t): Tensor network state sampler.
        attribute (SamplerAttribute): Configuration attribute.
        attribute_value (intptr_t): Pointer to the configuration attribute value (type-erased).
        attribute_size (size_t): The size of the configuration attribute value.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_sampler_attribute_dtype`.

    .. seealso:: `cutensornetSamplerConfigure`
    """
    with nogil:
        status = cutensornetSamplerConfigure(<const Handle>handle, <StateSampler>tensor_network_sampler, <_SamplerAttribute>attribute, <const void*>attribute_value, attribute_size)
    check_status(status)


cpdef sampler_prepare(intptr_t handle, intptr_t tensor_network_sampler, size_t max_workspace_size_device, intptr_t work_desc, intptr_t cuda_stream):
    """Prepares the tensor network state sampler.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_sampler (intptr_t): Tensor network state sampler.
        max_workspace_size_device (size_t): Upper limit on the amount of available GPU scratch memory (bytes).
        work_desc (intptr_t): Workspace descriptor (the required scratch/cache memory sizes will be set).
        cuda_stream (intptr_t): CUDA stream.

    .. seealso:: `cutensornetSamplerPrepare`
    """
    with nogil:
        status = cutensornetSamplerPrepare(<const Handle>handle, <StateSampler>tensor_network_sampler, max_workspace_size_device, <WorkspaceDescriptor>work_desc, <Stream>cuda_stream)
    check_status(status)


cpdef sampler_sample(intptr_t handle, intptr_t tensor_network_sampler, int64_t num_shots, intptr_t work_desc, intptr_t samples, intptr_t cuda_stream):
    """Performs sampling of the tensor network state, that is, generates the requested number of samples.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_sampler (intptr_t): Tensor network state sampler.
        num_shots (int64_t): Number of samples to generate.
        work_desc (intptr_t): Workspace descriptor (the required scratch/cache memory buffers must be set by the user).
        samples (intptr_t): Host memory pointer where the generated state tensor samples will be stored at. The samples will be stored as samples[SampleId][ModeId] in C notation and the originally specified order of the tensor network state modes to sample from will be respected.
        cuda_stream (intptr_t): CUDA stream.

    .. seealso:: `cutensornetSamplerSample`
    """
    with nogil:
        status = cutensornetSamplerSample(<const Handle>handle, <StateSampler>tensor_network_sampler, num_shots, <WorkspaceDescriptor>work_desc, <int64_t*>samples, <Stream>cuda_stream)
    check_status(status)


cpdef destroy_sampler(intptr_t tensor_network_sampler):
    """Destroys the tensor network state sampler.

    Args:
        tensor_network_sampler (intptr_t): Tensor network state sampler.

    .. seealso:: `cutensornetDestroySampler`
    """
    with nogil:
        status = cutensornetDestroySampler(<StateSampler>tensor_network_sampler)
    check_status(status)


cpdef state_finalize_mps(intptr_t handle, intptr_t tensor_network_state, int boundary_condition, extents_out, strides_out):
    """Imposes a user-defined MPS (Matrix Product State) factorization on the final tensor network state with the given shape.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        boundary_condition (BoundaryCondition): The boundary condition of the target MPS representation.
        extents_out (object): Array of size ``nStateModes`` specifying the maximal extents of all tensors defining the target MPS representation. ``extents_out[i]`` is expected to be consistent with the mode order (shared mode between (i-1)th and i-th MPS tensor, state mode of the i-th MPS tensor, shared mode between i-th and (i+1)th MPS tensor). For the open boundary condition, the modes for the first tensor get reduced to (state mode, shared mode with the second site) while the modes for the last tensor become (shared mode with the second last site, state mode). It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int64_t', or
            - a nested Python sequence of ``int64_t``.

        strides_out (object): Array of size ``nStateModes`` specifying the strides of all tensors defining the target MPS representation. Similar to ``extents_out``, ``strides_out`` is also expected to be consistent with the mode order of each MPS tensor. If NULL, the default generalized column-major strides will be assumed. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int64_t', or
            - a nested Python sequence of ``int64_t``.


    .. seealso:: `cutensornetStateFinalizeMPS`
    """
    cdef nested_resource[ int64_t ] _extents_out_
    get_nested_resource_ptr[int64_t](_extents_out_, extents_out, <int64_t*>NULL)
    cdef nested_resource[ int64_t ] _strides_out_
    get_nested_resource_ptr[int64_t](_strides_out_, strides_out, <int64_t*>NULL)
    with nogil:
        status = cutensornetStateFinalizeMPS(<const Handle>handle, <State>tensor_network_state, <_BoundaryCondition>boundary_condition, <const int64_t* const*>(_extents_out_.ptrs.data()), <const int64_t* const*>(_strides_out_.ptrs.data()))
    check_status(status)


######################### Python specific utility #########################

cdef dict state_attribute_sizes = {
    CUTENSORNET_STATE_MPS_CANONICAL_CENTER: _numpy.int32,
    CUTENSORNET_STATE_MPS_SVD_CONFIG_ABS_CUTOFF: _numpy.float64,
    CUTENSORNET_STATE_MPS_SVD_CONFIG_REL_CUTOFF: _numpy.float64,
    CUTENSORNET_STATE_MPS_SVD_CONFIG_S_NORMALIZATION: _numpy.int32,
    CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO: _numpy.int32,
    CUTENSORNET_STATE_MPS_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF: _numpy.float64,
    CUTENSORNET_STATE_NUM_HYPER_SAMPLES: _numpy.int32,
    CUTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER: _numpy.int32,
    CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF: _numpy.float64,
    CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF: _numpy.float64,
    CUTENSORNET_STATE_CONFIG_MPS_SVD_S_NORMALIZATION: _numpy.int32,
    CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO: _numpy.int32,
    CUTENSORNET_STATE_CONFIG_MPS_SVD_DISCARDED_WEIGHT_CUTOFF: _numpy.float64,
    CUTENSORNET_STATE_CONFIG_MPS_MPO_APPLICATION: _numpy.int32,
    CUTENSORNET_STATE_CONFIG_MPS_GAUGE_OPTION: _numpy.int32,
    CUTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES: _numpy.int32,
    CUTENSORNET_STATE_INFO_FLOPS: _numpy.float64,
}

cpdef get_state_attribute_dtype(int attr):
    """Get the Python data type of the corresponding StateAttribute attribute.

    Args:
        attr (StateAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`state_configure`, :func:`state_get_info`.
    """
    if attr == CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO_PARAMS:
        raise ValueError('use tensor_svd_algo_params_get_dtype to get the dtype')
    if attr == CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO_PARAMS:
        raise ValueError('use tensor_svd_algo_params_get_dtype to get the dtype')
    return state_attribute_sizes[attr]

###########################################################################


cpdef state_configure(intptr_t handle, intptr_t tensor_network_state, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Configures computation of the full tensor network state, either in the exact or a factorized form.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        attribute (StateAttribute): Configuration attribute.
        attribute_value (intptr_t): Pointer to the configuration attribute value (type-erased).
        attribute_size (size_t): The size of the configuration attribute value.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_state_attribute_dtype`.

    .. seealso:: `cutensornetStateConfigure`
    """
    with nogil:
        status = cutensornetStateConfigure(<const Handle>handle, <State>tensor_network_state, <_StateAttribute>attribute, <const void*>attribute_value, attribute_size)
    check_status(status)


cpdef state_prepare(intptr_t handle, intptr_t tensor_network_state, size_t max_workspace_size_device, intptr_t work_desc, intptr_t cuda_stream):
    """Prepares computation of the full tensor network state, either in the exact or a factorized form.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        max_workspace_size_device (size_t): Upper limit on the amount of available GPU scratch memory (bytes).
        work_desc (intptr_t): Workspace descriptor (the required scratch/cache memory sizes will be set).
        cuda_stream (intptr_t): CUDA stream.

    .. seealso:: `cutensornetStatePrepare`
    """
    with nogil:
        status = cutensornetStatePrepare(<const Handle>handle, <State>tensor_network_state, max_workspace_size_device, <WorkspaceDescriptor>work_desc, <Stream>cuda_stream)
    check_status(status)


cpdef intptr_t create_network_operator(intptr_t handle, int32_t num_state_modes, state_mode_extents, int data_type) except? 0:
    """Creates an uninitialized tensor network operator of a given shape defined by the number of state modes and their extents.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        num_state_modes (int32_t): The number of state modes the operator acts on.
        state_mode_extents (object): An array of size ``num_state_modes`` specifying the extent of each state mode acted on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        data_type (int): Data type of the operator.

    Returns:
        intptr_t: Tensor network operator (empty at this point).

    .. seealso:: `cutensornetCreateNetworkOperator`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _state_mode_extents_
    get_resource_ptr[int64_t](_state_mode_extents_, state_mode_extents, <int64_t*>NULL)
    cdef NetworkOperator tensor_network_operator
    with nogil:
        status = cutensornetCreateNetworkOperator(<const Handle>handle, num_state_modes, <const int64_t*>(_state_mode_extents_.data()), <DataType>data_type, &tensor_network_operator)
    check_status(status)
    return <intptr_t>tensor_network_operator


cpdef int64_t network_operator_append_product(intptr_t handle, intptr_t tensor_network_operator, complex coefficient, int32_t num_tensors, num_state_modes, state_modes, tensor_mode_strides, tensor_data) except? -1:
    """Appends a tensor product operator component to the tensor network operator.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_operator (intptr_t): Tensor network operator.
        coefficient (complex): Complex coefficient associated with the appended operator component.
        num_tensors (int32_t): Number of tensor factors in the tensor product.
        num_state_modes (object): Number of state modes each appended tensor factor acts on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        state_modes (object): Modes each appended tensor factor acts on (length = ``num_state_modes``). It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int32_t', or
            - a nested Python sequence of ``int32_t``.

        tensor_mode_strides (object): Tensor mode strides for each tensor factor (length = ``num_state_modes`` * 2). If NULL, the default generalized column-major strides will be used. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int64_t', or
            - a nested Python sequence of ``int64_t``.

        tensor_data (object): Tensor data stored in GPU memory for each tensor factor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).


    Returns:
        int64_t: Unique sequential integer identifier of the appended tensor network operator component.

    .. seealso:: `cutensornetNetworkOperatorAppendProduct`
    """
    cdef cuDoubleComplex _coefficient_ = cuDoubleComplex(coefficient.real, coefficient.imag)
    cdef nullable_unique_ptr[ vector[int32_t] ] _num_state_modes_
    get_resource_ptr[int32_t](_num_state_modes_, num_state_modes, <int32_t*>NULL)
    cdef nested_resource[ int32_t ] _state_modes_
    get_nested_resource_ptr[int32_t](_state_modes_, state_modes, <int32_t*>NULL)
    cdef nested_resource[ int64_t ] _tensor_mode_strides_
    get_nested_resource_ptr[int64_t](_tensor_mode_strides_, tensor_mode_strides, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _tensor_data_
    get_resource_ptrs[void](_tensor_data_, tensor_data, <void*>NULL)
    cdef int64_t component_id
    with nogil:
        status = cutensornetNetworkOperatorAppendProduct(<const Handle>handle, <NetworkOperator>tensor_network_operator, <cuDoubleComplex>_coefficient_, num_tensors, <const int32_t*>(_num_state_modes_.data()), <const int32_t**>(_state_modes_.ptrs.data()), <const int64_t**>(_tensor_mode_strides_.ptrs.data()), <const void**>(_tensor_data_.data()), &component_id)
    check_status(status)
    return component_id


cpdef destroy_network_operator(intptr_t tensor_network_operator):
    """Frees all resources owned by the tensor network operator.

    Args:
        tensor_network_operator (intptr_t): Tensor network operator.

    .. seealso:: `cutensornetDestroyNetworkOperator`
    """
    with nogil:
        status = cutensornetDestroyNetworkOperator(<NetworkOperator>tensor_network_operator)
    check_status(status)


cpdef intptr_t create_accessor(intptr_t handle, intptr_t tensor_network_state, int32_t num_projected_modes, projected_modes, amplitudes_tensor_strides) except? 0:
    """Creates a tensor network state amplitudes accessor.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Defined tensor network state.
        num_projected_modes (int32_t): Number of projected state modes (tensor network state modes projected to specific basis vectors).
        projected_modes (object): Projected state modes (may be NULL when none or all modes are projected). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        amplitudes_tensor_strides (object): Mode strides for the resulting amplitudes tensor. If NULL, the default generalized column-major strides will be assumed. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    Returns:
        intptr_t: Tensor network state amplitudes accessor.

    .. seealso:: `cutensornetCreateAccessor`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _projected_modes_
    get_resource_ptr[int32_t](_projected_modes_, projected_modes, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _amplitudes_tensor_strides_
    get_resource_ptr[int64_t](_amplitudes_tensor_strides_, amplitudes_tensor_strides, <int64_t*>NULL)
    cdef StateAccessor tensor_network_accessor
    with nogil:
        status = cutensornetCreateAccessor(<const Handle>handle, <State>tensor_network_state, num_projected_modes, <const int32_t*>(_projected_modes_.data()), <const int64_t*>(_amplitudes_tensor_strides_.data()), &tensor_network_accessor)
    check_status(status)
    return <intptr_t>tensor_network_accessor


######################### Python specific utility #########################

cdef dict accessor_attribute_sizes = {
    CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES: _numpy.int32,
    CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES: _numpy.int32,
    CUTENSORNET_ACCESSOR_INFO_FLOPS: _numpy.float64,
}

cpdef get_accessor_attribute_dtype(int attr):
    """Get the Python data type of the corresponding AccessorAttribute attribute.

    Args:
        attr (AccessorAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`accessor_configure`, :func:`accessor_get_info`.
    """
    return accessor_attribute_sizes[attr]

###########################################################################


cpdef accessor_configure(intptr_t handle, intptr_t tensor_network_accessor, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Configures computation of the requested tensor network state amplitudes tensor.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_accessor (intptr_t): Tensor network state amplitudes accessor.
        attribute (AccessorAttribute): Configuration attribute.
        attribute_value (intptr_t): Pointer to the configuration attribute value (type-erased).
        attribute_size (size_t): The size of the configuration attribute value.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_accessor_attribute_dtype`.

    .. seealso:: `cutensornetAccessorConfigure`
    """
    with nogil:
        status = cutensornetAccessorConfigure(<const Handle>handle, <StateAccessor>tensor_network_accessor, <_AccessorAttribute>attribute, <const void*>attribute_value, attribute_size)
    check_status(status)


cpdef accessor_prepare(intptr_t handle, intptr_t tensor_network_accessor, size_t max_workspace_size_device, intptr_t work_desc, intptr_t cuda_stream):
    """Prepares computation of the requested tensor network state amplitudes tensor.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_accessor (intptr_t): Tensor network state amplitudes accessor.
        max_workspace_size_device (size_t): Upper limit on the amount of available GPU scratch memory (bytes).
        work_desc (intptr_t): Workspace descriptor (the required scratch/cache memory sizes will be set).
        cuda_stream (intptr_t): CUDA stream.

    .. seealso:: `cutensornetAccessorPrepare`
    """
    with nogil:
        status = cutensornetAccessorPrepare(<const Handle>handle, <StateAccessor>tensor_network_accessor, max_workspace_size_device, <WorkspaceDescriptor>work_desc, <Stream>cuda_stream)
    check_status(status)


cpdef accessor_compute(intptr_t handle, intptr_t tensor_network_accessor, projected_mode_values, intptr_t work_desc, intptr_t amplitudes_tensor, intptr_t state_norm, intptr_t cuda_stream):
    """Computes the amplitudes of the tensor network state.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_accessor (intptr_t): Tensor network state amplitudes accessor.
        projected_mode_values (object): The values of the projected state modes or NULL pointer if there are no projected modes. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        work_desc (intptr_t): Workspace descriptor (the required scratch/cache memory buffers must be set by the user).
        amplitudes_tensor (intptr_t): Storage for the computed tensor network state amplitudes tensor.
        state_norm (intptr_t): The squared 2-norm of the underlying tensor circuit state (Host pointer). The returned scalar will have the same numerical data type as the tensor circuit state. Providing a NULL pointer will ignore norm calculation.
        cuda_stream (intptr_t): CUDA stream.

    .. seealso:: `cutensornetAccessorCompute`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _projected_mode_values_
    get_resource_ptr[int64_t](_projected_mode_values_, projected_mode_values, <int64_t*>NULL)
    with nogil:
        status = cutensornetAccessorCompute(<const Handle>handle, <StateAccessor>tensor_network_accessor, <const int64_t*>(_projected_mode_values_.data()), <WorkspaceDescriptor>work_desc, <void*>amplitudes_tensor, <void*>state_norm, <Stream>cuda_stream)
    check_status(status)


cpdef destroy_accessor(intptr_t tensor_network_accessor):
    """Destroyes the tensor network state amplitudes accessor.

    Args:
        tensor_network_accessor (intptr_t): Tensor network state amplitudes accessor.

    .. seealso:: `cutensornetDestroyAccessor`
    """
    with nogil:
        status = cutensornetDestroyAccessor(<StateAccessor>tensor_network_accessor)
    check_status(status)


cpdef intptr_t create_expectation(intptr_t handle, intptr_t tensor_network_state, intptr_t tensor_network_operator) except? 0:
    """Creates a representation of the tensor network state expectation value.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Defined tensor network state.
        tensor_network_operator (intptr_t): Defined tensor network operator.

    Returns:
        intptr_t: Tensor network expectation value representation.

    .. seealso:: `cutensornetCreateExpectation`
    """
    cdef StateExpectation tensor_network_expectation
    with nogil:
        status = cutensornetCreateExpectation(<const Handle>handle, <State>tensor_network_state, <NetworkOperator>tensor_network_operator, &tensor_network_expectation)
    check_status(status)
    return <intptr_t>tensor_network_expectation


######################### Python specific utility #########################

cdef dict expectation_attribute_sizes = {
    CUTENSORNET_EXPECTATION_OPT_NUM_HYPER_SAMPLES: _numpy.int32,
    CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES: _numpy.int32,
    CUTENSORNET_EXPECTATION_INFO_FLOPS: _numpy.float64,
}

cpdef get_expectation_attribute_dtype(int attr):
    """Get the Python data type of the corresponding ExpectationAttribute attribute.

    Args:
        attr (ExpectationAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`expectation_configure`, :func:`expectation_get_info`.
    """
    return expectation_attribute_sizes[attr]

###########################################################################


cpdef expectation_configure(intptr_t handle, intptr_t tensor_network_expectation, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Configures computation of the requested tensor network state expectation value.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_expectation (intptr_t): Tensor network state expectation value representation.
        attribute (ExpectationAttribute): Configuration attribute.
        attribute_value (intptr_t): Pointer to the configuration attribute value (type-erased).
        attribute_size (size_t): The size of the configuration attribute value.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_expectation_attribute_dtype`.

    .. seealso:: `cutensornetExpectationConfigure`
    """
    with nogil:
        status = cutensornetExpectationConfigure(<const Handle>handle, <StateExpectation>tensor_network_expectation, <_ExpectationAttribute>attribute, <const void*>attribute_value, attribute_size)
    check_status(status)


cpdef expectation_prepare(intptr_t handle, intptr_t tensor_network_expectation, size_t max_workspace_size_device, intptr_t work_desc, intptr_t cuda_stream):
    """Prepares computation of the requested tensor network state expectation value.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_expectation (intptr_t): Tensor network state expectation value representation.
        max_workspace_size_device (size_t): Upper limit on the amount of available GPU scratch memory (bytes).
        work_desc (intptr_t): Workspace descriptor (the required scratch/cache memory sizes will be set).
        cuda_stream (intptr_t): CUDA stream.

    .. seealso:: `cutensornetExpectationPrepare`
    """
    with nogil:
        status = cutensornetExpectationPrepare(<const Handle>handle, <StateExpectation>tensor_network_expectation, max_workspace_size_device, <WorkspaceDescriptor>work_desc, <Stream>cuda_stream)
    check_status(status)


cpdef expectation_compute(intptr_t handle, intptr_t tensor_network_expectation, intptr_t work_desc, intptr_t expectation_value, intptr_t state_norm, intptr_t cuda_stream):
    """Computes an (unnormalized) expectation value of a given tensor network operator over a given tensor network state.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_expectation (intptr_t): Tensor network state expectation value representation.
        work_desc (intptr_t): Workspace descriptor (the required scratch/cache memory buffers must be set by the user).
        expectation_value (intptr_t): Computed unnormalized tensor network state expectation value (Host pointer). The returned scalar will have the same numerical data type as the tensor circuit state.
        state_norm (intptr_t): The squared 2-norm of the underlying tensor circuit state (Host pointer). The returned scalar will have the same numerical data type as the tensor circuit state. Providing a NULL pointer will ignore norm calculation.
        cuda_stream (intptr_t): CUDA stream.

    .. seealso:: `cutensornetExpectationCompute`
    """
    with nogil:
        status = cutensornetExpectationCompute(<const Handle>handle, <StateExpectation>tensor_network_expectation, <WorkspaceDescriptor>work_desc, <void*>expectation_value, <void*>state_norm, <Stream>cuda_stream)
    check_status(status)


cpdef destroy_expectation(intptr_t tensor_network_expectation):
    """Destroyes the tensor network state expectation value representation.

    Args:
        tensor_network_expectation (intptr_t): Tensor network state expectation value representation.

    .. seealso:: `cutensornetDestroyExpectation`
    """
    with nogil:
        status = cutensornetDestroyExpectation(<StateExpectation>tensor_network_expectation)
    check_status(status)


cpdef int64_t state_apply_tensor_operator(intptr_t handle, intptr_t tensor_network_state, int32_t num_state_modes, state_modes, intptr_t tensor_data, tensor_mode_strides, int32_t immutable, int32_t adjoint, int32_t unitary) except? -1:
    """Applies a tensor operator to the tensor network state.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        num_state_modes (int32_t): Number of state modes the tensor operator acts on.
        state_modes (object): Pointer to the state modes the tensor operator acts on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        tensor_data (intptr_t): Elements of the tensor operator (must be of the same data type as the elements of the state tensor).
        tensor_mode_strides (object): Strides of the tensor operator data layout (note that the tensor operator has twice more modes than the number of state modes it acts on). Passing NULL will assume the default generalized columnwise storage layout. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        immutable (int32_t): Whether or not the tensor operator data may change during the lifetime of the tensor network state. Any data change must be registered via a call to ``cutensornetStateUpdateTensorOperator``.
        adjoint (int32_t): Whether or not the tensor operator is applied as an adjoint (ket and bra modes reversed, with all tensor elements complex conjugated).
        unitary (int32_t): Whether or not the tensor operator is unitary with respect to the first and second halves of its modes.

    Returns:
        int64_t: Unique integer id (for later identification of the tensor operator).

    .. seealso:: `cutensornetStateApplyTensorOperator`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_modes_
    get_resource_ptr[int32_t](_state_modes_, state_modes, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _tensor_mode_strides_
    get_resource_ptr[int64_t](_tensor_mode_strides_, tensor_mode_strides, <int64_t*>NULL)
    cdef int64_t tensor_id
    with nogil:
        status = cutensornetStateApplyTensorOperator(<const Handle>handle, <State>tensor_network_state, num_state_modes, <const int32_t*>(_state_modes_.data()), <void*>tensor_data, <const int64_t*>(_tensor_mode_strides_.data()), <const int32_t>immutable, <const int32_t>adjoint, <const int32_t>unitary, &tensor_id)
    check_status(status)
    return tensor_id


cpdef int64_t state_apply_controlled_tensor_operator(intptr_t handle, intptr_t tensor_network_state, int32_t num_control_modes, state_control_modes, state_control_values, int32_t num_target_modes, state_target_modes, intptr_t tensor_data, tensor_mode_strides, int32_t immutable, int32_t adjoint, int32_t unitary) except? -1:
    """Applies a controlled tensor operator to the tensor network state.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        num_control_modes (int32_t): Number of control state modes used by the tensor operator.
        state_control_modes (object): Controlling state modes used by the tensor operator. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        state_control_values (object): Control values for the controlling state modes. A control value is the sequential integer id of the qudit basis component which activates the action of the target tensor operator. If NULL, all control values are assumed to be set to the max id (last qudit basis component), which will be 1 for qubits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        num_target_modes (int32_t): Number of target state modes acted on by the tensor operator.
        state_target_modes (object): Target state modes acted on by the tensor operator. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        tensor_data (intptr_t): Elements of the target tensor of the controlled tensor operator (must be of the same data type as the elements of the state tensor).
        tensor_mode_strides (object): Strides of the tensor operator data layout (note that the tensor operator has twice more modes than the number of the target state modes it acts on). Passing NULL will assume the default generalized columnwise storage layout. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        immutable (int32_t): Whether or not the tensor operator data may change during the lifetime of the tensor network state. Any data change must be registered via a call to ``cutensornetStateUpdateTensorOperator``.
        adjoint (int32_t): Whether or not the tensor operator is applied as an adjoint (ket and bra modes reversed, with all tensor elements complex conjugated).
        unitary (int32_t): Whether or not the controlled tensor operator is unitary with respect to the first and second halves of its modes.

    Returns:
        int64_t: Unique integer id (for later identification of the tensor operator).

    .. seealso:: `cutensornetStateApplyControlledTensorOperator`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_control_modes_
    get_resource_ptr[int32_t](_state_control_modes_, state_control_modes, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _state_control_values_
    get_resource_ptr[int64_t](_state_control_values_, state_control_values, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_target_modes_
    get_resource_ptr[int32_t](_state_target_modes_, state_target_modes, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _tensor_mode_strides_
    get_resource_ptr[int64_t](_tensor_mode_strides_, tensor_mode_strides, <int64_t*>NULL)
    cdef int64_t tensor_id
    with nogil:
        status = cutensornetStateApplyControlledTensorOperator(<const Handle>handle, <State>tensor_network_state, num_control_modes, <const int32_t*>(_state_control_modes_.data()), <const int64_t*>(_state_control_values_.data()), num_target_modes, <const int32_t*>(_state_target_modes_.data()), <void*>tensor_data, <const int64_t*>(_tensor_mode_strides_.data()), <const int32_t>immutable, <const int32_t>adjoint, <const int32_t>unitary, &tensor_id)
    check_status(status)
    return tensor_id


cpdef state_update_tensor_operator(intptr_t handle, intptr_t tensor_network_state, int64_t tensor_id, intptr_t tensor_data, int32_t unitary):
    """Registers an external update of the elements of the specified tensor operator that was previously applied to the tensor network state.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        tensor_id (int64_t): Tensor id assigned during the ``cutensornetStateApplyTensorOperator`` call.
        tensor_data (intptr_t): Pointer to the updated elements of the tensor operator (tensor operator elements must be of the same type as the state tensor).
        unitary (int32_t): Whether or not the tensor operator is unitary with respect to the first and second halves of its modes. This parameter is not applicable to the tensors that are part of a matrix product operator (MPO).

    .. seealso:: `cutensornetStateUpdateTensorOperator`
    """
    with nogil:
        status = cutensornetStateUpdateTensorOperator(<const Handle>handle, <State>tensor_network_state, tensor_id, <void*>tensor_data, unitary)
    check_status(status)


cpdef int64_t state_apply_network_operator(intptr_t handle, intptr_t tensor_network_state, intptr_t tensor_network_operator, int32_t immutable, int32_t adjoint, int32_t unitary) except? -1:
    """Applies a tensor network operator to a tensor network state.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        tensor_network_operator (intptr_t): Tensor network operator containg only a single component.
        immutable (int32_t): Whether or not the tensor network operator data may change during the lifetime of the tensor network state.
        adjoint (int32_t): Whether or not the tensor network operator is applied as an adjoint.
        unitary (int32_t): Whether or not the tensor network operator is unitary with respect to the first and second halves of its modes.

    Returns:
        int64_t: Unique integer id (for later identification of the tensor network operator).

    .. seealso:: `cutensornetStateApplyNetworkOperator`
    """
    cdef int64_t operator_id
    with nogil:
        status = cutensornetStateApplyNetworkOperator(<const Handle>handle, <State>tensor_network_state, <const NetworkOperator>tensor_network_operator, <const int32_t>immutable, <const int32_t>adjoint, <const int32_t>unitary, &operator_id)
    check_status(status)
    return operator_id


cpdef state_initialize_mps(intptr_t handle, intptr_t tensor_network_state, int boundary_condition, extents_in, strides_in, state_tensors_in):
    """Imposes a user-defined MPS (Matrix Product State) factorization on the initial tensor network state with the given shape and data.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        boundary_condition (BoundaryCondition): The boundary condition of the chosen MPS representation.
        extents_in (object): Array of size ``nStateModes`` specifying the extents of all tensors defining the initial MPS representation. ``extents[i]`` is expected to be consistent with the mode order (shared mode between (i-1)th and i-th MPS tensor, state mode of the i-th MPS tensor, shared mode between i-th and the (i+1)th MPS tensor). For the open boundary condition, the modes of the first tensor get reduced to (state mode, shared mode with the second site) while the modes of the last tensor become (shared mode with the second to the last site, state mode). It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int64_t', or
            - a nested Python sequence of ``int64_t``.

        strides_in (object): Array of size ``nStateModes`` specifying the strides of all tensors in the chosen MPS representation. Similar to ``extents_in``, ``strides_in`` is also expected to be consistent with the mode order of each MPS tensor. If NULL, the default generalized column-major strides will be assumed. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int64_t', or
            - a nested Python sequence of ``int64_t``.

        state_tensors_in (object): Array of size ``nStateModes`` specifying the data for all tensors defining the chosen MPS representation. If NULL, the initial MPS-factorized state will represent the vacuum state. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).


    .. seealso:: `cutensornetStateInitializeMPS`
    """
    cdef nested_resource[ int64_t ] _extents_in_
    get_nested_resource_ptr[int64_t](_extents_in_, extents_in, <int64_t*>NULL)
    cdef nested_resource[ int64_t ] _strides_in_
    get_nested_resource_ptr[int64_t](_strides_in_, strides_in, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _state_tensors_in_
    get_resource_ptrs[void](_state_tensors_in_, state_tensors_in, <void*>NULL)
    with nogil:
        status = cutensornetStateInitializeMPS(<const Handle>handle, <State>tensor_network_state, <_BoundaryCondition>boundary_condition, <const int64_t* const*>(_extents_in_.ptrs.data()), <const int64_t* const*>(_strides_in_.ptrs.data()), <void**>(_state_tensors_in_.data()))
    check_status(status)


cpdef state_get_info(intptr_t handle, intptr_t tensor_network_state, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Retrieves an attribute related to computation of the full tensor network state, either in the exact or a factorized form.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        attribute (StateAttribute): Information attribute.
        attribute_value (intptr_t): Pointer to the information attribute value (type-erased).
        attribute_size (size_t): The size of the information attribute value.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_state_attribute_dtype`.

    .. seealso:: `cutensornetStateGetInfo`
    """
    with nogil:
        status = cutensornetStateGetInfo(<const Handle>handle, <const State>tensor_network_state, <_StateAttribute>attribute, <void*>attribute_value, attribute_size)
    check_status(status)


cpdef int64_t network_operator_append_mpo(intptr_t handle, intptr_t tensor_network_operator, complex coefficient, int32_t num_state_modes, state_modes, tensor_mode_extents, tensor_mode_strides, tensor_data, int boundary_condition) except? -1:
    """Appends a Matrix Product Operator (MPO) component to the tensor network operator.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_operator (intptr_t): Tensor network operator.
        coefficient (complex): Complex coefficient associated with the appended operator component.
        num_state_modes (int32_t): Number of state modes the MPO acts on (number of tensors in the MPO).
        state_modes (object): State modes the MPO acts on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        tensor_mode_extents (object): Tensor mode extents for each MPO tensor. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int64_t', or
            - a nested Python sequence of ``int64_t``.

        tensor_mode_strides (object): Storage strides for each MPO tensor or NULL (default generalized column-wise strides). It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int64_t', or
            - a nested Python sequence of ``int64_t``.

        tensor_data (object): Tensor data stored in GPU memory for each MPO tensor factor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        boundary_condition (BoundaryCondition): MPO boundary condition.

    Returns:
        int64_t: Unique sequential integer identifier of the appended tensor network operator component.

    .. seealso:: `cutensornetNetworkOperatorAppendMPO`
    """
    cdef cuDoubleComplex _coefficient_ = cuDoubleComplex(coefficient.real, coefficient.imag)
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_modes_
    get_resource_ptr[int32_t](_state_modes_, state_modes, <int32_t*>NULL)
    cdef nested_resource[ int64_t ] _tensor_mode_extents_
    get_nested_resource_ptr[int64_t](_tensor_mode_extents_, tensor_mode_extents, <int64_t*>NULL)
    cdef nested_resource[ int64_t ] _tensor_mode_strides_
    get_nested_resource_ptr[int64_t](_tensor_mode_strides_, tensor_mode_strides, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _tensor_data_
    get_resource_ptrs[void](_tensor_data_, tensor_data, <void*>NULL)
    cdef int64_t component_id
    with nogil:
        status = cutensornetNetworkOperatorAppendMPO(<const Handle>handle, <NetworkOperator>tensor_network_operator, <cuDoubleComplex>_coefficient_, num_state_modes, <const int32_t*>(_state_modes_.data()), <const int64_t**>(_tensor_mode_extents_.ptrs.data()), <const int64_t**>(_tensor_mode_strides_.ptrs.data()), <const void**>(_tensor_data_.data()), <_BoundaryCondition>boundary_condition, &component_id)
    check_status(status)
    return component_id


cpdef accessor_get_info(intptr_t handle, intptr_t tensor_network_accessor, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Retrieves an attribute related to computation of the requested tensor network state amplitudes tensor.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_accessor (intptr_t): Tensor network state amplitudes accessor.
        attribute (AccessorAttribute): Information attribute.
        attribute_value (intptr_t): Pointer to the information attribute value (type-erased).
        attribute_size (size_t): The size of the information attribute value.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_accessor_attribute_dtype`.

    .. seealso:: `cutensornetAccessorGetInfo`
    """
    with nogil:
        status = cutensornetAccessorGetInfo(<const Handle>handle, <const StateAccessor>tensor_network_accessor, <_AccessorAttribute>attribute, <void*>attribute_value, attribute_size)
    check_status(status)


cpdef expectation_get_info(intptr_t handle, intptr_t tensor_network_expectation, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Retrieves an attribute related to computation of the requested tensor network state expectation value.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_expectation (intptr_t): Tensor network state expectation value representation.
        attribute (ExpectationAttribute): Information attribute.
        attribute_value (intptr_t): Pointer to the information attribute value (type-erased).
        attribute_size (size_t): The size of the information attribute value.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_expectation_attribute_dtype`.

    .. seealso:: `cutensornetExpectationGetInfo`
    """
    with nogil:
        status = cutensornetExpectationGetInfo(<const Handle>handle, <const StateExpectation>tensor_network_expectation, <_ExpectationAttribute>attribute, <void*>attribute_value, attribute_size)
    check_status(status)


cpdef marginal_get_info(intptr_t handle, intptr_t tensor_network_marginal, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Retrieves an attribute related to computation of the requested tensor network state marginal tensor.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_marginal (intptr_t): Tensor network state marginal representation.
        attribute (MarginalAttribute): Information attribute.
        attribute_value (intptr_t): Pointer to the information attribute value (type-erased).
        attribute_size (size_t): The size of the information attribute value.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_marginal_attribute_dtype`.

    .. seealso:: `cutensornetMarginalGetInfo`
    """
    with nogil:
        status = cutensornetMarginalGetInfo(<const Handle>handle, <const StateMarginal>tensor_network_marginal, <_MarginalAttribute>attribute, <void*>attribute_value, attribute_size)
    check_status(status)


cpdef sampler_get_info(intptr_t handle, intptr_t tensor_network_sampler, int attribute, intptr_t attribute_value, size_t attribute_size):
    """Retrieves an attribute related to tensor network state sampling.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_sampler (intptr_t): Tensor network state sampler.
        attribute (SamplerAttribute): Information attribute.
        attribute_value (intptr_t): Pointer to the information attribute value (type-erased).
        attribute_size (size_t): The size of the information attribute value.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_sampler_attribute_dtype`.

    .. seealso:: `cutensornetSamplerGetInfo`
    """
    with nogil:
        status = cutensornetSamplerGetInfo(<const Handle>handle, <const StateSampler>tensor_network_sampler, <_SamplerAttribute>attribute, <void*>attribute_value, attribute_size)
    check_status(status)


cpdef int64_t state_apply_unitary_channel(intptr_t handle, intptr_t tensor_network_state, int32_t num_state_modes, state_modes, int32_t num_tensors, tensor_data, tensor_mode_strides, probabilities) except? -1:
    """Applies a tensor channel consisting of one or more unitary tensor operators to the tensor network state.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        num_state_modes (int32_t): Number of state modes the tensor channel acts on.
        state_modes (object): Pointer to the state modes the tensor channel acts on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        num_tensors (int32_t): Number of constituting tensor operators defining the tensor channel.
        tensor_data (object): Elements of the tensor operators constituting the tensor channel (must be of the same data type as the elements of the state tensor). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        tensor_mode_strides (object): Strides of the tensor data storage layout (note that the supplied tensors have twice more modes than the number of state modes they act on). Passing NULL will assume the default generalized columnwise storage layout. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        probabilities (object): Probabilities associated with the individual tensor operators. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``float``.


    Returns:
        int64_t: Unique integer id (for later identification of the tensor channel).

    .. seealso:: `cutensornetStateApplyUnitaryChannel`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_modes_
    get_resource_ptr[int32_t](_state_modes_, state_modes, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _tensor_data_
    get_resource_ptrs[void](_tensor_data_, tensor_data, <void*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _tensor_mode_strides_
    get_resource_ptr[int64_t](_tensor_mode_strides_, tensor_mode_strides, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[double] ] _probabilities_
    get_resource_ptr[double](_probabilities_, probabilities, <double*>NULL)
    cdef int64_t channel_id
    with nogil:
        status = cutensornetStateApplyUnitaryChannel(<const Handle>handle, <State>tensor_network_state, num_state_modes, <const int32_t*>(_state_modes_.data()), num_tensors, <void**>(_tensor_data_.data()), <const int64_t*>(_tensor_mode_strides_.data()), <const double*>(_probabilities_.data()), &channel_id)
    check_status(status)
    return channel_id


cpdef state_capture_mps(intptr_t handle, intptr_t tensor_network_state):
    """Resets the tensor network state to the MPS state previously computed via ``cutensornetStateCompute``.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.

    .. seealso:: `cutensornetStateCaptureMPS`
    """
    with nogil:
        status = cutensornetStateCaptureMPS(<const Handle>handle, <State>tensor_network_state)
    check_status(status)


cpdef int64_t state_apply_general_channel(intptr_t handle, intptr_t tensor_network_state, int32_t num_state_modes, state_modes, int32_t num_tensors, tensor_data, tensor_mode_strides) except? -1:
    """Applies a tensor channel consisting of one or more gneral Kraus operators to the tensor network state.

    Args:
        handle (intptr_t): cuTensorNet library handle.
        tensor_network_state (intptr_t): Tensor network state.
        num_state_modes (int32_t): Number of state modes the tensor channel acts on.
        state_modes (object): Pointer to the state modes the tensor channel acts on. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        num_tensors (int32_t): Number of constituting tensor operators defining the tensor channel.
        tensor_data (object): Elements of the tensor operators constituting the tensor channel (must be of the same data type as the elements of the state tensor). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        tensor_mode_strides (object): Strides of the tensor data storage layout (note that the supplied tensors have twice more modes than the number of state modes they act on). Passing NULL will assume the default generalized columnwise storage layout. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    Returns:
        int64_t: Unique integer id (for later identification of the tensor channel).

    .. seealso:: `cutensornetStateApplyGeneralChannel`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _state_modes_
    get_resource_ptr[int32_t](_state_modes_, state_modes, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _tensor_data_
    get_resource_ptrs[void](_tensor_data_, tensor_data, <void*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _tensor_mode_strides_
    get_resource_ptr[int64_t](_tensor_mode_strides_, tensor_mode_strides, <int64_t*>NULL)
    cdef int64_t channel_id
    with nogil:
        status = cutensornetStateApplyGeneralChannel(<const Handle>handle, <State>tensor_network_state, num_state_modes, <const int32_t*>(_state_modes_.data()), num_tensors, <void**>(_tensor_data_.data()), <const int64_t*>(_tensor_mode_strides_.data()), &channel_id)
    check_status(status)
    return channel_id


# for backward compat
contraction_optimizer_config_get_attribute_dtype = get_contraction_optimizer_config_attribute_dtype
contraction_optimizer_info_get_attribute_dtype = get_contraction_optimizer_info_attribute_dtype
contraction_autotune_preference_get_attribute_dtype = get_contraction_autotune_preference_attribute_dtype
tensor_svd_config_get_attribute_dtype = get_tensor_svd_config_attribute_dtype
tensor_svd_info_get_attribute_dtype = get_tensor_svd_info_attribute_dtype
network_get_attribute_dtype = get_network_attribute_dtype
marginal_get_attribute_dtype = get_marginal_attribute_dtype
sampler_get_attribute_dtype = get_sampler_attribute_dtype
state_get_attribute_dtype = get_state_attribute_dtype
accessor_get_attribute_dtype = get_accessor_attribute_dtype
expectation_get_attribute_dtype = get_expectation_attribute_dtype
MAJOR_VER = CUTENSORNET_MAJOR
MINOR_VER = CUTENSORNET_MINOR
PATCH_VER = CUTENSORNET_PATCH
VERSION = CUTENSORNET_VERSION
ALLOCATOR_NAME_LEN = CUTENSORNET_ALLOCATOR_NAME_LEN


cpdef tuple get_tensor_details(intptr_t handle, intptr_t desc):
    """Get the tensor's metadata.

    Args:
        handle (intptr_t): The library handle.
        desc (intptr_t): A tensor descriptor.

    Returns:
        tuple:
            The metadata of the tensor: ``(num_modes, modes, extents,
            strides)``.

    .. seealso:: `cutensornetGetTensorDetails`

    """
    cdef int32_t numModesOut = 0
    with nogil:
        status = cutensornetGetTensorDetails(
            <Handle>handle, <TensorDescriptor>desc,
            &numModesOut, NULL, NULL, NULL, NULL)
    check_status(status)
    modes = _numpy.empty(numModesOut, dtype=_numpy.int32)
    extents = _numpy.empty(numModesOut, dtype=_numpy.int64)
    strides = _numpy.empty(numModesOut, dtype=_numpy.int64)
    cdef int32_t* mPtr = <int32_t*><intptr_t>modes.ctypes.data
    cdef int64_t* ePtr = <int64_t*><intptr_t>extents.ctypes.data
    cdef int64_t* sPtr = <int64_t*><intptr_t>strides.ctypes.data
    with nogil:
        status = cutensornetGetTensorDetails(
            <Handle>handle, <TensorDescriptor>desc,
            &numModesOut, NULL, mPtr, ePtr, sPtr)
    check_status(status)
    return (numModesOut, modes, extents, strides)


######################### Python specific utility #########################

cdef dict svd_algo_params_sizes = {
    CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ: gesvdj_params_dtype,
    CUTENSORNET_TENSOR_SVD_ALGO_GESVDR: gesvdr_params_dtype
}

cpdef tensor_svd_algo_params_get_dtype(int svd_algo):
    """Get the Python data type of the corresponding tensor SVD parameters attribute.

    Args:
        svd_algo (TensorSVDAlgo): The SVD algorithm to query.

    Returns:
        The data type of algorithm parameters for the queried SVD algorithm. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for `CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS`.
    """
    if svd_algo not in svd_algo_params_sizes:
        raise ValueError(f"Algorithm {svd_algo} does not support tunable parameters.")
    return svd_algo_params_sizes[svd_algo]


cdef dict svd_algo_status_sizes = {
    CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ: gesvdj_status_dtype,
    CUTENSORNET_TENSOR_SVD_ALGO_GESVDP: gesvdp_status_dtype
}


cpdef tensor_svd_algo_status_get_dtype(int svd_algo):
    """Get the Python data type of the corresponding tensor SVD status attribute.
    
    Args:
        svd_algo (TensorSVDAlgo): The SVD algorithm to query.
    
    Returns:
        The data type of algorithm status for the queried SVD algorithm. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for `CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS`.
    """
    if svd_algo not in svd_algo_status_sizes:
        raise ValueError(f"Algorithm {svd_algo} does not support tunable parameters.")
    return svd_algo_status_sizes[svd_algo]

###########################################################################


cpdef intptr_t create_slice_group_from_ids(
        intptr_t handle, ids, size_t ids_size) except*:
    """Create a slice group from a sequence of slice IDs.

    Args:
        handle (intptr_t): The library handle.
        ids: A host sequence of slice IDs (as Python :class:`int`). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        ids_size (size_t): the length of the ID sequence.

    Returns:
        intptr_t: An opaque slice group descriptor.

    .. seealso:: `cutensornetCreateSliceGroupFromIDs`
    """
    # ids can be a pointer address, or a Python sequence
    cdef vector[int64_t] IDsData
    cdef int64_t* IdsPtr
    cdef size_t size
    if cpython.PySequence_Check(ids):
        IDsData = ids
        IDsPtr = <int64_t*>(IDsData.data())
        size = IDsData.size()
        assert size == ids_size
    else:  # a pointer address
        IDsPtr = <int64_t*><intptr_t>ids
        size = ids_size

    cdef SliceGroup slice_group
    with nogil:
        status = cutensornetCreateSliceGroupFromIDs(
            <Handle>handle, IDsPtr, IDsPtr+size, &slice_group)
    check_status(status)
    return <intptr_t>slice_group


cpdef tuple state_compute(
        intptr_t handle, intptr_t state, intptr_t workspace,
        state_tensors_out, intptr_t stream):
    """Computes the tensor network state representation.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.
        workspace (intptr_t): The workspace descriptor.
        state_tensors_out: A host array of pointer addresses (as Python :class:`int`) for
            each output tensor (on device). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    Returns:
        tuple:
            The metadata of the output tensors: ``(extents_out, strides_out)``.

    .. seealso:: `cutensornetStateCompute`
    """
    cdef int32_t num_tensors = 0
    with nogil:
        status = cutensornetGetOutputStateDetails(
            <Handle>handle, <State>state,
            &num_tensors, NULL, NULL, NULL)
    check_status(status)

    num_modes = _numpy.empty(num_tensors, dtype=_numpy.int32)
    cdef int32_t* numModesPtr = <int32_t*><intptr_t>num_modes.ctypes.data
    with nogil:
        status = cutensornetGetOutputStateDetails(
            <Handle>handle, <State>state,
            &num_tensors, numModesPtr, NULL, NULL)
    check_status(status)

    extents_out_py = [_numpy.empty(num_modes[i], dtype=_numpy.int64) for i in range(num_tensors)]
    strides_out_py = [_numpy.empty(num_modes[i], dtype=_numpy.int64) for i in range(num_tensors)]

    cdef vector[intptr_t] extentsOut
    cdef vector[intptr_t] stridesOut
    for i in range(num_tensors):
        extentsOut.push_back(<intptr_t>extents_out_py[i].ctypes.data)
        stridesOut.push_back(<intptr_t>strides_out_py[i].ctypes.data)

    cdef int64_t** extentsOutPtr = <int64_t**>(extentsOut.data())
    cdef int64_t** stridesOutPtr = <int64_t**>(stridesOut.data())

    cdef vector[intptr_t] stateTensorsOutData
    cdef void** stateTensorsOutPtr
    if cpython.PySequence_Check(state_tensors_out):
        stateTensorsOutData = state_tensors_out
        stateTensorsOutPtr = <void**>(stateTensorsOutData.data())
    else:  # a pointer address
        stateTensorsOutPtr = <void**><intptr_t>state_tensors_out

    with nogil:
        status = cutensornetStateCompute(
            <Handle>handle, <State>state, <WorkspaceDescriptor>workspace,
            extentsOutPtr, stridesOutPtr, stateTensorsOutPtr, <Stream>stream)
    check_status(status)
    return (extents_out_py, strides_out_py)


cpdef tuple get_output_state_details(intptr_t handle, intptr_t state):
    """Get the output state tensors' metadata.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.

    Returns:
        tuple:
            The metadata of the output tensor: ``(num_tensors, num_modes, extents,
            strides)``.

    .. seealso:: `cutensornetGetOutputStateDetails`
    """
    cdef int32_t num_tensors = 0
    with nogil:
        status = cutensornetGetOutputStateDetails(
            <Handle>handle, <State>state,
            &num_tensors, NULL, NULL, NULL)
    check_status(status)

    num_modes = _numpy.empty(num_tensors, dtype=_numpy.int32)
    cdef int32_t* numModesPtr = <int32_t*><intptr_t>num_modes.ctypes.data
    with nogil:
        status = cutensornetGetOutputStateDetails(
            <Handle>handle, <State>state,
            &num_tensors, numModesPtr, NULL, NULL)
    check_status(status)
    extents_out_py = [_numpy.empty(num_modes[i], dtype=_numpy.int64) for i in range(num_tensors)]
    strides_out_py = [_numpy.empty(num_modes[i], dtype=_numpy.int64) for i in range(num_tensors)]

    cdef vector[intptr_t] extentsOut
    cdef vector[intptr_t] stridesOut
    for i in range(num_tensors):
        extentsOut.push_back(<intptr_t>extents_out_py[i].ctypes.data)
        stridesOut.push_back(<intptr_t>strides_out_py[i].ctypes.data)

    cdef int64_t** extentsOutPtr = <int64_t**>(extentsOut.data())
    cdef int64_t** stridesOutPtr = <int64_t**>(stridesOut.data())
    with nogil:
        status = cutensornetGetOutputStateDetails(
            <Handle>handle, <State>state,
            &num_tensors, NULL, extentsOutPtr, stridesOutPtr)
    check_status(status)
    return (num_tensors, num_modes, extents_out_py, strides_out_py)


cpdef set_device_mem_handler(intptr_t handle, handler):
    """ Set the device memory handler for cuTensorNet.

    The ``handler`` object can be passed in multiple ways:

      - If ``handler`` is an :class:`int`, it refers to the address of a fully
        initialized `cutensornetDeviceMemHandler_t` struct.
      - If ``handler`` is a Python sequence:

        - If ``handler`` is a sequence of length 4, it is interpreted as ``(ctx, device_alloc,
          device_free, name)``, where the first three elements are the pointer
          addresses (:class:`int`) of the corresponding members. ``name`` is a
          :class:`str` as the name of the handler.
        - If ``handler`` is a sequence of length 3, it is interpreted as ``(malloc, free,
          name)``, where the first two objects are Python *callables* with the
          following calling convention:

            - ``ptr = malloc(size, stream)``
            - ``free(ptr, size, stream)``

          with all arguments and return value (``ptr``) being Python :class:`int`.
          ``name`` is the same as above.

    .. note:: Only when ``handler`` is a length-3 sequence will the GIL be
        held whenever a routine requires memory allocation and deallocation,
        so for all other cases be sure your ``handler`` does not manipulate
        any Python objects.

    Args:
        handle (intptr_t): The library handle.
        handler: The memory handler object, see above.

    .. seealso:: `cutensornetSetDeviceMemHandler`
    """
    cdef bytes name
    cdef _DeviceMemHandler our_handler
    cdef _DeviceMemHandler* handlerPtr = &our_handler

    if isinstance(handler, int):
        handlerPtr = <_DeviceMemHandler*><intptr_t>handler
    elif cpython.PySequence_Check(handler):
        name = handler[-1].encode('ascii')
        if len(name) > CUTENSORNET_ALLOCATOR_NAME_LEN:
            raise ValueError("the handler name is too long")
        our_handler.name[:len(name)] = name
        our_handler.name[len(name)] = 0

        if len(handler) == 4:
            # handler = (ctx_ptr, malloc_ptr, free_ptr, name)
            assert (isinstance(handler[1], int) and isinstance(handler[2], int))
            our_handler.ctx = <void*><intptr_t>(handler[0])
            our_handler.device_alloc = <DeviceAllocType><intptr_t>(handler[1])
            our_handler.device_free = <DeviceFreeType><intptr_t>(handler[2])
        elif len(handler) == 3:
            # handler = (malloc, free, name)
            assert (callable(handler[0]) and callable(handler[1]))
            ctx = (handler[0], handler[1])
            owner_pyobj[handle] = ctx  # keep it alive
            our_handler.ctx = <void*>ctx
            our_handler.device_alloc = cuqnt_alloc_wrapper
            our_handler.device_free = cuqnt_free_wrapper
        else:
            raise ValueError("handler must be a sequence of length 3 or 4, "
                             "see the documentation for detail")
    else:
        raise NotImplementedError("handler format not recognized")

    with nogil:
        status = cutensornetSetDeviceMemHandler(<Handle>handle, handlerPtr)
    check_status(status)


cpdef tuple get_device_mem_handler(intptr_t handle):
    """ Get the device memory handler for cuTensorNet.

    Args:
        handle (intptr_t): The library handle.

    Returns:
        tuple:
            The ``handler`` object, which has two forms:

              - If ``handler`` is a 3-tuple, it is interpreted as ``(malloc, free,
                name)``, where the first two objects are Python *callables*, and ``name``
                is the name of the handler. This 3-tuple handler would be compared equal
                (elementwisely) to the one previously passed to :func:`set_device_mem_handler`.
              - If ``handler`` is a 4-tuple, it is interpreted as ``(ctx, device_alloc,
                device_free, name)``, where the first three elements are the pointer
                addresses (:class:`int`) of the corresponding members. ``name`` is the
                same as above.

    .. seealso:: `cutensornetGetDeviceMemHandler`
    """
    cdef _DeviceMemHandler handler
    with nogil:
        status = cutensornetGetDeviceMemHandler(<Handle>handle, &handler)
    check_status(status)

    cdef tuple ctx
    cdef bytes name = handler.name
    if (handler.device_alloc == cuqnt_alloc_wrapper and
            handler.device_free == cuqnt_free_wrapper):
        ctx = <object>(handler.ctx)
        return (ctx[0], ctx[1], name.decode('ascii'))
    else:
        # TODO: consider other possibilities?
        return (<intptr_t>handler.ctx,
                <intptr_t>handler.device_alloc,
                <intptr_t>handler.device_free,
                name.decode('ascii'))


# can't be cpdef because args & kwargs can't be handled in a C signature
def logger_set_callback_data(callback, *args, **kwargs):
    """Set the logger callback along with arguments.

    Args:
        callback: A Python callable with the following signature (no return):

          - ``callback(log_level, func_name, message, *args, **kwargs)``

          where ``log_level`` (:class:`int`), ``func_name`` (`str`), and
          ``message`` (`str`) are provided by the logger API.

    .. seealso:: `cutensornetLoggerSetCallbackData`
    """
    func_arg = (callback, args, kwargs)
    # if only set once, the callback lifetime should be as long as this module,
    # because we don't know when the logger is done using it
    global logger_callback_holder
    logger_callback_holder = func_arg
    with nogil:
        status = cutensornetLoggerSetCallbackData(
            <LoggerCallbackData>logger_callback_with_data, <void*>(func_arg))
    check_status(status)


# who owns a reference to user-provided Python objects (k: owner, v: object)
cdef dict owner_pyobj = {}
