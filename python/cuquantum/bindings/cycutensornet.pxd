# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 25.03.0. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.stdio cimport FILE


###############################################################################
# Types (structs, enums, ...)
###############################################################################

cdef extern from *:
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef int cudaDataType_t 'cudaDataType_t'
    ctypedef int cudaDataType 'cudaDataType'
    ctypedef int libraryPropertyType_t 'libraryPropertyType_t'
    ctypedef int libraryPropertyType 'libraryPropertyType'

    ctypedef struct cuDoubleComplex:
        double x
        double y


cdef extern from '<cutensornet.h>' nogil:
    # enums
    ctypedef enum cutensornetStatus_t:
        CUTENSORNET_STATUS_SUCCESS
        CUTENSORNET_STATUS_NOT_INITIALIZED
        CUTENSORNET_STATUS_ALLOC_FAILED
        CUTENSORNET_STATUS_INVALID_VALUE
        CUTENSORNET_STATUS_ARCH_MISMATCH
        CUTENSORNET_STATUS_MAPPING_ERROR
        CUTENSORNET_STATUS_EXECUTION_FAILED
        CUTENSORNET_STATUS_INTERNAL_ERROR
        CUTENSORNET_STATUS_NOT_SUPPORTED
        CUTENSORNET_STATUS_LICENSE_ERROR
        CUTENSORNET_STATUS_CUBLAS_ERROR
        CUTENSORNET_STATUS_CUDA_ERROR
        CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE
        CUTENSORNET_STATUS_INSUFFICIENT_DRIVER
        CUTENSORNET_STATUS_IO_ERROR
        CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH
        CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR
        CUTENSORNET_STATUS_ALL_HYPER_SAMPLES_FAILED
        CUTENSORNET_STATUS_CUSOLVER_ERROR
        CUTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR
        CUTENSORNET_STATUS_DISTRIBUTED_FAILURE
        CUTENSORNET_STATUS_INTERRUPTED

    ctypedef enum cutensornetComputeType_t:
        CUTENSORNET_COMPUTE_16F
        CUTENSORNET_COMPUTE_16BF
        CUTENSORNET_COMPUTE_TF32
        CUTENSORNET_COMPUTE_3XTF32
        CUTENSORNET_COMPUTE_32F
        CUTENSORNET_COMPUTE_64F
        CUTENSORNET_COMPUTE_8U
        CUTENSORNET_COMPUTE_8I
        CUTENSORNET_COMPUTE_32U
        CUTENSORNET_COMPUTE_32I

    ctypedef enum cutensornetGraphAlgo_t:
        CUTENSORNET_GRAPH_ALGO_RB
        CUTENSORNET_GRAPH_ALGO_KWAY

    ctypedef enum cutensornetMemoryModel_t:
        CUTENSORNET_MEMORY_MODEL_HEURISTIC
        CUTENSORNET_MEMORY_MODEL_CUTENSOR

    ctypedef enum cutensornetOptimizerCost_t:
        CUTENSORNET_OPTIMIZER_COST_FLOPS
        CUTENSORNET_OPTIMIZER_COST_TIME
        CUTENSORNET_OPTIMIZER_COST_TIME_TUNED

    ctypedef enum cutensornetContractionOptimizerConfigAttributes_t:
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_PARTITIONS
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_CUTOFF_SIZE
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_ITERATIONS
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_SLICE_FACTOR
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION

    ctypedef enum cutensornetContractionOptimizerInfoAttributes_t:
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_OVERHEAD
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PHASE1_FLOP_COUNT
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS_EST
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_RUNTIME_EST
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_INTERMEDIATE_MODES

    ctypedef enum cutensornetContractionAutotunePreferenceAttributes_t:
        CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS
        CUTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES

    ctypedef enum cutensornetWorksizePref_t:
        CUTENSORNET_WORKSIZE_PREF_MIN
        CUTENSORNET_WORKSIZE_PREF_RECOMMENDED
        CUTENSORNET_WORKSIZE_PREF_MAX

    ctypedef enum cutensornetMemspace_t:
        CUTENSORNET_MEMSPACE_DEVICE
        CUTENSORNET_MEMSPACE_HOST

    ctypedef enum cutensornetWorkspaceKind_t:
        CUTENSORNET_WORKSPACE_SCRATCH
        CUTENSORNET_WORKSPACE_CACHE

    ctypedef enum cutensornetTensorSVDConfigAttributes_t:
        CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF
        CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF
        CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION
        CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION
        CUTENSORNET_TENSOR_SVD_CONFIG_ALGO
        CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS
        CUTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF

    ctypedef enum cutensornetTensorSVDPartition_t:
        CUTENSORNET_TENSOR_SVD_PARTITION_NONE
        CUTENSORNET_TENSOR_SVD_PARTITION_US
        CUTENSORNET_TENSOR_SVD_PARTITION_SV
        CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL

    ctypedef enum cutensornetTensorSVDNormalization_t:
        CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE
        CUTENSORNET_TENSOR_SVD_NORMALIZATION_L1
        CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2
        CUTENSORNET_TENSOR_SVD_NORMALIZATION_LINF

    ctypedef enum cutensornetTensorSVDInfoAttributes_t:
        CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT
        CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT
        CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT
        CUTENSORNET_TENSOR_SVD_INFO_ALGO
        CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS

    ctypedef enum cutensornetGateSplitAlgo_t:
        CUTENSORNET_GATE_SPLIT_ALGO_DIRECT
        CUTENSORNET_GATE_SPLIT_ALGO_REDUCED

    ctypedef enum cutensornetNetworkAttributes_t:
        CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONSTANT
        CUTENSORNET_NETWORK_INPUT_TENSORS_CONSTANT
        CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONJUGATED
        CUTENSORNET_NETWORK_INPUT_TENSORS_CONJUGATED
        CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_REQUIRE_GRAD
        CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD

    ctypedef enum cutensornetSmartOption_t:
        CUTENSORNET_SMART_OPTION_DISABLED
        CUTENSORNET_SMART_OPTION_ENABLED

    ctypedef enum cutensornetTensorSVDAlgo_t:
        CUTENSORNET_TENSOR_SVD_ALGO_GESVD
        CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ
        CUTENSORNET_TENSOR_SVD_ALGO_GESVDP
        CUTENSORNET_TENSOR_SVD_ALGO_GESVDR

    ctypedef enum cutensornetStatePurity_t:
        CUTENSORNET_STATE_PURITY_PURE

    ctypedef enum cutensornetMarginalAttributes_t:
        CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES
        CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES
        CUTENSORNET_MARGINAL_INFO_FLOPS

    ctypedef enum cutensornetSamplerAttributes_t:
        CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES
        CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES
        CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC
        CUTENSORNET_SAMPLER_INFO_FLOPS

    ctypedef enum cutensornetAccessorAttributes_t:
        CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES
        CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES
        CUTENSORNET_ACCESSOR_INFO_FLOPS

    ctypedef enum cutensornetExpectationAttributes_t:
        CUTENSORNET_EXPECTATION_OPT_NUM_HYPER_SAMPLES
        CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES
        CUTENSORNET_EXPECTATION_INFO_FLOPS

    ctypedef enum cutensornetBoundaryCondition_t:
        CUTENSORNET_BOUNDARY_CONDITION_OPEN

    ctypedef enum cutensornetStateAttributes_t:
        CUTENSORNET_STATE_MPS_CANONICAL_CENTER
        CUTENSORNET_STATE_MPS_SVD_CONFIG_ABS_CUTOFF
        CUTENSORNET_STATE_MPS_SVD_CONFIG_REL_CUTOFF
        CUTENSORNET_STATE_MPS_SVD_CONFIG_S_NORMALIZATION
        CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO
        CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO_PARAMS
        CUTENSORNET_STATE_MPS_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF
        CUTENSORNET_STATE_NUM_HYPER_SAMPLES
        CUTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER
        CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF
        CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF
        CUTENSORNET_STATE_CONFIG_MPS_SVD_S_NORMALIZATION
        CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO
        CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO_PARAMS
        CUTENSORNET_STATE_CONFIG_MPS_SVD_DISCARDED_WEIGHT_CUTOFF
        CUTENSORNET_STATE_CONFIG_MPS_MPO_APPLICATION
        CUTENSORNET_STATE_CONFIG_MPS_GAUGE_OPTION
        CUTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES
        CUTENSORNET_STATE_INFO_FLOPS

    ctypedef enum cutensornetStateMPOApplication_t:
        CUTENSORNET_STATE_MPO_APPLICATION_INEXACT
        CUTENSORNET_STATE_MPO_APPLICATION_EXACT

    ctypedef enum cutensornetStateMPSGaugeOption_t:
        CUTENSORNET_STATE_MPS_GAUGE_FREE
        CUTENSORNET_STATE_MPS_GAUGE_SIMPLE

    # types
    ctypedef void* cutensornetNetworkDescriptor_t 'cutensornetNetworkDescriptor_t'
    ctypedef void* cutensornetContractionPlan_t 'cutensornetContractionPlan_t'
    ctypedef void* cutensornetHandle_t 'cutensornetHandle_t'
    ctypedef void* cutensornetWorkspaceDescriptor_t 'cutensornetWorkspaceDescriptor_t'
    ctypedef void* cutensornetContractionOptimizerConfig_t 'cutensornetContractionOptimizerConfig_t'
    ctypedef void* cutensornetContractionOptimizerInfo_t 'cutensornetContractionOptimizerInfo_t'
    ctypedef void* cutensornetContractionAutotunePreference_t 'cutensornetContractionAutotunePreference_t'
    ctypedef void* cutensornetSliceGroup_t 'cutensornetSliceGroup_t'
    ctypedef void* cutensornetTensorDescriptor_t 'cutensornetTensorDescriptor_t'
    ctypedef void* cutensornetTensorSVDConfig_t 'cutensornetTensorSVDConfig_t'
    ctypedef void* cutensornetTensorSVDInfo_t 'cutensornetTensorSVDInfo_t'
    ctypedef void* cutensornetState_t 'cutensornetState_t'
    ctypedef void* cutensornetStateMarginal_t 'cutensornetStateMarginal_t'
    ctypedef void* cutensornetStateSampler_t 'cutensornetStateSampler_t'
    ctypedef void* cutensornetStateAccessor_t 'cutensornetStateAccessor_t'
    ctypedef void* cutensornetStateExpectation_t 'cutensornetStateExpectation_t'
    ctypedef void* cutensornetNetworkOperator_t 'cutensornetNetworkOperator_t'
    ctypedef struct cutensornetNodePair_t 'cutensornetNodePair_t':
        int32_t first
        int32_t second
    ctypedef struct cutensornetSliceInfoPair_t 'cutensornetSliceInfoPair_t':
        int32_t slicedMode
        int64_t slicedExtent
    ctypedef struct cutensornetTensorQualifiers_t 'cutensornetTensorQualifiers_t':
        int32_t isConjugate
        int32_t isConstant
        int32_t requiresGradient
    ctypedef struct cutensornetDeviceMemHandler_t 'cutensornetDeviceMemHandler_t':
        void* ctx
        int (*device_alloc)(void*, void**, size_t, cudaStream_t)
        int (*device_free)(void*, void*, size_t, cudaStream_t)
        char name[64]
    ctypedef struct cutensornetDistributedCommunicator_t 'cutensornetDistributedCommunicator_t':
        void* commPtr
        size_t commSize
    ctypedef struct cutensornetDistributedInterface_t 'cutensornetDistributedInterface_t':
        int version
        int (*getNumRanks)(const cutensornetDistributedCommunicator_t*, int32_t*)
        int (*getNumRanksShared)(const cutensornetDistributedCommunicator_t*, int32_t*)
        int (*getProcRank)(const cutensornetDistributedCommunicator_t*, int32_t*)
        int (*Barrier)(const cutensornetDistributedCommunicator_t*)
        int (*Bcast)(const cutensornetDistributedCommunicator_t*, void*, int32_t, cudaDataType_t, int32_t)
        int (*Allreduce)(const cutensornetDistributedCommunicator_t*, const void*, void*, int32_t, cudaDataType_t)
        int (*AllreduceInPlace)(const cutensornetDistributedCommunicator_t*, void*, int32_t, cudaDataType_t)
        int (*AllreduceInPlaceMin)(const cutensornetDistributedCommunicator_t*, void*, int32_t, cudaDataType_t)
        int (*AllreduceDoubleIntMinloc)(const cutensornetDistributedCommunicator_t*, const void*, void*)
        int (*Allgather)(const cutensornetDistributedCommunicator_t*, const void*, void*, int32_t, cudaDataType_t)
    ctypedef struct cutensornetTensorIDList_t 'cutensornetTensorIDList_t':
        int32_t numTensors
        int32_t* data
    ctypedef struct cutensornetGesvdjParams_t 'cutensornetGesvdjParams_t':
        double tol
        int32_t maxSweeps
    ctypedef struct cutensornetGesvdrParams_t 'cutensornetGesvdrParams_t':
        int64_t oversampling
        int64_t niters
    ctypedef struct cutensornetGesvdjStatus_t 'cutensornetGesvdjStatus_t':
        double residual
        int32_t sweeps
    ctypedef struct cutensornetGesvdpStatus_t 'cutensornetGesvdpStatus_t':
        double errSigma
    ctypedef void (*cutensornetLoggerCallback_t 'cutensornetLoggerCallback_t')(
        int32_t logLevel,
        const char* functionName,
        const char* message
    )
    ctypedef void (*cutensornetLoggerCallbackData_t 'cutensornetLoggerCallbackData_t')(
        int32_t logLevel,
        const char* functionName,
        const char* message,
        void* userData
    )
    ctypedef struct cutensornetContractionPath_t 'cutensornetContractionPath_t':
        int32_t numContractions
        cutensornetNodePair_t* data
    ctypedef struct cutensornetSlicingConfig_t 'cutensornetSlicingConfig_t':
        uint32_t numSlicedModes
        cutensornetSliceInfoPair_t* data

    # constants
    const int CUTENSORNET_ALLOCATOR_NAME_LEN
    const int CUTENSORNET_MAJOR
    const int CUTENSORNET_MINOR
    const int CUTENSORNET_PATCH
    const int CUTENSORNET_VERSION


###############################################################################
# Functions
###############################################################################

cdef cutensornetStatus_t cutensornetCreate(cutensornetHandle_t* handle) except* nogil
cdef cutensornetStatus_t cutensornetDestroy(cutensornetHandle_t handle) except* nogil
cdef cutensornetStatus_t cutensornetCreateNetworkDescriptor(const cutensornetHandle_t handle, int32_t numInputs, const int32_t numModesIn[], const int64_t* const extentsIn[], const int64_t* const stridesIn[], const int32_t* const modesIn[], const cutensornetTensorQualifiers_t qualifiersIn[], int32_t numModesOut, const int64_t extentsOut[], const int64_t stridesOut[], const int32_t modesOut[], cudaDataType_t dataType, cutensornetComputeType_t computeType, cutensornetNetworkDescriptor_t* descNet) except* nogil
cdef cutensornetStatus_t cutensornetDestroyNetworkDescriptor(cutensornetNetworkDescriptor_t desc) except* nogil
cdef cutensornetStatus_t cutensornetGetOutputTensorDescriptor(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, cutensornetTensorDescriptor_t* outputTensorDesc) except* nogil
cdef cutensornetStatus_t cutensornetGetTensorDetails(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t tensorDesc, int32_t* numModes, size_t* dataSize, int32_t* modeLabels, int64_t* extents, int64_t* strides) except* nogil
cdef cutensornetStatus_t cutensornetCreateWorkspaceDescriptor(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t* workDesc) except* nogil
cdef cutensornetStatus_t cutensornetWorkspaceComputeContractionSizes(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetWorkspaceDescriptor_t workDesc) except* nogil
cdef cutensornetStatus_t cutensornetWorkspaceGetMemorySize(const cutensornetHandle_t handle, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetWorksizePref_t workPref, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, int64_t* memorySize) except* nogil
cdef cutensornetStatus_t cutensornetWorkspaceSetMemory(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, void* const memoryPtr, int64_t memorySize) except* nogil
cdef cutensornetStatus_t cutensornetWorkspaceGetMemory(const cutensornetHandle_t handle, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, void** memoryPtr, int64_t* memorySize) except* nogil
cdef cutensornetStatus_t cutensornetDestroyWorkspaceDescriptor(cutensornetWorkspaceDescriptor_t desc) except* nogil
cdef cutensornetStatus_t cutensornetCreateContractionOptimizerConfig(const cutensornetHandle_t handle, cutensornetContractionOptimizerConfig_t* optimizerConfig) except* nogil
cdef cutensornetStatus_t cutensornetDestroyContractionOptimizerConfig(cutensornetContractionOptimizerConfig_t optimizerConfig) except* nogil
cdef cutensornetStatus_t cutensornetContractionOptimizerConfigGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionOptimizerConfig_t optimizerConfig, cutensornetContractionOptimizerConfigAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetContractionOptimizerConfigSetAttribute(const cutensornetHandle_t handle, cutensornetContractionOptimizerConfig_t optimizerConfig, cutensornetContractionOptimizerConfigAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetDestroyContractionOptimizerInfo(cutensornetContractionOptimizerInfo_t optimizerInfo) except* nogil
cdef cutensornetStatus_t cutensornetCreateContractionOptimizerInfo(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, cutensornetContractionOptimizerInfo_t* optimizerInfo) except* nogil
cdef cutensornetStatus_t cutensornetContractionOptimize(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const cutensornetContractionOptimizerConfig_t optimizerConfig, uint64_t workspaceSizeConstraint, cutensornetContractionOptimizerInfo_t optimizerInfo) except* nogil
cdef cutensornetStatus_t cutensornetContractionOptimizerInfoGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetContractionOptimizerInfoAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetContractionOptimizerInfoSetAttribute(const cutensornetHandle_t handle, cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetContractionOptimizerInfoAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetContractionOptimizerInfoGetPackedSize(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, size_t* sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetContractionOptimizerInfoPackData(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, void* buffer, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetCreateContractionOptimizerInfoFromPackedData(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const void* buffer, size_t sizeInBytes, cutensornetContractionOptimizerInfo_t* optimizerInfo) except* nogil
cdef cutensornetStatus_t cutensornetUpdateContractionOptimizerInfoFromPackedData(const cutensornetHandle_t handle, const void* buffer, size_t sizeInBytes, cutensornetContractionOptimizerInfo_t optimizerInfo) except* nogil
cdef cutensornetStatus_t cutensornetCreateContractionPlan(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const cutensornetContractionOptimizerInfo_t optimizerInfo, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetContractionPlan_t* plan) except* nogil
cdef cutensornetStatus_t cutensornetDestroyContractionPlan(cutensornetContractionPlan_t plan) except* nogil
cdef cutensornetStatus_t cutensornetContractionAutotune(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan, const void* const rawDataIn[], void* rawDataOut, cutensornetWorkspaceDescriptor_t workDesc, const cutensornetContractionAutotunePreference_t pref, cudaStream_t stream) except* nogil
cdef cutensornetStatus_t cutensornetCreateContractionAutotunePreference(const cutensornetHandle_t handle, cutensornetContractionAutotunePreference_t* autotunePreference) except* nogil
cdef cutensornetStatus_t cutensornetContractionAutotunePreferenceGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionAutotunePreference_t autotunePreference, cutensornetContractionAutotunePreferenceAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetContractionAutotunePreferenceSetAttribute(const cutensornetHandle_t handle, cutensornetContractionAutotunePreference_t autotunePreference, cutensornetContractionAutotunePreferenceAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetDestroyContractionAutotunePreference(cutensornetContractionAutotunePreference_t autotunePreference) except* nogil
cdef cutensornetStatus_t cutensornetCreateSliceGroupFromIDRange(const cutensornetHandle_t handle, int64_t sliceIdStart, int64_t sliceIdStop, int64_t sliceIdStep, cutensornetSliceGroup_t* sliceGroup) except* nogil
cdef cutensornetStatus_t cutensornetCreateSliceGroupFromIDs(const cutensornetHandle_t handle, const int64_t* beginIDSequence, const int64_t* endIDSequence, cutensornetSliceGroup_t* sliceGroup) except* nogil
cdef cutensornetStatus_t cutensornetDestroySliceGroup(cutensornetSliceGroup_t sliceGroup) except* nogil
cdef cutensornetStatus_t cutensornetContractSlices(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan, const void* const rawDataIn[], void* rawDataOut, int32_t accumulateOutput, cutensornetWorkspaceDescriptor_t workDesc, const cutensornetSliceGroup_t sliceGroup, cudaStream_t stream) except* nogil
cdef cutensornetStatus_t cutensornetCreateTensorDescriptor(const cutensornetHandle_t handle, int32_t numModes, const int64_t extents[], const int64_t strides[], const int32_t modes[], cudaDataType_t dataType, cutensornetTensorDescriptor_t* descTensor) except* nogil
cdef cutensornetStatus_t cutensornetDestroyTensorDescriptor(cutensornetTensorDescriptor_t descTensor) except* nogil
cdef cutensornetStatus_t cutensornetCreateTensorSVDConfig(const cutensornetHandle_t handle, cutensornetTensorSVDConfig_t* svdConfig) except* nogil
cdef cutensornetStatus_t cutensornetDestroyTensorSVDConfig(cutensornetTensorSVDConfig_t svdConfig) except* nogil
cdef cutensornetStatus_t cutensornetTensorSVDConfigGetAttribute(const cutensornetHandle_t handle, const cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDConfigAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetTensorSVDConfigSetAttribute(const cutensornetHandle_t handle, cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDConfigAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetWorkspaceComputeSVDSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const cutensornetTensorDescriptor_t descTensorU, const cutensornetTensorDescriptor_t descTensorV, const cutensornetTensorSVDConfig_t svdConfig, cutensornetWorkspaceDescriptor_t workDesc) except* nogil
cdef cutensornetStatus_t cutensornetWorkspaceComputeQRSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const cutensornetTensorDescriptor_t descTensorQ, const cutensornetTensorDescriptor_t descTensorR, cutensornetWorkspaceDescriptor_t workDesc) except* nogil
cdef cutensornetStatus_t cutensornetCreateTensorSVDInfo(const cutensornetHandle_t handle, cutensornetTensorSVDInfo_t* svdInfo) except* nogil
cdef cutensornetStatus_t cutensornetTensorSVDInfoGetAttribute(const cutensornetHandle_t handle, const cutensornetTensorSVDInfo_t svdInfo, cutensornetTensorSVDInfoAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetDestroyTensorSVDInfo(cutensornetTensorSVDInfo_t svdInfo) except* nogil
cdef cutensornetStatus_t cutensornetTensorSVD(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const void* const rawDataIn, cutensornetTensorDescriptor_t descTensorU, void* u, void* s, cutensornetTensorDescriptor_t descTensorV, void* v, const cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDInfo_t svdInfo, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil
cdef cutensornetStatus_t cutensornetTensorQR(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const void* const rawDataIn, const cutensornetTensorDescriptor_t descTensorQ, void* q, const cutensornetTensorDescriptor_t descTensorR, void* r, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil
cdef cutensornetStatus_t cutensornetWorkspaceComputeGateSplitSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA, const cutensornetTensorDescriptor_t descTensorInB, const cutensornetTensorDescriptor_t descTensorInG, const cutensornetTensorDescriptor_t descTensorU, const cutensornetTensorDescriptor_t descTensorV, const cutensornetGateSplitAlgo_t gateAlgo, const cutensornetTensorSVDConfig_t svdConfig, cutensornetComputeType_t computeType, cutensornetWorkspaceDescriptor_t workDesc) except* nogil
cdef cutensornetStatus_t cutensornetGateSplit(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA, const void* rawDataInA, const cutensornetTensorDescriptor_t descTensorInB, const void* rawDataInB, const cutensornetTensorDescriptor_t descTensorInG, const void* rawDataInG, cutensornetTensorDescriptor_t descTensorU, void* u, void* s, cutensornetTensorDescriptor_t descTensorV, void* v, const cutensornetGateSplitAlgo_t gateAlgo, const cutensornetTensorSVDConfig_t svdConfig, cutensornetComputeType_t computeType, cutensornetTensorSVDInfo_t svdInfo, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil
cdef cutensornetStatus_t cutensornetGetDeviceMemHandler(const cutensornetHandle_t handle, cutensornetDeviceMemHandler_t* devMemHandler) except* nogil
cdef cutensornetStatus_t cutensornetSetDeviceMemHandler(cutensornetHandle_t handle, const cutensornetDeviceMemHandler_t* devMemHandler) except* nogil
cdef cutensornetStatus_t cutensornetLoggerSetCallback(cutensornetLoggerCallback_t callback) except* nogil
cdef cutensornetStatus_t cutensornetLoggerSetCallbackData(cutensornetLoggerCallbackData_t callback, void* userData) except* nogil
cdef cutensornetStatus_t cutensornetLoggerSetFile(FILE* file) except* nogil
cdef cutensornetStatus_t cutensornetLoggerOpenFile(const char* logFile) except* nogil
cdef cutensornetStatus_t cutensornetLoggerSetLevel(int32_t level) except* nogil
cdef cutensornetStatus_t cutensornetLoggerSetMask(int32_t mask) except* nogil
cdef cutensornetStatus_t cutensornetLoggerForceDisable() except* nogil
cdef size_t cutensornetGetVersion() except* nogil
cdef size_t cutensornetGetCudartVersion() except* nogil
cdef const char* cutensornetGetErrorString(cutensornetStatus_t error) except* nogil
cdef cutensornetStatus_t cutensornetDistributedResetConfiguration(cutensornetHandle_t handle, const void* commPtr, size_t commSize) except* nogil
cdef cutensornetStatus_t cutensornetDistributedGetNumRanks(const cutensornetHandle_t handle, int32_t* numRanks) except* nogil
cdef cutensornetStatus_t cutensornetDistributedGetProcRank(const cutensornetHandle_t handle, int32_t* procRank) except* nogil
cdef cutensornetStatus_t cutensornetDistributedSynchronize(const cutensornetHandle_t handle) except* nogil
cdef cutensornetStatus_t cutensornetNetworkGetAttribute(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, cutensornetNetworkAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetNetworkSetAttribute(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, cutensornetNetworkAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil
cdef cutensornetStatus_t cutensornetWorkspacePurgeCache(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace) except* nogil
cdef cutensornetStatus_t cutensornetComputeGradientsBackward(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan, const void* const rawDataIn[], const void* outputGradient, void* const gradients[], int32_t accumulateOutput, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil
cdef cutensornetStatus_t cutensornetCreateState(const cutensornetHandle_t handle, cutensornetStatePurity_t purity, int32_t numStateModes, const int64_t* stateModeExtents, cudaDataType_t dataType, cutensornetState_t* tensorNetworkState) except* nogil
cdef cutensornetStatus_t cutensornetStateApplyTensor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except* nogil
cdef cutensornetStatus_t cutensornetStateUpdateTensor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int64_t tensorId, void* tensorData, int32_t unitary) except* nogil
cdef cutensornetStatus_t cutensornetDestroyState(cutensornetState_t tensorNetworkState) except* nogil
cdef cutensornetStatus_t cutensornetCreateMarginal(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numMarginalModes, const int32_t* marginalModes, int32_t numProjectedModes, const int32_t* projectedModes, const int64_t* marginalTensorStrides, cutensornetStateMarginal_t* tensorNetworkMarginal) except* nogil
cdef cutensornetStatus_t cutensornetMarginalConfigure(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, cutensornetMarginalAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil
cdef cutensornetStatus_t cutensornetMarginalPrepare(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil
cdef cutensornetStatus_t cutensornetMarginalCompute(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, const int64_t* projectedModeValues, cutensornetWorkspaceDescriptor_t workDesc, void* marginalTensor, cudaStream_t cudaStream) except* nogil
cdef cutensornetStatus_t cutensornetDestroyMarginal(cutensornetStateMarginal_t tensorNetworkMarginal) except* nogil
cdef cutensornetStatus_t cutensornetCreateSampler(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numModesToSample, const int32_t* modesToSample, cutensornetStateSampler_t* tensorNetworkSampler) except* nogil
cdef cutensornetStatus_t cutensornetSamplerConfigure(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, cutensornetSamplerAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil
cdef cutensornetStatus_t cutensornetSamplerPrepare(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil
cdef cutensornetStatus_t cutensornetSamplerSample(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, int64_t numShots, cutensornetWorkspaceDescriptor_t workDesc, int64_t* samples, cudaStream_t cudaStream) except* nogil
cdef cutensornetStatus_t cutensornetDestroySampler(cutensornetStateSampler_t tensorNetworkSampler) except* nogil
cdef cutensornetStatus_t cutensornetStateFinalizeMPS(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetBoundaryCondition_t boundaryCondition, const int64_t* const extentsOut[], const int64_t* const stridesOut[]) except* nogil
cdef cutensornetStatus_t cutensornetStateConfigure(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetStateAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil
cdef cutensornetStatus_t cutensornetStatePrepare(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil
cdef cutensornetStatus_t cutensornetStateCompute(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetWorkspaceDescriptor_t workDesc, int64_t* extentsOut[], int64_t* stridesOut[], void* stateTensorsOut[], cudaStream_t cudaStream) except* nogil
cdef cutensornetStatus_t cutensornetGetOutputStateDetails(const cutensornetHandle_t handle, const cutensornetState_t tensorNetworkState, int32_t* numTensorsOut, int32_t numModesOut[], int64_t* extentsOut[], int64_t* stridesOut[]) except* nogil
cdef cutensornetStatus_t cutensornetCreateNetworkOperator(const cutensornetHandle_t handle, int32_t numStateModes, const int64_t stateModeExtents[], cudaDataType_t dataType, cutensornetNetworkOperator_t* tensorNetworkOperator) except* nogil
cdef cutensornetStatus_t cutensornetNetworkOperatorAppendProduct(const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator, cuDoubleComplex coefficient, int32_t numTensors, const int32_t numStateModes[], const int32_t* stateModes[], const int64_t* tensorModeStrides[], const void* tensorData[], int64_t* componentId) except* nogil
cdef cutensornetStatus_t cutensornetDestroyNetworkOperator(cutensornetNetworkOperator_t tensorNetworkOperator) except* nogil
cdef cutensornetStatus_t cutensornetCreateAccessor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numProjectedModes, const int32_t* projectedModes, const int64_t* amplitudesTensorStrides, cutensornetStateAccessor_t* tensorNetworkAccessor) except* nogil
cdef cutensornetStatus_t cutensornetAccessorConfigure(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, cutensornetAccessorAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil
cdef cutensornetStatus_t cutensornetAccessorPrepare(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil
cdef cutensornetStatus_t cutensornetAccessorCompute(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, const int64_t* projectedModeValues, cutensornetWorkspaceDescriptor_t workDesc, void* amplitudesTensor, void* stateNorm, cudaStream_t cudaStream) except* nogil
cdef cutensornetStatus_t cutensornetDestroyAccessor(cutensornetStateAccessor_t tensorNetworkAccessor) except* nogil
cdef cutensornetStatus_t cutensornetCreateExpectation(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetNetworkOperator_t tensorNetworkOperator, cutensornetStateExpectation_t* tensorNetworkExpectation) except* nogil
cdef cutensornetStatus_t cutensornetExpectationConfigure(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetExpectationAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil
cdef cutensornetStatus_t cutensornetExpectationPrepare(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil
cdef cutensornetStatus_t cutensornetExpectationCompute(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetWorkspaceDescriptor_t workDesc, void* expectationValue, void* stateNorm, cudaStream_t cudaStream) except* nogil
cdef cutensornetStatus_t cutensornetDestroyExpectation(cutensornetStateExpectation_t tensorNetworkExpectation) except* nogil
cdef cutensornetStatus_t cutensornetStateApplyTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except* nogil
cdef cutensornetStatus_t cutensornetStateApplyControlledTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numControlModes, const int32_t* stateControlModes, const int64_t* stateControlValues, int32_t numTargetModes, const int32_t* stateTargetModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except* nogil
cdef cutensornetStatus_t cutensornetStateUpdateTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int64_t tensorId, void* tensorData, int32_t unitary) except* nogil
cdef cutensornetStatus_t cutensornetStateApplyNetworkOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, const cutensornetNetworkOperator_t tensorNetworkOperator, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* operatorId) except* nogil
cdef cutensornetStatus_t cutensornetStateInitializeMPS(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetBoundaryCondition_t boundaryCondition, const int64_t* const extentsIn[], const int64_t* const stridesIn[], void* stateTensorsIn[]) except* nogil
cdef cutensornetStatus_t cutensornetStateGetInfo(const cutensornetHandle_t handle, const cutensornetState_t tensorNetworkState, cutensornetStateAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil
cdef cutensornetStatus_t cutensornetNetworkOperatorAppendMPO(const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator, cuDoubleComplex coefficient, int32_t numStateModes, const int32_t stateModes[], const int64_t* tensorModeExtents[], const int64_t* tensorModeStrides[], const void* tensorData[], cutensornetBoundaryCondition_t boundaryCondition, int64_t* componentId) except* nogil
cdef cutensornetStatus_t cutensornetAccessorGetInfo(const cutensornetHandle_t handle, const cutensornetStateAccessor_t tensorNetworkAccessor, cutensornetAccessorAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil
cdef cutensornetStatus_t cutensornetExpectationGetInfo(const cutensornetHandle_t handle, const cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetExpectationAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil
cdef cutensornetStatus_t cutensornetMarginalGetInfo(const cutensornetHandle_t handle, const cutensornetStateMarginal_t tensorNetworkMarginal, cutensornetMarginalAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil
cdef cutensornetStatus_t cutensornetSamplerGetInfo(const cutensornetHandle_t handle, const cutensornetStateSampler_t tensorNetworkSampler, cutensornetSamplerAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil
cdef cutensornetStatus_t cutensornetStateApplyUnitaryChannel(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, int32_t numTensors, void* tensorData[], const int64_t* tensorModeStrides, const double probabilities[], int64_t* channelId) except* nogil
cdef cutensornetStatus_t cutensornetStateCaptureMPS(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState) except* nogil
cdef cutensornetStatus_t cutensornetStateApplyGeneralChannel(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, int32_t numTensors, void* tensorData[], const int64_t* tensorModeStrides, int64_t* channelId) except* nogil
