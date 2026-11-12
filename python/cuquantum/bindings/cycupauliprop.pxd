# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 26.09.0 to 26.11.0. Do not modify it directly.
# This layer exposes the C header to Cython as-is.


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    int32_t,
    int64_t,
    uint64_t,
)


# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdio cimport FILE


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum cupaulipropStatus_t "cupaulipropStatus_t":
    CUPAULIPROP_STATUS_SUCCESS "CUPAULIPROP_STATUS_SUCCESS" = 0
    CUPAULIPROP_STATUS_NOT_INITIALIZED "CUPAULIPROP_STATUS_NOT_INITIALIZED" = 1
    CUPAULIPROP_STATUS_INVALID_VALUE "CUPAULIPROP_STATUS_INVALID_VALUE" = 2
    CUPAULIPROP_STATUS_INTERNAL_ERROR "CUPAULIPROP_STATUS_INTERNAL_ERROR" = 3
    CUPAULIPROP_STATUS_NOT_SUPPORTED "CUPAULIPROP_STATUS_NOT_SUPPORTED" = 4
    CUPAULIPROP_STATUS_CUDA_ERROR "CUPAULIPROP_STATUS_CUDA_ERROR" = 5
    CUPAULIPROP_STATUS_FATAL_DISTRIBUTED_FAILURE "CUPAULIPROP_STATUS_FATAL_DISTRIBUTED_FAILURE" = 6
    CUPAULIPROP_STATUS_MANDATORY_MEMORY_OVERFLOWED "CUPAULIPROP_STATUS_MANDATORY_MEMORY_OVERFLOWED" = 7
    CUPAULIPROP_STATUS_INSUFFICIENT_WORKSPACE "CUPAULIPROP_STATUS_INSUFFICIENT_WORKSPACE" = 8
    CUPAULIPROP_STATUS_INSUFFICIENT_OUT_EXPANSION "CUPAULIPROP_STATUS_INSUFFICIENT_OUT_EXPANSION" = 9
    CUPAULIPROP_STATUS_INSUFFICIENT_DEVICE_PROPERTY "CUPAULIPROP_STATUS_INSUFFICIENT_DEVICE_PROPERTY" = 10
    _CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR "_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cupaulipropDistributedProvider_t "cupaulipropDistributedProvider_t":
    CUPAULIPROP_DISTRIBUTED_PROVIDER_NONE "CUPAULIPROP_DISTRIBUTED_PROVIDER_NONE" = 0
    CUPAULIPROP_DISTRIBUTED_PROVIDER_MPI "CUPAULIPROP_DISTRIBUTED_PROVIDER_MPI" = 1
    CUPAULIPROP_DISTRIBUTED_PROVIDER_NCCL "CUPAULIPROP_DISTRIBUTED_PROVIDER_NCCL" = 2

ctypedef enum cupaulipropMemspace_t "cupaulipropMemspace_t":
    CUPAULIPROP_MEMSPACE_DEVICE "CUPAULIPROP_MEMSPACE_DEVICE" = 0
    CUPAULIPROP_MEMSPACE_HOST "CUPAULIPROP_MEMSPACE_HOST" = 1

ctypedef enum cupaulipropWorkspaceKind_t "cupaulipropWorkspaceKind_t":
    CUPAULIPROP_WORKSPACE_SCRATCH "CUPAULIPROP_WORKSPACE_SCRATCH" = 0

ctypedef enum cupaulipropTruncationStrategyKind_t "cupaulipropTruncationStrategyKind_t":
    CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED "CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED" = 0
    CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED "CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED" = 1

ctypedef enum cupaulipropSortOrder_t "cupaulipropSortOrder_t":
    CUPAULIPROP_SORT_ORDER_NONE "CUPAULIPROP_SORT_ORDER_NONE" = 0
    CUPAULIPROP_SORT_ORDER_INTERNAL "CUPAULIPROP_SORT_ORDER_INTERNAL" = 1
    CUPAULIPROP_SORT_ORDER_LITTLE_ENDIAN_BITWISE "CUPAULIPROP_SORT_ORDER_LITTLE_ENDIAN_BITWISE" = 2

ctypedef enum cupaulipropPauliKind_t "cupaulipropPauliKind_t":
    CUPAULIPROP_PAULI_I "CUPAULIPROP_PAULI_I" = 0
    CUPAULIPROP_PAULI_X "CUPAULIPROP_PAULI_X" = 1
    CUPAULIPROP_PAULI_Y "CUPAULIPROP_PAULI_Y" = 2
    CUPAULIPROP_PAULI_Z "CUPAULIPROP_PAULI_Z" = 3

ctypedef enum cupaulipropCliffordGateKind_t "cupaulipropCliffordGateKind_t":
    CUPAULIPROP_CLIFFORD_GATE_I "CUPAULIPROP_CLIFFORD_GATE_I" = 0
    CUPAULIPROP_CLIFFORD_GATE_X "CUPAULIPROP_CLIFFORD_GATE_X" = 1
    CUPAULIPROP_CLIFFORD_GATE_Y "CUPAULIPROP_CLIFFORD_GATE_Y" = 2
    CUPAULIPROP_CLIFFORD_GATE_Z "CUPAULIPROP_CLIFFORD_GATE_Z" = 3
    CUPAULIPROP_CLIFFORD_GATE_H "CUPAULIPROP_CLIFFORD_GATE_H" = 4
    CUPAULIPROP_CLIFFORD_GATE_S "CUPAULIPROP_CLIFFORD_GATE_S" = 5
    CUPAULIPROP_CLIFFORD_GATE_CX "CUPAULIPROP_CLIFFORD_GATE_CX" = 7
    CUPAULIPROP_CLIFFORD_GATE_CY "CUPAULIPROP_CLIFFORD_GATE_CY" = 8
    CUPAULIPROP_CLIFFORD_GATE_CZ "CUPAULIPROP_CLIFFORD_GATE_CZ" = 9
    CUPAULIPROP_CLIFFORD_GATE_SWAP "CUPAULIPROP_CLIFFORD_GATE_SWAP" = 10
    CUPAULIPROP_CLIFFORD_GATE_ISWAP "CUPAULIPROP_CLIFFORD_GATE_ISWAP" = 11
    CUPAULIPROP_CLIFFORD_GATE_SQRTX "CUPAULIPROP_CLIFFORD_GATE_SQRTX" = 12
    CUPAULIPROP_CLIFFORD_GATE_SQRTY "CUPAULIPROP_CLIFFORD_GATE_SQRTY" = 13
    CUPAULIPROP_CLIFFORD_GATE_SQRTZ "CUPAULIPROP_CLIFFORD_GATE_SQRTZ" = 14


# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>

    #define CUPAULIPROP_ALLOCATOR_NAME_LEN 64
    """
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef int cudaDataType_t 'cudaDataType_t'
    ctypedef int cudaDataType 'cudaDataType'
    ctypedef int libraryPropertyType_t 'libraryPropertyType_t'
    ctypedef int libraryPropertyType 'libraryPropertyType'

    ctypedef struct cuDoubleComplex:
        double x
        double y
    
    cdef const int CUPAULIPROP_ALLOCATOR_NAME_LEN


ctypedef uint64_t cupaulipropPackedIntegerType_t 'cupaulipropPackedIntegerType_t'

ctypedef void* cupaulipropHandle_t 'cupaulipropHandle_t'

ctypedef void* cupaulipropWorkspaceDescriptor_t 'cupaulipropWorkspaceDescriptor_t'

ctypedef void* cupaulipropPauliExpansion_t 'cupaulipropPauliExpansion_t'

ctypedef void* cupaulipropPauliExpansionView_t 'cupaulipropPauliExpansionView_t'

ctypedef void* cupaulipropQuantumOperator_t 'cupaulipropQuantumOperator_t'

ctypedef void* cupaulipropDistributedRequest_t 'cupaulipropDistributedRequest_t'

ctypedef struct cupaulipropTruncationStrategy_t 'cupaulipropTruncationStrategy_t':
    cupaulipropTruncationStrategyKind_t strategy
    void* paramStruct

ctypedef struct cupaulipropCoefficientTruncationParams_t 'cupaulipropCoefficientTruncationParams_t':
    double cutoff

ctypedef struct cupaulipropPauliWeightTruncationParams_t 'cupaulipropPauliWeightTruncationParams_t':
    int32_t cutoff

ctypedef struct cupaulipropDistributedCommunicator_t 'cupaulipropDistributedCommunicator_t':
    void* commPtr
    size_t commSize

ctypedef struct cupaulipropDistributedInterface_t 'cupaulipropDistributedInterface_t':
    int version
    cupaulipropDistributedProvider_t provider
    int (*getNumRanks)(const cupaulipropDistributedCommunicator_t*, int32_t*)
    int (*getNumRanksShared)(const cupaulipropDistributedCommunicator_t*, int32_t*)
    int (*getProcRank)(const cupaulipropDistributedCommunicator_t*, int32_t*)
    int (*barrier)(const cupaulipropDistributedCommunicator_t*, void*)
    int (*createRequest)(cupaulipropDistributedRequest_t*)
    int (*destroyRequest)(cupaulipropDistributedRequest_t)
    int (*waitRequest)(cupaulipropDistributedRequest_t)
    int (*testRequest)(cupaulipropDistributedRequest_t, int32_t*)
    int (*groupStart)()
    int (*groupEnd)()
    int (*send)(const cupaulipropDistributedCommunicator_t*, const void*, int32_t, cudaDataType_t, int32_t, int32_t)
    int (*sendAsync)(const cupaulipropDistributedCommunicator_t*, const void*, int32_t, cudaDataType_t, int32_t, int32_t, cupaulipropDistributedRequest_t)
    int (*receive)(const cupaulipropDistributedCommunicator_t*, void*, int32_t, cudaDataType_t, int32_t, int32_t)
    int (*receiveAsync)(const cupaulipropDistributedCommunicator_t*, void*, int32_t, cudaDataType_t, int32_t, int32_t, cupaulipropDistributedRequest_t)
    int (*bcast)(const cupaulipropDistributedCommunicator_t*, void*, int32_t, cudaDataType_t, int32_t)
    int (*allreduce)(const cupaulipropDistributedCommunicator_t*, const void*, void*, int32_t, cudaDataType_t)
    int (*allreduceInPlace)(const cupaulipropDistributedCommunicator_t*, void*, int32_t, cudaDataType_t)
    int (*allreduceInPlaceMin)(const cupaulipropDistributedCommunicator_t*, void*, int32_t, cudaDataType_t)
    int (*allreduceDoubleIntMinloc)(const cupaulipropDistributedCommunicator_t*, const void*, void*)
    int (*allgather)(const cupaulipropDistributedCommunicator_t*, const void*, void*, int32_t, cudaDataType_t)

ctypedef struct cupaulipropPauliTerm_t 'cupaulipropPauliTerm_t':
    cupaulipropPackedIntegerType_t* xzbits
    void* coef

###############################################################################
# Functions
###############################################################################

cdef size_t cupaulipropGetVersion() except?0 nogil
cdef const char* cupaulipropGetErrorString(cupaulipropStatus_t error) except?NULL nogil
cdef cupaulipropStatus_t cupaulipropGetNumPackedIntegers(int32_t numQubits, int32_t* numPackedIntegers) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreate(cupaulipropHandle_t* handle) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropDestroy(cupaulipropHandle_t handle) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropResetDistributedConfiguration(cupaulipropHandle_t handle, cupaulipropDistributedProvider_t provider, const void* commPtr, size_t commSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropGetNumRanks(const cupaulipropHandle_t handle, int32_t* numRanks) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropGetProcRank(const cupaulipropHandle_t handle, int32_t* procRank) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreateWorkspaceDescriptor(cupaulipropHandle_t handle, cupaulipropWorkspaceDescriptor_t* workspaceDesc) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropDestroyWorkspaceDescriptor(cupaulipropWorkspaceDescriptor_t workspaceDesc) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropWorkspaceGetMemorySize(const cupaulipropHandle_t handle, const cupaulipropWorkspaceDescriptor_t workspaceDesc, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, int64_t* memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropWorkspaceSetMemory(const cupaulipropHandle_t handle, cupaulipropWorkspaceDescriptor_t workspaceDesc, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, void* memoryBuffer, int64_t memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropWorkspaceGetMemory(const cupaulipropHandle_t handle, const cupaulipropWorkspaceDescriptor_t workspaceDescr, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, void** memoryBuffer, int64_t* memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreatePauliExpansion(const cupaulipropHandle_t handle, int32_t numQubits, void* xzBitsBuffer, int64_t xzBitsBufferSize, void* coefBuffer, int64_t coefBufferSize, cudaDataType_t dataType, int64_t numLocalTerms, cupaulipropSortOrder_t sortOrder, int32_t hasDuplicates, cupaulipropPauliExpansion_t* pauliExpansion) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropDestroyPauliExpansion(cupaulipropPauliExpansion_t pauliExpansion) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetStorageBuffer(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, void** xzBitsBuffer, int64_t* xzBitsBufferSize, void** coefBuffer, int64_t* coefBufferSize, int64_t* numLocalTerms, cupaulipropMemspace_t* location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetNumQubits(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* numQubits) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetNumTerms(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t* numLocalTerms) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetDataType(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, cudaDataType_t* dataType) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetSortOrder(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, cupaulipropSortOrder_t* sortOrder) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionIsDeduplicated(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* isDeduplicated) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetTerm(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t termIndex, cupaulipropPauliTerm_t* term) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetContiguousRange(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t startIndex, int64_t endIndex, cupaulipropPauliExpansionView_t* view) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropDestroyPauliExpansionView(cupaulipropPauliExpansionView_t view) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewGetNumTerms(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t* numLocalTerms) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewGetLocation(const cupaulipropPauliExpansionView_t view, cupaulipropMemspace_t* location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewGetTerm(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t termIndex, cupaulipropPauliTerm_t* term) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareDeduplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropSortOrder_t sortOrder, int64_t maxWorkspaceDeviceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewExecuteDeduplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, cupaulipropSortOrder_t sortOrder, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareSort(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropSortOrder_t sortOrder, int64_t maxWorkspaceDeviceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewExecuteSort(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, cupaulipropSortOrder_t sortOrder, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionPopulateFromView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareTraceWithExpansionView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int64_t maxWorkspaceDeviceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeTraceWithExpansionView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int32_t takeAdjoint1, void* traceSignificand, double* traceExponent, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int64_t maxWorkspaceDeviceSize, int64_t* requiredXZBitsBufferSize1, int64_t* requiredCoefBufferSize1, int64_t* requiredXZBitsBufferSize2, int64_t* requiredCoefBufferSize2, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int32_t takeAdjoint1, const void* cotangentTraceSignificand, const double* cotangentTraceExponent, cupaulipropPauliExpansion_t cotangentExpansion1, cupaulipropPauliExpansion_t cotangentExpansion2, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareTraceWithZeroState(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t maxWorkspaceDeviceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeTraceWithZeroState(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, void* traceSignificand, double* traceExponent, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t maxWorkspaceDeviceSize, int64_t* requiredXZBitsBufferSize, int64_t* requiredCoefBufferSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, const void* cotangentTraceSignificand, const double* cotangentTraceExponent, cupaulipropPauliExpansion_t cotangentExpansion, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareOperatorApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, const cupaulipropQuantumOperator_t quantumOperator, cupaulipropSortOrder_t sortOrder, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceDeviceSize, int64_t* requiredXZBitsBufferSize, int64_t* requiredCoefBufferSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeOperatorApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, const cupaulipropQuantumOperator_t quantumOperator, int32_t adjoint, cupaulipropSortOrder_t sortOrder, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareOperatorFusedApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int32_t numQuantumOperators, const cupaulipropQuantumOperator_t quantumOperators[], const int32_t adjoints[], int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceDeviceSize, int64_t* minExpansionOutCapacity, cupaulipropWorkspaceDescriptor_t minWorkspace, int64_t* averageExpansionOutCapacity, cupaulipropWorkspaceDescriptor_t averageWorkspace, int64_t* maxExpansionOutCapacity, cupaulipropWorkspaceDescriptor_t maxWorkspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeOperatorFusedApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, int32_t numQuantumOperators, const cupaulipropQuantumOperator_t quantumOperators[], const int32_t adjoints[], int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, const cupaulipropPauliExpansionView_t cotangentOut, const cupaulipropQuantumOperator_t quantumOperator, cupaulipropSortOrder_t sortOrder, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceDeviceSize, int64_t* requiredXZBitsBufferSize, int64_t* requiredCoefBufferSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, const cupaulipropPauliExpansionView_t cotangentOut, cupaulipropPauliExpansion_t cotangentIn, cupaulipropQuantumOperator_t quantumOperator, int32_t adjoint, cupaulipropSortOrder_t sortOrder, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareTruncation(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceDeviceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewExecuteTruncation(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreateCliffordGateOperator(const cupaulipropHandle_t handle, cupaulipropCliffordGateKind_t cliffordGateKind, const int32_t qubitIndices[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreatePauliRotationGateOperator(const cupaulipropHandle_t handle, double angle, int32_t numQubits, const int32_t qubitIndices[], const cupaulipropPauliKind_t paulis[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreatePauliNoiseChannelOperator(const cupaulipropHandle_t handle, int32_t numQubits, const int32_t qubitIndices[], const double probabilities[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreateAmplitudeDampingChannelOperator(const cupaulipropHandle_t handle, int32_t qubitIndex, double dampingProb, double exciteProb, cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropQuantumOperatorAttachCotangentBuffer(const cupaulipropHandle_t handle, cupaulipropQuantumOperator_t oper, void* cotangentBuffer, int64_t cotangentBufferSize, cudaDataType_t dataType, cupaulipropMemspace_t location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropQuantumOperatorGetCotangentBuffer(const cupaulipropHandle_t handle, const cupaulipropQuantumOperator_t oper, void** cotangentBuffer, int64_t* cotangentBufferNumElements, cudaDataType_t* dataType, cupaulipropMemspace_t* location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropDestroyOperator(cupaulipropQuantumOperator_t oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
