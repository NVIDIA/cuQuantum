# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.11.0. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
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
    CUPAULIPROP_STATUS_DISTRIBUTED_FAILURE "CUPAULIPROP_STATUS_DISTRIBUTED_FAILURE" = 6
    _CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR "_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cupaulipropComputeType_t "cupaulipropComputeType_t":
    CUPAULIPROP_COMPUTE_32F "CUPAULIPROP_COMPUTE_32F" = (1U << 2U)
    CUPAULIPROP_COMPUTE_64F "CUPAULIPROP_COMPUTE_64F" = (1U << 4U)

ctypedef enum cupaulipropMemspace_t "cupaulipropMemspace_t":
    CUPAULIPROP_MEMSPACE_DEVICE "CUPAULIPROP_MEMSPACE_DEVICE" = 0
    CUPAULIPROP_MEMSPACE_HOST "CUPAULIPROP_MEMSPACE_HOST" = 1

ctypedef enum cupaulipropWorkspaceKind_t "cupaulipropWorkspaceKind_t":
    CUPAULIPROP_WORKSPACE_SCRATCH "CUPAULIPROP_WORKSPACE_SCRATCH" = 0

ctypedef enum cupaulipropTruncationStrategyKind_t "cupaulipropTruncationStrategyKind_t":
    CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED "CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED" = 0
    CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED "CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED" = 1

ctypedef enum cupaulipropPauliKind_t "cupaulipropPauliKind_t":
    CUPAULIPROP_PAULI_I "CUPAULIPROP_PAULI_I" = 0
    CUPAULIPROP_PAULI_X "CUPAULIPROP_PAULI_X" = 1
    CUPAULIPROP_PAULI_Z "CUPAULIPROP_PAULI_Z" = 2
    CUPAULIPROP_PAULI_Y "CUPAULIPROP_PAULI_Y" = 3

ctypedef enum cupaulipropCliffordGateKind_t "cupaulipropCliffordGateKind_t":
    CUPAULIPROP_CLIFFORD_GATE_I "CUPAULIPROP_CLIFFORD_GATE_I" = 0
    CUPAULIPROP_CLIFFORD_GATE_X "CUPAULIPROP_CLIFFORD_GATE_X" = 1
    CUPAULIPROP_CLIFFORD_GATE_Z "CUPAULIPROP_CLIFFORD_GATE_Z" = 2
    CUPAULIPROP_CLIFFORD_GATE_Y "CUPAULIPROP_CLIFFORD_GATE_Y" = 3
    CUPAULIPROP_CLIFFORD_GATE_H "CUPAULIPROP_CLIFFORD_GATE_H" = 4
    CUPAULIPROP_CLIFFORD_GATE_S "CUPAULIPROP_CLIFFORD_GATE_S" = 5
    CUPAULIPROP_CLIFFORD_GATE_CX "CUPAULIPROP_CLIFFORD_GATE_CX" = 7
    CUPAULIPROP_CLIFFORD_GATE_CZ "CUPAULIPROP_CLIFFORD_GATE_CZ" = 8
    CUPAULIPROP_CLIFFORD_GATE_CY "CUPAULIPROP_CLIFFORD_GATE_CY" = 9
    CUPAULIPROP_CLIFFORD_GATE_SWAP "CUPAULIPROP_CLIFFORD_GATE_SWAP" = 10
    CUPAULIPROP_CLIFFORD_GATE_ISWAP "CUPAULIPROP_CLIFFORD_GATE_ISWAP" = 11
    CUPAULIPROP_CLIFFORD_GATE_SQRTX "CUPAULIPROP_CLIFFORD_GATE_SQRTX" = 12
    CUPAULIPROP_CLIFFORD_GATE_SQRTZ "CUPAULIPROP_CLIFFORD_GATE_SQRTZ" = 13
    CUPAULIPROP_CLIFFORD_GATE_SQRTY "CUPAULIPROP_CLIFFORD_GATE_SQRTY" = 14

ctypedef enum cupaulipropQuantumOperatorKind_t "cupaulipropQuantumOperatorKind_t":
    CUPAULIPROP_EXPANSION_KIND_PAULI_ROTATION_GATE "CUPAULIPROP_EXPANSION_KIND_PAULI_ROTATION_GATE" = 0
    CUPAULIPROP_EXPANSION_KIND_CLIFFORD_GATE "CUPAULIPROP_EXPANSION_KIND_CLIFFORD_GATE" = 1
    CUPAULIPROP_EXPANSION_KIND_PAULI_NOISE_CHANNEL "CUPAULIPROP_EXPANSION_KIND_PAULI_NOISE_CHANNEL" = 2


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
ctypedef struct cupaulipropTruncationStrategy_t 'cupaulipropTruncationStrategy_t':
    cupaulipropTruncationStrategyKind_t strategy
    void* paramStruct
ctypedef struct cupaulipropCoefficientTruncationParams_t 'cupaulipropCoefficientTruncationParams_t':
    double cutoff
ctypedef struct cupaulipropPauliWeightTruncationParams_t 'cupaulipropPauliWeightTruncationParams_t':
    int32_t cutoff
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
cdef cupaulipropStatus_t cupaulipropSetStream(cupaulipropHandle_t handle, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreateWorkspaceDescriptor(cupaulipropHandle_t handle, cupaulipropWorkspaceDescriptor_t* workspaceDesc) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropDestroyWorkspaceDescriptor(cupaulipropWorkspaceDescriptor_t workspaceDesc) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropWorkspaceGetMemorySize(const cupaulipropHandle_t handle, const cupaulipropWorkspaceDescriptor_t workspaceDesc, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, int64_t* memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropWorkspaceSetMemory(const cupaulipropHandle_t handle, cupaulipropWorkspaceDescriptor_t workspaceDesc, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, void* memoryBuffer, int64_t memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropWorkspaceGetMemory(const cupaulipropHandle_t handle, const cupaulipropWorkspaceDescriptor_t workspaceDescr, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, void** memoryBuffer, int64_t* memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreatePauliExpansion(const cupaulipropHandle_t handle, int32_t numQubits, void* xzBitsBuffer, int64_t xzBitsBufferSize, void* coefBuffer, int64_t coefBufferSize, cudaDataType_t dataType, int64_t numTerms, int32_t isSorted, int32_t hasDuplicates, cupaulipropPauliExpansion_t* pauliExpansion) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropDestroyPauliExpansion(cupaulipropPauliExpansion_t pauliExpansion) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetStorageBuffer(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, void** xzBitsBuffer, int64_t* xzBitsBufferSize, void** coefBuffer, int64_t* coefBufferSize, int64_t* numTerms, cupaulipropMemspace_t* location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetNumQubits(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* numQubits) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetNumTerms(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t* numTerms) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetDataType(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, cudaDataType_t* dataType) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionIsSorted(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* isSorted) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionIsDeduplicated(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* isDeduplicated) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetTerm(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t termIndex, cupaulipropPauliTerm_t* term) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionGetContiguousRange(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t startIndex, int64_t endIndex, cupaulipropPauliExpansionView_t* view) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropDestroyPauliExpansionView(cupaulipropPauliExpansionView_t view) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewGetNumTerms(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t* numTerms) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewGetLocation(const cupaulipropPauliExpansionView_t view, cupaulipropMemspace_t* location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewGetTerm(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t termIndex, cupaulipropPauliTerm_t* term) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareDeduplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int32_t makeSorted, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewExecuteDeduplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, int32_t makeSorted, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareCanonicalSort(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewExecuteCanonicalSort(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionPopulateFromView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareTraceWithExpansionView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeTraceWithExpansionView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int32_t takeAdjoint1, void* trace, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareTraceWithZeroState(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeTraceWithZeroState(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, void* trace, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareOperatorApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, const cupaulipropQuantumOperator_t quantumOperator, int32_t makeSorted, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceSize, int64_t* requiredXZBitsBufferSize, int64_t* requiredCoefBufferSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeOperatorApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, const cupaulipropQuantumOperator_t quantumOperator, int32_t adjoint, int32_t makeSorted, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareTruncation(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropPauliExpansionViewExecuteTruncation(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreateCliffordGateOperator(const cupaulipropHandle_t handle, cupaulipropCliffordGateKind_t cliffordGateKind, const int32_t qubitIndices[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreatePauliRotationGateOperator(const cupaulipropHandle_t handle, double angle, int32_t numQubits, const int32_t qubitIndices[], const cupaulipropPauliKind_t paulis[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropCreatePauliNoiseChannelOperator(const cupaulipropHandle_t handle, int32_t numQubits, const int32_t qubitIndices[], const double probabilities[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropQuantumOperatorGetKind(const cupaulipropHandle_t handle, const cupaulipropQuantumOperator_t oper, cupaulipropQuantumOperatorKind_t* kind) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cupaulipropStatus_t cupaulipropDestroyOperator(cupaulipropQuantumOperator_t oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil
