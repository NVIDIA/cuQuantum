# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.11.0. Do not modify it directly.

from ._internal cimport cupauliprop as _cupauliprop


###############################################################################
# Wrapper functions
###############################################################################

cdef size_t cupaulipropGetVersion() except?0 nogil:
    return _cupauliprop._cupaulipropGetVersion()


cdef const char* cupaulipropGetErrorString(cupaulipropStatus_t error) except?NULL nogil:
    return _cupauliprop._cupaulipropGetErrorString(error)


cdef cupaulipropStatus_t cupaulipropGetNumPackedIntegers(int32_t numQubits, int32_t* numPackedIntegers) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropGetNumPackedIntegers(numQubits, numPackedIntegers)


cdef cupaulipropStatus_t cupaulipropCreate(cupaulipropHandle_t* handle) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropCreate(handle)


cdef cupaulipropStatus_t cupaulipropDestroy(cupaulipropHandle_t handle) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropDestroy(handle)


cdef cupaulipropStatus_t cupaulipropSetStream(cupaulipropHandle_t handle, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropSetStream(handle, stream)


cdef cupaulipropStatus_t cupaulipropCreateWorkspaceDescriptor(cupaulipropHandle_t handle, cupaulipropWorkspaceDescriptor_t* workspaceDesc) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropCreateWorkspaceDescriptor(handle, workspaceDesc)


cdef cupaulipropStatus_t cupaulipropDestroyWorkspaceDescriptor(cupaulipropWorkspaceDescriptor_t workspaceDesc) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropDestroyWorkspaceDescriptor(workspaceDesc)


cdef cupaulipropStatus_t cupaulipropWorkspaceGetMemorySize(const cupaulipropHandle_t handle, const cupaulipropWorkspaceDescriptor_t workspaceDesc, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, int64_t* memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropWorkspaceGetMemorySize(handle, workspaceDesc, memSpace, workspaceKind, memoryBufferSize)


cdef cupaulipropStatus_t cupaulipropWorkspaceSetMemory(const cupaulipropHandle_t handle, cupaulipropWorkspaceDescriptor_t workspaceDesc, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, void* memoryBuffer, int64_t memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropWorkspaceSetMemory(handle, workspaceDesc, memSpace, workspaceKind, memoryBuffer, memoryBufferSize)


cdef cupaulipropStatus_t cupaulipropWorkspaceGetMemory(const cupaulipropHandle_t handle, const cupaulipropWorkspaceDescriptor_t workspaceDescr, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, void** memoryBuffer, int64_t* memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropWorkspaceGetMemory(handle, workspaceDescr, memSpace, workspaceKind, memoryBuffer, memoryBufferSize)


cdef cupaulipropStatus_t cupaulipropCreatePauliExpansion(const cupaulipropHandle_t handle, int32_t numQubits, void* xzBitsBuffer, int64_t xzBitsBufferSize, void* coefBuffer, int64_t coefBufferSize, cudaDataType_t dataType, int64_t numTerms, int32_t isSorted, int32_t hasDuplicates, cupaulipropPauliExpansion_t* pauliExpansion) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropCreatePauliExpansion(handle, numQubits, xzBitsBuffer, xzBitsBufferSize, coefBuffer, coefBufferSize, dataType, numTerms, isSorted, hasDuplicates, pauliExpansion)


cdef cupaulipropStatus_t cupaulipropDestroyPauliExpansion(cupaulipropPauliExpansion_t pauliExpansion) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropDestroyPauliExpansion(pauliExpansion)


cdef cupaulipropStatus_t cupaulipropPauliExpansionGetStorageBuffer(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, void** xzBitsBuffer, int64_t* xzBitsBufferSize, void** coefBuffer, int64_t* coefBufferSize, int64_t* numTerms, cupaulipropMemspace_t* location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionGetStorageBuffer(handle, pauliExpansion, xzBitsBuffer, xzBitsBufferSize, coefBuffer, coefBufferSize, numTerms, location)


cdef cupaulipropStatus_t cupaulipropPauliExpansionGetNumQubits(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* numQubits) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionGetNumQubits(handle, pauliExpansion, numQubits)


cdef cupaulipropStatus_t cupaulipropPauliExpansionGetNumTerms(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t* numTerms) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionGetNumTerms(handle, pauliExpansion, numTerms)


cdef cupaulipropStatus_t cupaulipropPauliExpansionGetDataType(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, cudaDataType_t* dataType) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionGetDataType(handle, pauliExpansion, dataType)


cdef cupaulipropStatus_t cupaulipropPauliExpansionIsSorted(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* isSorted) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionIsSorted(handle, pauliExpansion, isSorted)


cdef cupaulipropStatus_t cupaulipropPauliExpansionIsDeduplicated(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* isDeduplicated) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionIsDeduplicated(handle, pauliExpansion, isDeduplicated)


cdef cupaulipropStatus_t cupaulipropPauliExpansionGetTerm(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t termIndex, cupaulipropPauliTerm_t* term) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionGetTerm(handle, pauliExpansion, termIndex, term)


cdef cupaulipropStatus_t cupaulipropPauliExpansionGetContiguousRange(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t startIndex, int64_t endIndex, cupaulipropPauliExpansionView_t* view) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionGetContiguousRange(handle, pauliExpansion, startIndex, endIndex, view)


cdef cupaulipropStatus_t cupaulipropDestroyPauliExpansionView(cupaulipropPauliExpansionView_t view) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropDestroyPauliExpansionView(view)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewGetNumTerms(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t* numTerms) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewGetNumTerms(handle, view, numTerms)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewGetLocation(const cupaulipropPauliExpansionView_t view, cupaulipropMemspace_t* location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewGetLocation(view, location)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewGetTerm(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t termIndex, cupaulipropPauliTerm_t* term) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewGetTerm(handle, view, termIndex, term)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareDeduplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int32_t makeSorted, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewPrepareDeduplication(handle, viewIn, makeSorted, maxWorkspaceSize, workspace)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewExecuteDeduplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, int32_t makeSorted, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewExecuteDeduplication(handle, viewIn, expansionOut, makeSorted, workspace)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareCanonicalSort(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewPrepareCanonicalSort(handle, viewIn, maxWorkspaceSize, workspace)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewExecuteCanonicalSort(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewExecuteCanonicalSort(handle, viewIn, expansionOut, workspace)


cdef cupaulipropStatus_t cupaulipropPauliExpansionPopulateFromView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionPopulateFromView(handle, viewIn, expansionOut)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareTraceWithExpansionView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewPrepareTraceWithExpansionView(handle, view1, view2, maxWorkspaceSize, workspace)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeTraceWithExpansionView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int32_t takeAdjoint1, void* trace, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewComputeTraceWithExpansionView(handle, view1, view2, takeAdjoint1, trace, workspace)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareTraceWithZeroState(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, view, maxWorkspaceSize, workspace)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeTraceWithZeroState(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, void* trace, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, view, trace, workspace)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareOperatorApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, const cupaulipropQuantumOperator_t quantumOperator, int32_t makeSorted, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceSize, int64_t* requiredXZBitsBufferSize, int64_t* requiredCoefBufferSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewPrepareOperatorApplication(handle, viewIn, quantumOperator, makeSorted, keepDuplicates, numTruncationStrategies, truncationStrategies, maxWorkspaceSize, requiredXZBitsBufferSize, requiredCoefBufferSize, workspace)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewComputeOperatorApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, const cupaulipropQuantumOperator_t quantumOperator, int32_t adjoint, int32_t makeSorted, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewComputeOperatorApplication(handle, viewIn, expansionOut, quantumOperator, adjoint, makeSorted, keepDuplicates, numTruncationStrategies, truncationStrategies, workspace)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewPrepareTruncation(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewPrepareTruncation(handle, viewIn, numTruncationStrategies, truncationStrategies, maxWorkspaceSize, workspace)


cdef cupaulipropStatus_t cupaulipropPauliExpansionViewExecuteTruncation(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropPauliExpansionViewExecuteTruncation(handle, viewIn, expansionOut, numTruncationStrategies, truncationStrategies, workspace)


cdef cupaulipropStatus_t cupaulipropCreateCliffordGateOperator(const cupaulipropHandle_t handle, cupaulipropCliffordGateKind_t cliffordGateKind, const int32_t qubitIndices[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropCreateCliffordGateOperator(handle, cliffordGateKind, qubitIndices, oper)


cdef cupaulipropStatus_t cupaulipropCreatePauliRotationGateOperator(const cupaulipropHandle_t handle, double angle, int32_t numQubits, const int32_t qubitIndices[], const cupaulipropPauliKind_t paulis[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropCreatePauliRotationGateOperator(handle, angle, numQubits, qubitIndices, paulis, oper)


cdef cupaulipropStatus_t cupaulipropCreatePauliNoiseChannelOperator(const cupaulipropHandle_t handle, int32_t numQubits, const int32_t qubitIndices[], const double probabilities[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropCreatePauliNoiseChannelOperator(handle, numQubits, qubitIndices, probabilities, oper)


cdef cupaulipropStatus_t cupaulipropQuantumOperatorGetKind(const cupaulipropHandle_t handle, const cupaulipropQuantumOperator_t oper, cupaulipropQuantumOperatorKind_t* kind) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropQuantumOperatorGetKind(handle, oper, kind)


cdef cupaulipropStatus_t cupaulipropDestroyOperator(cupaulipropQuantumOperator_t oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cupauliprop._cupaulipropDestroyOperator(oper)
