# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from ..cycudensitymat cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef cudensitymatStatus_t _cudensitymatCreate(cudensitymatHandle_t* handle) except* nogil
cdef cudensitymatStatus_t _cudensitymatDestroy(cudensitymatHandle_t handle) except* nogil
cdef cudensitymatStatus_t _cudensitymatResetDistributedConfiguration(cudensitymatHandle_t handle, cudensitymatDistributedProvider_t provider, const void* commPtr, size_t commSize) except* nogil
cdef cudensitymatStatus_t _cudensitymatGetNumRanks(const cudensitymatHandle_t handle, int32_t* numRanks) except* nogil
cdef cudensitymatStatus_t _cudensitymatGetProcRank(const cudensitymatHandle_t handle, int32_t* procRank) except* nogil
cdef cudensitymatStatus_t _cudensitymatResetRandomSeed(cudensitymatHandle_t handle, int32_t randomSeed) except* nogil
cdef cudensitymatStatus_t _cudensitymatCreateState(const cudensitymatHandle_t handle, cudensitymatStatePurity_t purity, int32_t numSpaceModes, const int64_t spaceModeExtents[], int64_t batchSize, cudaDataType_t dataType, cudensitymatState_t* state) except* nogil
cdef cudensitymatStatus_t _cudensitymatDestroyState(cudensitymatState_t state) except* nogil
cdef cudensitymatStatus_t _cudensitymatStateGetNumComponents(const cudensitymatHandle_t handle, const cudensitymatState_t state, int32_t* numStateComponents) except* nogil
cdef cudensitymatStatus_t _cudensitymatStateGetComponentStorageSize(const cudensitymatHandle_t handle, const cudensitymatState_t state, int32_t numStateComponents, size_t componentBufferSize[]) except* nogil
cdef cudensitymatStatus_t _cudensitymatStateAttachComponentStorage(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t numStateComponents, void* componentBuffer[], const size_t componentBufferSize[]) except* nogil
cdef cudensitymatStatus_t _cudensitymatStateGetComponentNumModes(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t stateComponentLocalId, int32_t* stateComponentGlobalId, int32_t* stateComponentNumModes, int32_t* batchModeLocation) except* nogil
cdef cudensitymatStatus_t _cudensitymatStateGetComponentInfo(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t stateComponentLocalId, int32_t* stateComponentGlobalId, int32_t* stateComponentNumModes, int64_t stateComponentModeExtents[], int64_t stateComponentModeOffsets[]) except* nogil
cdef cudensitymatStatus_t _cudensitymatStateInitializeZero(const cudensitymatHandle_t handle, cudensitymatState_t state, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatStateComputeScaling(const cudensitymatHandle_t handle, cudensitymatState_t state, const void* scalingFactors, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatStateComputeNorm(const cudensitymatHandle_t handle, const cudensitymatState_t state, void* norm, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatStateComputeTrace(const cudensitymatHandle_t handle, const cudensitymatState_t state, void* trace, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatStateComputeAccumulation(const cudensitymatHandle_t handle, const cudensitymatState_t stateIn, cudensitymatState_t stateOut, const void* scalingFactors, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatStateComputeInnerProduct(const cudensitymatHandle_t handle, const cudensitymatState_t stateLeft, const cudensitymatState_t stateRight, void* innerProduct, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatCreateElementaryOperator(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatElementaryOperatorSparsity_t sparsity, int32_t numDiagonals, const int32_t diagonalOffsets[], cudaDataType_t dataType, void* tensorData, cudensitymatWrappedTensorCallback_t tensorCallback, cudensitymatElementaryOperator_t* elemOperator) except* nogil
cdef cudensitymatStatus_t _cudensitymatDestroyElementaryOperator(cudensitymatElementaryOperator_t elemOperator) except* nogil
cdef cudensitymatStatus_t _cudensitymatCreateOperatorTerm(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatOperatorTerm_t* operatorTerm) except* nogil
cdef cudensitymatStatus_t _cudensitymatDestroyOperatorTerm(cudensitymatOperatorTerm_t operatorTerm) except* nogil
cdef cudensitymatStatus_t _cudensitymatOperatorTermAppendElementaryProduct(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numElemOperators, const cudensitymatElementaryOperator_t elemOperators[], const int32_t stateModesActedOn[], const int32_t modeActionDuality[], cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback) except* nogil
cdef cudensitymatStatus_t _cudensitymatOperatorTermAppendGeneralProduct(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numElemOperators, const int32_t numOperatorModes[], const int64_t* operatorModeExtents[], const int64_t* operatorModeStrides[], const int32_t stateModesActedOn[], const int32_t modeActionDuality[], cudaDataType_t dataType, void* tensorData[], cudensitymatWrappedTensorCallback_t tensorCallbacks[], cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback) except* nogil
cdef cudensitymatStatus_t _cudensitymatCreateOperator(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatOperator_t* superoperator) except* nogil
cdef cudensitymatStatus_t _cudensitymatDestroyOperator(cudensitymatOperator_t superoperator) except* nogil
cdef cudensitymatStatus_t _cudensitymatOperatorAppendTerm(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, cudensitymatOperatorTerm_t operatorTerm, int32_t duality, cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback) except* nogil
cdef cudensitymatStatus_t _cudensitymatOperatorPrepareAction(const cudensitymatHandle_t handle, const cudensitymatOperator_t superoperator, const cudensitymatState_t stateIn, const cudensitymatState_t stateOut, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatOperatorComputeAction(const cudensitymatHandle_t handle, const cudensitymatOperator_t superoperator, double time, int32_t numParams, const double params[], const cudensitymatState_t stateIn, cudensitymatState_t stateOut, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatCreateOperatorAction(const cudensitymatHandle_t handle, int32_t numOperators, cudensitymatOperator_t operators[], cudensitymatOperatorAction_t* operatorAction) except* nogil
cdef cudensitymatStatus_t _cudensitymatDestroyOperatorAction(cudensitymatOperatorAction_t operatorAction) except* nogil
cdef cudensitymatStatus_t _cudensitymatOperatorActionPrepare(const cudensitymatHandle_t handle, cudensitymatOperatorAction_t operatorAction, const cudensitymatState_t stateIn[], const cudensitymatState_t stateOut, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatOperatorActionCompute(const cudensitymatHandle_t handle, cudensitymatOperatorAction_t operatorAction, double time, int32_t numParams, const double params[], const cudensitymatState_t stateIn[], cudensitymatState_t stateOut, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatCreateExpectation(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, cudensitymatExpectation_t* expectation) except* nogil
cdef cudensitymatStatus_t _cudensitymatDestroyExpectation(cudensitymatExpectation_t expectation) except* nogil
cdef cudensitymatStatus_t _cudensitymatExpectationPrepare(const cudensitymatHandle_t handle, cudensitymatExpectation_t expectation, const cudensitymatState_t state, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatExpectationCompute(const cudensitymatHandle_t handle, cudensitymatExpectation_t expectation, double time, int32_t numParams, const double params[], const cudensitymatState_t state, void* expectationValue, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except* nogil
cdef cudensitymatStatus_t _cudensitymatCreateWorkspace(const cudensitymatHandle_t handle, cudensitymatWorkspaceDescriptor_t* workspaceDescr) except* nogil
cdef cudensitymatStatus_t _cudensitymatDestroyWorkspace(cudensitymatWorkspaceDescriptor_t workspaceDescr) except* nogil
cdef cudensitymatStatus_t _cudensitymatWorkspaceGetMemorySize(const cudensitymatHandle_t handle, const cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, size_t* memoryBufferSize) except* nogil
cdef cudensitymatStatus_t _cudensitymatWorkspaceSetMemory(const cudensitymatHandle_t handle, cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, void* memoryBuffer, size_t memoryBufferSize) except* nogil
cdef cudensitymatStatus_t _cudensitymatWorkspaceGetMemory(const cudensitymatHandle_t handle, const cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, void** memoryBuffer, size_t* memoryBufferSize) except* nogil
