# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.09.0. Do not modify it directly.

from ._internal cimport cudensitymat as _cudensitymat


###############################################################################
# Wrapper functions
###############################################################################

cdef size_t cudensitymatGetVersion() except?0 nogil:
    return _cudensitymat._cudensitymatGetVersion()


cdef cudensitymatStatus_t cudensitymatCreate(cudensitymatHandle_t* handle) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreate(handle)


cdef cudensitymatStatus_t cudensitymatDestroy(cudensitymatHandle_t handle) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatDestroy(handle)


cdef cudensitymatStatus_t cudensitymatResetDistributedConfiguration(cudensitymatHandle_t handle, cudensitymatDistributedProvider_t provider, const void* commPtr, size_t commSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatResetDistributedConfiguration(handle, provider, commPtr, commSize)


cdef cudensitymatStatus_t cudensitymatGetNumRanks(const cudensitymatHandle_t handle, int32_t* numRanks) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatGetNumRanks(handle, numRanks)


cdef cudensitymatStatus_t cudensitymatGetProcRank(const cudensitymatHandle_t handle, int32_t* procRank) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatGetProcRank(handle, procRank)


cdef cudensitymatStatus_t cudensitymatResetRandomSeed(cudensitymatHandle_t handle, int32_t randomSeed) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatResetRandomSeed(handle, randomSeed)


cdef cudensitymatStatus_t cudensitymatCreateState(const cudensitymatHandle_t handle, cudensitymatStatePurity_t purity, int32_t numSpaceModes, const int64_t spaceModeExtents[], int64_t batchSize, cudaDataType_t dataType, cudensitymatState_t* state) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreateState(handle, purity, numSpaceModes, spaceModeExtents, batchSize, dataType, state)


cdef cudensitymatStatus_t cudensitymatDestroyState(cudensitymatState_t state) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatDestroyState(state)


cdef cudensitymatStatus_t cudensitymatStateGetNumComponents(const cudensitymatHandle_t handle, const cudensitymatState_t state, int32_t* numStateComponents) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatStateGetNumComponents(handle, state, numStateComponents)


cdef cudensitymatStatus_t cudensitymatStateGetComponentStorageSize(const cudensitymatHandle_t handle, const cudensitymatState_t state, int32_t numStateComponents, size_t componentBufferSize[]) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatStateGetComponentStorageSize(handle, state, numStateComponents, componentBufferSize)


cdef cudensitymatStatus_t cudensitymatStateAttachComponentStorage(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t numStateComponents, void* componentBuffer[], const size_t componentBufferSize[]) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatStateAttachComponentStorage(handle, state, numStateComponents, componentBuffer, componentBufferSize)


cdef cudensitymatStatus_t cudensitymatStateGetComponentNumModes(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t stateComponentLocalId, int32_t* stateComponentGlobalId, int32_t* stateComponentNumModes, int32_t* batchModeLocation) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatStateGetComponentNumModes(handle, state, stateComponentLocalId, stateComponentGlobalId, stateComponentNumModes, batchModeLocation)


cdef cudensitymatStatus_t cudensitymatStateGetComponentInfo(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t stateComponentLocalId, int32_t* stateComponentGlobalId, int32_t* stateComponentNumModes, int64_t stateComponentModeExtents[], int64_t stateComponentModeOffsets[]) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatStateGetComponentInfo(handle, state, stateComponentLocalId, stateComponentGlobalId, stateComponentNumModes, stateComponentModeExtents, stateComponentModeOffsets)


cdef cudensitymatStatus_t cudensitymatStateInitializeZero(const cudensitymatHandle_t handle, cudensitymatState_t state, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatStateInitializeZero(handle, state, stream)


cdef cudensitymatStatus_t cudensitymatStateComputeScaling(const cudensitymatHandle_t handle, cudensitymatState_t state, const void* scalingFactors, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatStateComputeScaling(handle, state, scalingFactors, stream)


cdef cudensitymatStatus_t cudensitymatStateComputeNorm(const cudensitymatHandle_t handle, const cudensitymatState_t state, void* norm, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatStateComputeNorm(handle, state, norm, stream)


cdef cudensitymatStatus_t cudensitymatStateComputeTrace(const cudensitymatHandle_t handle, const cudensitymatState_t state, void* trace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatStateComputeTrace(handle, state, trace, stream)


cdef cudensitymatStatus_t cudensitymatStateComputeAccumulation(const cudensitymatHandle_t handle, const cudensitymatState_t stateIn, cudensitymatState_t stateOut, const void* scalingFactors, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatStateComputeAccumulation(handle, stateIn, stateOut, scalingFactors, stream)


cdef cudensitymatStatus_t cudensitymatStateComputeInnerProduct(const cudensitymatHandle_t handle, const cudensitymatState_t stateLeft, const cudensitymatState_t stateRight, void* innerProduct, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatStateComputeInnerProduct(handle, stateLeft, stateRight, innerProduct, stream)


cdef cudensitymatStatus_t cudensitymatCreateElementaryOperator(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatElementaryOperatorSparsity_t sparsity, int32_t numDiagonals, const int32_t diagonalOffsets[], cudaDataType_t dataType, void* tensorData, cudensitymatWrappedTensorCallback_t tensorCallback, cudensitymatWrappedTensorGradientCallback_t tensorGradientCallback, cudensitymatElementaryOperator_t* elemOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreateElementaryOperator(handle, numSpaceModes, spaceModeExtents, sparsity, numDiagonals, diagonalOffsets, dataType, tensorData, tensorCallback, tensorGradientCallback, elemOperator)


cdef cudensitymatStatus_t cudensitymatCreateElementaryOperatorBatch(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], int64_t batchSize, cudensitymatElementaryOperatorSparsity_t sparsity, int32_t numDiagonals, const int32_t diagonalOffsets[], cudaDataType_t dataType, void* tensorData, cudensitymatWrappedTensorCallback_t tensorCallback, cudensitymatWrappedTensorGradientCallback_t tensorGradientCallback, cudensitymatElementaryOperator_t* elemOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreateElementaryOperatorBatch(handle, numSpaceModes, spaceModeExtents, batchSize, sparsity, numDiagonals, diagonalOffsets, dataType, tensorData, tensorCallback, tensorGradientCallback, elemOperator)


cdef cudensitymatStatus_t cudensitymatDestroyElementaryOperator(cudensitymatElementaryOperator_t elemOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatDestroyElementaryOperator(elemOperator)


cdef cudensitymatStatus_t cudensitymatCreateMatrixOperatorDenseLocal(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudaDataType_t dataType, void* matrixData, cudensitymatWrappedTensorCallback_t matrixCallback, cudensitymatWrappedTensorGradientCallback_t matrixGradientCallback, cudensitymatMatrixOperator_t* matrixOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreateMatrixOperatorDenseLocal(handle, numSpaceModes, spaceModeExtents, dataType, matrixData, matrixCallback, matrixGradientCallback, matrixOperator)


cdef cudensitymatStatus_t cudensitymatCreateMatrixOperatorDenseLocalBatch(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], int64_t batchSize, cudaDataType_t dataType, void* matrixData, cudensitymatWrappedTensorCallback_t matrixCallback, cudensitymatWrappedTensorGradientCallback_t matrixGradientCallback, cudensitymatMatrixOperator_t* matrixOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreateMatrixOperatorDenseLocalBatch(handle, numSpaceModes, spaceModeExtents, batchSize, dataType, matrixData, matrixCallback, matrixGradientCallback, matrixOperator)


cdef cudensitymatStatus_t cudensitymatDestroyMatrixOperator(cudensitymatMatrixOperator_t matrixOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatDestroyMatrixOperator(matrixOperator)


cdef cudensitymatStatus_t cudensitymatCreateOperatorTerm(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatOperatorTerm_t* operatorTerm) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreateOperatorTerm(handle, numSpaceModes, spaceModeExtents, operatorTerm)


cdef cudensitymatStatus_t cudensitymatDestroyOperatorTerm(cudensitymatOperatorTerm_t operatorTerm) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatDestroyOperatorTerm(operatorTerm)


cdef cudensitymatStatus_t cudensitymatOperatorTermAppendElementaryProduct(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numElemOperators, const cudensitymatElementaryOperator_t elemOperators[], const int32_t stateModesActedOn[], const int32_t modeActionDuality[], cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorTermAppendElementaryProduct(handle, operatorTerm, numElemOperators, elemOperators, stateModesActedOn, modeActionDuality, coefficient, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t cudensitymatOperatorTermAppendElementaryProductBatch(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numElemOperators, const cudensitymatElementaryOperator_t elemOperators[], const int32_t stateModesActedOn[], const int32_t modeActionDuality[], int64_t batchSize, const cuDoubleComplex staticCoefficients[], cuDoubleComplex totalCoefficients[], cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorTermAppendElementaryProductBatch(handle, operatorTerm, numElemOperators, elemOperators, stateModesActedOn, modeActionDuality, batchSize, staticCoefficients, totalCoefficients, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t cudensitymatOperatorTermAppendMatrixProduct(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numMatrixOperators, const cudensitymatMatrixOperator_t matrixOperators[], const int32_t matrixConjugation[], const int32_t actionDuality[], cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorTermAppendMatrixProduct(handle, operatorTerm, numMatrixOperators, matrixOperators, matrixConjugation, actionDuality, coefficient, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t cudensitymatOperatorTermAppendMatrixProductBatch(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numMatrixOperators, const cudensitymatMatrixOperator_t matrixOperators[], const int32_t matrixConjugation[], const int32_t actionDuality[], int64_t batchSize, const cuDoubleComplex staticCoefficients[], cuDoubleComplex totalCoefficients[], cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorTermAppendMatrixProductBatch(handle, operatorTerm, numMatrixOperators, matrixOperators, matrixConjugation, actionDuality, batchSize, staticCoefficients, totalCoefficients, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t cudensitymatCreateOperator(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatOperator_t* superoperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreateOperator(handle, numSpaceModes, spaceModeExtents, superoperator)


cdef cudensitymatStatus_t cudensitymatDestroyOperator(cudensitymatOperator_t superoperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatDestroyOperator(superoperator)


cdef cudensitymatStatus_t cudensitymatOperatorAppendTerm(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, cudensitymatOperatorTerm_t operatorTerm, int32_t duality, cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorAppendTerm(handle, superoperator, operatorTerm, duality, coefficient, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t cudensitymatOperatorAppendTermBatch(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, cudensitymatOperatorTerm_t operatorTerm, int32_t duality, int64_t batchSize, const cuDoubleComplex staticCoefficients[], cuDoubleComplex totalCoefficients[], cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorAppendTermBatch(handle, superoperator, operatorTerm, duality, batchSize, staticCoefficients, totalCoefficients, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t cudensitymatOperatorPrepareAction(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, const cudensitymatState_t stateIn, const cudensitymatState_t stateOut, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorPrepareAction(handle, superoperator, stateIn, stateOut, computeType, workspaceSizeLimit, workspace, stream)


cdef cudensitymatStatus_t cudensitymatOperatorComputeAction(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t stateIn, cudensitymatState_t stateOut, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorComputeAction(handle, superoperator, time, batchSize, numParams, params, stateIn, stateOut, workspace, stream)


cdef cudensitymatStatus_t cudensitymatOperatorPrepareActionBackwardDiff(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, const cudensitymatState_t stateIn, const cudensitymatState_t stateOutAdj, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorPrepareActionBackwardDiff(handle, superoperator, stateIn, stateOutAdj, computeType, workspaceSizeLimit, workspace, stream)


cdef cudensitymatStatus_t cudensitymatOperatorComputeActionBackwardDiff(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t stateIn, const cudensitymatState_t stateOutAdj, cudensitymatState_t stateInAdj, double* paramsGrad, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorComputeActionBackwardDiff(handle, superoperator, time, batchSize, numParams, params, stateIn, stateOutAdj, stateInAdj, paramsGrad, workspace, stream)


cdef cudensitymatStatus_t cudensitymatCreateOperatorAction(const cudensitymatHandle_t handle, int32_t numOperators, cudensitymatOperator_t operators[], cudensitymatOperatorAction_t* operatorAction) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreateOperatorAction(handle, numOperators, operators, operatorAction)


cdef cudensitymatStatus_t cudensitymatDestroyOperatorAction(cudensitymatOperatorAction_t operatorAction) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatDestroyOperatorAction(operatorAction)


cdef cudensitymatStatus_t cudensitymatOperatorActionPrepare(const cudensitymatHandle_t handle, cudensitymatOperatorAction_t operatorAction, const cudensitymatState_t stateIn[], const cudensitymatState_t stateOut, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorActionPrepare(handle, operatorAction, stateIn, stateOut, computeType, workspaceSizeLimit, workspace, stream)


cdef cudensitymatStatus_t cudensitymatOperatorActionCompute(const cudensitymatHandle_t handle, cudensitymatOperatorAction_t operatorAction, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t stateIn[], cudensitymatState_t stateOut, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorActionCompute(handle, operatorAction, time, batchSize, numParams, params, stateIn, stateOut, workspace, stream)


cdef cudensitymatStatus_t cudensitymatCreateExpectation(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, cudensitymatExpectation_t* expectation) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreateExpectation(handle, superoperator, expectation)


cdef cudensitymatStatus_t cudensitymatDestroyExpectation(cudensitymatExpectation_t expectation) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatDestroyExpectation(expectation)


cdef cudensitymatStatus_t cudensitymatExpectationPrepare(const cudensitymatHandle_t handle, cudensitymatExpectation_t expectation, const cudensitymatState_t state, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatExpectationPrepare(handle, expectation, state, computeType, workspaceSizeLimit, workspace, stream)


cdef cudensitymatStatus_t cudensitymatExpectationCompute(const cudensitymatHandle_t handle, cudensitymatExpectation_t expectation, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t state, void* expectationValue, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatExpectationCompute(handle, expectation, time, batchSize, numParams, params, state, expectationValue, workspace, stream)


cdef cudensitymatStatus_t cudensitymatCreateOperatorSpectrum(const cudensitymatHandle_t handle, const cudensitymatOperator_t superoperator, int32_t isHermitian, cudensitymatOperatorSpectrumKind_t spectrumKind, cudensitymatOperatorSpectrum_t* spectrum) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreateOperatorSpectrum(handle, superoperator, isHermitian, spectrumKind, spectrum)


cdef cudensitymatStatus_t cudensitymatDestroyOperatorSpectrum(cudensitymatOperatorSpectrum_t spectrum) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatDestroyOperatorSpectrum(spectrum)


cdef cudensitymatStatus_t cudensitymatOperatorSpectrumConfigure(const cudensitymatHandle_t handle, cudensitymatOperatorSpectrum_t spectrum, cudensitymatOperatorSpectrumConfig_t attribute, const void* attributeValue, size_t attributeValueSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorSpectrumConfigure(handle, spectrum, attribute, attributeValue, attributeValueSize)


cdef cudensitymatStatus_t cudensitymatOperatorSpectrumPrepare(const cudensitymatHandle_t handle, cudensitymatOperatorSpectrum_t spectrum, int32_t maxEigenStates, const cudensitymatState_t state, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorSpectrumPrepare(handle, spectrum, maxEigenStates, state, computeType, workspaceSizeLimit, workspace, stream)


cdef cudensitymatStatus_t cudensitymatOperatorSpectrumCompute(const cudensitymatHandle_t handle, cudensitymatOperatorSpectrum_t spectrum, double time, int64_t batchSize, int32_t numParams, const double* params, int32_t numEigenStates, cudensitymatState_t eigenstates[], void* eigenvalues, double* tolerances, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatOperatorSpectrumCompute(handle, spectrum, time, batchSize, numParams, params, numEigenStates, eigenstates, eigenvalues, tolerances, workspace, stream)


cdef cudensitymatStatus_t cudensitymatCreateWorkspace(const cudensitymatHandle_t handle, cudensitymatWorkspaceDescriptor_t* workspaceDescr) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatCreateWorkspace(handle, workspaceDescr)


cdef cudensitymatStatus_t cudensitymatDestroyWorkspace(cudensitymatWorkspaceDescriptor_t workspaceDescr) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatDestroyWorkspace(workspaceDescr)


cdef cudensitymatStatus_t cudensitymatWorkspaceGetMemorySize(const cudensitymatHandle_t handle, const cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, size_t* memoryBufferSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatWorkspaceGetMemorySize(handle, workspaceDescr, memSpace, workspaceKind, memoryBufferSize)


cdef cudensitymatStatus_t cudensitymatWorkspaceSetMemory(const cudensitymatHandle_t handle, cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, void* memoryBuffer, size_t memoryBufferSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatWorkspaceSetMemory(handle, workspaceDescr, memSpace, workspaceKind, memoryBuffer, memoryBufferSize)


cdef cudensitymatStatus_t cudensitymatWorkspaceGetMemory(const cudensitymatHandle_t handle, const cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, void** memoryBuffer, size_t* memoryBufferSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudensitymat._cudensitymatWorkspaceGetMemory(handle, workspaceDescr, memSpace, workspaceKind, memoryBuffer, memoryBufferSize)
