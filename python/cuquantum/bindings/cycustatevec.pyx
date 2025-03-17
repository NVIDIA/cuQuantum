# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 25.03.0. Do not modify it directly.

from ._internal cimport custatevec as _custatevec


###############################################################################
# Wrapper functions
###############################################################################

cdef custatevecStatus_t custatevecCreate(custatevecHandle_t* handle) except* nogil:
    return _custatevec._custatevecCreate(handle)


cdef custatevecStatus_t custatevecDestroy(custatevecHandle_t handle) except* nogil:
    return _custatevec._custatevecDestroy(handle)


cdef custatevecStatus_t custatevecGetDefaultWorkspaceSize(custatevecHandle_t handle, size_t* workspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecGetDefaultWorkspaceSize(handle, workspaceSizeInBytes)


cdef custatevecStatus_t custatevecSetWorkspace(custatevecHandle_t handle, void* workspace, size_t workspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecSetWorkspace(handle, workspace, workspaceSizeInBytes)


cdef const char* custatevecGetErrorName(custatevecStatus_t status) except* nogil:
    return _custatevec._custatevecGetErrorName(status)


cdef const char* custatevecGetErrorString(custatevecStatus_t status) except* nogil:
    return _custatevec._custatevecGetErrorString(status)


cdef custatevecStatus_t custatevecGetProperty(libraryPropertyType type, int32_t* value) except* nogil:
    return _custatevec._custatevecGetProperty(type, value)


cdef size_t custatevecGetVersion() except* nogil:
    return _custatevec._custatevecGetVersion()


cdef custatevecStatus_t custatevecSetStream(custatevecHandle_t handle, cudaStream_t streamId) except* nogil:
    return _custatevec._custatevecSetStream(handle, streamId)


cdef custatevecStatus_t custatevecGetStream(custatevecHandle_t handle, cudaStream_t* streamId) except* nogil:
    return _custatevec._custatevecGetStream(handle, streamId)


cdef custatevecStatus_t custatevecLoggerSetCallbackData(custatevecLoggerCallbackData_t callback, void* userData) except* nogil:
    return _custatevec._custatevecLoggerSetCallbackData(callback, userData)


cdef custatevecStatus_t custatevecLoggerOpenFile(const char* logFile) except* nogil:
    return _custatevec._custatevecLoggerOpenFile(logFile)


cdef custatevecStatus_t custatevecLoggerSetLevel(int32_t level) except* nogil:
    return _custatevec._custatevecLoggerSetLevel(level)


cdef custatevecStatus_t custatevecLoggerSetMask(int32_t mask) except* nogil:
    return _custatevec._custatevecLoggerSetMask(mask)


cdef custatevecStatus_t custatevecLoggerForceDisable() except* nogil:
    return _custatevec._custatevecLoggerForceDisable()


cdef custatevecStatus_t custatevecGetDeviceMemHandler(custatevecHandle_t handle, custatevecDeviceMemHandler_t* handler) except* nogil:
    return _custatevec._custatevecGetDeviceMemHandler(handle, handler)


cdef custatevecStatus_t custatevecSetDeviceMemHandler(custatevecHandle_t handle, const custatevecDeviceMemHandler_t* handler) except* nogil:
    return _custatevec._custatevecSetDeviceMemHandler(handle, handler)


cdef custatevecStatus_t custatevecAbs2SumOnZBasis(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* abs2sum0, double* abs2sum1, const int32_t* basisBits, const uint32_t nBasisBits) except* nogil:
    return _custatevec._custatevecAbs2SumOnZBasis(handle, sv, svDataType, nIndexBits, abs2sum0, abs2sum1, basisBits, nBasisBits)


cdef custatevecStatus_t custatevecAbs2SumArray(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* abs2sum, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen) except* nogil:
    return _custatevec._custatevecAbs2SumArray(handle, sv, svDataType, nIndexBits, abs2sum, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen)


cdef custatevecStatus_t custatevecCollapseOnZBasis(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int32_t parity, const int32_t* basisBits, const uint32_t nBasisBits, double norm) except* nogil:
    return _custatevec._custatevecCollapseOnZBasis(handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, norm)


cdef custatevecStatus_t custatevecCollapseByBitString(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, double norm) except* nogil:
    return _custatevec._custatevecCollapseByBitString(handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, norm)


cdef custatevecStatus_t custatevecMeasureOnZBasis(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* parity, const int32_t* basisBits, const uint32_t nBasisBits, const double randnum, custatevecCollapseOp_t collapse) except* nogil:
    return _custatevec._custatevecMeasureOnZBasis(handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, randnum, collapse)


cdef custatevecStatus_t custatevecBatchMeasure(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, const double randnum, custatevecCollapseOp_t collapse) except* nogil:
    return _custatevec._custatevecBatchMeasure(handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse)


cdef custatevecStatus_t custatevecBatchMeasureWithOffset(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, const double randnum, custatevecCollapseOp_t collapse, const double offset, const double abs2sum) except* nogil:
    return _custatevec._custatevecBatchMeasureWithOffset(handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse, offset, abs2sum)


cdef custatevecStatus_t custatevecApplyPauliRotation(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double theta, const custatevecPauli_t* paulis, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls) except* nogil:
    return _custatevec._custatevecApplyPauliRotation(handle, sv, svDataType, nIndexBits, theta, paulis, targets, nTargets, controls, controlBitValues, nControls)


cdef custatevecStatus_t custatevecApplyMatrixGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nTargets, const uint32_t nControls, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecApplyMatrixGetWorkspaceSize(handle, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecApplyMatrix(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecApplyMatrix(handle, sv, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, targets, nTargets, controls, controlBitValues, nControls, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecComputeExpectationGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nBasisBits, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecComputeExpectationGetWorkspaceSize(handle, svDataType, nIndexBits, matrix, matrixDataType, layout, nBasisBits, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecComputeExpectation(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, void* expectationValue, cudaDataType_t expectationDataType, double* residualNorm, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t* basisBits, const uint32_t nBasisBits, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecComputeExpectation(handle, sv, svDataType, nIndexBits, expectationValue, expectationDataType, residualNorm, matrix, matrixDataType, layout, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecSamplerCreate(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecSamplerDescriptor_t* sampler, uint32_t nMaxShots, size_t* extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecSamplerCreate(handle, sv, svDataType, nIndexBits, sampler, nMaxShots, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecSamplerDestroy(custatevecSamplerDescriptor_t sampler) except* nogil:
    return _custatevec._custatevecSamplerDestroy(sampler)


cdef custatevecStatus_t custatevecSamplerPreprocess(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, void* extraWorkspace, const size_t extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecSamplerPreprocess(handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecSamplerGetSquaredNorm(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, double* norm) except* nogil:
    return _custatevec._custatevecSamplerGetSquaredNorm(handle, sampler, norm)


cdef custatevecStatus_t custatevecSamplerApplySubSVOffset(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, int32_t subSVOrd, uint32_t nSubSVs, double offset, double norm) except* nogil:
    return _custatevec._custatevecSamplerApplySubSVOffset(handle, sampler, subSVOrd, nSubSVs, offset, norm)


cdef custatevecStatus_t custatevecSamplerSample(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* randnums, const uint32_t nShots, custatevecSamplerOutput_t output) except* nogil:
    return _custatevec._custatevecSamplerSample(handle, sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots, output)


cdef custatevecStatus_t custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType, const int32_t* targets, const uint32_t nTargets, const uint32_t nControls, size_t* extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(handle, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, targets, nTargets, nControls, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecApplyGeneralizedPermutationMatrix(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType, const int32_t adjoint, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecApplyGeneralizedPermutationMatrix(handle, sv, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, adjoint, targets, nTargets, controls, controlBitValues, nControls, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecComputeExpectationsOnPauliBasis(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* expectationValues, const custatevecPauli_t** pauliOperatorsArray, const uint32_t nPauliOperatorArrays, const int32_t** basisBitsArray, const uint32_t* nBasisBitsArray) except* nogil:
    return _custatevec._custatevecComputeExpectationsOnPauliBasis(handle, sv, svDataType, nIndexBits, expectationValues, pauliOperatorsArray, nPauliOperatorArrays, basisBitsArray, nBasisBitsArray)


cdef custatevecStatus_t custatevecAccessorCreate(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecAccessorDescriptor_t* accessor, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, size_t* extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecAccessorCreate(handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecAccessorCreateView(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecAccessorDescriptor_t* accessor, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, size_t* extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecAccessorCreateView(handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecAccessorDestroy(custatevecAccessorDescriptor_t accessor) except* nogil:
    return _custatevec._custatevecAccessorDestroy(accessor)


cdef custatevecStatus_t custatevecAccessorSetExtraWorkspace(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecAccessorSetExtraWorkspace(handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecAccessorGet(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, void* externalBuffer, const custatevecIndex_t begin, const custatevecIndex_t end) except* nogil:
    return _custatevec._custatevecAccessorGet(handle, accessor, externalBuffer, begin, end)


cdef custatevecStatus_t custatevecAccessorSet(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, const void* externalBuffer, const custatevecIndex_t begin, const custatevecIndex_t end) except* nogil:
    return _custatevec._custatevecAccessorSet(handle, accessor, externalBuffer, begin, end)


cdef custatevecStatus_t custatevecSwapIndexBits(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int2* bitSwaps, const uint32_t nBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen) except* nogil:
    return _custatevec._custatevecSwapIndexBits(handle, sv, svDataType, nIndexBits, bitSwaps, nBitSwaps, maskBitString, maskOrdering, maskLen)


cdef custatevecStatus_t custatevecTestMatrixTypeGetWorkspaceSize(custatevecHandle_t handle, custatevecMatrixType_t matrixType, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nTargets, const int32_t adjoint, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecTestMatrixTypeGetWorkspaceSize(handle, matrixType, matrix, matrixDataType, layout, nTargets, adjoint, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecTestMatrixType(custatevecHandle_t handle, double* residualNorm, custatevecMatrixType_t matrixType, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nTargets, const int32_t adjoint, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecTestMatrixType(handle, residualNorm, matrixType, matrix, matrixDataType, layout, nTargets, adjoint, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecMultiDeviceSwapIndexBits(custatevecHandle_t* handles, const uint32_t nHandles, void** subSVs, const cudaDataType_t svDataType, const uint32_t nGlobalIndexBits, const uint32_t nLocalIndexBits, const int2* indexBitSwaps, const uint32_t nIndexBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, const custatevecDeviceNetworkType_t deviceNetworkType) except* nogil:
    return _custatevec._custatevecMultiDeviceSwapIndexBits(handles, nHandles, subSVs, svDataType, nGlobalIndexBits, nLocalIndexBits, indexBitSwaps, nIndexBitSwaps, maskBitString, maskOrdering, maskLen, deviceNetworkType)


cdef custatevecStatus_t custatevecCommunicatorCreate(custatevecHandle_t handle, custatevecCommunicatorDescriptor_t* communicator, custatevecCommunicatorType_t communicatorType, const char* soname) except* nogil:
    return _custatevec._custatevecCommunicatorCreate(handle, communicator, communicatorType, soname)


cdef custatevecStatus_t custatevecCommunicatorDestroy(custatevecHandle_t handle, custatevecCommunicatorDescriptor_t communicator) except* nogil:
    return _custatevec._custatevecCommunicatorDestroy(handle, communicator)


cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerCreate(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t* scheduler, const uint32_t nGlobalIndexBits, const uint32_t nLocalIndexBits) except* nogil:
    return _custatevec._custatevecDistIndexBitSwapSchedulerCreate(handle, scheduler, nGlobalIndexBits, nLocalIndexBits)


cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerDestroy(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler) except* nogil:
    return _custatevec._custatevecDistIndexBitSwapSchedulerDestroy(handle, scheduler)


cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler, const int2* indexBitSwaps, const uint32_t nIndexBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, uint32_t* nSwapBatches) except* nogil:
    return _custatevec._custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(handle, scheduler, indexBitSwaps, nIndexBitSwaps, maskBitString, maskOrdering, maskLen, nSwapBatches)


cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerGetParameters(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler, const int32_t swapBatchIndex, const int32_t orgSubSVIndex, custatevecSVSwapParameters_t* parameters) except* nogil:
    return _custatevec._custatevecDistIndexBitSwapSchedulerGetParameters(handle, scheduler, swapBatchIndex, orgSubSVIndex, parameters)


cdef custatevecStatus_t custatevecSVSwapWorkerCreate(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t* svSwapWorker, custatevecCommunicatorDescriptor_t communicator, void* orgSubSV, int32_t orgSubSVIndex, cudaEvent_t orgEvent, cudaDataType_t svDataType, cudaStream_t stream, size_t* extraWorkspaceSizeInBytes, size_t* minTransferWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecSVSwapWorkerCreate(handle, svSwapWorker, communicator, orgSubSV, orgSubSVIndex, orgEvent, svDataType, stream, extraWorkspaceSizeInBytes, minTransferWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecSVSwapWorkerDestroy(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker) except* nogil:
    return _custatevec._custatevecSVSwapWorkerDestroy(handle, svSwapWorker)


cdef custatevecStatus_t custatevecSVSwapWorkerSetExtraWorkspace(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecSVSwapWorkerSetExtraWorkspace(handle, svSwapWorker, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecSVSwapWorkerSetTransferWorkspace(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void* transferWorkspace, size_t transferWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecSVSwapWorkerSetTransferWorkspace(handle, svSwapWorker, transferWorkspace, transferWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecSVSwapWorkerSetSubSVsP2P(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void** dstSubSVsP2P, const int32_t* dstSubSVIndicesP2P, cudaEvent_t* dstEvents, const uint32_t nDstSubSVsP2P) except* nogil:
    return _custatevec._custatevecSVSwapWorkerSetSubSVsP2P(handle, svSwapWorker, dstSubSVsP2P, dstSubSVIndicesP2P, dstEvents, nDstSubSVsP2P)


cdef custatevecStatus_t custatevecSVSwapWorkerSetParameters(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, const custatevecSVSwapParameters_t* parameters, int peer) except* nogil:
    return _custatevec._custatevecSVSwapWorkerSetParameters(handle, svSwapWorker, parameters, peer)


cdef custatevecStatus_t custatevecSVSwapWorkerExecute(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, custatevecIndex_t begin, custatevecIndex_t end) except* nogil:
    return _custatevec._custatevecSVSwapWorkerExecute(handle, svSwapWorker, begin, end)


cdef custatevecStatus_t custatevecInitializeStateVector(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecStateVectorType_t svType) except* nogil:
    return _custatevec._custatevecInitializeStateVector(handle, sv, svDataType, nIndexBits, svType)


cdef custatevecStatus_t custatevecApplyMatrixBatchedGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, custatevecMatrixMapType_t mapType, const int32_t* matrixIndices, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nMatrices, const uint32_t nTargets, const uint32_t nControls, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecApplyMatrixBatchedGetWorkspaceSize(handle, svDataType, nIndexBits, nSVs, svStride, mapType, matrixIndices, matrices, matrixDataType, layout, adjoint, nMatrices, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecApplyMatrixBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, custatevecIndex_t svStride, custatevecMatrixMapType_t mapType, const int32_t* matrixIndices, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nMatrices, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecApplyMatrixBatched(handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, mapType, matrixIndices, matrices, matrixDataType, layout, adjoint, nMatrices, targets, nTargets, controls, controlBitValues, nControls, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecAbs2SumArrayBatched(custatevecHandle_t handle, const void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, double* abs2sumArrays, const custatevecIndex_t abs2sumArrayStride, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const custatevecIndex_t* maskBitStrings, const int32_t* maskOrdering, const uint32_t maskLen) except* nogil:
    return _custatevec._custatevecAbs2SumArrayBatched(handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, abs2sumArrays, abs2sumArrayStride, bitOrdering, bitOrderingLen, maskBitStrings, maskOrdering, maskLen)


cdef custatevecStatus_t custatevecCollapseByBitStringBatchedGetWorkspaceSize(custatevecHandle_t handle, const uint32_t nSVs, const custatevecIndex_t* bitStrings, const double* norms, size_t* extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecCollapseByBitStringBatchedGetWorkspaceSize(handle, nSVs, bitStrings, norms, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecCollapseByBitStringBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, const custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* norms, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecCollapseByBitStringBatched(handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, bitStrings, bitOrdering, bitStringLen, norms, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecMeasureBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* randnums, custatevecCollapseOp_t collapse) except* nogil:
    return _custatevec._custatevecMeasureBatched(handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, bitStrings, bitOrdering, bitStringLen, randnums, collapse)


cdef custatevecStatus_t custatevecSubSVMigratorCreate(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t* migrator, void* deviceSlots, cudaDataType_t svDataType, int nDeviceSlots, int nLocalIndexBits) except* nogil:
    return _custatevec._custatevecSubSVMigratorCreate(handle, migrator, deviceSlots, svDataType, nDeviceSlots, nLocalIndexBits)


cdef custatevecStatus_t custatevecSubSVMigratorDestroy(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t migrator) except* nogil:
    return _custatevec._custatevecSubSVMigratorDestroy(handle, migrator)


cdef custatevecStatus_t custatevecSubSVMigratorMigrate(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t migrator, int deviceSlotIndex, const void* srcSubSV, void* dstSubSV, custatevecIndex_t begin, custatevecIndex_t end) except* nogil:
    return _custatevec._custatevecSubSVMigratorMigrate(handle, migrator, deviceSlotIndex, srcSubSV, dstSubSV, begin, end)


cdef custatevecStatus_t custatevecComputeExpectationBatchedGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nMatrices, const uint32_t nBasisBits, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecComputeExpectationBatchedGetWorkspaceSize(handle, svDataType, nIndexBits, nSVs, svStride, matrices, matrixDataType, layout, nMatrices, nBasisBits, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecComputeExpectationBatched(custatevecHandle_t handle, const void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, custatevecIndex_t svStride, double2* expectationValues, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nMatrices, const int32_t* basisBits, const uint32_t nBasisBits, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    return _custatevec._custatevecComputeExpectationBatched(handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, expectationValues, matrices, matrixDataType, layout, nMatrices, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t custatevecSetMathMode(custatevecHandle_t handle, custatevecMathMode_t mode) except* nogil:
    return _custatevec._custatevecSetMathMode(handle, mode)


cdef custatevecStatus_t custatevecGetMathMode(custatevecHandle_t handle, custatevecMathMode_t* mode) except* nogil:
    return _custatevec._custatevecGetMathMode(handle, mode)
