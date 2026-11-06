# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 26.09.0 to 26.11.0. Do not modify it directly.


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    int32_t,
    uint32_t,
)


# <<<< END OF PREAMBLE CONTENT >>>>

from ._internal cimport custatevecEx as _custatevecEx


###############################################################################
# Wrapper functions
###############################################################################

cdef custatevecStatus_t custatevecExDictionaryDestroy(custatevecExDictionaryDescriptor_t dictionary) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExDictionaryDestroy(dictionary)


cdef custatevecStatus_t custatevecExCommunicatorInitialize(custatevecCommunicatorType_t communicatorType, const char* libraryPath, int* argc, char*** argv, custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExCommunicatorInitialize(communicatorType, libraryPath, argc, argv, exCommStatus)


cdef custatevecStatus_t custatevecExCommunicatorFinalize(custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExCommunicatorFinalize(exCommStatus)


cdef custatevecStatus_t custatevecExCommunicatorGetSizeAndRank(int32_t* size, int32_t* rank, custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExCommunicatorGetSizeAndRank(size, rank, exCommStatus)


cdef custatevecStatus_t custatevecExCommunicatorCreate(custatevecExCommunicatorDescriptor_t* exCommunicator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExCommunicatorCreate(exCommunicator)


cdef custatevecStatus_t custatevecExCommunicatorDestroy(custatevecExCommunicatorDescriptor_t exCommunicator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExCommunicatorDestroy(exCommunicator)


cdef custatevecStatus_t custatevecExConfigureStateVectorSingleDevice(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, int32_t deviceId, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExConfigureStateVectorSingleDevice(svConfig, svDataType, numWires, numDeviceWires, deviceId, capability)


cdef custatevecStatus_t custatevecExConfigureStateVectorMultiDevice(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, const int32_t* deviceIds, int32_t numDevices, custatevecDeviceNetworkType_t networkType, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExConfigureStateVectorMultiDevice(svConfig, svDataType, numWires, numDeviceWires, deviceIds, numDevices, networkType, capability)


cdef custatevecStatus_t custatevecExConfigureStateVectorMultiProcess(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, int32_t deviceId, custatevecExMemorySharingMethod_t memorySharingMethod, const custatevecExGlobalIndexBitClass_t* globalIndexBitClasses, const int32_t* numGlobalIndexBitsPerLayer, int32_t numGlobalIndexBitLayers, size_t transferWorkspaceSizeInBytes, const void* auxConfig, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExConfigureStateVectorMultiProcess(svConfig, svDataType, numWires, numDeviceWires, deviceId, memorySharingMethod, globalIndexBitClasses, numGlobalIndexBitsPerLayer, numGlobalIndexBitLayers, transferWorkspaceSizeInBytes, auxConfig, capability)


cdef custatevecStatus_t custatevecExStateVectorCreateSingleProcess(custatevecExStateVectorDescriptor_t* stateVector, const custatevecExDictionaryDescriptor_t svConfig, const cudaStream_t* streams, int32_t numStreams, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorCreateSingleProcess(stateVector, svConfig, streams, numStreams, resourceManager)


cdef custatevecStatus_t custatevecExStateVectorCreateMultiProcess(custatevecExStateVectorDescriptor_t* stateVector, const custatevecExDictionaryDescriptor_t svConfig, cudaStream_t stream, custatevecExCommunicatorDescriptor_t exCommunicator, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorCreateMultiProcess(stateVector, svConfig, stream, exCommunicator, resourceManager)


cdef custatevecStatus_t custatevecExStateVectorDestroy(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorDestroy(stateVector)


cdef custatevecStatus_t custatevecExStateVectorGetProperty(const custatevecExStateVectorDescriptor_t stateVector, custatevecExStateVectorProperty_t property, void* value, size_t sizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorGetProperty(stateVector, property, value, sizeInBytes)


cdef custatevecStatus_t custatevecExStateVectorSetMathMode(custatevecExStateVectorDescriptor_t stateVector, custatevecMathMode_t mode) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorSetMathMode(stateVector, mode)


cdef custatevecStatus_t custatevecExStateVectorSetZeroState(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorSetZeroState(stateVector)


cdef custatevecStatus_t custatevecExStateVectorGetState(const custatevecExStateVectorDescriptor_t stateVector, void* state, cudaDataType_t dataType, custatevecIndex_t begin, custatevecIndex_t end, int32_t maxNumConcurrentCopies) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorGetState(stateVector, state, dataType, begin, end, maxNumConcurrentCopies)


cdef custatevecStatus_t custatevecExStateVectorSetState(custatevecExStateVectorDescriptor_t stateVector, const void* state, cudaDataType_t dataType, custatevecIndex_t begin, custatevecIndex_t end, int32_t maxNumConcurrentCopies) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorSetState(stateVector, state, dataType, begin, end, maxNumConcurrentCopies)


cdef custatevecStatus_t custatevecExStateVectorReassignWireOrdering(custatevecExStateVectorDescriptor_t stateVector, const int32_t* wireOrdering, int32_t wireOrderingLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorReassignWireOrdering(stateVector, wireOrdering, wireOrderingLen)


cdef custatevecStatus_t custatevecExStateVectorPermuteIndexBits(custatevecExStateVectorDescriptor_t stateVector, const int32_t* permutation, int32_t permutationLen, custatevecExPermutationType_t permutationType) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorPermuteIndexBits(stateVector, permutation, permutationLen, permutationType)


cdef custatevecStatus_t custatevecExStateVectorStageSubSV(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorStageSubSV(stateVector, subSVIndex)


cdef custatevecStatus_t custatevecExStateVectorExposeResources(custatevecExStateVectorDescriptor_t stateVector, custatevecExExposeResources_t exposeResources) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorExposeResources(stateVector, exposeResources)


cdef custatevecStatus_t custatevecExStateVectorGetResourcesFromDeviceSubSV(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t* deviceId, void** d_subSV, cudaStream_t* stream, custatevecHandle_t* handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorGetResourcesFromDeviceSubSV(stateVector, subSVIndex, deviceId, d_subSV, stream, handle)


cdef custatevecStatus_t custatevecExStateVectorGetResourcesFromDeviceSubSVView(const custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t* deviceId, const void** d_subSV, cudaStream_t* stream, custatevecHandle_t* handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorGetResourcesFromDeviceSubSVView(stateVector, subSVIndex, deviceId, d_subSV, stream, handle)


cdef custatevecStatus_t custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t sliceIndex, void** subSVSlice, custatevecExMemoryPlacement_t* placement, int32_t* deviceId, cudaStream_t* stream) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice(stateVector, subSVIndex, sliceIndex, subSVSlice, placement, deviceId, stream)


cdef custatevecStatus_t custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView(const custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t sliceIndex, const void** subSVSlice, custatevecExMemoryPlacement_t* placement, int32_t* deviceId, cudaStream_t* stream) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView(stateVector, subSVIndex, sliceIndex, subSVSlice, placement, deviceId, stream)


cdef custatevecStatus_t custatevecExStateVectorSynchronize(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorSynchronize(stateVector)


cdef custatevecStatus_t custatevecExStateVectorSynchronizeScoped(custatevecExStateVectorDescriptor_t stateVector, custatevecExSynchronizationScope_t scope) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorSynchronizeScoped(stateVector, scope)


cdef custatevecStatus_t custatevecExStateVectorAddWires(custatevecExStateVectorDescriptor_t stateVector, custatevecExIndexBitDomain_t indexBitDomain, int32_t numWiresToAdd, custatevecExWireInitMode_t wireInitMode, int32_t* wiresAdded) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExStateVectorAddWires(stateVector, indexBitDomain, numWiresToAdd, wireInitMode, wiresAdded)


cdef custatevecStatus_t custatevecExAbs2SumArray(custatevecExStateVectorDescriptor_t stateVector, double* abs2sum, const int32_t* outputOrdering, int32_t outputOrderingLen, const int32_t* maskBitString, const int32_t* maskWireOrdering, int32_t maskLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExAbs2SumArray(stateVector, abs2sum, outputOrdering, outputOrderingLen, maskBitString, maskWireOrdering, maskLen)


cdef custatevecStatus_t custatevecExMeasure(custatevecExStateVectorDescriptor_t stateVector, custatevecIndex_t* bitString, const int32_t* bitStringOrdering, int32_t bitStringOrderingLen, double randnum, custatevecCollapseOp_t collapse, const void* reserved) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExMeasure(stateVector, bitString, bitStringOrdering, bitStringOrderingLen, randnum, collapse, reserved)


cdef custatevecStatus_t custatevecExSample(custatevecExStateVectorDescriptor_t stateVector, custatevecIndex_t* bitStrings, const int32_t* bitStringOrdering, int32_t bitStringOrderingLen, const double* randnums, int32_t numShots, custatevecSamplerOutput_t output, const double* abs2Sums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExSample(stateVector, bitStrings, bitStringOrdering, bitStringOrderingLen, randnums, numShots, output, abs2Sums)


cdef custatevecStatus_t custatevecExApplyMatrix(custatevecExStateVectorDescriptor_t stateVector, const void* matrix, cudaDataType_t matrixDataType, custatevecExMatrixType_t exMatrixType, custatevecMatrixLayout_t layout, int32_t adjoint, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExApplyMatrix(stateVector, matrix, matrixDataType, exMatrixType, layout, adjoint, targets, numTargets, controls, controlBitValues, numControls)


cdef custatevecStatus_t custatevecExApplyPauliRotation(custatevecExStateVectorDescriptor_t stateVector, double theta, const custatevecPauli_t* paulis, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExApplyPauliRotation(stateVector, theta, paulis, targets, numTargets, controls, controlBitValues, numControls)


cdef custatevecStatus_t custatevecExComputeExpectationOnPauliBasis(custatevecExStateVectorDescriptor_t stateVector, double* expectationValues, const custatevecPauli_t** pauliOperatorArrays, int32_t numPauliOperatorArrays, const int32_t** basisWiresArray, const int32_t* numBasisWiresArray) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExComputeExpectationOnPauliBasis(stateVector, expectationValues, pauliOperatorArrays, numPauliOperatorArrays, basisWiresArray, numBasisWiresArray)


cdef custatevecStatus_t custatevecExComputeExpectation(custatevecExStateVectorDescriptor_t stateVector, double2* expectationValues, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, int32_t numMatrices, const int32_t* basisWires, int32_t numBasisWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExComputeExpectation(stateVector, expectationValues, matrices, matrixDataType, layout, numMatrices, basisWires, numBasisWires)


cdef custatevecStatus_t custatevecExConfigureSVUpdater(custatevecExDictionaryDescriptor_t* svUpdaterConfig, cudaDataType_t dataType, const custatevecExSVUpdaterConfigItem_t* configItems, int32_t numConfigItems) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExConfigureSVUpdater(svUpdaterConfig, dataType, configItems, numConfigItems)


cdef custatevecStatus_t custatevecExSVUpdaterCreate(custatevecExSVUpdaterDescriptor_t* svUpdater, const custatevecExDictionaryDescriptor_t svUpdaterConfig, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExSVUpdaterCreate(svUpdater, svUpdaterConfig, resourceManager)


cdef custatevecStatus_t custatevecExSVUpdaterDestroy(custatevecExSVUpdaterDescriptor_t svUpdater) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExSVUpdaterDestroy(svUpdater)


cdef custatevecStatus_t custatevecExSVUpdaterClear(custatevecExSVUpdaterDescriptor_t svUpdater) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExSVUpdaterClear(svUpdater)


cdef custatevecStatus_t custatevecExSVUpdaterEnqueueMatrix(custatevecExSVUpdaterDescriptor_t svUpdater, const void* matrix, cudaDataType_t matrixDataType, custatevecExMatrixType_t exMatrixType, custatevecMatrixLayout_t layout, int32_t adjoint, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExSVUpdaterEnqueueMatrix(svUpdater, matrix, matrixDataType, exMatrixType, layout, adjoint, targets, numTargets, controls, controlBitValues, numControls)


cdef custatevecStatus_t custatevecExSVUpdaterEnqueueUnitaryChannel(custatevecExSVUpdaterDescriptor_t svUpdater, const void* const* unitaries, cudaDataType_t unitariesDataType, const custatevecExMatrixType_t* exMatrixTypes, int32_t numUnitaries, custatevecMatrixLayout_t layout, const double* probabilities, const int32_t* channelWires, int32_t numChannelWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExSVUpdaterEnqueueUnitaryChannel(svUpdater, unitaries, unitariesDataType, exMatrixTypes, numUnitaries, layout, probabilities, channelWires, numChannelWires)


cdef custatevecStatus_t custatevecExSVUpdaterEnqueueGeneralChannel(custatevecExSVUpdaterDescriptor_t svUpdater, const void* const* matrices, cudaDataType_t matrixDataType, const custatevecExMatrixType_t* exMatrixTypes, int32_t numMatrices, custatevecMatrixLayout_t layout, const int32_t* channelWires, int32_t numChannelWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExSVUpdaterEnqueueGeneralChannel(svUpdater, matrices, matrixDataType, exMatrixTypes, numMatrices, layout, channelWires, numChannelWires)


cdef custatevecStatus_t custatevecExSVUpdaterGetMaxNumRequiredRandnums(custatevecExSVUpdaterDescriptor_t svUpdater, int32_t* maxNumRequiredRandnums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExSVUpdaterGetMaxNumRequiredRandnums(svUpdater, maxNumRequiredRandnums)


cdef custatevecStatus_t custatevecExSVUpdaterApply(custatevecExSVUpdaterDescriptor_t svUpdater, custatevecExStateVectorDescriptor_t stateVector, const double* randnums, int32_t numRandnums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custatevecEx._custatevecExSVUpdaterApply(svUpdater, stateVector, randnums, numRandnums)
