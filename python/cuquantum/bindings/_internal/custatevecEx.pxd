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

from ..cycustatevecEx cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef custatevecStatus_t _custatevecExDictionaryDestroy(custatevecExDictionaryDescriptor_t dictionary) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExCommunicatorInitialize(custatevecCommunicatorType_t communicatorType, const char* libraryPath, int* argc, char*** argv, custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExCommunicatorFinalize(custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExCommunicatorGetSizeAndRank(int32_t* size, int32_t* rank, custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExCommunicatorCreate(custatevecExCommunicatorDescriptor_t* exCommunicator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExCommunicatorDestroy(custatevecExCommunicatorDescriptor_t exCommunicator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExConfigureStateVectorSingleDevice(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, int32_t deviceId, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExConfigureStateVectorMultiDevice(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, const int32_t* deviceIds, int32_t numDevices, custatevecDeviceNetworkType_t networkType, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExConfigureStateVectorMultiProcess(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, int32_t deviceId, custatevecExMemorySharingMethod_t memorySharingMethod, const custatevecExGlobalIndexBitClass_t* globalIndexBitClasses, const int32_t* numGlobalIndexBitsPerLayer, int32_t numGlobalIndexBitLayers, size_t transferWorkspaceSizeInBytes, const void* auxConfig, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorCreateSingleProcess(custatevecExStateVectorDescriptor_t* stateVector, const custatevecExDictionaryDescriptor_t svConfig, const cudaStream_t* streams, int32_t numStreams, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorCreateMultiProcess(custatevecExStateVectorDescriptor_t* stateVector, const custatevecExDictionaryDescriptor_t svConfig, cudaStream_t stream, custatevecExCommunicatorDescriptor_t exCommunicator, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorDestroy(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorGetProperty(const custatevecExStateVectorDescriptor_t stateVector, custatevecExStateVectorProperty_t property, void* value, size_t sizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorSetMathMode(custatevecExStateVectorDescriptor_t stateVector, custatevecMathMode_t mode) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorSetZeroState(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorGetState(const custatevecExStateVectorDescriptor_t stateVector, void* state, cudaDataType_t dataType, custatevecIndex_t begin, custatevecIndex_t end, int32_t maxNumConcurrentCopies) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorSetState(custatevecExStateVectorDescriptor_t stateVector, const void* state, cudaDataType_t dataType, custatevecIndex_t begin, custatevecIndex_t end, int32_t maxNumConcurrentCopies) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorReassignWireOrdering(custatevecExStateVectorDescriptor_t stateVector, const int32_t* wireOrdering, int32_t wireOrderingLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorPermuteIndexBits(custatevecExStateVectorDescriptor_t stateVector, const int32_t* permutation, int32_t permutationLen, custatevecExPermutationType_t permutationType) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorStageSubSV(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorExposeResources(custatevecExStateVectorDescriptor_t stateVector, custatevecExExposeResources_t exposeResources) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorGetResourcesFromDeviceSubSV(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t* deviceId, void** d_subSV, cudaStream_t* stream, custatevecHandle_t* handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorGetResourcesFromDeviceSubSVView(const custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t* deviceId, const void** d_subSV, cudaStream_t* stream, custatevecHandle_t* handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t sliceIndex, void** subSVSlice, custatevecExMemoryPlacement_t* placement, int32_t* deviceId, cudaStream_t* stream) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView(const custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t sliceIndex, const void** subSVSlice, custatevecExMemoryPlacement_t* placement, int32_t* deviceId, cudaStream_t* stream) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorSynchronize(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorSynchronizeScoped(custatevecExStateVectorDescriptor_t stateVector, custatevecExSynchronizationScope_t scope) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExStateVectorAddWires(custatevecExStateVectorDescriptor_t stateVector, custatevecExIndexBitDomain_t indexBitDomain, int32_t numWiresToAdd, custatevecExWireInitMode_t wireInitMode, int32_t* wiresAdded) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExAbs2SumArray(custatevecExStateVectorDescriptor_t stateVector, double* abs2sum, const int32_t* outputOrdering, int32_t outputOrderingLen, const int32_t* maskBitString, const int32_t* maskWireOrdering, int32_t maskLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExMeasure(custatevecExStateVectorDescriptor_t stateVector, custatevecIndex_t* bitString, const int32_t* bitStringOrdering, int32_t bitStringOrderingLen, double randnum, custatevecCollapseOp_t collapse, const void* reserved) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExSample(custatevecExStateVectorDescriptor_t stateVector, custatevecIndex_t* bitStrings, const int32_t* bitStringOrdering, int32_t bitStringOrderingLen, const double* randnums, int32_t numShots, custatevecSamplerOutput_t output, const double* abs2Sums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExApplyMatrix(custatevecExStateVectorDescriptor_t stateVector, const void* matrix, cudaDataType_t matrixDataType, custatevecExMatrixType_t exMatrixType, custatevecMatrixLayout_t layout, int32_t adjoint, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExApplyPauliRotation(custatevecExStateVectorDescriptor_t stateVector, double theta, const custatevecPauli_t* paulis, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExComputeExpectationOnPauliBasis(custatevecExStateVectorDescriptor_t stateVector, double* expectationValues, const custatevecPauli_t** pauliOperatorArrays, int32_t numPauliOperatorArrays, const int32_t** basisWiresArray, const int32_t* numBasisWiresArray) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExComputeExpectation(custatevecExStateVectorDescriptor_t stateVector, double2* expectationValues, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, int32_t numMatrices, const int32_t* basisWires, int32_t numBasisWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExConfigureSVUpdater(custatevecExDictionaryDescriptor_t* svUpdaterConfig, cudaDataType_t dataType, const custatevecExSVUpdaterConfigItem_t* configItems, int32_t numConfigItems) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExSVUpdaterCreate(custatevecExSVUpdaterDescriptor_t* svUpdater, const custatevecExDictionaryDescriptor_t svUpdaterConfig, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExSVUpdaterDestroy(custatevecExSVUpdaterDescriptor_t svUpdater) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExSVUpdaterClear(custatevecExSVUpdaterDescriptor_t svUpdater) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExSVUpdaterEnqueueMatrix(custatevecExSVUpdaterDescriptor_t svUpdater, const void* matrix, cudaDataType_t matrixDataType, custatevecExMatrixType_t exMatrixType, custatevecMatrixLayout_t layout, int32_t adjoint, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExSVUpdaterEnqueueUnitaryChannel(custatevecExSVUpdaterDescriptor_t svUpdater, const void* const* unitaries, cudaDataType_t unitariesDataType, const custatevecExMatrixType_t* exMatrixTypes, int32_t numUnitaries, custatevecMatrixLayout_t layout, const double* probabilities, const int32_t* channelWires, int32_t numChannelWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExSVUpdaterEnqueueGeneralChannel(custatevecExSVUpdaterDescriptor_t svUpdater, const void* const* matrices, cudaDataType_t matrixDataType, const custatevecExMatrixType_t* exMatrixTypes, int32_t numMatrices, custatevecMatrixLayout_t layout, const int32_t* channelWires, int32_t numChannelWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExSVUpdaterGetMaxNumRequiredRandnums(custatevecExSVUpdaterDescriptor_t svUpdater, int32_t* maxNumRequiredRandnums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t _custatevecExSVUpdaterApply(custatevecExSVUpdaterDescriptor_t svUpdater, custatevecExStateVectorDescriptor_t stateVector, const double* randnums, int32_t numRandnums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
