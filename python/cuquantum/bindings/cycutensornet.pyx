# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 25.03.0. Do not modify it directly.

from ._internal cimport cutensornet as _cutensornet


###############################################################################
# Wrapper functions
###############################################################################

cdef cutensornetStatus_t cutensornetCreate(cutensornetHandle_t* handle) except* nogil:
    return _cutensornet._cutensornetCreate(handle)


cdef cutensornetStatus_t cutensornetDestroy(cutensornetHandle_t handle) except* nogil:
    return _cutensornet._cutensornetDestroy(handle)


cdef cutensornetStatus_t cutensornetCreateNetworkDescriptor(const cutensornetHandle_t handle, int32_t numInputs, const int32_t numModesIn[], const int64_t* const extentsIn[], const int64_t* const stridesIn[], const int32_t* const modesIn[], const cutensornetTensorQualifiers_t qualifiersIn[], int32_t numModesOut, const int64_t extentsOut[], const int64_t stridesOut[], const int32_t modesOut[], cudaDataType_t dataType, cutensornetComputeType_t computeType, cutensornetNetworkDescriptor_t* descNet) except* nogil:
    return _cutensornet._cutensornetCreateNetworkDescriptor(handle, numInputs, numModesIn, extentsIn, stridesIn, modesIn, qualifiersIn, numModesOut, extentsOut, stridesOut, modesOut, dataType, computeType, descNet)


cdef cutensornetStatus_t cutensornetDestroyNetworkDescriptor(cutensornetNetworkDescriptor_t desc) except* nogil:
    return _cutensornet._cutensornetDestroyNetworkDescriptor(desc)


cdef cutensornetStatus_t cutensornetGetOutputTensorDescriptor(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, cutensornetTensorDescriptor_t* outputTensorDesc) except* nogil:
    return _cutensornet._cutensornetGetOutputTensorDescriptor(handle, descNet, outputTensorDesc)


cdef cutensornetStatus_t cutensornetGetTensorDetails(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t tensorDesc, int32_t* numModes, size_t* dataSize, int32_t* modeLabels, int64_t* extents, int64_t* strides) except* nogil:
    return _cutensornet._cutensornetGetTensorDetails(handle, tensorDesc, numModes, dataSize, modeLabels, extents, strides)


cdef cutensornetStatus_t cutensornetCreateWorkspaceDescriptor(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t* workDesc) except* nogil:
    return _cutensornet._cutensornetCreateWorkspaceDescriptor(handle, workDesc)


cdef cutensornetStatus_t cutensornetWorkspaceComputeContractionSizes(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetWorkspaceDescriptor_t workDesc) except* nogil:
    return _cutensornet._cutensornetWorkspaceComputeContractionSizes(handle, descNet, optimizerInfo, workDesc)


cdef cutensornetStatus_t cutensornetWorkspaceGetMemorySize(const cutensornetHandle_t handle, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetWorksizePref_t workPref, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, int64_t* memorySize) except* nogil:
    return _cutensornet._cutensornetWorkspaceGetMemorySize(handle, workDesc, workPref, memSpace, workKind, memorySize)


cdef cutensornetStatus_t cutensornetWorkspaceSetMemory(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, void* const memoryPtr, int64_t memorySize) except* nogil:
    return _cutensornet._cutensornetWorkspaceSetMemory(handle, workDesc, memSpace, workKind, memoryPtr, memorySize)


cdef cutensornetStatus_t cutensornetWorkspaceGetMemory(const cutensornetHandle_t handle, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, void** memoryPtr, int64_t* memorySize) except* nogil:
    return _cutensornet._cutensornetWorkspaceGetMemory(handle, workDesc, memSpace, workKind, memoryPtr, memorySize)


cdef cutensornetStatus_t cutensornetDestroyWorkspaceDescriptor(cutensornetWorkspaceDescriptor_t desc) except* nogil:
    return _cutensornet._cutensornetDestroyWorkspaceDescriptor(desc)


cdef cutensornetStatus_t cutensornetCreateContractionOptimizerConfig(const cutensornetHandle_t handle, cutensornetContractionOptimizerConfig_t* optimizerConfig) except* nogil:
    return _cutensornet._cutensornetCreateContractionOptimizerConfig(handle, optimizerConfig)


cdef cutensornetStatus_t cutensornetDestroyContractionOptimizerConfig(cutensornetContractionOptimizerConfig_t optimizerConfig) except* nogil:
    return _cutensornet._cutensornetDestroyContractionOptimizerConfig(optimizerConfig)


cdef cutensornetStatus_t cutensornetContractionOptimizerConfigGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionOptimizerConfig_t optimizerConfig, cutensornetContractionOptimizerConfigAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetContractionOptimizerConfigGetAttribute(handle, optimizerConfig, attr, buf, sizeInBytes)


cdef cutensornetStatus_t cutensornetContractionOptimizerConfigSetAttribute(const cutensornetHandle_t handle, cutensornetContractionOptimizerConfig_t optimizerConfig, cutensornetContractionOptimizerConfigAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetContractionOptimizerConfigSetAttribute(handle, optimizerConfig, attr, buf, sizeInBytes)


cdef cutensornetStatus_t cutensornetDestroyContractionOptimizerInfo(cutensornetContractionOptimizerInfo_t optimizerInfo) except* nogil:
    return _cutensornet._cutensornetDestroyContractionOptimizerInfo(optimizerInfo)


cdef cutensornetStatus_t cutensornetCreateContractionOptimizerInfo(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, cutensornetContractionOptimizerInfo_t* optimizerInfo) except* nogil:
    return _cutensornet._cutensornetCreateContractionOptimizerInfo(handle, descNet, optimizerInfo)


cdef cutensornetStatus_t cutensornetContractionOptimize(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const cutensornetContractionOptimizerConfig_t optimizerConfig, uint64_t workspaceSizeConstraint, cutensornetContractionOptimizerInfo_t optimizerInfo) except* nogil:
    return _cutensornet._cutensornetContractionOptimize(handle, descNet, optimizerConfig, workspaceSizeConstraint, optimizerInfo)


cdef cutensornetStatus_t cutensornetContractionOptimizerInfoGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetContractionOptimizerInfoAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetContractionOptimizerInfoGetAttribute(handle, optimizerInfo, attr, buf, sizeInBytes)


cdef cutensornetStatus_t cutensornetContractionOptimizerInfoSetAttribute(const cutensornetHandle_t handle, cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetContractionOptimizerInfoAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetContractionOptimizerInfoSetAttribute(handle, optimizerInfo, attr, buf, sizeInBytes)


cdef cutensornetStatus_t cutensornetContractionOptimizerInfoGetPackedSize(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, size_t* sizeInBytes) except* nogil:
    return _cutensornet._cutensornetContractionOptimizerInfoGetPackedSize(handle, optimizerInfo, sizeInBytes)


cdef cutensornetStatus_t cutensornetContractionOptimizerInfoPackData(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, void* buffer, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetContractionOptimizerInfoPackData(handle, optimizerInfo, buffer, sizeInBytes)


cdef cutensornetStatus_t cutensornetCreateContractionOptimizerInfoFromPackedData(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const void* buffer, size_t sizeInBytes, cutensornetContractionOptimizerInfo_t* optimizerInfo) except* nogil:
    return _cutensornet._cutensornetCreateContractionOptimizerInfoFromPackedData(handle, descNet, buffer, sizeInBytes, optimizerInfo)


cdef cutensornetStatus_t cutensornetUpdateContractionOptimizerInfoFromPackedData(const cutensornetHandle_t handle, const void* buffer, size_t sizeInBytes, cutensornetContractionOptimizerInfo_t optimizerInfo) except* nogil:
    return _cutensornet._cutensornetUpdateContractionOptimizerInfoFromPackedData(handle, buffer, sizeInBytes, optimizerInfo)


cdef cutensornetStatus_t cutensornetCreateContractionPlan(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const cutensornetContractionOptimizerInfo_t optimizerInfo, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetContractionPlan_t* plan) except* nogil:
    return _cutensornet._cutensornetCreateContractionPlan(handle, descNet, optimizerInfo, workDesc, plan)


cdef cutensornetStatus_t cutensornetDestroyContractionPlan(cutensornetContractionPlan_t plan) except* nogil:
    return _cutensornet._cutensornetDestroyContractionPlan(plan)


cdef cutensornetStatus_t cutensornetContractionAutotune(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan, const void* const rawDataIn[], void* rawDataOut, cutensornetWorkspaceDescriptor_t workDesc, const cutensornetContractionAutotunePreference_t pref, cudaStream_t stream) except* nogil:
    return _cutensornet._cutensornetContractionAutotune(handle, plan, rawDataIn, rawDataOut, workDesc, pref, stream)


cdef cutensornetStatus_t cutensornetCreateContractionAutotunePreference(const cutensornetHandle_t handle, cutensornetContractionAutotunePreference_t* autotunePreference) except* nogil:
    return _cutensornet._cutensornetCreateContractionAutotunePreference(handle, autotunePreference)


cdef cutensornetStatus_t cutensornetContractionAutotunePreferenceGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionAutotunePreference_t autotunePreference, cutensornetContractionAutotunePreferenceAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetContractionAutotunePreferenceGetAttribute(handle, autotunePreference, attr, buf, sizeInBytes)


cdef cutensornetStatus_t cutensornetContractionAutotunePreferenceSetAttribute(const cutensornetHandle_t handle, cutensornetContractionAutotunePreference_t autotunePreference, cutensornetContractionAutotunePreferenceAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetContractionAutotunePreferenceSetAttribute(handle, autotunePreference, attr, buf, sizeInBytes)


cdef cutensornetStatus_t cutensornetDestroyContractionAutotunePreference(cutensornetContractionAutotunePreference_t autotunePreference) except* nogil:
    return _cutensornet._cutensornetDestroyContractionAutotunePreference(autotunePreference)


cdef cutensornetStatus_t cutensornetCreateSliceGroupFromIDRange(const cutensornetHandle_t handle, int64_t sliceIdStart, int64_t sliceIdStop, int64_t sliceIdStep, cutensornetSliceGroup_t* sliceGroup) except* nogil:
    return _cutensornet._cutensornetCreateSliceGroupFromIDRange(handle, sliceIdStart, sliceIdStop, sliceIdStep, sliceGroup)


cdef cutensornetStatus_t cutensornetCreateSliceGroupFromIDs(const cutensornetHandle_t handle, const int64_t* beginIDSequence, const int64_t* endIDSequence, cutensornetSliceGroup_t* sliceGroup) except* nogil:
    return _cutensornet._cutensornetCreateSliceGroupFromIDs(handle, beginIDSequence, endIDSequence, sliceGroup)


cdef cutensornetStatus_t cutensornetDestroySliceGroup(cutensornetSliceGroup_t sliceGroup) except* nogil:
    return _cutensornet._cutensornetDestroySliceGroup(sliceGroup)


cdef cutensornetStatus_t cutensornetContractSlices(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan, const void* const rawDataIn[], void* rawDataOut, int32_t accumulateOutput, cutensornetWorkspaceDescriptor_t workDesc, const cutensornetSliceGroup_t sliceGroup, cudaStream_t stream) except* nogil:
    return _cutensornet._cutensornetContractSlices(handle, plan, rawDataIn, rawDataOut, accumulateOutput, workDesc, sliceGroup, stream)


cdef cutensornetStatus_t cutensornetCreateTensorDescriptor(const cutensornetHandle_t handle, int32_t numModes, const int64_t extents[], const int64_t strides[], const int32_t modes[], cudaDataType_t dataType, cutensornetTensorDescriptor_t* descTensor) except* nogil:
    return _cutensornet._cutensornetCreateTensorDescriptor(handle, numModes, extents, strides, modes, dataType, descTensor)


cdef cutensornetStatus_t cutensornetDestroyTensorDescriptor(cutensornetTensorDescriptor_t descTensor) except* nogil:
    return _cutensornet._cutensornetDestroyTensorDescriptor(descTensor)


cdef cutensornetStatus_t cutensornetCreateTensorSVDConfig(const cutensornetHandle_t handle, cutensornetTensorSVDConfig_t* svdConfig) except* nogil:
    return _cutensornet._cutensornetCreateTensorSVDConfig(handle, svdConfig)


cdef cutensornetStatus_t cutensornetDestroyTensorSVDConfig(cutensornetTensorSVDConfig_t svdConfig) except* nogil:
    return _cutensornet._cutensornetDestroyTensorSVDConfig(svdConfig)


cdef cutensornetStatus_t cutensornetTensorSVDConfigGetAttribute(const cutensornetHandle_t handle, const cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDConfigAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetTensorSVDConfigGetAttribute(handle, svdConfig, attr, buf, sizeInBytes)


cdef cutensornetStatus_t cutensornetTensorSVDConfigSetAttribute(const cutensornetHandle_t handle, cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDConfigAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetTensorSVDConfigSetAttribute(handle, svdConfig, attr, buf, sizeInBytes)


cdef cutensornetStatus_t cutensornetWorkspaceComputeSVDSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const cutensornetTensorDescriptor_t descTensorU, const cutensornetTensorDescriptor_t descTensorV, const cutensornetTensorSVDConfig_t svdConfig, cutensornetWorkspaceDescriptor_t workDesc) except* nogil:
    return _cutensornet._cutensornetWorkspaceComputeSVDSizes(handle, descTensorIn, descTensorU, descTensorV, svdConfig, workDesc)


cdef cutensornetStatus_t cutensornetWorkspaceComputeQRSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const cutensornetTensorDescriptor_t descTensorQ, const cutensornetTensorDescriptor_t descTensorR, cutensornetWorkspaceDescriptor_t workDesc) except* nogil:
    return _cutensornet._cutensornetWorkspaceComputeQRSizes(handle, descTensorIn, descTensorQ, descTensorR, workDesc)


cdef cutensornetStatus_t cutensornetCreateTensorSVDInfo(const cutensornetHandle_t handle, cutensornetTensorSVDInfo_t* svdInfo) except* nogil:
    return _cutensornet._cutensornetCreateTensorSVDInfo(handle, svdInfo)


cdef cutensornetStatus_t cutensornetTensorSVDInfoGetAttribute(const cutensornetHandle_t handle, const cutensornetTensorSVDInfo_t svdInfo, cutensornetTensorSVDInfoAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetTensorSVDInfoGetAttribute(handle, svdInfo, attr, buf, sizeInBytes)


cdef cutensornetStatus_t cutensornetDestroyTensorSVDInfo(cutensornetTensorSVDInfo_t svdInfo) except* nogil:
    return _cutensornet._cutensornetDestroyTensorSVDInfo(svdInfo)


cdef cutensornetStatus_t cutensornetTensorSVD(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const void* const rawDataIn, cutensornetTensorDescriptor_t descTensorU, void* u, void* s, cutensornetTensorDescriptor_t descTensorV, void* v, const cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDInfo_t svdInfo, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil:
    return _cutensornet._cutensornetTensorSVD(handle, descTensorIn, rawDataIn, descTensorU, u, s, descTensorV, v, svdConfig, svdInfo, workDesc, stream)


cdef cutensornetStatus_t cutensornetTensorQR(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const void* const rawDataIn, const cutensornetTensorDescriptor_t descTensorQ, void* q, const cutensornetTensorDescriptor_t descTensorR, void* r, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil:
    return _cutensornet._cutensornetTensorQR(handle, descTensorIn, rawDataIn, descTensorQ, q, descTensorR, r, workDesc, stream)


cdef cutensornetStatus_t cutensornetWorkspaceComputeGateSplitSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA, const cutensornetTensorDescriptor_t descTensorInB, const cutensornetTensorDescriptor_t descTensorInG, const cutensornetTensorDescriptor_t descTensorU, const cutensornetTensorDescriptor_t descTensorV, const cutensornetGateSplitAlgo_t gateAlgo, const cutensornetTensorSVDConfig_t svdConfig, cutensornetComputeType_t computeType, cutensornetWorkspaceDescriptor_t workDesc) except* nogil:
    return _cutensornet._cutensornetWorkspaceComputeGateSplitSizes(handle, descTensorInA, descTensorInB, descTensorInG, descTensorU, descTensorV, gateAlgo, svdConfig, computeType, workDesc)


cdef cutensornetStatus_t cutensornetGateSplit(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA, const void* rawDataInA, const cutensornetTensorDescriptor_t descTensorInB, const void* rawDataInB, const cutensornetTensorDescriptor_t descTensorInG, const void* rawDataInG, cutensornetTensorDescriptor_t descTensorU, void* u, void* s, cutensornetTensorDescriptor_t descTensorV, void* v, const cutensornetGateSplitAlgo_t gateAlgo, const cutensornetTensorSVDConfig_t svdConfig, cutensornetComputeType_t computeType, cutensornetTensorSVDInfo_t svdInfo, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil:
    return _cutensornet._cutensornetGateSplit(handle, descTensorInA, rawDataInA, descTensorInB, rawDataInB, descTensorInG, rawDataInG, descTensorU, u, s, descTensorV, v, gateAlgo, svdConfig, computeType, svdInfo, workDesc, stream)


cdef cutensornetStatus_t cutensornetGetDeviceMemHandler(const cutensornetHandle_t handle, cutensornetDeviceMemHandler_t* devMemHandler) except* nogil:
    return _cutensornet._cutensornetGetDeviceMemHandler(handle, devMemHandler)


cdef cutensornetStatus_t cutensornetSetDeviceMemHandler(cutensornetHandle_t handle, const cutensornetDeviceMemHandler_t* devMemHandler) except* nogil:
    return _cutensornet._cutensornetSetDeviceMemHandler(handle, devMemHandler)


cdef cutensornetStatus_t cutensornetLoggerSetCallback(cutensornetLoggerCallback_t callback) except* nogil:
    return _cutensornet._cutensornetLoggerSetCallback(callback)


cdef cutensornetStatus_t cutensornetLoggerSetCallbackData(cutensornetLoggerCallbackData_t callback, void* userData) except* nogil:
    return _cutensornet._cutensornetLoggerSetCallbackData(callback, userData)


cdef cutensornetStatus_t cutensornetLoggerSetFile(FILE* file) except* nogil:
    return _cutensornet._cutensornetLoggerSetFile(file)


cdef cutensornetStatus_t cutensornetLoggerOpenFile(const char* logFile) except* nogil:
    return _cutensornet._cutensornetLoggerOpenFile(logFile)


cdef cutensornetStatus_t cutensornetLoggerSetLevel(int32_t level) except* nogil:
    return _cutensornet._cutensornetLoggerSetLevel(level)


cdef cutensornetStatus_t cutensornetLoggerSetMask(int32_t mask) except* nogil:
    return _cutensornet._cutensornetLoggerSetMask(mask)


cdef cutensornetStatus_t cutensornetLoggerForceDisable() except* nogil:
    return _cutensornet._cutensornetLoggerForceDisable()


cdef size_t cutensornetGetVersion() except* nogil:
    return _cutensornet._cutensornetGetVersion()


cdef size_t cutensornetGetCudartVersion() except* nogil:
    return _cutensornet._cutensornetGetCudartVersion()


cdef const char* cutensornetGetErrorString(cutensornetStatus_t error) except* nogil:
    return _cutensornet._cutensornetGetErrorString(error)


cdef cutensornetStatus_t cutensornetDistributedResetConfiguration(cutensornetHandle_t handle, const void* commPtr, size_t commSize) except* nogil:
    return _cutensornet._cutensornetDistributedResetConfiguration(handle, commPtr, commSize)


cdef cutensornetStatus_t cutensornetDistributedGetNumRanks(const cutensornetHandle_t handle, int32_t* numRanks) except* nogil:
    return _cutensornet._cutensornetDistributedGetNumRanks(handle, numRanks)


cdef cutensornetStatus_t cutensornetDistributedGetProcRank(const cutensornetHandle_t handle, int32_t* procRank) except* nogil:
    return _cutensornet._cutensornetDistributedGetProcRank(handle, procRank)


cdef cutensornetStatus_t cutensornetDistributedSynchronize(const cutensornetHandle_t handle) except* nogil:
    return _cutensornet._cutensornetDistributedSynchronize(handle)


cdef cutensornetStatus_t cutensornetNetworkGetAttribute(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, cutensornetNetworkAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetNetworkGetAttribute(handle, networkDesc, attr, buf, sizeInBytes)


cdef cutensornetStatus_t cutensornetNetworkSetAttribute(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, cutensornetNetworkAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    return _cutensornet._cutensornetNetworkSetAttribute(handle, networkDesc, attr, buf, sizeInBytes)


cdef cutensornetStatus_t cutensornetWorkspacePurgeCache(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace) except* nogil:
    return _cutensornet._cutensornetWorkspacePurgeCache(handle, workDesc, memSpace)


cdef cutensornetStatus_t cutensornetComputeGradientsBackward(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan, const void* const rawDataIn[], const void* outputGradient, void* const gradients[], int32_t accumulateOutput, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil:
    return _cutensornet._cutensornetComputeGradientsBackward(handle, plan, rawDataIn, outputGradient, gradients, accumulateOutput, workDesc, stream)


cdef cutensornetStatus_t cutensornetCreateState(const cutensornetHandle_t handle, cutensornetStatePurity_t purity, int32_t numStateModes, const int64_t* stateModeExtents, cudaDataType_t dataType, cutensornetState_t* tensorNetworkState) except* nogil:
    return _cutensornet._cutensornetCreateState(handle, purity, numStateModes, stateModeExtents, dataType, tensorNetworkState)


cdef cutensornetStatus_t cutensornetStateApplyTensor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except* nogil:
    return _cutensornet._cutensornetStateApplyTensor(handle, tensorNetworkState, numStateModes, stateModes, tensorData, tensorModeStrides, immutable, adjoint, unitary, tensorId)


cdef cutensornetStatus_t cutensornetStateUpdateTensor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int64_t tensorId, void* tensorData, int32_t unitary) except* nogil:
    return _cutensornet._cutensornetStateUpdateTensor(handle, tensorNetworkState, tensorId, tensorData, unitary)


cdef cutensornetStatus_t cutensornetDestroyState(cutensornetState_t tensorNetworkState) except* nogil:
    return _cutensornet._cutensornetDestroyState(tensorNetworkState)


cdef cutensornetStatus_t cutensornetCreateMarginal(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numMarginalModes, const int32_t* marginalModes, int32_t numProjectedModes, const int32_t* projectedModes, const int64_t* marginalTensorStrides, cutensornetStateMarginal_t* tensorNetworkMarginal) except* nogil:
    return _cutensornet._cutensornetCreateMarginal(handle, tensorNetworkState, numMarginalModes, marginalModes, numProjectedModes, projectedModes, marginalTensorStrides, tensorNetworkMarginal)


cdef cutensornetStatus_t cutensornetMarginalConfigure(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, cutensornetMarginalAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil:
    return _cutensornet._cutensornetMarginalConfigure(handle, tensorNetworkMarginal, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t cutensornetMarginalPrepare(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil:
    return _cutensornet._cutensornetMarginalPrepare(handle, tensorNetworkMarginal, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t cutensornetMarginalCompute(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, const int64_t* projectedModeValues, cutensornetWorkspaceDescriptor_t workDesc, void* marginalTensor, cudaStream_t cudaStream) except* nogil:
    return _cutensornet._cutensornetMarginalCompute(handle, tensorNetworkMarginal, projectedModeValues, workDesc, marginalTensor, cudaStream)


cdef cutensornetStatus_t cutensornetDestroyMarginal(cutensornetStateMarginal_t tensorNetworkMarginal) except* nogil:
    return _cutensornet._cutensornetDestroyMarginal(tensorNetworkMarginal)


cdef cutensornetStatus_t cutensornetCreateSampler(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numModesToSample, const int32_t* modesToSample, cutensornetStateSampler_t* tensorNetworkSampler) except* nogil:
    return _cutensornet._cutensornetCreateSampler(handle, tensorNetworkState, numModesToSample, modesToSample, tensorNetworkSampler)


cdef cutensornetStatus_t cutensornetSamplerConfigure(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, cutensornetSamplerAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil:
    return _cutensornet._cutensornetSamplerConfigure(handle, tensorNetworkSampler, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t cutensornetSamplerPrepare(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil:
    return _cutensornet._cutensornetSamplerPrepare(handle, tensorNetworkSampler, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t cutensornetSamplerSample(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, int64_t numShots, cutensornetWorkspaceDescriptor_t workDesc, int64_t* samples, cudaStream_t cudaStream) except* nogil:
    return _cutensornet._cutensornetSamplerSample(handle, tensorNetworkSampler, numShots, workDesc, samples, cudaStream)


cdef cutensornetStatus_t cutensornetDestroySampler(cutensornetStateSampler_t tensorNetworkSampler) except* nogil:
    return _cutensornet._cutensornetDestroySampler(tensorNetworkSampler)


cdef cutensornetStatus_t cutensornetStateFinalizeMPS(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetBoundaryCondition_t boundaryCondition, const int64_t* const extentsOut[], const int64_t* const stridesOut[]) except* nogil:
    return _cutensornet._cutensornetStateFinalizeMPS(handle, tensorNetworkState, boundaryCondition, extentsOut, stridesOut)


cdef cutensornetStatus_t cutensornetStateConfigure(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetStateAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil:
    return _cutensornet._cutensornetStateConfigure(handle, tensorNetworkState, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t cutensornetStatePrepare(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil:
    return _cutensornet._cutensornetStatePrepare(handle, tensorNetworkState, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t cutensornetStateCompute(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetWorkspaceDescriptor_t workDesc, int64_t* extentsOut[], int64_t* stridesOut[], void* stateTensorsOut[], cudaStream_t cudaStream) except* nogil:
    return _cutensornet._cutensornetStateCompute(handle, tensorNetworkState, workDesc, extentsOut, stridesOut, stateTensorsOut, cudaStream)


cdef cutensornetStatus_t cutensornetGetOutputStateDetails(const cutensornetHandle_t handle, const cutensornetState_t tensorNetworkState, int32_t* numTensorsOut, int32_t numModesOut[], int64_t* extentsOut[], int64_t* stridesOut[]) except* nogil:
    return _cutensornet._cutensornetGetOutputStateDetails(handle, tensorNetworkState, numTensorsOut, numModesOut, extentsOut, stridesOut)


cdef cutensornetStatus_t cutensornetCreateNetworkOperator(const cutensornetHandle_t handle, int32_t numStateModes, const int64_t stateModeExtents[], cudaDataType_t dataType, cutensornetNetworkOperator_t* tensorNetworkOperator) except* nogil:
    return _cutensornet._cutensornetCreateNetworkOperator(handle, numStateModes, stateModeExtents, dataType, tensorNetworkOperator)


cdef cutensornetStatus_t cutensornetNetworkOperatorAppendProduct(const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator, cuDoubleComplex coefficient, int32_t numTensors, const int32_t numStateModes[], const int32_t* stateModes[], const int64_t* tensorModeStrides[], const void* tensorData[], int64_t* componentId) except* nogil:
    return _cutensornet._cutensornetNetworkOperatorAppendProduct(handle, tensorNetworkOperator, coefficient, numTensors, numStateModes, stateModes, tensorModeStrides, tensorData, componentId)


cdef cutensornetStatus_t cutensornetDestroyNetworkOperator(cutensornetNetworkOperator_t tensorNetworkOperator) except* nogil:
    return _cutensornet._cutensornetDestroyNetworkOperator(tensorNetworkOperator)


cdef cutensornetStatus_t cutensornetCreateAccessor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numProjectedModes, const int32_t* projectedModes, const int64_t* amplitudesTensorStrides, cutensornetStateAccessor_t* tensorNetworkAccessor) except* nogil:
    return _cutensornet._cutensornetCreateAccessor(handle, tensorNetworkState, numProjectedModes, projectedModes, amplitudesTensorStrides, tensorNetworkAccessor)


cdef cutensornetStatus_t cutensornetAccessorConfigure(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, cutensornetAccessorAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil:
    return _cutensornet._cutensornetAccessorConfigure(handle, tensorNetworkAccessor, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t cutensornetAccessorPrepare(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil:
    return _cutensornet._cutensornetAccessorPrepare(handle, tensorNetworkAccessor, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t cutensornetAccessorCompute(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, const int64_t* projectedModeValues, cutensornetWorkspaceDescriptor_t workDesc, void* amplitudesTensor, void* stateNorm, cudaStream_t cudaStream) except* nogil:
    return _cutensornet._cutensornetAccessorCompute(handle, tensorNetworkAccessor, projectedModeValues, workDesc, amplitudesTensor, stateNorm, cudaStream)


cdef cutensornetStatus_t cutensornetDestroyAccessor(cutensornetStateAccessor_t tensorNetworkAccessor) except* nogil:
    return _cutensornet._cutensornetDestroyAccessor(tensorNetworkAccessor)


cdef cutensornetStatus_t cutensornetCreateExpectation(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetNetworkOperator_t tensorNetworkOperator, cutensornetStateExpectation_t* tensorNetworkExpectation) except* nogil:
    return _cutensornet._cutensornetCreateExpectation(handle, tensorNetworkState, tensorNetworkOperator, tensorNetworkExpectation)


cdef cutensornetStatus_t cutensornetExpectationConfigure(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetExpectationAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil:
    return _cutensornet._cutensornetExpectationConfigure(handle, tensorNetworkExpectation, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t cutensornetExpectationPrepare(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil:
    return _cutensornet._cutensornetExpectationPrepare(handle, tensorNetworkExpectation, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t cutensornetExpectationCompute(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetWorkspaceDescriptor_t workDesc, void* expectationValue, void* stateNorm, cudaStream_t cudaStream) except* nogil:
    return _cutensornet._cutensornetExpectationCompute(handle, tensorNetworkExpectation, workDesc, expectationValue, stateNorm, cudaStream)


cdef cutensornetStatus_t cutensornetDestroyExpectation(cutensornetStateExpectation_t tensorNetworkExpectation) except* nogil:
    return _cutensornet._cutensornetDestroyExpectation(tensorNetworkExpectation)


cdef cutensornetStatus_t cutensornetStateApplyTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except* nogil:
    return _cutensornet._cutensornetStateApplyTensorOperator(handle, tensorNetworkState, numStateModes, stateModes, tensorData, tensorModeStrides, immutable, adjoint, unitary, tensorId)


cdef cutensornetStatus_t cutensornetStateApplyControlledTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numControlModes, const int32_t* stateControlModes, const int64_t* stateControlValues, int32_t numTargetModes, const int32_t* stateTargetModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except* nogil:
    return _cutensornet._cutensornetStateApplyControlledTensorOperator(handle, tensorNetworkState, numControlModes, stateControlModes, stateControlValues, numTargetModes, stateTargetModes, tensorData, tensorModeStrides, immutable, adjoint, unitary, tensorId)


cdef cutensornetStatus_t cutensornetStateUpdateTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int64_t tensorId, void* tensorData, int32_t unitary) except* nogil:
    return _cutensornet._cutensornetStateUpdateTensorOperator(handle, tensorNetworkState, tensorId, tensorData, unitary)


cdef cutensornetStatus_t cutensornetStateApplyNetworkOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, const cutensornetNetworkOperator_t tensorNetworkOperator, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* operatorId) except* nogil:
    return _cutensornet._cutensornetStateApplyNetworkOperator(handle, tensorNetworkState, tensorNetworkOperator, immutable, adjoint, unitary, operatorId)


cdef cutensornetStatus_t cutensornetStateInitializeMPS(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetBoundaryCondition_t boundaryCondition, const int64_t* const extentsIn[], const int64_t* const stridesIn[], void* stateTensorsIn[]) except* nogil:
    return _cutensornet._cutensornetStateInitializeMPS(handle, tensorNetworkState, boundaryCondition, extentsIn, stridesIn, stateTensorsIn)


cdef cutensornetStatus_t cutensornetStateGetInfo(const cutensornetHandle_t handle, const cutensornetState_t tensorNetworkState, cutensornetStateAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil:
    return _cutensornet._cutensornetStateGetInfo(handle, tensorNetworkState, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t cutensornetNetworkOperatorAppendMPO(const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator, cuDoubleComplex coefficient, int32_t numStateModes, const int32_t stateModes[], const int64_t* tensorModeExtents[], const int64_t* tensorModeStrides[], const void* tensorData[], cutensornetBoundaryCondition_t boundaryCondition, int64_t* componentId) except* nogil:
    return _cutensornet._cutensornetNetworkOperatorAppendMPO(handle, tensorNetworkOperator, coefficient, numStateModes, stateModes, tensorModeExtents, tensorModeStrides, tensorData, boundaryCondition, componentId)


cdef cutensornetStatus_t cutensornetAccessorGetInfo(const cutensornetHandle_t handle, const cutensornetStateAccessor_t tensorNetworkAccessor, cutensornetAccessorAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil:
    return _cutensornet._cutensornetAccessorGetInfo(handle, tensorNetworkAccessor, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t cutensornetExpectationGetInfo(const cutensornetHandle_t handle, const cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetExpectationAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil:
    return _cutensornet._cutensornetExpectationGetInfo(handle, tensorNetworkExpectation, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t cutensornetMarginalGetInfo(const cutensornetHandle_t handle, const cutensornetStateMarginal_t tensorNetworkMarginal, cutensornetMarginalAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil:
    return _cutensornet._cutensornetMarginalGetInfo(handle, tensorNetworkMarginal, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t cutensornetSamplerGetInfo(const cutensornetHandle_t handle, const cutensornetStateSampler_t tensorNetworkSampler, cutensornetSamplerAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil:
    return _cutensornet._cutensornetSamplerGetInfo(handle, tensorNetworkSampler, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t cutensornetStateApplyUnitaryChannel(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, int32_t numTensors, void* tensorData[], const int64_t* tensorModeStrides, const double probabilities[], int64_t* channelId) except* nogil:
    return _cutensornet._cutensornetStateApplyUnitaryChannel(handle, tensorNetworkState, numStateModes, stateModes, numTensors, tensorData, tensorModeStrides, probabilities, channelId)


cdef cutensornetStatus_t cutensornetStateCaptureMPS(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState) except* nogil:
    return _cutensornet._cutensornetStateCaptureMPS(handle, tensorNetworkState)


cdef cutensornetStatus_t cutensornetStateApplyGeneralChannel(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, int32_t numTensors, void* tensorData[], const int64_t* tensorModeStrides, int64_t* channelId) except* nogil:
    return _cutensornet._cutensornetStateApplyGeneralChannel(handle, tensorNetworkState, numStateModes, stateModes, numTensors, tensorData, tensorModeStrides, channelId)
