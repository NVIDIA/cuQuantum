# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 26.09.0 to 26.11.0. Do not modify it directly.
# This layer exposes the C header to Cython as-is.


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    int32_t,
    uint32_t,
)


# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t

from .cycustatevec cimport (custatevecStatus_t, _CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR,
                            custatevecCommunicatorType_t, custatevecHandle_t,
                            custatevecMathMode_t, custatevecIndex_t,
                            custatevecCollapseOp_t, custatevecSamplerOutput_t,
                            custatevecPauli_t, custatevecMatrixLayout_t,
                            custatevecDeviceNetworkType_t, double2)


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum custatevecExCommunicatorStatus_t "custatevecExCommunicatorStatus_t":
    CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS "CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS" = 0

ctypedef enum custatevecExStateVectorCapability_t "custatevecExStateVectorCapability_t":
    CUSTATEVEC_EX_SV_CAPABILITY_NONE "CUSTATEVEC_EX_SV_CAPABILITY_NONE" = 0
    CUSTATEVEC_EX_SV_CAPABILITY_RESIZABLE "CUSTATEVEC_EX_SV_CAPABILITY_RESIZABLE" = (1U << 0)

ctypedef enum custatevecExStateVectorDistributionType_t "custatevecExStateVectorDistributionType_t":
    CUSTATEVEC_EX_SV_DISTRIBUTION_SINGLE_DEVICE "CUSTATEVEC_EX_SV_DISTRIBUTION_SINGLE_DEVICE" = 0
    CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_DEVICE "CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_DEVICE" = 1
    CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS "CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS" = 2

ctypedef enum custatevecExIndexBitDomain_t "custatevecExIndexBitDomain_t":
    CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL "CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL" = 0
    CUSTATEVEC_EX_INDEX_BIT_DOMAIN_MIGRATION "CUSTATEVEC_EX_INDEX_BIT_DOMAIN_MIGRATION" = 1
    CUSTATEVEC_EX_INDEX_BIT_DOMAIN_GLOBAL_DEVICE "CUSTATEVEC_EX_INDEX_BIT_DOMAIN_GLOBAL_DEVICE" = 2

ctypedef enum custatevecExWireInitMode_t "custatevecExWireInitMode_t":
    CUSTATEVEC_EX_WIRE_INIT_MODE_ZERO "CUSTATEVEC_EX_WIRE_INIT_MODE_ZERO" = 0

ctypedef enum custatevecExGlobalIndexBitClass_t "custatevecExGlobalIndexBitClass_t":
    CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P "CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P" = 1
    CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_COMMUNICATOR "CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_COMMUNICATOR" = 2
    CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_MIGRATION "CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_MIGRATION" = 3

ctypedef enum custatevecExStateVectorProperty_t "custatevecExStateVectorProperty_t":
    CUSTATEVEC_EX_SV_PROP_DISTRIBUTION_TYPE "CUSTATEVEC_EX_SV_PROP_DISTRIBUTION_TYPE" = 0
    CUSTATEVEC_EX_SV_PROP_DATA_TYPE "CUSTATEVEC_EX_SV_PROP_DATA_TYPE" = 1
    CUSTATEVEC_EX_SV_PROP_NUM_WIRES "CUSTATEVEC_EX_SV_PROP_NUM_WIRES" = 2
    CUSTATEVEC_EX_SV_PROP_WIRE_ORDERING "CUSTATEVEC_EX_SV_PROP_WIRE_ORDERING" = 3
    CUSTATEVEC_EX_SV_PROP_NUM_LOCAL_WIRES "CUSTATEVEC_EX_SV_PROP_NUM_LOCAL_WIRES" = 4
    CUSTATEVEC_EX_SV_PROP_NUM_DEVICE_SUBSVS "CUSTATEVEC_EX_SV_PROP_NUM_DEVICE_SUBSVS" = 5
    CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES "CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES" = 6
    CUSTATEVEC_EX_SV_PROP_NUM_INTERPROC_DEVICE_WIRES "CUSTATEVEC_EX_SV_PROP_NUM_INTERPROC_DEVICE_WIRES" = 7
    CUSTATEVEC_EX_SV_PROP_NUM_INPROC_DEVICE_WIRES "CUSTATEVEC_EX_SV_PROP_NUM_INPROC_DEVICE_WIRES" = 8
    CUSTATEVEC_EX_SV_PROP_NUM_MIGRATION_WIRES "CUSTATEVEC_EX_SV_PROP_NUM_MIGRATION_WIRES" = 9
    CUSTATEVEC_EX_SV_PROP_NUM_SUBSV_SLICES "CUSTATEVEC_EX_SV_PROP_NUM_SUBSV_SLICES" = 10
    CUSTATEVEC_EX_SV_PROP_NUM_SUBSVS "CUSTATEVEC_EX_SV_PROP_NUM_SUBSVS" = 11
    CUSTATEVEC_EX_SV_PROP_SUBSV_INDICES "CUSTATEVEC_EX_SV_PROP_SUBSV_INDICES" = 12
    CUSTATEVEC_EX_SV_PROP_NUM_UNSTAGED_SUBSVS "CUSTATEVEC_EX_SV_PROP_NUM_UNSTAGED_SUBSVS" = 13
    CUSTATEVEC_EX_SV_PROP_UNSTAGED_SUBSV_INDICES "CUSTATEVEC_EX_SV_PROP_UNSTAGED_SUBSV_INDICES" = 14
    CUSTATEVEC_EX_SV_PROP_MAX_NUM_WIRES "CUSTATEVEC_EX_SV_PROP_MAX_NUM_WIRES" = 15
    CUSTATEVEC_EX_SV_PROP_MAX_NUM_LOCAL_WIRES "CUSTATEVEC_EX_SV_PROP_MAX_NUM_LOCAL_WIRES" = 16
    CUSTATEVEC_EX_SV_PROP_MAX_NUM_INTERPROC_DEVICE_WIRES "CUSTATEVEC_EX_SV_PROP_MAX_NUM_INTERPROC_DEVICE_WIRES" = 17
    CUSTATEVEC_EX_SV_PROP_MAX_NUM_INPROC_DEVICE_WIRES "CUSTATEVEC_EX_SV_PROP_MAX_NUM_INPROC_DEVICE_WIRES" = 18
    CUSTATEVEC_EX_SV_PROP_MAX_NUM_MIGRATION_WIRES "CUSTATEVEC_EX_SV_PROP_MAX_NUM_MIGRATION_WIRES" = 19

ctypedef enum custatevecExPermutationType_t "custatevecExPermutationType_t":
    CUSTATEVEC_EX_PERMUTATION_SCATTER "CUSTATEVEC_EX_PERMUTATION_SCATTER" = 0
    CUSTATEVEC_EX_PERMUTATION_GATHER "CUSTATEVEC_EX_PERMUTATION_GATHER" = 1

ctypedef enum custatevecExExposeResources_t "custatevecExExposeResources_t":
    CUSTATEVEC_EX_EXPOSE_RESOURCES_ACCESSIBLE "CUSTATEVEC_EX_EXPOSE_RESOURCES_ACCESSIBLE" = 0

ctypedef enum custatevecExMatrixType_t "custatevecExMatrixType_t":
    CUSTATEVEC_EX_MATRIX_DENSE "CUSTATEVEC_EX_MATRIX_DENSE" = 1
    CUSTATEVEC_EX_MATRIX_DIAGONAL "CUSTATEVEC_EX_MATRIX_DIAGONAL" = 2
    CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL "CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL" = 4

ctypedef enum custatevecExSVUpdaterConfigName_t "custatevecExSVUpdaterConfigName_t":
    CUSTATEVEC_EX_SVUPDATER_CONFIG_MAX_NUM_HOST_THREADS "CUSTATEVEC_EX_SVUPDATER_CONFIG_MAX_NUM_HOST_THREADS" = 0
    CUSTATEVEC_EX_SVUPDATER_CONFIG_DENSE_FUSION_SIZE "CUSTATEVEC_EX_SVUPDATER_CONFIG_DENSE_FUSION_SIZE" = 1
    CUSTATEVEC_EX_SVUPDATER_CONFIG_DIAGONAL_FUSION_SIZE "CUSTATEVEC_EX_SVUPDATER_CONFIG_DIAGONAL_FUSION_SIZE" = 2

ctypedef enum custatevecExMemorySharingMethod_t "custatevecExMemorySharingMethod_t":
    CUSTATEVEC_EX_MEMORY_SHARING_METHOD_AUTODETECT "CUSTATEVEC_EX_MEMORY_SHARING_METHOD_AUTODETECT" = 0
    CUSTATEVEC_EX_MEMORY_SHARING_METHOD_NONE "CUSTATEVEC_EX_MEMORY_SHARING_METHOD_NONE" = 1
    CUSTATEVEC_EX_MEMORY_SHARING_METHOD_FABRIC_HANDLE "CUSTATEVEC_EX_MEMORY_SHARING_METHOD_FABRIC_HANDLE" = 2
    CUSTATEVEC_EX_MEMORY_SHARING_METHOD_PIDFD "CUSTATEVEC_EX_MEMORY_SHARING_METHOD_PIDFD" = 3

ctypedef enum custatevecExMemoryPlacement_t "custatevecExMemoryPlacement_t":
    CUSTATEVEC_EX_MEMORY_PLACEMENT_ON_HOST "CUSTATEVEC_EX_MEMORY_PLACEMENT_ON_HOST" = 0

ctypedef enum custatevecExSynchronizationScope_t "custatevecExSynchronizationScope_t":
    CUSTATEVEC_EX_SYNCHRONIZATION_SCOPE_PROCESS_LOCAL "CUSTATEVEC_EX_SYNCHRONIZATION_SCOPE_PROCESS_LOCAL" = 0
    CUSTATEVEC_EX_SYNCHRONIZATION_SCOPE_GLOBAL "CUSTATEVEC_EX_SYNCHRONIZATION_SCOPE_GLOBAL" = 1


cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>
    """
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef int cudaDataType_t 'cudaDataType_t'
    ctypedef int cudaDataType 'cudaDataType'
    ctypedef int libraryPropertyType_t 'libraryPropertyType_t'
    ctypedef int libraryPropertyType 'libraryPropertyType'


# types
ctypedef void* custatevecExDictionaryDescriptor_t 'custatevecExDictionaryDescriptor_t'

ctypedef void* custatevecExCommunicatorDescriptor_t 'custatevecExCommunicatorDescriptor_t'

ctypedef void* custatevecExStateVectorDescriptor_t 'custatevecExStateVectorDescriptor_t'

ctypedef void* custatevecExSVUpdaterDescriptor_t 'custatevecExSVUpdaterDescriptor_t'

ctypedef void* custatevecExResourceManagerDescriptor_t 'custatevecExResourceManagerDescriptor_t'

cdef extern from *:
    """
    typedef union {
        int32_t int32;
        char placeholder[32];
    } __attribute__((aligned(16))) custatevecExSVUpdaterConfigValue;
    """
    ctypedef union custatevecExSVUpdaterConfigValue:
        int32_t int32
        char placeholder[32]

ctypedef struct custatevecExSVUpdaterConfigItem_t 'custatevecExSVUpdaterConfigItem_t':
    custatevecExSVUpdaterConfigName_t name
    custatevecExSVUpdaterConfigValue value


###############################################################################
# Functions
###############################################################################

cdef custatevecStatus_t custatevecExDictionaryDestroy(custatevecExDictionaryDescriptor_t dictionary) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExCommunicatorInitialize(custatevecCommunicatorType_t communicatorType, const char* libraryPath, int* argc, char*** argv, custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExCommunicatorFinalize(custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExCommunicatorGetSizeAndRank(int32_t* size, int32_t* rank, custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExCommunicatorCreate(custatevecExCommunicatorDescriptor_t* exCommunicator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExCommunicatorDestroy(custatevecExCommunicatorDescriptor_t exCommunicator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExConfigureStateVectorSingleDevice(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, int32_t deviceId, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExConfigureStateVectorMultiDevice(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, const int32_t* deviceIds, int32_t numDevices, custatevecDeviceNetworkType_t networkType, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExConfigureStateVectorMultiProcess(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, int32_t deviceId, custatevecExMemorySharingMethod_t memorySharingMethod, const custatevecExGlobalIndexBitClass_t* globalIndexBitClasses, const int32_t* numGlobalIndexBitsPerLayer, int32_t numGlobalIndexBitLayers, size_t transferWorkspaceSizeInBytes, const void* auxConfig, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorCreateSingleProcess(custatevecExStateVectorDescriptor_t* stateVector, const custatevecExDictionaryDescriptor_t svConfig, const cudaStream_t* streams, int32_t numStreams, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorCreateMultiProcess(custatevecExStateVectorDescriptor_t* stateVector, const custatevecExDictionaryDescriptor_t svConfig, cudaStream_t stream, custatevecExCommunicatorDescriptor_t exCommunicator, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorDestroy(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorGetProperty(const custatevecExStateVectorDescriptor_t stateVector, custatevecExStateVectorProperty_t property, void* value, size_t sizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorSetMathMode(custatevecExStateVectorDescriptor_t stateVector, custatevecMathMode_t mode) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorSetZeroState(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorGetState(const custatevecExStateVectorDescriptor_t stateVector, void* state, cudaDataType_t dataType, custatevecIndex_t begin, custatevecIndex_t end, int32_t maxNumConcurrentCopies) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorSetState(custatevecExStateVectorDescriptor_t stateVector, const void* state, cudaDataType_t dataType, custatevecIndex_t begin, custatevecIndex_t end, int32_t maxNumConcurrentCopies) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorReassignWireOrdering(custatevecExStateVectorDescriptor_t stateVector, const int32_t* wireOrdering, int32_t wireOrderingLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorPermuteIndexBits(custatevecExStateVectorDescriptor_t stateVector, const int32_t* permutation, int32_t permutationLen, custatevecExPermutationType_t permutationType) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorStageSubSV(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorExposeResources(custatevecExStateVectorDescriptor_t stateVector, custatevecExExposeResources_t exposeResources) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorGetResourcesFromDeviceSubSV(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t* deviceId, void** d_subSV, cudaStream_t* stream, custatevecHandle_t* handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorGetResourcesFromDeviceSubSVView(const custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t* deviceId, const void** d_subSV, cudaStream_t* stream, custatevecHandle_t* handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t sliceIndex, void** subSVSlice, custatevecExMemoryPlacement_t* placement, int32_t* deviceId, cudaStream_t* stream) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView(const custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t sliceIndex, const void** subSVSlice, custatevecExMemoryPlacement_t* placement, int32_t* deviceId, cudaStream_t* stream) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorSynchronize(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorSynchronizeScoped(custatevecExStateVectorDescriptor_t stateVector, custatevecExSynchronizationScope_t scope) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExStateVectorAddWires(custatevecExStateVectorDescriptor_t stateVector, custatevecExIndexBitDomain_t indexBitDomain, int32_t numWiresToAdd, custatevecExWireInitMode_t wireInitMode, int32_t* wiresAdded) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExAbs2SumArray(custatevecExStateVectorDescriptor_t stateVector, double* abs2sum, const int32_t* outputOrdering, int32_t outputOrderingLen, const int32_t* maskBitString, const int32_t* maskWireOrdering, int32_t maskLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExMeasure(custatevecExStateVectorDescriptor_t stateVector, custatevecIndex_t* bitString, const int32_t* bitStringOrdering, int32_t bitStringOrderingLen, double randnum, custatevecCollapseOp_t collapse, const void* reserved) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExSample(custatevecExStateVectorDescriptor_t stateVector, custatevecIndex_t* bitStrings, const int32_t* bitStringOrdering, int32_t bitStringOrderingLen, const double* randnums, int32_t numShots, custatevecSamplerOutput_t output, const double* abs2Sums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExApplyMatrix(custatevecExStateVectorDescriptor_t stateVector, const void* matrix, cudaDataType_t matrixDataType, custatevecExMatrixType_t exMatrixType, custatevecMatrixLayout_t layout, int32_t adjoint, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExApplyPauliRotation(custatevecExStateVectorDescriptor_t stateVector, double theta, const custatevecPauli_t* paulis, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExComputeExpectationOnPauliBasis(custatevecExStateVectorDescriptor_t stateVector, double* expectationValues, const custatevecPauli_t** pauliOperatorArrays, int32_t numPauliOperatorArrays, const int32_t** basisWiresArray, const int32_t* numBasisWiresArray) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExComputeExpectation(custatevecExStateVectorDescriptor_t stateVector, double2* expectationValues, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, int32_t numMatrices, const int32_t* basisWires, int32_t numBasisWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExConfigureSVUpdater(custatevecExDictionaryDescriptor_t* svUpdaterConfig, cudaDataType_t dataType, const custatevecExSVUpdaterConfigItem_t* configItems, int32_t numConfigItems) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExSVUpdaterCreate(custatevecExSVUpdaterDescriptor_t* svUpdater, const custatevecExDictionaryDescriptor_t svUpdaterConfig, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExSVUpdaterDestroy(custatevecExSVUpdaterDescriptor_t svUpdater) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExSVUpdaterClear(custatevecExSVUpdaterDescriptor_t svUpdater) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExSVUpdaterEnqueueMatrix(custatevecExSVUpdaterDescriptor_t svUpdater, const void* matrix, cudaDataType_t matrixDataType, custatevecExMatrixType_t exMatrixType, custatevecMatrixLayout_t layout, int32_t adjoint, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExSVUpdaterEnqueueUnitaryChannel(custatevecExSVUpdaterDescriptor_t svUpdater, const void* const* unitaries, cudaDataType_t unitariesDataType, const custatevecExMatrixType_t* exMatrixTypes, int32_t numUnitaries, custatevecMatrixLayout_t layout, const double* probabilities, const int32_t* channelWires, int32_t numChannelWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExSVUpdaterEnqueueGeneralChannel(custatevecExSVUpdaterDescriptor_t svUpdater, const void* const* matrices, cudaDataType_t matrixDataType, const custatevecExMatrixType_t* exMatrixTypes, int32_t numMatrices, custatevecMatrixLayout_t layout, const int32_t* channelWires, int32_t numChannelWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExSVUpdaterGetMaxNumRequiredRandnums(custatevecExSVUpdaterDescriptor_t svUpdater, int32_t* maxNumRequiredRandnums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecExSVUpdaterApply(custatevecExSVUpdaterDescriptor_t svUpdater, custatevecExStateVectorDescriptor_t stateVector, const double* randnums, int32_t numRandnums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
