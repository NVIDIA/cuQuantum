# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 25.03.0. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.stdio cimport FILE


###############################################################################
# Types (structs, enums, ...)
###############################################################################

cdef extern from *:
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef void* cudaEvent_t 'cudaEvent_t'
    ctypedef int cudaDataType_t 'cudaDataType_t'
    ctypedef int cudaDataType 'cudaDataType'
    ctypedef int libraryPropertyType_t 'libraryPropertyType_t'
    ctypedef int libraryPropertyType 'libraryPropertyType'
    ctypedef struct int2 'int2':
        pass
    ctypedef struct double2 'double2':
        pass


cdef extern from '<custatevec.h>' nogil:
    # enums
    ctypedef enum custatevecStatus_t:
        CUSTATEVEC_STATUS_SUCCESS
        CUSTATEVEC_STATUS_NOT_INITIALIZED
        CUSTATEVEC_STATUS_ALLOC_FAILED
        CUSTATEVEC_STATUS_INVALID_VALUE
        CUSTATEVEC_STATUS_ARCH_MISMATCH
        CUSTATEVEC_STATUS_EXECUTION_FAILED
        CUSTATEVEC_STATUS_INTERNAL_ERROR
        CUSTATEVEC_STATUS_NOT_SUPPORTED
        CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE
        CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED
        CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR
        CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR
        CUSTATEVEC_STATUS_COMMUNICATOR_ERROR
        CUSTATEVEC_STATUS_LOADING_LIBRARY_FAILED
        CUSTATEVEC_STATUS_MAX_VALUE

    ctypedef enum custatevecPauli_t:
        CUSTATEVEC_PAULI_I
        CUSTATEVEC_PAULI_X
        CUSTATEVEC_PAULI_Y
        CUSTATEVEC_PAULI_Z

    ctypedef enum custatevecMatrixLayout_t:
        CUSTATEVEC_MATRIX_LAYOUT_COL
        CUSTATEVEC_MATRIX_LAYOUT_ROW

    ctypedef enum custatevecMatrixType_t:
        CUSTATEVEC_MATRIX_TYPE_GENERAL
        CUSTATEVEC_MATRIX_TYPE_UNITARY
        CUSTATEVEC_MATRIX_TYPE_HERMITIAN

    ctypedef enum custatevecCollapseOp_t:
        CUSTATEVEC_COLLAPSE_NONE
        CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO

    ctypedef enum custatevecComputeType_t:
        CUSTATEVEC_COMPUTE_DEFAULT
        CUSTATEVEC_COMPUTE_32F
        CUSTATEVEC_COMPUTE_64F
        CUSTATEVEC_COMPUTE_TF32

    ctypedef enum custatevecSamplerOutput_t:
        CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER
        CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER

    ctypedef enum custatevecDeviceNetworkType_t:
        CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH
        CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH

    ctypedef enum custatevecCommunicatorType_t:
        CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL
        CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI
        CUSTATEVEC_COMMUNICATOR_TYPE_MPICH

    ctypedef enum custatevecDataTransferType_t:
        CUSTATEVEC_DATA_TRANSFER_TYPE_NONE
        CUSTATEVEC_DATA_TRANSFER_TYPE_SEND
        CUSTATEVEC_DATA_TRANSFER_TYPE_RECV
        CUSTATEVEC_DATA_TRANSFER_TYPE_SEND_RECV

    ctypedef enum custatevecMatrixMapType_t:
        CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST
        CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED

    ctypedef enum custatevecStateVectorType_t:
        CUSTATEVEC_STATE_VECTOR_TYPE_ZERO
        CUSTATEVEC_STATE_VECTOR_TYPE_UNIFORM
        CUSTATEVEC_STATE_VECTOR_TYPE_GHZ
        CUSTATEVEC_STATE_VECTOR_TYPE_W

    ctypedef enum custatevecMathMode_t:
        CUSTATEVEC_MATH_MODE_DEFAULT
        CUSTATEVEC_MATH_MODE_ALLOW_FP32_EMULATED_BF16X9
        CUSTATEVEC_MATH_MODE_DISALLOW_FP32_EMULATED_BF16X9

    # types
    ctypedef int64_t custatevecIndex_t 'custatevecIndex_t'
    ctypedef void* custatevecHandle_t 'custatevecHandle_t'
    ctypedef void* custatevecSamplerDescriptor_t 'custatevecSamplerDescriptor_t'
    ctypedef void* custatevecAccessorDescriptor_t 'custatevecAccessorDescriptor_t'
    ctypedef void* custatevecCommunicatorDescriptor_t 'custatevecCommunicatorDescriptor_t'
    ctypedef void* custatevecDistIndexBitSwapSchedulerDescriptor_t 'custatevecDistIndexBitSwapSchedulerDescriptor_t'
    ctypedef void* custatevecSVSwapWorkerDescriptor_t 'custatevecSVSwapWorkerDescriptor_t'
    ctypedef void* custatevecSubSVMigratorDescriptor_t 'custatevecSubSVMigratorDescriptor_t'
    ctypedef struct custatevecDeviceMemHandler_t 'custatevecDeviceMemHandler_t':
        void* ctx
        int (*device_alloc)(void*, void**, size_t, cudaStream_t)
        int (*device_free)(void*, void*, size_t, cudaStream_t)
        char name[64]
    ctypedef struct custatevecSVSwapParameters_t 'custatevecSVSwapParameters_t':
        int32_t swapBatchIndex
        int32_t orgSubSVIndex
        int32_t dstSubSVIndex
        int32_t orgSegmentMaskString[48]
        int32_t dstSegmentMaskString[48]
        int32_t segmentMaskOrdering[48]
        uint32_t segmentMaskLen
        uint32_t nSegmentBits
        custatevecDataTransferType_t dataTransferType
        custatevecIndex_t transferSize
    ctypedef void (*custatevecLoggerCallback_t 'custatevecLoggerCallback_t')(
        int32_t logLevel,
        const char* functionName,
        const char* message
    )
    ctypedef void (*custatevecLoggerCallbackData_t 'custatevecLoggerCallbackData_t')(
        int32_t logLevel,
        const char* functionName,
        const char* message,
        void* userData
    )

    # constants
    const int CUSTATEVEC_MAX_SEGMENT_MASK_SIZE
    const int CUSTATEVEC_ALLOCATOR_NAME_LEN
    const int CUSTATEVEC_VER_MAJOR
    const int CUSTATEVEC_VER_MINOR
    const int CUSTATEVEC_VER_PATCH
    const int CUSTATEVEC_VERSION


###############################################################################
# Functions
###############################################################################

cdef custatevecStatus_t custatevecCreate(custatevecHandle_t* handle) except* nogil
cdef custatevecStatus_t custatevecDestroy(custatevecHandle_t handle) except* nogil
cdef custatevecStatus_t custatevecGetDefaultWorkspaceSize(custatevecHandle_t handle, size_t* workspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecSetWorkspace(custatevecHandle_t handle, void* workspace, size_t workspaceSizeInBytes) except* nogil
cdef const char* custatevecGetErrorName(custatevecStatus_t status) except* nogil
cdef const char* custatevecGetErrorString(custatevecStatus_t status) except* nogil
cdef custatevecStatus_t custatevecGetProperty(libraryPropertyType type, int32_t* value) except* nogil
cdef size_t custatevecGetVersion() except* nogil
cdef custatevecStatus_t custatevecSetStream(custatevecHandle_t handle, cudaStream_t streamId) except* nogil
cdef custatevecStatus_t custatevecGetStream(custatevecHandle_t handle, cudaStream_t* streamId) except* nogil
cdef custatevecStatus_t custatevecLoggerSetCallbackData(custatevecLoggerCallbackData_t callback, void* userData) except* nogil
cdef custatevecStatus_t custatevecLoggerOpenFile(const char* logFile) except* nogil
cdef custatevecStatus_t custatevecLoggerSetLevel(int32_t level) except* nogil
cdef custatevecStatus_t custatevecLoggerSetMask(int32_t mask) except* nogil
cdef custatevecStatus_t custatevecLoggerForceDisable() except* nogil
cdef custatevecStatus_t custatevecGetDeviceMemHandler(custatevecHandle_t handle, custatevecDeviceMemHandler_t* handler) except* nogil
cdef custatevecStatus_t custatevecSetDeviceMemHandler(custatevecHandle_t handle, const custatevecDeviceMemHandler_t* handler) except* nogil
cdef custatevecStatus_t custatevecAbs2SumOnZBasis(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* abs2sum0, double* abs2sum1, const int32_t* basisBits, const uint32_t nBasisBits) except* nogil
cdef custatevecStatus_t custatevecAbs2SumArray(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* abs2sum, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen) except* nogil
cdef custatevecStatus_t custatevecCollapseOnZBasis(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int32_t parity, const int32_t* basisBits, const uint32_t nBasisBits, double norm) except* nogil
cdef custatevecStatus_t custatevecCollapseByBitString(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, double norm) except* nogil
cdef custatevecStatus_t custatevecMeasureOnZBasis(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* parity, const int32_t* basisBits, const uint32_t nBasisBits, const double randnum, custatevecCollapseOp_t collapse) except* nogil
cdef custatevecStatus_t custatevecBatchMeasure(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, const double randnum, custatevecCollapseOp_t collapse) except* nogil
cdef custatevecStatus_t custatevecBatchMeasureWithOffset(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, const double randnum, custatevecCollapseOp_t collapse, const double offset, const double abs2sum) except* nogil
cdef custatevecStatus_t custatevecApplyPauliRotation(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double theta, const custatevecPauli_t* paulis, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls) except* nogil
cdef custatevecStatus_t custatevecApplyMatrixGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nTargets, const uint32_t nControls, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecApplyMatrix(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecComputeExpectationGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nBasisBits, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecComputeExpectation(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, void* expectationValue, cudaDataType_t expectationDataType, double* residualNorm, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t* basisBits, const uint32_t nBasisBits, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecSamplerCreate(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecSamplerDescriptor_t* sampler, uint32_t nMaxShots, size_t* extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecSamplerDestroy(custatevecSamplerDescriptor_t sampler) except* nogil
cdef custatevecStatus_t custatevecSamplerPreprocess(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, void* extraWorkspace, const size_t extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecSamplerGetSquaredNorm(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, double* norm) except* nogil
cdef custatevecStatus_t custatevecSamplerApplySubSVOffset(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, int32_t subSVOrd, uint32_t nSubSVs, double offset, double norm) except* nogil
cdef custatevecStatus_t custatevecSamplerSample(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* randnums, const uint32_t nShots, custatevecSamplerOutput_t output) except* nogil
cdef custatevecStatus_t custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType, const int32_t* targets, const uint32_t nTargets, const uint32_t nControls, size_t* extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecApplyGeneralizedPermutationMatrix(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType, const int32_t adjoint, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecComputeExpectationsOnPauliBasis(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* expectationValues, const custatevecPauli_t** pauliOperatorsArray, const uint32_t nPauliOperatorArrays, const int32_t** basisBitsArray, const uint32_t* nBasisBitsArray) except* nogil
cdef custatevecStatus_t custatevecAccessorCreate(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecAccessorDescriptor_t* accessor, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, size_t* extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecAccessorCreateView(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecAccessorDescriptor_t* accessor, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, size_t* extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecAccessorDestroy(custatevecAccessorDescriptor_t accessor) except* nogil
cdef custatevecStatus_t custatevecAccessorSetExtraWorkspace(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecAccessorGet(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, void* externalBuffer, const custatevecIndex_t begin, const custatevecIndex_t end) except* nogil
cdef custatevecStatus_t custatevecAccessorSet(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, const void* externalBuffer, const custatevecIndex_t begin, const custatevecIndex_t end) except* nogil
cdef custatevecStatus_t custatevecSwapIndexBits(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int2* bitSwaps, const uint32_t nBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen) except* nogil
cdef custatevecStatus_t custatevecTestMatrixTypeGetWorkspaceSize(custatevecHandle_t handle, custatevecMatrixType_t matrixType, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nTargets, const int32_t adjoint, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecTestMatrixType(custatevecHandle_t handle, double* residualNorm, custatevecMatrixType_t matrixType, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nTargets, const int32_t adjoint, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecMultiDeviceSwapIndexBits(custatevecHandle_t* handles, const uint32_t nHandles, void** subSVs, const cudaDataType_t svDataType, const uint32_t nGlobalIndexBits, const uint32_t nLocalIndexBits, const int2* indexBitSwaps, const uint32_t nIndexBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, const custatevecDeviceNetworkType_t deviceNetworkType) except* nogil
cdef custatevecStatus_t custatevecCommunicatorCreate(custatevecHandle_t handle, custatevecCommunicatorDescriptor_t* communicator, custatevecCommunicatorType_t communicatorType, const char* soname) except* nogil
cdef custatevecStatus_t custatevecCommunicatorDestroy(custatevecHandle_t handle, custatevecCommunicatorDescriptor_t communicator) except* nogil
cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerCreate(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t* scheduler, const uint32_t nGlobalIndexBits, const uint32_t nLocalIndexBits) except* nogil
cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerDestroy(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler) except* nogil
cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler, const int2* indexBitSwaps, const uint32_t nIndexBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, uint32_t* nSwapBatches) except* nogil
cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerGetParameters(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler, const int32_t swapBatchIndex, const int32_t orgSubSVIndex, custatevecSVSwapParameters_t* parameters) except* nogil
cdef custatevecStatus_t custatevecSVSwapWorkerCreate(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t* svSwapWorker, custatevecCommunicatorDescriptor_t communicator, void* orgSubSV, int32_t orgSubSVIndex, cudaEvent_t orgEvent, cudaDataType_t svDataType, cudaStream_t stream, size_t* extraWorkspaceSizeInBytes, size_t* minTransferWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecSVSwapWorkerDestroy(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker) except* nogil
cdef custatevecStatus_t custatevecSVSwapWorkerSetExtraWorkspace(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecSVSwapWorkerSetTransferWorkspace(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void* transferWorkspace, size_t transferWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecSVSwapWorkerSetSubSVsP2P(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void** dstSubSVsP2P, const int32_t* dstSubSVIndicesP2P, cudaEvent_t* dstEvents, const uint32_t nDstSubSVsP2P) except* nogil
cdef custatevecStatus_t custatevecSVSwapWorkerSetParameters(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, const custatevecSVSwapParameters_t* parameters, int peer) except* nogil
cdef custatevecStatus_t custatevecSVSwapWorkerExecute(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, custatevecIndex_t begin, custatevecIndex_t end) except* nogil
cdef custatevecStatus_t custatevecInitializeStateVector(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecStateVectorType_t svType) except* nogil
cdef custatevecStatus_t custatevecApplyMatrixBatchedGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, custatevecMatrixMapType_t mapType, const int32_t* matrixIndices, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nMatrices, const uint32_t nTargets, const uint32_t nControls, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecApplyMatrixBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, custatevecIndex_t svStride, custatevecMatrixMapType_t mapType, const int32_t* matrixIndices, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nMatrices, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecAbs2SumArrayBatched(custatevecHandle_t handle, const void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, double* abs2sumArrays, const custatevecIndex_t abs2sumArrayStride, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const custatevecIndex_t* maskBitStrings, const int32_t* maskOrdering, const uint32_t maskLen) except* nogil
cdef custatevecStatus_t custatevecCollapseByBitStringBatchedGetWorkspaceSize(custatevecHandle_t handle, const uint32_t nSVs, const custatevecIndex_t* bitStrings, const double* norms, size_t* extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecCollapseByBitStringBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, const custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* norms, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecMeasureBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* randnums, custatevecCollapseOp_t collapse) except* nogil
cdef custatevecStatus_t custatevecSubSVMigratorCreate(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t* migrator, void* deviceSlots, cudaDataType_t svDataType, int nDeviceSlots, int nLocalIndexBits) except* nogil
cdef custatevecStatus_t custatevecSubSVMigratorDestroy(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t migrator) except* nogil
cdef custatevecStatus_t custatevecSubSVMigratorMigrate(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t migrator, int deviceSlotIndex, const void* srcSubSV, void* dstSubSV, custatevecIndex_t begin, custatevecIndex_t end) except* nogil
cdef custatevecStatus_t custatevecComputeExpectationBatchedGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nMatrices, const uint32_t nBasisBits, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecComputeExpectationBatched(custatevecHandle_t handle, const void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, custatevecIndex_t svStride, double2* expectationValues, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nMatrices, const int32_t* basisBits, const uint32_t nBasisBits, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil
cdef custatevecStatus_t custatevecSetMathMode(custatevecHandle_t handle, custatevecMathMode_t mode) except* nogil
cdef custatevecStatus_t custatevecGetMathMode(custatevecHandle_t handle, custatevecMathMode_t* mode) except* nogil
