# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 25.11.0. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.stdio cimport FILE


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum custatevecStatus_t "custatevecStatus_t":
    CUSTATEVEC_STATUS_SUCCESS "CUSTATEVEC_STATUS_SUCCESS" = 0
    CUSTATEVEC_STATUS_NOT_INITIALIZED "CUSTATEVEC_STATUS_NOT_INITIALIZED" = 1
    CUSTATEVEC_STATUS_ALLOC_FAILED "CUSTATEVEC_STATUS_ALLOC_FAILED" = 2
    CUSTATEVEC_STATUS_INVALID_VALUE "CUSTATEVEC_STATUS_INVALID_VALUE" = 3
    CUSTATEVEC_STATUS_ARCH_MISMATCH "CUSTATEVEC_STATUS_ARCH_MISMATCH" = 4
    CUSTATEVEC_STATUS_EXECUTION_FAILED "CUSTATEVEC_STATUS_EXECUTION_FAILED" = 5
    CUSTATEVEC_STATUS_INTERNAL_ERROR "CUSTATEVEC_STATUS_INTERNAL_ERROR" = 6
    CUSTATEVEC_STATUS_NOT_SUPPORTED "CUSTATEVEC_STATUS_NOT_SUPPORTED" = 7
    CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE "CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE" = 8
    CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED "CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED" = 9
    CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR "CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR" = 10
    CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR "CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR" = 11
    CUSTATEVEC_STATUS_COMMUNICATOR_ERROR "CUSTATEVEC_STATUS_COMMUNICATOR_ERROR" = 12
    CUSTATEVEC_STATUS_LOADING_LIBRARY_FAILED "CUSTATEVEC_STATUS_LOADING_LIBRARY_FAILED" = 13
    CUSTATEVEC_STATUS_INVALID_CONFIGURATION "CUSTATEVEC_STATUS_INVALID_CONFIGURATION" = 14
    CUSTATEVEC_STATUS_ALREADY_INITIALIZED "CUSTATEVEC_STATUS_ALREADY_INITIALIZED" = 15
    CUSTATEVEC_STATUS_INVALID_WIRE "CUSTATEVEC_STATUS_INVALID_WIRE" = 16
    CUSTATEVEC_STATUS_SYSTEM_ERROR "CUSTATEVEC_STATUS_SYSTEM_ERROR" = 17
    CUSTATEVEC_STATUS_CUDA_ERROR "CUSTATEVEC_STATUS_CUDA_ERROR" = 18
    CUSTATEVEC_STATUS_NUMERICAL_ERROR "CUSTATEVEC_STATUS_NUMERICAL_ERROR" = 19
    CUSTATEVEC_STATUS_MAX_VALUE "CUSTATEVEC_STATUS_MAX_VALUE" = 20
    _CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR "_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum custatevecPauli_t "custatevecPauli_t":
    CUSTATEVEC_PAULI_I "CUSTATEVEC_PAULI_I" = 0
    CUSTATEVEC_PAULI_X "CUSTATEVEC_PAULI_X" = 1
    CUSTATEVEC_PAULI_Y "CUSTATEVEC_PAULI_Y" = 2
    CUSTATEVEC_PAULI_Z "CUSTATEVEC_PAULI_Z" = 3

ctypedef enum custatevecMatrixLayout_t "custatevecMatrixLayout_t":
    CUSTATEVEC_MATRIX_LAYOUT_COL "CUSTATEVEC_MATRIX_LAYOUT_COL" = 0
    CUSTATEVEC_MATRIX_LAYOUT_ROW "CUSTATEVEC_MATRIX_LAYOUT_ROW" = 1

ctypedef enum custatevecMatrixType_t "custatevecMatrixType_t":
    CUSTATEVEC_MATRIX_TYPE_GENERAL "CUSTATEVEC_MATRIX_TYPE_GENERAL" = 0
    CUSTATEVEC_MATRIX_TYPE_UNITARY "CUSTATEVEC_MATRIX_TYPE_UNITARY" = 1
    CUSTATEVEC_MATRIX_TYPE_HERMITIAN "CUSTATEVEC_MATRIX_TYPE_HERMITIAN" = 2

ctypedef enum custatevecCollapseOp_t "custatevecCollapseOp_t":
    CUSTATEVEC_COLLAPSE_NONE "CUSTATEVEC_COLLAPSE_NONE" = 0
    CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO "CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO" = 1
    CUSTATEVEC_COLLAPSE_RESET "CUSTATEVEC_COLLAPSE_RESET" = 2

ctypedef enum custatevecComputeType_t "custatevecComputeType_t":
    CUSTATEVEC_COMPUTE_DEFAULT "CUSTATEVEC_COMPUTE_DEFAULT" = 0
    CUSTATEVEC_COMPUTE_32F "CUSTATEVEC_COMPUTE_32F" = (1U << 2U)
    CUSTATEVEC_COMPUTE_64F "CUSTATEVEC_COMPUTE_64F" = (1U << 4U)
    CUSTATEVEC_COMPUTE_TF32 "CUSTATEVEC_COMPUTE_TF32" = (1U << 12U)

ctypedef enum custatevecSamplerOutput_t "custatevecSamplerOutput_t":
    CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER "CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER" = 0
    CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER "CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER" = 1

ctypedef enum custatevecDeviceNetworkType_t "custatevecDeviceNetworkType_t":
    CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH "CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH" = 1
    CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH "CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH" = 2

ctypedef enum custatevecCommunicatorType_t "custatevecCommunicatorType_t":
    CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL "CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL" = 0
    CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI "CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI" = 1
    CUSTATEVEC_COMMUNICATOR_TYPE_MPICH "CUSTATEVEC_COMMUNICATOR_TYPE_MPICH" = 2

ctypedef enum custatevecDataTransferType_t "custatevecDataTransferType_t":
    CUSTATEVEC_DATA_TRANSFER_TYPE_NONE "CUSTATEVEC_DATA_TRANSFER_TYPE_NONE" = 0
    CUSTATEVEC_DATA_TRANSFER_TYPE_SEND "CUSTATEVEC_DATA_TRANSFER_TYPE_SEND" = 1
    CUSTATEVEC_DATA_TRANSFER_TYPE_RECV "CUSTATEVEC_DATA_TRANSFER_TYPE_RECV" = 2
    CUSTATEVEC_DATA_TRANSFER_TYPE_SEND_RECV "CUSTATEVEC_DATA_TRANSFER_TYPE_SEND_RECV" = (CUSTATEVEC_DATA_TRANSFER_TYPE_SEND | CUSTATEVEC_DATA_TRANSFER_TYPE_RECV)

ctypedef enum custatevecMatrixMapType_t "custatevecMatrixMapType_t":
    CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST "CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST" = 0
    CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED "CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED" = 1

ctypedef enum custatevecStateVectorType_t "custatevecStateVectorType_t":
    CUSTATEVEC_STATE_VECTOR_TYPE_ZERO "CUSTATEVEC_STATE_VECTOR_TYPE_ZERO" = 0
    CUSTATEVEC_STATE_VECTOR_TYPE_UNIFORM "CUSTATEVEC_STATE_VECTOR_TYPE_UNIFORM" = 1
    CUSTATEVEC_STATE_VECTOR_TYPE_GHZ "CUSTATEVEC_STATE_VECTOR_TYPE_GHZ" = 2
    CUSTATEVEC_STATE_VECTOR_TYPE_W "CUSTATEVEC_STATE_VECTOR_TYPE_W" = 3

ctypedef enum custatevecMathMode_t "custatevecMathMode_t":
    CUSTATEVEC_MATH_MODE_DEFAULT "CUSTATEVEC_MATH_MODE_DEFAULT" = 0
    CUSTATEVEC_MATH_MODE_ALLOW_FP32_EMULATED_BF16X9 "CUSTATEVEC_MATH_MODE_ALLOW_FP32_EMULATED_BF16X9" = (1U << 0)
    CUSTATEVEC_MATH_MODE_DISALLOW_FP32_EMULATED_BF16X9 "CUSTATEVEC_MATH_MODE_DISALLOW_FP32_EMULATED_BF16X9" = (1U << 1)


cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>

    #define CUSTATEVEC_MAX_SEGMENT_MASK_SIZE 48
    #define CUSTATEVEC_ALLOCATOR_NAME_LEN 64
    """
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
    
    cdef const int CUSTATEVEC_MAX_SEGMENT_MASK_SIZE
    cdef const int CUSTATEVEC_ALLOCATOR_NAME_LEN


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


###############################################################################
# Functions
###############################################################################

cdef custatevecStatus_t custatevecCreate(custatevecHandle_t* handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecDestroy(custatevecHandle_t handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecGetDefaultWorkspaceSize(custatevecHandle_t handle, size_t* workspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSetWorkspace(custatevecHandle_t handle, void* workspace, size_t workspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef const char* custatevecGetErrorName(custatevecStatus_t status) except?NULL nogil
cdef const char* custatevecGetErrorString(custatevecStatus_t status) except?NULL nogil
cdef custatevecStatus_t custatevecGetProperty(libraryPropertyType type, int32_t* value) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef size_t custatevecGetVersion() except?0 nogil
cdef custatevecStatus_t custatevecSetStream(custatevecHandle_t handle, cudaStream_t streamId) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecGetStream(custatevecHandle_t handle, cudaStream_t* streamId) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecLoggerSetCallbackData(custatevecLoggerCallbackData_t callback, void* userData) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecLoggerOpenFile(const char* logFile) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecLoggerSetLevel(int32_t level) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecLoggerSetMask(int32_t mask) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecLoggerForceDisable() except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecGetDeviceMemHandler(custatevecHandle_t handle, custatevecDeviceMemHandler_t* handler) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSetDeviceMemHandler(custatevecHandle_t handle, const custatevecDeviceMemHandler_t* handler) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecAbs2SumOnZBasis(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* abs2sum0, double* abs2sum1, const int32_t* basisBits, const uint32_t nBasisBits) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecAbs2SumArray(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* abs2sum, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecCollapseOnZBasis(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int32_t parity, const int32_t* basisBits, const uint32_t nBasisBits, double norm) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecCollapseByBitString(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, double norm) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecMeasureOnZBasis(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* parity, const int32_t* basisBits, const uint32_t nBasisBits, const double randnum, custatevecCollapseOp_t collapse) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecBatchMeasure(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, const double randnum, custatevecCollapseOp_t collapse) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecBatchMeasureWithOffset(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, const double randnum, custatevecCollapseOp_t collapse, const double offset, const double abs2sum) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecApplyPauliRotation(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double theta, const custatevecPauli_t* paulis, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecApplyMatrixGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nTargets, const uint32_t nControls, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecApplyMatrix(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecComputeExpectationGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nBasisBits, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecComputeExpectation(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, void* expectationValue, cudaDataType_t expectationDataType, double* residualNorm, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t* basisBits, const uint32_t nBasisBits, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSamplerCreate(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecSamplerDescriptor_t* sampler, uint32_t nMaxShots, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSamplerDestroy(custatevecSamplerDescriptor_t sampler) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSamplerPreprocess(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, void* extraWorkspace, const size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSamplerGetSquaredNorm(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, double* norm) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSamplerApplySubSVOffset(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, int32_t subSVOrd, uint32_t nSubSVs, double offset, double norm) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSamplerSample(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* randnums, const uint32_t nShots, custatevecSamplerOutput_t output) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType, const int32_t* targets, const uint32_t nTargets, const uint32_t nControls, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecApplyGeneralizedPermutationMatrix(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType, const int32_t adjoint, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecComputeExpectationsOnPauliBasis(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* expectationValues, const custatevecPauli_t** pauliOperatorsArray, const uint32_t nPauliOperatorArrays, const int32_t** basisBitsArray, const uint32_t* nBasisBitsArray) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecAccessorCreate(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecAccessorDescriptor_t* accessor, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecAccessorCreateView(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecAccessorDescriptor_t* accessor, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecAccessorDestroy(custatevecAccessorDescriptor_t accessor) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecAccessorSetExtraWorkspace(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecAccessorGet(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, void* externalBuffer, const custatevecIndex_t begin, const custatevecIndex_t end) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecAccessorSet(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, const void* externalBuffer, const custatevecIndex_t begin, const custatevecIndex_t end) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSwapIndexBits(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int2* bitSwaps, const uint32_t nBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecTestMatrixTypeGetWorkspaceSize(custatevecHandle_t handle, custatevecMatrixType_t matrixType, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nTargets, const int32_t adjoint, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecTestMatrixType(custatevecHandle_t handle, double* residualNorm, custatevecMatrixType_t matrixType, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nTargets, const int32_t adjoint, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecMultiDeviceSwapIndexBits(custatevecHandle_t* handles, const uint32_t nHandles, void** subSVs, const cudaDataType_t svDataType, const uint32_t nGlobalIndexBits, const uint32_t nLocalIndexBits, const int2* indexBitSwaps, const uint32_t nIndexBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, const custatevecDeviceNetworkType_t deviceNetworkType) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecCommunicatorCreate(custatevecHandle_t handle, custatevecCommunicatorDescriptor_t* communicator, custatevecCommunicatorType_t communicatorType, const char* soname) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecCommunicatorDestroy(custatevecHandle_t handle, custatevecCommunicatorDescriptor_t communicator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerCreate(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t* scheduler, const uint32_t nGlobalIndexBits, const uint32_t nLocalIndexBits) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerDestroy(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler, const int2* indexBitSwaps, const uint32_t nIndexBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, uint32_t* nSwapBatches) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecDistIndexBitSwapSchedulerGetParameters(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler, const int32_t swapBatchIndex, const int32_t orgSubSVIndex, custatevecSVSwapParameters_t* parameters) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSVSwapWorkerCreate(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t* svSwapWorker, custatevecCommunicatorDescriptor_t communicator, void* orgSubSV, int32_t orgSubSVIndex, cudaEvent_t orgEvent, cudaDataType_t svDataType, cudaStream_t stream, size_t* extraWorkspaceSizeInBytes, size_t* minTransferWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSVSwapWorkerDestroy(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSVSwapWorkerSetExtraWorkspace(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSVSwapWorkerSetTransferWorkspace(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void* transferWorkspace, size_t transferWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSVSwapWorkerSetSubSVsP2P(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void** dstSubSVsP2P, const int32_t* dstSubSVIndicesP2P, cudaEvent_t* dstEvents, const uint32_t nDstSubSVsP2P) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSVSwapWorkerSetParameters(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, const custatevecSVSwapParameters_t* parameters, int peer) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSVSwapWorkerExecute(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, custatevecIndex_t begin, custatevecIndex_t end) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecInitializeStateVector(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecStateVectorType_t svType) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecApplyMatrixBatchedGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, custatevecMatrixMapType_t mapType, const int32_t* matrixIndices, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nMatrices, const uint32_t nTargets, const uint32_t nControls, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecApplyMatrixBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, custatevecIndex_t svStride, custatevecMatrixMapType_t mapType, const int32_t* matrixIndices, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nMatrices, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecAbs2SumArrayBatched(custatevecHandle_t handle, const void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, double* abs2sumArrays, const custatevecIndex_t abs2sumArrayStride, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const custatevecIndex_t* maskBitStrings, const int32_t* maskOrdering, const uint32_t maskLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecCollapseByBitStringBatchedGetWorkspaceSize(custatevecHandle_t handle, const uint32_t nSVs, const custatevecIndex_t* bitStrings, const double* norms, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecCollapseByBitStringBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, const custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* norms, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecMeasureBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* randnums, custatevecCollapseOp_t collapse) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSubSVMigratorCreate(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t* migrator, void* deviceSlots, cudaDataType_t svDataType, int nDeviceSlots, int nLocalIndexBits) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSubSVMigratorDestroy(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t migrator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSubSVMigratorMigrate(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t migrator, int deviceSlotIndex, const void* srcSubSV, void* dstSubSV, custatevecIndex_t begin, custatevecIndex_t end) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecComputeExpectationBatchedGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nMatrices, const uint32_t nBasisBits, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecComputeExpectationBatched(custatevecHandle_t handle, const void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, custatevecIndex_t svStride, double2* expectationValues, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nMatrices, const int32_t* basisBits, const uint32_t nBasisBits, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecSetMathMode(custatevecHandle_t handle, custatevecMathMode_t mode) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custatevecStatus_t custatevecGetMathMode(custatevecHandle_t handle, custatevecMathMode_t* mode) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil
