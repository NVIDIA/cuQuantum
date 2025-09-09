# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.09.0. Do not modify it directly.

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.stdio cimport FILE


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum cudensitymatStatus_t "cudensitymatStatus_t":
    CUDENSITYMAT_STATUS_SUCCESS "CUDENSITYMAT_STATUS_SUCCESS" = 0
    CUDENSITYMAT_STATUS_NOT_INITIALIZED "CUDENSITYMAT_STATUS_NOT_INITIALIZED" = 1
    CUDENSITYMAT_STATUS_ALLOC_FAILED "CUDENSITYMAT_STATUS_ALLOC_FAILED" = 3
    CUDENSITYMAT_STATUS_INVALID_VALUE "CUDENSITYMAT_STATUS_INVALID_VALUE" = 7
    CUDENSITYMAT_STATUS_ARCH_MISMATCH "CUDENSITYMAT_STATUS_ARCH_MISMATCH" = 8
    CUDENSITYMAT_STATUS_EXECUTION_FAILED "CUDENSITYMAT_STATUS_EXECUTION_FAILED" = 13
    CUDENSITYMAT_STATUS_INTERNAL_ERROR "CUDENSITYMAT_STATUS_INTERNAL_ERROR" = 14
    CUDENSITYMAT_STATUS_NOT_SUPPORTED "CUDENSITYMAT_STATUS_NOT_SUPPORTED" = 15
    CUDENSITYMAT_STATUS_CALLBACK_ERROR "CUDENSITYMAT_STATUS_CALLBACK_ERROR" = 16
    CUDENSITYMAT_STATUS_CUBLAS_ERROR "CUDENSITYMAT_STATUS_CUBLAS_ERROR" = 17
    CUDENSITYMAT_STATUS_CUDA_ERROR "CUDENSITYMAT_STATUS_CUDA_ERROR" = 18
    CUDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE "CUDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE" = 19
    CUDENSITYMAT_STATUS_INSUFFICIENT_DRIVER "CUDENSITYMAT_STATUS_INSUFFICIENT_DRIVER" = 20
    CUDENSITYMAT_STATUS_IO_ERROR "CUDENSITYMAT_STATUS_IO_ERROR" = 21
    CUDENSITYMAT_STATUS_CUTENSOR_VERSION_MISMATCH "CUDENSITYMAT_STATUS_CUTENSOR_VERSION_MISMATCH" = 22
    CUDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR "CUDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR" = 23
    CUDENSITYMAT_STATUS_CUTENSOR_ERROR "CUDENSITYMAT_STATUS_CUTENSOR_ERROR" = 24
    CUDENSITYMAT_STATUS_CUSOLVER_ERROR "CUDENSITYMAT_STATUS_CUSOLVER_ERROR" = 25
    CUDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR "CUDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR" = 26
    CUDENSITYMAT_STATUS_DISTRIBUTED_FAILURE "CUDENSITYMAT_STATUS_DISTRIBUTED_FAILURE" = 27
    CUDENSITYMAT_STATUS_INTERRUPTED "CUDENSITYMAT_STATUS_INTERRUPTED" = 28
    CUDENSITYMAT_STATUS_CUTENSORNET_ERROR "CUDENSITYMAT_STATUS_CUTENSORNET_ERROR" = 29
    _CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR "_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cudensitymatComputeType_t "cudensitymatComputeType_t":
    CUDENSITYMAT_COMPUTE_32F "CUDENSITYMAT_COMPUTE_32F" = (1U << 2U)
    CUDENSITYMAT_COMPUTE_64F "CUDENSITYMAT_COMPUTE_64F" = (1U << 4U)

ctypedef enum cudensitymatDistributedProvider_t "cudensitymatDistributedProvider_t":
    CUDENSITYMAT_DISTRIBUTED_PROVIDER_NONE "CUDENSITYMAT_DISTRIBUTED_PROVIDER_NONE" = 0
    CUDENSITYMAT_DISTRIBUTED_PROVIDER_MPI "CUDENSITYMAT_DISTRIBUTED_PROVIDER_MPI" = 1

ctypedef enum cudensitymatCallbackDevice_t "cudensitymatCallbackDevice_t":
    CUDENSITYMAT_CALLBACK_DEVICE_CPU "CUDENSITYMAT_CALLBACK_DEVICE_CPU"
    CUDENSITYMAT_CALLBACK_DEVICE_GPU "CUDENSITYMAT_CALLBACK_DEVICE_GPU"

ctypedef enum cudensitymatDifferentiationDir_t "cudensitymatDifferentiationDir_t":
    CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD "CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD" = 1

ctypedef enum cudensitymatStatePurity_t "cudensitymatStatePurity_t":
    CUDENSITYMAT_STATE_PURITY_PURE "CUDENSITYMAT_STATE_PURITY_PURE"
    CUDENSITYMAT_STATE_PURITY_MIXED "CUDENSITYMAT_STATE_PURITY_MIXED"

ctypedef enum cudensitymatElementaryOperatorSparsity_t "cudensitymatElementaryOperatorSparsity_t":
    CUDENSITYMAT_OPERATOR_SPARSITY_NONE "CUDENSITYMAT_OPERATOR_SPARSITY_NONE" = 0
    CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL "CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL" = 1

ctypedef enum cudensitymatOperatorSpectrumKind_t "cudensitymatOperatorSpectrumKind_t":
    CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST "CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST" = 0
    CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST "CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST" = 1
    CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL "CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL" = 2
    CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST_REAL "CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST_REAL" = 3

ctypedef enum cudensitymatOperatorSpectrumConfig_t "cudensitymatOperatorSpectrumConfig_t":
    CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_EXPANSION "CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_EXPANSION" = 0
    CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_RESTARTS "CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_RESTARTS" = 1
    CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MIN_BLOCK_SIZE "CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MIN_BLOCK_SIZE" = 2

ctypedef enum cudensitymatMemspace_t "cudensitymatMemspace_t":
    CUDENSITYMAT_MEMSPACE_DEVICE "CUDENSITYMAT_MEMSPACE_DEVICE" = 0
    CUDENSITYMAT_MEMSPACE_HOST "CUDENSITYMAT_MEMSPACE_HOST" = 1

ctypedef enum cudensitymatWorkspaceKind_t "cudensitymatWorkspaceKind_t":
    CUDENSITYMAT_WORKSPACE_SCRATCH "CUDENSITYMAT_WORKSPACE_SCRATCH" = 0

cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>

    #define CUDENSITYMAT_ALLOCATOR_NAME_LEN 64
    """
    ctypedef void* cudaStream_t 'cudaStream_t'

    ctypedef enum cudaDataType_t:
        CUDA_R_32F
        CUDA_C_32F
        CUDA_R_64F
        CUDA_C_64F
    ctypedef cudaDataType_t cudaDataType 'cudaDataType'

    ctypedef int libraryPropertyType_t 'libraryPropertyType_t'
    ctypedef int libraryPropertyType 'libraryPropertyType'

    ctypedef struct cuDoubleComplex:
        double x
        double y

    # constants
    const int CUDENSITYMAT_ALLOCATOR_NAME_LEN

# types
ctypedef void* cudensitymatHandle_t 'cudensitymatHandle_t'
ctypedef void* cudensitymatState_t 'cudensitymatState_t'
ctypedef void* cudensitymatElementaryOperator_t 'cudensitymatElementaryOperator_t'
ctypedef void* cudensitymatMatrixOperator_t 'cudensitymatMatrixOperator_t'
ctypedef void* cudensitymatOperatorTerm_t 'cudensitymatOperatorTerm_t'
ctypedef void* cudensitymatOperator_t 'cudensitymatOperator_t'
ctypedef void* cudensitymatOperatorAction_t 'cudensitymatOperatorAction_t'
ctypedef void* cudensitymatExpectation_t 'cudensitymatExpectation_t'
ctypedef void* cudensitymatOperatorSpectrum_t 'cudensitymatOperatorSpectrum_t'
ctypedef void* cudensitymatWorkspaceDescriptor_t 'cudensitymatWorkspaceDescriptor_t'
ctypedef void* cudensitymatDistributedRequest_t 'cudensitymatDistributedRequest_t'
ctypedef struct cudensitymatTimeRange_t 'cudensitymatTimeRange_t':
    double timeStart
    double timeFinish
    double timeStep
    int64_t numPoints
    double* points
ctypedef struct cudensitymatDistributedCommunicator_t 'cudensitymatDistributedCommunicator_t':
    void* commPtr
    size_t commSize
ctypedef int32_t (*cudensitymatScalarCallback_t 'cudensitymatScalarCallback_t')(
    double time,
    int64_t batchSize,
    int32_t numParams,
    const double* params,
    cudaDataType_t dataType,
    void* scalarStorage,
    cudaStream_t stream
)
ctypedef int32_t (*cudensitymatTensorCallback_t 'cudensitymatTensorCallback_t')(
    cudensitymatElementaryOperatorSparsity_t sparsity,
    int32_t numModes,
    const int64_t modeExtents[],
    const int32_t diagonalOffsets[],
    double time,
    int64_t batchSize,
    int32_t numParams,
    const double* params,
    cudaDataType_t dataType,
    void* tensorStorage,
    cudaStream_t stream
)
ctypedef int32_t (*cudensitymatScalarGradientCallback_t 'cudensitymatScalarGradientCallback_t')(
    double time,
    int64_t batchSize,
    int32_t numParams,
    const double* params,
    cudaDataType_t dataType,
    void* scalarGrad,
    double* paramsGrad,
    cudaStream_t stream
)
ctypedef int32_t (*cudensitymatTensorGradientCallback_t 'cudensitymatTensorGradientCallback_t')(
    cudensitymatElementaryOperatorSparsity_t sparsity,
    int32_t numModes,
    const int64_t modeExtents[],
    const int32_t diagonalOffsets[],
    double time,
    int64_t batchSize,
    int32_t numParams,
    const double* params,
    cudaDataType_t dataType,
    void* tensorGrad,
    double* paramsGrad,
    cudaStream_t stream
)
ctypedef struct cudensitymatDistributedInterface_t 'cudensitymatDistributedInterface_t':
    int version
    int (*getNumRanks)(const cudensitymatDistributedCommunicator_t*, int32_t*)
    int (*getNumRanksShared)(const cudensitymatDistributedCommunicator_t*, int32_t*)
    int (*getProcRank)(const cudensitymatDistributedCommunicator_t*, int32_t*)
    int (*barrier)(const cudensitymatDistributedCommunicator_t*)
    int (*createRequest)(cudensitymatDistributedRequest_t*)
    int (*destroyRequest)(cudensitymatDistributedRequest_t)
    int (*waitRequest)(cudensitymatDistributedRequest_t)
    int (*testRequest)(cudensitymatDistributedRequest_t, int32_t*)
    int (*send)(const cudensitymatDistributedCommunicator_t*, const void*, int32_t, cudaDataType_t, int32_t, int32_t)
    int (*sendAsync)(const cudensitymatDistributedCommunicator_t*, const void*, int32_t, cudaDataType_t, int32_t, int32_t, cudensitymatDistributedRequest_t)
    int (*receive)(const cudensitymatDistributedCommunicator_t*, void*, int32_t, cudaDataType_t, int32_t, int32_t)
    int (*receiveAsync)(const cudensitymatDistributedCommunicator_t*, void*, int32_t, cudaDataType_t, int32_t, int32_t, cudensitymatDistributedRequest_t)
    int (*bcast)(const cudensitymatDistributedCommunicator_t*, void*, int32_t, cudaDataType_t, int32_t)
    int (*allreduce)(const cudensitymatDistributedCommunicator_t*, const void*, void*, int32_t, cudaDataType_t)
    int (*allreduceInPlace)(const cudensitymatDistributedCommunicator_t*, void*, int32_t, cudaDataType_t)
    int (*allreduceInPlaceMin)(const cudensitymatDistributedCommunicator_t*, void*, int32_t, cudaDataType_t)
    int (*allreduceDoubleIntMinloc)(const cudensitymatDistributedCommunicator_t*, const void*, void*)
    int (*allgather)(const cudensitymatDistributedCommunicator_t*, const void*, void*, int32_t, cudaDataType_t)
ctypedef void (*cudensitymatLoggerCallback_t 'cudensitymatLoggerCallback_t')(
    int32_t logLevel,
    const char* functionName,
    const char* message
)
ctypedef void (*cudensitymatLoggerCallbackData_t 'cudensitymatLoggerCallbackData_t')(
    int32_t logLevel,
    const char* functionName,
    const char* message,
    void* userData
)
ctypedef struct cudensitymatWrappedScalarCallback_t 'cudensitymatWrappedScalarCallback_t':
    cudensitymatScalarCallback_t callback
    cudensitymatCallbackDevice_t device
    void* wrapper
ctypedef struct cudensitymatWrappedTensorCallback_t 'cudensitymatWrappedTensorCallback_t':
    cudensitymatTensorCallback_t callback
    cudensitymatCallbackDevice_t device
    void* wrapper
ctypedef struct cudensitymatWrappedScalarGradientCallback_t 'cudensitymatWrappedScalarGradientCallback_t':
    cudensitymatScalarGradientCallback_t callback
    cudensitymatCallbackDevice_t device
    void* wrapper
    cudensitymatDifferentiationDir_t direction
ctypedef struct cudensitymatWrappedTensorGradientCallback_t 'cudensitymatWrappedTensorGradientCallback_t':
    cudensitymatTensorGradientCallback_t callback
    cudensitymatCallbackDevice_t device
    void* wrapper
    cudensitymatDifferentiationDir_t direction    


###############################################################################
# Functions
###############################################################################

cdef size_t cudensitymatGetVersion() except?0 nogil
cdef cudensitymatStatus_t cudensitymatCreate(cudensitymatHandle_t* handle) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatDestroy(cudensitymatHandle_t handle) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatResetDistributedConfiguration(cudensitymatHandle_t handle, cudensitymatDistributedProvider_t provider, const void* commPtr, size_t commSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatGetNumRanks(const cudensitymatHandle_t handle, int32_t* numRanks) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatGetProcRank(const cudensitymatHandle_t handle, int32_t* procRank) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatResetRandomSeed(cudensitymatHandle_t handle, int32_t randomSeed) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatCreateState(const cudensitymatHandle_t handle, cudensitymatStatePurity_t purity, int32_t numSpaceModes, const int64_t spaceModeExtents[], int64_t batchSize, cudaDataType_t dataType, cudensitymatState_t* state) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatDestroyState(cudensitymatState_t state) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatStateGetNumComponents(const cudensitymatHandle_t handle, const cudensitymatState_t state, int32_t* numStateComponents) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatStateGetComponentStorageSize(const cudensitymatHandle_t handle, const cudensitymatState_t state, int32_t numStateComponents, size_t componentBufferSize[]) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatStateAttachComponentStorage(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t numStateComponents, void* componentBuffer[], const size_t componentBufferSize[]) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatStateGetComponentNumModes(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t stateComponentLocalId, int32_t* stateComponentGlobalId, int32_t* stateComponentNumModes, int32_t* batchModeLocation) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatStateGetComponentInfo(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t stateComponentLocalId, int32_t* stateComponentGlobalId, int32_t* stateComponentNumModes, int64_t stateComponentModeExtents[], int64_t stateComponentModeOffsets[]) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatStateInitializeZero(const cudensitymatHandle_t handle, cudensitymatState_t state, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatStateComputeScaling(const cudensitymatHandle_t handle, cudensitymatState_t state, const void* scalingFactors, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatStateComputeNorm(const cudensitymatHandle_t handle, const cudensitymatState_t state, void* norm, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatStateComputeTrace(const cudensitymatHandle_t handle, const cudensitymatState_t state, void* trace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatStateComputeAccumulation(const cudensitymatHandle_t handle, const cudensitymatState_t stateIn, cudensitymatState_t stateOut, const void* scalingFactors, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatStateComputeInnerProduct(const cudensitymatHandle_t handle, const cudensitymatState_t stateLeft, const cudensitymatState_t stateRight, void* innerProduct, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatCreateElementaryOperator(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatElementaryOperatorSparsity_t sparsity, int32_t numDiagonals, const int32_t diagonalOffsets[], cudaDataType_t dataType, void* tensorData, cudensitymatWrappedTensorCallback_t tensorCallback, cudensitymatWrappedTensorGradientCallback_t tensorGradientCallback, cudensitymatElementaryOperator_t* elemOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatCreateElementaryOperatorBatch(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], int64_t batchSize, cudensitymatElementaryOperatorSparsity_t sparsity, int32_t numDiagonals, const int32_t diagonalOffsets[], cudaDataType_t dataType, void* tensorData, cudensitymatWrappedTensorCallback_t tensorCallback, cudensitymatWrappedTensorGradientCallback_t tensorGradientCallback, cudensitymatElementaryOperator_t* elemOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatDestroyElementaryOperator(cudensitymatElementaryOperator_t elemOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatCreateMatrixOperatorDenseLocal(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudaDataType_t dataType, void* matrixData, cudensitymatWrappedTensorCallback_t matrixCallback, cudensitymatWrappedTensorGradientCallback_t matrixGradientCallback, cudensitymatMatrixOperator_t* matrixOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatCreateMatrixOperatorDenseLocalBatch(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], int64_t batchSize, cudaDataType_t dataType, void* matrixData, cudensitymatWrappedTensorCallback_t matrixCallback, cudensitymatWrappedTensorGradientCallback_t matrixGradientCallback, cudensitymatMatrixOperator_t* matrixOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatDestroyMatrixOperator(cudensitymatMatrixOperator_t matrixOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatCreateOperatorTerm(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatOperatorTerm_t* operatorTerm) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatDestroyOperatorTerm(cudensitymatOperatorTerm_t operatorTerm) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorTermAppendElementaryProduct(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numElemOperators, const cudensitymatElementaryOperator_t elemOperators[], const int32_t stateModesActedOn[], const int32_t modeActionDuality[], cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorTermAppendElementaryProductBatch(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numElemOperators, const cudensitymatElementaryOperator_t elemOperators[], const int32_t stateModesActedOn[], const int32_t modeActionDuality[], int64_t batchSize, const cuDoubleComplex staticCoefficients[], cuDoubleComplex totalCoefficients[], cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorTermAppendMatrixProduct(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numMatrixOperators, const cudensitymatMatrixOperator_t matrixOperators[], const int32_t matrixConjugation[], const int32_t actionDuality[], cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorTermAppendMatrixProductBatch(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numMatrixOperators, const cudensitymatMatrixOperator_t matrixOperators[], const int32_t matrixConjugation[], const int32_t actionDuality[], int64_t batchSize, const cuDoubleComplex staticCoefficients[], cuDoubleComplex totalCoefficients[], cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatCreateOperator(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatOperator_t* superoperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatDestroyOperator(cudensitymatOperator_t superoperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorAppendTerm(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, cudensitymatOperatorTerm_t operatorTerm, int32_t duality, cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorAppendTermBatch(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, cudensitymatOperatorTerm_t operatorTerm, int32_t duality, int64_t batchSize, const cuDoubleComplex staticCoefficients[], cuDoubleComplex totalCoefficients[], cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorPrepareAction(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, const cudensitymatState_t stateIn, const cudensitymatState_t stateOut, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorComputeAction(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t stateIn, cudensitymatState_t stateOut, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorPrepareActionBackwardDiff(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, const cudensitymatState_t stateIn, const cudensitymatState_t stateOutAdj, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorComputeActionBackwardDiff(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t stateIn, const cudensitymatState_t stateOutAdj, cudensitymatState_t stateInAdj, double* paramsGrad, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatCreateOperatorAction(const cudensitymatHandle_t handle, int32_t numOperators, cudensitymatOperator_t operators[], cudensitymatOperatorAction_t* operatorAction) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatDestroyOperatorAction(cudensitymatOperatorAction_t operatorAction) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorActionPrepare(const cudensitymatHandle_t handle, cudensitymatOperatorAction_t operatorAction, const cudensitymatState_t stateIn[], const cudensitymatState_t stateOut, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorActionCompute(const cudensitymatHandle_t handle, cudensitymatOperatorAction_t operatorAction, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t stateIn[], cudensitymatState_t stateOut, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatCreateExpectation(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, cudensitymatExpectation_t* expectation) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatDestroyExpectation(cudensitymatExpectation_t expectation) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatExpectationPrepare(const cudensitymatHandle_t handle, cudensitymatExpectation_t expectation, const cudensitymatState_t state, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatExpectationCompute(const cudensitymatHandle_t handle, cudensitymatExpectation_t expectation, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t state, void* expectationValue, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatCreateOperatorSpectrum(const cudensitymatHandle_t handle, const cudensitymatOperator_t superoperator, int32_t isHermitian, cudensitymatOperatorSpectrumKind_t spectrumKind, cudensitymatOperatorSpectrum_t* spectrum) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatDestroyOperatorSpectrum(cudensitymatOperatorSpectrum_t spectrum) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorSpectrumConfigure(const cudensitymatHandle_t handle, cudensitymatOperatorSpectrum_t spectrum, cudensitymatOperatorSpectrumConfig_t attribute, const void* attributeValue, size_t attributeValueSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorSpectrumPrepare(const cudensitymatHandle_t handle, cudensitymatOperatorSpectrum_t spectrum, int32_t maxEigenStates, const cudensitymatState_t state, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatOperatorSpectrumCompute(const cudensitymatHandle_t handle, cudensitymatOperatorSpectrum_t spectrum, double time, int64_t batchSize, int32_t numParams, const double* params, int32_t numEigenStates, cudensitymatState_t eigenstates[], void* eigenvalues, double* tolerances, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatCreateWorkspace(const cudensitymatHandle_t handle, cudensitymatWorkspaceDescriptor_t* workspaceDescr) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatDestroyWorkspace(cudensitymatWorkspaceDescriptor_t workspaceDescr) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatWorkspaceGetMemorySize(const cudensitymatHandle_t handle, const cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, size_t* memoryBufferSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatWorkspaceSetMemory(const cudensitymatHandle_t handle, cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, void* memoryBuffer, size_t memoryBufferSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudensitymatStatus_t cudensitymatWorkspaceGetMemory(const cudensitymatHandle_t handle, const cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, void** memoryBuffer, size_t* memoryBufferSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil
