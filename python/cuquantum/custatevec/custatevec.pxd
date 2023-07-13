# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# The C types are prefixed with an underscore because we are not
# yet protected by the module namespaces as done in CUDA Python.
# Once we switch over the names would be prettier (in the Cython
# layer).

from libc.stdint cimport intptr_t, int32_t, uint32_t, int64_t

from cuquantum.utils cimport (DataType, DeviceAllocType, DeviceFreeType, int2,
                              LibPropType, Stream, Event)


cdef extern from '<custatevec.h>' nogil:
    # cuStateVec consts
    const int CUSTATEVEC_VER_MAJOR
    const int CUSTATEVEC_VER_MINOR
    const int CUSTATEVEC_VER_PATCH
    const int CUSTATEVEC_VERSION
    const int CUSTATEVEC_ALLOCATOR_NAME_LEN
    const int CUSTATEVEC_MAX_SEGMENT_MASK_SIZE

    # cuStateVec types
    ctypedef void* _Handle 'custatevecHandle_t'
    ctypedef int64_t _Index 'custatevecIndex_t'
    ctypedef int _Status 'custatevecStatus_t'
    ctypedef void* _SamplerDescriptor 'custatevecSamplerDescriptor_t'
    ctypedef void* _AccessorDescriptor 'custatevecAccessorDescriptor_t'
    ctypedef struct _DeviceMemHandler 'custatevecDeviceMemHandler_t':
        void* ctx
        DeviceAllocType device_alloc
        DeviceFreeType device_free
        char name[CUSTATEVEC_ALLOCATOR_NAME_LEN]
    ctypedef void(*LoggerCallbackData 'custatevecLoggerCallbackData_t')(
        int32_t logLevel,
        const char* functionName,
        const char* message,
        void* userData)
    ctypedef void* _CommunicatorDescriptor 'custatevecCommunicatorDescriptor_t'
    ctypedef struct _SVSwapParameters 'custatevecSVSwapParameters_t':
        int32_t swapBatchIndex
        int32_t orgSubSVIndex
        int32_t dstSubSVIndex
        # Same Cython limitation as above
        int32_t orgSegmentMaskString[48]
        int32_t dstSegmentMaskString[48]
        int32_t segmentMaskOrdering[48]
        uint32_t segmentMaskLen
        uint32_t nSegmentBits
        _DataTransferType dataTransferType
        _Index transferSize
    ctypedef void* _DistIndexBitSwapSchedulerDescriptor 'custatevecDistIndexBitSwapSchedulerDescriptor_t'
    ctypedef void* _SVSwapWorkerDescriptor 'custatevecSVSwapWorkerDescriptor_t'


    # cuStateVec enums
    ctypedef enum _ComputeType 'custatevecComputeType_t':
        pass

    ctypedef enum _Pauli 'custatevecPauli_t':
        CUSTATEVEC_PAULI_I
        CUSTATEVEC_PAULI_X
        CUSTATEVEC_PAULI_Y
        CUSTATEVEC_PAULI_Z

    ctypedef enum _MatrixLayout 'custatevecMatrixLayout_t':
        CUSTATEVEC_MATRIX_LAYOUT_COL
        CUSTATEVEC_MATRIX_LAYOUT_ROW

    ctypedef enum _MatrixType 'custatevecMatrixType_t':
        CUSTATEVEC_MATRIX_TYPE_GENERAL
        CUSTATEVEC_MATRIX_TYPE_UNITARY
        CUSTATEVEC_MATRIX_TYPE_HERMITIAN

    ctypedef enum _MatrixMapType 'custatevecMatrixMapType_t':
        CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST
        CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED

    ctypedef enum _CollapseOp 'custatevecCollapseOp_t':
        CUSTATEVEC_COLLAPSE_NONE
        CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO

    ctypedef enum _SamplerOutput 'custatevecSamplerOutput_t':
        CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER
        CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER

    ctypedef enum _DeviceNetworkType 'custatevecDeviceNetworkType_t':
        CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH
        CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH

    ctypedef enum _CommunicatorType 'custatevecCommunicatorType_t':
        CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL
        CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI
        CUSTATEVEC_COMMUNICATOR_TYPE_MPICH

    ctypedef enum _DataTransferType 'custatevecDataTransferType_t':
        CUSTATEVEC_DATA_TRANSFER_TYPE_NONE
        CUSTATEVEC_DATA_TRANSFER_TYPE_SEND
        CUSTATEVEC_DATA_TRANSFER_TYPE_RECV
        CUSTATEVEC_DATA_TRANSFER_TYPE_SEND_RECV

    ctypedef enum _StateVectorType 'custatevecStateVectorType_t':
        CUSTATEVEC_STATE_VECTOR_TYPE_ZERO
        CUSTATEVEC_STATE_VECTOR_TYPE_UNIFORM
        CUSTATEVEC_STATE_VECTOR_TYPE_GHZ
        CUSTATEVEC_STATE_VECTOR_TYPE_W

    # cuStateVec consts
    int CUSTATEVEC_VER_MAJOR
    int CUSTATEVEC_VER_MINOR
    int CUSTATEVEC_VER_PATCH
    int CUSTATEVEC_VERSION
    int CUSTATEVEC_ALLOCATOR_NAME_LEN
    int CUSTATEVEC_MAX_SEGMENT_MASK_SIZE
