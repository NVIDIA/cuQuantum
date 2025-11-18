# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.11.0. Do not modify it directly.

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.stdio cimport FILE


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum custabilizerStatus_t "custabilizerStatus_t":
    CUSTABILIZER_STATUS_SUCCESS "CUSTABILIZER_STATUS_SUCCESS" = 0
    CUSTABILIZER_STATUS_ERROR "CUSTABILIZER_STATUS_ERROR" = 1
    CUSTABILIZER_STATUS_NOT_INITIALIZED "CUSTABILIZER_STATUS_NOT_INITIALIZED" = 2
    CUSTABILIZER_STATUS_INVALID_VALUE "CUSTABILIZER_STATUS_INVALID_VALUE" = 3
    CUSTABILIZER_STATUS_NOT_SUPPORTED "CUSTABILIZER_STATUS_NOT_SUPPORTED" = 4
    CUSTABILIZER_STATUS_ALLOC_FAILED "CUSTABILIZER_STATUS_ALLOC_FAILED" = 5
    CUSTABILIZER_STATUS_INTERNAL_ERROR "CUSTABILIZER_STATUS_INTERNAL_ERROR" = 6
    CUSTABILIZER_STATUS_INSUFFICIENT_WORKSPACE "CUSTABILIZER_STATUS_INSUFFICIENT_WORKSPACE" = 7
    CUSTABILIZER_STATUS_CUDA_ERROR "CUSTABILIZER_STATUS_CUDA_ERROR" = 8
    _CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR "_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR" = -42

cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>

    """

    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef void* cudaEvent_t 'cudaEvent_t'

    ctypedef enum cudaDataType_t:
        CUDA_R_32F
        CUDA_C_32F
        CUDA_R_64F
        CUDA_C_64F
    ctypedef cudaDataType_t cudaDataType 'cudaDataType'

# types
ctypedef uint32_t custabilizerBitInt_t 'custabilizerBitInt_t'
ctypedef void* custabilizerCircuit_t 'custabilizerCircuit_t'
ctypedef void* custabilizerFrameSimulator_t 'custabilizerFrameSimulator_t'
ctypedef void* custabilizerHandle_t 'custabilizerHandle_t'    


###############################################################################
# Functions
###############################################################################

cdef int custabilizerGetVersion() except?-42 nogil
cdef const char* custabilizerGetErrorString(custabilizerStatus_t status) except?NULL nogil
cdef custabilizerStatus_t custabilizerCreate(custabilizerHandle_t* handle) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerDestroy(custabilizerHandle_t handle) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerCircuitSizeFromString(const custabilizerHandle_t handle, const char* circuitString, int64_t* bufferSize) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerCreateCircuitFromString(const custabilizerHandle_t handle, const char* circuitString, void* bufferDevice, int64_t bufferSize, custabilizerCircuit_t* circuit) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerDestroyCircuit(custabilizerCircuit_t circuit) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerCreateFrameSimulator(const custabilizerHandle_t handle, int64_t numQubits, int64_t numShots, int64_t numMeasurements, int64_t tableStrideMajor, custabilizerFrameSimulator_t* frameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerDestroyFrameSimulator(custabilizerFrameSimulator_t frameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerFrameSimulatorApplyCircuit(const custabilizerHandle_t handle, custabilizerFrameSimulator_t frameSimulator, const custabilizerCircuit_t circuit, int randomizeFrameAfterMeasurement, uint64_t seed, custabilizerBitInt_t* xTableDevice, custabilizerBitInt_t* zTableDevice, custabilizerBitInt_t* mTableDevice, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
