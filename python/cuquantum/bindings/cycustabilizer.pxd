# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 25.11.0 to 26.11.0. Do not modify it directly.


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    int32_t,
    int64_t,
    uint32_t,
    uint64_t,
)


# <<<< END OF PREAMBLE CONTENT >>>>

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
    CUSTABILIZER_STATUS_INSUFFICIENT_SPARSE_STORAGE "CUSTABILIZER_STATUS_INSUFFICIENT_SPARSE_STORAGE" = 9
    _CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR "_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum custabilizerCircuitAttributes_t "custabilizerCircuitAttributes_t":
    CUSTABILIZER_CIRCUIT_NUM_QUBITS "CUSTABILIZER_CIRCUIT_NUM_QUBITS" = 0
    CUSTABILIZER_CIRCUIT_NUM_MEASUREMENT_GATES "CUSTABILIZER_CIRCUIT_NUM_MEASUREMENT_GATES" = 1
    CUSTABILIZER_CIRCUIT_NUM_DETECTORS "CUSTABILIZER_CIRCUIT_NUM_DETECTORS" = 2
    CUSTABILIZER_CIRCUIT_NUM_REPEAT_BLOCKS "CUSTABILIZER_CIRCUIT_NUM_REPEAT_BLOCKS" = 3
    CUSTABILIZER_CIRCUIT_NUM_RESETS "CUSTABILIZER_CIRCUIT_NUM_RESETS" = 4
    CUSTABILIZER_CIRCUIT_NUM_1Q_GATES "CUSTABILIZER_CIRCUIT_NUM_1Q_GATES" = 5
    CUSTABILIZER_CIRCUIT_NUM_2Q_GATES "CUSTABILIZER_CIRCUIT_NUM_2Q_GATES" = 6
    CUSTABILIZER_CIRCUIT_HAS_NOISE "CUSTABILIZER_CIRCUIT_HAS_NOISE" = 7
    CUSTABILIZER_CIRCUIT_NUM_LEAKAGE_INSTRUCTIONS "CUSTABILIZER_CIRCUIT_NUM_LEAKAGE_INSTRUCTIONS" = 8
    CUSTABILIZER_CIRCUIT_NUM_LEAKAGE_READOUTS "CUSTABILIZER_CIRCUIT_NUM_LEAKAGE_READOUTS" = 9
    CUSTABILIZER_CIRCUIT_NUM_MEASUREMENT_BITS "CUSTABILIZER_CIRCUIT_NUM_MEASUREMENT_BITS" = 10

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

ctypedef void* custabilizerLeakageFrameSimulator_t 'custabilizerLeakageFrameSimulator_t'    


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
cdef custabilizerStatus_t custabilizerSampleProbArray(custabilizerHandle_t handle, int64_t numSamples, int64_t numProbs, const double* probs, uint64_t seed, custabilizerBitInt_t* samples, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerSampleProbArraySparsePrepare(custabilizerHandle_t handle, int64_t numSamples, int64_t numProbs, size_t* workspaceSize) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerSampleProbArraySparseCompute(custabilizerHandle_t handle, int64_t numSamples, int64_t numProbs, const double* probs, uint64_t seed, uint64_t* nnz, uint64_t* columnIndices, uint64_t* rowOffsets, void* workspace, size_t workspaceSize, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerGF2SparseDenseMatrixMultiply(custabilizerHandle_t handle, uint64_t m, uint64_t n, uint64_t k, uint64_t nnz, const uint64_t* columnIndices, const uint64_t* rowOffsets, const custabilizerBitInt_t* B, int32_t beta, custabilizerBitInt_t* C, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerGF2SparseSparseMatrixMultiply(custabilizerHandle_t handle, uint64_t m, uint64_t n, uint64_t k, const uint64_t* aColumnIndices, const uint64_t* aRowOffsets, uint64_t bNNZ, const uint64_t* bColumnIndices, const uint64_t* bRowOffsets, int32_t beta, custabilizerBitInt_t* C, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerCircuitGetAttribute(const custabilizerHandle_t handle, const custabilizerCircuit_t circuit, custabilizerCircuitAttributes_t attribute, void* buffer, size_t sizeInBytes) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerCreateLeakageFrameSimulator(const custabilizerHandle_t handle, int64_t numQubits, int64_t numShots, int64_t numMeasurements, int64_t tableStrideMajor, custabilizerLeakageFrameSimulator_t* leakageFrameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerDestroyLeakageFrameSimulator(custabilizerLeakageFrameSimulator_t leakageFrameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t custabilizerLeakageFrameSimulatorApplyCircuit(const custabilizerHandle_t handle, custabilizerLeakageFrameSimulator_t leakageFrameSimulator, const custabilizerCircuit_t circuit, int randomizeFrameAfterMeasurement, uint64_t seed, custabilizerBitInt_t* xTableDevice, custabilizerBitInt_t* zTableDevice, custabilizerBitInt_t* lTableDevice, custabilizerBitInt_t* mTableDevice, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
