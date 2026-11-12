# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 25.11.0 to 26.11.0. Do not modify it directly.


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    int32_t,
    int64_t,
    uint64_t,
)


# <<<< END OF PREAMBLE CONTENT >>>>

from ..cycustabilizer cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef int _custabilizerGetVersion() except?-42 nogil
cdef const char* _custabilizerGetErrorString(custabilizerStatus_t status) except?NULL nogil
cdef custabilizerStatus_t _custabilizerCreate(custabilizerHandle_t* handle) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerDestroy(custabilizerHandle_t handle) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerCircuitSizeFromString(const custabilizerHandle_t handle, const char* circuitString, int64_t* bufferSize) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerCreateCircuitFromString(const custabilizerHandle_t handle, const char* circuitString, void* bufferDevice, int64_t bufferSize, custabilizerCircuit_t* circuit) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerDestroyCircuit(custabilizerCircuit_t circuit) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerCreateFrameSimulator(const custabilizerHandle_t handle, int64_t numQubits, int64_t numShots, int64_t numMeasurements, int64_t tableStrideMajor, custabilizerFrameSimulator_t* frameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerDestroyFrameSimulator(custabilizerFrameSimulator_t frameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerFrameSimulatorApplyCircuit(const custabilizerHandle_t handle, custabilizerFrameSimulator_t frameSimulator, const custabilizerCircuit_t circuit, int randomizeFrameAfterMeasurement, uint64_t seed, custabilizerBitInt_t* xTableDevice, custabilizerBitInt_t* zTableDevice, custabilizerBitInt_t* mTableDevice, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerSampleProbArray(custabilizerHandle_t handle, int64_t numSamples, int64_t numProbs, const double* probs, uint64_t seed, custabilizerBitInt_t* samples, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerSampleProbArraySparsePrepare(custabilizerHandle_t handle, int64_t numSamples, int64_t numProbs, size_t* workspaceSize) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerSampleProbArraySparseCompute(custabilizerHandle_t handle, int64_t numSamples, int64_t numProbs, const double* probs, uint64_t seed, uint64_t* nnz, uint64_t* columnIndices, uint64_t* rowOffsets, void* workspace, size_t workspaceSize, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerGF2SparseDenseMatrixMultiply(custabilizerHandle_t handle, uint64_t m, uint64_t n, uint64_t k, uint64_t nnz, const uint64_t* columnIndices, const uint64_t* rowOffsets, const custabilizerBitInt_t* B, int32_t beta, custabilizerBitInt_t* C, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerGF2SparseSparseMatrixMultiply(custabilizerHandle_t handle, uint64_t m, uint64_t n, uint64_t k, const uint64_t* aColumnIndices, const uint64_t* aRowOffsets, uint64_t bNNZ, const uint64_t* bColumnIndices, const uint64_t* bRowOffsets, int32_t beta, custabilizerBitInt_t* C, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerCircuitGetAttribute(const custabilizerHandle_t handle, const custabilizerCircuit_t circuit, custabilizerCircuitAttributes_t attribute, void* buffer, size_t sizeInBytes) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerCreateLeakageFrameSimulator(const custabilizerHandle_t handle, int64_t numQubits, int64_t numShots, int64_t numMeasurements, int64_t tableStrideMajor, custabilizerLeakageFrameSimulator_t* leakageFrameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerDestroyLeakageFrameSimulator(custabilizerLeakageFrameSimulator_t leakageFrameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef custabilizerStatus_t _custabilizerLeakageFrameSimulatorApplyCircuit(const custabilizerHandle_t handle, custabilizerLeakageFrameSimulator_t leakageFrameSimulator, const custabilizerCircuit_t circuit, int randomizeFrameAfterMeasurement, uint64_t seed, custabilizerBitInt_t* xTableDevice, custabilizerBitInt_t* zTableDevice, custabilizerBitInt_t* lTableDevice, custabilizerBitInt_t* mTableDevice, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil
