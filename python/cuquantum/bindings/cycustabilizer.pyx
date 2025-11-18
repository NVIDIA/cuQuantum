# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.11.0. Do not modify it directly.

from ._internal cimport custabilizer as _custabilizer


###############################################################################
# Wrapper functions
###############################################################################

cdef int custabilizerGetVersion() except?-42 nogil:
    return _custabilizer._custabilizerGetVersion()


cdef const char* custabilizerGetErrorString(custabilizerStatus_t status) except?NULL nogil:
    return _custabilizer._custabilizerGetErrorString(status)


cdef custabilizerStatus_t custabilizerCreate(custabilizerHandle_t* handle) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custabilizer._custabilizerCreate(handle)


cdef custabilizerStatus_t custabilizerDestroy(custabilizerHandle_t handle) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custabilizer._custabilizerDestroy(handle)


cdef custabilizerStatus_t custabilizerCircuitSizeFromString(const custabilizerHandle_t handle, const char* circuitString, int64_t* bufferSize) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custabilizer._custabilizerCircuitSizeFromString(handle, circuitString, bufferSize)


cdef custabilizerStatus_t custabilizerCreateCircuitFromString(const custabilizerHandle_t handle, const char* circuitString, void* bufferDevice, int64_t bufferSize, custabilizerCircuit_t* circuit) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custabilizer._custabilizerCreateCircuitFromString(handle, circuitString, bufferDevice, bufferSize, circuit)


cdef custabilizerStatus_t custabilizerDestroyCircuit(custabilizerCircuit_t circuit) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custabilizer._custabilizerDestroyCircuit(circuit)


cdef custabilizerStatus_t custabilizerCreateFrameSimulator(const custabilizerHandle_t handle, int64_t numQubits, int64_t numShots, int64_t numMeasurements, int64_t tableStrideMajor, custabilizerFrameSimulator_t* frameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custabilizer._custabilizerCreateFrameSimulator(handle, numQubits, numShots, numMeasurements, tableStrideMajor, frameSimulator)


cdef custabilizerStatus_t custabilizerDestroyFrameSimulator(custabilizerFrameSimulator_t frameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custabilizer._custabilizerDestroyFrameSimulator(frameSimulator)


cdef custabilizerStatus_t custabilizerFrameSimulatorApplyCircuit(const custabilizerHandle_t handle, custabilizerFrameSimulator_t frameSimulator, const custabilizerCircuit_t circuit, int randomizeFrameAfterMeasurement, uint64_t seed, custabilizerBitInt_t* xTableDevice, custabilizerBitInt_t* zTableDevice, custabilizerBitInt_t* mTableDevice, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _custabilizer._custabilizerFrameSimulatorApplyCircuit(handle, frameSimulator, circuit, randomizeFrameAfterMeasurement, seed, xTableDevice, zTableDevice, mTableDevice, stream)
