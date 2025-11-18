# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.11.0. Do not modify it directly.

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
