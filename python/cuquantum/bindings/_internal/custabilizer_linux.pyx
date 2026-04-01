# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 25.11.0 to 26.02.0, generator version 0.3.1.dev1380+g6ceff55cb.d20260311. Do not modify it directly.

from libc.stdint cimport intptr_t

import threading

from .._utils import FunctionNotFoundError, NotSupportedError


###############################################################################
# Extern
###############################################################################

cdef extern from "<dlfcn.h>" nogil:
    void* dlopen(const char*, int)
    char* dlerror()
    void* dlsym(void*, const char*)
    int dlclose(void*)

    enum:
        RTLD_LAZY
        RTLD_NOW
        RTLD_GLOBAL
        RTLD_LOCAL

    const void* RTLD_DEFAULT 'RTLD_DEFAULT'


###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_custabilizer_init = False

cdef void* __custabilizerGetVersion = NULL
cdef void* __custabilizerGetErrorString = NULL
cdef void* __custabilizerCreate = NULL
cdef void* __custabilizerDestroy = NULL
cdef void* __custabilizerCircuitSizeFromString = NULL
cdef void* __custabilizerCreateCircuitFromString = NULL
cdef void* __custabilizerDestroyCircuit = NULL
cdef void* __custabilizerCreateFrameSimulator = NULL
cdef void* __custabilizerDestroyFrameSimulator = NULL
cdef void* __custabilizerFrameSimulatorApplyCircuit = NULL
cdef void* __custabilizerSampleProbArray = NULL
cdef void* __custabilizerSampleProbArraySparsePrepare = NULL
cdef void* __custabilizerSampleProbArraySparseCompute = NULL
cdef void* __custabilizerGF2SparseDenseMatrixMultiply = NULL
cdef void* __custabilizerGF2SparseSparseMatrixMultiply = NULL


cdef void* load_library() except* nogil:
    cdef void* handle
    handle = dlopen("libcustabilizer.so.0", RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        with gil:
            err_msg = dlerror()
            raise RuntimeError(f'Failed to dlopen libcustabilizer ({err_msg.decode()})')
    return handle


cdef int _check_or_init_custabilizer() except -1 nogil:
    global __py_custabilizer_init
    if __py_custabilizer_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Load function
        global __custabilizerGetVersion
        __custabilizerGetVersion = dlsym(RTLD_DEFAULT, 'custabilizerGetVersion')
        if __custabilizerGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerGetVersion = dlsym(handle, 'custabilizerGetVersion')

        global __custabilizerGetErrorString
        __custabilizerGetErrorString = dlsym(RTLD_DEFAULT, 'custabilizerGetErrorString')
        if __custabilizerGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerGetErrorString = dlsym(handle, 'custabilizerGetErrorString')

        global __custabilizerCreate
        __custabilizerCreate = dlsym(RTLD_DEFAULT, 'custabilizerCreate')
        if __custabilizerCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerCreate = dlsym(handle, 'custabilizerCreate')

        global __custabilizerDestroy
        __custabilizerDestroy = dlsym(RTLD_DEFAULT, 'custabilizerDestroy')
        if __custabilizerDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerDestroy = dlsym(handle, 'custabilizerDestroy')

        global __custabilizerCircuitSizeFromString
        __custabilizerCircuitSizeFromString = dlsym(RTLD_DEFAULT, 'custabilizerCircuitSizeFromString')
        if __custabilizerCircuitSizeFromString == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerCircuitSizeFromString = dlsym(handle, 'custabilizerCircuitSizeFromString')

        global __custabilizerCreateCircuitFromString
        __custabilizerCreateCircuitFromString = dlsym(RTLD_DEFAULT, 'custabilizerCreateCircuitFromString')
        if __custabilizerCreateCircuitFromString == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerCreateCircuitFromString = dlsym(handle, 'custabilizerCreateCircuitFromString')

        global __custabilizerDestroyCircuit
        __custabilizerDestroyCircuit = dlsym(RTLD_DEFAULT, 'custabilizerDestroyCircuit')
        if __custabilizerDestroyCircuit == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerDestroyCircuit = dlsym(handle, 'custabilizerDestroyCircuit')

        global __custabilizerCreateFrameSimulator
        __custabilizerCreateFrameSimulator = dlsym(RTLD_DEFAULT, 'custabilizerCreateFrameSimulator')
        if __custabilizerCreateFrameSimulator == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerCreateFrameSimulator = dlsym(handle, 'custabilizerCreateFrameSimulator')

        global __custabilizerDestroyFrameSimulator
        __custabilizerDestroyFrameSimulator = dlsym(RTLD_DEFAULT, 'custabilizerDestroyFrameSimulator')
        if __custabilizerDestroyFrameSimulator == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerDestroyFrameSimulator = dlsym(handle, 'custabilizerDestroyFrameSimulator')

        global __custabilizerFrameSimulatorApplyCircuit
        __custabilizerFrameSimulatorApplyCircuit = dlsym(RTLD_DEFAULT, 'custabilizerFrameSimulatorApplyCircuit')
        if __custabilizerFrameSimulatorApplyCircuit == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerFrameSimulatorApplyCircuit = dlsym(handle, 'custabilizerFrameSimulatorApplyCircuit')

        global __custabilizerSampleProbArray
        __custabilizerSampleProbArray = dlsym(RTLD_DEFAULT, 'custabilizerSampleProbArray')
        if __custabilizerSampleProbArray == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerSampleProbArray = dlsym(handle, 'custabilizerSampleProbArray')

        global __custabilizerSampleProbArraySparsePrepare
        __custabilizerSampleProbArraySparsePrepare = dlsym(RTLD_DEFAULT, 'custabilizerSampleProbArraySparsePrepare')
        if __custabilizerSampleProbArraySparsePrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerSampleProbArraySparsePrepare = dlsym(handle, 'custabilizerSampleProbArraySparsePrepare')

        global __custabilizerSampleProbArraySparseCompute
        __custabilizerSampleProbArraySparseCompute = dlsym(RTLD_DEFAULT, 'custabilizerSampleProbArraySparseCompute')
        if __custabilizerSampleProbArraySparseCompute == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerSampleProbArraySparseCompute = dlsym(handle, 'custabilizerSampleProbArraySparseCompute')

        global __custabilizerGF2SparseDenseMatrixMultiply
        __custabilizerGF2SparseDenseMatrixMultiply = dlsym(RTLD_DEFAULT, 'custabilizerGF2SparseDenseMatrixMultiply')
        if __custabilizerGF2SparseDenseMatrixMultiply == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerGF2SparseDenseMatrixMultiply = dlsym(handle, 'custabilizerGF2SparseDenseMatrixMultiply')

        global __custabilizerGF2SparseSparseMatrixMultiply
        __custabilizerGF2SparseSparseMatrixMultiply = dlsym(RTLD_DEFAULT, 'custabilizerGF2SparseSparseMatrixMultiply')
        if __custabilizerGF2SparseSparseMatrixMultiply == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerGF2SparseSparseMatrixMultiply = dlsym(handle, 'custabilizerGF2SparseSparseMatrixMultiply')
        __py_custabilizer_init = True
        return 0


cpdef dict _inspect_function_pointers():
    _check_or_init_custabilizer()
    cdef dict data = {}

    global __custabilizerGetVersion
    data["__custabilizerGetVersion"] = <intptr_t>__custabilizerGetVersion

    global __custabilizerGetErrorString
    data["__custabilizerGetErrorString"] = <intptr_t>__custabilizerGetErrorString

    global __custabilizerCreate
    data["__custabilizerCreate"] = <intptr_t>__custabilizerCreate

    global __custabilizerDestroy
    data["__custabilizerDestroy"] = <intptr_t>__custabilizerDestroy

    global __custabilizerCircuitSizeFromString
    data["__custabilizerCircuitSizeFromString"] = <intptr_t>__custabilizerCircuitSizeFromString

    global __custabilizerCreateCircuitFromString
    data["__custabilizerCreateCircuitFromString"] = <intptr_t>__custabilizerCreateCircuitFromString

    global __custabilizerDestroyCircuit
    data["__custabilizerDestroyCircuit"] = <intptr_t>__custabilizerDestroyCircuit

    global __custabilizerCreateFrameSimulator
    data["__custabilizerCreateFrameSimulator"] = <intptr_t>__custabilizerCreateFrameSimulator

    global __custabilizerDestroyFrameSimulator
    data["__custabilizerDestroyFrameSimulator"] = <intptr_t>__custabilizerDestroyFrameSimulator

    global __custabilizerFrameSimulatorApplyCircuit
    data["__custabilizerFrameSimulatorApplyCircuit"] = <intptr_t>__custabilizerFrameSimulatorApplyCircuit

    global __custabilizerSampleProbArray
    data["__custabilizerSampleProbArray"] = <intptr_t>__custabilizerSampleProbArray

    global __custabilizerSampleProbArraySparsePrepare
    data["__custabilizerSampleProbArraySparsePrepare"] = <intptr_t>__custabilizerSampleProbArraySparsePrepare

    global __custabilizerSampleProbArraySparseCompute
    data["__custabilizerSampleProbArraySparseCompute"] = <intptr_t>__custabilizerSampleProbArraySparseCompute

    global __custabilizerGF2SparseDenseMatrixMultiply
    data["__custabilizerGF2SparseDenseMatrixMultiply"] = <intptr_t>__custabilizerGF2SparseDenseMatrixMultiply

    global __custabilizerGF2SparseSparseMatrixMultiply
    data["__custabilizerGF2SparseSparseMatrixMultiply"] = <intptr_t>__custabilizerGF2SparseSparseMatrixMultiply

    return data


###############################################################################
# Wrapper functions
###############################################################################

cdef int _custabilizerGetVersion() except?-42 nogil:
    global __custabilizerGetVersion
    _check_or_init_custabilizer()
    if __custabilizerGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerGetVersion is not found")
    return (<int (*)() noexcept nogil>__custabilizerGetVersion)(
        )


cdef const char* _custabilizerGetErrorString(custabilizerStatus_t status) except?NULL nogil:
    global __custabilizerGetErrorString
    _check_or_init_custabilizer()
    if __custabilizerGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerGetErrorString is not found")
    return (<const char* (*)(custabilizerStatus_t) noexcept nogil>__custabilizerGetErrorString)(
        status)


cdef custabilizerStatus_t _custabilizerCreate(custabilizerHandle_t* handle) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerCreate
    _check_or_init_custabilizer()
    if __custabilizerCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerCreate is not found")
    return (<custabilizerStatus_t (*)(custabilizerHandle_t*) noexcept nogil>__custabilizerCreate)(
        handle)


cdef custabilizerStatus_t _custabilizerDestroy(custabilizerHandle_t handle) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerDestroy
    _check_or_init_custabilizer()
    if __custabilizerDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerDestroy is not found")
    return (<custabilizerStatus_t (*)(custabilizerHandle_t) noexcept nogil>__custabilizerDestroy)(
        handle)


cdef custabilizerStatus_t _custabilizerCircuitSizeFromString(const custabilizerHandle_t handle, const char* circuitString, int64_t* bufferSize) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerCircuitSizeFromString
    _check_or_init_custabilizer()
    if __custabilizerCircuitSizeFromString == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerCircuitSizeFromString is not found")
    return (<custabilizerStatus_t (*)(const custabilizerHandle_t, const char*, int64_t*) noexcept nogil>__custabilizerCircuitSizeFromString)(
        handle, circuitString, bufferSize)


cdef custabilizerStatus_t _custabilizerCreateCircuitFromString(const custabilizerHandle_t handle, const char* circuitString, void* bufferDevice, int64_t bufferSize, custabilizerCircuit_t* circuit) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerCreateCircuitFromString
    _check_or_init_custabilizer()
    if __custabilizerCreateCircuitFromString == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerCreateCircuitFromString is not found")
    return (<custabilizerStatus_t (*)(const custabilizerHandle_t, const char*, void*, int64_t, custabilizerCircuit_t*) noexcept nogil>__custabilizerCreateCircuitFromString)(
        handle, circuitString, bufferDevice, bufferSize, circuit)


cdef custabilizerStatus_t _custabilizerDestroyCircuit(custabilizerCircuit_t circuit) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerDestroyCircuit
    _check_or_init_custabilizer()
    if __custabilizerDestroyCircuit == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerDestroyCircuit is not found")
    return (<custabilizerStatus_t (*)(custabilizerCircuit_t) noexcept nogil>__custabilizerDestroyCircuit)(
        circuit)


cdef custabilizerStatus_t _custabilizerCreateFrameSimulator(const custabilizerHandle_t handle, int64_t numQubits, int64_t numShots, int64_t numMeasurements, int64_t tableStrideMajor, custabilizerFrameSimulator_t* frameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerCreateFrameSimulator
    _check_or_init_custabilizer()
    if __custabilizerCreateFrameSimulator == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerCreateFrameSimulator is not found")
    return (<custabilizerStatus_t (*)(const custabilizerHandle_t, int64_t, int64_t, int64_t, int64_t, custabilizerFrameSimulator_t*) noexcept nogil>__custabilizerCreateFrameSimulator)(
        handle, numQubits, numShots, numMeasurements, tableStrideMajor, frameSimulator)


cdef custabilizerStatus_t _custabilizerDestroyFrameSimulator(custabilizerFrameSimulator_t frameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerDestroyFrameSimulator
    _check_or_init_custabilizer()
    if __custabilizerDestroyFrameSimulator == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerDestroyFrameSimulator is not found")
    return (<custabilizerStatus_t (*)(custabilizerFrameSimulator_t) noexcept nogil>__custabilizerDestroyFrameSimulator)(
        frameSimulator)


cdef custabilizerStatus_t _custabilizerFrameSimulatorApplyCircuit(const custabilizerHandle_t handle, custabilizerFrameSimulator_t frameSimulator, const custabilizerCircuit_t circuit, int randomizeFrameAfterMeasurement, uint64_t seed, custabilizerBitInt_t* xTableDevice, custabilizerBitInt_t* zTableDevice, custabilizerBitInt_t* mTableDevice, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerFrameSimulatorApplyCircuit
    _check_or_init_custabilizer()
    if __custabilizerFrameSimulatorApplyCircuit == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerFrameSimulatorApplyCircuit is not found")
    return (<custabilizerStatus_t (*)(const custabilizerHandle_t, custabilizerFrameSimulator_t, const custabilizerCircuit_t, int, uint64_t, custabilizerBitInt_t*, custabilizerBitInt_t*, custabilizerBitInt_t*, cudaStream_t) noexcept nogil>__custabilizerFrameSimulatorApplyCircuit)(
        handle, frameSimulator, circuit, randomizeFrameAfterMeasurement, seed, xTableDevice, zTableDevice, mTableDevice, stream)


cdef custabilizerStatus_t _custabilizerSampleProbArray(custabilizerHandle_t handle, int64_t numSamples, int64_t numProbs, const double* probs, uint64_t seed, custabilizerBitInt_t* samples, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerSampleProbArray
    _check_or_init_custabilizer()
    if __custabilizerSampleProbArray == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerSampleProbArray is not found")
    return (<custabilizerStatus_t (*)(custabilizerHandle_t, int64_t, int64_t, const double*, uint64_t, custabilizerBitInt_t*, cudaStream_t) noexcept nogil>__custabilizerSampleProbArray)(
        handle, numSamples, numProbs, probs, seed, samples, stream)


cdef custabilizerStatus_t _custabilizerSampleProbArraySparsePrepare(custabilizerHandle_t handle, int64_t numSamples, int64_t numProbs, size_t* workspaceSize) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerSampleProbArraySparsePrepare
    _check_or_init_custabilizer()
    if __custabilizerSampleProbArraySparsePrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerSampleProbArraySparsePrepare is not found")
    return (<custabilizerStatus_t (*)(custabilizerHandle_t, int64_t, int64_t, size_t*) noexcept nogil>__custabilizerSampleProbArraySparsePrepare)(
        handle, numSamples, numProbs, workspaceSize)


cdef custabilizerStatus_t _custabilizerSampleProbArraySparseCompute(custabilizerHandle_t handle, int64_t numSamples, int64_t numProbs, const double* probs, uint64_t seed, uint64_t* nnz, uint64_t* columnIndices, uint64_t* rowOffsets, void* workspace, size_t workspaceSize, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerSampleProbArraySparseCompute
    _check_or_init_custabilizer()
    if __custabilizerSampleProbArraySparseCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerSampleProbArraySparseCompute is not found")
    return (<custabilizerStatus_t (*)(custabilizerHandle_t, int64_t, int64_t, const double*, uint64_t, uint64_t*, uint64_t*, uint64_t*, void*, size_t, cudaStream_t) noexcept nogil>__custabilizerSampleProbArraySparseCompute)(
        handle, numSamples, numProbs, probs, seed, nnz, columnIndices, rowOffsets, workspace, workspaceSize, stream)


cdef custabilizerStatus_t _custabilizerGF2SparseDenseMatrixMultiply(custabilizerHandle_t handle, uint64_t m, uint64_t n, uint64_t k, uint64_t nnz, const uint64_t* columnIndices, const uint64_t* rowOffsets, const custabilizerBitInt_t* B, int32_t beta, custabilizerBitInt_t* C, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerGF2SparseDenseMatrixMultiply
    _check_or_init_custabilizer()
    if __custabilizerGF2SparseDenseMatrixMultiply == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerGF2SparseDenseMatrixMultiply is not found")
    return (<custabilizerStatus_t (*)(custabilizerHandle_t, uint64_t, uint64_t, uint64_t, uint64_t, const uint64_t*, const uint64_t*, const custabilizerBitInt_t*, int32_t, custabilizerBitInt_t*, cudaStream_t) noexcept nogil>__custabilizerGF2SparseDenseMatrixMultiply)(
        handle, m, n, k, nnz, columnIndices, rowOffsets, B, beta, C, stream)


cdef custabilizerStatus_t _custabilizerGF2SparseSparseMatrixMultiply(custabilizerHandle_t handle, uint64_t m, uint64_t n, uint64_t k, const uint64_t* aColumnIndices, const uint64_t* aRowOffsets, uint64_t bNNZ, const uint64_t* bColumnIndices, const uint64_t* bRowOffsets, int32_t beta, custabilizerBitInt_t* C, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerGF2SparseSparseMatrixMultiply
    _check_or_init_custabilizer()
    if __custabilizerGF2SparseSparseMatrixMultiply == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerGF2SparseSparseMatrixMultiply is not found")
    return (<custabilizerStatus_t (*)(custabilizerHandle_t, uint64_t, uint64_t, uint64_t, const uint64_t*, const uint64_t*, uint64_t, const uint64_t*, const uint64_t*, int32_t, custabilizerBitInt_t*, cudaStream_t) noexcept nogil>__custabilizerGF2SparseSparseMatrixMultiply)(
        handle, m, n, k, aColumnIndices, aRowOffsets, bNNZ, bColumnIndices, bRowOffsets, beta, C, stream)
