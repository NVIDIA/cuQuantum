# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 25.11.0 to 26.11.0. Do not modify it directly.



# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    int32_t,
    int64_t,
    intptr_t,
    uint64_t,
)

cdef extern from * nogil:
    """
    #if defined(_MSC_VER) && !defined(__clang__)
        #include <intrin.h>
        static __forceinline int atomic_int_load(int *p) {
            int v = *(int volatile *)p; _ReadBarrier(); return v;
        }
        static __forceinline void atomic_int_store(int *p, int v) {
            _WriteBarrier(); *(int volatile *)p = v;
        }
    #elif defined(__cplusplus)
        /* GCC/Clang __atomic builtins work in any C++ standard without headers */
        static inline int atomic_int_load(int *p) {
            return __atomic_load_n(p, __ATOMIC_ACQUIRE);
        }
        static inline void atomic_int_store(int *p, int v) {
            __atomic_store_n(p, v, __ATOMIC_RELEASE);
        }
    #else
        #include <stdatomic.h>
        static inline int atomic_int_load(int *p) {
            return (int)atomic_load_explicit((atomic_int *)p, memory_order_acquire);
        }
        static inline void atomic_int_store(int *p, int v) {
            atomic_store_explicit((atomic_int *)p, v, memory_order_release);
        }
    #endif

    """
    cdef int _cyb_atomic_int_load "atomic_int_load"(int *p) nogil
    cdef void _cyb_atomic_int_store "atomic_int_store"(int *p, int v) nogil

cdef extern from "<dlfcn.h>":
    void* _cyb_dlsym "dlsym"(void*, const char*) nogil
    const void * _cyb_RTLD_DEFAULT "RTLD_DEFAULT"

import threading as _cyb_threading

cdef int _cyb___py_custabilizer_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .._utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################


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
cdef void* __custabilizerCircuitGetAttribute = NULL
cdef void* __custabilizerCreateLeakageFrameSimulator = NULL
cdef void* __custabilizerDestroyLeakageFrameSimulator = NULL
cdef void* __custabilizerLeakageFrameSimulatorApplyCircuit = NULL

cdef int _init_custabilizer() except -1 nogil:
    global _cyb___py_custabilizer_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_custabilizer_init: return 0

        global __custabilizerGetVersion
        __custabilizerGetVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerGetVersion')
        if __custabilizerGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerGetVersion = _cyb_dlsym(handle, 'custabilizerGetVersion')

        global __custabilizerGetErrorString
        __custabilizerGetErrorString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerGetErrorString')
        if __custabilizerGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerGetErrorString = _cyb_dlsym(handle, 'custabilizerGetErrorString')

        global __custabilizerCreate
        __custabilizerCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerCreate')
        if __custabilizerCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerCreate = _cyb_dlsym(handle, 'custabilizerCreate')

        global __custabilizerDestroy
        __custabilizerDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerDestroy')
        if __custabilizerDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerDestroy = _cyb_dlsym(handle, 'custabilizerDestroy')

        global __custabilizerCircuitSizeFromString
        __custabilizerCircuitSizeFromString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerCircuitSizeFromString')
        if __custabilizerCircuitSizeFromString == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerCircuitSizeFromString = _cyb_dlsym(handle, 'custabilizerCircuitSizeFromString')

        global __custabilizerCreateCircuitFromString
        __custabilizerCreateCircuitFromString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerCreateCircuitFromString')
        if __custabilizerCreateCircuitFromString == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerCreateCircuitFromString = _cyb_dlsym(handle, 'custabilizerCreateCircuitFromString')

        global __custabilizerDestroyCircuit
        __custabilizerDestroyCircuit = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerDestroyCircuit')
        if __custabilizerDestroyCircuit == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerDestroyCircuit = _cyb_dlsym(handle, 'custabilizerDestroyCircuit')

        global __custabilizerCreateFrameSimulator
        __custabilizerCreateFrameSimulator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerCreateFrameSimulator')
        if __custabilizerCreateFrameSimulator == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerCreateFrameSimulator = _cyb_dlsym(handle, 'custabilizerCreateFrameSimulator')

        global __custabilizerDestroyFrameSimulator
        __custabilizerDestroyFrameSimulator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerDestroyFrameSimulator')
        if __custabilizerDestroyFrameSimulator == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerDestroyFrameSimulator = _cyb_dlsym(handle, 'custabilizerDestroyFrameSimulator')

        global __custabilizerFrameSimulatorApplyCircuit
        __custabilizerFrameSimulatorApplyCircuit = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerFrameSimulatorApplyCircuit')
        if __custabilizerFrameSimulatorApplyCircuit == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerFrameSimulatorApplyCircuit = _cyb_dlsym(handle, 'custabilizerFrameSimulatorApplyCircuit')

        global __custabilizerSampleProbArray
        __custabilizerSampleProbArray = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerSampleProbArray')
        if __custabilizerSampleProbArray == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerSampleProbArray = _cyb_dlsym(handle, 'custabilizerSampleProbArray')

        global __custabilizerSampleProbArraySparsePrepare
        __custabilizerSampleProbArraySparsePrepare = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerSampleProbArraySparsePrepare')
        if __custabilizerSampleProbArraySparsePrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerSampleProbArraySparsePrepare = _cyb_dlsym(handle, 'custabilizerSampleProbArraySparsePrepare')

        global __custabilizerSampleProbArraySparseCompute
        __custabilizerSampleProbArraySparseCompute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerSampleProbArraySparseCompute')
        if __custabilizerSampleProbArraySparseCompute == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerSampleProbArraySparseCompute = _cyb_dlsym(handle, 'custabilizerSampleProbArraySparseCompute')

        global __custabilizerGF2SparseDenseMatrixMultiply
        __custabilizerGF2SparseDenseMatrixMultiply = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerGF2SparseDenseMatrixMultiply')
        if __custabilizerGF2SparseDenseMatrixMultiply == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerGF2SparseDenseMatrixMultiply = _cyb_dlsym(handle, 'custabilizerGF2SparseDenseMatrixMultiply')

        global __custabilizerGF2SparseSparseMatrixMultiply
        __custabilizerGF2SparseSparseMatrixMultiply = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerGF2SparseSparseMatrixMultiply')
        if __custabilizerGF2SparseSparseMatrixMultiply == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerGF2SparseSparseMatrixMultiply = _cyb_dlsym(handle, 'custabilizerGF2SparseSparseMatrixMultiply')

        global __custabilizerCircuitGetAttribute
        __custabilizerCircuitGetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerCircuitGetAttribute')
        if __custabilizerCircuitGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerCircuitGetAttribute = _cyb_dlsym(handle, 'custabilizerCircuitGetAttribute')

        global __custabilizerCreateLeakageFrameSimulator
        __custabilizerCreateLeakageFrameSimulator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerCreateLeakageFrameSimulator')
        if __custabilizerCreateLeakageFrameSimulator == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerCreateLeakageFrameSimulator = _cyb_dlsym(handle, 'custabilizerCreateLeakageFrameSimulator')

        global __custabilizerDestroyLeakageFrameSimulator
        __custabilizerDestroyLeakageFrameSimulator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerDestroyLeakageFrameSimulator')
        if __custabilizerDestroyLeakageFrameSimulator == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerDestroyLeakageFrameSimulator = _cyb_dlsym(handle, 'custabilizerDestroyLeakageFrameSimulator')

        global __custabilizerLeakageFrameSimulatorApplyCircuit
        __custabilizerLeakageFrameSimulatorApplyCircuit = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custabilizerLeakageFrameSimulatorApplyCircuit')
        if __custabilizerLeakageFrameSimulatorApplyCircuit == NULL:
            if handle == NULL:
                handle = load_library()
            __custabilizerLeakageFrameSimulatorApplyCircuit = _cyb_dlsym(handle, 'custabilizerLeakageFrameSimulatorApplyCircuit')

        _cyb_atomic_int_store(<int *>&_cyb___py_custabilizer_init, 1)
        return 0

cdef inline int _check_or_init_custabilizer() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_custabilizer_init):
        return 0

    return _init_custabilizer()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

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

    global __custabilizerCircuitGetAttribute
    data["__custabilizerCircuitGetAttribute"] = <intptr_t>__custabilizerCircuitGetAttribute

    global __custabilizerCreateLeakageFrameSimulator
    data["__custabilizerCreateLeakageFrameSimulator"] = <intptr_t>__custabilizerCreateLeakageFrameSimulator

    global __custabilizerDestroyLeakageFrameSimulator
    data["__custabilizerDestroyLeakageFrameSimulator"] = <intptr_t>__custabilizerDestroyLeakageFrameSimulator

    global __custabilizerLeakageFrameSimulatorApplyCircuit
    data["__custabilizerLeakageFrameSimulatorApplyCircuit"] = <intptr_t>__custabilizerLeakageFrameSimulatorApplyCircuit
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("custabilizer")._handle_uint
    return <void*>handle


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


cdef custabilizerStatus_t _custabilizerCircuitGetAttribute(const custabilizerHandle_t handle, const custabilizerCircuit_t circuit, custabilizerCircuitAttributes_t attribute, void* buffer, size_t sizeInBytes) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerCircuitGetAttribute
    _check_or_init_custabilizer()
    if __custabilizerCircuitGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerCircuitGetAttribute is not found")
    return (<custabilizerStatus_t (*)(const custabilizerHandle_t, const custabilizerCircuit_t, custabilizerCircuitAttributes_t, void*, size_t) noexcept nogil>__custabilizerCircuitGetAttribute)(
        handle, circuit, attribute, buffer, sizeInBytes)


cdef custabilizerStatus_t _custabilizerCreateLeakageFrameSimulator(const custabilizerHandle_t handle, int64_t numQubits, int64_t numShots, int64_t numMeasurements, int64_t tableStrideMajor, custabilizerLeakageFrameSimulator_t* leakageFrameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerCreateLeakageFrameSimulator
    _check_or_init_custabilizer()
    if __custabilizerCreateLeakageFrameSimulator == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerCreateLeakageFrameSimulator is not found")
    return (<custabilizerStatus_t (*)(const custabilizerHandle_t, int64_t, int64_t, int64_t, int64_t, custabilizerLeakageFrameSimulator_t*) noexcept nogil>__custabilizerCreateLeakageFrameSimulator)(
        handle, numQubits, numShots, numMeasurements, tableStrideMajor, leakageFrameSimulator)


cdef custabilizerStatus_t _custabilizerDestroyLeakageFrameSimulator(custabilizerLeakageFrameSimulator_t leakageFrameSimulator) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerDestroyLeakageFrameSimulator
    _check_or_init_custabilizer()
    if __custabilizerDestroyLeakageFrameSimulator == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerDestroyLeakageFrameSimulator is not found")
    return (<custabilizerStatus_t (*)(custabilizerLeakageFrameSimulator_t) noexcept nogil>__custabilizerDestroyLeakageFrameSimulator)(
        leakageFrameSimulator)


cdef custabilizerStatus_t _custabilizerLeakageFrameSimulatorApplyCircuit(const custabilizerHandle_t handle, custabilizerLeakageFrameSimulator_t leakageFrameSimulator, const custabilizerCircuit_t circuit, int randomizeFrameAfterMeasurement, uint64_t seed, custabilizerBitInt_t* xTableDevice, custabilizerBitInt_t* zTableDevice, custabilizerBitInt_t* lTableDevice, custabilizerBitInt_t* mTableDevice, cudaStream_t stream) except?_CUSTABILIZERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custabilizerLeakageFrameSimulatorApplyCircuit
    _check_or_init_custabilizer()
    if __custabilizerLeakageFrameSimulatorApplyCircuit == NULL:
        with gil:
            raise FunctionNotFoundError("function custabilizerLeakageFrameSimulatorApplyCircuit is not found")
    return (<custabilizerStatus_t (*)(const custabilizerHandle_t, custabilizerLeakageFrameSimulator_t, const custabilizerCircuit_t, int, uint64_t, custabilizerBitInt_t*, custabilizerBitInt_t*, custabilizerBitInt_t*, custabilizerBitInt_t*, cudaStream_t) noexcept nogil>__custabilizerLeakageFrameSimulatorApplyCircuit)(
        handle, leakageFrameSimulator, circuit, randomizeFrameAfterMeasurement, seed, xTableDevice, zTableDevice, lTableDevice, mTableDevice, stream)
