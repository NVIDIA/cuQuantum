# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.11.0. Do not modify it directly.

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

cdef int get_cuda_version():
    cdef void* handle = NULL
    cdef int err, driver_ver = 0

    # Load driver to check version
    handle = dlopen('libcuda.so.1', RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        err_msg = dlerror()
        raise NotSupportedError(f'CUDA driver is not found ({err_msg.decode()})')
    cuDriverGetVersion = dlsym(handle, "cuDriverGetVersion")
    if cuDriverGetVersion == NULL:
        raise RuntimeError('Did not find cuDriverGetVersion symbol in libcuda.so.1')
    err = (<int (*)(int*) noexcept nogil>cuDriverGetVersion)(&driver_ver)
    if err != 0:
        raise RuntimeError(f'cuDriverGetVersion returned error code {err}')

    return driver_ver


###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_cupauliprop_init = False

cdef void* __cupaulipropGetVersion = NULL
cdef void* __cupaulipropGetErrorString = NULL
cdef void* __cupaulipropGetNumPackedIntegers = NULL
cdef void* __cupaulipropCreate = NULL
cdef void* __cupaulipropDestroy = NULL
cdef void* __cupaulipropSetStream = NULL
cdef void* __cupaulipropCreateWorkspaceDescriptor = NULL
cdef void* __cupaulipropDestroyWorkspaceDescriptor = NULL
cdef void* __cupaulipropWorkspaceGetMemorySize = NULL
cdef void* __cupaulipropWorkspaceSetMemory = NULL
cdef void* __cupaulipropWorkspaceGetMemory = NULL
cdef void* __cupaulipropCreatePauliExpansion = NULL
cdef void* __cupaulipropDestroyPauliExpansion = NULL
cdef void* __cupaulipropPauliExpansionGetStorageBuffer = NULL
cdef void* __cupaulipropPauliExpansionGetNumQubits = NULL
cdef void* __cupaulipropPauliExpansionGetNumTerms = NULL
cdef void* __cupaulipropPauliExpansionGetDataType = NULL
cdef void* __cupaulipropPauliExpansionIsSorted = NULL
cdef void* __cupaulipropPauliExpansionIsDeduplicated = NULL
cdef void* __cupaulipropPauliExpansionGetTerm = NULL
cdef void* __cupaulipropPauliExpansionGetContiguousRange = NULL
cdef void* __cupaulipropDestroyPauliExpansionView = NULL
cdef void* __cupaulipropPauliExpansionViewGetNumTerms = NULL
cdef void* __cupaulipropPauliExpansionViewGetLocation = NULL
cdef void* __cupaulipropPauliExpansionViewGetTerm = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareDeduplication = NULL
cdef void* __cupaulipropPauliExpansionViewExecuteDeduplication = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareCanonicalSort = NULL
cdef void* __cupaulipropPauliExpansionViewExecuteCanonicalSort = NULL
cdef void* __cupaulipropPauliExpansionPopulateFromView = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView = NULL
cdef void* __cupaulipropPauliExpansionViewComputeTraceWithExpansionView = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareTraceWithZeroState = NULL
cdef void* __cupaulipropPauliExpansionViewComputeTraceWithZeroState = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareOperatorApplication = NULL
cdef void* __cupaulipropPauliExpansionViewComputeOperatorApplication = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareTruncation = NULL
cdef void* __cupaulipropPauliExpansionViewExecuteTruncation = NULL
cdef void* __cupaulipropCreateCliffordGateOperator = NULL
cdef void* __cupaulipropCreatePauliRotationGateOperator = NULL
cdef void* __cupaulipropCreatePauliNoiseChannelOperator = NULL
cdef void* __cupaulipropQuantumOperatorGetKind = NULL
cdef void* __cupaulipropDestroyOperator = NULL


cdef void* load_library() except* nogil:
    cdef void* handle
    handle = dlopen("libcupauliprop.so.0", RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        with gil:
            err_msg = dlerror()
            raise RuntimeError(f'Failed to dlopen libcupauliprop ({err_msg.decode()})')
    return handle


cdef int _check_or_init_cupauliprop() except -1 nogil:
    global __py_cupauliprop_init
    if __py_cupauliprop_init:
        return 0

    cdef void* handle = NULL
    with gil, __symbol_lock:
        # Load function
        global __cupaulipropGetVersion
        __cupaulipropGetVersion = dlsym(RTLD_DEFAULT, 'cupaulipropGetVersion')
        if __cupaulipropGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropGetVersion = dlsym(handle, 'cupaulipropGetVersion')

        global __cupaulipropGetErrorString
        __cupaulipropGetErrorString = dlsym(RTLD_DEFAULT, 'cupaulipropGetErrorString')
        if __cupaulipropGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropGetErrorString = dlsym(handle, 'cupaulipropGetErrorString')

        global __cupaulipropGetNumPackedIntegers
        __cupaulipropGetNumPackedIntegers = dlsym(RTLD_DEFAULT, 'cupaulipropGetNumPackedIntegers')
        if __cupaulipropGetNumPackedIntegers == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropGetNumPackedIntegers = dlsym(handle, 'cupaulipropGetNumPackedIntegers')

        global __cupaulipropCreate
        __cupaulipropCreate = dlsym(RTLD_DEFAULT, 'cupaulipropCreate')
        if __cupaulipropCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreate = dlsym(handle, 'cupaulipropCreate')

        global __cupaulipropDestroy
        __cupaulipropDestroy = dlsym(RTLD_DEFAULT, 'cupaulipropDestroy')
        if __cupaulipropDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropDestroy = dlsym(handle, 'cupaulipropDestroy')

        global __cupaulipropSetStream
        __cupaulipropSetStream = dlsym(RTLD_DEFAULT, 'cupaulipropSetStream')
        if __cupaulipropSetStream == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropSetStream = dlsym(handle, 'cupaulipropSetStream')

        global __cupaulipropCreateWorkspaceDescriptor
        __cupaulipropCreateWorkspaceDescriptor = dlsym(RTLD_DEFAULT, 'cupaulipropCreateWorkspaceDescriptor')
        if __cupaulipropCreateWorkspaceDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreateWorkspaceDescriptor = dlsym(handle, 'cupaulipropCreateWorkspaceDescriptor')

        global __cupaulipropDestroyWorkspaceDescriptor
        __cupaulipropDestroyWorkspaceDescriptor = dlsym(RTLD_DEFAULT, 'cupaulipropDestroyWorkspaceDescriptor')
        if __cupaulipropDestroyWorkspaceDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropDestroyWorkspaceDescriptor = dlsym(handle, 'cupaulipropDestroyWorkspaceDescriptor')

        global __cupaulipropWorkspaceGetMemorySize
        __cupaulipropWorkspaceGetMemorySize = dlsym(RTLD_DEFAULT, 'cupaulipropWorkspaceGetMemorySize')
        if __cupaulipropWorkspaceGetMemorySize == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropWorkspaceGetMemorySize = dlsym(handle, 'cupaulipropWorkspaceGetMemorySize')

        global __cupaulipropWorkspaceSetMemory
        __cupaulipropWorkspaceSetMemory = dlsym(RTLD_DEFAULT, 'cupaulipropWorkspaceSetMemory')
        if __cupaulipropWorkspaceSetMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropWorkspaceSetMemory = dlsym(handle, 'cupaulipropWorkspaceSetMemory')

        global __cupaulipropWorkspaceGetMemory
        __cupaulipropWorkspaceGetMemory = dlsym(RTLD_DEFAULT, 'cupaulipropWorkspaceGetMemory')
        if __cupaulipropWorkspaceGetMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropWorkspaceGetMemory = dlsym(handle, 'cupaulipropWorkspaceGetMemory')

        global __cupaulipropCreatePauliExpansion
        __cupaulipropCreatePauliExpansion = dlsym(RTLD_DEFAULT, 'cupaulipropCreatePauliExpansion')
        if __cupaulipropCreatePauliExpansion == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreatePauliExpansion = dlsym(handle, 'cupaulipropCreatePauliExpansion')

        global __cupaulipropDestroyPauliExpansion
        __cupaulipropDestroyPauliExpansion = dlsym(RTLD_DEFAULT, 'cupaulipropDestroyPauliExpansion')
        if __cupaulipropDestroyPauliExpansion == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropDestroyPauliExpansion = dlsym(handle, 'cupaulipropDestroyPauliExpansion')

        global __cupaulipropPauliExpansionGetStorageBuffer
        __cupaulipropPauliExpansionGetStorageBuffer = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionGetStorageBuffer')
        if __cupaulipropPauliExpansionGetStorageBuffer == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetStorageBuffer = dlsym(handle, 'cupaulipropPauliExpansionGetStorageBuffer')

        global __cupaulipropPauliExpansionGetNumQubits
        __cupaulipropPauliExpansionGetNumQubits = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionGetNumQubits')
        if __cupaulipropPauliExpansionGetNumQubits == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetNumQubits = dlsym(handle, 'cupaulipropPauliExpansionGetNumQubits')

        global __cupaulipropPauliExpansionGetNumTerms
        __cupaulipropPauliExpansionGetNumTerms = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionGetNumTerms')
        if __cupaulipropPauliExpansionGetNumTerms == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetNumTerms = dlsym(handle, 'cupaulipropPauliExpansionGetNumTerms')

        global __cupaulipropPauliExpansionGetDataType
        __cupaulipropPauliExpansionGetDataType = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionGetDataType')
        if __cupaulipropPauliExpansionGetDataType == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetDataType = dlsym(handle, 'cupaulipropPauliExpansionGetDataType')

        global __cupaulipropPauliExpansionIsSorted
        __cupaulipropPauliExpansionIsSorted = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionIsSorted')
        if __cupaulipropPauliExpansionIsSorted == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionIsSorted = dlsym(handle, 'cupaulipropPauliExpansionIsSorted')

        global __cupaulipropPauliExpansionIsDeduplicated
        __cupaulipropPauliExpansionIsDeduplicated = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionIsDeduplicated')
        if __cupaulipropPauliExpansionIsDeduplicated == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionIsDeduplicated = dlsym(handle, 'cupaulipropPauliExpansionIsDeduplicated')

        global __cupaulipropPauliExpansionGetTerm
        __cupaulipropPauliExpansionGetTerm = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionGetTerm')
        if __cupaulipropPauliExpansionGetTerm == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetTerm = dlsym(handle, 'cupaulipropPauliExpansionGetTerm')

        global __cupaulipropPauliExpansionGetContiguousRange
        __cupaulipropPauliExpansionGetContiguousRange = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionGetContiguousRange')
        if __cupaulipropPauliExpansionGetContiguousRange == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetContiguousRange = dlsym(handle, 'cupaulipropPauliExpansionGetContiguousRange')

        global __cupaulipropDestroyPauliExpansionView
        __cupaulipropDestroyPauliExpansionView = dlsym(RTLD_DEFAULT, 'cupaulipropDestroyPauliExpansionView')
        if __cupaulipropDestroyPauliExpansionView == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropDestroyPauliExpansionView = dlsym(handle, 'cupaulipropDestroyPauliExpansionView')

        global __cupaulipropPauliExpansionViewGetNumTerms
        __cupaulipropPauliExpansionViewGetNumTerms = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewGetNumTerms')
        if __cupaulipropPauliExpansionViewGetNumTerms == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewGetNumTerms = dlsym(handle, 'cupaulipropPauliExpansionViewGetNumTerms')

        global __cupaulipropPauliExpansionViewGetLocation
        __cupaulipropPauliExpansionViewGetLocation = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewGetLocation')
        if __cupaulipropPauliExpansionViewGetLocation == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewGetLocation = dlsym(handle, 'cupaulipropPauliExpansionViewGetLocation')

        global __cupaulipropPauliExpansionViewGetTerm
        __cupaulipropPauliExpansionViewGetTerm = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewGetTerm')
        if __cupaulipropPauliExpansionViewGetTerm == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewGetTerm = dlsym(handle, 'cupaulipropPauliExpansionViewGetTerm')

        global __cupaulipropPauliExpansionViewPrepareDeduplication
        __cupaulipropPauliExpansionViewPrepareDeduplication = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareDeduplication')
        if __cupaulipropPauliExpansionViewPrepareDeduplication == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareDeduplication = dlsym(handle, 'cupaulipropPauliExpansionViewPrepareDeduplication')

        global __cupaulipropPauliExpansionViewExecuteDeduplication
        __cupaulipropPauliExpansionViewExecuteDeduplication = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewExecuteDeduplication')
        if __cupaulipropPauliExpansionViewExecuteDeduplication == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewExecuteDeduplication = dlsym(handle, 'cupaulipropPauliExpansionViewExecuteDeduplication')

        global __cupaulipropPauliExpansionViewPrepareCanonicalSort
        __cupaulipropPauliExpansionViewPrepareCanonicalSort = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareCanonicalSort')
        if __cupaulipropPauliExpansionViewPrepareCanonicalSort == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareCanonicalSort = dlsym(handle, 'cupaulipropPauliExpansionViewPrepareCanonicalSort')

        global __cupaulipropPauliExpansionViewExecuteCanonicalSort
        __cupaulipropPauliExpansionViewExecuteCanonicalSort = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewExecuteCanonicalSort')
        if __cupaulipropPauliExpansionViewExecuteCanonicalSort == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewExecuteCanonicalSort = dlsym(handle, 'cupaulipropPauliExpansionViewExecuteCanonicalSort')

        global __cupaulipropPauliExpansionPopulateFromView
        __cupaulipropPauliExpansionPopulateFromView = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionPopulateFromView')
        if __cupaulipropPauliExpansionPopulateFromView == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionPopulateFromView = dlsym(handle, 'cupaulipropPauliExpansionPopulateFromView')

        global __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView
        __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareTraceWithExpansionView')
        if __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView = dlsym(handle, 'cupaulipropPauliExpansionViewPrepareTraceWithExpansionView')

        global __cupaulipropPauliExpansionViewComputeTraceWithExpansionView
        __cupaulipropPauliExpansionViewComputeTraceWithExpansionView = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewComputeTraceWithExpansionView')
        if __cupaulipropPauliExpansionViewComputeTraceWithExpansionView == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewComputeTraceWithExpansionView = dlsym(handle, 'cupaulipropPauliExpansionViewComputeTraceWithExpansionView')

        global __cupaulipropPauliExpansionViewPrepareTraceWithZeroState
        __cupaulipropPauliExpansionViewPrepareTraceWithZeroState = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareTraceWithZeroState')
        if __cupaulipropPauliExpansionViewPrepareTraceWithZeroState == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareTraceWithZeroState = dlsym(handle, 'cupaulipropPauliExpansionViewPrepareTraceWithZeroState')

        global __cupaulipropPauliExpansionViewComputeTraceWithZeroState
        __cupaulipropPauliExpansionViewComputeTraceWithZeroState = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewComputeTraceWithZeroState')
        if __cupaulipropPauliExpansionViewComputeTraceWithZeroState == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewComputeTraceWithZeroState = dlsym(handle, 'cupaulipropPauliExpansionViewComputeTraceWithZeroState')

        global __cupaulipropPauliExpansionViewPrepareOperatorApplication
        __cupaulipropPauliExpansionViewPrepareOperatorApplication = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareOperatorApplication')
        if __cupaulipropPauliExpansionViewPrepareOperatorApplication == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareOperatorApplication = dlsym(handle, 'cupaulipropPauliExpansionViewPrepareOperatorApplication')

        global __cupaulipropPauliExpansionViewComputeOperatorApplication
        __cupaulipropPauliExpansionViewComputeOperatorApplication = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewComputeOperatorApplication')
        if __cupaulipropPauliExpansionViewComputeOperatorApplication == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewComputeOperatorApplication = dlsym(handle, 'cupaulipropPauliExpansionViewComputeOperatorApplication')

        global __cupaulipropPauliExpansionViewPrepareTruncation
        __cupaulipropPauliExpansionViewPrepareTruncation = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareTruncation')
        if __cupaulipropPauliExpansionViewPrepareTruncation == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareTruncation = dlsym(handle, 'cupaulipropPauliExpansionViewPrepareTruncation')

        global __cupaulipropPauliExpansionViewExecuteTruncation
        __cupaulipropPauliExpansionViewExecuteTruncation = dlsym(RTLD_DEFAULT, 'cupaulipropPauliExpansionViewExecuteTruncation')
        if __cupaulipropPauliExpansionViewExecuteTruncation == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewExecuteTruncation = dlsym(handle, 'cupaulipropPauliExpansionViewExecuteTruncation')

        global __cupaulipropCreateCliffordGateOperator
        __cupaulipropCreateCliffordGateOperator = dlsym(RTLD_DEFAULT, 'cupaulipropCreateCliffordGateOperator')
        if __cupaulipropCreateCliffordGateOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreateCliffordGateOperator = dlsym(handle, 'cupaulipropCreateCliffordGateOperator')

        global __cupaulipropCreatePauliRotationGateOperator
        __cupaulipropCreatePauliRotationGateOperator = dlsym(RTLD_DEFAULT, 'cupaulipropCreatePauliRotationGateOperator')
        if __cupaulipropCreatePauliRotationGateOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreatePauliRotationGateOperator = dlsym(handle, 'cupaulipropCreatePauliRotationGateOperator')

        global __cupaulipropCreatePauliNoiseChannelOperator
        __cupaulipropCreatePauliNoiseChannelOperator = dlsym(RTLD_DEFAULT, 'cupaulipropCreatePauliNoiseChannelOperator')
        if __cupaulipropCreatePauliNoiseChannelOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreatePauliNoiseChannelOperator = dlsym(handle, 'cupaulipropCreatePauliNoiseChannelOperator')

        global __cupaulipropQuantumOperatorGetKind
        __cupaulipropQuantumOperatorGetKind = dlsym(RTLD_DEFAULT, 'cupaulipropQuantumOperatorGetKind')
        if __cupaulipropQuantumOperatorGetKind == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropQuantumOperatorGetKind = dlsym(handle, 'cupaulipropQuantumOperatorGetKind')

        global __cupaulipropDestroyOperator
        __cupaulipropDestroyOperator = dlsym(RTLD_DEFAULT, 'cupaulipropDestroyOperator')
        if __cupaulipropDestroyOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropDestroyOperator = dlsym(handle, 'cupaulipropDestroyOperator')
        __py_cupauliprop_init = True
        return 0


cpdef dict _inspect_function_pointers():
    _check_or_init_cupauliprop()
    cdef dict data = {}

    global __cupaulipropGetVersion
    data["__cupaulipropGetVersion"] = <intptr_t>__cupaulipropGetVersion

    global __cupaulipropGetErrorString
    data["__cupaulipropGetErrorString"] = <intptr_t>__cupaulipropGetErrorString

    global __cupaulipropGetNumPackedIntegers
    data["__cupaulipropGetNumPackedIntegers"] = <intptr_t>__cupaulipropGetNumPackedIntegers

    global __cupaulipropCreate
    data["__cupaulipropCreate"] = <intptr_t>__cupaulipropCreate

    global __cupaulipropDestroy
    data["__cupaulipropDestroy"] = <intptr_t>__cupaulipropDestroy

    global __cupaulipropSetStream
    data["__cupaulipropSetStream"] = <intptr_t>__cupaulipropSetStream

    global __cupaulipropCreateWorkspaceDescriptor
    data["__cupaulipropCreateWorkspaceDescriptor"] = <intptr_t>__cupaulipropCreateWorkspaceDescriptor

    global __cupaulipropDestroyWorkspaceDescriptor
    data["__cupaulipropDestroyWorkspaceDescriptor"] = <intptr_t>__cupaulipropDestroyWorkspaceDescriptor

    global __cupaulipropWorkspaceGetMemorySize
    data["__cupaulipropWorkspaceGetMemorySize"] = <intptr_t>__cupaulipropWorkspaceGetMemorySize

    global __cupaulipropWorkspaceSetMemory
    data["__cupaulipropWorkspaceSetMemory"] = <intptr_t>__cupaulipropWorkspaceSetMemory

    global __cupaulipropWorkspaceGetMemory
    data["__cupaulipropWorkspaceGetMemory"] = <intptr_t>__cupaulipropWorkspaceGetMemory

    global __cupaulipropCreatePauliExpansion
    data["__cupaulipropCreatePauliExpansion"] = <intptr_t>__cupaulipropCreatePauliExpansion

    global __cupaulipropDestroyPauliExpansion
    data["__cupaulipropDestroyPauliExpansion"] = <intptr_t>__cupaulipropDestroyPauliExpansion

    global __cupaulipropPauliExpansionGetStorageBuffer
    data["__cupaulipropPauliExpansionGetStorageBuffer"] = <intptr_t>__cupaulipropPauliExpansionGetStorageBuffer

    global __cupaulipropPauliExpansionGetNumQubits
    data["__cupaulipropPauliExpansionGetNumQubits"] = <intptr_t>__cupaulipropPauliExpansionGetNumQubits

    global __cupaulipropPauliExpansionGetNumTerms
    data["__cupaulipropPauliExpansionGetNumTerms"] = <intptr_t>__cupaulipropPauliExpansionGetNumTerms

    global __cupaulipropPauliExpansionGetDataType
    data["__cupaulipropPauliExpansionGetDataType"] = <intptr_t>__cupaulipropPauliExpansionGetDataType

    global __cupaulipropPauliExpansionIsSorted
    data["__cupaulipropPauliExpansionIsSorted"] = <intptr_t>__cupaulipropPauliExpansionIsSorted

    global __cupaulipropPauliExpansionIsDeduplicated
    data["__cupaulipropPauliExpansionIsDeduplicated"] = <intptr_t>__cupaulipropPauliExpansionIsDeduplicated

    global __cupaulipropPauliExpansionGetTerm
    data["__cupaulipropPauliExpansionGetTerm"] = <intptr_t>__cupaulipropPauliExpansionGetTerm

    global __cupaulipropPauliExpansionGetContiguousRange
    data["__cupaulipropPauliExpansionGetContiguousRange"] = <intptr_t>__cupaulipropPauliExpansionGetContiguousRange

    global __cupaulipropDestroyPauliExpansionView
    data["__cupaulipropDestroyPauliExpansionView"] = <intptr_t>__cupaulipropDestroyPauliExpansionView

    global __cupaulipropPauliExpansionViewGetNumTerms
    data["__cupaulipropPauliExpansionViewGetNumTerms"] = <intptr_t>__cupaulipropPauliExpansionViewGetNumTerms

    global __cupaulipropPauliExpansionViewGetLocation
    data["__cupaulipropPauliExpansionViewGetLocation"] = <intptr_t>__cupaulipropPauliExpansionViewGetLocation

    global __cupaulipropPauliExpansionViewGetTerm
    data["__cupaulipropPauliExpansionViewGetTerm"] = <intptr_t>__cupaulipropPauliExpansionViewGetTerm

    global __cupaulipropPauliExpansionViewPrepareDeduplication
    data["__cupaulipropPauliExpansionViewPrepareDeduplication"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareDeduplication

    global __cupaulipropPauliExpansionViewExecuteDeduplication
    data["__cupaulipropPauliExpansionViewExecuteDeduplication"] = <intptr_t>__cupaulipropPauliExpansionViewExecuteDeduplication

    global __cupaulipropPauliExpansionViewPrepareCanonicalSort
    data["__cupaulipropPauliExpansionViewPrepareCanonicalSort"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareCanonicalSort

    global __cupaulipropPauliExpansionViewExecuteCanonicalSort
    data["__cupaulipropPauliExpansionViewExecuteCanonicalSort"] = <intptr_t>__cupaulipropPauliExpansionViewExecuteCanonicalSort

    global __cupaulipropPauliExpansionPopulateFromView
    data["__cupaulipropPauliExpansionPopulateFromView"] = <intptr_t>__cupaulipropPauliExpansionPopulateFromView

    global __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView
    data["__cupaulipropPauliExpansionViewPrepareTraceWithExpansionView"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareTraceWithExpansionView

    global __cupaulipropPauliExpansionViewComputeTraceWithExpansionView
    data["__cupaulipropPauliExpansionViewComputeTraceWithExpansionView"] = <intptr_t>__cupaulipropPauliExpansionViewComputeTraceWithExpansionView

    global __cupaulipropPauliExpansionViewPrepareTraceWithZeroState
    data["__cupaulipropPauliExpansionViewPrepareTraceWithZeroState"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareTraceWithZeroState

    global __cupaulipropPauliExpansionViewComputeTraceWithZeroState
    data["__cupaulipropPauliExpansionViewComputeTraceWithZeroState"] = <intptr_t>__cupaulipropPauliExpansionViewComputeTraceWithZeroState

    global __cupaulipropPauliExpansionViewPrepareOperatorApplication
    data["__cupaulipropPauliExpansionViewPrepareOperatorApplication"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareOperatorApplication

    global __cupaulipropPauliExpansionViewComputeOperatorApplication
    data["__cupaulipropPauliExpansionViewComputeOperatorApplication"] = <intptr_t>__cupaulipropPauliExpansionViewComputeOperatorApplication

    global __cupaulipropPauliExpansionViewPrepareTruncation
    data["__cupaulipropPauliExpansionViewPrepareTruncation"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareTruncation

    global __cupaulipropPauliExpansionViewExecuteTruncation
    data["__cupaulipropPauliExpansionViewExecuteTruncation"] = <intptr_t>__cupaulipropPauliExpansionViewExecuteTruncation

    global __cupaulipropCreateCliffordGateOperator
    data["__cupaulipropCreateCliffordGateOperator"] = <intptr_t>__cupaulipropCreateCliffordGateOperator

    global __cupaulipropCreatePauliRotationGateOperator
    data["__cupaulipropCreatePauliRotationGateOperator"] = <intptr_t>__cupaulipropCreatePauliRotationGateOperator

    global __cupaulipropCreatePauliNoiseChannelOperator
    data["__cupaulipropCreatePauliNoiseChannelOperator"] = <intptr_t>__cupaulipropCreatePauliNoiseChannelOperator

    global __cupaulipropQuantumOperatorGetKind
    data["__cupaulipropQuantumOperatorGetKind"] = <intptr_t>__cupaulipropQuantumOperatorGetKind

    global __cupaulipropDestroyOperator
    data["__cupaulipropDestroyOperator"] = <intptr_t>__cupaulipropDestroyOperator

    return data


###############################################################################
# Wrapper functions
###############################################################################

cdef size_t _cupaulipropGetVersion() except?0 nogil:
    global __cupaulipropGetVersion
    _check_or_init_cupauliprop()
    if __cupaulipropGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropGetVersion is not found")
    return (<size_t (*)() noexcept nogil>__cupaulipropGetVersion)(
        )


cdef const char* _cupaulipropGetErrorString(cupaulipropStatus_t error) except?NULL nogil:
    global __cupaulipropGetErrorString
    _check_or_init_cupauliprop()
    if __cupaulipropGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropGetErrorString is not found")
    return (<const char* (*)(cupaulipropStatus_t) noexcept nogil>__cupaulipropGetErrorString)(
        error)


cdef cupaulipropStatus_t _cupaulipropGetNumPackedIntegers(int32_t numQubits, int32_t* numPackedIntegers) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropGetNumPackedIntegers
    _check_or_init_cupauliprop()
    if __cupaulipropGetNumPackedIntegers == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropGetNumPackedIntegers is not found")
    return (<cupaulipropStatus_t (*)(int32_t, int32_t*) noexcept nogil>__cupaulipropGetNumPackedIntegers)(
        numQubits, numPackedIntegers)


cdef cupaulipropStatus_t _cupaulipropCreate(cupaulipropHandle_t* handle) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropCreate
    _check_or_init_cupauliprop()
    if __cupaulipropCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropCreate is not found")
    return (<cupaulipropStatus_t (*)(cupaulipropHandle_t*) noexcept nogil>__cupaulipropCreate)(
        handle)


cdef cupaulipropStatus_t _cupaulipropDestroy(cupaulipropHandle_t handle) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropDestroy
    _check_or_init_cupauliprop()
    if __cupaulipropDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropDestroy is not found")
    return (<cupaulipropStatus_t (*)(cupaulipropHandle_t) noexcept nogil>__cupaulipropDestroy)(
        handle)


cdef cupaulipropStatus_t _cupaulipropSetStream(cupaulipropHandle_t handle, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropSetStream
    _check_or_init_cupauliprop()
    if __cupaulipropSetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropSetStream is not found")
    return (<cupaulipropStatus_t (*)(cupaulipropHandle_t, cudaStream_t) noexcept nogil>__cupaulipropSetStream)(
        handle, stream)


cdef cupaulipropStatus_t _cupaulipropCreateWorkspaceDescriptor(cupaulipropHandle_t handle, cupaulipropWorkspaceDescriptor_t* workspaceDesc) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropCreateWorkspaceDescriptor
    _check_or_init_cupauliprop()
    if __cupaulipropCreateWorkspaceDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropCreateWorkspaceDescriptor is not found")
    return (<cupaulipropStatus_t (*)(cupaulipropHandle_t, cupaulipropWorkspaceDescriptor_t*) noexcept nogil>__cupaulipropCreateWorkspaceDescriptor)(
        handle, workspaceDesc)


cdef cupaulipropStatus_t _cupaulipropDestroyWorkspaceDescriptor(cupaulipropWorkspaceDescriptor_t workspaceDesc) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropDestroyWorkspaceDescriptor
    _check_or_init_cupauliprop()
    if __cupaulipropDestroyWorkspaceDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropDestroyWorkspaceDescriptor is not found")
    return (<cupaulipropStatus_t (*)(cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropDestroyWorkspaceDescriptor)(
        workspaceDesc)


cdef cupaulipropStatus_t _cupaulipropWorkspaceGetMemorySize(const cupaulipropHandle_t handle, const cupaulipropWorkspaceDescriptor_t workspaceDesc, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, int64_t* memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropWorkspaceGetMemorySize
    _check_or_init_cupauliprop()
    if __cupaulipropWorkspaceGetMemorySize == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropWorkspaceGetMemorySize is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropWorkspaceDescriptor_t, cupaulipropMemspace_t, cupaulipropWorkspaceKind_t, int64_t*) noexcept nogil>__cupaulipropWorkspaceGetMemorySize)(
        handle, workspaceDesc, memSpace, workspaceKind, memoryBufferSize)


cdef cupaulipropStatus_t _cupaulipropWorkspaceSetMemory(const cupaulipropHandle_t handle, cupaulipropWorkspaceDescriptor_t workspaceDesc, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, void* memoryBuffer, int64_t memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropWorkspaceSetMemory
    _check_or_init_cupauliprop()
    if __cupaulipropWorkspaceSetMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropWorkspaceSetMemory is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, cupaulipropWorkspaceDescriptor_t, cupaulipropMemspace_t, cupaulipropWorkspaceKind_t, void*, int64_t) noexcept nogil>__cupaulipropWorkspaceSetMemory)(
        handle, workspaceDesc, memSpace, workspaceKind, memoryBuffer, memoryBufferSize)


cdef cupaulipropStatus_t _cupaulipropWorkspaceGetMemory(const cupaulipropHandle_t handle, const cupaulipropWorkspaceDescriptor_t workspaceDescr, cupaulipropMemspace_t memSpace, cupaulipropWorkspaceKind_t workspaceKind, void** memoryBuffer, int64_t* memoryBufferSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropWorkspaceGetMemory
    _check_or_init_cupauliprop()
    if __cupaulipropWorkspaceGetMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropWorkspaceGetMemory is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropWorkspaceDescriptor_t, cupaulipropMemspace_t, cupaulipropWorkspaceKind_t, void**, int64_t*) noexcept nogil>__cupaulipropWorkspaceGetMemory)(
        handle, workspaceDescr, memSpace, workspaceKind, memoryBuffer, memoryBufferSize)


cdef cupaulipropStatus_t _cupaulipropCreatePauliExpansion(const cupaulipropHandle_t handle, int32_t numQubits, void* xzBitsBuffer, int64_t xzBitsBufferSize, void* coefBuffer, int64_t coefBufferSize, cudaDataType_t dataType, int64_t numTerms, int32_t isSorted, int32_t hasDuplicates, cupaulipropPauliExpansion_t* pauliExpansion) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropCreatePauliExpansion
    _check_or_init_cupauliprop()
    if __cupaulipropCreatePauliExpansion == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropCreatePauliExpansion is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, int32_t, void*, int64_t, void*, int64_t, cudaDataType_t, int64_t, int32_t, int32_t, cupaulipropPauliExpansion_t*) noexcept nogil>__cupaulipropCreatePauliExpansion)(
        handle, numQubits, xzBitsBuffer, xzBitsBufferSize, coefBuffer, coefBufferSize, dataType, numTerms, isSorted, hasDuplicates, pauliExpansion)


cdef cupaulipropStatus_t _cupaulipropDestroyPauliExpansion(cupaulipropPauliExpansion_t pauliExpansion) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropDestroyPauliExpansion
    _check_or_init_cupauliprop()
    if __cupaulipropDestroyPauliExpansion == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropDestroyPauliExpansion is not found")
    return (<cupaulipropStatus_t (*)(cupaulipropPauliExpansion_t) noexcept nogil>__cupaulipropDestroyPauliExpansion)(
        pauliExpansion)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionGetStorageBuffer(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, void** xzBitsBuffer, int64_t* xzBitsBufferSize, void** coefBuffer, int64_t* coefBufferSize, int64_t* numTerms, cupaulipropMemspace_t* location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionGetStorageBuffer
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionGetStorageBuffer == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionGetStorageBuffer is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, void**, int64_t*, void**, int64_t*, int64_t*, cupaulipropMemspace_t*) noexcept nogil>__cupaulipropPauliExpansionGetStorageBuffer)(
        handle, pauliExpansion, xzBitsBuffer, xzBitsBufferSize, coefBuffer, coefBufferSize, numTerms, location)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionGetNumQubits(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* numQubits) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionGetNumQubits
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionGetNumQubits == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionGetNumQubits is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, int32_t*) noexcept nogil>__cupaulipropPauliExpansionGetNumQubits)(
        handle, pauliExpansion, numQubits)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionGetNumTerms(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t* numTerms) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionGetNumTerms
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionGetNumTerms == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionGetNumTerms is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, int64_t*) noexcept nogil>__cupaulipropPauliExpansionGetNumTerms)(
        handle, pauliExpansion, numTerms)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionGetDataType(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, cudaDataType_t* dataType) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionGetDataType
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionGetDataType == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionGetDataType is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, cudaDataType_t*) noexcept nogil>__cupaulipropPauliExpansionGetDataType)(
        handle, pauliExpansion, dataType)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionIsSorted(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* isSorted) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionIsSorted
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionIsSorted == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionIsSorted is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, int32_t*) noexcept nogil>__cupaulipropPauliExpansionIsSorted)(
        handle, pauliExpansion, isSorted)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionIsDeduplicated(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* isDeduplicated) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionIsDeduplicated
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionIsDeduplicated == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionIsDeduplicated is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, int32_t*) noexcept nogil>__cupaulipropPauliExpansionIsDeduplicated)(
        handle, pauliExpansion, isDeduplicated)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionGetTerm(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t termIndex, cupaulipropPauliTerm_t* term) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionGetTerm
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionGetTerm == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionGetTerm is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, int64_t, cupaulipropPauliTerm_t*) noexcept nogil>__cupaulipropPauliExpansionGetTerm)(
        handle, pauliExpansion, termIndex, term)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionGetContiguousRange(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t startIndex, int64_t endIndex, cupaulipropPauliExpansionView_t* view) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionGetContiguousRange
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionGetContiguousRange == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionGetContiguousRange is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, int64_t, int64_t, cupaulipropPauliExpansionView_t*) noexcept nogil>__cupaulipropPauliExpansionGetContiguousRange)(
        handle, pauliExpansion, startIndex, endIndex, view)


cdef cupaulipropStatus_t _cupaulipropDestroyPauliExpansionView(cupaulipropPauliExpansionView_t view) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropDestroyPauliExpansionView
    _check_or_init_cupauliprop()
    if __cupaulipropDestroyPauliExpansionView == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropDestroyPauliExpansionView is not found")
    return (<cupaulipropStatus_t (*)(cupaulipropPauliExpansionView_t) noexcept nogil>__cupaulipropDestroyPauliExpansionView)(
        view)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewGetNumTerms(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t* numTerms) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewGetNumTerms
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewGetNumTerms == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewGetNumTerms is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, int64_t*) noexcept nogil>__cupaulipropPauliExpansionViewGetNumTerms)(
        handle, view, numTerms)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewGetLocation(const cupaulipropPauliExpansionView_t view, cupaulipropMemspace_t* location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewGetLocation
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewGetLocation == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewGetLocation is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropPauliExpansionView_t, cupaulipropMemspace_t*) noexcept nogil>__cupaulipropPauliExpansionViewGetLocation)(
        view, location)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewGetTerm(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t termIndex, cupaulipropPauliTerm_t* term) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewGetTerm
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewGetTerm == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewGetTerm is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, int64_t, cupaulipropPauliTerm_t*) noexcept nogil>__cupaulipropPauliExpansionViewGetTerm)(
        handle, view, termIndex, term)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareDeduplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int32_t makeSorted, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareDeduplication
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareDeduplication == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareDeduplication is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, int32_t, int64_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareDeduplication)(
        handle, viewIn, makeSorted, maxWorkspaceSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewExecuteDeduplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, int32_t makeSorted, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewExecuteDeduplication
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewExecuteDeduplication == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewExecuteDeduplication is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t, int32_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewExecuteDeduplication)(
        handle, viewIn, expansionOut, makeSorted, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareCanonicalSort(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareCanonicalSort
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareCanonicalSort == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareCanonicalSort is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, int64_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareCanonicalSort)(
        handle, viewIn, maxWorkspaceSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewExecuteCanonicalSort(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewExecuteCanonicalSort
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewExecuteCanonicalSort == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewExecuteCanonicalSort is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewExecuteCanonicalSort)(
        handle, viewIn, expansionOut, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionPopulateFromView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionPopulateFromView
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionPopulateFromView == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionPopulateFromView is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t) noexcept nogil>__cupaulipropPauliExpansionPopulateFromView)(
        handle, viewIn, expansionOut)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareTraceWithExpansionView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareTraceWithExpansionView is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, const cupaulipropPauliExpansionView_t, int64_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareTraceWithExpansionView)(
        handle, view1, view2, maxWorkspaceSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewComputeTraceWithExpansionView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int32_t takeAdjoint1, void* trace, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewComputeTraceWithExpansionView
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewComputeTraceWithExpansionView == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewComputeTraceWithExpansionView is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, const cupaulipropPauliExpansionView_t, int32_t, void*, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewComputeTraceWithExpansionView)(
        handle, view1, view2, takeAdjoint1, trace, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareTraceWithZeroState(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareTraceWithZeroState
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareTraceWithZeroState == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareTraceWithZeroState is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, int64_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareTraceWithZeroState)(
        handle, view, maxWorkspaceSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewComputeTraceWithZeroState(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, void* trace, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewComputeTraceWithZeroState
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewComputeTraceWithZeroState == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewComputeTraceWithZeroState is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, void*, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewComputeTraceWithZeroState)(
        handle, view, trace, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareOperatorApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, const cupaulipropQuantumOperator_t quantumOperator, int32_t makeSorted, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceSize, int64_t* requiredXZBitsBufferSize, int64_t* requiredCoefBufferSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareOperatorApplication
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareOperatorApplication == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareOperatorApplication is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, const cupaulipropQuantumOperator_t, int32_t, int32_t, int32_t, const cupaulipropTruncationStrategy_t*, int64_t, int64_t*, int64_t*, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareOperatorApplication)(
        handle, viewIn, quantumOperator, makeSorted, keepDuplicates, numTruncationStrategies, truncationStrategies, maxWorkspaceSize, requiredXZBitsBufferSize, requiredCoefBufferSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewComputeOperatorApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, const cupaulipropQuantumOperator_t quantumOperator, int32_t adjoint, int32_t makeSorted, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewComputeOperatorApplication
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewComputeOperatorApplication == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewComputeOperatorApplication is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t, const cupaulipropQuantumOperator_t, int32_t, int32_t, int32_t, int32_t, const cupaulipropTruncationStrategy_t*, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewComputeOperatorApplication)(
        handle, viewIn, expansionOut, quantumOperator, adjoint, makeSorted, keepDuplicates, numTruncationStrategies, truncationStrategies, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareTruncation(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareTruncation
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareTruncation == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareTruncation is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, int32_t, const cupaulipropTruncationStrategy_t*, int64_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareTruncation)(
        handle, viewIn, numTruncationStrategies, truncationStrategies, maxWorkspaceSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewExecuteTruncation(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewExecuteTruncation
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewExecuteTruncation == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewExecuteTruncation is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t, int32_t, const cupaulipropTruncationStrategy_t*, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewExecuteTruncation)(
        handle, viewIn, expansionOut, numTruncationStrategies, truncationStrategies, workspace)


cdef cupaulipropStatus_t _cupaulipropCreateCliffordGateOperator(const cupaulipropHandle_t handle, cupaulipropCliffordGateKind_t cliffordGateKind, const int32_t qubitIndices[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropCreateCliffordGateOperator
    _check_or_init_cupauliprop()
    if __cupaulipropCreateCliffordGateOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropCreateCliffordGateOperator is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, cupaulipropCliffordGateKind_t, const int32_t*, cupaulipropQuantumOperator_t*) noexcept nogil>__cupaulipropCreateCliffordGateOperator)(
        handle, cliffordGateKind, qubitIndices, oper)


cdef cupaulipropStatus_t _cupaulipropCreatePauliRotationGateOperator(const cupaulipropHandle_t handle, double angle, int32_t numQubits, const int32_t qubitIndices[], const cupaulipropPauliKind_t paulis[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropCreatePauliRotationGateOperator
    _check_or_init_cupauliprop()
    if __cupaulipropCreatePauliRotationGateOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropCreatePauliRotationGateOperator is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, double, int32_t, const int32_t*, const cupaulipropPauliKind_t*, cupaulipropQuantumOperator_t*) noexcept nogil>__cupaulipropCreatePauliRotationGateOperator)(
        handle, angle, numQubits, qubitIndices, paulis, oper)


cdef cupaulipropStatus_t _cupaulipropCreatePauliNoiseChannelOperator(const cupaulipropHandle_t handle, int32_t numQubits, const int32_t qubitIndices[], const double probabilities[], cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropCreatePauliNoiseChannelOperator
    _check_or_init_cupauliprop()
    if __cupaulipropCreatePauliNoiseChannelOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropCreatePauliNoiseChannelOperator is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, int32_t, const int32_t*, const double*, cupaulipropQuantumOperator_t*) noexcept nogil>__cupaulipropCreatePauliNoiseChannelOperator)(
        handle, numQubits, qubitIndices, probabilities, oper)


cdef cupaulipropStatus_t _cupaulipropQuantumOperatorGetKind(const cupaulipropHandle_t handle, const cupaulipropQuantumOperator_t oper, cupaulipropQuantumOperatorKind_t* kind) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropQuantumOperatorGetKind
    _check_or_init_cupauliprop()
    if __cupaulipropQuantumOperatorGetKind == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropQuantumOperatorGetKind is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropQuantumOperator_t, cupaulipropQuantumOperatorKind_t*) noexcept nogil>__cupaulipropQuantumOperatorGetKind)(
        handle, oper, kind)


cdef cupaulipropStatus_t _cupaulipropDestroyOperator(cupaulipropQuantumOperator_t oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropDestroyOperator
    _check_or_init_cupauliprop()
    if __cupaulipropDestroyOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropDestroyOperator is not found")
    return (<cupaulipropStatus_t (*)(cupaulipropQuantumOperator_t) noexcept nogil>__cupaulipropDestroyOperator)(
        oper)
