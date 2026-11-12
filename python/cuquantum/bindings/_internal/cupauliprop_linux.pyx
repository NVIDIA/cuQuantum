# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 26.09.0 to 26.11.0. Do not modify it directly.



# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    int32_t,
    int64_t,
    intptr_t,
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

cdef int _cyb___py_cupauliprop_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .._utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################


cdef void* __cupaulipropGetVersion = NULL
cdef void* __cupaulipropGetErrorString = NULL
cdef void* __cupaulipropGetNumPackedIntegers = NULL
cdef void* __cupaulipropCreate = NULL
cdef void* __cupaulipropDestroy = NULL
cdef void* __cupaulipropResetDistributedConfiguration = NULL
cdef void* __cupaulipropGetNumRanks = NULL
cdef void* __cupaulipropGetProcRank = NULL
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
cdef void* __cupaulipropPauliExpansionGetSortOrder = NULL
cdef void* __cupaulipropPauliExpansionIsDeduplicated = NULL
cdef void* __cupaulipropPauliExpansionGetTerm = NULL
cdef void* __cupaulipropPauliExpansionGetContiguousRange = NULL
cdef void* __cupaulipropDestroyPauliExpansionView = NULL
cdef void* __cupaulipropPauliExpansionViewGetNumTerms = NULL
cdef void* __cupaulipropPauliExpansionViewGetLocation = NULL
cdef void* __cupaulipropPauliExpansionViewGetTerm = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareDeduplication = NULL
cdef void* __cupaulipropPauliExpansionViewExecuteDeduplication = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareSort = NULL
cdef void* __cupaulipropPauliExpansionViewExecuteSort = NULL
cdef void* __cupaulipropPauliExpansionPopulateFromView = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView = NULL
cdef void* __cupaulipropPauliExpansionViewComputeTraceWithExpansionView = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff = NULL
cdef void* __cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareTraceWithZeroState = NULL
cdef void* __cupaulipropPauliExpansionViewComputeTraceWithZeroState = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff = NULL
cdef void* __cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareOperatorApplication = NULL
cdef void* __cupaulipropPauliExpansionViewComputeOperatorApplication = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareOperatorFusedApplication = NULL
cdef void* __cupaulipropPauliExpansionViewComputeOperatorFusedApplication = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff = NULL
cdef void* __cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff = NULL
cdef void* __cupaulipropPauliExpansionViewPrepareTruncation = NULL
cdef void* __cupaulipropPauliExpansionViewExecuteTruncation = NULL
cdef void* __cupaulipropCreateCliffordGateOperator = NULL
cdef void* __cupaulipropCreatePauliRotationGateOperator = NULL
cdef void* __cupaulipropCreatePauliNoiseChannelOperator = NULL
cdef void* __cupaulipropCreateAmplitudeDampingChannelOperator = NULL
cdef void* __cupaulipropQuantumOperatorAttachCotangentBuffer = NULL
cdef void* __cupaulipropQuantumOperatorGetCotangentBuffer = NULL
cdef void* __cupaulipropDestroyOperator = NULL

cdef int _init_cupauliprop() except -1 nogil:
    global _cyb___py_cupauliprop_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_cupauliprop_init: return 0

        global __cupaulipropGetVersion
        __cupaulipropGetVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropGetVersion')
        if __cupaulipropGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropGetVersion = _cyb_dlsym(handle, 'cupaulipropGetVersion')

        global __cupaulipropGetErrorString
        __cupaulipropGetErrorString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropGetErrorString')
        if __cupaulipropGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropGetErrorString = _cyb_dlsym(handle, 'cupaulipropGetErrorString')

        global __cupaulipropGetNumPackedIntegers
        __cupaulipropGetNumPackedIntegers = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropGetNumPackedIntegers')
        if __cupaulipropGetNumPackedIntegers == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropGetNumPackedIntegers = _cyb_dlsym(handle, 'cupaulipropGetNumPackedIntegers')

        global __cupaulipropCreate
        __cupaulipropCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropCreate')
        if __cupaulipropCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreate = _cyb_dlsym(handle, 'cupaulipropCreate')

        global __cupaulipropDestroy
        __cupaulipropDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropDestroy')
        if __cupaulipropDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropDestroy = _cyb_dlsym(handle, 'cupaulipropDestroy')

        global __cupaulipropResetDistributedConfiguration
        __cupaulipropResetDistributedConfiguration = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropResetDistributedConfiguration')
        if __cupaulipropResetDistributedConfiguration == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropResetDistributedConfiguration = _cyb_dlsym(handle, 'cupaulipropResetDistributedConfiguration')

        global __cupaulipropGetNumRanks
        __cupaulipropGetNumRanks = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropGetNumRanks')
        if __cupaulipropGetNumRanks == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropGetNumRanks = _cyb_dlsym(handle, 'cupaulipropGetNumRanks')

        global __cupaulipropGetProcRank
        __cupaulipropGetProcRank = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropGetProcRank')
        if __cupaulipropGetProcRank == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropGetProcRank = _cyb_dlsym(handle, 'cupaulipropGetProcRank')

        global __cupaulipropCreateWorkspaceDescriptor
        __cupaulipropCreateWorkspaceDescriptor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropCreateWorkspaceDescriptor')
        if __cupaulipropCreateWorkspaceDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreateWorkspaceDescriptor = _cyb_dlsym(handle, 'cupaulipropCreateWorkspaceDescriptor')

        global __cupaulipropDestroyWorkspaceDescriptor
        __cupaulipropDestroyWorkspaceDescriptor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropDestroyWorkspaceDescriptor')
        if __cupaulipropDestroyWorkspaceDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropDestroyWorkspaceDescriptor = _cyb_dlsym(handle, 'cupaulipropDestroyWorkspaceDescriptor')

        global __cupaulipropWorkspaceGetMemorySize
        __cupaulipropWorkspaceGetMemorySize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropWorkspaceGetMemorySize')
        if __cupaulipropWorkspaceGetMemorySize == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropWorkspaceGetMemorySize = _cyb_dlsym(handle, 'cupaulipropWorkspaceGetMemorySize')

        global __cupaulipropWorkspaceSetMemory
        __cupaulipropWorkspaceSetMemory = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropWorkspaceSetMemory')
        if __cupaulipropWorkspaceSetMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropWorkspaceSetMemory = _cyb_dlsym(handle, 'cupaulipropWorkspaceSetMemory')

        global __cupaulipropWorkspaceGetMemory
        __cupaulipropWorkspaceGetMemory = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropWorkspaceGetMemory')
        if __cupaulipropWorkspaceGetMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropWorkspaceGetMemory = _cyb_dlsym(handle, 'cupaulipropWorkspaceGetMemory')

        global __cupaulipropCreatePauliExpansion
        __cupaulipropCreatePauliExpansion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropCreatePauliExpansion')
        if __cupaulipropCreatePauliExpansion == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreatePauliExpansion = _cyb_dlsym(handle, 'cupaulipropCreatePauliExpansion')

        global __cupaulipropDestroyPauliExpansion
        __cupaulipropDestroyPauliExpansion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropDestroyPauliExpansion')
        if __cupaulipropDestroyPauliExpansion == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropDestroyPauliExpansion = _cyb_dlsym(handle, 'cupaulipropDestroyPauliExpansion')

        global __cupaulipropPauliExpansionGetStorageBuffer
        __cupaulipropPauliExpansionGetStorageBuffer = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionGetStorageBuffer')
        if __cupaulipropPauliExpansionGetStorageBuffer == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetStorageBuffer = _cyb_dlsym(handle, 'cupaulipropPauliExpansionGetStorageBuffer')

        global __cupaulipropPauliExpansionGetNumQubits
        __cupaulipropPauliExpansionGetNumQubits = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionGetNumQubits')
        if __cupaulipropPauliExpansionGetNumQubits == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetNumQubits = _cyb_dlsym(handle, 'cupaulipropPauliExpansionGetNumQubits')

        global __cupaulipropPauliExpansionGetNumTerms
        __cupaulipropPauliExpansionGetNumTerms = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionGetNumTerms')
        if __cupaulipropPauliExpansionGetNumTerms == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetNumTerms = _cyb_dlsym(handle, 'cupaulipropPauliExpansionGetNumTerms')

        global __cupaulipropPauliExpansionGetDataType
        __cupaulipropPauliExpansionGetDataType = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionGetDataType')
        if __cupaulipropPauliExpansionGetDataType == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetDataType = _cyb_dlsym(handle, 'cupaulipropPauliExpansionGetDataType')

        global __cupaulipropPauliExpansionGetSortOrder
        __cupaulipropPauliExpansionGetSortOrder = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionGetSortOrder')
        if __cupaulipropPauliExpansionGetSortOrder == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetSortOrder = _cyb_dlsym(handle, 'cupaulipropPauliExpansionGetSortOrder')

        global __cupaulipropPauliExpansionIsDeduplicated
        __cupaulipropPauliExpansionIsDeduplicated = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionIsDeduplicated')
        if __cupaulipropPauliExpansionIsDeduplicated == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionIsDeduplicated = _cyb_dlsym(handle, 'cupaulipropPauliExpansionIsDeduplicated')

        global __cupaulipropPauliExpansionGetTerm
        __cupaulipropPauliExpansionGetTerm = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionGetTerm')
        if __cupaulipropPauliExpansionGetTerm == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetTerm = _cyb_dlsym(handle, 'cupaulipropPauliExpansionGetTerm')

        global __cupaulipropPauliExpansionGetContiguousRange
        __cupaulipropPauliExpansionGetContiguousRange = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionGetContiguousRange')
        if __cupaulipropPauliExpansionGetContiguousRange == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionGetContiguousRange = _cyb_dlsym(handle, 'cupaulipropPauliExpansionGetContiguousRange')

        global __cupaulipropDestroyPauliExpansionView
        __cupaulipropDestroyPauliExpansionView = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropDestroyPauliExpansionView')
        if __cupaulipropDestroyPauliExpansionView == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropDestroyPauliExpansionView = _cyb_dlsym(handle, 'cupaulipropDestroyPauliExpansionView')

        global __cupaulipropPauliExpansionViewGetNumTerms
        __cupaulipropPauliExpansionViewGetNumTerms = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewGetNumTerms')
        if __cupaulipropPauliExpansionViewGetNumTerms == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewGetNumTerms = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewGetNumTerms')

        global __cupaulipropPauliExpansionViewGetLocation
        __cupaulipropPauliExpansionViewGetLocation = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewGetLocation')
        if __cupaulipropPauliExpansionViewGetLocation == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewGetLocation = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewGetLocation')

        global __cupaulipropPauliExpansionViewGetTerm
        __cupaulipropPauliExpansionViewGetTerm = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewGetTerm')
        if __cupaulipropPauliExpansionViewGetTerm == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewGetTerm = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewGetTerm')

        global __cupaulipropPauliExpansionViewPrepareDeduplication
        __cupaulipropPauliExpansionViewPrepareDeduplication = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareDeduplication')
        if __cupaulipropPauliExpansionViewPrepareDeduplication == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareDeduplication = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewPrepareDeduplication')

        global __cupaulipropPauliExpansionViewExecuteDeduplication
        __cupaulipropPauliExpansionViewExecuteDeduplication = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewExecuteDeduplication')
        if __cupaulipropPauliExpansionViewExecuteDeduplication == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewExecuteDeduplication = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewExecuteDeduplication')

        global __cupaulipropPauliExpansionViewPrepareSort
        __cupaulipropPauliExpansionViewPrepareSort = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareSort')
        if __cupaulipropPauliExpansionViewPrepareSort == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareSort = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewPrepareSort')

        global __cupaulipropPauliExpansionViewExecuteSort
        __cupaulipropPauliExpansionViewExecuteSort = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewExecuteSort')
        if __cupaulipropPauliExpansionViewExecuteSort == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewExecuteSort = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewExecuteSort')

        global __cupaulipropPauliExpansionPopulateFromView
        __cupaulipropPauliExpansionPopulateFromView = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionPopulateFromView')
        if __cupaulipropPauliExpansionPopulateFromView == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionPopulateFromView = _cyb_dlsym(handle, 'cupaulipropPauliExpansionPopulateFromView')

        global __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView
        __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareTraceWithExpansionView')
        if __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewPrepareTraceWithExpansionView')

        global __cupaulipropPauliExpansionViewComputeTraceWithExpansionView
        __cupaulipropPauliExpansionViewComputeTraceWithExpansionView = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewComputeTraceWithExpansionView')
        if __cupaulipropPauliExpansionViewComputeTraceWithExpansionView == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewComputeTraceWithExpansionView = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewComputeTraceWithExpansionView')

        global __cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff
        __cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff')
        if __cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff')

        global __cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff
        __cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff')
        if __cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff')

        global __cupaulipropPauliExpansionViewPrepareTraceWithZeroState
        __cupaulipropPauliExpansionViewPrepareTraceWithZeroState = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareTraceWithZeroState')
        if __cupaulipropPauliExpansionViewPrepareTraceWithZeroState == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareTraceWithZeroState = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewPrepareTraceWithZeroState')

        global __cupaulipropPauliExpansionViewComputeTraceWithZeroState
        __cupaulipropPauliExpansionViewComputeTraceWithZeroState = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewComputeTraceWithZeroState')
        if __cupaulipropPauliExpansionViewComputeTraceWithZeroState == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewComputeTraceWithZeroState = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewComputeTraceWithZeroState')

        global __cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff
        __cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff')
        if __cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff')

        global __cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff
        __cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff')
        if __cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff')

        global __cupaulipropPauliExpansionViewPrepareOperatorApplication
        __cupaulipropPauliExpansionViewPrepareOperatorApplication = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareOperatorApplication')
        if __cupaulipropPauliExpansionViewPrepareOperatorApplication == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareOperatorApplication = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewPrepareOperatorApplication')

        global __cupaulipropPauliExpansionViewComputeOperatorApplication
        __cupaulipropPauliExpansionViewComputeOperatorApplication = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewComputeOperatorApplication')
        if __cupaulipropPauliExpansionViewComputeOperatorApplication == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewComputeOperatorApplication = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewComputeOperatorApplication')

        global __cupaulipropPauliExpansionViewPrepareOperatorFusedApplication
        __cupaulipropPauliExpansionViewPrepareOperatorFusedApplication = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareOperatorFusedApplication')
        if __cupaulipropPauliExpansionViewPrepareOperatorFusedApplication == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareOperatorFusedApplication = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewPrepareOperatorFusedApplication')

        global __cupaulipropPauliExpansionViewComputeOperatorFusedApplication
        __cupaulipropPauliExpansionViewComputeOperatorFusedApplication = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewComputeOperatorFusedApplication')
        if __cupaulipropPauliExpansionViewComputeOperatorFusedApplication == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewComputeOperatorFusedApplication = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewComputeOperatorFusedApplication')

        global __cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff
        __cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff')
        if __cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff')

        global __cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff
        __cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff')
        if __cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff')

        global __cupaulipropPauliExpansionViewPrepareTruncation
        __cupaulipropPauliExpansionViewPrepareTruncation = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewPrepareTruncation')
        if __cupaulipropPauliExpansionViewPrepareTruncation == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewPrepareTruncation = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewPrepareTruncation')

        global __cupaulipropPauliExpansionViewExecuteTruncation
        __cupaulipropPauliExpansionViewExecuteTruncation = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropPauliExpansionViewExecuteTruncation')
        if __cupaulipropPauliExpansionViewExecuteTruncation == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropPauliExpansionViewExecuteTruncation = _cyb_dlsym(handle, 'cupaulipropPauliExpansionViewExecuteTruncation')

        global __cupaulipropCreateCliffordGateOperator
        __cupaulipropCreateCliffordGateOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropCreateCliffordGateOperator')
        if __cupaulipropCreateCliffordGateOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreateCliffordGateOperator = _cyb_dlsym(handle, 'cupaulipropCreateCliffordGateOperator')

        global __cupaulipropCreatePauliRotationGateOperator
        __cupaulipropCreatePauliRotationGateOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropCreatePauliRotationGateOperator')
        if __cupaulipropCreatePauliRotationGateOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreatePauliRotationGateOperator = _cyb_dlsym(handle, 'cupaulipropCreatePauliRotationGateOperator')

        global __cupaulipropCreatePauliNoiseChannelOperator
        __cupaulipropCreatePauliNoiseChannelOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropCreatePauliNoiseChannelOperator')
        if __cupaulipropCreatePauliNoiseChannelOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreatePauliNoiseChannelOperator = _cyb_dlsym(handle, 'cupaulipropCreatePauliNoiseChannelOperator')

        global __cupaulipropCreateAmplitudeDampingChannelOperator
        __cupaulipropCreateAmplitudeDampingChannelOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropCreateAmplitudeDampingChannelOperator')
        if __cupaulipropCreateAmplitudeDampingChannelOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropCreateAmplitudeDampingChannelOperator = _cyb_dlsym(handle, 'cupaulipropCreateAmplitudeDampingChannelOperator')

        global __cupaulipropQuantumOperatorAttachCotangentBuffer
        __cupaulipropQuantumOperatorAttachCotangentBuffer = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropQuantumOperatorAttachCotangentBuffer')
        if __cupaulipropQuantumOperatorAttachCotangentBuffer == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropQuantumOperatorAttachCotangentBuffer = _cyb_dlsym(handle, 'cupaulipropQuantumOperatorAttachCotangentBuffer')

        global __cupaulipropQuantumOperatorGetCotangentBuffer
        __cupaulipropQuantumOperatorGetCotangentBuffer = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropQuantumOperatorGetCotangentBuffer')
        if __cupaulipropQuantumOperatorGetCotangentBuffer == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropQuantumOperatorGetCotangentBuffer = _cyb_dlsym(handle, 'cupaulipropQuantumOperatorGetCotangentBuffer')

        global __cupaulipropDestroyOperator
        __cupaulipropDestroyOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cupaulipropDestroyOperator')
        if __cupaulipropDestroyOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cupaulipropDestroyOperator = _cyb_dlsym(handle, 'cupaulipropDestroyOperator')

        _cyb_atomic_int_store(<int *>&_cyb___py_cupauliprop_init, 1)
        return 0

cdef inline int _check_or_init_cupauliprop() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_cupauliprop_init):
        return 0

    return _init_cupauliprop()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

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

    global __cupaulipropResetDistributedConfiguration
    data["__cupaulipropResetDistributedConfiguration"] = <intptr_t>__cupaulipropResetDistributedConfiguration

    global __cupaulipropGetNumRanks
    data["__cupaulipropGetNumRanks"] = <intptr_t>__cupaulipropGetNumRanks

    global __cupaulipropGetProcRank
    data["__cupaulipropGetProcRank"] = <intptr_t>__cupaulipropGetProcRank

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

    global __cupaulipropPauliExpansionGetSortOrder
    data["__cupaulipropPauliExpansionGetSortOrder"] = <intptr_t>__cupaulipropPauliExpansionGetSortOrder

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

    global __cupaulipropPauliExpansionViewPrepareSort
    data["__cupaulipropPauliExpansionViewPrepareSort"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareSort

    global __cupaulipropPauliExpansionViewExecuteSort
    data["__cupaulipropPauliExpansionViewExecuteSort"] = <intptr_t>__cupaulipropPauliExpansionViewExecuteSort

    global __cupaulipropPauliExpansionPopulateFromView
    data["__cupaulipropPauliExpansionPopulateFromView"] = <intptr_t>__cupaulipropPauliExpansionPopulateFromView

    global __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView
    data["__cupaulipropPauliExpansionViewPrepareTraceWithExpansionView"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareTraceWithExpansionView

    global __cupaulipropPauliExpansionViewComputeTraceWithExpansionView
    data["__cupaulipropPauliExpansionViewComputeTraceWithExpansionView"] = <intptr_t>__cupaulipropPauliExpansionViewComputeTraceWithExpansionView

    global __cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff
    data["__cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff

    global __cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff
    data["__cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff"] = <intptr_t>__cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff

    global __cupaulipropPauliExpansionViewPrepareTraceWithZeroState
    data["__cupaulipropPauliExpansionViewPrepareTraceWithZeroState"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareTraceWithZeroState

    global __cupaulipropPauliExpansionViewComputeTraceWithZeroState
    data["__cupaulipropPauliExpansionViewComputeTraceWithZeroState"] = <intptr_t>__cupaulipropPauliExpansionViewComputeTraceWithZeroState

    global __cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff
    data["__cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff

    global __cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff
    data["__cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff"] = <intptr_t>__cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff

    global __cupaulipropPauliExpansionViewPrepareOperatorApplication
    data["__cupaulipropPauliExpansionViewPrepareOperatorApplication"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareOperatorApplication

    global __cupaulipropPauliExpansionViewComputeOperatorApplication
    data["__cupaulipropPauliExpansionViewComputeOperatorApplication"] = <intptr_t>__cupaulipropPauliExpansionViewComputeOperatorApplication

    global __cupaulipropPauliExpansionViewPrepareOperatorFusedApplication
    data["__cupaulipropPauliExpansionViewPrepareOperatorFusedApplication"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareOperatorFusedApplication

    global __cupaulipropPauliExpansionViewComputeOperatorFusedApplication
    data["__cupaulipropPauliExpansionViewComputeOperatorFusedApplication"] = <intptr_t>__cupaulipropPauliExpansionViewComputeOperatorFusedApplication

    global __cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff
    data["__cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff"] = <intptr_t>__cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff

    global __cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff
    data["__cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff"] = <intptr_t>__cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff

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

    global __cupaulipropCreateAmplitudeDampingChannelOperator
    data["__cupaulipropCreateAmplitudeDampingChannelOperator"] = <intptr_t>__cupaulipropCreateAmplitudeDampingChannelOperator

    global __cupaulipropQuantumOperatorAttachCotangentBuffer
    data["__cupaulipropQuantumOperatorAttachCotangentBuffer"] = <intptr_t>__cupaulipropQuantumOperatorAttachCotangentBuffer

    global __cupaulipropQuantumOperatorGetCotangentBuffer
    data["__cupaulipropQuantumOperatorGetCotangentBuffer"] = <intptr_t>__cupaulipropQuantumOperatorGetCotangentBuffer

    global __cupaulipropDestroyOperator
    data["__cupaulipropDestroyOperator"] = <intptr_t>__cupaulipropDestroyOperator
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cupauliprop")._handle_uint
    return <void*>handle


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


cdef cupaulipropStatus_t _cupaulipropResetDistributedConfiguration(cupaulipropHandle_t handle, cupaulipropDistributedProvider_t provider, const void* commPtr, size_t commSize) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropResetDistributedConfiguration
    _check_or_init_cupauliprop()
    if __cupaulipropResetDistributedConfiguration == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropResetDistributedConfiguration is not found")
    return (<cupaulipropStatus_t (*)(cupaulipropHandle_t, cupaulipropDistributedProvider_t, const void*, size_t) noexcept nogil>__cupaulipropResetDistributedConfiguration)(
        handle, provider, commPtr, commSize)


cdef cupaulipropStatus_t _cupaulipropGetNumRanks(const cupaulipropHandle_t handle, int32_t* numRanks) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropGetNumRanks
    _check_or_init_cupauliprop()
    if __cupaulipropGetNumRanks == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropGetNumRanks is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, int32_t*) noexcept nogil>__cupaulipropGetNumRanks)(
        handle, numRanks)


cdef cupaulipropStatus_t _cupaulipropGetProcRank(const cupaulipropHandle_t handle, int32_t* procRank) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropGetProcRank
    _check_or_init_cupauliprop()
    if __cupaulipropGetProcRank == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropGetProcRank is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, int32_t*) noexcept nogil>__cupaulipropGetProcRank)(
        handle, procRank)


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


cdef cupaulipropStatus_t _cupaulipropCreatePauliExpansion(const cupaulipropHandle_t handle, int32_t numQubits, void* xzBitsBuffer, int64_t xzBitsBufferSize, void* coefBuffer, int64_t coefBufferSize, cudaDataType_t dataType, int64_t numLocalTerms, cupaulipropSortOrder_t sortOrder, int32_t hasDuplicates, cupaulipropPauliExpansion_t* pauliExpansion) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropCreatePauliExpansion
    _check_or_init_cupauliprop()
    if __cupaulipropCreatePauliExpansion == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropCreatePauliExpansion is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, int32_t, void*, int64_t, void*, int64_t, cudaDataType_t, int64_t, cupaulipropSortOrder_t, int32_t, cupaulipropPauliExpansion_t*) noexcept nogil>__cupaulipropCreatePauliExpansion)(
        handle, numQubits, xzBitsBuffer, xzBitsBufferSize, coefBuffer, coefBufferSize, dataType, numLocalTerms, sortOrder, hasDuplicates, pauliExpansion)


cdef cupaulipropStatus_t _cupaulipropDestroyPauliExpansion(cupaulipropPauliExpansion_t pauliExpansion) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropDestroyPauliExpansion
    _check_or_init_cupauliprop()
    if __cupaulipropDestroyPauliExpansion == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropDestroyPauliExpansion is not found")
    return (<cupaulipropStatus_t (*)(cupaulipropPauliExpansion_t) noexcept nogil>__cupaulipropDestroyPauliExpansion)(
        pauliExpansion)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionGetStorageBuffer(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, void** xzBitsBuffer, int64_t* xzBitsBufferSize, void** coefBuffer, int64_t* coefBufferSize, int64_t* numLocalTerms, cupaulipropMemspace_t* location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionGetStorageBuffer
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionGetStorageBuffer == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionGetStorageBuffer is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, void**, int64_t*, void**, int64_t*, int64_t*, cupaulipropMemspace_t*) noexcept nogil>__cupaulipropPauliExpansionGetStorageBuffer)(
        handle, pauliExpansion, xzBitsBuffer, xzBitsBufferSize, coefBuffer, coefBufferSize, numLocalTerms, location)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionGetNumQubits(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int32_t* numQubits) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionGetNumQubits
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionGetNumQubits == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionGetNumQubits is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, int32_t*) noexcept nogil>__cupaulipropPauliExpansionGetNumQubits)(
        handle, pauliExpansion, numQubits)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionGetNumTerms(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, int64_t* numLocalTerms) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionGetNumTerms
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionGetNumTerms == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionGetNumTerms is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, int64_t*) noexcept nogil>__cupaulipropPauliExpansionGetNumTerms)(
        handle, pauliExpansion, numLocalTerms)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionGetDataType(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, cudaDataType_t* dataType) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionGetDataType
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionGetDataType == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionGetDataType is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, cudaDataType_t*) noexcept nogil>__cupaulipropPauliExpansionGetDataType)(
        handle, pauliExpansion, dataType)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionGetSortOrder(const cupaulipropHandle_t handle, const cupaulipropPauliExpansion_t pauliExpansion, cupaulipropSortOrder_t* sortOrder) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionGetSortOrder
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionGetSortOrder == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionGetSortOrder is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansion_t, cupaulipropSortOrder_t*) noexcept nogil>__cupaulipropPauliExpansionGetSortOrder)(
        handle, pauliExpansion, sortOrder)


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


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewGetNumTerms(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t* numLocalTerms) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewGetNumTerms
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewGetNumTerms == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewGetNumTerms is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, int64_t*) noexcept nogil>__cupaulipropPauliExpansionViewGetNumTerms)(
        handle, view, numLocalTerms)


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


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareDeduplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropSortOrder_t sortOrder, int64_t maxWorkspaceDeviceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareDeduplication
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareDeduplication == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareDeduplication is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropSortOrder_t, int64_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareDeduplication)(
        handle, viewIn, sortOrder, maxWorkspaceDeviceSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewExecuteDeduplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, cupaulipropSortOrder_t sortOrder, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewExecuteDeduplication
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewExecuteDeduplication == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewExecuteDeduplication is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t, cupaulipropSortOrder_t, cupaulipropWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cupaulipropPauliExpansionViewExecuteDeduplication)(
        handle, viewIn, expansionOut, sortOrder, workspace, stream)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareSort(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropSortOrder_t sortOrder, int64_t maxWorkspaceDeviceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareSort
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareSort == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareSort is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropSortOrder_t, int64_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareSort)(
        handle, viewIn, sortOrder, maxWorkspaceDeviceSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewExecuteSort(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, cupaulipropSortOrder_t sortOrder, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewExecuteSort
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewExecuteSort == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewExecuteSort is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t, cupaulipropSortOrder_t, cupaulipropWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cupaulipropPauliExpansionViewExecuteSort)(
        handle, viewIn, expansionOut, sortOrder, workspace, stream)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionPopulateFromView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionPopulateFromView
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionPopulateFromView == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionPopulateFromView is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t, cudaStream_t) noexcept nogil>__cupaulipropPauliExpansionPopulateFromView)(
        handle, viewIn, expansionOut, stream)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareTraceWithExpansionView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int64_t maxWorkspaceDeviceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareTraceWithExpansionView == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareTraceWithExpansionView is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, const cupaulipropPauliExpansionView_t, int64_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareTraceWithExpansionView)(
        handle, view1, view2, maxWorkspaceDeviceSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewComputeTraceWithExpansionView(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int32_t takeAdjoint1, void* traceSignificand, double* traceExponent, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewComputeTraceWithExpansionView
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewComputeTraceWithExpansionView == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewComputeTraceWithExpansionView is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, const cupaulipropPauliExpansionView_t, int32_t, void*, double*, cupaulipropWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cupaulipropPauliExpansionViewComputeTraceWithExpansionView)(
        handle, view1, view2, takeAdjoint1, traceSignificand, traceExponent, workspace, stream)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int64_t maxWorkspaceDeviceSize, int64_t* requiredXZBitsBufferSize1, int64_t* requiredCoefBufferSize1, int64_t* requiredXZBitsBufferSize2, int64_t* requiredCoefBufferSize2, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, const cupaulipropPauliExpansionView_t, int64_t, int64_t*, int64_t*, int64_t*, int64_t*, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff)(
        handle, view1, view2, maxWorkspaceDeviceSize, requiredXZBitsBufferSize1, requiredCoefBufferSize1, requiredXZBitsBufferSize2, requiredCoefBufferSize2, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view1, const cupaulipropPauliExpansionView_t view2, int32_t takeAdjoint1, const void* cotangentTraceSignificand, const double* cotangentTraceExponent, cupaulipropPauliExpansion_t cotangentExpansion1, cupaulipropPauliExpansion_t cotangentExpansion2, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, const cupaulipropPauliExpansionView_t, int32_t, const void*, const double*, cupaulipropPauliExpansion_t, cupaulipropPauliExpansion_t, cupaulipropWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff)(
        handle, view1, view2, takeAdjoint1, cotangentTraceSignificand, cotangentTraceExponent, cotangentExpansion1, cotangentExpansion2, workspace, stream)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareTraceWithZeroState(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t maxWorkspaceDeviceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareTraceWithZeroState
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareTraceWithZeroState == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareTraceWithZeroState is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, int64_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareTraceWithZeroState)(
        handle, view, maxWorkspaceDeviceSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewComputeTraceWithZeroState(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, void* traceSignificand, double* traceExponent, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewComputeTraceWithZeroState
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewComputeTraceWithZeroState == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewComputeTraceWithZeroState is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, void*, double*, cupaulipropWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cupaulipropPauliExpansionViewComputeTraceWithZeroState)(
        handle, view, traceSignificand, traceExponent, workspace, stream)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, int64_t maxWorkspaceDeviceSize, int64_t* requiredXZBitsBufferSize, int64_t* requiredCoefBufferSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, int64_t, int64_t*, int64_t*, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff)(
        handle, view, maxWorkspaceDeviceSize, requiredXZBitsBufferSize, requiredCoefBufferSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t view, const void* cotangentTraceSignificand, const double* cotangentTraceExponent, cupaulipropPauliExpansion_t cotangentExpansion, cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, const void*, const double*, cupaulipropPauliExpansion_t, cupaulipropWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff)(
        handle, view, cotangentTraceSignificand, cotangentTraceExponent, cotangentExpansion, workspace, stream)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareOperatorApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, const cupaulipropQuantumOperator_t quantumOperator, cupaulipropSortOrder_t sortOrder, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceDeviceSize, int64_t* requiredXZBitsBufferSize, int64_t* requiredCoefBufferSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareOperatorApplication
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareOperatorApplication == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareOperatorApplication is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, const cupaulipropQuantumOperator_t, cupaulipropSortOrder_t, int32_t, int32_t, const cupaulipropTruncationStrategy_t*, int64_t, int64_t*, int64_t*, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareOperatorApplication)(
        handle, viewIn, quantumOperator, sortOrder, keepDuplicates, numTruncationStrategies, truncationStrategies, maxWorkspaceDeviceSize, requiredXZBitsBufferSize, requiredCoefBufferSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewComputeOperatorApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, const cupaulipropQuantumOperator_t quantumOperator, int32_t adjoint, cupaulipropSortOrder_t sortOrder, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewComputeOperatorApplication
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewComputeOperatorApplication == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewComputeOperatorApplication is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t, const cupaulipropQuantumOperator_t, int32_t, cupaulipropSortOrder_t, int32_t, int32_t, const cupaulipropTruncationStrategy_t*, cupaulipropWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cupaulipropPauliExpansionViewComputeOperatorApplication)(
        handle, viewIn, expansionOut, quantumOperator, adjoint, sortOrder, keepDuplicates, numTruncationStrategies, truncationStrategies, workspace, stream)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareOperatorFusedApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int32_t numQuantumOperators, const cupaulipropQuantumOperator_t quantumOperators[], const int32_t adjoints[], int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceDeviceSize, int64_t* minExpansionOutCapacity, cupaulipropWorkspaceDescriptor_t minWorkspace, int64_t* averageExpansionOutCapacity, cupaulipropWorkspaceDescriptor_t averageWorkspace, int64_t* maxExpansionOutCapacity, cupaulipropWorkspaceDescriptor_t maxWorkspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareOperatorFusedApplication
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareOperatorFusedApplication == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareOperatorFusedApplication is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, int32_t, const cupaulipropQuantumOperator_t*, const int32_t*, int32_t, const cupaulipropTruncationStrategy_t*, int64_t, int64_t*, cupaulipropWorkspaceDescriptor_t, int64_t*, cupaulipropWorkspaceDescriptor_t, int64_t*, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareOperatorFusedApplication)(
        handle, viewIn, numQuantumOperators, quantumOperators, adjoints, numTruncationStrategies, truncationStrategies, maxWorkspaceDeviceSize, minExpansionOutCapacity, minWorkspace, averageExpansionOutCapacity, averageWorkspace, maxExpansionOutCapacity, maxWorkspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewComputeOperatorFusedApplication(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, int32_t numQuantumOperators, const cupaulipropQuantumOperator_t quantumOperators[], const int32_t adjoints[], int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewComputeOperatorFusedApplication
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewComputeOperatorFusedApplication == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewComputeOperatorFusedApplication is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t, int32_t, const cupaulipropQuantumOperator_t*, const int32_t*, int32_t, const cupaulipropTruncationStrategy_t*, cupaulipropWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cupaulipropPauliExpansionViewComputeOperatorFusedApplication)(
        handle, viewIn, expansionOut, numQuantumOperators, quantumOperators, adjoints, numTruncationStrategies, truncationStrategies, workspace, stream)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, const cupaulipropPauliExpansionView_t cotangentOut, const cupaulipropQuantumOperator_t quantumOperator, cupaulipropSortOrder_t sortOrder, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceDeviceSize, int64_t* requiredXZBitsBufferSize, int64_t* requiredCoefBufferSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, const cupaulipropPauliExpansionView_t, const cupaulipropQuantumOperator_t, cupaulipropSortOrder_t, int32_t, int32_t, const cupaulipropTruncationStrategy_t*, int64_t, int64_t*, int64_t*, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff)(
        handle, viewIn, cotangentOut, quantumOperator, sortOrder, keepDuplicates, numTruncationStrategies, truncationStrategies, maxWorkspaceDeviceSize, requiredXZBitsBufferSize, requiredCoefBufferSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, const cupaulipropPauliExpansionView_t cotangentOut, cupaulipropPauliExpansion_t cotangentIn, cupaulipropQuantumOperator_t quantumOperator, int32_t adjoint, cupaulipropSortOrder_t sortOrder, int32_t keepDuplicates, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t, cupaulipropQuantumOperator_t, int32_t, cupaulipropSortOrder_t, int32_t, int32_t, const cupaulipropTruncationStrategy_t*, cupaulipropWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff)(
        handle, viewIn, cotangentOut, cotangentIn, quantumOperator, adjoint, sortOrder, keepDuplicates, numTruncationStrategies, truncationStrategies, workspace, stream)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewPrepareTruncation(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], int64_t maxWorkspaceDeviceSize, cupaulipropWorkspaceDescriptor_t workspace) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewPrepareTruncation
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewPrepareTruncation == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewPrepareTruncation is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, int32_t, const cupaulipropTruncationStrategy_t*, int64_t, cupaulipropWorkspaceDescriptor_t) noexcept nogil>__cupaulipropPauliExpansionViewPrepareTruncation)(
        handle, viewIn, numTruncationStrategies, truncationStrategies, maxWorkspaceDeviceSize, workspace)


cdef cupaulipropStatus_t _cupaulipropPauliExpansionViewExecuteTruncation(const cupaulipropHandle_t handle, const cupaulipropPauliExpansionView_t viewIn, cupaulipropPauliExpansion_t expansionOut, int32_t numTruncationStrategies, const cupaulipropTruncationStrategy_t truncationStrategies[], cupaulipropWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropPauliExpansionViewExecuteTruncation
    _check_or_init_cupauliprop()
    if __cupaulipropPauliExpansionViewExecuteTruncation == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropPauliExpansionViewExecuteTruncation is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropPauliExpansionView_t, cupaulipropPauliExpansion_t, int32_t, const cupaulipropTruncationStrategy_t*, cupaulipropWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cupaulipropPauliExpansionViewExecuteTruncation)(
        handle, viewIn, expansionOut, numTruncationStrategies, truncationStrategies, workspace, stream)


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


cdef cupaulipropStatus_t _cupaulipropCreateAmplitudeDampingChannelOperator(const cupaulipropHandle_t handle, int32_t qubitIndex, double dampingProb, double exciteProb, cupaulipropQuantumOperator_t* oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropCreateAmplitudeDampingChannelOperator
    _check_or_init_cupauliprop()
    if __cupaulipropCreateAmplitudeDampingChannelOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropCreateAmplitudeDampingChannelOperator is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, int32_t, double, double, cupaulipropQuantumOperator_t*) noexcept nogil>__cupaulipropCreateAmplitudeDampingChannelOperator)(
        handle, qubitIndex, dampingProb, exciteProb, oper)


cdef cupaulipropStatus_t _cupaulipropQuantumOperatorAttachCotangentBuffer(const cupaulipropHandle_t handle, cupaulipropQuantumOperator_t oper, void* cotangentBuffer, int64_t cotangentBufferSize, cudaDataType_t dataType, cupaulipropMemspace_t location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropQuantumOperatorAttachCotangentBuffer
    _check_or_init_cupauliprop()
    if __cupaulipropQuantumOperatorAttachCotangentBuffer == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropQuantumOperatorAttachCotangentBuffer is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, cupaulipropQuantumOperator_t, void*, int64_t, cudaDataType_t, cupaulipropMemspace_t) noexcept nogil>__cupaulipropQuantumOperatorAttachCotangentBuffer)(
        handle, oper, cotangentBuffer, cotangentBufferSize, dataType, location)


cdef cupaulipropStatus_t _cupaulipropQuantumOperatorGetCotangentBuffer(const cupaulipropHandle_t handle, const cupaulipropQuantumOperator_t oper, void** cotangentBuffer, int64_t* cotangentBufferNumElements, cudaDataType_t* dataType, cupaulipropMemspace_t* location) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropQuantumOperatorGetCotangentBuffer
    _check_or_init_cupauliprop()
    if __cupaulipropQuantumOperatorGetCotangentBuffer == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropQuantumOperatorGetCotangentBuffer is not found")
    return (<cupaulipropStatus_t (*)(const cupaulipropHandle_t, const cupaulipropQuantumOperator_t, void**, int64_t*, cudaDataType_t*, cupaulipropMemspace_t*) noexcept nogil>__cupaulipropQuantumOperatorGetCotangentBuffer)(
        handle, oper, cotangentBuffer, cotangentBufferNumElements, dataType, location)


cdef cupaulipropStatus_t _cupaulipropDestroyOperator(cupaulipropQuantumOperator_t oper) except?_CUPAULIPROPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cupaulipropDestroyOperator
    _check_or_init_cupauliprop()
    if __cupaulipropDestroyOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cupaulipropDestroyOperator is not found")
    return (<cupaulipropStatus_t (*)(cupaulipropQuantumOperator_t) noexcept nogil>__cupaulipropDestroyOperator)(
        oper)
