# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 26.09.0 to 26.11.0. Do not modify it directly.



# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    int32_t,
    intptr_t,
    uint32_t,
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

cdef int _cyb___py_custatevecEx_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .._utils import FunctionNotFoundError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################


cdef void* __custatevecExDictionaryDestroy = NULL
cdef void* __custatevecExCommunicatorInitialize = NULL
cdef void* __custatevecExCommunicatorFinalize = NULL
cdef void* __custatevecExCommunicatorGetSizeAndRank = NULL
cdef void* __custatevecExCommunicatorCreate = NULL
cdef void* __custatevecExCommunicatorDestroy = NULL
cdef void* __custatevecExConfigureStateVectorSingleDevice = NULL
cdef void* __custatevecExConfigureStateVectorMultiDevice = NULL
cdef void* __custatevecExConfigureStateVectorMultiProcess = NULL
cdef void* __custatevecExStateVectorCreateSingleProcess = NULL
cdef void* __custatevecExStateVectorCreateMultiProcess = NULL
cdef void* __custatevecExStateVectorDestroy = NULL
cdef void* __custatevecExStateVectorGetProperty = NULL
cdef void* __custatevecExStateVectorSetMathMode = NULL
cdef void* __custatevecExStateVectorSetZeroState = NULL
cdef void* __custatevecExStateVectorGetState = NULL
cdef void* __custatevecExStateVectorSetState = NULL
cdef void* __custatevecExStateVectorReassignWireOrdering = NULL
cdef void* __custatevecExStateVectorPermuteIndexBits = NULL
cdef void* __custatevecExStateVectorStageSubSV = NULL
cdef void* __custatevecExStateVectorExposeResources = NULL
cdef void* __custatevecExStateVectorGetResourcesFromDeviceSubSV = NULL
cdef void* __custatevecExStateVectorGetResourcesFromDeviceSubSVView = NULL
cdef void* __custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice = NULL
cdef void* __custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView = NULL
cdef void* __custatevecExStateVectorSynchronize = NULL
cdef void* __custatevecExStateVectorSynchronizeScoped = NULL
cdef void* __custatevecExStateVectorAddWires = NULL
cdef void* __custatevecExAbs2SumArray = NULL
cdef void* __custatevecExMeasure = NULL
cdef void* __custatevecExSample = NULL
cdef void* __custatevecExApplyMatrix = NULL
cdef void* __custatevecExApplyPauliRotation = NULL
cdef void* __custatevecExComputeExpectationOnPauliBasis = NULL
cdef void* __custatevecExComputeExpectation = NULL
cdef void* __custatevecExConfigureSVUpdater = NULL
cdef void* __custatevecExSVUpdaterCreate = NULL
cdef void* __custatevecExSVUpdaterDestroy = NULL
cdef void* __custatevecExSVUpdaterClear = NULL
cdef void* __custatevecExSVUpdaterEnqueueMatrix = NULL
cdef void* __custatevecExSVUpdaterEnqueueUnitaryChannel = NULL
cdef void* __custatevecExSVUpdaterEnqueueGeneralChannel = NULL
cdef void* __custatevecExSVUpdaterGetMaxNumRequiredRandnums = NULL
cdef void* __custatevecExSVUpdaterApply = NULL

cdef int _init_custatevecEx() except -1 nogil:
    global _cyb___py_custatevecEx_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_custatevecEx_init: return 0

        global __custatevecExDictionaryDestroy
        __custatevecExDictionaryDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExDictionaryDestroy')
        if __custatevecExDictionaryDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExDictionaryDestroy = _cyb_dlsym(handle, 'custatevecExDictionaryDestroy')

        global __custatevecExCommunicatorInitialize
        __custatevecExCommunicatorInitialize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExCommunicatorInitialize')
        if __custatevecExCommunicatorInitialize == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExCommunicatorInitialize = _cyb_dlsym(handle, 'custatevecExCommunicatorInitialize')

        global __custatevecExCommunicatorFinalize
        __custatevecExCommunicatorFinalize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExCommunicatorFinalize')
        if __custatevecExCommunicatorFinalize == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExCommunicatorFinalize = _cyb_dlsym(handle, 'custatevecExCommunicatorFinalize')

        global __custatevecExCommunicatorGetSizeAndRank
        __custatevecExCommunicatorGetSizeAndRank = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExCommunicatorGetSizeAndRank')
        if __custatevecExCommunicatorGetSizeAndRank == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExCommunicatorGetSizeAndRank = _cyb_dlsym(handle, 'custatevecExCommunicatorGetSizeAndRank')

        global __custatevecExCommunicatorCreate
        __custatevecExCommunicatorCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExCommunicatorCreate')
        if __custatevecExCommunicatorCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExCommunicatorCreate = _cyb_dlsym(handle, 'custatevecExCommunicatorCreate')

        global __custatevecExCommunicatorDestroy
        __custatevecExCommunicatorDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExCommunicatorDestroy')
        if __custatevecExCommunicatorDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExCommunicatorDestroy = _cyb_dlsym(handle, 'custatevecExCommunicatorDestroy')

        global __custatevecExConfigureStateVectorSingleDevice
        __custatevecExConfigureStateVectorSingleDevice = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExConfigureStateVectorSingleDevice')
        if __custatevecExConfigureStateVectorSingleDevice == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExConfigureStateVectorSingleDevice = _cyb_dlsym(handle, 'custatevecExConfigureStateVectorSingleDevice')

        global __custatevecExConfigureStateVectorMultiDevice
        __custatevecExConfigureStateVectorMultiDevice = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExConfigureStateVectorMultiDevice')
        if __custatevecExConfigureStateVectorMultiDevice == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExConfigureStateVectorMultiDevice = _cyb_dlsym(handle, 'custatevecExConfigureStateVectorMultiDevice')

        global __custatevecExConfigureStateVectorMultiProcess
        __custatevecExConfigureStateVectorMultiProcess = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExConfigureStateVectorMultiProcess')
        if __custatevecExConfigureStateVectorMultiProcess == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExConfigureStateVectorMultiProcess = _cyb_dlsym(handle, 'custatevecExConfigureStateVectorMultiProcess')

        global __custatevecExStateVectorCreateSingleProcess
        __custatevecExStateVectorCreateSingleProcess = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorCreateSingleProcess')
        if __custatevecExStateVectorCreateSingleProcess == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorCreateSingleProcess = _cyb_dlsym(handle, 'custatevecExStateVectorCreateSingleProcess')

        global __custatevecExStateVectorCreateMultiProcess
        __custatevecExStateVectorCreateMultiProcess = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorCreateMultiProcess')
        if __custatevecExStateVectorCreateMultiProcess == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorCreateMultiProcess = _cyb_dlsym(handle, 'custatevecExStateVectorCreateMultiProcess')

        global __custatevecExStateVectorDestroy
        __custatevecExStateVectorDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorDestroy')
        if __custatevecExStateVectorDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorDestroy = _cyb_dlsym(handle, 'custatevecExStateVectorDestroy')

        global __custatevecExStateVectorGetProperty
        __custatevecExStateVectorGetProperty = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorGetProperty')
        if __custatevecExStateVectorGetProperty == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorGetProperty = _cyb_dlsym(handle, 'custatevecExStateVectorGetProperty')

        global __custatevecExStateVectorSetMathMode
        __custatevecExStateVectorSetMathMode = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorSetMathMode')
        if __custatevecExStateVectorSetMathMode == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorSetMathMode = _cyb_dlsym(handle, 'custatevecExStateVectorSetMathMode')

        global __custatevecExStateVectorSetZeroState
        __custatevecExStateVectorSetZeroState = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorSetZeroState')
        if __custatevecExStateVectorSetZeroState == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorSetZeroState = _cyb_dlsym(handle, 'custatevecExStateVectorSetZeroState')

        global __custatevecExStateVectorGetState
        __custatevecExStateVectorGetState = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorGetState')
        if __custatevecExStateVectorGetState == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorGetState = _cyb_dlsym(handle, 'custatevecExStateVectorGetState')

        global __custatevecExStateVectorSetState
        __custatevecExStateVectorSetState = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorSetState')
        if __custatevecExStateVectorSetState == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorSetState = _cyb_dlsym(handle, 'custatevecExStateVectorSetState')

        global __custatevecExStateVectorReassignWireOrdering
        __custatevecExStateVectorReassignWireOrdering = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorReassignWireOrdering')
        if __custatevecExStateVectorReassignWireOrdering == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorReassignWireOrdering = _cyb_dlsym(handle, 'custatevecExStateVectorReassignWireOrdering')

        global __custatevecExStateVectorPermuteIndexBits
        __custatevecExStateVectorPermuteIndexBits = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorPermuteIndexBits')
        if __custatevecExStateVectorPermuteIndexBits == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorPermuteIndexBits = _cyb_dlsym(handle, 'custatevecExStateVectorPermuteIndexBits')

        global __custatevecExStateVectorStageSubSV
        __custatevecExStateVectorStageSubSV = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorStageSubSV')
        if __custatevecExStateVectorStageSubSV == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorStageSubSV = _cyb_dlsym(handle, 'custatevecExStateVectorStageSubSV')

        global __custatevecExStateVectorExposeResources
        __custatevecExStateVectorExposeResources = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorExposeResources')
        if __custatevecExStateVectorExposeResources == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorExposeResources = _cyb_dlsym(handle, 'custatevecExStateVectorExposeResources')

        global __custatevecExStateVectorGetResourcesFromDeviceSubSV
        __custatevecExStateVectorGetResourcesFromDeviceSubSV = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorGetResourcesFromDeviceSubSV')
        if __custatevecExStateVectorGetResourcesFromDeviceSubSV == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorGetResourcesFromDeviceSubSV = _cyb_dlsym(handle, 'custatevecExStateVectorGetResourcesFromDeviceSubSV')

        global __custatevecExStateVectorGetResourcesFromDeviceSubSVView
        __custatevecExStateVectorGetResourcesFromDeviceSubSVView = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorGetResourcesFromDeviceSubSVView')
        if __custatevecExStateVectorGetResourcesFromDeviceSubSVView == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorGetResourcesFromDeviceSubSVView = _cyb_dlsym(handle, 'custatevecExStateVectorGetResourcesFromDeviceSubSVView')

        global __custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice
        __custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice')
        if __custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice = _cyb_dlsym(handle, 'custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice')

        global __custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView
        __custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView')
        if __custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView = _cyb_dlsym(handle, 'custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView')

        global __custatevecExStateVectorSynchronize
        __custatevecExStateVectorSynchronize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorSynchronize')
        if __custatevecExStateVectorSynchronize == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorSynchronize = _cyb_dlsym(handle, 'custatevecExStateVectorSynchronize')

        global __custatevecExStateVectorSynchronizeScoped
        __custatevecExStateVectorSynchronizeScoped = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorSynchronizeScoped')
        if __custatevecExStateVectorSynchronizeScoped == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorSynchronizeScoped = _cyb_dlsym(handle, 'custatevecExStateVectorSynchronizeScoped')

        global __custatevecExStateVectorAddWires
        __custatevecExStateVectorAddWires = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExStateVectorAddWires')
        if __custatevecExStateVectorAddWires == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExStateVectorAddWires = _cyb_dlsym(handle, 'custatevecExStateVectorAddWires')

        global __custatevecExAbs2SumArray
        __custatevecExAbs2SumArray = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExAbs2SumArray')
        if __custatevecExAbs2SumArray == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExAbs2SumArray = _cyb_dlsym(handle, 'custatevecExAbs2SumArray')

        global __custatevecExMeasure
        __custatevecExMeasure = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExMeasure')
        if __custatevecExMeasure == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExMeasure = _cyb_dlsym(handle, 'custatevecExMeasure')

        global __custatevecExSample
        __custatevecExSample = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExSample')
        if __custatevecExSample == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExSample = _cyb_dlsym(handle, 'custatevecExSample')

        global __custatevecExApplyMatrix
        __custatevecExApplyMatrix = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExApplyMatrix')
        if __custatevecExApplyMatrix == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExApplyMatrix = _cyb_dlsym(handle, 'custatevecExApplyMatrix')

        global __custatevecExApplyPauliRotation
        __custatevecExApplyPauliRotation = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExApplyPauliRotation')
        if __custatevecExApplyPauliRotation == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExApplyPauliRotation = _cyb_dlsym(handle, 'custatevecExApplyPauliRotation')

        global __custatevecExComputeExpectationOnPauliBasis
        __custatevecExComputeExpectationOnPauliBasis = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExComputeExpectationOnPauliBasis')
        if __custatevecExComputeExpectationOnPauliBasis == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExComputeExpectationOnPauliBasis = _cyb_dlsym(handle, 'custatevecExComputeExpectationOnPauliBasis')

        global __custatevecExComputeExpectation
        __custatevecExComputeExpectation = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExComputeExpectation')
        if __custatevecExComputeExpectation == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExComputeExpectation = _cyb_dlsym(handle, 'custatevecExComputeExpectation')

        global __custatevecExConfigureSVUpdater
        __custatevecExConfigureSVUpdater = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExConfigureSVUpdater')
        if __custatevecExConfigureSVUpdater == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExConfigureSVUpdater = _cyb_dlsym(handle, 'custatevecExConfigureSVUpdater')

        global __custatevecExSVUpdaterCreate
        __custatevecExSVUpdaterCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExSVUpdaterCreate')
        if __custatevecExSVUpdaterCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExSVUpdaterCreate = _cyb_dlsym(handle, 'custatevecExSVUpdaterCreate')

        global __custatevecExSVUpdaterDestroy
        __custatevecExSVUpdaterDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExSVUpdaterDestroy')
        if __custatevecExSVUpdaterDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExSVUpdaterDestroy = _cyb_dlsym(handle, 'custatevecExSVUpdaterDestroy')

        global __custatevecExSVUpdaterClear
        __custatevecExSVUpdaterClear = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExSVUpdaterClear')
        if __custatevecExSVUpdaterClear == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExSVUpdaterClear = _cyb_dlsym(handle, 'custatevecExSVUpdaterClear')

        global __custatevecExSVUpdaterEnqueueMatrix
        __custatevecExSVUpdaterEnqueueMatrix = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExSVUpdaterEnqueueMatrix')
        if __custatevecExSVUpdaterEnqueueMatrix == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExSVUpdaterEnqueueMatrix = _cyb_dlsym(handle, 'custatevecExSVUpdaterEnqueueMatrix')

        global __custatevecExSVUpdaterEnqueueUnitaryChannel
        __custatevecExSVUpdaterEnqueueUnitaryChannel = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExSVUpdaterEnqueueUnitaryChannel')
        if __custatevecExSVUpdaterEnqueueUnitaryChannel == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExSVUpdaterEnqueueUnitaryChannel = _cyb_dlsym(handle, 'custatevecExSVUpdaterEnqueueUnitaryChannel')

        global __custatevecExSVUpdaterEnqueueGeneralChannel
        __custatevecExSVUpdaterEnqueueGeneralChannel = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExSVUpdaterEnqueueGeneralChannel')
        if __custatevecExSVUpdaterEnqueueGeneralChannel == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExSVUpdaterEnqueueGeneralChannel = _cyb_dlsym(handle, 'custatevecExSVUpdaterEnqueueGeneralChannel')

        global __custatevecExSVUpdaterGetMaxNumRequiredRandnums
        __custatevecExSVUpdaterGetMaxNumRequiredRandnums = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExSVUpdaterGetMaxNumRequiredRandnums')
        if __custatevecExSVUpdaterGetMaxNumRequiredRandnums == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExSVUpdaterGetMaxNumRequiredRandnums = _cyb_dlsym(handle, 'custatevecExSVUpdaterGetMaxNumRequiredRandnums')

        global __custatevecExSVUpdaterApply
        __custatevecExSVUpdaterApply = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecExSVUpdaterApply')
        if __custatevecExSVUpdaterApply == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecExSVUpdaterApply = _cyb_dlsym(handle, 'custatevecExSVUpdaterApply')

        _cyb_atomic_int_store(<int *>&_cyb___py_custatevecEx_init, 1)
        return 0

cdef inline int _check_or_init_custatevecEx() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_custatevecEx_init):
        return 0

    return _init_custatevecEx()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_custatevecEx()
    cdef dict data = {}
    global __custatevecExDictionaryDestroy
    data["__custatevecExDictionaryDestroy"] = <intptr_t>__custatevecExDictionaryDestroy

    global __custatevecExCommunicatorInitialize
    data["__custatevecExCommunicatorInitialize"] = <intptr_t>__custatevecExCommunicatorInitialize

    global __custatevecExCommunicatorFinalize
    data["__custatevecExCommunicatorFinalize"] = <intptr_t>__custatevecExCommunicatorFinalize

    global __custatevecExCommunicatorGetSizeAndRank
    data["__custatevecExCommunicatorGetSizeAndRank"] = <intptr_t>__custatevecExCommunicatorGetSizeAndRank

    global __custatevecExCommunicatorCreate
    data["__custatevecExCommunicatorCreate"] = <intptr_t>__custatevecExCommunicatorCreate

    global __custatevecExCommunicatorDestroy
    data["__custatevecExCommunicatorDestroy"] = <intptr_t>__custatevecExCommunicatorDestroy

    global __custatevecExConfigureStateVectorSingleDevice
    data["__custatevecExConfigureStateVectorSingleDevice"] = <intptr_t>__custatevecExConfigureStateVectorSingleDevice

    global __custatevecExConfigureStateVectorMultiDevice
    data["__custatevecExConfigureStateVectorMultiDevice"] = <intptr_t>__custatevecExConfigureStateVectorMultiDevice

    global __custatevecExConfigureStateVectorMultiProcess
    data["__custatevecExConfigureStateVectorMultiProcess"] = <intptr_t>__custatevecExConfigureStateVectorMultiProcess

    global __custatevecExStateVectorCreateSingleProcess
    data["__custatevecExStateVectorCreateSingleProcess"] = <intptr_t>__custatevecExStateVectorCreateSingleProcess

    global __custatevecExStateVectorCreateMultiProcess
    data["__custatevecExStateVectorCreateMultiProcess"] = <intptr_t>__custatevecExStateVectorCreateMultiProcess

    global __custatevecExStateVectorDestroy
    data["__custatevecExStateVectorDestroy"] = <intptr_t>__custatevecExStateVectorDestroy

    global __custatevecExStateVectorGetProperty
    data["__custatevecExStateVectorGetProperty"] = <intptr_t>__custatevecExStateVectorGetProperty

    global __custatevecExStateVectorSetMathMode
    data["__custatevecExStateVectorSetMathMode"] = <intptr_t>__custatevecExStateVectorSetMathMode

    global __custatevecExStateVectorSetZeroState
    data["__custatevecExStateVectorSetZeroState"] = <intptr_t>__custatevecExStateVectorSetZeroState

    global __custatevecExStateVectorGetState
    data["__custatevecExStateVectorGetState"] = <intptr_t>__custatevecExStateVectorGetState

    global __custatevecExStateVectorSetState
    data["__custatevecExStateVectorSetState"] = <intptr_t>__custatevecExStateVectorSetState

    global __custatevecExStateVectorReassignWireOrdering
    data["__custatevecExStateVectorReassignWireOrdering"] = <intptr_t>__custatevecExStateVectorReassignWireOrdering

    global __custatevecExStateVectorPermuteIndexBits
    data["__custatevecExStateVectorPermuteIndexBits"] = <intptr_t>__custatevecExStateVectorPermuteIndexBits

    global __custatevecExStateVectorStageSubSV
    data["__custatevecExStateVectorStageSubSV"] = <intptr_t>__custatevecExStateVectorStageSubSV

    global __custatevecExStateVectorExposeResources
    data["__custatevecExStateVectorExposeResources"] = <intptr_t>__custatevecExStateVectorExposeResources

    global __custatevecExStateVectorGetResourcesFromDeviceSubSV
    data["__custatevecExStateVectorGetResourcesFromDeviceSubSV"] = <intptr_t>__custatevecExStateVectorGetResourcesFromDeviceSubSV

    global __custatevecExStateVectorGetResourcesFromDeviceSubSVView
    data["__custatevecExStateVectorGetResourcesFromDeviceSubSVView"] = <intptr_t>__custatevecExStateVectorGetResourcesFromDeviceSubSVView

    global __custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice
    data["__custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice"] = <intptr_t>__custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice

    global __custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView
    data["__custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView"] = <intptr_t>__custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView

    global __custatevecExStateVectorSynchronize
    data["__custatevecExStateVectorSynchronize"] = <intptr_t>__custatevecExStateVectorSynchronize

    global __custatevecExStateVectorSynchronizeScoped
    data["__custatevecExStateVectorSynchronizeScoped"] = <intptr_t>__custatevecExStateVectorSynchronizeScoped

    global __custatevecExStateVectorAddWires
    data["__custatevecExStateVectorAddWires"] = <intptr_t>__custatevecExStateVectorAddWires

    global __custatevecExAbs2SumArray
    data["__custatevecExAbs2SumArray"] = <intptr_t>__custatevecExAbs2SumArray

    global __custatevecExMeasure
    data["__custatevecExMeasure"] = <intptr_t>__custatevecExMeasure

    global __custatevecExSample
    data["__custatevecExSample"] = <intptr_t>__custatevecExSample

    global __custatevecExApplyMatrix
    data["__custatevecExApplyMatrix"] = <intptr_t>__custatevecExApplyMatrix

    global __custatevecExApplyPauliRotation
    data["__custatevecExApplyPauliRotation"] = <intptr_t>__custatevecExApplyPauliRotation

    global __custatevecExComputeExpectationOnPauliBasis
    data["__custatevecExComputeExpectationOnPauliBasis"] = <intptr_t>__custatevecExComputeExpectationOnPauliBasis

    global __custatevecExComputeExpectation
    data["__custatevecExComputeExpectation"] = <intptr_t>__custatevecExComputeExpectation

    global __custatevecExConfigureSVUpdater
    data["__custatevecExConfigureSVUpdater"] = <intptr_t>__custatevecExConfigureSVUpdater

    global __custatevecExSVUpdaterCreate
    data["__custatevecExSVUpdaterCreate"] = <intptr_t>__custatevecExSVUpdaterCreate

    global __custatevecExSVUpdaterDestroy
    data["__custatevecExSVUpdaterDestroy"] = <intptr_t>__custatevecExSVUpdaterDestroy

    global __custatevecExSVUpdaterClear
    data["__custatevecExSVUpdaterClear"] = <intptr_t>__custatevecExSVUpdaterClear

    global __custatevecExSVUpdaterEnqueueMatrix
    data["__custatevecExSVUpdaterEnqueueMatrix"] = <intptr_t>__custatevecExSVUpdaterEnqueueMatrix

    global __custatevecExSVUpdaterEnqueueUnitaryChannel
    data["__custatevecExSVUpdaterEnqueueUnitaryChannel"] = <intptr_t>__custatevecExSVUpdaterEnqueueUnitaryChannel

    global __custatevecExSVUpdaterEnqueueGeneralChannel
    data["__custatevecExSVUpdaterEnqueueGeneralChannel"] = <intptr_t>__custatevecExSVUpdaterEnqueueGeneralChannel

    global __custatevecExSVUpdaterGetMaxNumRequiredRandnums
    data["__custatevecExSVUpdaterGetMaxNumRequiredRandnums"] = <intptr_t>__custatevecExSVUpdaterGetMaxNumRequiredRandnums

    global __custatevecExSVUpdaterApply
    data["__custatevecExSVUpdaterApply"] = <intptr_t>__custatevecExSVUpdaterApply
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]



cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("custatevec")._handle_uint
    return <void*>handle


###############################################################################
# Wrapper functions
###############################################################################

cdef custatevecStatus_t _custatevecExDictionaryDestroy(custatevecExDictionaryDescriptor_t dictionary) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExDictionaryDestroy
    _check_or_init_custatevecEx()
    if __custatevecExDictionaryDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExDictionaryDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecExDictionaryDescriptor_t) noexcept nogil>__custatevecExDictionaryDestroy)(
        dictionary)


cdef custatevecStatus_t _custatevecExCommunicatorInitialize(custatevecCommunicatorType_t communicatorType, const char* libraryPath, int* argc, char*** argv, custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExCommunicatorInitialize
    _check_or_init_custatevecEx()
    if __custatevecExCommunicatorInitialize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExCommunicatorInitialize is not found")
    return (<custatevecStatus_t (*)(custatevecCommunicatorType_t, const char*, int*, char***, custatevecExCommunicatorStatus_t*) noexcept nogil>__custatevecExCommunicatorInitialize)(
        communicatorType, libraryPath, argc, argv, exCommStatus)


cdef custatevecStatus_t _custatevecExCommunicatorFinalize(custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExCommunicatorFinalize
    _check_or_init_custatevecEx()
    if __custatevecExCommunicatorFinalize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExCommunicatorFinalize is not found")
    return (<custatevecStatus_t (*)(custatevecExCommunicatorStatus_t*) noexcept nogil>__custatevecExCommunicatorFinalize)(
        exCommStatus)


cdef custatevecStatus_t _custatevecExCommunicatorGetSizeAndRank(int32_t* size, int32_t* rank, custatevecExCommunicatorStatus_t* exCommStatus) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExCommunicatorGetSizeAndRank
    _check_or_init_custatevecEx()
    if __custatevecExCommunicatorGetSizeAndRank == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExCommunicatorGetSizeAndRank is not found")
    return (<custatevecStatus_t (*)(int32_t*, int32_t*, custatevecExCommunicatorStatus_t*) noexcept nogil>__custatevecExCommunicatorGetSizeAndRank)(
        size, rank, exCommStatus)


cdef custatevecStatus_t _custatevecExCommunicatorCreate(custatevecExCommunicatorDescriptor_t* exCommunicator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExCommunicatorCreate
    _check_or_init_custatevecEx()
    if __custatevecExCommunicatorCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExCommunicatorCreate is not found")
    return (<custatevecStatus_t (*)(custatevecExCommunicatorDescriptor_t*) noexcept nogil>__custatevecExCommunicatorCreate)(
        exCommunicator)


cdef custatevecStatus_t _custatevecExCommunicatorDestroy(custatevecExCommunicatorDescriptor_t exCommunicator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExCommunicatorDestroy
    _check_or_init_custatevecEx()
    if __custatevecExCommunicatorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExCommunicatorDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecExCommunicatorDescriptor_t) noexcept nogil>__custatevecExCommunicatorDestroy)(
        exCommunicator)


cdef custatevecStatus_t _custatevecExConfigureStateVectorSingleDevice(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, int32_t deviceId, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExConfigureStateVectorSingleDevice
    _check_or_init_custatevecEx()
    if __custatevecExConfigureStateVectorSingleDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExConfigureStateVectorSingleDevice is not found")
    return (<custatevecStatus_t (*)(custatevecExDictionaryDescriptor_t*, cudaDataType_t, int32_t, int32_t, int32_t, uint32_t) noexcept nogil>__custatevecExConfigureStateVectorSingleDevice)(
        svConfig, svDataType, numWires, numDeviceWires, deviceId, capability)


cdef custatevecStatus_t _custatevecExConfigureStateVectorMultiDevice(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, const int32_t* deviceIds, int32_t numDevices, custatevecDeviceNetworkType_t networkType, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExConfigureStateVectorMultiDevice
    _check_or_init_custatevecEx()
    if __custatevecExConfigureStateVectorMultiDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExConfigureStateVectorMultiDevice is not found")
    return (<custatevecStatus_t (*)(custatevecExDictionaryDescriptor_t*, cudaDataType_t, int32_t, int32_t, const int32_t*, int32_t, custatevecDeviceNetworkType_t, uint32_t) noexcept nogil>__custatevecExConfigureStateVectorMultiDevice)(
        svConfig, svDataType, numWires, numDeviceWires, deviceIds, numDevices, networkType, capability)


cdef custatevecStatus_t _custatevecExConfigureStateVectorMultiProcess(custatevecExDictionaryDescriptor_t* svConfig, cudaDataType_t svDataType, int32_t numWires, int32_t numDeviceWires, int32_t deviceId, custatevecExMemorySharingMethod_t memorySharingMethod, const custatevecExGlobalIndexBitClass_t* globalIndexBitClasses, const int32_t* numGlobalIndexBitsPerLayer, int32_t numGlobalIndexBitLayers, size_t transferWorkspaceSizeInBytes, const void* auxConfig, uint32_t capability) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExConfigureStateVectorMultiProcess
    _check_or_init_custatevecEx()
    if __custatevecExConfigureStateVectorMultiProcess == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExConfigureStateVectorMultiProcess is not found")
    return (<custatevecStatus_t (*)(custatevecExDictionaryDescriptor_t*, cudaDataType_t, int32_t, int32_t, int32_t, custatevecExMemorySharingMethod_t, const custatevecExGlobalIndexBitClass_t*, const int32_t*, int32_t, size_t, const void*, uint32_t) noexcept nogil>__custatevecExConfigureStateVectorMultiProcess)(
        svConfig, svDataType, numWires, numDeviceWires, deviceId, memorySharingMethod, globalIndexBitClasses, numGlobalIndexBitsPerLayer, numGlobalIndexBitLayers, transferWorkspaceSizeInBytes, auxConfig, capability)


cdef custatevecStatus_t _custatevecExStateVectorCreateSingleProcess(custatevecExStateVectorDescriptor_t* stateVector, const custatevecExDictionaryDescriptor_t svConfig, const cudaStream_t* streams, int32_t numStreams, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorCreateSingleProcess
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorCreateSingleProcess == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorCreateSingleProcess is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t*, const custatevecExDictionaryDescriptor_t, const cudaStream_t*, int32_t, custatevecExResourceManagerDescriptor_t) noexcept nogil>__custatevecExStateVectorCreateSingleProcess)(
        stateVector, svConfig, streams, numStreams, resourceManager)


cdef custatevecStatus_t _custatevecExStateVectorCreateMultiProcess(custatevecExStateVectorDescriptor_t* stateVector, const custatevecExDictionaryDescriptor_t svConfig, cudaStream_t stream, custatevecExCommunicatorDescriptor_t exCommunicator, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorCreateMultiProcess
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorCreateMultiProcess == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorCreateMultiProcess is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t*, const custatevecExDictionaryDescriptor_t, cudaStream_t, custatevecExCommunicatorDescriptor_t, custatevecExResourceManagerDescriptor_t) noexcept nogil>__custatevecExStateVectorCreateMultiProcess)(
        stateVector, svConfig, stream, exCommunicator, resourceManager)


cdef custatevecStatus_t _custatevecExStateVectorDestroy(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorDestroy
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t) noexcept nogil>__custatevecExStateVectorDestroy)(
        stateVector)


cdef custatevecStatus_t _custatevecExStateVectorGetProperty(const custatevecExStateVectorDescriptor_t stateVector, custatevecExStateVectorProperty_t property, void* value, size_t sizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorGetProperty
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorGetProperty is not found")
    return (<custatevecStatus_t (*)(const custatevecExStateVectorDescriptor_t, custatevecExStateVectorProperty_t, void*, size_t) noexcept nogil>__custatevecExStateVectorGetProperty)(
        stateVector, property, value, sizeInBytes)


cdef custatevecStatus_t _custatevecExStateVectorSetMathMode(custatevecExStateVectorDescriptor_t stateVector, custatevecMathMode_t mode) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorSetMathMode
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorSetMathMode == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorSetMathMode is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, custatevecMathMode_t) noexcept nogil>__custatevecExStateVectorSetMathMode)(
        stateVector, mode)


cdef custatevecStatus_t _custatevecExStateVectorSetZeroState(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorSetZeroState
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorSetZeroState == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorSetZeroState is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t) noexcept nogil>__custatevecExStateVectorSetZeroState)(
        stateVector)


cdef custatevecStatus_t _custatevecExStateVectorGetState(const custatevecExStateVectorDescriptor_t stateVector, void* state, cudaDataType_t dataType, custatevecIndex_t begin, custatevecIndex_t end, int32_t maxNumConcurrentCopies) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorGetState
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorGetState == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorGetState is not found")
    return (<custatevecStatus_t (*)(const custatevecExStateVectorDescriptor_t, void*, cudaDataType_t, custatevecIndex_t, custatevecIndex_t, int32_t) noexcept nogil>__custatevecExStateVectorGetState)(
        stateVector, state, dataType, begin, end, maxNumConcurrentCopies)


cdef custatevecStatus_t _custatevecExStateVectorSetState(custatevecExStateVectorDescriptor_t stateVector, const void* state, cudaDataType_t dataType, custatevecIndex_t begin, custatevecIndex_t end, int32_t maxNumConcurrentCopies) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorSetState
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorSetState == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorSetState is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, const void*, cudaDataType_t, custatevecIndex_t, custatevecIndex_t, int32_t) noexcept nogil>__custatevecExStateVectorSetState)(
        stateVector, state, dataType, begin, end, maxNumConcurrentCopies)


cdef custatevecStatus_t _custatevecExStateVectorReassignWireOrdering(custatevecExStateVectorDescriptor_t stateVector, const int32_t* wireOrdering, int32_t wireOrderingLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorReassignWireOrdering
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorReassignWireOrdering == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorReassignWireOrdering is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, const int32_t*, int32_t) noexcept nogil>__custatevecExStateVectorReassignWireOrdering)(
        stateVector, wireOrdering, wireOrderingLen)


cdef custatevecStatus_t _custatevecExStateVectorPermuteIndexBits(custatevecExStateVectorDescriptor_t stateVector, const int32_t* permutation, int32_t permutationLen, custatevecExPermutationType_t permutationType) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorPermuteIndexBits
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorPermuteIndexBits == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorPermuteIndexBits is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, const int32_t*, int32_t, custatevecExPermutationType_t) noexcept nogil>__custatevecExStateVectorPermuteIndexBits)(
        stateVector, permutation, permutationLen, permutationType)


cdef custatevecStatus_t _custatevecExStateVectorStageSubSV(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorStageSubSV
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorStageSubSV == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorStageSubSV is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, int32_t) noexcept nogil>__custatevecExStateVectorStageSubSV)(
        stateVector, subSVIndex)


cdef custatevecStatus_t _custatevecExStateVectorExposeResources(custatevecExStateVectorDescriptor_t stateVector, custatevecExExposeResources_t exposeResources) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorExposeResources
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorExposeResources == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorExposeResources is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, custatevecExExposeResources_t) noexcept nogil>__custatevecExStateVectorExposeResources)(
        stateVector, exposeResources)


cdef custatevecStatus_t _custatevecExStateVectorGetResourcesFromDeviceSubSV(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t* deviceId, void** d_subSV, cudaStream_t* stream, custatevecHandle_t* handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorGetResourcesFromDeviceSubSV
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorGetResourcesFromDeviceSubSV == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorGetResourcesFromDeviceSubSV is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, int32_t, int32_t*, void**, cudaStream_t*, custatevecHandle_t*) noexcept nogil>__custatevecExStateVectorGetResourcesFromDeviceSubSV)(
        stateVector, subSVIndex, deviceId, d_subSV, stream, handle)


cdef custatevecStatus_t _custatevecExStateVectorGetResourcesFromDeviceSubSVView(const custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t* deviceId, const void** d_subSV, cudaStream_t* stream, custatevecHandle_t* handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorGetResourcesFromDeviceSubSVView
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorGetResourcesFromDeviceSubSVView == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorGetResourcesFromDeviceSubSVView is not found")
    return (<custatevecStatus_t (*)(const custatevecExStateVectorDescriptor_t, int32_t, int32_t*, const void**, cudaStream_t*, custatevecHandle_t*) noexcept nogil>__custatevecExStateVectorGetResourcesFromDeviceSubSVView)(
        stateVector, subSVIndex, deviceId, d_subSV, stream, handle)


cdef custatevecStatus_t _custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice(custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t sliceIndex, void** subSVSlice, custatevecExMemoryPlacement_t* placement, int32_t* deviceId, cudaStream_t* stream) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, int32_t, int32_t, void**, custatevecExMemoryPlacement_t*, int32_t*, cudaStream_t*) noexcept nogil>__custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice)(
        stateVector, subSVIndex, sliceIndex, subSVSlice, placement, deviceId, stream)


cdef custatevecStatus_t _custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView(const custatevecExStateVectorDescriptor_t stateVector, int32_t subSVIndex, int32_t sliceIndex, const void** subSVSlice, custatevecExMemoryPlacement_t* placement, int32_t* deviceId, cudaStream_t* stream) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView is not found")
    return (<custatevecStatus_t (*)(const custatevecExStateVectorDescriptor_t, int32_t, int32_t, const void**, custatevecExMemoryPlacement_t*, int32_t*, cudaStream_t*) noexcept nogil>__custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView)(
        stateVector, subSVIndex, sliceIndex, subSVSlice, placement, deviceId, stream)


cdef custatevecStatus_t _custatevecExStateVectorSynchronize(custatevecExStateVectorDescriptor_t stateVector) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorSynchronize
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorSynchronize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorSynchronize is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t) noexcept nogil>__custatevecExStateVectorSynchronize)(
        stateVector)


cdef custatevecStatus_t _custatevecExStateVectorSynchronizeScoped(custatevecExStateVectorDescriptor_t stateVector, custatevecExSynchronizationScope_t scope) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorSynchronizeScoped
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorSynchronizeScoped == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorSynchronizeScoped is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, custatevecExSynchronizationScope_t) noexcept nogil>__custatevecExStateVectorSynchronizeScoped)(
        stateVector, scope)


cdef custatevecStatus_t _custatevecExStateVectorAddWires(custatevecExStateVectorDescriptor_t stateVector, custatevecExIndexBitDomain_t indexBitDomain, int32_t numWiresToAdd, custatevecExWireInitMode_t wireInitMode, int32_t* wiresAdded) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExStateVectorAddWires
    _check_or_init_custatevecEx()
    if __custatevecExStateVectorAddWires == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExStateVectorAddWires is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, custatevecExIndexBitDomain_t, int32_t, custatevecExWireInitMode_t, int32_t*) noexcept nogil>__custatevecExStateVectorAddWires)(
        stateVector, indexBitDomain, numWiresToAdd, wireInitMode, wiresAdded)


cdef custatevecStatus_t _custatevecExAbs2SumArray(custatevecExStateVectorDescriptor_t stateVector, double* abs2sum, const int32_t* outputOrdering, int32_t outputOrderingLen, const int32_t* maskBitString, const int32_t* maskWireOrdering, int32_t maskLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExAbs2SumArray
    _check_or_init_custatevecEx()
    if __custatevecExAbs2SumArray == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExAbs2SumArray is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, double*, const int32_t*, int32_t, const int32_t*, const int32_t*, int32_t) noexcept nogil>__custatevecExAbs2SumArray)(
        stateVector, abs2sum, outputOrdering, outputOrderingLen, maskBitString, maskWireOrdering, maskLen)


cdef custatevecStatus_t _custatevecExMeasure(custatevecExStateVectorDescriptor_t stateVector, custatevecIndex_t* bitString, const int32_t* bitStringOrdering, int32_t bitStringOrderingLen, double randnum, custatevecCollapseOp_t collapse, const void* reserved) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExMeasure
    _check_or_init_custatevecEx()
    if __custatevecExMeasure == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExMeasure is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, custatevecIndex_t*, const int32_t*, int32_t, double, custatevecCollapseOp_t, const void*) noexcept nogil>__custatevecExMeasure)(
        stateVector, bitString, bitStringOrdering, bitStringOrderingLen, randnum, collapse, reserved)


cdef custatevecStatus_t _custatevecExSample(custatevecExStateVectorDescriptor_t stateVector, custatevecIndex_t* bitStrings, const int32_t* bitStringOrdering, int32_t bitStringOrderingLen, const double* randnums, int32_t numShots, custatevecSamplerOutput_t output, const double* abs2Sums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExSample
    _check_or_init_custatevecEx()
    if __custatevecExSample == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExSample is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, custatevecIndex_t*, const int32_t*, int32_t, const double*, int32_t, custatevecSamplerOutput_t, const double*) noexcept nogil>__custatevecExSample)(
        stateVector, bitStrings, bitStringOrdering, bitStringOrderingLen, randnums, numShots, output, abs2Sums)


cdef custatevecStatus_t _custatevecExApplyMatrix(custatevecExStateVectorDescriptor_t stateVector, const void* matrix, cudaDataType_t matrixDataType, custatevecExMatrixType_t exMatrixType, custatevecMatrixLayout_t layout, int32_t adjoint, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExApplyMatrix
    _check_or_init_custatevecEx()
    if __custatevecExApplyMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExApplyMatrix is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, const void*, cudaDataType_t, custatevecExMatrixType_t, custatevecMatrixLayout_t, int32_t, const int32_t*, int32_t, const int32_t*, const int32_t*, int32_t) noexcept nogil>__custatevecExApplyMatrix)(
        stateVector, matrix, matrixDataType, exMatrixType, layout, adjoint, targets, numTargets, controls, controlBitValues, numControls)


cdef custatevecStatus_t _custatevecExApplyPauliRotation(custatevecExStateVectorDescriptor_t stateVector, double theta, const custatevecPauli_t* paulis, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExApplyPauliRotation
    _check_or_init_custatevecEx()
    if __custatevecExApplyPauliRotation == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExApplyPauliRotation is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, double, const custatevecPauli_t*, const int32_t*, int32_t, const int32_t*, const int32_t*, int32_t) noexcept nogil>__custatevecExApplyPauliRotation)(
        stateVector, theta, paulis, targets, numTargets, controls, controlBitValues, numControls)


cdef custatevecStatus_t _custatevecExComputeExpectationOnPauliBasis(custatevecExStateVectorDescriptor_t stateVector, double* expectationValues, const custatevecPauli_t** pauliOperatorArrays, int32_t numPauliOperatorArrays, const int32_t** basisWiresArray, const int32_t* numBasisWiresArray) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExComputeExpectationOnPauliBasis
    _check_or_init_custatevecEx()
    if __custatevecExComputeExpectationOnPauliBasis == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExComputeExpectationOnPauliBasis is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, double*, const custatevecPauli_t**, int32_t, const int32_t**, const int32_t*) noexcept nogil>__custatevecExComputeExpectationOnPauliBasis)(
        stateVector, expectationValues, pauliOperatorArrays, numPauliOperatorArrays, basisWiresArray, numBasisWiresArray)


cdef custatevecStatus_t _custatevecExComputeExpectation(custatevecExStateVectorDescriptor_t stateVector, double2* expectationValues, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, int32_t numMatrices, const int32_t* basisWires, int32_t numBasisWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExComputeExpectation
    _check_or_init_custatevecEx()
    if __custatevecExComputeExpectation == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExComputeExpectation is not found")
    return (<custatevecStatus_t (*)(custatevecExStateVectorDescriptor_t, double2*, const void*, cudaDataType_t, custatevecMatrixLayout_t, int32_t, const int32_t*, int32_t) noexcept nogil>__custatevecExComputeExpectation)(
        stateVector, expectationValues, matrices, matrixDataType, layout, numMatrices, basisWires, numBasisWires)


cdef custatevecStatus_t _custatevecExConfigureSVUpdater(custatevecExDictionaryDescriptor_t* svUpdaterConfig, cudaDataType_t dataType, const custatevecExSVUpdaterConfigItem_t* configItems, int32_t numConfigItems) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExConfigureSVUpdater
    _check_or_init_custatevecEx()
    if __custatevecExConfigureSVUpdater == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExConfigureSVUpdater is not found")
    return (<custatevecStatus_t (*)(custatevecExDictionaryDescriptor_t*, cudaDataType_t, const custatevecExSVUpdaterConfigItem_t*, int32_t) noexcept nogil>__custatevecExConfigureSVUpdater)(
        svUpdaterConfig, dataType, configItems, numConfigItems)


cdef custatevecStatus_t _custatevecExSVUpdaterCreate(custatevecExSVUpdaterDescriptor_t* svUpdater, const custatevecExDictionaryDescriptor_t svUpdaterConfig, custatevecExResourceManagerDescriptor_t resourceManager) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExSVUpdaterCreate
    _check_or_init_custatevecEx()
    if __custatevecExSVUpdaterCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExSVUpdaterCreate is not found")
    return (<custatevecStatus_t (*)(custatevecExSVUpdaterDescriptor_t*, const custatevecExDictionaryDescriptor_t, custatevecExResourceManagerDescriptor_t) noexcept nogil>__custatevecExSVUpdaterCreate)(
        svUpdater, svUpdaterConfig, resourceManager)


cdef custatevecStatus_t _custatevecExSVUpdaterDestroy(custatevecExSVUpdaterDescriptor_t svUpdater) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExSVUpdaterDestroy
    _check_or_init_custatevecEx()
    if __custatevecExSVUpdaterDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExSVUpdaterDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecExSVUpdaterDescriptor_t) noexcept nogil>__custatevecExSVUpdaterDestroy)(
        svUpdater)


cdef custatevecStatus_t _custatevecExSVUpdaterClear(custatevecExSVUpdaterDescriptor_t svUpdater) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExSVUpdaterClear
    _check_or_init_custatevecEx()
    if __custatevecExSVUpdaterClear == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExSVUpdaterClear is not found")
    return (<custatevecStatus_t (*)(custatevecExSVUpdaterDescriptor_t) noexcept nogil>__custatevecExSVUpdaterClear)(
        svUpdater)


cdef custatevecStatus_t _custatevecExSVUpdaterEnqueueMatrix(custatevecExSVUpdaterDescriptor_t svUpdater, const void* matrix, cudaDataType_t matrixDataType, custatevecExMatrixType_t exMatrixType, custatevecMatrixLayout_t layout, int32_t adjoint, const int32_t* targets, int32_t numTargets, const int32_t* controls, const int32_t* controlBitValues, int32_t numControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExSVUpdaterEnqueueMatrix
    _check_or_init_custatevecEx()
    if __custatevecExSVUpdaterEnqueueMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExSVUpdaterEnqueueMatrix is not found")
    return (<custatevecStatus_t (*)(custatevecExSVUpdaterDescriptor_t, const void*, cudaDataType_t, custatevecExMatrixType_t, custatevecMatrixLayout_t, int32_t, const int32_t*, int32_t, const int32_t*, const int32_t*, int32_t) noexcept nogil>__custatevecExSVUpdaterEnqueueMatrix)(
        svUpdater, matrix, matrixDataType, exMatrixType, layout, adjoint, targets, numTargets, controls, controlBitValues, numControls)


cdef custatevecStatus_t _custatevecExSVUpdaterEnqueueUnitaryChannel(custatevecExSVUpdaterDescriptor_t svUpdater, const void* const* unitaries, cudaDataType_t unitariesDataType, const custatevecExMatrixType_t* exMatrixTypes, int32_t numUnitaries, custatevecMatrixLayout_t layout, const double* probabilities, const int32_t* channelWires, int32_t numChannelWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExSVUpdaterEnqueueUnitaryChannel
    _check_or_init_custatevecEx()
    if __custatevecExSVUpdaterEnqueueUnitaryChannel == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExSVUpdaterEnqueueUnitaryChannel is not found")
    return (<custatevecStatus_t (*)(custatevecExSVUpdaterDescriptor_t, const void* const*, cudaDataType_t, const custatevecExMatrixType_t*, int32_t, custatevecMatrixLayout_t, const double*, const int32_t*, int32_t) noexcept nogil>__custatevecExSVUpdaterEnqueueUnitaryChannel)(
        svUpdater, unitaries, unitariesDataType, exMatrixTypes, numUnitaries, layout, probabilities, channelWires, numChannelWires)


cdef custatevecStatus_t _custatevecExSVUpdaterEnqueueGeneralChannel(custatevecExSVUpdaterDescriptor_t svUpdater, const void* const* matrices, cudaDataType_t matrixDataType, const custatevecExMatrixType_t* exMatrixTypes, int32_t numMatrices, custatevecMatrixLayout_t layout, const int32_t* channelWires, int32_t numChannelWires) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExSVUpdaterEnqueueGeneralChannel
    _check_or_init_custatevecEx()
    if __custatevecExSVUpdaterEnqueueGeneralChannel == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExSVUpdaterEnqueueGeneralChannel is not found")
    return (<custatevecStatus_t (*)(custatevecExSVUpdaterDescriptor_t, const void* const*, cudaDataType_t, const custatevecExMatrixType_t*, int32_t, custatevecMatrixLayout_t, const int32_t*, int32_t) noexcept nogil>__custatevecExSVUpdaterEnqueueGeneralChannel)(
        svUpdater, matrices, matrixDataType, exMatrixTypes, numMatrices, layout, channelWires, numChannelWires)


cdef custatevecStatus_t _custatevecExSVUpdaterGetMaxNumRequiredRandnums(custatevecExSVUpdaterDescriptor_t svUpdater, int32_t* maxNumRequiredRandnums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExSVUpdaterGetMaxNumRequiredRandnums
    _check_or_init_custatevecEx()
    if __custatevecExSVUpdaterGetMaxNumRequiredRandnums == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExSVUpdaterGetMaxNumRequiredRandnums is not found")
    return (<custatevecStatus_t (*)(custatevecExSVUpdaterDescriptor_t, int32_t*) noexcept nogil>__custatevecExSVUpdaterGetMaxNumRequiredRandnums)(
        svUpdater, maxNumRequiredRandnums)


cdef custatevecStatus_t _custatevecExSVUpdaterApply(custatevecExSVUpdaterDescriptor_t svUpdater, custatevecExStateVectorDescriptor_t stateVector, const double* randnums, int32_t numRandnums) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecExSVUpdaterApply
    _check_or_init_custatevecEx()
    if __custatevecExSVUpdaterApply == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecExSVUpdaterApply is not found")
    return (<custatevecStatus_t (*)(custatevecExSVUpdaterDescriptor_t, custatevecExStateVectorDescriptor_t, const double*, int32_t) noexcept nogil>__custatevecExSVUpdaterApply)(
        svUpdater, stateVector, randnums, numRandnums)
