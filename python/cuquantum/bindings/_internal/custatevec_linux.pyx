# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 26.11.0. Do not modify it directly.



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

cdef int _cyb___py_custatevec_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .._utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################


cdef void* __custatevecCreate = NULL
cdef void* __custatevecDestroy = NULL
cdef void* __custatevecGetDefaultWorkspaceSize = NULL
cdef void* __custatevecSetWorkspace = NULL
cdef void* __custatevecGetErrorName = NULL
cdef void* __custatevecGetErrorString = NULL
cdef void* __custatevecGetProperty = NULL
cdef void* __custatevecGetVersion = NULL
cdef void* __custatevecSetStream = NULL
cdef void* __custatevecGetStream = NULL
cdef void* __custatevecLoggerSetCallback = NULL
cdef void* __custatevecLoggerSetCallbackData = NULL
cdef void* __custatevecLoggerSetFile = NULL
cdef void* __custatevecLoggerOpenFile = NULL
cdef void* __custatevecLoggerSetLevel = NULL
cdef void* __custatevecLoggerSetMask = NULL
cdef void* __custatevecLoggerForceDisable = NULL
cdef void* __custatevecGetDeviceMemHandler = NULL
cdef void* __custatevecSetDeviceMemHandler = NULL
cdef void* __custatevecAbs2SumOnZBasis = NULL
cdef void* __custatevecAbs2SumArray = NULL
cdef void* __custatevecCollapseOnZBasis = NULL
cdef void* __custatevecCollapseByBitString = NULL
cdef void* __custatevecMeasureOnZBasis = NULL
cdef void* __custatevecBatchMeasure = NULL
cdef void* __custatevecBatchMeasureWithOffset = NULL
cdef void* __custatevecApplyPauliRotation = NULL
cdef void* __custatevecApplyMatrixGetWorkspaceSize = NULL
cdef void* __custatevecApplyMatrix = NULL
cdef void* __custatevecComputeExpectationGetWorkspaceSize = NULL
cdef void* __custatevecComputeExpectation = NULL
cdef void* __custatevecSamplerCreate = NULL
cdef void* __custatevecSamplerDestroy = NULL
cdef void* __custatevecSamplerPreprocess = NULL
cdef void* __custatevecSamplerGetSquaredNorm = NULL
cdef void* __custatevecSamplerApplySubSVOffset = NULL
cdef void* __custatevecSamplerSample = NULL
cdef void* __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize = NULL
cdef void* __custatevecApplyGeneralizedPermutationMatrix = NULL
cdef void* __custatevecComputeExpectationsOnPauliBasis = NULL
cdef void* __custatevecAccessorCreate = NULL
cdef void* __custatevecAccessorCreateView = NULL
cdef void* __custatevecAccessorDestroy = NULL
cdef void* __custatevecAccessorSetExtraWorkspace = NULL
cdef void* __custatevecAccessorGet = NULL
cdef void* __custatevecAccessorSet = NULL
cdef void* __custatevecSwapIndexBits = NULL
cdef void* __custatevecTestMatrixTypeGetWorkspaceSize = NULL
cdef void* __custatevecTestMatrixType = NULL
cdef void* __custatevecMultiDeviceSwapIndexBits = NULL
cdef void* __custatevecCommunicatorCreate = NULL
cdef void* __custatevecCommunicatorDestroy = NULL
cdef void* __custatevecDistIndexBitSwapSchedulerCreate = NULL
cdef void* __custatevecDistIndexBitSwapSchedulerDestroy = NULL
cdef void* __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps = NULL
cdef void* __custatevecDistIndexBitSwapSchedulerGetParameters = NULL
cdef void* __custatevecSVSwapWorkerCreate = NULL
cdef void* __custatevecSVSwapWorkerDestroy = NULL
cdef void* __custatevecSVSwapWorkerSetExtraWorkspace = NULL
cdef void* __custatevecSVSwapWorkerSetTransferWorkspace = NULL
cdef void* __custatevecSVSwapWorkerSetSubSVsP2P = NULL
cdef void* __custatevecSVSwapWorkerSetParameters = NULL
cdef void* __custatevecSVSwapWorkerExecute = NULL
cdef void* __custatevecInitializeStateVector = NULL
cdef void* __custatevecApplyMatrixBatchedGetWorkspaceSize = NULL
cdef void* __custatevecApplyMatrixBatched = NULL
cdef void* __custatevecAbs2SumArrayBatched = NULL
cdef void* __custatevecCollapseByBitStringBatchedGetWorkspaceSize = NULL
cdef void* __custatevecCollapseByBitStringBatched = NULL
cdef void* __custatevecMeasureBatched = NULL
cdef void* __custatevecSubSVMigratorCreate = NULL
cdef void* __custatevecSubSVMigratorDestroy = NULL
cdef void* __custatevecSubSVMigratorMigrate = NULL
cdef void* __custatevecComputeExpectationBatchedGetWorkspaceSize = NULL
cdef void* __custatevecComputeExpectationBatched = NULL
cdef void* __custatevecSetMathMode = NULL
cdef void* __custatevecGetMathMode = NULL
cdef void* __custatevecSVSwapWorkerCreateWithSemaphore = NULL
cdef void* __custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores = NULL

cdef int _init_custatevec() except -1 nogil:
    global _cyb___py_custatevec_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_custatevec_init: return 0

        global __custatevecCreate
        __custatevecCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecCreate')
        if __custatevecCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecCreate = _cyb_dlsym(handle, 'custatevecCreate')

        global __custatevecDestroy
        __custatevecDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecDestroy')
        if __custatevecDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecDestroy = _cyb_dlsym(handle, 'custatevecDestroy')

        global __custatevecGetDefaultWorkspaceSize
        __custatevecGetDefaultWorkspaceSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecGetDefaultWorkspaceSize')
        if __custatevecGetDefaultWorkspaceSize == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecGetDefaultWorkspaceSize = _cyb_dlsym(handle, 'custatevecGetDefaultWorkspaceSize')

        global __custatevecSetWorkspace
        __custatevecSetWorkspace = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSetWorkspace')
        if __custatevecSetWorkspace == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSetWorkspace = _cyb_dlsym(handle, 'custatevecSetWorkspace')

        global __custatevecGetErrorName
        __custatevecGetErrorName = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecGetErrorName')
        if __custatevecGetErrorName == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecGetErrorName = _cyb_dlsym(handle, 'custatevecGetErrorName')

        global __custatevecGetErrorString
        __custatevecGetErrorString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecGetErrorString')
        if __custatevecGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecGetErrorString = _cyb_dlsym(handle, 'custatevecGetErrorString')

        global __custatevecGetProperty
        __custatevecGetProperty = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecGetProperty')
        if __custatevecGetProperty == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecGetProperty = _cyb_dlsym(handle, 'custatevecGetProperty')

        global __custatevecGetVersion
        __custatevecGetVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecGetVersion')
        if __custatevecGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecGetVersion = _cyb_dlsym(handle, 'custatevecGetVersion')

        global __custatevecSetStream
        __custatevecSetStream = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSetStream')
        if __custatevecSetStream == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSetStream = _cyb_dlsym(handle, 'custatevecSetStream')

        global __custatevecGetStream
        __custatevecGetStream = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecGetStream')
        if __custatevecGetStream == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecGetStream = _cyb_dlsym(handle, 'custatevecGetStream')

        global __custatevecLoggerSetCallback
        __custatevecLoggerSetCallback = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecLoggerSetCallback')
        if __custatevecLoggerSetCallback == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecLoggerSetCallback = _cyb_dlsym(handle, 'custatevecLoggerSetCallback')

        global __custatevecLoggerSetCallbackData
        __custatevecLoggerSetCallbackData = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecLoggerSetCallbackData')
        if __custatevecLoggerSetCallbackData == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecLoggerSetCallbackData = _cyb_dlsym(handle, 'custatevecLoggerSetCallbackData')

        global __custatevecLoggerSetFile
        __custatevecLoggerSetFile = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecLoggerSetFile')
        if __custatevecLoggerSetFile == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecLoggerSetFile = _cyb_dlsym(handle, 'custatevecLoggerSetFile')

        global __custatevecLoggerOpenFile
        __custatevecLoggerOpenFile = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecLoggerOpenFile')
        if __custatevecLoggerOpenFile == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecLoggerOpenFile = _cyb_dlsym(handle, 'custatevecLoggerOpenFile')

        global __custatevecLoggerSetLevel
        __custatevecLoggerSetLevel = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecLoggerSetLevel')
        if __custatevecLoggerSetLevel == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecLoggerSetLevel = _cyb_dlsym(handle, 'custatevecLoggerSetLevel')

        global __custatevecLoggerSetMask
        __custatevecLoggerSetMask = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecLoggerSetMask')
        if __custatevecLoggerSetMask == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecLoggerSetMask = _cyb_dlsym(handle, 'custatevecLoggerSetMask')

        global __custatevecLoggerForceDisable
        __custatevecLoggerForceDisable = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecLoggerForceDisable')
        if __custatevecLoggerForceDisable == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecLoggerForceDisable = _cyb_dlsym(handle, 'custatevecLoggerForceDisable')

        global __custatevecGetDeviceMemHandler
        __custatevecGetDeviceMemHandler = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecGetDeviceMemHandler')
        if __custatevecGetDeviceMemHandler == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecGetDeviceMemHandler = _cyb_dlsym(handle, 'custatevecGetDeviceMemHandler')

        global __custatevecSetDeviceMemHandler
        __custatevecSetDeviceMemHandler = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSetDeviceMemHandler')
        if __custatevecSetDeviceMemHandler == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSetDeviceMemHandler = _cyb_dlsym(handle, 'custatevecSetDeviceMemHandler')

        global __custatevecAbs2SumOnZBasis
        __custatevecAbs2SumOnZBasis = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecAbs2SumOnZBasis')
        if __custatevecAbs2SumOnZBasis == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecAbs2SumOnZBasis = _cyb_dlsym(handle, 'custatevecAbs2SumOnZBasis')

        global __custatevecAbs2SumArray
        __custatevecAbs2SumArray = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecAbs2SumArray')
        if __custatevecAbs2SumArray == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecAbs2SumArray = _cyb_dlsym(handle, 'custatevecAbs2SumArray')

        global __custatevecCollapseOnZBasis
        __custatevecCollapseOnZBasis = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecCollapseOnZBasis')
        if __custatevecCollapseOnZBasis == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecCollapseOnZBasis = _cyb_dlsym(handle, 'custatevecCollapseOnZBasis')

        global __custatevecCollapseByBitString
        __custatevecCollapseByBitString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecCollapseByBitString')
        if __custatevecCollapseByBitString == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecCollapseByBitString = _cyb_dlsym(handle, 'custatevecCollapseByBitString')

        global __custatevecMeasureOnZBasis
        __custatevecMeasureOnZBasis = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecMeasureOnZBasis')
        if __custatevecMeasureOnZBasis == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecMeasureOnZBasis = _cyb_dlsym(handle, 'custatevecMeasureOnZBasis')

        global __custatevecBatchMeasure
        __custatevecBatchMeasure = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecBatchMeasure')
        if __custatevecBatchMeasure == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecBatchMeasure = _cyb_dlsym(handle, 'custatevecBatchMeasure')

        global __custatevecBatchMeasureWithOffset
        __custatevecBatchMeasureWithOffset = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecBatchMeasureWithOffset')
        if __custatevecBatchMeasureWithOffset == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecBatchMeasureWithOffset = _cyb_dlsym(handle, 'custatevecBatchMeasureWithOffset')

        global __custatevecApplyPauliRotation
        __custatevecApplyPauliRotation = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecApplyPauliRotation')
        if __custatevecApplyPauliRotation == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecApplyPauliRotation = _cyb_dlsym(handle, 'custatevecApplyPauliRotation')

        global __custatevecApplyMatrixGetWorkspaceSize
        __custatevecApplyMatrixGetWorkspaceSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecApplyMatrixGetWorkspaceSize')
        if __custatevecApplyMatrixGetWorkspaceSize == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecApplyMatrixGetWorkspaceSize = _cyb_dlsym(handle, 'custatevecApplyMatrixGetWorkspaceSize')

        global __custatevecApplyMatrix
        __custatevecApplyMatrix = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecApplyMatrix')
        if __custatevecApplyMatrix == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecApplyMatrix = _cyb_dlsym(handle, 'custatevecApplyMatrix')

        global __custatevecComputeExpectationGetWorkspaceSize
        __custatevecComputeExpectationGetWorkspaceSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecComputeExpectationGetWorkspaceSize')
        if __custatevecComputeExpectationGetWorkspaceSize == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecComputeExpectationGetWorkspaceSize = _cyb_dlsym(handle, 'custatevecComputeExpectationGetWorkspaceSize')

        global __custatevecComputeExpectation
        __custatevecComputeExpectation = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecComputeExpectation')
        if __custatevecComputeExpectation == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecComputeExpectation = _cyb_dlsym(handle, 'custatevecComputeExpectation')

        global __custatevecSamplerCreate
        __custatevecSamplerCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSamplerCreate')
        if __custatevecSamplerCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSamplerCreate = _cyb_dlsym(handle, 'custatevecSamplerCreate')

        global __custatevecSamplerDestroy
        __custatevecSamplerDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSamplerDestroy')
        if __custatevecSamplerDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSamplerDestroy = _cyb_dlsym(handle, 'custatevecSamplerDestroy')

        global __custatevecSamplerPreprocess
        __custatevecSamplerPreprocess = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSamplerPreprocess')
        if __custatevecSamplerPreprocess == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSamplerPreprocess = _cyb_dlsym(handle, 'custatevecSamplerPreprocess')

        global __custatevecSamplerGetSquaredNorm
        __custatevecSamplerGetSquaredNorm = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSamplerGetSquaredNorm')
        if __custatevecSamplerGetSquaredNorm == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSamplerGetSquaredNorm = _cyb_dlsym(handle, 'custatevecSamplerGetSquaredNorm')

        global __custatevecSamplerApplySubSVOffset
        __custatevecSamplerApplySubSVOffset = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSamplerApplySubSVOffset')
        if __custatevecSamplerApplySubSVOffset == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSamplerApplySubSVOffset = _cyb_dlsym(handle, 'custatevecSamplerApplySubSVOffset')

        global __custatevecSamplerSample
        __custatevecSamplerSample = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSamplerSample')
        if __custatevecSamplerSample == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSamplerSample = _cyb_dlsym(handle, 'custatevecSamplerSample')

        global __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize
        __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize')
        if __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize = _cyb_dlsym(handle, 'custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize')

        global __custatevecApplyGeneralizedPermutationMatrix
        __custatevecApplyGeneralizedPermutationMatrix = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecApplyGeneralizedPermutationMatrix')
        if __custatevecApplyGeneralizedPermutationMatrix == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecApplyGeneralizedPermutationMatrix = _cyb_dlsym(handle, 'custatevecApplyGeneralizedPermutationMatrix')

        global __custatevecComputeExpectationsOnPauliBasis
        __custatevecComputeExpectationsOnPauliBasis = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecComputeExpectationsOnPauliBasis')
        if __custatevecComputeExpectationsOnPauliBasis == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecComputeExpectationsOnPauliBasis = _cyb_dlsym(handle, 'custatevecComputeExpectationsOnPauliBasis')

        global __custatevecAccessorCreate
        __custatevecAccessorCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecAccessorCreate')
        if __custatevecAccessorCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecAccessorCreate = _cyb_dlsym(handle, 'custatevecAccessorCreate')

        global __custatevecAccessorCreateView
        __custatevecAccessorCreateView = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecAccessorCreateView')
        if __custatevecAccessorCreateView == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecAccessorCreateView = _cyb_dlsym(handle, 'custatevecAccessorCreateView')

        global __custatevecAccessorDestroy
        __custatevecAccessorDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecAccessorDestroy')
        if __custatevecAccessorDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecAccessorDestroy = _cyb_dlsym(handle, 'custatevecAccessorDestroy')

        global __custatevecAccessorSetExtraWorkspace
        __custatevecAccessorSetExtraWorkspace = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecAccessorSetExtraWorkspace')
        if __custatevecAccessorSetExtraWorkspace == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecAccessorSetExtraWorkspace = _cyb_dlsym(handle, 'custatevecAccessorSetExtraWorkspace')

        global __custatevecAccessorGet
        __custatevecAccessorGet = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecAccessorGet')
        if __custatevecAccessorGet == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecAccessorGet = _cyb_dlsym(handle, 'custatevecAccessorGet')

        global __custatevecAccessorSet
        __custatevecAccessorSet = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecAccessorSet')
        if __custatevecAccessorSet == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecAccessorSet = _cyb_dlsym(handle, 'custatevecAccessorSet')

        global __custatevecSwapIndexBits
        __custatevecSwapIndexBits = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSwapIndexBits')
        if __custatevecSwapIndexBits == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSwapIndexBits = _cyb_dlsym(handle, 'custatevecSwapIndexBits')

        global __custatevecTestMatrixTypeGetWorkspaceSize
        __custatevecTestMatrixTypeGetWorkspaceSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecTestMatrixTypeGetWorkspaceSize')
        if __custatevecTestMatrixTypeGetWorkspaceSize == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecTestMatrixTypeGetWorkspaceSize = _cyb_dlsym(handle, 'custatevecTestMatrixTypeGetWorkspaceSize')

        global __custatevecTestMatrixType
        __custatevecTestMatrixType = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecTestMatrixType')
        if __custatevecTestMatrixType == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecTestMatrixType = _cyb_dlsym(handle, 'custatevecTestMatrixType')

        global __custatevecMultiDeviceSwapIndexBits
        __custatevecMultiDeviceSwapIndexBits = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecMultiDeviceSwapIndexBits')
        if __custatevecMultiDeviceSwapIndexBits == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecMultiDeviceSwapIndexBits = _cyb_dlsym(handle, 'custatevecMultiDeviceSwapIndexBits')

        global __custatevecCommunicatorCreate
        __custatevecCommunicatorCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecCommunicatorCreate')
        if __custatevecCommunicatorCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecCommunicatorCreate = _cyb_dlsym(handle, 'custatevecCommunicatorCreate')

        global __custatevecCommunicatorDestroy
        __custatevecCommunicatorDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecCommunicatorDestroy')
        if __custatevecCommunicatorDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecCommunicatorDestroy = _cyb_dlsym(handle, 'custatevecCommunicatorDestroy')

        global __custatevecDistIndexBitSwapSchedulerCreate
        __custatevecDistIndexBitSwapSchedulerCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecDistIndexBitSwapSchedulerCreate')
        if __custatevecDistIndexBitSwapSchedulerCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecDistIndexBitSwapSchedulerCreate = _cyb_dlsym(handle, 'custatevecDistIndexBitSwapSchedulerCreate')

        global __custatevecDistIndexBitSwapSchedulerDestroy
        __custatevecDistIndexBitSwapSchedulerDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecDistIndexBitSwapSchedulerDestroy')
        if __custatevecDistIndexBitSwapSchedulerDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecDistIndexBitSwapSchedulerDestroy = _cyb_dlsym(handle, 'custatevecDistIndexBitSwapSchedulerDestroy')

        global __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps
        __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps')
        if __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps = _cyb_dlsym(handle, 'custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps')

        global __custatevecDistIndexBitSwapSchedulerGetParameters
        __custatevecDistIndexBitSwapSchedulerGetParameters = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecDistIndexBitSwapSchedulerGetParameters')
        if __custatevecDistIndexBitSwapSchedulerGetParameters == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecDistIndexBitSwapSchedulerGetParameters = _cyb_dlsym(handle, 'custatevecDistIndexBitSwapSchedulerGetParameters')

        global __custatevecSVSwapWorkerCreate
        __custatevecSVSwapWorkerCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSVSwapWorkerCreate')
        if __custatevecSVSwapWorkerCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSVSwapWorkerCreate = _cyb_dlsym(handle, 'custatevecSVSwapWorkerCreate')

        global __custatevecSVSwapWorkerDestroy
        __custatevecSVSwapWorkerDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSVSwapWorkerDestroy')
        if __custatevecSVSwapWorkerDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSVSwapWorkerDestroy = _cyb_dlsym(handle, 'custatevecSVSwapWorkerDestroy')

        global __custatevecSVSwapWorkerSetExtraWorkspace
        __custatevecSVSwapWorkerSetExtraWorkspace = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSVSwapWorkerSetExtraWorkspace')
        if __custatevecSVSwapWorkerSetExtraWorkspace == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSVSwapWorkerSetExtraWorkspace = _cyb_dlsym(handle, 'custatevecSVSwapWorkerSetExtraWorkspace')

        global __custatevecSVSwapWorkerSetTransferWorkspace
        __custatevecSVSwapWorkerSetTransferWorkspace = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSVSwapWorkerSetTransferWorkspace')
        if __custatevecSVSwapWorkerSetTransferWorkspace == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSVSwapWorkerSetTransferWorkspace = _cyb_dlsym(handle, 'custatevecSVSwapWorkerSetTransferWorkspace')

        global __custatevecSVSwapWorkerSetSubSVsP2P
        __custatevecSVSwapWorkerSetSubSVsP2P = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSVSwapWorkerSetSubSVsP2P')
        if __custatevecSVSwapWorkerSetSubSVsP2P == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSVSwapWorkerSetSubSVsP2P = _cyb_dlsym(handle, 'custatevecSVSwapWorkerSetSubSVsP2P')

        global __custatevecSVSwapWorkerSetParameters
        __custatevecSVSwapWorkerSetParameters = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSVSwapWorkerSetParameters')
        if __custatevecSVSwapWorkerSetParameters == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSVSwapWorkerSetParameters = _cyb_dlsym(handle, 'custatevecSVSwapWorkerSetParameters')

        global __custatevecSVSwapWorkerExecute
        __custatevecSVSwapWorkerExecute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSVSwapWorkerExecute')
        if __custatevecSVSwapWorkerExecute == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSVSwapWorkerExecute = _cyb_dlsym(handle, 'custatevecSVSwapWorkerExecute')

        global __custatevecInitializeStateVector
        __custatevecInitializeStateVector = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecInitializeStateVector')
        if __custatevecInitializeStateVector == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecInitializeStateVector = _cyb_dlsym(handle, 'custatevecInitializeStateVector')

        global __custatevecApplyMatrixBatchedGetWorkspaceSize
        __custatevecApplyMatrixBatchedGetWorkspaceSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecApplyMatrixBatchedGetWorkspaceSize')
        if __custatevecApplyMatrixBatchedGetWorkspaceSize == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecApplyMatrixBatchedGetWorkspaceSize = _cyb_dlsym(handle, 'custatevecApplyMatrixBatchedGetWorkspaceSize')

        global __custatevecApplyMatrixBatched
        __custatevecApplyMatrixBatched = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecApplyMatrixBatched')
        if __custatevecApplyMatrixBatched == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecApplyMatrixBatched = _cyb_dlsym(handle, 'custatevecApplyMatrixBatched')

        global __custatevecAbs2SumArrayBatched
        __custatevecAbs2SumArrayBatched = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecAbs2SumArrayBatched')
        if __custatevecAbs2SumArrayBatched == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecAbs2SumArrayBatched = _cyb_dlsym(handle, 'custatevecAbs2SumArrayBatched')

        global __custatevecCollapseByBitStringBatchedGetWorkspaceSize
        __custatevecCollapseByBitStringBatchedGetWorkspaceSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecCollapseByBitStringBatchedGetWorkspaceSize')
        if __custatevecCollapseByBitStringBatchedGetWorkspaceSize == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecCollapseByBitStringBatchedGetWorkspaceSize = _cyb_dlsym(handle, 'custatevecCollapseByBitStringBatchedGetWorkspaceSize')

        global __custatevecCollapseByBitStringBatched
        __custatevecCollapseByBitStringBatched = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecCollapseByBitStringBatched')
        if __custatevecCollapseByBitStringBatched == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecCollapseByBitStringBatched = _cyb_dlsym(handle, 'custatevecCollapseByBitStringBatched')

        global __custatevecMeasureBatched
        __custatevecMeasureBatched = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecMeasureBatched')
        if __custatevecMeasureBatched == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecMeasureBatched = _cyb_dlsym(handle, 'custatevecMeasureBatched')

        global __custatevecSubSVMigratorCreate
        __custatevecSubSVMigratorCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSubSVMigratorCreate')
        if __custatevecSubSVMigratorCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSubSVMigratorCreate = _cyb_dlsym(handle, 'custatevecSubSVMigratorCreate')

        global __custatevecSubSVMigratorDestroy
        __custatevecSubSVMigratorDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSubSVMigratorDestroy')
        if __custatevecSubSVMigratorDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSubSVMigratorDestroy = _cyb_dlsym(handle, 'custatevecSubSVMigratorDestroy')

        global __custatevecSubSVMigratorMigrate
        __custatevecSubSVMigratorMigrate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSubSVMigratorMigrate')
        if __custatevecSubSVMigratorMigrate == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSubSVMigratorMigrate = _cyb_dlsym(handle, 'custatevecSubSVMigratorMigrate')

        global __custatevecComputeExpectationBatchedGetWorkspaceSize
        __custatevecComputeExpectationBatchedGetWorkspaceSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecComputeExpectationBatchedGetWorkspaceSize')
        if __custatevecComputeExpectationBatchedGetWorkspaceSize == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecComputeExpectationBatchedGetWorkspaceSize = _cyb_dlsym(handle, 'custatevecComputeExpectationBatchedGetWorkspaceSize')

        global __custatevecComputeExpectationBatched
        __custatevecComputeExpectationBatched = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecComputeExpectationBatched')
        if __custatevecComputeExpectationBatched == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecComputeExpectationBatched = _cyb_dlsym(handle, 'custatevecComputeExpectationBatched')

        global __custatevecSetMathMode
        __custatevecSetMathMode = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSetMathMode')
        if __custatevecSetMathMode == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSetMathMode = _cyb_dlsym(handle, 'custatevecSetMathMode')

        global __custatevecGetMathMode
        __custatevecGetMathMode = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecGetMathMode')
        if __custatevecGetMathMode == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecGetMathMode = _cyb_dlsym(handle, 'custatevecGetMathMode')

        global __custatevecSVSwapWorkerCreateWithSemaphore
        __custatevecSVSwapWorkerCreateWithSemaphore = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSVSwapWorkerCreateWithSemaphore')
        if __custatevecSVSwapWorkerCreateWithSemaphore == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSVSwapWorkerCreateWithSemaphore = _cyb_dlsym(handle, 'custatevecSVSwapWorkerCreateWithSemaphore')

        global __custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores
        __custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores')
        if __custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores == NULL:
            if handle == NULL:
                handle = load_library()
            __custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores = _cyb_dlsym(handle, 'custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores')

        _cyb_atomic_int_store(<int *>&_cyb___py_custatevec_init, 1)
        return 0

cdef inline int _check_or_init_custatevec() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_custatevec_init):
        return 0

    return _init_custatevec()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_custatevec()
    cdef dict data = {}
    global __custatevecCreate
    data["__custatevecCreate"] = <intptr_t>__custatevecCreate

    global __custatevecDestroy
    data["__custatevecDestroy"] = <intptr_t>__custatevecDestroy

    global __custatevecGetDefaultWorkspaceSize
    data["__custatevecGetDefaultWorkspaceSize"] = <intptr_t>__custatevecGetDefaultWorkspaceSize

    global __custatevecSetWorkspace
    data["__custatevecSetWorkspace"] = <intptr_t>__custatevecSetWorkspace

    global __custatevecGetErrorName
    data["__custatevecGetErrorName"] = <intptr_t>__custatevecGetErrorName

    global __custatevecGetErrorString
    data["__custatevecGetErrorString"] = <intptr_t>__custatevecGetErrorString

    global __custatevecGetProperty
    data["__custatevecGetProperty"] = <intptr_t>__custatevecGetProperty

    global __custatevecGetVersion
    data["__custatevecGetVersion"] = <intptr_t>__custatevecGetVersion

    global __custatevecSetStream
    data["__custatevecSetStream"] = <intptr_t>__custatevecSetStream

    global __custatevecGetStream
    data["__custatevecGetStream"] = <intptr_t>__custatevecGetStream

    global __custatevecLoggerSetCallback
    data["__custatevecLoggerSetCallback"] = <intptr_t>__custatevecLoggerSetCallback

    global __custatevecLoggerSetCallbackData
    data["__custatevecLoggerSetCallbackData"] = <intptr_t>__custatevecLoggerSetCallbackData

    global __custatevecLoggerSetFile
    data["__custatevecLoggerSetFile"] = <intptr_t>__custatevecLoggerSetFile

    global __custatevecLoggerOpenFile
    data["__custatevecLoggerOpenFile"] = <intptr_t>__custatevecLoggerOpenFile

    global __custatevecLoggerSetLevel
    data["__custatevecLoggerSetLevel"] = <intptr_t>__custatevecLoggerSetLevel

    global __custatevecLoggerSetMask
    data["__custatevecLoggerSetMask"] = <intptr_t>__custatevecLoggerSetMask

    global __custatevecLoggerForceDisable
    data["__custatevecLoggerForceDisable"] = <intptr_t>__custatevecLoggerForceDisable

    global __custatevecGetDeviceMemHandler
    data["__custatevecGetDeviceMemHandler"] = <intptr_t>__custatevecGetDeviceMemHandler

    global __custatevecSetDeviceMemHandler
    data["__custatevecSetDeviceMemHandler"] = <intptr_t>__custatevecSetDeviceMemHandler

    global __custatevecAbs2SumOnZBasis
    data["__custatevecAbs2SumOnZBasis"] = <intptr_t>__custatevecAbs2SumOnZBasis

    global __custatevecAbs2SumArray
    data["__custatevecAbs2SumArray"] = <intptr_t>__custatevecAbs2SumArray

    global __custatevecCollapseOnZBasis
    data["__custatevecCollapseOnZBasis"] = <intptr_t>__custatevecCollapseOnZBasis

    global __custatevecCollapseByBitString
    data["__custatevecCollapseByBitString"] = <intptr_t>__custatevecCollapseByBitString

    global __custatevecMeasureOnZBasis
    data["__custatevecMeasureOnZBasis"] = <intptr_t>__custatevecMeasureOnZBasis

    global __custatevecBatchMeasure
    data["__custatevecBatchMeasure"] = <intptr_t>__custatevecBatchMeasure

    global __custatevecBatchMeasureWithOffset
    data["__custatevecBatchMeasureWithOffset"] = <intptr_t>__custatevecBatchMeasureWithOffset

    global __custatevecApplyPauliRotation
    data["__custatevecApplyPauliRotation"] = <intptr_t>__custatevecApplyPauliRotation

    global __custatevecApplyMatrixGetWorkspaceSize
    data["__custatevecApplyMatrixGetWorkspaceSize"] = <intptr_t>__custatevecApplyMatrixGetWorkspaceSize

    global __custatevecApplyMatrix
    data["__custatevecApplyMatrix"] = <intptr_t>__custatevecApplyMatrix

    global __custatevecComputeExpectationGetWorkspaceSize
    data["__custatevecComputeExpectationGetWorkspaceSize"] = <intptr_t>__custatevecComputeExpectationGetWorkspaceSize

    global __custatevecComputeExpectation
    data["__custatevecComputeExpectation"] = <intptr_t>__custatevecComputeExpectation

    global __custatevecSamplerCreate
    data["__custatevecSamplerCreate"] = <intptr_t>__custatevecSamplerCreate

    global __custatevecSamplerDestroy
    data["__custatevecSamplerDestroy"] = <intptr_t>__custatevecSamplerDestroy

    global __custatevecSamplerPreprocess
    data["__custatevecSamplerPreprocess"] = <intptr_t>__custatevecSamplerPreprocess

    global __custatevecSamplerGetSquaredNorm
    data["__custatevecSamplerGetSquaredNorm"] = <intptr_t>__custatevecSamplerGetSquaredNorm

    global __custatevecSamplerApplySubSVOffset
    data["__custatevecSamplerApplySubSVOffset"] = <intptr_t>__custatevecSamplerApplySubSVOffset

    global __custatevecSamplerSample
    data["__custatevecSamplerSample"] = <intptr_t>__custatevecSamplerSample

    global __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize
    data["__custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize"] = <intptr_t>__custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize

    global __custatevecApplyGeneralizedPermutationMatrix
    data["__custatevecApplyGeneralizedPermutationMatrix"] = <intptr_t>__custatevecApplyGeneralizedPermutationMatrix

    global __custatevecComputeExpectationsOnPauliBasis
    data["__custatevecComputeExpectationsOnPauliBasis"] = <intptr_t>__custatevecComputeExpectationsOnPauliBasis

    global __custatevecAccessorCreate
    data["__custatevecAccessorCreate"] = <intptr_t>__custatevecAccessorCreate

    global __custatevecAccessorCreateView
    data["__custatevecAccessorCreateView"] = <intptr_t>__custatevecAccessorCreateView

    global __custatevecAccessorDestroy
    data["__custatevecAccessorDestroy"] = <intptr_t>__custatevecAccessorDestroy

    global __custatevecAccessorSetExtraWorkspace
    data["__custatevecAccessorSetExtraWorkspace"] = <intptr_t>__custatevecAccessorSetExtraWorkspace

    global __custatevecAccessorGet
    data["__custatevecAccessorGet"] = <intptr_t>__custatevecAccessorGet

    global __custatevecAccessorSet
    data["__custatevecAccessorSet"] = <intptr_t>__custatevecAccessorSet

    global __custatevecSwapIndexBits
    data["__custatevecSwapIndexBits"] = <intptr_t>__custatevecSwapIndexBits

    global __custatevecTestMatrixTypeGetWorkspaceSize
    data["__custatevecTestMatrixTypeGetWorkspaceSize"] = <intptr_t>__custatevecTestMatrixTypeGetWorkspaceSize

    global __custatevecTestMatrixType
    data["__custatevecTestMatrixType"] = <intptr_t>__custatevecTestMatrixType

    global __custatevecMultiDeviceSwapIndexBits
    data["__custatevecMultiDeviceSwapIndexBits"] = <intptr_t>__custatevecMultiDeviceSwapIndexBits

    global __custatevecCommunicatorCreate
    data["__custatevecCommunicatorCreate"] = <intptr_t>__custatevecCommunicatorCreate

    global __custatevecCommunicatorDestroy
    data["__custatevecCommunicatorDestroy"] = <intptr_t>__custatevecCommunicatorDestroy

    global __custatevecDistIndexBitSwapSchedulerCreate
    data["__custatevecDistIndexBitSwapSchedulerCreate"] = <intptr_t>__custatevecDistIndexBitSwapSchedulerCreate

    global __custatevecDistIndexBitSwapSchedulerDestroy
    data["__custatevecDistIndexBitSwapSchedulerDestroy"] = <intptr_t>__custatevecDistIndexBitSwapSchedulerDestroy

    global __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps
    data["__custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps"] = <intptr_t>__custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps

    global __custatevecDistIndexBitSwapSchedulerGetParameters
    data["__custatevecDistIndexBitSwapSchedulerGetParameters"] = <intptr_t>__custatevecDistIndexBitSwapSchedulerGetParameters

    global __custatevecSVSwapWorkerCreate
    data["__custatevecSVSwapWorkerCreate"] = <intptr_t>__custatevecSVSwapWorkerCreate

    global __custatevecSVSwapWorkerDestroy
    data["__custatevecSVSwapWorkerDestroy"] = <intptr_t>__custatevecSVSwapWorkerDestroy

    global __custatevecSVSwapWorkerSetExtraWorkspace
    data["__custatevecSVSwapWorkerSetExtraWorkspace"] = <intptr_t>__custatevecSVSwapWorkerSetExtraWorkspace

    global __custatevecSVSwapWorkerSetTransferWorkspace
    data["__custatevecSVSwapWorkerSetTransferWorkspace"] = <intptr_t>__custatevecSVSwapWorkerSetTransferWorkspace

    global __custatevecSVSwapWorkerSetSubSVsP2P
    data["__custatevecSVSwapWorkerSetSubSVsP2P"] = <intptr_t>__custatevecSVSwapWorkerSetSubSVsP2P

    global __custatevecSVSwapWorkerSetParameters
    data["__custatevecSVSwapWorkerSetParameters"] = <intptr_t>__custatevecSVSwapWorkerSetParameters

    global __custatevecSVSwapWorkerExecute
    data["__custatevecSVSwapWorkerExecute"] = <intptr_t>__custatevecSVSwapWorkerExecute

    global __custatevecInitializeStateVector
    data["__custatevecInitializeStateVector"] = <intptr_t>__custatevecInitializeStateVector

    global __custatevecApplyMatrixBatchedGetWorkspaceSize
    data["__custatevecApplyMatrixBatchedGetWorkspaceSize"] = <intptr_t>__custatevecApplyMatrixBatchedGetWorkspaceSize

    global __custatevecApplyMatrixBatched
    data["__custatevecApplyMatrixBatched"] = <intptr_t>__custatevecApplyMatrixBatched

    global __custatevecAbs2SumArrayBatched
    data["__custatevecAbs2SumArrayBatched"] = <intptr_t>__custatevecAbs2SumArrayBatched

    global __custatevecCollapseByBitStringBatchedGetWorkspaceSize
    data["__custatevecCollapseByBitStringBatchedGetWorkspaceSize"] = <intptr_t>__custatevecCollapseByBitStringBatchedGetWorkspaceSize

    global __custatevecCollapseByBitStringBatched
    data["__custatevecCollapseByBitStringBatched"] = <intptr_t>__custatevecCollapseByBitStringBatched

    global __custatevecMeasureBatched
    data["__custatevecMeasureBatched"] = <intptr_t>__custatevecMeasureBatched

    global __custatevecSubSVMigratorCreate
    data["__custatevecSubSVMigratorCreate"] = <intptr_t>__custatevecSubSVMigratorCreate

    global __custatevecSubSVMigratorDestroy
    data["__custatevecSubSVMigratorDestroy"] = <intptr_t>__custatevecSubSVMigratorDestroy

    global __custatevecSubSVMigratorMigrate
    data["__custatevecSubSVMigratorMigrate"] = <intptr_t>__custatevecSubSVMigratorMigrate

    global __custatevecComputeExpectationBatchedGetWorkspaceSize
    data["__custatevecComputeExpectationBatchedGetWorkspaceSize"] = <intptr_t>__custatevecComputeExpectationBatchedGetWorkspaceSize

    global __custatevecComputeExpectationBatched
    data["__custatevecComputeExpectationBatched"] = <intptr_t>__custatevecComputeExpectationBatched

    global __custatevecSetMathMode
    data["__custatevecSetMathMode"] = <intptr_t>__custatevecSetMathMode

    global __custatevecGetMathMode
    data["__custatevecGetMathMode"] = <intptr_t>__custatevecGetMathMode

    global __custatevecSVSwapWorkerCreateWithSemaphore
    data["__custatevecSVSwapWorkerCreateWithSemaphore"] = <intptr_t>__custatevecSVSwapWorkerCreateWithSemaphore

    global __custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores
    data["__custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores"] = <intptr_t>__custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores
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

cdef custatevecStatus_t _custatevecCreate(custatevecHandle_t* handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecCreate
    _check_or_init_custatevec()
    if __custatevecCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t*) noexcept nogil>__custatevecCreate)(
        handle)


cdef custatevecStatus_t _custatevecDestroy(custatevecHandle_t handle) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecDestroy
    _check_or_init_custatevec()
    if __custatevecDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t) noexcept nogil>__custatevecDestroy)(
        handle)


cdef custatevecStatus_t _custatevecGetDefaultWorkspaceSize(custatevecHandle_t handle, size_t* workspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecGetDefaultWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecGetDefaultWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetDefaultWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, size_t*) noexcept nogil>__custatevecGetDefaultWorkspaceSize)(
        handle, workspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSetWorkspace(custatevecHandle_t handle, void* workspace, size_t workspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSetWorkspace
    _check_or_init_custatevec()
    if __custatevecSetWorkspace == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSetWorkspace is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, size_t) noexcept nogil>__custatevecSetWorkspace)(
        handle, workspace, workspaceSizeInBytes)


cdef const char* _custatevecGetErrorName(custatevecStatus_t status) except?NULL nogil:
    global __custatevecGetErrorName
    _check_or_init_custatevec()
    if __custatevecGetErrorName == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetErrorName is not found")
    return (<const char* (*)(custatevecStatus_t) noexcept nogil>__custatevecGetErrorName)(
        status)


cdef const char* _custatevecGetErrorString(custatevecStatus_t status) except?NULL nogil:
    global __custatevecGetErrorString
    _check_or_init_custatevec()
    if __custatevecGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetErrorString is not found")
    return (<const char* (*)(custatevecStatus_t) noexcept nogil>__custatevecGetErrorString)(
        status)


cdef custatevecStatus_t _custatevecGetProperty(libraryPropertyType type, int32_t* value) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecGetProperty
    _check_or_init_custatevec()
    if __custatevecGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetProperty is not found")
    return (<custatevecStatus_t (*)(libraryPropertyType, int32_t*) noexcept nogil>__custatevecGetProperty)(
        type, value)


cdef size_t _custatevecGetVersion() except?0 nogil:
    global __custatevecGetVersion
    _check_or_init_custatevec()
    if __custatevecGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetVersion is not found")
    return (<size_t (*)() noexcept nogil>__custatevecGetVersion)(
        )


cdef custatevecStatus_t _custatevecSetStream(custatevecHandle_t handle, cudaStream_t streamId) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSetStream
    _check_or_init_custatevec()
    if __custatevecSetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSetStream is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaStream_t) noexcept nogil>__custatevecSetStream)(
        handle, streamId)


cdef custatevecStatus_t _custatevecGetStream(custatevecHandle_t handle, cudaStream_t* streamId) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecGetStream
    _check_or_init_custatevec()
    if __custatevecGetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetStream is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaStream_t*) noexcept nogil>__custatevecGetStream)(
        handle, streamId)


cdef custatevecStatus_t _custatevecLoggerSetCallback(custatevecLoggerCallback_t callback) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecLoggerSetCallback
    _check_or_init_custatevec()
    if __custatevecLoggerSetCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerSetCallback is not found")
    return (<custatevecStatus_t (*)(custatevecLoggerCallback_t) noexcept nogil>__custatevecLoggerSetCallback)(
        callback)


cdef custatevecStatus_t _custatevecLoggerSetCallbackData(custatevecLoggerCallbackData_t callback, void* userData) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecLoggerSetCallbackData
    _check_or_init_custatevec()
    if __custatevecLoggerSetCallbackData == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerSetCallbackData is not found")
    return (<custatevecStatus_t (*)(custatevecLoggerCallbackData_t, void*) noexcept nogil>__custatevecLoggerSetCallbackData)(
        callback, userData)


cdef custatevecStatus_t _custatevecLoggerSetFile(FILE* file) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecLoggerSetFile
    _check_or_init_custatevec()
    if __custatevecLoggerSetFile == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerSetFile is not found")
    return (<custatevecStatus_t (*)(FILE*) noexcept nogil>__custatevecLoggerSetFile)(
        file)


cdef custatevecStatus_t _custatevecLoggerOpenFile(const char* logFile) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecLoggerOpenFile
    _check_or_init_custatevec()
    if __custatevecLoggerOpenFile == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerOpenFile is not found")
    return (<custatevecStatus_t (*)(const char*) noexcept nogil>__custatevecLoggerOpenFile)(
        logFile)


cdef custatevecStatus_t _custatevecLoggerSetLevel(int32_t level) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecLoggerSetLevel
    _check_or_init_custatevec()
    if __custatevecLoggerSetLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerSetLevel is not found")
    return (<custatevecStatus_t (*)(int32_t) noexcept nogil>__custatevecLoggerSetLevel)(
        level)


cdef custatevecStatus_t _custatevecLoggerSetMask(int32_t mask) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecLoggerSetMask
    _check_or_init_custatevec()
    if __custatevecLoggerSetMask == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerSetMask is not found")
    return (<custatevecStatus_t (*)(int32_t) noexcept nogil>__custatevecLoggerSetMask)(
        mask)


cdef custatevecStatus_t _custatevecLoggerForceDisable() except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecLoggerForceDisable
    _check_or_init_custatevec()
    if __custatevecLoggerForceDisable == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerForceDisable is not found")
    return (<custatevecStatus_t (*)() noexcept nogil>__custatevecLoggerForceDisable)(
        )


cdef custatevecStatus_t _custatevecGetDeviceMemHandler(custatevecHandle_t handle, custatevecDeviceMemHandler_t* handler) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecGetDeviceMemHandler
    _check_or_init_custatevec()
    if __custatevecGetDeviceMemHandler == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetDeviceMemHandler is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecDeviceMemHandler_t*) noexcept nogil>__custatevecGetDeviceMemHandler)(
        handle, handler)


cdef custatevecStatus_t _custatevecSetDeviceMemHandler(custatevecHandle_t handle, const custatevecDeviceMemHandler_t* handler) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSetDeviceMemHandler
    _check_or_init_custatevec()
    if __custatevecSetDeviceMemHandler == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSetDeviceMemHandler is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const custatevecDeviceMemHandler_t*) noexcept nogil>__custatevecSetDeviceMemHandler)(
        handle, handler)


cdef custatevecStatus_t _custatevecAbs2SumOnZBasis(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* abs2sum0, double* abs2sum1, const int32_t* basisBits, const uint32_t nBasisBits) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecAbs2SumOnZBasis
    _check_or_init_custatevec()
    if __custatevecAbs2SumOnZBasis == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAbs2SumOnZBasis is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, double*, double*, const int32_t*, const uint32_t) noexcept nogil>__custatevecAbs2SumOnZBasis)(
        handle, sv, svDataType, nIndexBits, abs2sum0, abs2sum1, basisBits, nBasisBits)


cdef custatevecStatus_t _custatevecAbs2SumArray(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* abs2sum, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecAbs2SumArray
    _check_or_init_custatevec()
    if __custatevecAbs2SumArray == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAbs2SumArray is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, double*, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t) noexcept nogil>__custatevecAbs2SumArray)(
        handle, sv, svDataType, nIndexBits, abs2sum, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen)


cdef custatevecStatus_t _custatevecCollapseOnZBasis(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int32_t parity, const int32_t* basisBits, const uint32_t nBasisBits, double norm) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecCollapseOnZBasis
    _check_or_init_custatevec()
    if __custatevecCollapseOnZBasis == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCollapseOnZBasis is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const int32_t, const int32_t*, const uint32_t, double) noexcept nogil>__custatevecCollapseOnZBasis)(
        handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, norm)


cdef custatevecStatus_t _custatevecCollapseByBitString(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, double norm) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecCollapseByBitString
    _check_or_init_custatevec()
    if __custatevecCollapseByBitString == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCollapseByBitString is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const int32_t*, const int32_t*, const uint32_t, double) noexcept nogil>__custatevecCollapseByBitString)(
        handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, norm)


cdef custatevecStatus_t _custatevecMeasureOnZBasis(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* parity, const int32_t* basisBits, const uint32_t nBasisBits, const double randnum, custatevecCollapseOp_t collapse) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecMeasureOnZBasis
    _check_or_init_custatevec()
    if __custatevecMeasureOnZBasis == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecMeasureOnZBasis is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, int32_t*, const int32_t*, const uint32_t, const double, custatevecCollapseOp_t) noexcept nogil>__custatevecMeasureOnZBasis)(
        handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, randnum, collapse)


cdef custatevecStatus_t _custatevecBatchMeasure(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, const double randnum, custatevecCollapseOp_t collapse) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecBatchMeasure
    _check_or_init_custatevec()
    if __custatevecBatchMeasure == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecBatchMeasure is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, int32_t*, const int32_t*, const uint32_t, const double, custatevecCollapseOp_t) noexcept nogil>__custatevecBatchMeasure)(
        handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse)


cdef custatevecStatus_t _custatevecBatchMeasureWithOffset(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, const double randnum, custatevecCollapseOp_t collapse, const double offset, const double abs2sum) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecBatchMeasureWithOffset
    _check_or_init_custatevec()
    if __custatevecBatchMeasureWithOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecBatchMeasureWithOffset is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, int32_t*, const int32_t*, const uint32_t, const double, custatevecCollapseOp_t, const double, const double) noexcept nogil>__custatevecBatchMeasureWithOffset)(
        handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse, offset, abs2sum)


cdef custatevecStatus_t _custatevecApplyPauliRotation(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double theta, const custatevecPauli_t* paulis, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecApplyPauliRotation
    _check_or_init_custatevec()
    if __custatevecApplyPauliRotation == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyPauliRotation is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, double, const custatevecPauli_t*, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t) noexcept nogil>__custatevecApplyPauliRotation)(
        handle, sv, svDataType, nIndexBits, theta, paulis, targets, nTargets, controls, controlBitValues, nControls)


cdef custatevecStatus_t _custatevecApplyMatrixGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nTargets, const uint32_t nControls, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecApplyMatrixGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecApplyMatrixGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyMatrixGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaDataType_t, const uint32_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const int32_t, const uint32_t, const uint32_t, custatevecComputeType_t, size_t*) noexcept nogil>__custatevecApplyMatrixGetWorkspaceSize)(
        handle, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecApplyMatrix(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecApplyMatrix
    _check_or_init_custatevec()
    if __custatevecApplyMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyMatrix is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const int32_t, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, custatevecComputeType_t, void*, size_t) noexcept nogil>__custatevecApplyMatrix)(
        handle, sv, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, targets, nTargets, controls, controlBitValues, nControls, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecComputeExpectationGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nBasisBits, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecComputeExpectationGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecComputeExpectationGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecComputeExpectationGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaDataType_t, const uint32_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const uint32_t, custatevecComputeType_t, size_t*) noexcept nogil>__custatevecComputeExpectationGetWorkspaceSize)(
        handle, svDataType, nIndexBits, matrix, matrixDataType, layout, nBasisBits, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecComputeExpectation(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, void* expectationValue, cudaDataType_t expectationDataType, double* residualNorm, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t* basisBits, const uint32_t nBasisBits, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecComputeExpectation
    _check_or_init_custatevec()
    if __custatevecComputeExpectation == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecComputeExpectation is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, void*, cudaDataType_t, double*, const void*, cudaDataType_t, custatevecMatrixLayout_t, const int32_t*, const uint32_t, custatevecComputeType_t, void*, size_t) noexcept nogil>__custatevecComputeExpectation)(
        handle, sv, svDataType, nIndexBits, expectationValue, expectationDataType, residualNorm, matrix, matrixDataType, layout, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSamplerCreate(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecSamplerDescriptor_t* sampler, uint32_t nMaxShots, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSamplerCreate
    _check_or_init_custatevec()
    if __custatevecSamplerCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, custatevecSamplerDescriptor_t*, uint32_t, size_t*) noexcept nogil>__custatevecSamplerCreate)(
        handle, sv, svDataType, nIndexBits, sampler, nMaxShots, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSamplerDestroy(custatevecSamplerDescriptor_t sampler) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSamplerDestroy
    _check_or_init_custatevec()
    if __custatevecSamplerDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecSamplerDescriptor_t) noexcept nogil>__custatevecSamplerDestroy)(
        sampler)


cdef custatevecStatus_t _custatevecSamplerPreprocess(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, void* extraWorkspace, const size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSamplerPreprocess
    _check_or_init_custatevec()
    if __custatevecSamplerPreprocess == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerPreprocess is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSamplerDescriptor_t, void*, const size_t) noexcept nogil>__custatevecSamplerPreprocess)(
        handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSamplerGetSquaredNorm(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, double* norm) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSamplerGetSquaredNorm
    _check_or_init_custatevec()
    if __custatevecSamplerGetSquaredNorm == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerGetSquaredNorm is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSamplerDescriptor_t, double*) noexcept nogil>__custatevecSamplerGetSquaredNorm)(
        handle, sampler, norm)


cdef custatevecStatus_t _custatevecSamplerApplySubSVOffset(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, int32_t subSVOrd, uint32_t nSubSVs, double offset, double norm) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSamplerApplySubSVOffset
    _check_or_init_custatevec()
    if __custatevecSamplerApplySubSVOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerApplySubSVOffset is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSamplerDescriptor_t, int32_t, uint32_t, double, double) noexcept nogil>__custatevecSamplerApplySubSVOffset)(
        handle, sampler, subSVOrd, nSubSVs, offset, norm)


cdef custatevecStatus_t _custatevecSamplerSample(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* randnums, const uint32_t nShots, custatevecSamplerOutput_t output) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSamplerSample
    _check_or_init_custatevec()
    if __custatevecSamplerSample == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerSample is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSamplerDescriptor_t, custatevecIndex_t*, const int32_t*, const uint32_t, const double*, const uint32_t, custatevecSamplerOutput_t) noexcept nogil>__custatevecSamplerSample)(
        handle, sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots, output)


cdef custatevecStatus_t _custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType, const int32_t* targets, const uint32_t nTargets, const uint32_t nControls, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaDataType_t, const uint32_t, const custatevecIndex_t*, const void*, cudaDataType_t, const int32_t*, const uint32_t, const uint32_t, size_t*) noexcept nogil>__custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize)(
        handle, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, targets, nTargets, nControls, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecApplyGeneralizedPermutationMatrix(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType, const int32_t adjoint, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecApplyGeneralizedPermutationMatrix
    _check_or_init_custatevec()
    if __custatevecApplyGeneralizedPermutationMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyGeneralizedPermutationMatrix is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, custatevecIndex_t*, const void*, cudaDataType_t, const int32_t, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, void*, size_t) noexcept nogil>__custatevecApplyGeneralizedPermutationMatrix)(
        handle, sv, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, adjoint, targets, nTargets, controls, controlBitValues, nControls, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecComputeExpectationsOnPauliBasis(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* expectationValues, const custatevecPauli_t** pauliOperatorsArray, const uint32_t nPauliOperatorArrays, const int32_t** basisBitsArray, const uint32_t* nBasisBitsArray) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecComputeExpectationsOnPauliBasis
    _check_or_init_custatevec()
    if __custatevecComputeExpectationsOnPauliBasis == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecComputeExpectationsOnPauliBasis is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, double*, const custatevecPauli_t**, const uint32_t, const int32_t**, const uint32_t*) noexcept nogil>__custatevecComputeExpectationsOnPauliBasis)(
        handle, sv, svDataType, nIndexBits, expectationValues, pauliOperatorsArray, nPauliOperatorArrays, basisBitsArray, nBasisBitsArray)


cdef custatevecStatus_t _custatevecAccessorCreate(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecAccessorDescriptor_t* accessor, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecAccessorCreate
    _check_or_init_custatevec()
    if __custatevecAccessorCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, custatevecAccessorDescriptor_t*, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, size_t*) noexcept nogil>__custatevecAccessorCreate)(
        handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecAccessorCreateView(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecAccessorDescriptor_t* accessor, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecAccessorCreateView
    _check_or_init_custatevec()
    if __custatevecAccessorCreateView == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorCreateView is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, custatevecAccessorDescriptor_t*, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, size_t*) noexcept nogil>__custatevecAccessorCreateView)(
        handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecAccessorDestroy(custatevecAccessorDescriptor_t accessor) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecAccessorDestroy
    _check_or_init_custatevec()
    if __custatevecAccessorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecAccessorDescriptor_t) noexcept nogil>__custatevecAccessorDestroy)(
        accessor)


cdef custatevecStatus_t _custatevecAccessorSetExtraWorkspace(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecAccessorSetExtraWorkspace
    _check_or_init_custatevec()
    if __custatevecAccessorSetExtraWorkspace == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorSetExtraWorkspace is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecAccessorDescriptor_t, void*, size_t) noexcept nogil>__custatevecAccessorSetExtraWorkspace)(
        handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecAccessorGet(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, void* externalBuffer, const custatevecIndex_t begin, const custatevecIndex_t end) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecAccessorGet
    _check_or_init_custatevec()
    if __custatevecAccessorGet == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorGet is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecAccessorDescriptor_t, void*, const custatevecIndex_t, const custatevecIndex_t) noexcept nogil>__custatevecAccessorGet)(
        handle, accessor, externalBuffer, begin, end)


cdef custatevecStatus_t _custatevecAccessorSet(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, const void* externalBuffer, const custatevecIndex_t begin, const custatevecIndex_t end) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecAccessorSet
    _check_or_init_custatevec()
    if __custatevecAccessorSet == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorSet is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecAccessorDescriptor_t, const void*, const custatevecIndex_t, const custatevecIndex_t) noexcept nogil>__custatevecAccessorSet)(
        handle, accessor, externalBuffer, begin, end)


cdef custatevecStatus_t _custatevecSwapIndexBits(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int2* bitSwaps, const uint32_t nBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSwapIndexBits
    _check_or_init_custatevec()
    if __custatevecSwapIndexBits == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSwapIndexBits is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const int2*, const uint32_t, const int32_t*, const int32_t*, const uint32_t) noexcept nogil>__custatevecSwapIndexBits)(
        handle, sv, svDataType, nIndexBits, bitSwaps, nBitSwaps, maskBitString, maskOrdering, maskLen)


cdef custatevecStatus_t _custatevecTestMatrixTypeGetWorkspaceSize(custatevecHandle_t handle, custatevecMatrixType_t matrixType, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nTargets, const int32_t adjoint, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecTestMatrixTypeGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecTestMatrixTypeGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecTestMatrixTypeGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecMatrixType_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const uint32_t, const int32_t, custatevecComputeType_t, size_t*) noexcept nogil>__custatevecTestMatrixTypeGetWorkspaceSize)(
        handle, matrixType, matrix, matrixDataType, layout, nTargets, adjoint, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecTestMatrixType(custatevecHandle_t handle, double* residualNorm, custatevecMatrixType_t matrixType, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nTargets, const int32_t adjoint, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecTestMatrixType
    _check_or_init_custatevec()
    if __custatevecTestMatrixType == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecTestMatrixType is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, double*, custatevecMatrixType_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const uint32_t, const int32_t, custatevecComputeType_t, void*, size_t) noexcept nogil>__custatevecTestMatrixType)(
        handle, residualNorm, matrixType, matrix, matrixDataType, layout, nTargets, adjoint, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecMultiDeviceSwapIndexBits(custatevecHandle_t* handles, const uint32_t nHandles, void** subSVs, const cudaDataType_t svDataType, const uint32_t nGlobalIndexBits, const uint32_t nLocalIndexBits, const int2* indexBitSwaps, const uint32_t nIndexBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, const custatevecDeviceNetworkType_t deviceNetworkType) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecMultiDeviceSwapIndexBits
    _check_or_init_custatevec()
    if __custatevecMultiDeviceSwapIndexBits == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecMultiDeviceSwapIndexBits is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t*, const uint32_t, void**, const cudaDataType_t, const uint32_t, const uint32_t, const int2*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, const custatevecDeviceNetworkType_t) noexcept nogil>__custatevecMultiDeviceSwapIndexBits)(
        handles, nHandles, subSVs, svDataType, nGlobalIndexBits, nLocalIndexBits, indexBitSwaps, nIndexBitSwaps, maskBitString, maskOrdering, maskLen, deviceNetworkType)


cdef custatevecStatus_t _custatevecCommunicatorCreate(custatevecHandle_t handle, custatevecCommunicatorDescriptor_t* communicator, custatevecCommunicatorType_t communicatorType, const char* soname) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecCommunicatorCreate
    _check_or_init_custatevec()
    if __custatevecCommunicatorCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCommunicatorCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecCommunicatorDescriptor_t*, custatevecCommunicatorType_t, const char*) noexcept nogil>__custatevecCommunicatorCreate)(
        handle, communicator, communicatorType, soname)


cdef custatevecStatus_t _custatevecCommunicatorDestroy(custatevecHandle_t handle, custatevecCommunicatorDescriptor_t communicator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecCommunicatorDestroy
    _check_or_init_custatevec()
    if __custatevecCommunicatorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCommunicatorDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecCommunicatorDescriptor_t) noexcept nogil>__custatevecCommunicatorDestroy)(
        handle, communicator)


cdef custatevecStatus_t _custatevecDistIndexBitSwapSchedulerCreate(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t* scheduler, const uint32_t nGlobalIndexBits, const uint32_t nLocalIndexBits) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecDistIndexBitSwapSchedulerCreate
    _check_or_init_custatevec()
    if __custatevecDistIndexBitSwapSchedulerCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecDistIndexBitSwapSchedulerCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecDistIndexBitSwapSchedulerDescriptor_t*, const uint32_t, const uint32_t) noexcept nogil>__custatevecDistIndexBitSwapSchedulerCreate)(
        handle, scheduler, nGlobalIndexBits, nLocalIndexBits)


cdef custatevecStatus_t _custatevecDistIndexBitSwapSchedulerDestroy(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecDistIndexBitSwapSchedulerDestroy
    _check_or_init_custatevec()
    if __custatevecDistIndexBitSwapSchedulerDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecDistIndexBitSwapSchedulerDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecDistIndexBitSwapSchedulerDescriptor_t) noexcept nogil>__custatevecDistIndexBitSwapSchedulerDestroy)(
        handle, scheduler)


cdef custatevecStatus_t _custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler, const int2* indexBitSwaps, const uint32_t nIndexBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, uint32_t* nSwapBatches) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps
    _check_or_init_custatevec()
    if __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecDistIndexBitSwapSchedulerDescriptor_t, const int2*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, uint32_t*) noexcept nogil>__custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps)(
        handle, scheduler, indexBitSwaps, nIndexBitSwaps, maskBitString, maskOrdering, maskLen, nSwapBatches)


cdef custatevecStatus_t _custatevecDistIndexBitSwapSchedulerGetParameters(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler, const int32_t swapBatchIndex, const int32_t orgSubSVIndex, custatevecSVSwapParameters_t* parameters) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecDistIndexBitSwapSchedulerGetParameters
    _check_or_init_custatevec()
    if __custatevecDistIndexBitSwapSchedulerGetParameters == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecDistIndexBitSwapSchedulerGetParameters is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecDistIndexBitSwapSchedulerDescriptor_t, const int32_t, const int32_t, custatevecSVSwapParameters_t*) noexcept nogil>__custatevecDistIndexBitSwapSchedulerGetParameters)(
        handle, scheduler, swapBatchIndex, orgSubSVIndex, parameters)


cdef custatevecStatus_t _custatevecSVSwapWorkerCreate(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t* svSwapWorker, custatevecCommunicatorDescriptor_t communicator, void* orgSubSV, int32_t orgSubSVIndex, cudaEvent_t orgEvent, cudaDataType_t svDataType, cudaStream_t stream, size_t* extraWorkspaceSizeInBytes, size_t* minTransferWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSVSwapWorkerCreate
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t*, custatevecCommunicatorDescriptor_t, void*, int32_t, cudaEvent_t, cudaDataType_t, cudaStream_t, size_t*, size_t*) noexcept nogil>__custatevecSVSwapWorkerCreate)(
        handle, svSwapWorker, communicator, orgSubSV, orgSubSVIndex, orgEvent, svDataType, stream, extraWorkspaceSizeInBytes, minTransferWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSVSwapWorkerDestroy(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSVSwapWorkerDestroy
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t) noexcept nogil>__custatevecSVSwapWorkerDestroy)(
        handle, svSwapWorker)


cdef custatevecStatus_t _custatevecSVSwapWorkerSetExtraWorkspace(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSVSwapWorkerSetExtraWorkspace
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerSetExtraWorkspace == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerSetExtraWorkspace is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t, void*, size_t) noexcept nogil>__custatevecSVSwapWorkerSetExtraWorkspace)(
        handle, svSwapWorker, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSVSwapWorkerSetTransferWorkspace(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void* transferWorkspace, size_t transferWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSVSwapWorkerSetTransferWorkspace
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerSetTransferWorkspace == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerSetTransferWorkspace is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t, void*, size_t) noexcept nogil>__custatevecSVSwapWorkerSetTransferWorkspace)(
        handle, svSwapWorker, transferWorkspace, transferWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSVSwapWorkerSetSubSVsP2P(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void** dstSubSVsP2P, const int32_t* dstSubSVIndicesP2P, cudaEvent_t* dstEvents, const uint32_t nDstSubSVsP2P) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSVSwapWorkerSetSubSVsP2P
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerSetSubSVsP2P == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerSetSubSVsP2P is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t, void**, const int32_t*, cudaEvent_t*, const uint32_t) noexcept nogil>__custatevecSVSwapWorkerSetSubSVsP2P)(
        handle, svSwapWorker, dstSubSVsP2P, dstSubSVIndicesP2P, dstEvents, nDstSubSVsP2P)


cdef custatevecStatus_t _custatevecSVSwapWorkerSetParameters(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, const custatevecSVSwapParameters_t* parameters, int peer) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSVSwapWorkerSetParameters
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerSetParameters == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerSetParameters is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t, const custatevecSVSwapParameters_t*, int) noexcept nogil>__custatevecSVSwapWorkerSetParameters)(
        handle, svSwapWorker, parameters, peer)


cdef custatevecStatus_t _custatevecSVSwapWorkerExecute(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, custatevecIndex_t begin, custatevecIndex_t end) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSVSwapWorkerExecute
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerExecute == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerExecute is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t, custatevecIndex_t, custatevecIndex_t) noexcept nogil>__custatevecSVSwapWorkerExecute)(
        handle, svSwapWorker, begin, end)


cdef custatevecStatus_t _custatevecInitializeStateVector(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecStateVectorType_t svType) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecInitializeStateVector
    _check_or_init_custatevec()
    if __custatevecInitializeStateVector == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecInitializeStateVector is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, custatevecStateVectorType_t) noexcept nogil>__custatevecInitializeStateVector)(
        handle, sv, svDataType, nIndexBits, svType)


cdef custatevecStatus_t _custatevecApplyMatrixBatchedGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, custatevecMatrixMapType_t mapType, const int32_t* matrixIndices, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nMatrices, const uint32_t nTargets, const uint32_t nControls, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecApplyMatrixBatchedGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecApplyMatrixBatchedGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyMatrixBatchedGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaDataType_t, const uint32_t, const uint32_t, const custatevecIndex_t, custatevecMatrixMapType_t, const int32_t*, const void*, cudaDataType_t, custatevecMatrixLayout_t, const int32_t, const uint32_t, const uint32_t, const uint32_t, custatevecComputeType_t, size_t*) noexcept nogil>__custatevecApplyMatrixBatchedGetWorkspaceSize)(
        handle, svDataType, nIndexBits, nSVs, svStride, mapType, matrixIndices, matrices, matrixDataType, layout, adjoint, nMatrices, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecApplyMatrixBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, custatevecIndex_t svStride, custatevecMatrixMapType_t mapType, const int32_t* matrixIndices, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nMatrices, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecApplyMatrixBatched
    _check_or_init_custatevec()
    if __custatevecApplyMatrixBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyMatrixBatched is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const uint32_t, custatevecIndex_t, custatevecMatrixMapType_t, const int32_t*, const void*, cudaDataType_t, custatevecMatrixLayout_t, const int32_t, const uint32_t, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, custatevecComputeType_t, void*, size_t) noexcept nogil>__custatevecApplyMatrixBatched)(
        handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, mapType, matrixIndices, matrices, matrixDataType, layout, adjoint, nMatrices, targets, nTargets, controls, controlBitValues, nControls, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecAbs2SumArrayBatched(custatevecHandle_t handle, const void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, double* abs2sumArrays, const custatevecIndex_t abs2sumArrayStride, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const custatevecIndex_t* maskBitStrings, const int32_t* maskOrdering, const uint32_t maskLen) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecAbs2SumArrayBatched
    _check_or_init_custatevec()
    if __custatevecAbs2SumArrayBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAbs2SumArrayBatched is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, const uint32_t, const custatevecIndex_t, double*, const custatevecIndex_t, const int32_t*, const uint32_t, const custatevecIndex_t*, const int32_t*, const uint32_t) noexcept nogil>__custatevecAbs2SumArrayBatched)(
        handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, abs2sumArrays, abs2sumArrayStride, bitOrdering, bitOrderingLen, maskBitStrings, maskOrdering, maskLen)


cdef custatevecStatus_t _custatevecCollapseByBitStringBatchedGetWorkspaceSize(custatevecHandle_t handle, const uint32_t nSVs, const custatevecIndex_t* bitStrings, const double* norms, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecCollapseByBitStringBatchedGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecCollapseByBitStringBatchedGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCollapseByBitStringBatchedGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const uint32_t, const custatevecIndex_t*, const double*, size_t*) noexcept nogil>__custatevecCollapseByBitStringBatchedGetWorkspaceSize)(
        handle, nSVs, bitStrings, norms, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecCollapseByBitStringBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, const custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* norms, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecCollapseByBitStringBatched
    _check_or_init_custatevec()
    if __custatevecCollapseByBitStringBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCollapseByBitStringBatched is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const uint32_t, const custatevecIndex_t, const custatevecIndex_t*, const int32_t*, const uint32_t, const double*, void*, size_t) noexcept nogil>__custatevecCollapseByBitStringBatched)(
        handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, bitStrings, bitOrdering, bitStringLen, norms, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecMeasureBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* randnums, custatevecCollapseOp_t collapse) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecMeasureBatched
    _check_or_init_custatevec()
    if __custatevecMeasureBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecMeasureBatched is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const uint32_t, const custatevecIndex_t, custatevecIndex_t*, const int32_t*, const uint32_t, const double*, custatevecCollapseOp_t) noexcept nogil>__custatevecMeasureBatched)(
        handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, bitStrings, bitOrdering, bitStringLen, randnums, collapse)


cdef custatevecStatus_t _custatevecSubSVMigratorCreate(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t* migrator, void* deviceSlices, cudaDataType_t svDataType, int nDeviceSlices, int nSliceLocalIndexBits) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSubSVMigratorCreate
    _check_or_init_custatevec()
    if __custatevecSubSVMigratorCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSubSVMigratorCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSubSVMigratorDescriptor_t*, void*, cudaDataType_t, int, int) noexcept nogil>__custatevecSubSVMigratorCreate)(
        handle, migrator, deviceSlices, svDataType, nDeviceSlices, nSliceLocalIndexBits)


cdef custatevecStatus_t _custatevecSubSVMigratorDestroy(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t migrator) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSubSVMigratorDestroy
    _check_or_init_custatevec()
    if __custatevecSubSVMigratorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSubSVMigratorDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSubSVMigratorDescriptor_t) noexcept nogil>__custatevecSubSVMigratorDestroy)(
        handle, migrator)


cdef custatevecStatus_t _custatevecSubSVMigratorMigrate(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t migrator, int deviceSliceIndex, const void* srcSubSVSlice, void* dstSubSVSlice, custatevecIndex_t begin, custatevecIndex_t end) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSubSVMigratorMigrate
    _check_or_init_custatevec()
    if __custatevecSubSVMigratorMigrate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSubSVMigratorMigrate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSubSVMigratorDescriptor_t, int, const void*, void*, custatevecIndex_t, custatevecIndex_t) noexcept nogil>__custatevecSubSVMigratorMigrate)(
        handle, migrator, deviceSliceIndex, srcSubSVSlice, dstSubSVSlice, begin, end)


cdef custatevecStatus_t _custatevecComputeExpectationBatchedGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nMatrices, const uint32_t nBasisBits, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecComputeExpectationBatchedGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecComputeExpectationBatchedGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecComputeExpectationBatchedGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaDataType_t, const uint32_t, const uint32_t, const custatevecIndex_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const uint32_t, const uint32_t, custatevecComputeType_t, size_t*) noexcept nogil>__custatevecComputeExpectationBatchedGetWorkspaceSize)(
        handle, svDataType, nIndexBits, nSVs, svStride, matrices, matrixDataType, layout, nMatrices, nBasisBits, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecComputeExpectationBatched(custatevecHandle_t handle, const void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, custatevecIndex_t svStride, double2* expectationValues, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nMatrices, const int32_t* basisBits, const uint32_t nBasisBits, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecComputeExpectationBatched
    _check_or_init_custatevec()
    if __custatevecComputeExpectationBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecComputeExpectationBatched is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, const uint32_t, custatevecIndex_t, double2*, const void*, cudaDataType_t, custatevecMatrixLayout_t, const uint32_t, const int32_t*, const uint32_t, custatevecComputeType_t, void*, size_t) noexcept nogil>__custatevecComputeExpectationBatched)(
        handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, expectationValues, matrices, matrixDataType, layout, nMatrices, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSetMathMode(custatevecHandle_t handle, custatevecMathMode_t mode) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSetMathMode
    _check_or_init_custatevec()
    if __custatevecSetMathMode == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSetMathMode is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecMathMode_t) noexcept nogil>__custatevecSetMathMode)(
        handle, mode)


cdef custatevecStatus_t _custatevecGetMathMode(custatevecHandle_t handle, custatevecMathMode_t* mode) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecGetMathMode
    _check_or_init_custatevec()
    if __custatevecGetMathMode == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetMathMode is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecMathMode_t*) noexcept nogil>__custatevecGetMathMode)(
        handle, mode)


cdef custatevecStatus_t _custatevecSVSwapWorkerCreateWithSemaphore(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t* svSwapWorker, custatevecCommunicatorDescriptor_t communicator, void* orgSubSV, int32_t orgSubSVIndex, void* orgSemaphore, cudaDataType_t svDataType, cudaStream_t stream, size_t* extraWorkspaceSizeInBytes, size_t* minTransferWorkspaceSizeInBytes) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSVSwapWorkerCreateWithSemaphore
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerCreateWithSemaphore == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerCreateWithSemaphore is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t*, custatevecCommunicatorDescriptor_t, void*, int32_t, void*, cudaDataType_t, cudaStream_t, size_t*, size_t*) noexcept nogil>__custatevecSVSwapWorkerCreateWithSemaphore)(
        handle, svSwapWorker, communicator, orgSubSV, orgSubSVIndex, orgSemaphore, svDataType, stream, extraWorkspaceSizeInBytes, minTransferWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void** dstSubSVsP2P, const int32_t* dstSubSVIndicesP2P, void** dstSemaphores, const uint32_t nDstSubSVsP2P) except?_CUSTATEVECSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t, void**, const int32_t*, void**, const uint32_t) noexcept nogil>__custatevecSVSwapWorkerSetSubSVsP2PWithSemaphores)(
        handle, svSwapWorker, dstSubSVsP2P, dstSubSVIndicesP2P, dstSemaphores, nDstSubSVsP2P)
