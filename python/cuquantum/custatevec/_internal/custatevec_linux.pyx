# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# This code was automatically generated. Do not modify it directly.

from libc.stdint cimport intptr_t

from ..._utils import FunctionNotFoundError, NotSupportedError


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

cdef bint __py_custatevec_init = False

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
cdef void* __custatevecLoggerSetCallbackData = NULL
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


cdef void* load_library() except* nogil:
    cdef void* handle
    handle = dlopen("libcustatevec.so.1", RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        with gil:
            err_msg = dlerror()
            raise RuntimeError(f'Failed to dlopen libcustatevec ({err_msg.decode()})')
    return handle


cdef int _check_or_init_custatevec() except -1 nogil:
    global __py_custatevec_init
    if __py_custatevec_init:
        return 0

    # Load function
    cdef void* handle = NULL
    global __custatevecCreate
    __custatevecCreate = dlsym(RTLD_DEFAULT, 'custatevecCreate')
    if __custatevecCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecCreate = dlsym(handle, 'custatevecCreate')
    
    global __custatevecDestroy
    __custatevecDestroy = dlsym(RTLD_DEFAULT, 'custatevecDestroy')
    if __custatevecDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecDestroy = dlsym(handle, 'custatevecDestroy')
    
    global __custatevecGetDefaultWorkspaceSize
    __custatevecGetDefaultWorkspaceSize = dlsym(RTLD_DEFAULT, 'custatevecGetDefaultWorkspaceSize')
    if __custatevecGetDefaultWorkspaceSize == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecGetDefaultWorkspaceSize = dlsym(handle, 'custatevecGetDefaultWorkspaceSize')
    
    global __custatevecSetWorkspace
    __custatevecSetWorkspace = dlsym(RTLD_DEFAULT, 'custatevecSetWorkspace')
    if __custatevecSetWorkspace == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSetWorkspace = dlsym(handle, 'custatevecSetWorkspace')
    
    global __custatevecGetErrorName
    __custatevecGetErrorName = dlsym(RTLD_DEFAULT, 'custatevecGetErrorName')
    if __custatevecGetErrorName == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecGetErrorName = dlsym(handle, 'custatevecGetErrorName')
    
    global __custatevecGetErrorString
    __custatevecGetErrorString = dlsym(RTLD_DEFAULT, 'custatevecGetErrorString')
    if __custatevecGetErrorString == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecGetErrorString = dlsym(handle, 'custatevecGetErrorString')
    
    global __custatevecGetProperty
    __custatevecGetProperty = dlsym(RTLD_DEFAULT, 'custatevecGetProperty')
    if __custatevecGetProperty == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecGetProperty = dlsym(handle, 'custatevecGetProperty')
    
    global __custatevecGetVersion
    __custatevecGetVersion = dlsym(RTLD_DEFAULT, 'custatevecGetVersion')
    if __custatevecGetVersion == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecGetVersion = dlsym(handle, 'custatevecGetVersion')
    
    global __custatevecSetStream
    __custatevecSetStream = dlsym(RTLD_DEFAULT, 'custatevecSetStream')
    if __custatevecSetStream == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSetStream = dlsym(handle, 'custatevecSetStream')
    
    global __custatevecGetStream
    __custatevecGetStream = dlsym(RTLD_DEFAULT, 'custatevecGetStream')
    if __custatevecGetStream == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecGetStream = dlsym(handle, 'custatevecGetStream')
    
    global __custatevecLoggerSetCallbackData
    __custatevecLoggerSetCallbackData = dlsym(RTLD_DEFAULT, 'custatevecLoggerSetCallbackData')
    if __custatevecLoggerSetCallbackData == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecLoggerSetCallbackData = dlsym(handle, 'custatevecLoggerSetCallbackData')
    
    global __custatevecLoggerOpenFile
    __custatevecLoggerOpenFile = dlsym(RTLD_DEFAULT, 'custatevecLoggerOpenFile')
    if __custatevecLoggerOpenFile == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecLoggerOpenFile = dlsym(handle, 'custatevecLoggerOpenFile')
    
    global __custatevecLoggerSetLevel
    __custatevecLoggerSetLevel = dlsym(RTLD_DEFAULT, 'custatevecLoggerSetLevel')
    if __custatevecLoggerSetLevel == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecLoggerSetLevel = dlsym(handle, 'custatevecLoggerSetLevel')
    
    global __custatevecLoggerSetMask
    __custatevecLoggerSetMask = dlsym(RTLD_DEFAULT, 'custatevecLoggerSetMask')
    if __custatevecLoggerSetMask == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecLoggerSetMask = dlsym(handle, 'custatevecLoggerSetMask')
    
    global __custatevecLoggerForceDisable
    __custatevecLoggerForceDisable = dlsym(RTLD_DEFAULT, 'custatevecLoggerForceDisable')
    if __custatevecLoggerForceDisable == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecLoggerForceDisable = dlsym(handle, 'custatevecLoggerForceDisable')
    
    global __custatevecGetDeviceMemHandler
    __custatevecGetDeviceMemHandler = dlsym(RTLD_DEFAULT, 'custatevecGetDeviceMemHandler')
    if __custatevecGetDeviceMemHandler == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecGetDeviceMemHandler = dlsym(handle, 'custatevecGetDeviceMemHandler')
    
    global __custatevecSetDeviceMemHandler
    __custatevecSetDeviceMemHandler = dlsym(RTLD_DEFAULT, 'custatevecSetDeviceMemHandler')
    if __custatevecSetDeviceMemHandler == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSetDeviceMemHandler = dlsym(handle, 'custatevecSetDeviceMemHandler')
    
    global __custatevecAbs2SumOnZBasis
    __custatevecAbs2SumOnZBasis = dlsym(RTLD_DEFAULT, 'custatevecAbs2SumOnZBasis')
    if __custatevecAbs2SumOnZBasis == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecAbs2SumOnZBasis = dlsym(handle, 'custatevecAbs2SumOnZBasis')
    
    global __custatevecAbs2SumArray
    __custatevecAbs2SumArray = dlsym(RTLD_DEFAULT, 'custatevecAbs2SumArray')
    if __custatevecAbs2SumArray == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecAbs2SumArray = dlsym(handle, 'custatevecAbs2SumArray')
    
    global __custatevecCollapseOnZBasis
    __custatevecCollapseOnZBasis = dlsym(RTLD_DEFAULT, 'custatevecCollapseOnZBasis')
    if __custatevecCollapseOnZBasis == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecCollapseOnZBasis = dlsym(handle, 'custatevecCollapseOnZBasis')
    
    global __custatevecCollapseByBitString
    __custatevecCollapseByBitString = dlsym(RTLD_DEFAULT, 'custatevecCollapseByBitString')
    if __custatevecCollapseByBitString == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecCollapseByBitString = dlsym(handle, 'custatevecCollapseByBitString')
    
    global __custatevecMeasureOnZBasis
    __custatevecMeasureOnZBasis = dlsym(RTLD_DEFAULT, 'custatevecMeasureOnZBasis')
    if __custatevecMeasureOnZBasis == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecMeasureOnZBasis = dlsym(handle, 'custatevecMeasureOnZBasis')
    
    global __custatevecBatchMeasure
    __custatevecBatchMeasure = dlsym(RTLD_DEFAULT, 'custatevecBatchMeasure')
    if __custatevecBatchMeasure == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecBatchMeasure = dlsym(handle, 'custatevecBatchMeasure')
    
    global __custatevecBatchMeasureWithOffset
    __custatevecBatchMeasureWithOffset = dlsym(RTLD_DEFAULT, 'custatevecBatchMeasureWithOffset')
    if __custatevecBatchMeasureWithOffset == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecBatchMeasureWithOffset = dlsym(handle, 'custatevecBatchMeasureWithOffset')
    
    global __custatevecApplyPauliRotation
    __custatevecApplyPauliRotation = dlsym(RTLD_DEFAULT, 'custatevecApplyPauliRotation')
    if __custatevecApplyPauliRotation == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecApplyPauliRotation = dlsym(handle, 'custatevecApplyPauliRotation')
    
    global __custatevecApplyMatrixGetWorkspaceSize
    __custatevecApplyMatrixGetWorkspaceSize = dlsym(RTLD_DEFAULT, 'custatevecApplyMatrixGetWorkspaceSize')
    if __custatevecApplyMatrixGetWorkspaceSize == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecApplyMatrixGetWorkspaceSize = dlsym(handle, 'custatevecApplyMatrixGetWorkspaceSize')
    
    global __custatevecApplyMatrix
    __custatevecApplyMatrix = dlsym(RTLD_DEFAULT, 'custatevecApplyMatrix')
    if __custatevecApplyMatrix == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecApplyMatrix = dlsym(handle, 'custatevecApplyMatrix')
    
    global __custatevecComputeExpectationGetWorkspaceSize
    __custatevecComputeExpectationGetWorkspaceSize = dlsym(RTLD_DEFAULT, 'custatevecComputeExpectationGetWorkspaceSize')
    if __custatevecComputeExpectationGetWorkspaceSize == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecComputeExpectationGetWorkspaceSize = dlsym(handle, 'custatevecComputeExpectationGetWorkspaceSize')
    
    global __custatevecComputeExpectation
    __custatevecComputeExpectation = dlsym(RTLD_DEFAULT, 'custatevecComputeExpectation')
    if __custatevecComputeExpectation == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecComputeExpectation = dlsym(handle, 'custatevecComputeExpectation')
    
    global __custatevecSamplerCreate
    __custatevecSamplerCreate = dlsym(RTLD_DEFAULT, 'custatevecSamplerCreate')
    if __custatevecSamplerCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSamplerCreate = dlsym(handle, 'custatevecSamplerCreate')
    
    global __custatevecSamplerDestroy
    __custatevecSamplerDestroy = dlsym(RTLD_DEFAULT, 'custatevecSamplerDestroy')
    if __custatevecSamplerDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSamplerDestroy = dlsym(handle, 'custatevecSamplerDestroy')
    
    global __custatevecSamplerPreprocess
    __custatevecSamplerPreprocess = dlsym(RTLD_DEFAULT, 'custatevecSamplerPreprocess')
    if __custatevecSamplerPreprocess == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSamplerPreprocess = dlsym(handle, 'custatevecSamplerPreprocess')
    
    global __custatevecSamplerGetSquaredNorm
    __custatevecSamplerGetSquaredNorm = dlsym(RTLD_DEFAULT, 'custatevecSamplerGetSquaredNorm')
    if __custatevecSamplerGetSquaredNorm == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSamplerGetSquaredNorm = dlsym(handle, 'custatevecSamplerGetSquaredNorm')
    
    global __custatevecSamplerApplySubSVOffset
    __custatevecSamplerApplySubSVOffset = dlsym(RTLD_DEFAULT, 'custatevecSamplerApplySubSVOffset')
    if __custatevecSamplerApplySubSVOffset == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSamplerApplySubSVOffset = dlsym(handle, 'custatevecSamplerApplySubSVOffset')
    
    global __custatevecSamplerSample
    __custatevecSamplerSample = dlsym(RTLD_DEFAULT, 'custatevecSamplerSample')
    if __custatevecSamplerSample == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSamplerSample = dlsym(handle, 'custatevecSamplerSample')
    
    global __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize
    __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize = dlsym(RTLD_DEFAULT, 'custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize')
    if __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize = dlsym(handle, 'custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize')
    
    global __custatevecApplyGeneralizedPermutationMatrix
    __custatevecApplyGeneralizedPermutationMatrix = dlsym(RTLD_DEFAULT, 'custatevecApplyGeneralizedPermutationMatrix')
    if __custatevecApplyGeneralizedPermutationMatrix == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecApplyGeneralizedPermutationMatrix = dlsym(handle, 'custatevecApplyGeneralizedPermutationMatrix')
    
    global __custatevecComputeExpectationsOnPauliBasis
    __custatevecComputeExpectationsOnPauliBasis = dlsym(RTLD_DEFAULT, 'custatevecComputeExpectationsOnPauliBasis')
    if __custatevecComputeExpectationsOnPauliBasis == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecComputeExpectationsOnPauliBasis = dlsym(handle, 'custatevecComputeExpectationsOnPauliBasis')
    
    global __custatevecAccessorCreate
    __custatevecAccessorCreate = dlsym(RTLD_DEFAULT, 'custatevecAccessorCreate')
    if __custatevecAccessorCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecAccessorCreate = dlsym(handle, 'custatevecAccessorCreate')
    
    global __custatevecAccessorCreateView
    __custatevecAccessorCreateView = dlsym(RTLD_DEFAULT, 'custatevecAccessorCreateView')
    if __custatevecAccessorCreateView == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecAccessorCreateView = dlsym(handle, 'custatevecAccessorCreateView')
    
    global __custatevecAccessorDestroy
    __custatevecAccessorDestroy = dlsym(RTLD_DEFAULT, 'custatevecAccessorDestroy')
    if __custatevecAccessorDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecAccessorDestroy = dlsym(handle, 'custatevecAccessorDestroy')
    
    global __custatevecAccessorSetExtraWorkspace
    __custatevecAccessorSetExtraWorkspace = dlsym(RTLD_DEFAULT, 'custatevecAccessorSetExtraWorkspace')
    if __custatevecAccessorSetExtraWorkspace == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecAccessorSetExtraWorkspace = dlsym(handle, 'custatevecAccessorSetExtraWorkspace')
    
    global __custatevecAccessorGet
    __custatevecAccessorGet = dlsym(RTLD_DEFAULT, 'custatevecAccessorGet')
    if __custatevecAccessorGet == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecAccessorGet = dlsym(handle, 'custatevecAccessorGet')
    
    global __custatevecAccessorSet
    __custatevecAccessorSet = dlsym(RTLD_DEFAULT, 'custatevecAccessorSet')
    if __custatevecAccessorSet == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecAccessorSet = dlsym(handle, 'custatevecAccessorSet')
    
    global __custatevecSwapIndexBits
    __custatevecSwapIndexBits = dlsym(RTLD_DEFAULT, 'custatevecSwapIndexBits')
    if __custatevecSwapIndexBits == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSwapIndexBits = dlsym(handle, 'custatevecSwapIndexBits')
    
    global __custatevecTestMatrixTypeGetWorkspaceSize
    __custatevecTestMatrixTypeGetWorkspaceSize = dlsym(RTLD_DEFAULT, 'custatevecTestMatrixTypeGetWorkspaceSize')
    if __custatevecTestMatrixTypeGetWorkspaceSize == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecTestMatrixTypeGetWorkspaceSize = dlsym(handle, 'custatevecTestMatrixTypeGetWorkspaceSize')
    
    global __custatevecTestMatrixType
    __custatevecTestMatrixType = dlsym(RTLD_DEFAULT, 'custatevecTestMatrixType')
    if __custatevecTestMatrixType == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecTestMatrixType = dlsym(handle, 'custatevecTestMatrixType')
    
    global __custatevecMultiDeviceSwapIndexBits
    __custatevecMultiDeviceSwapIndexBits = dlsym(RTLD_DEFAULT, 'custatevecMultiDeviceSwapIndexBits')
    if __custatevecMultiDeviceSwapIndexBits == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecMultiDeviceSwapIndexBits = dlsym(handle, 'custatevecMultiDeviceSwapIndexBits')
    
    global __custatevecCommunicatorCreate
    __custatevecCommunicatorCreate = dlsym(RTLD_DEFAULT, 'custatevecCommunicatorCreate')
    if __custatevecCommunicatorCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecCommunicatorCreate = dlsym(handle, 'custatevecCommunicatorCreate')
    
    global __custatevecCommunicatorDestroy
    __custatevecCommunicatorDestroy = dlsym(RTLD_DEFAULT, 'custatevecCommunicatorDestroy')
    if __custatevecCommunicatorDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecCommunicatorDestroy = dlsym(handle, 'custatevecCommunicatorDestroy')
    
    global __custatevecDistIndexBitSwapSchedulerCreate
    __custatevecDistIndexBitSwapSchedulerCreate = dlsym(RTLD_DEFAULT, 'custatevecDistIndexBitSwapSchedulerCreate')
    if __custatevecDistIndexBitSwapSchedulerCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecDistIndexBitSwapSchedulerCreate = dlsym(handle, 'custatevecDistIndexBitSwapSchedulerCreate')
    
    global __custatevecDistIndexBitSwapSchedulerDestroy
    __custatevecDistIndexBitSwapSchedulerDestroy = dlsym(RTLD_DEFAULT, 'custatevecDistIndexBitSwapSchedulerDestroy')
    if __custatevecDistIndexBitSwapSchedulerDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecDistIndexBitSwapSchedulerDestroy = dlsym(handle, 'custatevecDistIndexBitSwapSchedulerDestroy')
    
    global __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps
    __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps = dlsym(RTLD_DEFAULT, 'custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps')
    if __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps = dlsym(handle, 'custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps')
    
    global __custatevecDistIndexBitSwapSchedulerGetParameters
    __custatevecDistIndexBitSwapSchedulerGetParameters = dlsym(RTLD_DEFAULT, 'custatevecDistIndexBitSwapSchedulerGetParameters')
    if __custatevecDistIndexBitSwapSchedulerGetParameters == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecDistIndexBitSwapSchedulerGetParameters = dlsym(handle, 'custatevecDistIndexBitSwapSchedulerGetParameters')
    
    global __custatevecSVSwapWorkerCreate
    __custatevecSVSwapWorkerCreate = dlsym(RTLD_DEFAULT, 'custatevecSVSwapWorkerCreate')
    if __custatevecSVSwapWorkerCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSVSwapWorkerCreate = dlsym(handle, 'custatevecSVSwapWorkerCreate')
    
    global __custatevecSVSwapWorkerDestroy
    __custatevecSVSwapWorkerDestroy = dlsym(RTLD_DEFAULT, 'custatevecSVSwapWorkerDestroy')
    if __custatevecSVSwapWorkerDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSVSwapWorkerDestroy = dlsym(handle, 'custatevecSVSwapWorkerDestroy')
    
    global __custatevecSVSwapWorkerSetExtraWorkspace
    __custatevecSVSwapWorkerSetExtraWorkspace = dlsym(RTLD_DEFAULT, 'custatevecSVSwapWorkerSetExtraWorkspace')
    if __custatevecSVSwapWorkerSetExtraWorkspace == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSVSwapWorkerSetExtraWorkspace = dlsym(handle, 'custatevecSVSwapWorkerSetExtraWorkspace')
    
    global __custatevecSVSwapWorkerSetTransferWorkspace
    __custatevecSVSwapWorkerSetTransferWorkspace = dlsym(RTLD_DEFAULT, 'custatevecSVSwapWorkerSetTransferWorkspace')
    if __custatevecSVSwapWorkerSetTransferWorkspace == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSVSwapWorkerSetTransferWorkspace = dlsym(handle, 'custatevecSVSwapWorkerSetTransferWorkspace')
    
    global __custatevecSVSwapWorkerSetSubSVsP2P
    __custatevecSVSwapWorkerSetSubSVsP2P = dlsym(RTLD_DEFAULT, 'custatevecSVSwapWorkerSetSubSVsP2P')
    if __custatevecSVSwapWorkerSetSubSVsP2P == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSVSwapWorkerSetSubSVsP2P = dlsym(handle, 'custatevecSVSwapWorkerSetSubSVsP2P')
    
    global __custatevecSVSwapWorkerSetParameters
    __custatevecSVSwapWorkerSetParameters = dlsym(RTLD_DEFAULT, 'custatevecSVSwapWorkerSetParameters')
    if __custatevecSVSwapWorkerSetParameters == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSVSwapWorkerSetParameters = dlsym(handle, 'custatevecSVSwapWorkerSetParameters')
    
    global __custatevecSVSwapWorkerExecute
    __custatevecSVSwapWorkerExecute = dlsym(RTLD_DEFAULT, 'custatevecSVSwapWorkerExecute')
    if __custatevecSVSwapWorkerExecute == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSVSwapWorkerExecute = dlsym(handle, 'custatevecSVSwapWorkerExecute')
    
    global __custatevecInitializeStateVector
    __custatevecInitializeStateVector = dlsym(RTLD_DEFAULT, 'custatevecInitializeStateVector')
    if __custatevecInitializeStateVector == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecInitializeStateVector = dlsym(handle, 'custatevecInitializeStateVector')
    
    global __custatevecApplyMatrixBatchedGetWorkspaceSize
    __custatevecApplyMatrixBatchedGetWorkspaceSize = dlsym(RTLD_DEFAULT, 'custatevecApplyMatrixBatchedGetWorkspaceSize')
    if __custatevecApplyMatrixBatchedGetWorkspaceSize == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecApplyMatrixBatchedGetWorkspaceSize = dlsym(handle, 'custatevecApplyMatrixBatchedGetWorkspaceSize')
    
    global __custatevecApplyMatrixBatched
    __custatevecApplyMatrixBatched = dlsym(RTLD_DEFAULT, 'custatevecApplyMatrixBatched')
    if __custatevecApplyMatrixBatched == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecApplyMatrixBatched = dlsym(handle, 'custatevecApplyMatrixBatched')
    
    global __custatevecAbs2SumArrayBatched
    __custatevecAbs2SumArrayBatched = dlsym(RTLD_DEFAULT, 'custatevecAbs2SumArrayBatched')
    if __custatevecAbs2SumArrayBatched == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecAbs2SumArrayBatched = dlsym(handle, 'custatevecAbs2SumArrayBatched')
    
    global __custatevecCollapseByBitStringBatchedGetWorkspaceSize
    __custatevecCollapseByBitStringBatchedGetWorkspaceSize = dlsym(RTLD_DEFAULT, 'custatevecCollapseByBitStringBatchedGetWorkspaceSize')
    if __custatevecCollapseByBitStringBatchedGetWorkspaceSize == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecCollapseByBitStringBatchedGetWorkspaceSize = dlsym(handle, 'custatevecCollapseByBitStringBatchedGetWorkspaceSize')
    
    global __custatevecCollapseByBitStringBatched
    __custatevecCollapseByBitStringBatched = dlsym(RTLD_DEFAULT, 'custatevecCollapseByBitStringBatched')
    if __custatevecCollapseByBitStringBatched == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecCollapseByBitStringBatched = dlsym(handle, 'custatevecCollapseByBitStringBatched')
    
    global __custatevecMeasureBatched
    __custatevecMeasureBatched = dlsym(RTLD_DEFAULT, 'custatevecMeasureBatched')
    if __custatevecMeasureBatched == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecMeasureBatched = dlsym(handle, 'custatevecMeasureBatched')
    
    global __custatevecSubSVMigratorCreate
    __custatevecSubSVMigratorCreate = dlsym(RTLD_DEFAULT, 'custatevecSubSVMigratorCreate')
    if __custatevecSubSVMigratorCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSubSVMigratorCreate = dlsym(handle, 'custatevecSubSVMigratorCreate')
    
    global __custatevecSubSVMigratorDestroy
    __custatevecSubSVMigratorDestroy = dlsym(RTLD_DEFAULT, 'custatevecSubSVMigratorDestroy')
    if __custatevecSubSVMigratorDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSubSVMigratorDestroy = dlsym(handle, 'custatevecSubSVMigratorDestroy')
    
    global __custatevecSubSVMigratorMigrate
    __custatevecSubSVMigratorMigrate = dlsym(RTLD_DEFAULT, 'custatevecSubSVMigratorMigrate')
    if __custatevecSubSVMigratorMigrate == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecSubSVMigratorMigrate = dlsym(handle, 'custatevecSubSVMigratorMigrate')
    
    global __custatevecComputeExpectationBatchedGetWorkspaceSize
    __custatevecComputeExpectationBatchedGetWorkspaceSize = dlsym(RTLD_DEFAULT, 'custatevecComputeExpectationBatchedGetWorkspaceSize')
    if __custatevecComputeExpectationBatchedGetWorkspaceSize == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecComputeExpectationBatchedGetWorkspaceSize = dlsym(handle, 'custatevecComputeExpectationBatchedGetWorkspaceSize')
    
    global __custatevecComputeExpectationBatched
    __custatevecComputeExpectationBatched = dlsym(RTLD_DEFAULT, 'custatevecComputeExpectationBatched')
    if __custatevecComputeExpectationBatched == NULL:
        if handle == NULL:
            handle = load_library()
        __custatevecComputeExpectationBatched = dlsym(handle, 'custatevecComputeExpectationBatched')

    __py_custatevec_init = True
    return 0


cpdef dict _inspect_function_pointers():
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
    
    global __custatevecLoggerSetCallbackData
    data["__custatevecLoggerSetCallbackData"] = <intptr_t>__custatevecLoggerSetCallbackData
    
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

    return data


###############################################################################
# Wrapper functions
###############################################################################

cdef custatevecStatus_t _custatevecCreate(custatevecHandle_t* handle) except* nogil:
    global __custatevecCreate
    _check_or_init_custatevec()
    if __custatevecCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t*) nogil>__custatevecCreate)(
        handle)


cdef custatevecStatus_t _custatevecDestroy(custatevecHandle_t handle) except* nogil:
    global __custatevecDestroy
    _check_or_init_custatevec()
    if __custatevecDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t) nogil>__custatevecDestroy)(
        handle)


cdef custatevecStatus_t _custatevecGetDefaultWorkspaceSize(custatevecHandle_t handle, size_t* workspaceSizeInBytes) except* nogil:
    global __custatevecGetDefaultWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecGetDefaultWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetDefaultWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, size_t*) nogil>__custatevecGetDefaultWorkspaceSize)(
        handle, workspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSetWorkspace(custatevecHandle_t handle, void* workspace, size_t workspaceSizeInBytes) except* nogil:
    global __custatevecSetWorkspace
    _check_or_init_custatevec()
    if __custatevecSetWorkspace == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSetWorkspace is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, size_t) nogil>__custatevecSetWorkspace)(
        handle, workspace, workspaceSizeInBytes)


cdef const char* _custatevecGetErrorName(custatevecStatus_t status) except* nogil:
    global __custatevecGetErrorName
    _check_or_init_custatevec()
    if __custatevecGetErrorName == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetErrorName is not found")
    return (<const char* (*)(custatevecStatus_t) nogil>__custatevecGetErrorName)(
        status)


cdef const char* _custatevecGetErrorString(custatevecStatus_t status) except* nogil:
    global __custatevecGetErrorString
    _check_or_init_custatevec()
    if __custatevecGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetErrorString is not found")
    return (<const char* (*)(custatevecStatus_t) nogil>__custatevecGetErrorString)(
        status)


cdef custatevecStatus_t _custatevecGetProperty(libraryPropertyType type, int32_t* value) except* nogil:
    global __custatevecGetProperty
    _check_or_init_custatevec()
    if __custatevecGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetProperty is not found")
    return (<custatevecStatus_t (*)(libraryPropertyType, int32_t*) nogil>__custatevecGetProperty)(
        type, value)


cdef size_t _custatevecGetVersion() except* nogil:
    global __custatevecGetVersion
    _check_or_init_custatevec()
    if __custatevecGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetVersion is not found")
    return (<size_t (*)() nogil>__custatevecGetVersion)(
        )


cdef custatevecStatus_t _custatevecSetStream(custatevecHandle_t handle, cudaStream_t streamId) except* nogil:
    global __custatevecSetStream
    _check_or_init_custatevec()
    if __custatevecSetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSetStream is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaStream_t) nogil>__custatevecSetStream)(
        handle, streamId)


cdef custatevecStatus_t _custatevecGetStream(custatevecHandle_t handle, cudaStream_t* streamId) except* nogil:
    global __custatevecGetStream
    _check_or_init_custatevec()
    if __custatevecGetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetStream is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaStream_t*) nogil>__custatevecGetStream)(
        handle, streamId)


cdef custatevecStatus_t _custatevecLoggerSetCallbackData(custatevecLoggerCallbackData_t callback, void* userData) except* nogil:
    global __custatevecLoggerSetCallbackData
    _check_or_init_custatevec()
    if __custatevecLoggerSetCallbackData == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerSetCallbackData is not found")
    return (<custatevecStatus_t (*)(custatevecLoggerCallbackData_t, void*) nogil>__custatevecLoggerSetCallbackData)(
        callback, userData)


cdef custatevecStatus_t _custatevecLoggerOpenFile(const char* logFile) except* nogil:
    global __custatevecLoggerOpenFile
    _check_or_init_custatevec()
    if __custatevecLoggerOpenFile == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerOpenFile is not found")
    return (<custatevecStatus_t (*)(const char*) nogil>__custatevecLoggerOpenFile)(
        logFile)


cdef custatevecStatus_t _custatevecLoggerSetLevel(int32_t level) except* nogil:
    global __custatevecLoggerSetLevel
    _check_or_init_custatevec()
    if __custatevecLoggerSetLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerSetLevel is not found")
    return (<custatevecStatus_t (*)(int32_t) nogil>__custatevecLoggerSetLevel)(
        level)


cdef custatevecStatus_t _custatevecLoggerSetMask(int32_t mask) except* nogil:
    global __custatevecLoggerSetMask
    _check_or_init_custatevec()
    if __custatevecLoggerSetMask == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerSetMask is not found")
    return (<custatevecStatus_t (*)(int32_t) nogil>__custatevecLoggerSetMask)(
        mask)


cdef custatevecStatus_t _custatevecLoggerForceDisable() except* nogil:
    global __custatevecLoggerForceDisable
    _check_or_init_custatevec()
    if __custatevecLoggerForceDisable == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecLoggerForceDisable is not found")
    return (<custatevecStatus_t (*)() nogil>__custatevecLoggerForceDisable)(
        )


cdef custatevecStatus_t _custatevecGetDeviceMemHandler(custatevecHandle_t handle, custatevecDeviceMemHandler_t* handler) except* nogil:
    global __custatevecGetDeviceMemHandler
    _check_or_init_custatevec()
    if __custatevecGetDeviceMemHandler == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecGetDeviceMemHandler is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecDeviceMemHandler_t*) nogil>__custatevecGetDeviceMemHandler)(
        handle, handler)


cdef custatevecStatus_t _custatevecSetDeviceMemHandler(custatevecHandle_t handle, const custatevecDeviceMemHandler_t* handler) except* nogil:
    global __custatevecSetDeviceMemHandler
    _check_or_init_custatevec()
    if __custatevecSetDeviceMemHandler == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSetDeviceMemHandler is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const custatevecDeviceMemHandler_t*) nogil>__custatevecSetDeviceMemHandler)(
        handle, handler)


cdef custatevecStatus_t _custatevecAbs2SumOnZBasis(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* abs2sum0, double* abs2sum1, const int32_t* basisBits, const uint32_t nBasisBits) except* nogil:
    global __custatevecAbs2SumOnZBasis
    _check_or_init_custatevec()
    if __custatevecAbs2SumOnZBasis == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAbs2SumOnZBasis is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, double*, double*, const int32_t*, const uint32_t) nogil>__custatevecAbs2SumOnZBasis)(
        handle, sv, svDataType, nIndexBits, abs2sum0, abs2sum1, basisBits, nBasisBits)


cdef custatevecStatus_t _custatevecAbs2SumArray(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* abs2sum, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen) except* nogil:
    global __custatevecAbs2SumArray
    _check_or_init_custatevec()
    if __custatevecAbs2SumArray == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAbs2SumArray is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, double*, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t) nogil>__custatevecAbs2SumArray)(
        handle, sv, svDataType, nIndexBits, abs2sum, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen)


cdef custatevecStatus_t _custatevecCollapseOnZBasis(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int32_t parity, const int32_t* basisBits, const uint32_t nBasisBits, double norm) except* nogil:
    global __custatevecCollapseOnZBasis
    _check_or_init_custatevec()
    if __custatevecCollapseOnZBasis == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCollapseOnZBasis is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const int32_t, const int32_t*, const uint32_t, double) nogil>__custatevecCollapseOnZBasis)(
        handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, norm)


cdef custatevecStatus_t _custatevecCollapseByBitString(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, double norm) except* nogil:
    global __custatevecCollapseByBitString
    _check_or_init_custatevec()
    if __custatevecCollapseByBitString == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCollapseByBitString is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const int32_t*, const int32_t*, const uint32_t, double) nogil>__custatevecCollapseByBitString)(
        handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, norm)


cdef custatevecStatus_t _custatevecMeasureOnZBasis(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* parity, const int32_t* basisBits, const uint32_t nBasisBits, const double randnum, custatevecCollapseOp_t collapse) except* nogil:
    global __custatevecMeasureOnZBasis
    _check_or_init_custatevec()
    if __custatevecMeasureOnZBasis == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecMeasureOnZBasis is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, int32_t*, const int32_t*, const uint32_t, const double, custatevecCollapseOp_t) nogil>__custatevecMeasureOnZBasis)(
        handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, randnum, collapse)


cdef custatevecStatus_t _custatevecBatchMeasure(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, const double randnum, custatevecCollapseOp_t collapse) except* nogil:
    global __custatevecBatchMeasure
    _check_or_init_custatevec()
    if __custatevecBatchMeasure == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecBatchMeasure is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, int32_t*, const int32_t*, const uint32_t, const double, custatevecCollapseOp_t) nogil>__custatevecBatchMeasure)(
        handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse)


cdef custatevecStatus_t _custatevecBatchMeasureWithOffset(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen, const double randnum, custatevecCollapseOp_t collapse, const double offset, const double abs2sum) except* nogil:
    global __custatevecBatchMeasureWithOffset
    _check_or_init_custatevec()
    if __custatevecBatchMeasureWithOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecBatchMeasureWithOffset is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, int32_t*, const int32_t*, const uint32_t, const double, custatevecCollapseOp_t, const double, const double) nogil>__custatevecBatchMeasureWithOffset)(
        handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse, offset, abs2sum)


cdef custatevecStatus_t _custatevecApplyPauliRotation(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double theta, const custatevecPauli_t* paulis, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls) except* nogil:
    global __custatevecApplyPauliRotation
    _check_or_init_custatevec()
    if __custatevecApplyPauliRotation == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyPauliRotation is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, double, const custatevecPauli_t*, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t) nogil>__custatevecApplyPauliRotation)(
        handle, sv, svDataType, nIndexBits, theta, paulis, targets, nTargets, controls, controlBitValues, nControls)


cdef custatevecStatus_t _custatevecApplyMatrixGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nTargets, const uint32_t nControls, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecApplyMatrixGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecApplyMatrixGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyMatrixGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaDataType_t, const uint32_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const int32_t, const uint32_t, const uint32_t, custatevecComputeType_t, size_t*) nogil>__custatevecApplyMatrixGetWorkspaceSize)(
        handle, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecApplyMatrix(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecApplyMatrix
    _check_or_init_custatevec()
    if __custatevecApplyMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyMatrix is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const int32_t, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, custatevecComputeType_t, void*, size_t) nogil>__custatevecApplyMatrix)(
        handle, sv, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, targets, nTargets, controls, controlBitValues, nControls, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecComputeExpectationGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nBasisBits, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecComputeExpectationGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecComputeExpectationGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecComputeExpectationGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaDataType_t, const uint32_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const uint32_t, custatevecComputeType_t, size_t*) nogil>__custatevecComputeExpectationGetWorkspaceSize)(
        handle, svDataType, nIndexBits, matrix, matrixDataType, layout, nBasisBits, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecComputeExpectation(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, void* expectationValue, cudaDataType_t expectationDataType, double* residualNorm, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t* basisBits, const uint32_t nBasisBits, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecComputeExpectation
    _check_or_init_custatevec()
    if __custatevecComputeExpectation == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecComputeExpectation is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, void*, cudaDataType_t, double*, const void*, cudaDataType_t, custatevecMatrixLayout_t, const int32_t*, const uint32_t, custatevecComputeType_t, void*, size_t) nogil>__custatevecComputeExpectation)(
        handle, sv, svDataType, nIndexBits, expectationValue, expectationDataType, residualNorm, matrix, matrixDataType, layout, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSamplerCreate(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecSamplerDescriptor_t* sampler, uint32_t nMaxShots, size_t* extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecSamplerCreate
    _check_or_init_custatevec()
    if __custatevecSamplerCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, custatevecSamplerDescriptor_t*, uint32_t, size_t*) nogil>__custatevecSamplerCreate)(
        handle, sv, svDataType, nIndexBits, sampler, nMaxShots, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSamplerDestroy(custatevecSamplerDescriptor_t sampler) except* nogil:
    global __custatevecSamplerDestroy
    _check_or_init_custatevec()
    if __custatevecSamplerDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecSamplerDescriptor_t) nogil>__custatevecSamplerDestroy)(
        sampler)


cdef custatevecStatus_t _custatevecSamplerPreprocess(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, void* extraWorkspace, const size_t extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecSamplerPreprocess
    _check_or_init_custatevec()
    if __custatevecSamplerPreprocess == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerPreprocess is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSamplerDescriptor_t, void*, const size_t) nogil>__custatevecSamplerPreprocess)(
        handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSamplerGetSquaredNorm(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, double* norm) except* nogil:
    global __custatevecSamplerGetSquaredNorm
    _check_or_init_custatevec()
    if __custatevecSamplerGetSquaredNorm == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerGetSquaredNorm is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSamplerDescriptor_t, double*) nogil>__custatevecSamplerGetSquaredNorm)(
        handle, sampler, norm)


cdef custatevecStatus_t _custatevecSamplerApplySubSVOffset(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, int32_t subSVOrd, uint32_t nSubSVs, double offset, double norm) except* nogil:
    global __custatevecSamplerApplySubSVOffset
    _check_or_init_custatevec()
    if __custatevecSamplerApplySubSVOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerApplySubSVOffset is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSamplerDescriptor_t, int32_t, uint32_t, double, double) nogil>__custatevecSamplerApplySubSVOffset)(
        handle, sampler, subSVOrd, nSubSVs, offset, norm)


cdef custatevecStatus_t _custatevecSamplerSample(custatevecHandle_t handle, custatevecSamplerDescriptor_t sampler, custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* randnums, const uint32_t nShots, custatevecSamplerOutput_t output) except* nogil:
    global __custatevecSamplerSample
    _check_or_init_custatevec()
    if __custatevecSamplerSample == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSamplerSample is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSamplerDescriptor_t, custatevecIndex_t*, const int32_t*, const uint32_t, const double*, const uint32_t, custatevecSamplerOutput_t) nogil>__custatevecSamplerSample)(
        handle, sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots, output)


cdef custatevecStatus_t _custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType, const int32_t* targets, const uint32_t nTargets, const uint32_t nControls, size_t* extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaDataType_t, const uint32_t, const custatevecIndex_t*, const void*, cudaDataType_t, const int32_t*, const uint32_t, const uint32_t, size_t*) nogil>__custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize)(
        handle, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, targets, nTargets, nControls, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecApplyGeneralizedPermutationMatrix(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType, const int32_t adjoint, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecApplyGeneralizedPermutationMatrix
    _check_or_init_custatevec()
    if __custatevecApplyGeneralizedPermutationMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyGeneralizedPermutationMatrix is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, custatevecIndex_t*, const void*, cudaDataType_t, const int32_t, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, void*, size_t) nogil>__custatevecApplyGeneralizedPermutationMatrix)(
        handle, sv, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, adjoint, targets, nTargets, controls, controlBitValues, nControls, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecComputeExpectationsOnPauliBasis(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, double* expectationValues, const custatevecPauli_t** pauliOperatorsArray, const uint32_t nPauliOperatorArrays, const int32_t** basisBitsArray, const uint32_t* nBasisBitsArray) except* nogil:
    global __custatevecComputeExpectationsOnPauliBasis
    _check_or_init_custatevec()
    if __custatevecComputeExpectationsOnPauliBasis == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecComputeExpectationsOnPauliBasis is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, double*, const custatevecPauli_t**, const uint32_t, const int32_t**, const uint32_t*) nogil>__custatevecComputeExpectationsOnPauliBasis)(
        handle, sv, svDataType, nIndexBits, expectationValues, pauliOperatorsArray, nPauliOperatorArrays, basisBitsArray, nBasisBitsArray)


cdef custatevecStatus_t _custatevecAccessorCreate(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecAccessorDescriptor_t* accessor, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, size_t* extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecAccessorCreate
    _check_or_init_custatevec()
    if __custatevecAccessorCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, custatevecAccessorDescriptor_t*, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, size_t*) nogil>__custatevecAccessorCreate)(
        handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecAccessorCreateView(custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecAccessorDescriptor_t* accessor, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, size_t* extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecAccessorCreateView
    _check_or_init_custatevec()
    if __custatevecAccessorCreateView == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorCreateView is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, custatevecAccessorDescriptor_t*, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, size_t*) nogil>__custatevecAccessorCreateView)(
        handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecAccessorDestroy(custatevecAccessorDescriptor_t accessor) except* nogil:
    global __custatevecAccessorDestroy
    _check_or_init_custatevec()
    if __custatevecAccessorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecAccessorDescriptor_t) nogil>__custatevecAccessorDestroy)(
        accessor)


cdef custatevecStatus_t _custatevecAccessorSetExtraWorkspace(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecAccessorSetExtraWorkspace
    _check_or_init_custatevec()
    if __custatevecAccessorSetExtraWorkspace == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorSetExtraWorkspace is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecAccessorDescriptor_t, void*, size_t) nogil>__custatevecAccessorSetExtraWorkspace)(
        handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecAccessorGet(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, void* externalBuffer, const custatevecIndex_t begin, const custatevecIndex_t end) except* nogil:
    global __custatevecAccessorGet
    _check_or_init_custatevec()
    if __custatevecAccessorGet == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorGet is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecAccessorDescriptor_t, void*, const custatevecIndex_t, const custatevecIndex_t) nogil>__custatevecAccessorGet)(
        handle, accessor, externalBuffer, begin, end)


cdef custatevecStatus_t _custatevecAccessorSet(custatevecHandle_t handle, custatevecAccessorDescriptor_t accessor, const void* externalBuffer, const custatevecIndex_t begin, const custatevecIndex_t end) except* nogil:
    global __custatevecAccessorSet
    _check_or_init_custatevec()
    if __custatevecAccessorSet == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAccessorSet is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecAccessorDescriptor_t, const void*, const custatevecIndex_t, const custatevecIndex_t) nogil>__custatevecAccessorSet)(
        handle, accessor, externalBuffer, begin, end)


cdef custatevecStatus_t _custatevecSwapIndexBits(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, const int2* bitSwaps, const uint32_t nBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen) except* nogil:
    global __custatevecSwapIndexBits
    _check_or_init_custatevec()
    if __custatevecSwapIndexBits == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSwapIndexBits is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const int2*, const uint32_t, const int32_t*, const int32_t*, const uint32_t) nogil>__custatevecSwapIndexBits)(
        handle, sv, svDataType, nIndexBits, bitSwaps, nBitSwaps, maskBitString, maskOrdering, maskLen)


cdef custatevecStatus_t _custatevecTestMatrixTypeGetWorkspaceSize(custatevecHandle_t handle, custatevecMatrixType_t matrixType, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nTargets, const int32_t adjoint, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecTestMatrixTypeGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecTestMatrixTypeGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecTestMatrixTypeGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecMatrixType_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const uint32_t, const int32_t, custatevecComputeType_t, size_t*) nogil>__custatevecTestMatrixTypeGetWorkspaceSize)(
        handle, matrixType, matrix, matrixDataType, layout, nTargets, adjoint, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecTestMatrixType(custatevecHandle_t handle, double* residualNorm, custatevecMatrixType_t matrixType, const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nTargets, const int32_t adjoint, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecTestMatrixType
    _check_or_init_custatevec()
    if __custatevecTestMatrixType == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecTestMatrixType is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, double*, custatevecMatrixType_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const uint32_t, const int32_t, custatevecComputeType_t, void*, size_t) nogil>__custatevecTestMatrixType)(
        handle, residualNorm, matrixType, matrix, matrixDataType, layout, nTargets, adjoint, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecMultiDeviceSwapIndexBits(custatevecHandle_t* handles, const uint32_t nHandles, void** subSVs, const cudaDataType_t svDataType, const uint32_t nGlobalIndexBits, const uint32_t nLocalIndexBits, const int2* indexBitSwaps, const uint32_t nIndexBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, const custatevecDeviceNetworkType_t deviceNetworkType) except* nogil:
    global __custatevecMultiDeviceSwapIndexBits
    _check_or_init_custatevec()
    if __custatevecMultiDeviceSwapIndexBits == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecMultiDeviceSwapIndexBits is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t*, const uint32_t, void**, const cudaDataType_t, const uint32_t, const uint32_t, const int2*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, const custatevecDeviceNetworkType_t) nogil>__custatevecMultiDeviceSwapIndexBits)(
        handles, nHandles, subSVs, svDataType, nGlobalIndexBits, nLocalIndexBits, indexBitSwaps, nIndexBitSwaps, maskBitString, maskOrdering, maskLen, deviceNetworkType)


cdef custatevecStatus_t _custatevecCommunicatorCreate(custatevecHandle_t handle, custatevecCommunicatorDescriptor_t* communicator, custatevecCommunicatorType_t communicatorType, const char* soname) except* nogil:
    global __custatevecCommunicatorCreate
    _check_or_init_custatevec()
    if __custatevecCommunicatorCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCommunicatorCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecCommunicatorDescriptor_t*, custatevecCommunicatorType_t, const char*) nogil>__custatevecCommunicatorCreate)(
        handle, communicator, communicatorType, soname)


cdef custatevecStatus_t _custatevecCommunicatorDestroy(custatevecHandle_t handle, custatevecCommunicatorDescriptor_t communicator) except* nogil:
    global __custatevecCommunicatorDestroy
    _check_or_init_custatevec()
    if __custatevecCommunicatorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCommunicatorDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecCommunicatorDescriptor_t) nogil>__custatevecCommunicatorDestroy)(
        handle, communicator)


cdef custatevecStatus_t _custatevecDistIndexBitSwapSchedulerCreate(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t* scheduler, const uint32_t nGlobalIndexBits, const uint32_t nLocalIndexBits) except* nogil:
    global __custatevecDistIndexBitSwapSchedulerCreate
    _check_or_init_custatevec()
    if __custatevecDistIndexBitSwapSchedulerCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecDistIndexBitSwapSchedulerCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecDistIndexBitSwapSchedulerDescriptor_t*, const uint32_t, const uint32_t) nogil>__custatevecDistIndexBitSwapSchedulerCreate)(
        handle, scheduler, nGlobalIndexBits, nLocalIndexBits)


cdef custatevecStatus_t _custatevecDistIndexBitSwapSchedulerDestroy(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler) except* nogil:
    global __custatevecDistIndexBitSwapSchedulerDestroy
    _check_or_init_custatevec()
    if __custatevecDistIndexBitSwapSchedulerDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecDistIndexBitSwapSchedulerDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecDistIndexBitSwapSchedulerDescriptor_t) nogil>__custatevecDistIndexBitSwapSchedulerDestroy)(
        handle, scheduler)


cdef custatevecStatus_t _custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler, const int2* indexBitSwaps, const uint32_t nIndexBitSwaps, const int32_t* maskBitString, const int32_t* maskOrdering, const uint32_t maskLen, uint32_t* nSwapBatches) except* nogil:
    global __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps
    _check_or_init_custatevec()
    if __custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecDistIndexBitSwapSchedulerDescriptor_t, const int2*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, uint32_t*) nogil>__custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps)(
        handle, scheduler, indexBitSwaps, nIndexBitSwaps, maskBitString, maskOrdering, maskLen, nSwapBatches)


cdef custatevecStatus_t _custatevecDistIndexBitSwapSchedulerGetParameters(custatevecHandle_t handle, custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler, const int32_t swapBatchIndex, const int32_t orgSubSVIndex, custatevecSVSwapParameters_t* parameters) except* nogil:
    global __custatevecDistIndexBitSwapSchedulerGetParameters
    _check_or_init_custatevec()
    if __custatevecDistIndexBitSwapSchedulerGetParameters == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecDistIndexBitSwapSchedulerGetParameters is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecDistIndexBitSwapSchedulerDescriptor_t, const int32_t, const int32_t, custatevecSVSwapParameters_t*) nogil>__custatevecDistIndexBitSwapSchedulerGetParameters)(
        handle, scheduler, swapBatchIndex, orgSubSVIndex, parameters)


cdef custatevecStatus_t _custatevecSVSwapWorkerCreate(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t* svSwapWorker, custatevecCommunicatorDescriptor_t communicator, void* orgSubSV, int32_t orgSubSVIndex, cudaEvent_t orgEvent, cudaDataType_t svDataType, cudaStream_t stream, size_t* extraWorkspaceSizeInBytes, size_t* minTransferWorkspaceSizeInBytes) except* nogil:
    global __custatevecSVSwapWorkerCreate
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t*, custatevecCommunicatorDescriptor_t, void*, int32_t, cudaEvent_t, cudaDataType_t, cudaStream_t, size_t*, size_t*) nogil>__custatevecSVSwapWorkerCreate)(
        handle, svSwapWorker, communicator, orgSubSV, orgSubSVIndex, orgEvent, svDataType, stream, extraWorkspaceSizeInBytes, minTransferWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSVSwapWorkerDestroy(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker) except* nogil:
    global __custatevecSVSwapWorkerDestroy
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t) nogil>__custatevecSVSwapWorkerDestroy)(
        handle, svSwapWorker)


cdef custatevecStatus_t _custatevecSVSwapWorkerSetExtraWorkspace(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecSVSwapWorkerSetExtraWorkspace
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerSetExtraWorkspace == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerSetExtraWorkspace is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t, void*, size_t) nogil>__custatevecSVSwapWorkerSetExtraWorkspace)(
        handle, svSwapWorker, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSVSwapWorkerSetTransferWorkspace(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void* transferWorkspace, size_t transferWorkspaceSizeInBytes) except* nogil:
    global __custatevecSVSwapWorkerSetTransferWorkspace
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerSetTransferWorkspace == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerSetTransferWorkspace is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t, void*, size_t) nogil>__custatevecSVSwapWorkerSetTransferWorkspace)(
        handle, svSwapWorker, transferWorkspace, transferWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecSVSwapWorkerSetSubSVsP2P(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, void** dstSubSVsP2P, const int32_t* dstSubSVIndicesP2P, cudaEvent_t* dstEvents, const uint32_t nDstSubSVsP2P) except* nogil:
    global __custatevecSVSwapWorkerSetSubSVsP2P
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerSetSubSVsP2P == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerSetSubSVsP2P is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t, void**, const int32_t*, cudaEvent_t*, const uint32_t) nogil>__custatevecSVSwapWorkerSetSubSVsP2P)(
        handle, svSwapWorker, dstSubSVsP2P, dstSubSVIndicesP2P, dstEvents, nDstSubSVsP2P)


cdef custatevecStatus_t _custatevecSVSwapWorkerSetParameters(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, const custatevecSVSwapParameters_t* parameters, int peer) except* nogil:
    global __custatevecSVSwapWorkerSetParameters
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerSetParameters == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerSetParameters is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t, const custatevecSVSwapParameters_t*, int) nogil>__custatevecSVSwapWorkerSetParameters)(
        handle, svSwapWorker, parameters, peer)


cdef custatevecStatus_t _custatevecSVSwapWorkerExecute(custatevecHandle_t handle, custatevecSVSwapWorkerDescriptor_t svSwapWorker, custatevecIndex_t begin, custatevecIndex_t end) except* nogil:
    global __custatevecSVSwapWorkerExecute
    _check_or_init_custatevec()
    if __custatevecSVSwapWorkerExecute == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSVSwapWorkerExecute is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSVSwapWorkerDescriptor_t, custatevecIndex_t, custatevecIndex_t) nogil>__custatevecSVSwapWorkerExecute)(
        handle, svSwapWorker, begin, end)


cdef custatevecStatus_t _custatevecInitializeStateVector(custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits, custatevecStateVectorType_t svType) except* nogil:
    global __custatevecInitializeStateVector
    _check_or_init_custatevec()
    if __custatevecInitializeStateVector == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecInitializeStateVector is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, custatevecStateVectorType_t) nogil>__custatevecInitializeStateVector)(
        handle, sv, svDataType, nIndexBits, svType)


cdef custatevecStatus_t _custatevecApplyMatrixBatchedGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, custatevecMatrixMapType_t mapType, const int32_t* matrixIndices, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nMatrices, const uint32_t nTargets, const uint32_t nControls, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecApplyMatrixBatchedGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecApplyMatrixBatchedGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyMatrixBatchedGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaDataType_t, const uint32_t, const uint32_t, const custatevecIndex_t, custatevecMatrixMapType_t, const int32_t*, const void*, cudaDataType_t, custatevecMatrixLayout_t, const int32_t, const uint32_t, const uint32_t, const uint32_t, custatevecComputeType_t, size_t*) nogil>__custatevecApplyMatrixBatchedGetWorkspaceSize)(
        handle, svDataType, nIndexBits, nSVs, svStride, mapType, matrixIndices, matrices, matrixDataType, layout, adjoint, nMatrices, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecApplyMatrixBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, custatevecIndex_t svStride, custatevecMatrixMapType_t mapType, const int32_t* matrixIndices, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const int32_t adjoint, const uint32_t nMatrices, const int32_t* targets, const uint32_t nTargets, const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecApplyMatrixBatched
    _check_or_init_custatevec()
    if __custatevecApplyMatrixBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecApplyMatrixBatched is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const uint32_t, custatevecIndex_t, custatevecMatrixMapType_t, const int32_t*, const void*, cudaDataType_t, custatevecMatrixLayout_t, const int32_t, const uint32_t, const int32_t*, const uint32_t, const int32_t*, const int32_t*, const uint32_t, custatevecComputeType_t, void*, size_t) nogil>__custatevecApplyMatrixBatched)(
        handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, mapType, matrixIndices, matrices, matrixDataType, layout, adjoint, nMatrices, targets, nTargets, controls, controlBitValues, nControls, computeType, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecAbs2SumArrayBatched(custatevecHandle_t handle, const void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, double* abs2sumArrays, const custatevecIndex_t abs2sumArrayStride, const int32_t* bitOrdering, const uint32_t bitOrderingLen, const custatevecIndex_t* maskBitStrings, const int32_t* maskOrdering, const uint32_t maskLen) except* nogil:
    global __custatevecAbs2SumArrayBatched
    _check_or_init_custatevec()
    if __custatevecAbs2SumArrayBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecAbs2SumArrayBatched is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, const uint32_t, const custatevecIndex_t, double*, const custatevecIndex_t, const int32_t*, const uint32_t, const custatevecIndex_t*, const int32_t*, const uint32_t) nogil>__custatevecAbs2SumArrayBatched)(
        handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, abs2sumArrays, abs2sumArrayStride, bitOrdering, bitOrderingLen, maskBitStrings, maskOrdering, maskLen)


cdef custatevecStatus_t _custatevecCollapseByBitStringBatchedGetWorkspaceSize(custatevecHandle_t handle, const uint32_t nSVs, const custatevecIndex_t* bitStrings, const double* norms, size_t* extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecCollapseByBitStringBatchedGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecCollapseByBitStringBatchedGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCollapseByBitStringBatchedGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const uint32_t, const custatevecIndex_t*, const double*, size_t*) nogil>__custatevecCollapseByBitStringBatchedGetWorkspaceSize)(
        handle, nSVs, bitStrings, norms, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecCollapseByBitStringBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, const custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* norms, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecCollapseByBitStringBatched
    _check_or_init_custatevec()
    if __custatevecCollapseByBitStringBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecCollapseByBitStringBatched is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const uint32_t, const custatevecIndex_t, const custatevecIndex_t*, const int32_t*, const uint32_t, const double*, void*, size_t) nogil>__custatevecCollapseByBitStringBatched)(
        handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, bitStrings, bitOrdering, bitStringLen, norms, extraWorkspace, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecMeasureBatched(custatevecHandle_t handle, void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, custatevecIndex_t* bitStrings, const int32_t* bitOrdering, const uint32_t bitStringLen, const double* randnums, custatevecCollapseOp_t collapse) except* nogil:
    global __custatevecMeasureBatched
    _check_or_init_custatevec()
    if __custatevecMeasureBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecMeasureBatched is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, void*, cudaDataType_t, const uint32_t, const uint32_t, const custatevecIndex_t, custatevecIndex_t*, const int32_t*, const uint32_t, const double*, custatevecCollapseOp_t) nogil>__custatevecMeasureBatched)(
        handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, bitStrings, bitOrdering, bitStringLen, randnums, collapse)


cdef custatevecStatus_t _custatevecSubSVMigratorCreate(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t* migrator, void* deviceSlots, cudaDataType_t svDataType, int nDeviceSlots, int nLocalIndexBits) except* nogil:
    global __custatevecSubSVMigratorCreate
    _check_or_init_custatevec()
    if __custatevecSubSVMigratorCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSubSVMigratorCreate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSubSVMigratorDescriptor_t*, void*, cudaDataType_t, int, int) nogil>__custatevecSubSVMigratorCreate)(
        handle, migrator, deviceSlots, svDataType, nDeviceSlots, nLocalIndexBits)


cdef custatevecStatus_t _custatevecSubSVMigratorDestroy(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t migrator) except* nogil:
    global __custatevecSubSVMigratorDestroy
    _check_or_init_custatevec()
    if __custatevecSubSVMigratorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSubSVMigratorDestroy is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSubSVMigratorDescriptor_t) nogil>__custatevecSubSVMigratorDestroy)(
        handle, migrator)


cdef custatevecStatus_t _custatevecSubSVMigratorMigrate(custatevecHandle_t handle, custatevecSubSVMigratorDescriptor_t migrator, int deviceSlotIndex, const void* srcSubSV, void* dstSubSV, custatevecIndex_t begin, custatevecIndex_t end) except* nogil:
    global __custatevecSubSVMigratorMigrate
    _check_or_init_custatevec()
    if __custatevecSubSVMigratorMigrate == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecSubSVMigratorMigrate is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, custatevecSubSVMigratorDescriptor_t, int, const void*, void*, custatevecIndex_t, custatevecIndex_t) nogil>__custatevecSubSVMigratorMigrate)(
        handle, migrator, deviceSlotIndex, srcSubSV, dstSubSV, begin, end)


cdef custatevecStatus_t _custatevecComputeExpectationBatchedGetWorkspaceSize(custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, const custatevecIndex_t svStride, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nMatrices, const uint32_t nBasisBits, custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecComputeExpectationBatchedGetWorkspaceSize
    _check_or_init_custatevec()
    if __custatevecComputeExpectationBatchedGetWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecComputeExpectationBatchedGetWorkspaceSize is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, cudaDataType_t, const uint32_t, const uint32_t, const custatevecIndex_t, const void*, cudaDataType_t, custatevecMatrixLayout_t, const uint32_t, const uint32_t, custatevecComputeType_t, size_t*) nogil>__custatevecComputeExpectationBatchedGetWorkspaceSize)(
        handle, svDataType, nIndexBits, nSVs, svStride, matrices, matrixDataType, layout, nMatrices, nBasisBits, computeType, extraWorkspaceSizeInBytes)


cdef custatevecStatus_t _custatevecComputeExpectationBatched(custatevecHandle_t handle, const void* batchedSv, cudaDataType_t svDataType, const uint32_t nIndexBits, const uint32_t nSVs, custatevecIndex_t svStride, double2* expectationValues, const void* matrices, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout, const uint32_t nMatrices, const int32_t* basisBits, const uint32_t nBasisBits, custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes) except* nogil:
    global __custatevecComputeExpectationBatched
    _check_or_init_custatevec()
    if __custatevecComputeExpectationBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function custatevecComputeExpectationBatched is not found")
    return (<custatevecStatus_t (*)(custatevecHandle_t, const void*, cudaDataType_t, const uint32_t, const uint32_t, custatevecIndex_t, double2*, const void*, cudaDataType_t, custatevecMatrixLayout_t, const uint32_t, const int32_t*, const uint32_t, custatevecComputeType_t, void*, size_t) nogil>__custatevecComputeExpectationBatched)(
        handle, batchedSv, svDataType, nIndexBits, nSVs, svStride, expectationValues, matrices, matrixDataType, layout, nMatrices, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
