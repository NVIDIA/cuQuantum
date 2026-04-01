# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 26.03.0, generator version 0.3.1.dev1332+g03874867a.d20260311. Do not modify it directly.

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
cdef bint __py_cudensitymat_init = False

cdef void* __cudensitymatGetVersion = NULL
cdef void* __cudensitymatCreate = NULL
cdef void* __cudensitymatDestroy = NULL
cdef void* __cudensitymatResetDistributedConfiguration = NULL
cdef void* __cudensitymatGetNumRanks = NULL
cdef void* __cudensitymatGetProcRank = NULL
cdef void* __cudensitymatResetRandomSeed = NULL
cdef void* __cudensitymatCreateState = NULL
cdef void* __cudensitymatCreateStateMPS = NULL
cdef void* __cudensitymatDestroyState = NULL
cdef void* __cudensitymatStateGetNumComponents = NULL
cdef void* __cudensitymatStateGetComponentStorageSize = NULL
cdef void* __cudensitymatStateAttachComponentStorage = NULL
cdef void* __cudensitymatStateGetComponentNumModes = NULL
cdef void* __cudensitymatStateGetComponentInfo = NULL
cdef void* __cudensitymatStateInitializeZero = NULL
cdef void* __cudensitymatStateComputeScaling = NULL
cdef void* __cudensitymatStateComputeNorm = NULL
cdef void* __cudensitymatStateComputeTrace = NULL
cdef void* __cudensitymatStateComputeAccumulation = NULL
cdef void* __cudensitymatStateComputeInnerProduct = NULL
cdef void* __cudensitymatCreateElementaryOperator = NULL
cdef void* __cudensitymatCreateElementaryOperatorBatch = NULL
cdef void* __cudensitymatDestroyElementaryOperator = NULL
cdef void* __cudensitymatCreateMatrixOperatorDenseLocal = NULL
cdef void* __cudensitymatCreateMatrixOperatorDenseLocalBatch = NULL
cdef void* __cudensitymatDestroyMatrixOperator = NULL
cdef void* __cudensitymatCreateMatrixProductOperator = NULL
cdef void* __cudensitymatDestroyMatrixProductOperator = NULL
cdef void* __cudensitymatCreateOperatorTerm = NULL
cdef void* __cudensitymatDestroyOperatorTerm = NULL
cdef void* __cudensitymatOperatorTermAppendElementaryProduct = NULL
cdef void* __cudensitymatOperatorTermAppendElementaryProductBatch = NULL
cdef void* __cudensitymatOperatorTermAppendMatrixProduct = NULL
cdef void* __cudensitymatOperatorTermAppendMatrixProductBatch = NULL
cdef void* __cudensitymatOperatorTermAppendMPOProduct = NULL
cdef void* __cudensitymatCreateOperator = NULL
cdef void* __cudensitymatDestroyOperator = NULL
cdef void* __cudensitymatOperatorAppendTerm = NULL
cdef void* __cudensitymatOperatorAppendTermBatch = NULL
cdef void* __cudensitymatAttachBatchedCoefficients = NULL
cdef void* __cudensitymatOperatorConfigureAction = NULL
cdef void* __cudensitymatOperatorPrepareAction = NULL
cdef void* __cudensitymatOperatorComputeAction = NULL
cdef void* __cudensitymatOperatorPrepareActionBackwardDiff = NULL
cdef void* __cudensitymatOperatorComputeActionBackwardDiff = NULL
cdef void* __cudensitymatCreateOperatorAction = NULL
cdef void* __cudensitymatDestroyOperatorAction = NULL
cdef void* __cudensitymatOperatorActionPrepare = NULL
cdef void* __cudensitymatOperatorActionCompute = NULL
cdef void* __cudensitymatCreateExpectation = NULL
cdef void* __cudensitymatDestroyExpectation = NULL
cdef void* __cudensitymatExpectationPrepare = NULL
cdef void* __cudensitymatExpectationCompute = NULL
cdef void* __cudensitymatCreateOperatorSpectrum = NULL
cdef void* __cudensitymatDestroyOperatorSpectrum = NULL
cdef void* __cudensitymatOperatorSpectrumConfigure = NULL
cdef void* __cudensitymatOperatorSpectrumPrepare = NULL
cdef void* __cudensitymatOperatorSpectrumCompute = NULL
cdef void* __cudensitymatCreateTimePropagationScopeSplitTDVPConfig = NULL
cdef void* __cudensitymatDestroyTimePropagationScopeSplitTDVPConfig = NULL
cdef void* __cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute = NULL
cdef void* __cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute = NULL
cdef void* __cudensitymatCreateTimePropagationApproachKrylovConfig = NULL
cdef void* __cudensitymatDestroyTimePropagationApproachKrylovConfig = NULL
cdef void* __cudensitymatTimePropagationApproachKrylovConfigSetAttribute = NULL
cdef void* __cudensitymatTimePropagationApproachKrylovConfigGetAttribute = NULL
cdef void* __cudensitymatCreateTimePropagation = NULL
cdef void* __cudensitymatDestroyTimePropagation = NULL
cdef void* __cudensitymatTimePropagationConfigure = NULL
cdef void* __cudensitymatTimePropagationPrepare = NULL
cdef void* __cudensitymatTimePropagationCompute = NULL
cdef void* __cudensitymatCreateWorkspace = NULL
cdef void* __cudensitymatDestroyWorkspace = NULL
cdef void* __cudensitymatWorkspaceGetMemorySize = NULL
cdef void* __cudensitymatWorkspaceSetMemory = NULL
cdef void* __cudensitymatWorkspaceGetMemory = NULL
cdef void* __cudensitymatElementaryOperatorAttachBuffer = NULL
cdef void* __cudensitymatMatrixOperatorDenseLocalAttachBuffer = NULL


cdef void* load_library() except* nogil:
    cdef void* handle
    handle = dlopen("libcudensitymat.so.0", RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        with gil:
            err_msg = dlerror()
            raise RuntimeError(f'Failed to dlopen libcudensitymat ({err_msg.decode()})')
    return handle


cdef int _check_or_init_cudensitymat() except -1 nogil:
    global __py_cudensitymat_init
    if __py_cudensitymat_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_cudensitymat_init:
            return 0
            
        # Load function
        global __cudensitymatGetVersion
        __cudensitymatGetVersion = dlsym(RTLD_DEFAULT, 'cudensitymatGetVersion')
        if __cudensitymatGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatGetVersion = dlsym(handle, 'cudensitymatGetVersion')

        global __cudensitymatCreate
        __cudensitymatCreate = dlsym(RTLD_DEFAULT, 'cudensitymatCreate')
        if __cudensitymatCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreate = dlsym(handle, 'cudensitymatCreate')

        global __cudensitymatDestroy
        __cudensitymatDestroy = dlsym(RTLD_DEFAULT, 'cudensitymatDestroy')
        if __cudensitymatDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroy = dlsym(handle, 'cudensitymatDestroy')

        global __cudensitymatResetDistributedConfiguration
        __cudensitymatResetDistributedConfiguration = dlsym(RTLD_DEFAULT, 'cudensitymatResetDistributedConfiguration')
        if __cudensitymatResetDistributedConfiguration == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatResetDistributedConfiguration = dlsym(handle, 'cudensitymatResetDistributedConfiguration')

        global __cudensitymatGetNumRanks
        __cudensitymatGetNumRanks = dlsym(RTLD_DEFAULT, 'cudensitymatGetNumRanks')
        if __cudensitymatGetNumRanks == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatGetNumRanks = dlsym(handle, 'cudensitymatGetNumRanks')

        global __cudensitymatGetProcRank
        __cudensitymatGetProcRank = dlsym(RTLD_DEFAULT, 'cudensitymatGetProcRank')
        if __cudensitymatGetProcRank == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatGetProcRank = dlsym(handle, 'cudensitymatGetProcRank')

        global __cudensitymatResetRandomSeed
        __cudensitymatResetRandomSeed = dlsym(RTLD_DEFAULT, 'cudensitymatResetRandomSeed')
        if __cudensitymatResetRandomSeed == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatResetRandomSeed = dlsym(handle, 'cudensitymatResetRandomSeed')

        global __cudensitymatCreateState
        __cudensitymatCreateState = dlsym(RTLD_DEFAULT, 'cudensitymatCreateState')
        if __cudensitymatCreateState == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateState = dlsym(handle, 'cudensitymatCreateState')

        global __cudensitymatCreateStateMPS
        __cudensitymatCreateStateMPS = dlsym(RTLD_DEFAULT, 'cudensitymatCreateStateMPS')
        if __cudensitymatCreateStateMPS == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateStateMPS = dlsym(handle, 'cudensitymatCreateStateMPS')

        global __cudensitymatDestroyState
        __cudensitymatDestroyState = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyState')
        if __cudensitymatDestroyState == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyState = dlsym(handle, 'cudensitymatDestroyState')

        global __cudensitymatStateGetNumComponents
        __cudensitymatStateGetNumComponents = dlsym(RTLD_DEFAULT, 'cudensitymatStateGetNumComponents')
        if __cudensitymatStateGetNumComponents == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatStateGetNumComponents = dlsym(handle, 'cudensitymatStateGetNumComponents')

        global __cudensitymatStateGetComponentStorageSize
        __cudensitymatStateGetComponentStorageSize = dlsym(RTLD_DEFAULT, 'cudensitymatStateGetComponentStorageSize')
        if __cudensitymatStateGetComponentStorageSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatStateGetComponentStorageSize = dlsym(handle, 'cudensitymatStateGetComponentStorageSize')

        global __cudensitymatStateAttachComponentStorage
        __cudensitymatStateAttachComponentStorage = dlsym(RTLD_DEFAULT, 'cudensitymatStateAttachComponentStorage')
        if __cudensitymatStateAttachComponentStorage == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatStateAttachComponentStorage = dlsym(handle, 'cudensitymatStateAttachComponentStorage')

        global __cudensitymatStateGetComponentNumModes
        __cudensitymatStateGetComponentNumModes = dlsym(RTLD_DEFAULT, 'cudensitymatStateGetComponentNumModes')
        if __cudensitymatStateGetComponentNumModes == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatStateGetComponentNumModes = dlsym(handle, 'cudensitymatStateGetComponentNumModes')

        global __cudensitymatStateGetComponentInfo
        __cudensitymatStateGetComponentInfo = dlsym(RTLD_DEFAULT, 'cudensitymatStateGetComponentInfo')
        if __cudensitymatStateGetComponentInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatStateGetComponentInfo = dlsym(handle, 'cudensitymatStateGetComponentInfo')

        global __cudensitymatStateInitializeZero
        __cudensitymatStateInitializeZero = dlsym(RTLD_DEFAULT, 'cudensitymatStateInitializeZero')
        if __cudensitymatStateInitializeZero == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatStateInitializeZero = dlsym(handle, 'cudensitymatStateInitializeZero')

        global __cudensitymatStateComputeScaling
        __cudensitymatStateComputeScaling = dlsym(RTLD_DEFAULT, 'cudensitymatStateComputeScaling')
        if __cudensitymatStateComputeScaling == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatStateComputeScaling = dlsym(handle, 'cudensitymatStateComputeScaling')

        global __cudensitymatStateComputeNorm
        __cudensitymatStateComputeNorm = dlsym(RTLD_DEFAULT, 'cudensitymatStateComputeNorm')
        if __cudensitymatStateComputeNorm == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatStateComputeNorm = dlsym(handle, 'cudensitymatStateComputeNorm')

        global __cudensitymatStateComputeTrace
        __cudensitymatStateComputeTrace = dlsym(RTLD_DEFAULT, 'cudensitymatStateComputeTrace')
        if __cudensitymatStateComputeTrace == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatStateComputeTrace = dlsym(handle, 'cudensitymatStateComputeTrace')

        global __cudensitymatStateComputeAccumulation
        __cudensitymatStateComputeAccumulation = dlsym(RTLD_DEFAULT, 'cudensitymatStateComputeAccumulation')
        if __cudensitymatStateComputeAccumulation == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatStateComputeAccumulation = dlsym(handle, 'cudensitymatStateComputeAccumulation')

        global __cudensitymatStateComputeInnerProduct
        __cudensitymatStateComputeInnerProduct = dlsym(RTLD_DEFAULT, 'cudensitymatStateComputeInnerProduct')
        if __cudensitymatStateComputeInnerProduct == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatStateComputeInnerProduct = dlsym(handle, 'cudensitymatStateComputeInnerProduct')

        global __cudensitymatCreateElementaryOperator
        __cudensitymatCreateElementaryOperator = dlsym(RTLD_DEFAULT, 'cudensitymatCreateElementaryOperator')
        if __cudensitymatCreateElementaryOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateElementaryOperator = dlsym(handle, 'cudensitymatCreateElementaryOperator')

        global __cudensitymatCreateElementaryOperatorBatch
        __cudensitymatCreateElementaryOperatorBatch = dlsym(RTLD_DEFAULT, 'cudensitymatCreateElementaryOperatorBatch')
        if __cudensitymatCreateElementaryOperatorBatch == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateElementaryOperatorBatch = dlsym(handle, 'cudensitymatCreateElementaryOperatorBatch')

        global __cudensitymatDestroyElementaryOperator
        __cudensitymatDestroyElementaryOperator = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyElementaryOperator')
        if __cudensitymatDestroyElementaryOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyElementaryOperator = dlsym(handle, 'cudensitymatDestroyElementaryOperator')

        global __cudensitymatCreateMatrixOperatorDenseLocal
        __cudensitymatCreateMatrixOperatorDenseLocal = dlsym(RTLD_DEFAULT, 'cudensitymatCreateMatrixOperatorDenseLocal')
        if __cudensitymatCreateMatrixOperatorDenseLocal == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateMatrixOperatorDenseLocal = dlsym(handle, 'cudensitymatCreateMatrixOperatorDenseLocal')

        global __cudensitymatCreateMatrixOperatorDenseLocalBatch
        __cudensitymatCreateMatrixOperatorDenseLocalBatch = dlsym(RTLD_DEFAULT, 'cudensitymatCreateMatrixOperatorDenseLocalBatch')
        if __cudensitymatCreateMatrixOperatorDenseLocalBatch == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateMatrixOperatorDenseLocalBatch = dlsym(handle, 'cudensitymatCreateMatrixOperatorDenseLocalBatch')

        global __cudensitymatDestroyMatrixOperator
        __cudensitymatDestroyMatrixOperator = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyMatrixOperator')
        if __cudensitymatDestroyMatrixOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyMatrixOperator = dlsym(handle, 'cudensitymatDestroyMatrixOperator')

        global __cudensitymatCreateMatrixProductOperator
        __cudensitymatCreateMatrixProductOperator = dlsym(RTLD_DEFAULT, 'cudensitymatCreateMatrixProductOperator')
        if __cudensitymatCreateMatrixProductOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateMatrixProductOperator = dlsym(handle, 'cudensitymatCreateMatrixProductOperator')

        global __cudensitymatDestroyMatrixProductOperator
        __cudensitymatDestroyMatrixProductOperator = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyMatrixProductOperator')
        if __cudensitymatDestroyMatrixProductOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyMatrixProductOperator = dlsym(handle, 'cudensitymatDestroyMatrixProductOperator')

        global __cudensitymatCreateOperatorTerm
        __cudensitymatCreateOperatorTerm = dlsym(RTLD_DEFAULT, 'cudensitymatCreateOperatorTerm')
        if __cudensitymatCreateOperatorTerm == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateOperatorTerm = dlsym(handle, 'cudensitymatCreateOperatorTerm')

        global __cudensitymatDestroyOperatorTerm
        __cudensitymatDestroyOperatorTerm = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyOperatorTerm')
        if __cudensitymatDestroyOperatorTerm == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyOperatorTerm = dlsym(handle, 'cudensitymatDestroyOperatorTerm')

        global __cudensitymatOperatorTermAppendElementaryProduct
        __cudensitymatOperatorTermAppendElementaryProduct = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorTermAppendElementaryProduct')
        if __cudensitymatOperatorTermAppendElementaryProduct == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorTermAppendElementaryProduct = dlsym(handle, 'cudensitymatOperatorTermAppendElementaryProduct')

        global __cudensitymatOperatorTermAppendElementaryProductBatch
        __cudensitymatOperatorTermAppendElementaryProductBatch = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorTermAppendElementaryProductBatch')
        if __cudensitymatOperatorTermAppendElementaryProductBatch == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorTermAppendElementaryProductBatch = dlsym(handle, 'cudensitymatOperatorTermAppendElementaryProductBatch')

        global __cudensitymatOperatorTermAppendMatrixProduct
        __cudensitymatOperatorTermAppendMatrixProduct = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorTermAppendMatrixProduct')
        if __cudensitymatOperatorTermAppendMatrixProduct == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorTermAppendMatrixProduct = dlsym(handle, 'cudensitymatOperatorTermAppendMatrixProduct')

        global __cudensitymatOperatorTermAppendMatrixProductBatch
        __cudensitymatOperatorTermAppendMatrixProductBatch = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorTermAppendMatrixProductBatch')
        if __cudensitymatOperatorTermAppendMatrixProductBatch == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorTermAppendMatrixProductBatch = dlsym(handle, 'cudensitymatOperatorTermAppendMatrixProductBatch')

        global __cudensitymatOperatorTermAppendMPOProduct
        __cudensitymatOperatorTermAppendMPOProduct = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorTermAppendMPOProduct')
        if __cudensitymatOperatorTermAppendMPOProduct == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorTermAppendMPOProduct = dlsym(handle, 'cudensitymatOperatorTermAppendMPOProduct')

        global __cudensitymatCreateOperator
        __cudensitymatCreateOperator = dlsym(RTLD_DEFAULT, 'cudensitymatCreateOperator')
        if __cudensitymatCreateOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateOperator = dlsym(handle, 'cudensitymatCreateOperator')

        global __cudensitymatDestroyOperator
        __cudensitymatDestroyOperator = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyOperator')
        if __cudensitymatDestroyOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyOperator = dlsym(handle, 'cudensitymatDestroyOperator')

        global __cudensitymatOperatorAppendTerm
        __cudensitymatOperatorAppendTerm = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorAppendTerm')
        if __cudensitymatOperatorAppendTerm == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorAppendTerm = dlsym(handle, 'cudensitymatOperatorAppendTerm')

        global __cudensitymatOperatorAppendTermBatch
        __cudensitymatOperatorAppendTermBatch = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorAppendTermBatch')
        if __cudensitymatOperatorAppendTermBatch == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorAppendTermBatch = dlsym(handle, 'cudensitymatOperatorAppendTermBatch')

        global __cudensitymatAttachBatchedCoefficients
        __cudensitymatAttachBatchedCoefficients = dlsym(RTLD_DEFAULT, 'cudensitymatAttachBatchedCoefficients')
        if __cudensitymatAttachBatchedCoefficients == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatAttachBatchedCoefficients = dlsym(handle, 'cudensitymatAttachBatchedCoefficients')

        global __cudensitymatOperatorConfigureAction
        __cudensitymatOperatorConfigureAction = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorConfigureAction')
        if __cudensitymatOperatorConfigureAction == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorConfigureAction = dlsym(handle, 'cudensitymatOperatorConfigureAction')

        global __cudensitymatOperatorPrepareAction
        __cudensitymatOperatorPrepareAction = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorPrepareAction')
        if __cudensitymatOperatorPrepareAction == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorPrepareAction = dlsym(handle, 'cudensitymatOperatorPrepareAction')

        global __cudensitymatOperatorComputeAction
        __cudensitymatOperatorComputeAction = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorComputeAction')
        if __cudensitymatOperatorComputeAction == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorComputeAction = dlsym(handle, 'cudensitymatOperatorComputeAction')

        global __cudensitymatOperatorPrepareActionBackwardDiff
        __cudensitymatOperatorPrepareActionBackwardDiff = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorPrepareActionBackwardDiff')
        if __cudensitymatOperatorPrepareActionBackwardDiff == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorPrepareActionBackwardDiff = dlsym(handle, 'cudensitymatOperatorPrepareActionBackwardDiff')

        global __cudensitymatOperatorComputeActionBackwardDiff
        __cudensitymatOperatorComputeActionBackwardDiff = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorComputeActionBackwardDiff')
        if __cudensitymatOperatorComputeActionBackwardDiff == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorComputeActionBackwardDiff = dlsym(handle, 'cudensitymatOperatorComputeActionBackwardDiff')

        global __cudensitymatCreateOperatorAction
        __cudensitymatCreateOperatorAction = dlsym(RTLD_DEFAULT, 'cudensitymatCreateOperatorAction')
        if __cudensitymatCreateOperatorAction == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateOperatorAction = dlsym(handle, 'cudensitymatCreateOperatorAction')

        global __cudensitymatDestroyOperatorAction
        __cudensitymatDestroyOperatorAction = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyOperatorAction')
        if __cudensitymatDestroyOperatorAction == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyOperatorAction = dlsym(handle, 'cudensitymatDestroyOperatorAction')

        global __cudensitymatOperatorActionPrepare
        __cudensitymatOperatorActionPrepare = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorActionPrepare')
        if __cudensitymatOperatorActionPrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorActionPrepare = dlsym(handle, 'cudensitymatOperatorActionPrepare')

        global __cudensitymatOperatorActionCompute
        __cudensitymatOperatorActionCompute = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorActionCompute')
        if __cudensitymatOperatorActionCompute == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorActionCompute = dlsym(handle, 'cudensitymatOperatorActionCompute')

        global __cudensitymatCreateExpectation
        __cudensitymatCreateExpectation = dlsym(RTLD_DEFAULT, 'cudensitymatCreateExpectation')
        if __cudensitymatCreateExpectation == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateExpectation = dlsym(handle, 'cudensitymatCreateExpectation')

        global __cudensitymatDestroyExpectation
        __cudensitymatDestroyExpectation = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyExpectation')
        if __cudensitymatDestroyExpectation == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyExpectation = dlsym(handle, 'cudensitymatDestroyExpectation')

        global __cudensitymatExpectationPrepare
        __cudensitymatExpectationPrepare = dlsym(RTLD_DEFAULT, 'cudensitymatExpectationPrepare')
        if __cudensitymatExpectationPrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatExpectationPrepare = dlsym(handle, 'cudensitymatExpectationPrepare')

        global __cudensitymatExpectationCompute
        __cudensitymatExpectationCompute = dlsym(RTLD_DEFAULT, 'cudensitymatExpectationCompute')
        if __cudensitymatExpectationCompute == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatExpectationCompute = dlsym(handle, 'cudensitymatExpectationCompute')

        global __cudensitymatCreateOperatorSpectrum
        __cudensitymatCreateOperatorSpectrum = dlsym(RTLD_DEFAULT, 'cudensitymatCreateOperatorSpectrum')
        if __cudensitymatCreateOperatorSpectrum == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateOperatorSpectrum = dlsym(handle, 'cudensitymatCreateOperatorSpectrum')

        global __cudensitymatDestroyOperatorSpectrum
        __cudensitymatDestroyOperatorSpectrum = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyOperatorSpectrum')
        if __cudensitymatDestroyOperatorSpectrum == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyOperatorSpectrum = dlsym(handle, 'cudensitymatDestroyOperatorSpectrum')

        global __cudensitymatOperatorSpectrumConfigure
        __cudensitymatOperatorSpectrumConfigure = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorSpectrumConfigure')
        if __cudensitymatOperatorSpectrumConfigure == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorSpectrumConfigure = dlsym(handle, 'cudensitymatOperatorSpectrumConfigure')

        global __cudensitymatOperatorSpectrumPrepare
        __cudensitymatOperatorSpectrumPrepare = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorSpectrumPrepare')
        if __cudensitymatOperatorSpectrumPrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorSpectrumPrepare = dlsym(handle, 'cudensitymatOperatorSpectrumPrepare')

        global __cudensitymatOperatorSpectrumCompute
        __cudensitymatOperatorSpectrumCompute = dlsym(RTLD_DEFAULT, 'cudensitymatOperatorSpectrumCompute')
        if __cudensitymatOperatorSpectrumCompute == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatOperatorSpectrumCompute = dlsym(handle, 'cudensitymatOperatorSpectrumCompute')

        global __cudensitymatCreateTimePropagationScopeSplitTDVPConfig
        __cudensitymatCreateTimePropagationScopeSplitTDVPConfig = dlsym(RTLD_DEFAULT, 'cudensitymatCreateTimePropagationScopeSplitTDVPConfig')
        if __cudensitymatCreateTimePropagationScopeSplitTDVPConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateTimePropagationScopeSplitTDVPConfig = dlsym(handle, 'cudensitymatCreateTimePropagationScopeSplitTDVPConfig')

        global __cudensitymatDestroyTimePropagationScopeSplitTDVPConfig
        __cudensitymatDestroyTimePropagationScopeSplitTDVPConfig = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyTimePropagationScopeSplitTDVPConfig')
        if __cudensitymatDestroyTimePropagationScopeSplitTDVPConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyTimePropagationScopeSplitTDVPConfig = dlsym(handle, 'cudensitymatDestroyTimePropagationScopeSplitTDVPConfig')

        global __cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute
        __cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute = dlsym(RTLD_DEFAULT, 'cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute')
        if __cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute = dlsym(handle, 'cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute')

        global __cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute
        __cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute = dlsym(RTLD_DEFAULT, 'cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute')
        if __cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute = dlsym(handle, 'cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute')

        global __cudensitymatCreateTimePropagationApproachKrylovConfig
        __cudensitymatCreateTimePropagationApproachKrylovConfig = dlsym(RTLD_DEFAULT, 'cudensitymatCreateTimePropagationApproachKrylovConfig')
        if __cudensitymatCreateTimePropagationApproachKrylovConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateTimePropagationApproachKrylovConfig = dlsym(handle, 'cudensitymatCreateTimePropagationApproachKrylovConfig')

        global __cudensitymatDestroyTimePropagationApproachKrylovConfig
        __cudensitymatDestroyTimePropagationApproachKrylovConfig = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyTimePropagationApproachKrylovConfig')
        if __cudensitymatDestroyTimePropagationApproachKrylovConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyTimePropagationApproachKrylovConfig = dlsym(handle, 'cudensitymatDestroyTimePropagationApproachKrylovConfig')

        global __cudensitymatTimePropagationApproachKrylovConfigSetAttribute
        __cudensitymatTimePropagationApproachKrylovConfigSetAttribute = dlsym(RTLD_DEFAULT, 'cudensitymatTimePropagationApproachKrylovConfigSetAttribute')
        if __cudensitymatTimePropagationApproachKrylovConfigSetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatTimePropagationApproachKrylovConfigSetAttribute = dlsym(handle, 'cudensitymatTimePropagationApproachKrylovConfigSetAttribute')

        global __cudensitymatTimePropagationApproachKrylovConfigGetAttribute
        __cudensitymatTimePropagationApproachKrylovConfigGetAttribute = dlsym(RTLD_DEFAULT, 'cudensitymatTimePropagationApproachKrylovConfigGetAttribute')
        if __cudensitymatTimePropagationApproachKrylovConfigGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatTimePropagationApproachKrylovConfigGetAttribute = dlsym(handle, 'cudensitymatTimePropagationApproachKrylovConfigGetAttribute')

        global __cudensitymatCreateTimePropagation
        __cudensitymatCreateTimePropagation = dlsym(RTLD_DEFAULT, 'cudensitymatCreateTimePropagation')
        if __cudensitymatCreateTimePropagation == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateTimePropagation = dlsym(handle, 'cudensitymatCreateTimePropagation')

        global __cudensitymatDestroyTimePropagation
        __cudensitymatDestroyTimePropagation = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyTimePropagation')
        if __cudensitymatDestroyTimePropagation == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyTimePropagation = dlsym(handle, 'cudensitymatDestroyTimePropagation')

        global __cudensitymatTimePropagationConfigure
        __cudensitymatTimePropagationConfigure = dlsym(RTLD_DEFAULT, 'cudensitymatTimePropagationConfigure')
        if __cudensitymatTimePropagationConfigure == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatTimePropagationConfigure = dlsym(handle, 'cudensitymatTimePropagationConfigure')

        global __cudensitymatTimePropagationPrepare
        __cudensitymatTimePropagationPrepare = dlsym(RTLD_DEFAULT, 'cudensitymatTimePropagationPrepare')
        if __cudensitymatTimePropagationPrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatTimePropagationPrepare = dlsym(handle, 'cudensitymatTimePropagationPrepare')

        global __cudensitymatTimePropagationCompute
        __cudensitymatTimePropagationCompute = dlsym(RTLD_DEFAULT, 'cudensitymatTimePropagationCompute')
        if __cudensitymatTimePropagationCompute == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatTimePropagationCompute = dlsym(handle, 'cudensitymatTimePropagationCompute')

        global __cudensitymatCreateWorkspace
        __cudensitymatCreateWorkspace = dlsym(RTLD_DEFAULT, 'cudensitymatCreateWorkspace')
        if __cudensitymatCreateWorkspace == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatCreateWorkspace = dlsym(handle, 'cudensitymatCreateWorkspace')

        global __cudensitymatDestroyWorkspace
        __cudensitymatDestroyWorkspace = dlsym(RTLD_DEFAULT, 'cudensitymatDestroyWorkspace')
        if __cudensitymatDestroyWorkspace == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatDestroyWorkspace = dlsym(handle, 'cudensitymatDestroyWorkspace')

        global __cudensitymatWorkspaceGetMemorySize
        __cudensitymatWorkspaceGetMemorySize = dlsym(RTLD_DEFAULT, 'cudensitymatWorkspaceGetMemorySize')
        if __cudensitymatWorkspaceGetMemorySize == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatWorkspaceGetMemorySize = dlsym(handle, 'cudensitymatWorkspaceGetMemorySize')

        global __cudensitymatWorkspaceSetMemory
        __cudensitymatWorkspaceSetMemory = dlsym(RTLD_DEFAULT, 'cudensitymatWorkspaceSetMemory')
        if __cudensitymatWorkspaceSetMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatWorkspaceSetMemory = dlsym(handle, 'cudensitymatWorkspaceSetMemory')

        global __cudensitymatWorkspaceGetMemory
        __cudensitymatWorkspaceGetMemory = dlsym(RTLD_DEFAULT, 'cudensitymatWorkspaceGetMemory')
        if __cudensitymatWorkspaceGetMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatWorkspaceGetMemory = dlsym(handle, 'cudensitymatWorkspaceGetMemory')

        global __cudensitymatElementaryOperatorAttachBuffer
        __cudensitymatElementaryOperatorAttachBuffer = dlsym(RTLD_DEFAULT, 'cudensitymatElementaryOperatorAttachBuffer')
        if __cudensitymatElementaryOperatorAttachBuffer == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatElementaryOperatorAttachBuffer = dlsym(handle, 'cudensitymatElementaryOperatorAttachBuffer')

        global __cudensitymatMatrixOperatorDenseLocalAttachBuffer
        __cudensitymatMatrixOperatorDenseLocalAttachBuffer = dlsym(RTLD_DEFAULT, 'cudensitymatMatrixOperatorDenseLocalAttachBuffer')
        if __cudensitymatMatrixOperatorDenseLocalAttachBuffer == NULL:
            if handle == NULL:
                handle = load_library()
            __cudensitymatMatrixOperatorDenseLocalAttachBuffer = dlsym(handle, 'cudensitymatMatrixOperatorDenseLocalAttachBuffer')
        __py_cudensitymat_init = True
        return 0


cpdef dict _inspect_function_pointers():
    _check_or_init_cudensitymat()
    cdef dict data = {}

    global __cudensitymatGetVersion
    data["__cudensitymatGetVersion"] = <intptr_t>__cudensitymatGetVersion

    global __cudensitymatCreate
    data["__cudensitymatCreate"] = <intptr_t>__cudensitymatCreate

    global __cudensitymatDestroy
    data["__cudensitymatDestroy"] = <intptr_t>__cudensitymatDestroy

    global __cudensitymatResetDistributedConfiguration
    data["__cudensitymatResetDistributedConfiguration"] = <intptr_t>__cudensitymatResetDistributedConfiguration

    global __cudensitymatGetNumRanks
    data["__cudensitymatGetNumRanks"] = <intptr_t>__cudensitymatGetNumRanks

    global __cudensitymatGetProcRank
    data["__cudensitymatGetProcRank"] = <intptr_t>__cudensitymatGetProcRank

    global __cudensitymatResetRandomSeed
    data["__cudensitymatResetRandomSeed"] = <intptr_t>__cudensitymatResetRandomSeed

    global __cudensitymatCreateState
    data["__cudensitymatCreateState"] = <intptr_t>__cudensitymatCreateState

    global __cudensitymatCreateStateMPS
    data["__cudensitymatCreateStateMPS"] = <intptr_t>__cudensitymatCreateStateMPS

    global __cudensitymatDestroyState
    data["__cudensitymatDestroyState"] = <intptr_t>__cudensitymatDestroyState

    global __cudensitymatStateGetNumComponents
    data["__cudensitymatStateGetNumComponents"] = <intptr_t>__cudensitymatStateGetNumComponents

    global __cudensitymatStateGetComponentStorageSize
    data["__cudensitymatStateGetComponentStorageSize"] = <intptr_t>__cudensitymatStateGetComponentStorageSize

    global __cudensitymatStateAttachComponentStorage
    data["__cudensitymatStateAttachComponentStorage"] = <intptr_t>__cudensitymatStateAttachComponentStorage

    global __cudensitymatStateGetComponentNumModes
    data["__cudensitymatStateGetComponentNumModes"] = <intptr_t>__cudensitymatStateGetComponentNumModes

    global __cudensitymatStateGetComponentInfo
    data["__cudensitymatStateGetComponentInfo"] = <intptr_t>__cudensitymatStateGetComponentInfo

    global __cudensitymatStateInitializeZero
    data["__cudensitymatStateInitializeZero"] = <intptr_t>__cudensitymatStateInitializeZero

    global __cudensitymatStateComputeScaling
    data["__cudensitymatStateComputeScaling"] = <intptr_t>__cudensitymatStateComputeScaling

    global __cudensitymatStateComputeNorm
    data["__cudensitymatStateComputeNorm"] = <intptr_t>__cudensitymatStateComputeNorm

    global __cudensitymatStateComputeTrace
    data["__cudensitymatStateComputeTrace"] = <intptr_t>__cudensitymatStateComputeTrace

    global __cudensitymatStateComputeAccumulation
    data["__cudensitymatStateComputeAccumulation"] = <intptr_t>__cudensitymatStateComputeAccumulation

    global __cudensitymatStateComputeInnerProduct
    data["__cudensitymatStateComputeInnerProduct"] = <intptr_t>__cudensitymatStateComputeInnerProduct

    global __cudensitymatCreateElementaryOperator
    data["__cudensitymatCreateElementaryOperator"] = <intptr_t>__cudensitymatCreateElementaryOperator

    global __cudensitymatCreateElementaryOperatorBatch
    data["__cudensitymatCreateElementaryOperatorBatch"] = <intptr_t>__cudensitymatCreateElementaryOperatorBatch

    global __cudensitymatDestroyElementaryOperator
    data["__cudensitymatDestroyElementaryOperator"] = <intptr_t>__cudensitymatDestroyElementaryOperator

    global __cudensitymatCreateMatrixOperatorDenseLocal
    data["__cudensitymatCreateMatrixOperatorDenseLocal"] = <intptr_t>__cudensitymatCreateMatrixOperatorDenseLocal

    global __cudensitymatCreateMatrixOperatorDenseLocalBatch
    data["__cudensitymatCreateMatrixOperatorDenseLocalBatch"] = <intptr_t>__cudensitymatCreateMatrixOperatorDenseLocalBatch

    global __cudensitymatDestroyMatrixOperator
    data["__cudensitymatDestroyMatrixOperator"] = <intptr_t>__cudensitymatDestroyMatrixOperator

    global __cudensitymatCreateMatrixProductOperator
    data["__cudensitymatCreateMatrixProductOperator"] = <intptr_t>__cudensitymatCreateMatrixProductOperator

    global __cudensitymatDestroyMatrixProductOperator
    data["__cudensitymatDestroyMatrixProductOperator"] = <intptr_t>__cudensitymatDestroyMatrixProductOperator

    global __cudensitymatCreateOperatorTerm
    data["__cudensitymatCreateOperatorTerm"] = <intptr_t>__cudensitymatCreateOperatorTerm

    global __cudensitymatDestroyOperatorTerm
    data["__cudensitymatDestroyOperatorTerm"] = <intptr_t>__cudensitymatDestroyOperatorTerm

    global __cudensitymatOperatorTermAppendElementaryProduct
    data["__cudensitymatOperatorTermAppendElementaryProduct"] = <intptr_t>__cudensitymatOperatorTermAppendElementaryProduct

    global __cudensitymatOperatorTermAppendElementaryProductBatch
    data["__cudensitymatOperatorTermAppendElementaryProductBatch"] = <intptr_t>__cudensitymatOperatorTermAppendElementaryProductBatch

    global __cudensitymatOperatorTermAppendMatrixProduct
    data["__cudensitymatOperatorTermAppendMatrixProduct"] = <intptr_t>__cudensitymatOperatorTermAppendMatrixProduct

    global __cudensitymatOperatorTermAppendMatrixProductBatch
    data["__cudensitymatOperatorTermAppendMatrixProductBatch"] = <intptr_t>__cudensitymatOperatorTermAppendMatrixProductBatch

    global __cudensitymatOperatorTermAppendMPOProduct
    data["__cudensitymatOperatorTermAppendMPOProduct"] = <intptr_t>__cudensitymatOperatorTermAppendMPOProduct

    global __cudensitymatCreateOperator
    data["__cudensitymatCreateOperator"] = <intptr_t>__cudensitymatCreateOperator

    global __cudensitymatDestroyOperator
    data["__cudensitymatDestroyOperator"] = <intptr_t>__cudensitymatDestroyOperator

    global __cudensitymatOperatorAppendTerm
    data["__cudensitymatOperatorAppendTerm"] = <intptr_t>__cudensitymatOperatorAppendTerm

    global __cudensitymatOperatorAppendTermBatch
    data["__cudensitymatOperatorAppendTermBatch"] = <intptr_t>__cudensitymatOperatorAppendTermBatch

    global __cudensitymatAttachBatchedCoefficients
    data["__cudensitymatAttachBatchedCoefficients"] = <intptr_t>__cudensitymatAttachBatchedCoefficients

    global __cudensitymatOperatorConfigureAction
    data["__cudensitymatOperatorConfigureAction"] = <intptr_t>__cudensitymatOperatorConfigureAction

    global __cudensitymatOperatorPrepareAction
    data["__cudensitymatOperatorPrepareAction"] = <intptr_t>__cudensitymatOperatorPrepareAction

    global __cudensitymatOperatorComputeAction
    data["__cudensitymatOperatorComputeAction"] = <intptr_t>__cudensitymatOperatorComputeAction

    global __cudensitymatOperatorPrepareActionBackwardDiff
    data["__cudensitymatOperatorPrepareActionBackwardDiff"] = <intptr_t>__cudensitymatOperatorPrepareActionBackwardDiff

    global __cudensitymatOperatorComputeActionBackwardDiff
    data["__cudensitymatOperatorComputeActionBackwardDiff"] = <intptr_t>__cudensitymatOperatorComputeActionBackwardDiff

    global __cudensitymatCreateOperatorAction
    data["__cudensitymatCreateOperatorAction"] = <intptr_t>__cudensitymatCreateOperatorAction

    global __cudensitymatDestroyOperatorAction
    data["__cudensitymatDestroyOperatorAction"] = <intptr_t>__cudensitymatDestroyOperatorAction

    global __cudensitymatOperatorActionPrepare
    data["__cudensitymatOperatorActionPrepare"] = <intptr_t>__cudensitymatOperatorActionPrepare

    global __cudensitymatOperatorActionCompute
    data["__cudensitymatOperatorActionCompute"] = <intptr_t>__cudensitymatOperatorActionCompute

    global __cudensitymatCreateExpectation
    data["__cudensitymatCreateExpectation"] = <intptr_t>__cudensitymatCreateExpectation

    global __cudensitymatDestroyExpectation
    data["__cudensitymatDestroyExpectation"] = <intptr_t>__cudensitymatDestroyExpectation

    global __cudensitymatExpectationPrepare
    data["__cudensitymatExpectationPrepare"] = <intptr_t>__cudensitymatExpectationPrepare

    global __cudensitymatExpectationCompute
    data["__cudensitymatExpectationCompute"] = <intptr_t>__cudensitymatExpectationCompute

    global __cudensitymatCreateOperatorSpectrum
    data["__cudensitymatCreateOperatorSpectrum"] = <intptr_t>__cudensitymatCreateOperatorSpectrum

    global __cudensitymatDestroyOperatorSpectrum
    data["__cudensitymatDestroyOperatorSpectrum"] = <intptr_t>__cudensitymatDestroyOperatorSpectrum

    global __cudensitymatOperatorSpectrumConfigure
    data["__cudensitymatOperatorSpectrumConfigure"] = <intptr_t>__cudensitymatOperatorSpectrumConfigure

    global __cudensitymatOperatorSpectrumPrepare
    data["__cudensitymatOperatorSpectrumPrepare"] = <intptr_t>__cudensitymatOperatorSpectrumPrepare

    global __cudensitymatOperatorSpectrumCompute
    data["__cudensitymatOperatorSpectrumCompute"] = <intptr_t>__cudensitymatOperatorSpectrumCompute

    global __cudensitymatCreateTimePropagationScopeSplitTDVPConfig
    data["__cudensitymatCreateTimePropagationScopeSplitTDVPConfig"] = <intptr_t>__cudensitymatCreateTimePropagationScopeSplitTDVPConfig

    global __cudensitymatDestroyTimePropagationScopeSplitTDVPConfig
    data["__cudensitymatDestroyTimePropagationScopeSplitTDVPConfig"] = <intptr_t>__cudensitymatDestroyTimePropagationScopeSplitTDVPConfig

    global __cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute
    data["__cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute"] = <intptr_t>__cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute

    global __cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute
    data["__cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute"] = <intptr_t>__cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute

    global __cudensitymatCreateTimePropagationApproachKrylovConfig
    data["__cudensitymatCreateTimePropagationApproachKrylovConfig"] = <intptr_t>__cudensitymatCreateTimePropagationApproachKrylovConfig

    global __cudensitymatDestroyTimePropagationApproachKrylovConfig
    data["__cudensitymatDestroyTimePropagationApproachKrylovConfig"] = <intptr_t>__cudensitymatDestroyTimePropagationApproachKrylovConfig

    global __cudensitymatTimePropagationApproachKrylovConfigSetAttribute
    data["__cudensitymatTimePropagationApproachKrylovConfigSetAttribute"] = <intptr_t>__cudensitymatTimePropagationApproachKrylovConfigSetAttribute

    global __cudensitymatTimePropagationApproachKrylovConfigGetAttribute
    data["__cudensitymatTimePropagationApproachKrylovConfigGetAttribute"] = <intptr_t>__cudensitymatTimePropagationApproachKrylovConfigGetAttribute

    global __cudensitymatCreateTimePropagation
    data["__cudensitymatCreateTimePropagation"] = <intptr_t>__cudensitymatCreateTimePropagation

    global __cudensitymatDestroyTimePropagation
    data["__cudensitymatDestroyTimePropagation"] = <intptr_t>__cudensitymatDestroyTimePropagation

    global __cudensitymatTimePropagationConfigure
    data["__cudensitymatTimePropagationConfigure"] = <intptr_t>__cudensitymatTimePropagationConfigure

    global __cudensitymatTimePropagationPrepare
    data["__cudensitymatTimePropagationPrepare"] = <intptr_t>__cudensitymatTimePropagationPrepare

    global __cudensitymatTimePropagationCompute
    data["__cudensitymatTimePropagationCompute"] = <intptr_t>__cudensitymatTimePropagationCompute

    global __cudensitymatCreateWorkspace
    data["__cudensitymatCreateWorkspace"] = <intptr_t>__cudensitymatCreateWorkspace

    global __cudensitymatDestroyWorkspace
    data["__cudensitymatDestroyWorkspace"] = <intptr_t>__cudensitymatDestroyWorkspace

    global __cudensitymatWorkspaceGetMemorySize
    data["__cudensitymatWorkspaceGetMemorySize"] = <intptr_t>__cudensitymatWorkspaceGetMemorySize

    global __cudensitymatWorkspaceSetMemory
    data["__cudensitymatWorkspaceSetMemory"] = <intptr_t>__cudensitymatWorkspaceSetMemory

    global __cudensitymatWorkspaceGetMemory
    data["__cudensitymatWorkspaceGetMemory"] = <intptr_t>__cudensitymatWorkspaceGetMemory

    global __cudensitymatElementaryOperatorAttachBuffer
    data["__cudensitymatElementaryOperatorAttachBuffer"] = <intptr_t>__cudensitymatElementaryOperatorAttachBuffer

    global __cudensitymatMatrixOperatorDenseLocalAttachBuffer
    data["__cudensitymatMatrixOperatorDenseLocalAttachBuffer"] = <intptr_t>__cudensitymatMatrixOperatorDenseLocalAttachBuffer

    return data


###############################################################################
# Wrapper functions
###############################################################################

cdef size_t _cudensitymatGetVersion() except?0 nogil:
    global __cudensitymatGetVersion
    _check_or_init_cudensitymat()
    if __cudensitymatGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatGetVersion is not found")
    return (<size_t (*)() noexcept nogil>__cudensitymatGetVersion)(
        )


cdef cudensitymatStatus_t _cudensitymatCreate(cudensitymatHandle_t* handle) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreate
    _check_or_init_cudensitymat()
    if __cudensitymatCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreate is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatHandle_t*) noexcept nogil>__cudensitymatCreate)(
        handle)


cdef cudensitymatStatus_t _cudensitymatDestroy(cudensitymatHandle_t handle) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroy
    _check_or_init_cudensitymat()
    if __cudensitymatDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroy is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatHandle_t) noexcept nogil>__cudensitymatDestroy)(
        handle)


cdef cudensitymatStatus_t _cudensitymatResetDistributedConfiguration(cudensitymatHandle_t handle, cudensitymatDistributedProvider_t provider, const void* commPtr, size_t commSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatResetDistributedConfiguration
    _check_or_init_cudensitymat()
    if __cudensitymatResetDistributedConfiguration == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatResetDistributedConfiguration is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatHandle_t, cudensitymatDistributedProvider_t, const void*, size_t) noexcept nogil>__cudensitymatResetDistributedConfiguration)(
        handle, provider, commPtr, commSize)


cdef cudensitymatStatus_t _cudensitymatGetNumRanks(const cudensitymatHandle_t handle, int32_t* numRanks) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatGetNumRanks
    _check_or_init_cudensitymat()
    if __cudensitymatGetNumRanks == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatGetNumRanks is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, int32_t*) noexcept nogil>__cudensitymatGetNumRanks)(
        handle, numRanks)


cdef cudensitymatStatus_t _cudensitymatGetProcRank(const cudensitymatHandle_t handle, int32_t* procRank) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatGetProcRank
    _check_or_init_cudensitymat()
    if __cudensitymatGetProcRank == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatGetProcRank is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, int32_t*) noexcept nogil>__cudensitymatGetProcRank)(
        handle, procRank)


cdef cudensitymatStatus_t _cudensitymatResetRandomSeed(cudensitymatHandle_t handle, int32_t randomSeed) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatResetRandomSeed
    _check_or_init_cudensitymat()
    if __cudensitymatResetRandomSeed == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatResetRandomSeed is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatHandle_t, int32_t) noexcept nogil>__cudensitymatResetRandomSeed)(
        handle, randomSeed)


cdef cudensitymatStatus_t _cudensitymatCreateState(const cudensitymatHandle_t handle, cudensitymatStatePurity_t purity, int32_t numSpaceModes, const int64_t spaceModeExtents[], int64_t batchSize, cudaDataType_t dataType, cudensitymatState_t* state) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateState
    _check_or_init_cudensitymat()
    if __cudensitymatCreateState == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateState is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatStatePurity_t, int32_t, const int64_t*, int64_t, cudaDataType_t, cudensitymatState_t*) noexcept nogil>__cudensitymatCreateState)(
        handle, purity, numSpaceModes, spaceModeExtents, batchSize, dataType, state)


cdef cudensitymatStatus_t _cudensitymatCreateStateMPS(const cudensitymatHandle_t handle, cudensitymatStatePurity_t purity, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatBoundaryCondition_t boundaryCondition, const int64_t bondExtents[], cudaDataType_t dataType, int64_t batchSize, cudensitymatState_t* state) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateStateMPS
    _check_or_init_cudensitymat()
    if __cudensitymatCreateStateMPS == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateStateMPS is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatStatePurity_t, int32_t, const int64_t*, cudensitymatBoundaryCondition_t, const int64_t*, cudaDataType_t, int64_t, cudensitymatState_t*) noexcept nogil>__cudensitymatCreateStateMPS)(
        handle, purity, numSpaceModes, spaceModeExtents, boundaryCondition, bondExtents, dataType, batchSize, state)


cdef cudensitymatStatus_t _cudensitymatDestroyState(cudensitymatState_t state) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyState
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyState == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyState is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatState_t) noexcept nogil>__cudensitymatDestroyState)(
        state)


cdef cudensitymatStatus_t _cudensitymatStateGetNumComponents(const cudensitymatHandle_t handle, const cudensitymatState_t state, int32_t* numStateComponents) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatStateGetNumComponents
    _check_or_init_cudensitymat()
    if __cudensitymatStateGetNumComponents == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatStateGetNumComponents is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, const cudensitymatState_t, int32_t*) noexcept nogil>__cudensitymatStateGetNumComponents)(
        handle, state, numStateComponents)


cdef cudensitymatStatus_t _cudensitymatStateGetComponentStorageSize(const cudensitymatHandle_t handle, const cudensitymatState_t state, int32_t numStateComponents, size_t componentBufferSize[]) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatStateGetComponentStorageSize
    _check_or_init_cudensitymat()
    if __cudensitymatStateGetComponentStorageSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatStateGetComponentStorageSize is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, const cudensitymatState_t, int32_t, size_t*) noexcept nogil>__cudensitymatStateGetComponentStorageSize)(
        handle, state, numStateComponents, componentBufferSize)


cdef cudensitymatStatus_t _cudensitymatStateAttachComponentStorage(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t numStateComponents, void* componentBuffer[], const size_t componentBufferSize[]) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatStateAttachComponentStorage
    _check_or_init_cudensitymat()
    if __cudensitymatStateAttachComponentStorage == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatStateAttachComponentStorage is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatState_t, int32_t, void**, const size_t*) noexcept nogil>__cudensitymatStateAttachComponentStorage)(
        handle, state, numStateComponents, componentBuffer, componentBufferSize)


cdef cudensitymatStatus_t _cudensitymatStateGetComponentNumModes(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t stateComponentLocalId, int32_t* stateComponentGlobalId, int32_t* stateComponentNumModes, int32_t* batchModeLocation) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatStateGetComponentNumModes
    _check_or_init_cudensitymat()
    if __cudensitymatStateGetComponentNumModes == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatStateGetComponentNumModes is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatState_t, int32_t, int32_t*, int32_t*, int32_t*) noexcept nogil>__cudensitymatStateGetComponentNumModes)(
        handle, state, stateComponentLocalId, stateComponentGlobalId, stateComponentNumModes, batchModeLocation)


cdef cudensitymatStatus_t _cudensitymatStateGetComponentInfo(const cudensitymatHandle_t handle, cudensitymatState_t state, int32_t stateComponentLocalId, int32_t* stateComponentGlobalId, int32_t* stateComponentNumModes, int64_t stateComponentModeExtents[], int64_t stateComponentModeOffsets[]) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatStateGetComponentInfo
    _check_or_init_cudensitymat()
    if __cudensitymatStateGetComponentInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatStateGetComponentInfo is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatState_t, int32_t, int32_t*, int32_t*, int64_t*, int64_t*) noexcept nogil>__cudensitymatStateGetComponentInfo)(
        handle, state, stateComponentLocalId, stateComponentGlobalId, stateComponentNumModes, stateComponentModeExtents, stateComponentModeOffsets)


cdef cudensitymatStatus_t _cudensitymatStateInitializeZero(const cudensitymatHandle_t handle, cudensitymatState_t state, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatStateInitializeZero
    _check_or_init_cudensitymat()
    if __cudensitymatStateInitializeZero == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatStateInitializeZero is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatState_t, cudaStream_t) noexcept nogil>__cudensitymatStateInitializeZero)(
        handle, state, stream)


cdef cudensitymatStatus_t _cudensitymatStateComputeScaling(const cudensitymatHandle_t handle, cudensitymatState_t state, const void* scalingFactors, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatStateComputeScaling
    _check_or_init_cudensitymat()
    if __cudensitymatStateComputeScaling == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatStateComputeScaling is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatState_t, const void*, cudaStream_t) noexcept nogil>__cudensitymatStateComputeScaling)(
        handle, state, scalingFactors, stream)


cdef cudensitymatStatus_t _cudensitymatStateComputeNorm(const cudensitymatHandle_t handle, const cudensitymatState_t state, void* norm, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatStateComputeNorm
    _check_or_init_cudensitymat()
    if __cudensitymatStateComputeNorm == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatStateComputeNorm is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, const cudensitymatState_t, void*, cudaStream_t) noexcept nogil>__cudensitymatStateComputeNorm)(
        handle, state, norm, stream)


cdef cudensitymatStatus_t _cudensitymatStateComputeTrace(const cudensitymatHandle_t handle, const cudensitymatState_t state, void* trace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatStateComputeTrace
    _check_or_init_cudensitymat()
    if __cudensitymatStateComputeTrace == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatStateComputeTrace is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, const cudensitymatState_t, void*, cudaStream_t) noexcept nogil>__cudensitymatStateComputeTrace)(
        handle, state, trace, stream)


cdef cudensitymatStatus_t _cudensitymatStateComputeAccumulation(const cudensitymatHandle_t handle, const cudensitymatState_t stateIn, cudensitymatState_t stateOut, const void* scalingFactors, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatStateComputeAccumulation
    _check_or_init_cudensitymat()
    if __cudensitymatStateComputeAccumulation == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatStateComputeAccumulation is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, const cudensitymatState_t, cudensitymatState_t, const void*, cudaStream_t) noexcept nogil>__cudensitymatStateComputeAccumulation)(
        handle, stateIn, stateOut, scalingFactors, stream)


cdef cudensitymatStatus_t _cudensitymatStateComputeInnerProduct(const cudensitymatHandle_t handle, const cudensitymatState_t stateLeft, const cudensitymatState_t stateRight, void* innerProduct, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatStateComputeInnerProduct
    _check_or_init_cudensitymat()
    if __cudensitymatStateComputeInnerProduct == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatStateComputeInnerProduct is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, const cudensitymatState_t, const cudensitymatState_t, void*, cudaStream_t) noexcept nogil>__cudensitymatStateComputeInnerProduct)(
        handle, stateLeft, stateRight, innerProduct, stream)


cdef cudensitymatStatus_t _cudensitymatCreateElementaryOperator(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatElementaryOperatorSparsity_t sparsity, int32_t numDiagonals, const int32_t diagonalOffsets[], cudaDataType_t dataType, void* tensorData, cudensitymatWrappedTensorCallback_t tensorCallback, cudensitymatWrappedTensorGradientCallback_t tensorGradientCallback, cudensitymatElementaryOperator_t* elemOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateElementaryOperator
    _check_or_init_cudensitymat()
    if __cudensitymatCreateElementaryOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateElementaryOperator is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, int32_t, const int64_t*, cudensitymatElementaryOperatorSparsity_t, int32_t, const int32_t*, cudaDataType_t, void*, cudensitymatWrappedTensorCallback_t, cudensitymatWrappedTensorGradientCallback_t, cudensitymatElementaryOperator_t*) noexcept nogil>__cudensitymatCreateElementaryOperator)(
        handle, numSpaceModes, spaceModeExtents, sparsity, numDiagonals, diagonalOffsets, dataType, tensorData, tensorCallback, tensorGradientCallback, elemOperator)


cdef cudensitymatStatus_t _cudensitymatCreateElementaryOperatorBatch(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], int64_t batchSize, cudensitymatElementaryOperatorSparsity_t sparsity, int32_t numDiagonals, const int32_t diagonalOffsets[], cudaDataType_t dataType, void* tensorData, cudensitymatWrappedTensorCallback_t tensorCallback, cudensitymatWrappedTensorGradientCallback_t tensorGradientCallback, cudensitymatElementaryOperator_t* elemOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateElementaryOperatorBatch
    _check_or_init_cudensitymat()
    if __cudensitymatCreateElementaryOperatorBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateElementaryOperatorBatch is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, int32_t, const int64_t*, int64_t, cudensitymatElementaryOperatorSparsity_t, int32_t, const int32_t*, cudaDataType_t, void*, cudensitymatWrappedTensorCallback_t, cudensitymatWrappedTensorGradientCallback_t, cudensitymatElementaryOperator_t*) noexcept nogil>__cudensitymatCreateElementaryOperatorBatch)(
        handle, numSpaceModes, spaceModeExtents, batchSize, sparsity, numDiagonals, diagonalOffsets, dataType, tensorData, tensorCallback, tensorGradientCallback, elemOperator)


cdef cudensitymatStatus_t _cudensitymatDestroyElementaryOperator(cudensitymatElementaryOperator_t elemOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyElementaryOperator
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyElementaryOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyElementaryOperator is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatElementaryOperator_t) noexcept nogil>__cudensitymatDestroyElementaryOperator)(
        elemOperator)


cdef cudensitymatStatus_t _cudensitymatCreateMatrixOperatorDenseLocal(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudaDataType_t dataType, void* matrixData, cudensitymatWrappedTensorCallback_t matrixCallback, cudensitymatWrappedTensorGradientCallback_t matrixGradientCallback, cudensitymatMatrixOperator_t* matrixOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateMatrixOperatorDenseLocal
    _check_or_init_cudensitymat()
    if __cudensitymatCreateMatrixOperatorDenseLocal == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateMatrixOperatorDenseLocal is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, int32_t, const int64_t*, cudaDataType_t, void*, cudensitymatWrappedTensorCallback_t, cudensitymatWrappedTensorGradientCallback_t, cudensitymatMatrixOperator_t*) noexcept nogil>__cudensitymatCreateMatrixOperatorDenseLocal)(
        handle, numSpaceModes, spaceModeExtents, dataType, matrixData, matrixCallback, matrixGradientCallback, matrixOperator)


cdef cudensitymatStatus_t _cudensitymatCreateMatrixOperatorDenseLocalBatch(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], int64_t batchSize, cudaDataType_t dataType, void* matrixData, cudensitymatWrappedTensorCallback_t matrixCallback, cudensitymatWrappedTensorGradientCallback_t matrixGradientCallback, cudensitymatMatrixOperator_t* matrixOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateMatrixOperatorDenseLocalBatch
    _check_or_init_cudensitymat()
    if __cudensitymatCreateMatrixOperatorDenseLocalBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateMatrixOperatorDenseLocalBatch is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, int32_t, const int64_t*, int64_t, cudaDataType_t, void*, cudensitymatWrappedTensorCallback_t, cudensitymatWrappedTensorGradientCallback_t, cudensitymatMatrixOperator_t*) noexcept nogil>__cudensitymatCreateMatrixOperatorDenseLocalBatch)(
        handle, numSpaceModes, spaceModeExtents, batchSize, dataType, matrixData, matrixCallback, matrixGradientCallback, matrixOperator)


cdef cudensitymatStatus_t _cudensitymatDestroyMatrixOperator(cudensitymatMatrixOperator_t matrixOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyMatrixOperator
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyMatrixOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyMatrixOperator is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatMatrixOperator_t) noexcept nogil>__cudensitymatDestroyMatrixOperator)(
        matrixOperator)


cdef cudensitymatStatus_t _cudensitymatCreateMatrixProductOperator(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatBoundaryCondition_t boundaryCondition, const int64_t bondExtents[], cudaDataType_t dataType, void* tensorData[], cudensitymatWrappedTensorCallback_t tensorCallbacks[], cudensitymatWrappedTensorGradientCallback_t tensorGradientCallbacks[], cudensitymatMatrixProductOperator_t* matrixProductOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateMatrixProductOperator
    _check_or_init_cudensitymat()
    if __cudensitymatCreateMatrixProductOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateMatrixProductOperator is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, int32_t, const int64_t*, cudensitymatBoundaryCondition_t, const int64_t*, cudaDataType_t, void**, cudensitymatWrappedTensorCallback_t*, cudensitymatWrappedTensorGradientCallback_t*, cudensitymatMatrixProductOperator_t*) noexcept nogil>__cudensitymatCreateMatrixProductOperator)(
        handle, numSpaceModes, spaceModeExtents, boundaryCondition, bondExtents, dataType, tensorData, tensorCallbacks, tensorGradientCallbacks, matrixProductOperator)


cdef cudensitymatStatus_t _cudensitymatDestroyMatrixProductOperator(cudensitymatMatrixProductOperator_t matrixProductOperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyMatrixProductOperator
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyMatrixProductOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyMatrixProductOperator is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatMatrixProductOperator_t) noexcept nogil>__cudensitymatDestroyMatrixProductOperator)(
        matrixProductOperator)


cdef cudensitymatStatus_t _cudensitymatCreateOperatorTerm(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatOperatorTerm_t* operatorTerm) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateOperatorTerm
    _check_or_init_cudensitymat()
    if __cudensitymatCreateOperatorTerm == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateOperatorTerm is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, int32_t, const int64_t*, cudensitymatOperatorTerm_t*) noexcept nogil>__cudensitymatCreateOperatorTerm)(
        handle, numSpaceModes, spaceModeExtents, operatorTerm)


cdef cudensitymatStatus_t _cudensitymatDestroyOperatorTerm(cudensitymatOperatorTerm_t operatorTerm) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyOperatorTerm
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyOperatorTerm == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyOperatorTerm is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatOperatorTerm_t) noexcept nogil>__cudensitymatDestroyOperatorTerm)(
        operatorTerm)


cdef cudensitymatStatus_t _cudensitymatOperatorTermAppendElementaryProduct(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numElemOperators, const cudensitymatElementaryOperator_t elemOperators[], const int32_t stateModesActedOn[], const int32_t modeActionDuality[], cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorTermAppendElementaryProduct
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorTermAppendElementaryProduct == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorTermAppendElementaryProduct is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperatorTerm_t, int32_t, const cudensitymatElementaryOperator_t*, const int32_t*, const int32_t*, cuDoubleComplex, cudensitymatWrappedScalarCallback_t, cudensitymatWrappedScalarGradientCallback_t) noexcept nogil>__cudensitymatOperatorTermAppendElementaryProduct)(
        handle, operatorTerm, numElemOperators, elemOperators, stateModesActedOn, modeActionDuality, coefficient, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t _cudensitymatOperatorTermAppendElementaryProductBatch(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numElemOperators, const cudensitymatElementaryOperator_t elemOperators[], const int32_t stateModesActedOn[], const int32_t modeActionDuality[], int64_t batchSize, const cuDoubleComplex staticCoefficients[], cuDoubleComplex totalCoefficients[], cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorTermAppendElementaryProductBatch
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorTermAppendElementaryProductBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorTermAppendElementaryProductBatch is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperatorTerm_t, int32_t, const cudensitymatElementaryOperator_t*, const int32_t*, const int32_t*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, cudensitymatWrappedScalarCallback_t, cudensitymatWrappedScalarGradientCallback_t) noexcept nogil>__cudensitymatOperatorTermAppendElementaryProductBatch)(
        handle, operatorTerm, numElemOperators, elemOperators, stateModesActedOn, modeActionDuality, batchSize, staticCoefficients, totalCoefficients, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t _cudensitymatOperatorTermAppendMatrixProduct(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numMatrixOperators, const cudensitymatMatrixOperator_t matrixOperators[], const int32_t matrixConjugation[], const int32_t actionDuality[], cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorTermAppendMatrixProduct
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorTermAppendMatrixProduct == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorTermAppendMatrixProduct is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperatorTerm_t, int32_t, const cudensitymatMatrixOperator_t*, const int32_t*, const int32_t*, cuDoubleComplex, cudensitymatWrappedScalarCallback_t, cudensitymatWrappedScalarGradientCallback_t) noexcept nogil>__cudensitymatOperatorTermAppendMatrixProduct)(
        handle, operatorTerm, numMatrixOperators, matrixOperators, matrixConjugation, actionDuality, coefficient, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t _cudensitymatOperatorTermAppendMatrixProductBatch(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numMatrixOperators, const cudensitymatMatrixOperator_t matrixOperators[], const int32_t matrixConjugation[], const int32_t actionDuality[], int64_t batchSize, const cuDoubleComplex staticCoefficients[], cuDoubleComplex totalCoefficients[], cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorTermAppendMatrixProductBatch
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorTermAppendMatrixProductBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorTermAppendMatrixProductBatch is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperatorTerm_t, int32_t, const cudensitymatMatrixOperator_t*, const int32_t*, const int32_t*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, cudensitymatWrappedScalarCallback_t, cudensitymatWrappedScalarGradientCallback_t) noexcept nogil>__cudensitymatOperatorTermAppendMatrixProductBatch)(
        handle, operatorTerm, numMatrixOperators, matrixOperators, matrixConjugation, actionDuality, batchSize, staticCoefficients, totalCoefficients, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t _cudensitymatOperatorTermAppendMPOProduct(const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm, int32_t numMPOOperators, const cudensitymatMatrixProductOperator_t mpoOperators[], const int32_t mpoConjugation[], const int32_t stateModesActedOn[], const int32_t modeActionDuality[], cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorTermAppendMPOProduct
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorTermAppendMPOProduct == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorTermAppendMPOProduct is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperatorTerm_t, int32_t, const cudensitymatMatrixProductOperator_t*, const int32_t*, const int32_t*, const int32_t*, cuDoubleComplex, cudensitymatWrappedScalarCallback_t, cudensitymatWrappedScalarGradientCallback_t) noexcept nogil>__cudensitymatOperatorTermAppendMPOProduct)(
        handle, operatorTerm, numMPOOperators, mpoOperators, mpoConjugation, stateModesActedOn, modeActionDuality, coefficient, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t _cudensitymatCreateOperator(const cudensitymatHandle_t handle, int32_t numSpaceModes, const int64_t spaceModeExtents[], cudensitymatOperator_t* superoperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateOperator
    _check_or_init_cudensitymat()
    if __cudensitymatCreateOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateOperator is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, int32_t, const int64_t*, cudensitymatOperator_t*) noexcept nogil>__cudensitymatCreateOperator)(
        handle, numSpaceModes, spaceModeExtents, superoperator)


cdef cudensitymatStatus_t _cudensitymatDestroyOperator(cudensitymatOperator_t superoperator) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyOperator
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyOperator is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatOperator_t) noexcept nogil>__cudensitymatDestroyOperator)(
        superoperator)


cdef cudensitymatStatus_t _cudensitymatOperatorAppendTerm(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, cudensitymatOperatorTerm_t operatorTerm, int32_t duality, cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorAppendTerm
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorAppendTerm == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorAppendTerm is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperator_t, cudensitymatOperatorTerm_t, int32_t, cuDoubleComplex, cudensitymatWrappedScalarCallback_t, cudensitymatWrappedScalarGradientCallback_t) noexcept nogil>__cudensitymatOperatorAppendTerm)(
        handle, superoperator, operatorTerm, duality, coefficient, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t _cudensitymatOperatorAppendTermBatch(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, cudensitymatOperatorTerm_t operatorTerm, int32_t duality, int64_t batchSize, const cuDoubleComplex staticCoefficients[], cuDoubleComplex totalCoefficients[], cudensitymatWrappedScalarCallback_t coefficientCallback, cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorAppendTermBatch
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorAppendTermBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorAppendTermBatch is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperator_t, cudensitymatOperatorTerm_t, int32_t, int64_t, const cuDoubleComplex*, cuDoubleComplex*, cudensitymatWrappedScalarCallback_t, cudensitymatWrappedScalarGradientCallback_t) noexcept nogil>__cudensitymatOperatorAppendTermBatch)(
        handle, superoperator, operatorTerm, duality, batchSize, staticCoefficients, totalCoefficients, coefficientCallback, coefficientGradientCallback)


cdef cudensitymatStatus_t _cudensitymatAttachBatchedCoefficients(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, int32_t numOperatorTermBatchedCoeffs, void* operatorTermBatchedCoeffsTmp[], void* operatorTermBatchedCoeffs[], int32_t numOperatorProductBatchedCoeffs, void* operatorProductBatchedCoeffsTmp[], void* operatorProductBatchedCoeffs[]) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatAttachBatchedCoefficients
    _check_or_init_cudensitymat()
    if __cudensitymatAttachBatchedCoefficients == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatAttachBatchedCoefficients is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperator_t, int32_t, void**, void**, int32_t, void**, void**) noexcept nogil>__cudensitymatAttachBatchedCoefficients)(
        handle, superoperator, numOperatorTermBatchedCoeffs, operatorTermBatchedCoeffsTmp, operatorTermBatchedCoeffs, numOperatorProductBatchedCoeffs, operatorProductBatchedCoeffsTmp, operatorProductBatchedCoeffs)


cdef cudensitymatStatus_t _cudensitymatOperatorConfigureAction(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, const cudensitymatState_t stateIn, const cudensitymatState_t stateOut, const void* attributeValue, size_t attributeSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorConfigureAction
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorConfigureAction == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorConfigureAction is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperator_t, const cudensitymatState_t, const cudensitymatState_t, const void*, size_t) noexcept nogil>__cudensitymatOperatorConfigureAction)(
        handle, superoperator, stateIn, stateOut, attributeValue, attributeSize)


cdef cudensitymatStatus_t _cudensitymatOperatorPrepareAction(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, const cudensitymatState_t stateIn, const cudensitymatState_t stateOut, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorPrepareAction
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorPrepareAction == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorPrepareAction is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperator_t, const cudensitymatState_t, const cudensitymatState_t, cudensitymatComputeType_t, size_t, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatOperatorPrepareAction)(
        handle, superoperator, stateIn, stateOut, computeType, workspaceSizeLimit, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatOperatorComputeAction(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t stateIn, cudensitymatState_t stateOut, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorComputeAction
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorComputeAction == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorComputeAction is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperator_t, double, int64_t, int32_t, const double*, const cudensitymatState_t, cudensitymatState_t, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatOperatorComputeAction)(
        handle, superoperator, time, batchSize, numParams, params, stateIn, stateOut, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatOperatorPrepareActionBackwardDiff(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, const cudensitymatState_t stateIn, const cudensitymatState_t stateOutAdj, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorPrepareActionBackwardDiff
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorPrepareActionBackwardDiff == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorPrepareActionBackwardDiff is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperator_t, const cudensitymatState_t, const cudensitymatState_t, cudensitymatComputeType_t, size_t, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatOperatorPrepareActionBackwardDiff)(
        handle, superoperator, stateIn, stateOutAdj, computeType, workspaceSizeLimit, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatOperatorComputeActionBackwardDiff(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t stateIn, const cudensitymatState_t stateOutAdj, cudensitymatState_t stateInAdj, double* paramsGrad, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorComputeActionBackwardDiff
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorComputeActionBackwardDiff == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorComputeActionBackwardDiff is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperator_t, double, int64_t, int32_t, const double*, const cudensitymatState_t, const cudensitymatState_t, cudensitymatState_t, double*, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatOperatorComputeActionBackwardDiff)(
        handle, superoperator, time, batchSize, numParams, params, stateIn, stateOutAdj, stateInAdj, paramsGrad, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatCreateOperatorAction(const cudensitymatHandle_t handle, int32_t numOperators, cudensitymatOperator_t operators[], cudensitymatOperatorAction_t* operatorAction) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateOperatorAction
    _check_or_init_cudensitymat()
    if __cudensitymatCreateOperatorAction == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateOperatorAction is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, int32_t, cudensitymatOperator_t*, cudensitymatOperatorAction_t*) noexcept nogil>__cudensitymatCreateOperatorAction)(
        handle, numOperators, operators, operatorAction)


cdef cudensitymatStatus_t _cudensitymatDestroyOperatorAction(cudensitymatOperatorAction_t operatorAction) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyOperatorAction
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyOperatorAction == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyOperatorAction is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatOperatorAction_t) noexcept nogil>__cudensitymatDestroyOperatorAction)(
        operatorAction)


cdef cudensitymatStatus_t _cudensitymatOperatorActionPrepare(const cudensitymatHandle_t handle, cudensitymatOperatorAction_t operatorAction, const cudensitymatState_t stateIn[], const cudensitymatState_t stateOut, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorActionPrepare
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorActionPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorActionPrepare is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperatorAction_t, const cudensitymatState_t*, const cudensitymatState_t, cudensitymatComputeType_t, size_t, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatOperatorActionPrepare)(
        handle, operatorAction, stateIn, stateOut, computeType, workspaceSizeLimit, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatOperatorActionCompute(const cudensitymatHandle_t handle, cudensitymatOperatorAction_t operatorAction, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t stateIn[], cudensitymatState_t stateOut, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorActionCompute
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorActionCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorActionCompute is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperatorAction_t, double, int64_t, int32_t, const double*, const cudensitymatState_t*, cudensitymatState_t, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatOperatorActionCompute)(
        handle, operatorAction, time, batchSize, numParams, params, stateIn, stateOut, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatCreateExpectation(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, cudensitymatExpectation_t* expectation) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateExpectation
    _check_or_init_cudensitymat()
    if __cudensitymatCreateExpectation == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateExpectation is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperator_t, cudensitymatExpectation_t*) noexcept nogil>__cudensitymatCreateExpectation)(
        handle, superoperator, expectation)


cdef cudensitymatStatus_t _cudensitymatDestroyExpectation(cudensitymatExpectation_t expectation) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyExpectation
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyExpectation == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyExpectation is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatExpectation_t) noexcept nogil>__cudensitymatDestroyExpectation)(
        expectation)


cdef cudensitymatStatus_t _cudensitymatExpectationPrepare(const cudensitymatHandle_t handle, cudensitymatExpectation_t expectation, const cudensitymatState_t state, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatExpectationPrepare
    _check_or_init_cudensitymat()
    if __cudensitymatExpectationPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatExpectationPrepare is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatExpectation_t, const cudensitymatState_t, cudensitymatComputeType_t, size_t, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatExpectationPrepare)(
        handle, expectation, state, computeType, workspaceSizeLimit, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatExpectationCompute(const cudensitymatHandle_t handle, cudensitymatExpectation_t expectation, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t state, void* expectationValue, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatExpectationCompute
    _check_or_init_cudensitymat()
    if __cudensitymatExpectationCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatExpectationCompute is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatExpectation_t, double, int64_t, int32_t, const double*, const cudensitymatState_t, void*, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatExpectationCompute)(
        handle, expectation, time, batchSize, numParams, params, state, expectationValue, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatCreateOperatorSpectrum(const cudensitymatHandle_t handle, const cudensitymatOperator_t superoperator, int32_t isHermitian, cudensitymatOperatorSpectrumKind_t spectrumKind, cudensitymatOperatorSpectrum_t* spectrum) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateOperatorSpectrum
    _check_or_init_cudensitymat()
    if __cudensitymatCreateOperatorSpectrum == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateOperatorSpectrum is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, const cudensitymatOperator_t, int32_t, cudensitymatOperatorSpectrumKind_t, cudensitymatOperatorSpectrum_t*) noexcept nogil>__cudensitymatCreateOperatorSpectrum)(
        handle, superoperator, isHermitian, spectrumKind, spectrum)


cdef cudensitymatStatus_t _cudensitymatDestroyOperatorSpectrum(cudensitymatOperatorSpectrum_t spectrum) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyOperatorSpectrum
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyOperatorSpectrum == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyOperatorSpectrum is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatOperatorSpectrum_t) noexcept nogil>__cudensitymatDestroyOperatorSpectrum)(
        spectrum)


cdef cudensitymatStatus_t _cudensitymatOperatorSpectrumConfigure(const cudensitymatHandle_t handle, cudensitymatOperatorSpectrum_t spectrum, cudensitymatOperatorSpectrumConfig_t attribute, const void* attributeValue, size_t attributeValueSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorSpectrumConfigure
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorSpectrumConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorSpectrumConfigure is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperatorSpectrum_t, cudensitymatOperatorSpectrumConfig_t, const void*, size_t) noexcept nogil>__cudensitymatOperatorSpectrumConfigure)(
        handle, spectrum, attribute, attributeValue, attributeValueSize)


cdef cudensitymatStatus_t _cudensitymatOperatorSpectrumPrepare(const cudensitymatHandle_t handle, cudensitymatOperatorSpectrum_t spectrum, int32_t maxEigenStates, const cudensitymatState_t state, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorSpectrumPrepare
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorSpectrumPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorSpectrumPrepare is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperatorSpectrum_t, int32_t, const cudensitymatState_t, cudensitymatComputeType_t, size_t, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatOperatorSpectrumPrepare)(
        handle, spectrum, maxEigenStates, state, computeType, workspaceSizeLimit, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatOperatorSpectrumCompute(const cudensitymatHandle_t handle, cudensitymatOperatorSpectrum_t spectrum, double time, int64_t batchSize, int32_t numParams, const double* params, int32_t numEigenStates, cudensitymatState_t eigenstates[], void* eigenvalues, double* tolerances, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatOperatorSpectrumCompute
    _check_or_init_cudensitymat()
    if __cudensitymatOperatorSpectrumCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatOperatorSpectrumCompute is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperatorSpectrum_t, double, int64_t, int32_t, const double*, int32_t, cudensitymatState_t*, void*, double*, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatOperatorSpectrumCompute)(
        handle, spectrum, time, batchSize, numParams, params, numEigenStates, eigenstates, eigenvalues, tolerances, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatCreateTimePropagationScopeSplitTDVPConfig(const cudensitymatHandle_t handle, cudensitymatTimePropagationScopeSplitTDVPConfig_t* config) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateTimePropagationScopeSplitTDVPConfig
    _check_or_init_cudensitymat()
    if __cudensitymatCreateTimePropagationScopeSplitTDVPConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateTimePropagationScopeSplitTDVPConfig is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatTimePropagationScopeSplitTDVPConfig_t*) noexcept nogil>__cudensitymatCreateTimePropagationScopeSplitTDVPConfig)(
        handle, config)


cdef cudensitymatStatus_t _cudensitymatDestroyTimePropagationScopeSplitTDVPConfig(cudensitymatTimePropagationScopeSplitTDVPConfig_t config) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyTimePropagationScopeSplitTDVPConfig
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyTimePropagationScopeSplitTDVPConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyTimePropagationScopeSplitTDVPConfig is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatTimePropagationScopeSplitTDVPConfig_t) noexcept nogil>__cudensitymatDestroyTimePropagationScopeSplitTDVPConfig)(
        config)


cdef cudensitymatStatus_t _cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute(const cudensitymatHandle_t handle, cudensitymatTimePropagationScopeSplitTDVPConfig_t config, cudensitymatTimePropagationScopeSplitTDVPConfigAttribute_t attribute, const void* attributeValue, size_t attributeSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute
    _check_or_init_cudensitymat()
    if __cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatTimePropagationScopeSplitTDVPConfig_t, cudensitymatTimePropagationScopeSplitTDVPConfigAttribute_t, const void*, size_t) noexcept nogil>__cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute)(
        handle, config, attribute, attributeValue, attributeSize)


cdef cudensitymatStatus_t _cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute(const cudensitymatHandle_t handle, const cudensitymatTimePropagationScopeSplitTDVPConfig_t config, cudensitymatTimePropagationScopeSplitTDVPConfigAttribute_t attribute, void* attributeValue, size_t attributeSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute
    _check_or_init_cudensitymat()
    if __cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, const cudensitymatTimePropagationScopeSplitTDVPConfig_t, cudensitymatTimePropagationScopeSplitTDVPConfigAttribute_t, void*, size_t) noexcept nogil>__cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute)(
        handle, config, attribute, attributeValue, attributeSize)


cdef cudensitymatStatus_t _cudensitymatCreateTimePropagationApproachKrylovConfig(const cudensitymatHandle_t handle, cudensitymatTimePropagationApproachKrylovConfig_t* config) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateTimePropagationApproachKrylovConfig
    _check_or_init_cudensitymat()
    if __cudensitymatCreateTimePropagationApproachKrylovConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateTimePropagationApproachKrylovConfig is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatTimePropagationApproachKrylovConfig_t*) noexcept nogil>__cudensitymatCreateTimePropagationApproachKrylovConfig)(
        handle, config)


cdef cudensitymatStatus_t _cudensitymatDestroyTimePropagationApproachKrylovConfig(cudensitymatTimePropagationApproachKrylovConfig_t config) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyTimePropagationApproachKrylovConfig
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyTimePropagationApproachKrylovConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyTimePropagationApproachKrylovConfig is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatTimePropagationApproachKrylovConfig_t) noexcept nogil>__cudensitymatDestroyTimePropagationApproachKrylovConfig)(
        config)


cdef cudensitymatStatus_t _cudensitymatTimePropagationApproachKrylovConfigSetAttribute(const cudensitymatHandle_t handle, cudensitymatTimePropagationApproachKrylovConfig_t config, cudensitymatTimePropagationApproachKrylovConfigAttribute_t attribute, const void* attributeValue, size_t attributeSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatTimePropagationApproachKrylovConfigSetAttribute
    _check_or_init_cudensitymat()
    if __cudensitymatTimePropagationApproachKrylovConfigSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatTimePropagationApproachKrylovConfigSetAttribute is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatTimePropagationApproachKrylovConfig_t, cudensitymatTimePropagationApproachKrylovConfigAttribute_t, const void*, size_t) noexcept nogil>__cudensitymatTimePropagationApproachKrylovConfigSetAttribute)(
        handle, config, attribute, attributeValue, attributeSize)


cdef cudensitymatStatus_t _cudensitymatTimePropagationApproachKrylovConfigGetAttribute(const cudensitymatHandle_t handle, const cudensitymatTimePropagationApproachKrylovConfig_t config, cudensitymatTimePropagationApproachKrylovConfigAttribute_t attribute, void* attributeValue, size_t attributeSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatTimePropagationApproachKrylovConfigGetAttribute
    _check_or_init_cudensitymat()
    if __cudensitymatTimePropagationApproachKrylovConfigGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatTimePropagationApproachKrylovConfigGetAttribute is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, const cudensitymatTimePropagationApproachKrylovConfig_t, cudensitymatTimePropagationApproachKrylovConfigAttribute_t, void*, size_t) noexcept nogil>__cudensitymatTimePropagationApproachKrylovConfigGetAttribute)(
        handle, config, attribute, attributeValue, attributeSize)


cdef cudensitymatStatus_t _cudensitymatCreateTimePropagation(const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator, int32_t isHermitian, cudensitymatTimePropagationScopeKind_t scopeKind, cudensitymatTimePropagationApproachKind_t approachKind, cudensitymatTimePropagation_t* timePropagation) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateTimePropagation
    _check_or_init_cudensitymat()
    if __cudensitymatCreateTimePropagation == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateTimePropagation is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatOperator_t, int32_t, cudensitymatTimePropagationScopeKind_t, cudensitymatTimePropagationApproachKind_t, cudensitymatTimePropagation_t*) noexcept nogil>__cudensitymatCreateTimePropagation)(
        handle, superoperator, isHermitian, scopeKind, approachKind, timePropagation)


cdef cudensitymatStatus_t _cudensitymatDestroyTimePropagation(cudensitymatTimePropagation_t timePropagation) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyTimePropagation
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyTimePropagation == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyTimePropagation is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatTimePropagation_t) noexcept nogil>__cudensitymatDestroyTimePropagation)(
        timePropagation)


cdef cudensitymatStatus_t _cudensitymatTimePropagationConfigure(const cudensitymatHandle_t handle, cudensitymatTimePropagation_t timePropagation, cudensitymatTimePropagationAttribute_t attribute, const void* attributeValue, size_t attributeSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatTimePropagationConfigure
    _check_or_init_cudensitymat()
    if __cudensitymatTimePropagationConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatTimePropagationConfigure is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatTimePropagation_t, cudensitymatTimePropagationAttribute_t, const void*, size_t) noexcept nogil>__cudensitymatTimePropagationConfigure)(
        handle, timePropagation, attribute, attributeValue, attributeSize)


cdef cudensitymatStatus_t _cudensitymatTimePropagationPrepare(const cudensitymatHandle_t handle, cudensitymatTimePropagation_t timePropagation, const cudensitymatState_t stateIn, const cudensitymatState_t stateOut, cudensitymatComputeType_t computeType, size_t workspaceSizeLimit, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatTimePropagationPrepare
    _check_or_init_cudensitymat()
    if __cudensitymatTimePropagationPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatTimePropagationPrepare is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatTimePropagation_t, const cudensitymatState_t, const cudensitymatState_t, cudensitymatComputeType_t, size_t, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatTimePropagationPrepare)(
        handle, timePropagation, stateIn, stateOut, computeType, workspaceSizeLimit, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatTimePropagationCompute(const cudensitymatHandle_t handle, cudensitymatTimePropagation_t timePropagation, double timeStepReal, double timeStepImag, double time, int64_t batchSize, int32_t numParams, const double* params, const cudensitymatState_t stateIn, cudensitymatState_t stateOut, cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatTimePropagationCompute
    _check_or_init_cudensitymat()
    if __cudensitymatTimePropagationCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatTimePropagationCompute is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatTimePropagation_t, double, double, double, int64_t, int32_t, const double*, const cudensitymatState_t, cudensitymatState_t, cudensitymatWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cudensitymatTimePropagationCompute)(
        handle, timePropagation, timeStepReal, timeStepImag, time, batchSize, numParams, params, stateIn, stateOut, workspace, stream)


cdef cudensitymatStatus_t _cudensitymatCreateWorkspace(const cudensitymatHandle_t handle, cudensitymatWorkspaceDescriptor_t* workspaceDescr) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatCreateWorkspace
    _check_or_init_cudensitymat()
    if __cudensitymatCreateWorkspace == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatCreateWorkspace is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatWorkspaceDescriptor_t*) noexcept nogil>__cudensitymatCreateWorkspace)(
        handle, workspaceDescr)


cdef cudensitymatStatus_t _cudensitymatDestroyWorkspace(cudensitymatWorkspaceDescriptor_t workspaceDescr) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatDestroyWorkspace
    _check_or_init_cudensitymat()
    if __cudensitymatDestroyWorkspace == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatDestroyWorkspace is not found")
    return (<cudensitymatStatus_t (*)(cudensitymatWorkspaceDescriptor_t) noexcept nogil>__cudensitymatDestroyWorkspace)(
        workspaceDescr)


cdef cudensitymatStatus_t _cudensitymatWorkspaceGetMemorySize(const cudensitymatHandle_t handle, const cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, size_t* memoryBufferSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatWorkspaceGetMemorySize
    _check_or_init_cudensitymat()
    if __cudensitymatWorkspaceGetMemorySize == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatWorkspaceGetMemorySize is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, const cudensitymatWorkspaceDescriptor_t, cudensitymatMemspace_t, cudensitymatWorkspaceKind_t, size_t*) noexcept nogil>__cudensitymatWorkspaceGetMemorySize)(
        handle, workspaceDescr, memSpace, workspaceKind, memoryBufferSize)


cdef cudensitymatStatus_t _cudensitymatWorkspaceSetMemory(const cudensitymatHandle_t handle, cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, void* memoryBuffer, size_t memoryBufferSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatWorkspaceSetMemory
    _check_or_init_cudensitymat()
    if __cudensitymatWorkspaceSetMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatWorkspaceSetMemory is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatWorkspaceDescriptor_t, cudensitymatMemspace_t, cudensitymatWorkspaceKind_t, void*, size_t) noexcept nogil>__cudensitymatWorkspaceSetMemory)(
        handle, workspaceDescr, memSpace, workspaceKind, memoryBuffer, memoryBufferSize)


cdef cudensitymatStatus_t _cudensitymatWorkspaceGetMemory(const cudensitymatHandle_t handle, const cudensitymatWorkspaceDescriptor_t workspaceDescr, cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind, void** memoryBuffer, size_t* memoryBufferSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatWorkspaceGetMemory
    _check_or_init_cudensitymat()
    if __cudensitymatWorkspaceGetMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatWorkspaceGetMemory is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, const cudensitymatWorkspaceDescriptor_t, cudensitymatMemspace_t, cudensitymatWorkspaceKind_t, void**, size_t*) noexcept nogil>__cudensitymatWorkspaceGetMemory)(
        handle, workspaceDescr, memSpace, workspaceKind, memoryBuffer, memoryBufferSize)


cdef cudensitymatStatus_t _cudensitymatElementaryOperatorAttachBuffer(const cudensitymatHandle_t handle, cudensitymatElementaryOperator_t elemOperator, void* buffer, size_t bufferSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatElementaryOperatorAttachBuffer
    _check_or_init_cudensitymat()
    if __cudensitymatElementaryOperatorAttachBuffer == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatElementaryOperatorAttachBuffer is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatElementaryOperator_t, void*, size_t) noexcept nogil>__cudensitymatElementaryOperatorAttachBuffer)(
        handle, elemOperator, buffer, bufferSize)


cdef cudensitymatStatus_t _cudensitymatMatrixOperatorDenseLocalAttachBuffer(const cudensitymatHandle_t handle, cudensitymatMatrixOperator_t matrixOperator, void* buffer, size_t bufferSize) except?_CUDENSITYMATSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudensitymatMatrixOperatorDenseLocalAttachBuffer
    _check_or_init_cudensitymat()
    if __cudensitymatMatrixOperatorDenseLocalAttachBuffer == NULL:
        with gil:
            raise FunctionNotFoundError("function cudensitymatMatrixOperatorDenseLocalAttachBuffer is not found")
    return (<cudensitymatStatus_t (*)(const cudensitymatHandle_t, cudensitymatMatrixOperator_t, void*, size_t) noexcept nogil>__cudensitymatMatrixOperatorDenseLocalAttachBuffer)(
        handle, matrixOperator, buffer, bufferSize)
