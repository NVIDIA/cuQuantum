# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 26.11.0. Do not modify it directly.



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

cdef int _cyb___py_cutensornet_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .._utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################


cdef void* __cutensornetCreate = NULL
cdef void* __cutensornetDestroy = NULL
cdef void* __cutensornetCreateNetworkDescriptor = NULL
cdef void* __cutensornetDestroyNetworkDescriptor = NULL
cdef void* __cutensornetGetOutputTensorDescriptor = NULL
cdef void* __cutensornetGetTensorDetails = NULL
cdef void* __cutensornetCreateWorkspaceDescriptor = NULL
cdef void* __cutensornetWorkspaceComputeContractionSizes = NULL
cdef void* __cutensornetWorkspaceGetMemorySize = NULL
cdef void* __cutensornetWorkspaceSetMemory = NULL
cdef void* __cutensornetWorkspaceGetMemory = NULL
cdef void* __cutensornetDestroyWorkspaceDescriptor = NULL
cdef void* __cutensornetCreateContractionOptimizerConfig = NULL
cdef void* __cutensornetDestroyContractionOptimizerConfig = NULL
cdef void* __cutensornetContractionOptimizerConfigGetAttribute = NULL
cdef void* __cutensornetContractionOptimizerConfigSetAttribute = NULL
cdef void* __cutensornetDestroyContractionOptimizerInfo = NULL
cdef void* __cutensornetCreateContractionOptimizerInfo = NULL
cdef void* __cutensornetContractionOptimize = NULL
cdef void* __cutensornetContractionOptimizerInfoGetAttribute = NULL
cdef void* __cutensornetContractionOptimizerInfoSetAttribute = NULL
cdef void* __cutensornetContractionOptimizerInfoGetPackedSize = NULL
cdef void* __cutensornetContractionOptimizerInfoPackData = NULL
cdef void* __cutensornetCreateContractionOptimizerInfoFromPackedData = NULL
cdef void* __cutensornetUpdateContractionOptimizerInfoFromPackedData = NULL
cdef void* __cutensornetCreateContractionPlan = NULL
cdef void* __cutensornetDestroyContractionPlan = NULL
cdef void* __cutensornetContractionAutotune = NULL
cdef void* __cutensornetCreateContractionAutotunePreference = NULL
cdef void* __cutensornetContractionAutotunePreferenceGetAttribute = NULL
cdef void* __cutensornetContractionAutotunePreferenceSetAttribute = NULL
cdef void* __cutensornetDestroyContractionAutotunePreference = NULL
cdef void* __cutensornetCreateSliceGroupFromIDRange = NULL
cdef void* __cutensornetCreateSliceGroupFromIDs = NULL
cdef void* __cutensornetDestroySliceGroup = NULL
cdef void* __cutensornetContractSlices = NULL
cdef void* __cutensornetCreateTensorDescriptor = NULL
cdef void* __cutensornetDestroyTensorDescriptor = NULL
cdef void* __cutensornetCreateTensorSVDConfig = NULL
cdef void* __cutensornetDestroyTensorSVDConfig = NULL
cdef void* __cutensornetTensorSVDConfigGetAttribute = NULL
cdef void* __cutensornetTensorSVDConfigSetAttribute = NULL
cdef void* __cutensornetWorkspaceComputeSVDSizes = NULL
cdef void* __cutensornetWorkspaceComputeQRSizes = NULL
cdef void* __cutensornetCreateTensorSVDInfo = NULL
cdef void* __cutensornetTensorSVDInfoGetAttribute = NULL
cdef void* __cutensornetDestroyTensorSVDInfo = NULL
cdef void* __cutensornetTensorSVD = NULL
cdef void* __cutensornetTensorQR = NULL
cdef void* __cutensornetWorkspaceComputeGateSplitSizes = NULL
cdef void* __cutensornetGateSplit = NULL
cdef void* __cutensornetGetDeviceMemHandler = NULL
cdef void* __cutensornetSetDeviceMemHandler = NULL
cdef void* __cutensornetLoggerSetCallback = NULL
cdef void* __cutensornetLoggerSetCallbackData = NULL
cdef void* __cutensornetLoggerSetFile = NULL
cdef void* __cutensornetLoggerOpenFile = NULL
cdef void* __cutensornetLoggerSetLevel = NULL
cdef void* __cutensornetLoggerSetMask = NULL
cdef void* __cutensornetLoggerForceDisable = NULL
cdef void* __cutensornetGetVersion = NULL
cdef void* __cutensornetGetCudartVersion = NULL
cdef void* __cutensornetGetErrorString = NULL
cdef void* __cutensornetDistributedResetConfiguration = NULL
cdef void* __cutensornetDistributedGetNumRanks = NULL
cdef void* __cutensornetDistributedGetProcRank = NULL
cdef void* __cutensornetDistributedSynchronize = NULL
cdef void* __cutensornetNetworkGetAttribute = NULL
cdef void* __cutensornetNetworkSetAttribute = NULL
cdef void* __cutensornetWorkspacePurgeCache = NULL
cdef void* __cutensornetCreateState = NULL
cdef void* __cutensornetStateApplyTensor = NULL
cdef void* __cutensornetStateUpdateTensor = NULL
cdef void* __cutensornetDestroyState = NULL
cdef void* __cutensornetCreateMarginal = NULL
cdef void* __cutensornetMarginalConfigure = NULL
cdef void* __cutensornetMarginalPrepare = NULL
cdef void* __cutensornetMarginalCompute = NULL
cdef void* __cutensornetDestroyMarginal = NULL
cdef void* __cutensornetCreateSampler = NULL
cdef void* __cutensornetSamplerConfigure = NULL
cdef void* __cutensornetSamplerPrepare = NULL
cdef void* __cutensornetSamplerSample = NULL
cdef void* __cutensornetDestroySampler = NULL
cdef void* __cutensornetStateFinalizeMPS = NULL
cdef void* __cutensornetStateConfigure = NULL
cdef void* __cutensornetStatePrepare = NULL
cdef void* __cutensornetStateCompute = NULL
cdef void* __cutensornetGetOutputStateDetails = NULL
cdef void* __cutensornetCreateNetworkOperator = NULL
cdef void* __cutensornetNetworkOperatorAppendProduct = NULL
cdef void* __cutensornetDestroyNetworkOperator = NULL
cdef void* __cutensornetCreateAccessor = NULL
cdef void* __cutensornetAccessorConfigure = NULL
cdef void* __cutensornetAccessorPrepare = NULL
cdef void* __cutensornetAccessorCompute = NULL
cdef void* __cutensornetDestroyAccessor = NULL
cdef void* __cutensornetCreateExpectation = NULL
cdef void* __cutensornetExpectationConfigure = NULL
cdef void* __cutensornetExpectationPrepare = NULL
cdef void* __cutensornetExpectationCompute = NULL
cdef void* __cutensornetDestroyExpectation = NULL
cdef void* __cutensornetStateApplyTensorOperator = NULL
cdef void* __cutensornetStateApplyControlledTensorOperator = NULL
cdef void* __cutensornetStateUpdateTensorOperator = NULL
cdef void* __cutensornetStateApplyNetworkOperator = NULL
cdef void* __cutensornetStateInitializeMPS = NULL
cdef void* __cutensornetStateGetInfo = NULL
cdef void* __cutensornetNetworkOperatorAppendMPO = NULL
cdef void* __cutensornetAccessorGetInfo = NULL
cdef void* __cutensornetExpectationGetInfo = NULL
cdef void* __cutensornetMarginalGetInfo = NULL
cdef void* __cutensornetSamplerGetInfo = NULL
cdef void* __cutensornetStateApplyUnitaryChannel = NULL
cdef void* __cutensornetStateCaptureMPS = NULL
cdef void* __cutensornetStateApplyGeneralChannel = NULL
cdef void* __cutensornetCreateStateProjectionMPS = NULL
cdef void* __cutensornetStateProjectionMPSConfigure = NULL
cdef void* __cutensornetStateProjectionMPSPrepare = NULL
cdef void* __cutensornetStateProjectionMPSComputeTensorEnv = NULL
cdef void* __cutensornetStateProjectionMPSGetTensorInfo = NULL
cdef void* __cutensornetStateProjectionMPSExtractTensor = NULL
cdef void* __cutensornetStateProjectionMPSInsertTensor = NULL
cdef void* __cutensornetDestroyStateProjectionMPS = NULL
cdef void* __cutensornetCreateNetwork = NULL
cdef void* __cutensornetDestroyNetwork = NULL
cdef void* __cutensornetNetworkAppendTensor = NULL
cdef void* __cutensornetNetworkSetOutputTensor = NULL
cdef void* __cutensornetNetworkSetOptimizerInfo = NULL
cdef void* __cutensornetNetworkPrepareContraction = NULL
cdef void* __cutensornetNetworkAutotuneContraction = NULL
cdef void* __cutensornetCreateNetworkAutotunePreference = NULL
cdef void* __cutensornetNetworkAutotunePreferenceGetAttribute = NULL
cdef void* __cutensornetNetworkAutotunePreferenceSetAttribute = NULL
cdef void* __cutensornetDestroyNetworkAutotunePreference = NULL
cdef void* __cutensornetNetworkSetInputTensorMemory = NULL
cdef void* __cutensornetNetworkSetOutputTensorMemory = NULL
cdef void* __cutensornetNetworkSetGradientTensorMemory = NULL
cdef void* __cutensornetNetworkSetAdjointTensorMemory = NULL
cdef void* __cutensornetNetworkContract = NULL
cdef void* __cutensornetNetworkPrepareGradientsBackward = NULL
cdef void* __cutensornetNetworkComputeGradientsBackward = NULL
cdef void* __cutensornetStateApplyDiagonalTensorOperator = NULL
cdef void* __cutensornetStateApplyTensorOperatorWithGradient = NULL
cdef void* __cutensornetStateUpdateTensorOperatorGradient = NULL
cdef void* __cutensornetExpectationComputeWithGradientsBackward = NULL
cdef void* __cutensornetStateProjectionMPSUpdateCoefficients = NULL
cdef void* __cutensornetStateProjectionMPSUpdateDualTensors = NULL
cdef void* __cutensornetGetLastError = NULL
cdef void* __cutensornetCreateMarginalDiagonal = NULL
cdef void* __cutensornetCreateDistributedTensorDescriptor = NULL
cdef void* __cutensornetCreateBinaryTensorContraction = NULL
cdef void* __cutensornetBinaryTensorContractionPrepare = NULL
cdef void* __cutensornetBinaryTensorContractionCompute = NULL
cdef void* __cutensornetDestroyBinaryTensorContraction = NULL
cdef void* __cutensornetTensorDescriptorGetAttribute = NULL

cdef int _init_cutensornet() except -1 nogil:
    global _cyb___py_cutensornet_init
    cdef void* handle = NULL
    with gil, _cyb_symbol_lock:
        if _cyb___py_cutensornet_init: return 0

        global __cutensornetCreate
        __cutensornetCreate = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreate')
        if __cutensornetCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreate = _cyb_dlsym(handle, 'cutensornetCreate')

        global __cutensornetDestroy
        __cutensornetDestroy = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroy')
        if __cutensornetDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroy = _cyb_dlsym(handle, 'cutensornetDestroy')

        global __cutensornetCreateNetworkDescriptor
        __cutensornetCreateNetworkDescriptor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateNetworkDescriptor')
        if __cutensornetCreateNetworkDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateNetworkDescriptor = _cyb_dlsym(handle, 'cutensornetCreateNetworkDescriptor')

        global __cutensornetDestroyNetworkDescriptor
        __cutensornetDestroyNetworkDescriptor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyNetworkDescriptor')
        if __cutensornetDestroyNetworkDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyNetworkDescriptor = _cyb_dlsym(handle, 'cutensornetDestroyNetworkDescriptor')

        global __cutensornetGetOutputTensorDescriptor
        __cutensornetGetOutputTensorDescriptor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetGetOutputTensorDescriptor')
        if __cutensornetGetOutputTensorDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetGetOutputTensorDescriptor = _cyb_dlsym(handle, 'cutensornetGetOutputTensorDescriptor')

        global __cutensornetGetTensorDetails
        __cutensornetGetTensorDetails = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetGetTensorDetails')
        if __cutensornetGetTensorDetails == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetGetTensorDetails = _cyb_dlsym(handle, 'cutensornetGetTensorDetails')

        global __cutensornetCreateWorkspaceDescriptor
        __cutensornetCreateWorkspaceDescriptor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateWorkspaceDescriptor')
        if __cutensornetCreateWorkspaceDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateWorkspaceDescriptor = _cyb_dlsym(handle, 'cutensornetCreateWorkspaceDescriptor')

        global __cutensornetWorkspaceComputeContractionSizes
        __cutensornetWorkspaceComputeContractionSizes = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetWorkspaceComputeContractionSizes')
        if __cutensornetWorkspaceComputeContractionSizes == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetWorkspaceComputeContractionSizes = _cyb_dlsym(handle, 'cutensornetWorkspaceComputeContractionSizes')

        global __cutensornetWorkspaceGetMemorySize
        __cutensornetWorkspaceGetMemorySize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetWorkspaceGetMemorySize')
        if __cutensornetWorkspaceGetMemorySize == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetWorkspaceGetMemorySize = _cyb_dlsym(handle, 'cutensornetWorkspaceGetMemorySize')

        global __cutensornetWorkspaceSetMemory
        __cutensornetWorkspaceSetMemory = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetWorkspaceSetMemory')
        if __cutensornetWorkspaceSetMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetWorkspaceSetMemory = _cyb_dlsym(handle, 'cutensornetWorkspaceSetMemory')

        global __cutensornetWorkspaceGetMemory
        __cutensornetWorkspaceGetMemory = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetWorkspaceGetMemory')
        if __cutensornetWorkspaceGetMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetWorkspaceGetMemory = _cyb_dlsym(handle, 'cutensornetWorkspaceGetMemory')

        global __cutensornetDestroyWorkspaceDescriptor
        __cutensornetDestroyWorkspaceDescriptor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyWorkspaceDescriptor')
        if __cutensornetDestroyWorkspaceDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyWorkspaceDescriptor = _cyb_dlsym(handle, 'cutensornetDestroyWorkspaceDescriptor')

        global __cutensornetCreateContractionOptimizerConfig
        __cutensornetCreateContractionOptimizerConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateContractionOptimizerConfig')
        if __cutensornetCreateContractionOptimizerConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateContractionOptimizerConfig = _cyb_dlsym(handle, 'cutensornetCreateContractionOptimizerConfig')

        global __cutensornetDestroyContractionOptimizerConfig
        __cutensornetDestroyContractionOptimizerConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyContractionOptimizerConfig')
        if __cutensornetDestroyContractionOptimizerConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyContractionOptimizerConfig = _cyb_dlsym(handle, 'cutensornetDestroyContractionOptimizerConfig')

        global __cutensornetContractionOptimizerConfigGetAttribute
        __cutensornetContractionOptimizerConfigGetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetContractionOptimizerConfigGetAttribute')
        if __cutensornetContractionOptimizerConfigGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetContractionOptimizerConfigGetAttribute = _cyb_dlsym(handle, 'cutensornetContractionOptimizerConfigGetAttribute')

        global __cutensornetContractionOptimizerConfigSetAttribute
        __cutensornetContractionOptimizerConfigSetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetContractionOptimizerConfigSetAttribute')
        if __cutensornetContractionOptimizerConfigSetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetContractionOptimizerConfigSetAttribute = _cyb_dlsym(handle, 'cutensornetContractionOptimizerConfigSetAttribute')

        global __cutensornetDestroyContractionOptimizerInfo
        __cutensornetDestroyContractionOptimizerInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyContractionOptimizerInfo')
        if __cutensornetDestroyContractionOptimizerInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyContractionOptimizerInfo = _cyb_dlsym(handle, 'cutensornetDestroyContractionOptimizerInfo')

        global __cutensornetCreateContractionOptimizerInfo
        __cutensornetCreateContractionOptimizerInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateContractionOptimizerInfo')
        if __cutensornetCreateContractionOptimizerInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateContractionOptimizerInfo = _cyb_dlsym(handle, 'cutensornetCreateContractionOptimizerInfo')

        global __cutensornetContractionOptimize
        __cutensornetContractionOptimize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetContractionOptimize')
        if __cutensornetContractionOptimize == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetContractionOptimize = _cyb_dlsym(handle, 'cutensornetContractionOptimize')

        global __cutensornetContractionOptimizerInfoGetAttribute
        __cutensornetContractionOptimizerInfoGetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetContractionOptimizerInfoGetAttribute')
        if __cutensornetContractionOptimizerInfoGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetContractionOptimizerInfoGetAttribute = _cyb_dlsym(handle, 'cutensornetContractionOptimizerInfoGetAttribute')

        global __cutensornetContractionOptimizerInfoSetAttribute
        __cutensornetContractionOptimizerInfoSetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetContractionOptimizerInfoSetAttribute')
        if __cutensornetContractionOptimizerInfoSetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetContractionOptimizerInfoSetAttribute = _cyb_dlsym(handle, 'cutensornetContractionOptimizerInfoSetAttribute')

        global __cutensornetContractionOptimizerInfoGetPackedSize
        __cutensornetContractionOptimizerInfoGetPackedSize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetContractionOptimizerInfoGetPackedSize')
        if __cutensornetContractionOptimizerInfoGetPackedSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetContractionOptimizerInfoGetPackedSize = _cyb_dlsym(handle, 'cutensornetContractionOptimizerInfoGetPackedSize')

        global __cutensornetContractionOptimizerInfoPackData
        __cutensornetContractionOptimizerInfoPackData = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetContractionOptimizerInfoPackData')
        if __cutensornetContractionOptimizerInfoPackData == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetContractionOptimizerInfoPackData = _cyb_dlsym(handle, 'cutensornetContractionOptimizerInfoPackData')

        global __cutensornetCreateContractionOptimizerInfoFromPackedData
        __cutensornetCreateContractionOptimizerInfoFromPackedData = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateContractionOptimizerInfoFromPackedData')
        if __cutensornetCreateContractionOptimizerInfoFromPackedData == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateContractionOptimizerInfoFromPackedData = _cyb_dlsym(handle, 'cutensornetCreateContractionOptimizerInfoFromPackedData')

        global __cutensornetUpdateContractionOptimizerInfoFromPackedData
        __cutensornetUpdateContractionOptimizerInfoFromPackedData = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetUpdateContractionOptimizerInfoFromPackedData')
        if __cutensornetUpdateContractionOptimizerInfoFromPackedData == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetUpdateContractionOptimizerInfoFromPackedData = _cyb_dlsym(handle, 'cutensornetUpdateContractionOptimizerInfoFromPackedData')

        global __cutensornetCreateContractionPlan
        __cutensornetCreateContractionPlan = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateContractionPlan')
        if __cutensornetCreateContractionPlan == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateContractionPlan = _cyb_dlsym(handle, 'cutensornetCreateContractionPlan')

        global __cutensornetDestroyContractionPlan
        __cutensornetDestroyContractionPlan = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyContractionPlan')
        if __cutensornetDestroyContractionPlan == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyContractionPlan = _cyb_dlsym(handle, 'cutensornetDestroyContractionPlan')

        global __cutensornetContractionAutotune
        __cutensornetContractionAutotune = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetContractionAutotune')
        if __cutensornetContractionAutotune == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetContractionAutotune = _cyb_dlsym(handle, 'cutensornetContractionAutotune')

        global __cutensornetCreateContractionAutotunePreference
        __cutensornetCreateContractionAutotunePreference = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateContractionAutotunePreference')
        if __cutensornetCreateContractionAutotunePreference == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateContractionAutotunePreference = _cyb_dlsym(handle, 'cutensornetCreateContractionAutotunePreference')

        global __cutensornetContractionAutotunePreferenceGetAttribute
        __cutensornetContractionAutotunePreferenceGetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetContractionAutotunePreferenceGetAttribute')
        if __cutensornetContractionAutotunePreferenceGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetContractionAutotunePreferenceGetAttribute = _cyb_dlsym(handle, 'cutensornetContractionAutotunePreferenceGetAttribute')

        global __cutensornetContractionAutotunePreferenceSetAttribute
        __cutensornetContractionAutotunePreferenceSetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetContractionAutotunePreferenceSetAttribute')
        if __cutensornetContractionAutotunePreferenceSetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetContractionAutotunePreferenceSetAttribute = _cyb_dlsym(handle, 'cutensornetContractionAutotunePreferenceSetAttribute')

        global __cutensornetDestroyContractionAutotunePreference
        __cutensornetDestroyContractionAutotunePreference = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyContractionAutotunePreference')
        if __cutensornetDestroyContractionAutotunePreference == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyContractionAutotunePreference = _cyb_dlsym(handle, 'cutensornetDestroyContractionAutotunePreference')

        global __cutensornetCreateSliceGroupFromIDRange
        __cutensornetCreateSliceGroupFromIDRange = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateSliceGroupFromIDRange')
        if __cutensornetCreateSliceGroupFromIDRange == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateSliceGroupFromIDRange = _cyb_dlsym(handle, 'cutensornetCreateSliceGroupFromIDRange')

        global __cutensornetCreateSliceGroupFromIDs
        __cutensornetCreateSliceGroupFromIDs = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateSliceGroupFromIDs')
        if __cutensornetCreateSliceGroupFromIDs == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateSliceGroupFromIDs = _cyb_dlsym(handle, 'cutensornetCreateSliceGroupFromIDs')

        global __cutensornetDestroySliceGroup
        __cutensornetDestroySliceGroup = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroySliceGroup')
        if __cutensornetDestroySliceGroup == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroySliceGroup = _cyb_dlsym(handle, 'cutensornetDestroySliceGroup')

        global __cutensornetContractSlices
        __cutensornetContractSlices = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetContractSlices')
        if __cutensornetContractSlices == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetContractSlices = _cyb_dlsym(handle, 'cutensornetContractSlices')

        global __cutensornetCreateTensorDescriptor
        __cutensornetCreateTensorDescriptor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateTensorDescriptor')
        if __cutensornetCreateTensorDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateTensorDescriptor = _cyb_dlsym(handle, 'cutensornetCreateTensorDescriptor')

        global __cutensornetDestroyTensorDescriptor
        __cutensornetDestroyTensorDescriptor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyTensorDescriptor')
        if __cutensornetDestroyTensorDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyTensorDescriptor = _cyb_dlsym(handle, 'cutensornetDestroyTensorDescriptor')

        global __cutensornetCreateTensorSVDConfig
        __cutensornetCreateTensorSVDConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateTensorSVDConfig')
        if __cutensornetCreateTensorSVDConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateTensorSVDConfig = _cyb_dlsym(handle, 'cutensornetCreateTensorSVDConfig')

        global __cutensornetDestroyTensorSVDConfig
        __cutensornetDestroyTensorSVDConfig = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyTensorSVDConfig')
        if __cutensornetDestroyTensorSVDConfig == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyTensorSVDConfig = _cyb_dlsym(handle, 'cutensornetDestroyTensorSVDConfig')

        global __cutensornetTensorSVDConfigGetAttribute
        __cutensornetTensorSVDConfigGetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetTensorSVDConfigGetAttribute')
        if __cutensornetTensorSVDConfigGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetTensorSVDConfigGetAttribute = _cyb_dlsym(handle, 'cutensornetTensorSVDConfigGetAttribute')

        global __cutensornetTensorSVDConfigSetAttribute
        __cutensornetTensorSVDConfigSetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetTensorSVDConfigSetAttribute')
        if __cutensornetTensorSVDConfigSetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetTensorSVDConfigSetAttribute = _cyb_dlsym(handle, 'cutensornetTensorSVDConfigSetAttribute')

        global __cutensornetWorkspaceComputeSVDSizes
        __cutensornetWorkspaceComputeSVDSizes = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetWorkspaceComputeSVDSizes')
        if __cutensornetWorkspaceComputeSVDSizes == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetWorkspaceComputeSVDSizes = _cyb_dlsym(handle, 'cutensornetWorkspaceComputeSVDSizes')

        global __cutensornetWorkspaceComputeQRSizes
        __cutensornetWorkspaceComputeQRSizes = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetWorkspaceComputeQRSizes')
        if __cutensornetWorkspaceComputeQRSizes == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetWorkspaceComputeQRSizes = _cyb_dlsym(handle, 'cutensornetWorkspaceComputeQRSizes')

        global __cutensornetCreateTensorSVDInfo
        __cutensornetCreateTensorSVDInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateTensorSVDInfo')
        if __cutensornetCreateTensorSVDInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateTensorSVDInfo = _cyb_dlsym(handle, 'cutensornetCreateTensorSVDInfo')

        global __cutensornetTensorSVDInfoGetAttribute
        __cutensornetTensorSVDInfoGetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetTensorSVDInfoGetAttribute')
        if __cutensornetTensorSVDInfoGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetTensorSVDInfoGetAttribute = _cyb_dlsym(handle, 'cutensornetTensorSVDInfoGetAttribute')

        global __cutensornetDestroyTensorSVDInfo
        __cutensornetDestroyTensorSVDInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyTensorSVDInfo')
        if __cutensornetDestroyTensorSVDInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyTensorSVDInfo = _cyb_dlsym(handle, 'cutensornetDestroyTensorSVDInfo')

        global __cutensornetTensorSVD
        __cutensornetTensorSVD = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetTensorSVD')
        if __cutensornetTensorSVD == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetTensorSVD = _cyb_dlsym(handle, 'cutensornetTensorSVD')

        global __cutensornetTensorQR
        __cutensornetTensorQR = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetTensorQR')
        if __cutensornetTensorQR == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetTensorQR = _cyb_dlsym(handle, 'cutensornetTensorQR')

        global __cutensornetWorkspaceComputeGateSplitSizes
        __cutensornetWorkspaceComputeGateSplitSizes = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetWorkspaceComputeGateSplitSizes')
        if __cutensornetWorkspaceComputeGateSplitSizes == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetWorkspaceComputeGateSplitSizes = _cyb_dlsym(handle, 'cutensornetWorkspaceComputeGateSplitSizes')

        global __cutensornetGateSplit
        __cutensornetGateSplit = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetGateSplit')
        if __cutensornetGateSplit == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetGateSplit = _cyb_dlsym(handle, 'cutensornetGateSplit')

        global __cutensornetGetDeviceMemHandler
        __cutensornetGetDeviceMemHandler = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetGetDeviceMemHandler')
        if __cutensornetGetDeviceMemHandler == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetGetDeviceMemHandler = _cyb_dlsym(handle, 'cutensornetGetDeviceMemHandler')

        global __cutensornetSetDeviceMemHandler
        __cutensornetSetDeviceMemHandler = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetSetDeviceMemHandler')
        if __cutensornetSetDeviceMemHandler == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetSetDeviceMemHandler = _cyb_dlsym(handle, 'cutensornetSetDeviceMemHandler')

        global __cutensornetLoggerSetCallback
        __cutensornetLoggerSetCallback = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetLoggerSetCallback')
        if __cutensornetLoggerSetCallback == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetLoggerSetCallback = _cyb_dlsym(handle, 'cutensornetLoggerSetCallback')

        global __cutensornetLoggerSetCallbackData
        __cutensornetLoggerSetCallbackData = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetLoggerSetCallbackData')
        if __cutensornetLoggerSetCallbackData == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetLoggerSetCallbackData = _cyb_dlsym(handle, 'cutensornetLoggerSetCallbackData')

        global __cutensornetLoggerSetFile
        __cutensornetLoggerSetFile = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetLoggerSetFile')
        if __cutensornetLoggerSetFile == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetLoggerSetFile = _cyb_dlsym(handle, 'cutensornetLoggerSetFile')

        global __cutensornetLoggerOpenFile
        __cutensornetLoggerOpenFile = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetLoggerOpenFile')
        if __cutensornetLoggerOpenFile == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetLoggerOpenFile = _cyb_dlsym(handle, 'cutensornetLoggerOpenFile')

        global __cutensornetLoggerSetLevel
        __cutensornetLoggerSetLevel = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetLoggerSetLevel')
        if __cutensornetLoggerSetLevel == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetLoggerSetLevel = _cyb_dlsym(handle, 'cutensornetLoggerSetLevel')

        global __cutensornetLoggerSetMask
        __cutensornetLoggerSetMask = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetLoggerSetMask')
        if __cutensornetLoggerSetMask == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetLoggerSetMask = _cyb_dlsym(handle, 'cutensornetLoggerSetMask')

        global __cutensornetLoggerForceDisable
        __cutensornetLoggerForceDisable = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetLoggerForceDisable')
        if __cutensornetLoggerForceDisable == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetLoggerForceDisable = _cyb_dlsym(handle, 'cutensornetLoggerForceDisable')

        global __cutensornetGetVersion
        __cutensornetGetVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetGetVersion')
        if __cutensornetGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetGetVersion = _cyb_dlsym(handle, 'cutensornetGetVersion')

        global __cutensornetGetCudartVersion
        __cutensornetGetCudartVersion = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetGetCudartVersion')
        if __cutensornetGetCudartVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetGetCudartVersion = _cyb_dlsym(handle, 'cutensornetGetCudartVersion')

        global __cutensornetGetErrorString
        __cutensornetGetErrorString = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetGetErrorString')
        if __cutensornetGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetGetErrorString = _cyb_dlsym(handle, 'cutensornetGetErrorString')

        global __cutensornetDistributedResetConfiguration
        __cutensornetDistributedResetConfiguration = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDistributedResetConfiguration')
        if __cutensornetDistributedResetConfiguration == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDistributedResetConfiguration = _cyb_dlsym(handle, 'cutensornetDistributedResetConfiguration')

        global __cutensornetDistributedGetNumRanks
        __cutensornetDistributedGetNumRanks = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDistributedGetNumRanks')
        if __cutensornetDistributedGetNumRanks == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDistributedGetNumRanks = _cyb_dlsym(handle, 'cutensornetDistributedGetNumRanks')

        global __cutensornetDistributedGetProcRank
        __cutensornetDistributedGetProcRank = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDistributedGetProcRank')
        if __cutensornetDistributedGetProcRank == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDistributedGetProcRank = _cyb_dlsym(handle, 'cutensornetDistributedGetProcRank')

        global __cutensornetDistributedSynchronize
        __cutensornetDistributedSynchronize = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDistributedSynchronize')
        if __cutensornetDistributedSynchronize == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDistributedSynchronize = _cyb_dlsym(handle, 'cutensornetDistributedSynchronize')

        global __cutensornetNetworkGetAttribute
        __cutensornetNetworkGetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkGetAttribute')
        if __cutensornetNetworkGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkGetAttribute = _cyb_dlsym(handle, 'cutensornetNetworkGetAttribute')

        global __cutensornetNetworkSetAttribute
        __cutensornetNetworkSetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkSetAttribute')
        if __cutensornetNetworkSetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkSetAttribute = _cyb_dlsym(handle, 'cutensornetNetworkSetAttribute')

        global __cutensornetWorkspacePurgeCache
        __cutensornetWorkspacePurgeCache = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetWorkspacePurgeCache')
        if __cutensornetWorkspacePurgeCache == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetWorkspacePurgeCache = _cyb_dlsym(handle, 'cutensornetWorkspacePurgeCache')

        global __cutensornetCreateState
        __cutensornetCreateState = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateState')
        if __cutensornetCreateState == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateState = _cyb_dlsym(handle, 'cutensornetCreateState')

        global __cutensornetStateApplyTensor
        __cutensornetStateApplyTensor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateApplyTensor')
        if __cutensornetStateApplyTensor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateApplyTensor = _cyb_dlsym(handle, 'cutensornetStateApplyTensor')

        global __cutensornetStateUpdateTensor
        __cutensornetStateUpdateTensor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateUpdateTensor')
        if __cutensornetStateUpdateTensor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateUpdateTensor = _cyb_dlsym(handle, 'cutensornetStateUpdateTensor')

        global __cutensornetDestroyState
        __cutensornetDestroyState = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyState')
        if __cutensornetDestroyState == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyState = _cyb_dlsym(handle, 'cutensornetDestroyState')

        global __cutensornetCreateMarginal
        __cutensornetCreateMarginal = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateMarginal')
        if __cutensornetCreateMarginal == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateMarginal = _cyb_dlsym(handle, 'cutensornetCreateMarginal')

        global __cutensornetMarginalConfigure
        __cutensornetMarginalConfigure = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetMarginalConfigure')
        if __cutensornetMarginalConfigure == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetMarginalConfigure = _cyb_dlsym(handle, 'cutensornetMarginalConfigure')

        global __cutensornetMarginalPrepare
        __cutensornetMarginalPrepare = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetMarginalPrepare')
        if __cutensornetMarginalPrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetMarginalPrepare = _cyb_dlsym(handle, 'cutensornetMarginalPrepare')

        global __cutensornetMarginalCompute
        __cutensornetMarginalCompute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetMarginalCompute')
        if __cutensornetMarginalCompute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetMarginalCompute = _cyb_dlsym(handle, 'cutensornetMarginalCompute')

        global __cutensornetDestroyMarginal
        __cutensornetDestroyMarginal = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyMarginal')
        if __cutensornetDestroyMarginal == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyMarginal = _cyb_dlsym(handle, 'cutensornetDestroyMarginal')

        global __cutensornetCreateSampler
        __cutensornetCreateSampler = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateSampler')
        if __cutensornetCreateSampler == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateSampler = _cyb_dlsym(handle, 'cutensornetCreateSampler')

        global __cutensornetSamplerConfigure
        __cutensornetSamplerConfigure = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetSamplerConfigure')
        if __cutensornetSamplerConfigure == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetSamplerConfigure = _cyb_dlsym(handle, 'cutensornetSamplerConfigure')

        global __cutensornetSamplerPrepare
        __cutensornetSamplerPrepare = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetSamplerPrepare')
        if __cutensornetSamplerPrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetSamplerPrepare = _cyb_dlsym(handle, 'cutensornetSamplerPrepare')

        global __cutensornetSamplerSample
        __cutensornetSamplerSample = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetSamplerSample')
        if __cutensornetSamplerSample == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetSamplerSample = _cyb_dlsym(handle, 'cutensornetSamplerSample')

        global __cutensornetDestroySampler
        __cutensornetDestroySampler = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroySampler')
        if __cutensornetDestroySampler == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroySampler = _cyb_dlsym(handle, 'cutensornetDestroySampler')

        global __cutensornetStateFinalizeMPS
        __cutensornetStateFinalizeMPS = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateFinalizeMPS')
        if __cutensornetStateFinalizeMPS == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateFinalizeMPS = _cyb_dlsym(handle, 'cutensornetStateFinalizeMPS')

        global __cutensornetStateConfigure
        __cutensornetStateConfigure = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateConfigure')
        if __cutensornetStateConfigure == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateConfigure = _cyb_dlsym(handle, 'cutensornetStateConfigure')

        global __cutensornetStatePrepare
        __cutensornetStatePrepare = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStatePrepare')
        if __cutensornetStatePrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStatePrepare = _cyb_dlsym(handle, 'cutensornetStatePrepare')

        global __cutensornetStateCompute
        __cutensornetStateCompute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateCompute')
        if __cutensornetStateCompute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateCompute = _cyb_dlsym(handle, 'cutensornetStateCompute')

        global __cutensornetGetOutputStateDetails
        __cutensornetGetOutputStateDetails = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetGetOutputStateDetails')
        if __cutensornetGetOutputStateDetails == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetGetOutputStateDetails = _cyb_dlsym(handle, 'cutensornetGetOutputStateDetails')

        global __cutensornetCreateNetworkOperator
        __cutensornetCreateNetworkOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateNetworkOperator')
        if __cutensornetCreateNetworkOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateNetworkOperator = _cyb_dlsym(handle, 'cutensornetCreateNetworkOperator')

        global __cutensornetNetworkOperatorAppendProduct
        __cutensornetNetworkOperatorAppendProduct = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkOperatorAppendProduct')
        if __cutensornetNetworkOperatorAppendProduct == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkOperatorAppendProduct = _cyb_dlsym(handle, 'cutensornetNetworkOperatorAppendProduct')

        global __cutensornetDestroyNetworkOperator
        __cutensornetDestroyNetworkOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyNetworkOperator')
        if __cutensornetDestroyNetworkOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyNetworkOperator = _cyb_dlsym(handle, 'cutensornetDestroyNetworkOperator')

        global __cutensornetCreateAccessor
        __cutensornetCreateAccessor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateAccessor')
        if __cutensornetCreateAccessor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateAccessor = _cyb_dlsym(handle, 'cutensornetCreateAccessor')

        global __cutensornetAccessorConfigure
        __cutensornetAccessorConfigure = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetAccessorConfigure')
        if __cutensornetAccessorConfigure == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetAccessorConfigure = _cyb_dlsym(handle, 'cutensornetAccessorConfigure')

        global __cutensornetAccessorPrepare
        __cutensornetAccessorPrepare = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetAccessorPrepare')
        if __cutensornetAccessorPrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetAccessorPrepare = _cyb_dlsym(handle, 'cutensornetAccessorPrepare')

        global __cutensornetAccessorCompute
        __cutensornetAccessorCompute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetAccessorCompute')
        if __cutensornetAccessorCompute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetAccessorCompute = _cyb_dlsym(handle, 'cutensornetAccessorCompute')

        global __cutensornetDestroyAccessor
        __cutensornetDestroyAccessor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyAccessor')
        if __cutensornetDestroyAccessor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyAccessor = _cyb_dlsym(handle, 'cutensornetDestroyAccessor')

        global __cutensornetCreateExpectation
        __cutensornetCreateExpectation = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateExpectation')
        if __cutensornetCreateExpectation == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateExpectation = _cyb_dlsym(handle, 'cutensornetCreateExpectation')

        global __cutensornetExpectationConfigure
        __cutensornetExpectationConfigure = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetExpectationConfigure')
        if __cutensornetExpectationConfigure == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetExpectationConfigure = _cyb_dlsym(handle, 'cutensornetExpectationConfigure')

        global __cutensornetExpectationPrepare
        __cutensornetExpectationPrepare = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetExpectationPrepare')
        if __cutensornetExpectationPrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetExpectationPrepare = _cyb_dlsym(handle, 'cutensornetExpectationPrepare')

        global __cutensornetExpectationCompute
        __cutensornetExpectationCompute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetExpectationCompute')
        if __cutensornetExpectationCompute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetExpectationCompute = _cyb_dlsym(handle, 'cutensornetExpectationCompute')

        global __cutensornetDestroyExpectation
        __cutensornetDestroyExpectation = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyExpectation')
        if __cutensornetDestroyExpectation == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyExpectation = _cyb_dlsym(handle, 'cutensornetDestroyExpectation')

        global __cutensornetStateApplyTensorOperator
        __cutensornetStateApplyTensorOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateApplyTensorOperator')
        if __cutensornetStateApplyTensorOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateApplyTensorOperator = _cyb_dlsym(handle, 'cutensornetStateApplyTensorOperator')

        global __cutensornetStateApplyControlledTensorOperator
        __cutensornetStateApplyControlledTensorOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateApplyControlledTensorOperator')
        if __cutensornetStateApplyControlledTensorOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateApplyControlledTensorOperator = _cyb_dlsym(handle, 'cutensornetStateApplyControlledTensorOperator')

        global __cutensornetStateUpdateTensorOperator
        __cutensornetStateUpdateTensorOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateUpdateTensorOperator')
        if __cutensornetStateUpdateTensorOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateUpdateTensorOperator = _cyb_dlsym(handle, 'cutensornetStateUpdateTensorOperator')

        global __cutensornetStateApplyNetworkOperator
        __cutensornetStateApplyNetworkOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateApplyNetworkOperator')
        if __cutensornetStateApplyNetworkOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateApplyNetworkOperator = _cyb_dlsym(handle, 'cutensornetStateApplyNetworkOperator')

        global __cutensornetStateInitializeMPS
        __cutensornetStateInitializeMPS = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateInitializeMPS')
        if __cutensornetStateInitializeMPS == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateInitializeMPS = _cyb_dlsym(handle, 'cutensornetStateInitializeMPS')

        global __cutensornetStateGetInfo
        __cutensornetStateGetInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateGetInfo')
        if __cutensornetStateGetInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateGetInfo = _cyb_dlsym(handle, 'cutensornetStateGetInfo')

        global __cutensornetNetworkOperatorAppendMPO
        __cutensornetNetworkOperatorAppendMPO = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkOperatorAppendMPO')
        if __cutensornetNetworkOperatorAppendMPO == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkOperatorAppendMPO = _cyb_dlsym(handle, 'cutensornetNetworkOperatorAppendMPO')

        global __cutensornetAccessorGetInfo
        __cutensornetAccessorGetInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetAccessorGetInfo')
        if __cutensornetAccessorGetInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetAccessorGetInfo = _cyb_dlsym(handle, 'cutensornetAccessorGetInfo')

        global __cutensornetExpectationGetInfo
        __cutensornetExpectationGetInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetExpectationGetInfo')
        if __cutensornetExpectationGetInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetExpectationGetInfo = _cyb_dlsym(handle, 'cutensornetExpectationGetInfo')

        global __cutensornetMarginalGetInfo
        __cutensornetMarginalGetInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetMarginalGetInfo')
        if __cutensornetMarginalGetInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetMarginalGetInfo = _cyb_dlsym(handle, 'cutensornetMarginalGetInfo')

        global __cutensornetSamplerGetInfo
        __cutensornetSamplerGetInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetSamplerGetInfo')
        if __cutensornetSamplerGetInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetSamplerGetInfo = _cyb_dlsym(handle, 'cutensornetSamplerGetInfo')

        global __cutensornetStateApplyUnitaryChannel
        __cutensornetStateApplyUnitaryChannel = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateApplyUnitaryChannel')
        if __cutensornetStateApplyUnitaryChannel == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateApplyUnitaryChannel = _cyb_dlsym(handle, 'cutensornetStateApplyUnitaryChannel')

        global __cutensornetStateCaptureMPS
        __cutensornetStateCaptureMPS = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateCaptureMPS')
        if __cutensornetStateCaptureMPS == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateCaptureMPS = _cyb_dlsym(handle, 'cutensornetStateCaptureMPS')

        global __cutensornetStateApplyGeneralChannel
        __cutensornetStateApplyGeneralChannel = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateApplyGeneralChannel')
        if __cutensornetStateApplyGeneralChannel == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateApplyGeneralChannel = _cyb_dlsym(handle, 'cutensornetStateApplyGeneralChannel')

        global __cutensornetCreateStateProjectionMPS
        __cutensornetCreateStateProjectionMPS = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateStateProjectionMPS')
        if __cutensornetCreateStateProjectionMPS == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateStateProjectionMPS = _cyb_dlsym(handle, 'cutensornetCreateStateProjectionMPS')

        global __cutensornetStateProjectionMPSConfigure
        __cutensornetStateProjectionMPSConfigure = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateProjectionMPSConfigure')
        if __cutensornetStateProjectionMPSConfigure == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateProjectionMPSConfigure = _cyb_dlsym(handle, 'cutensornetStateProjectionMPSConfigure')

        global __cutensornetStateProjectionMPSPrepare
        __cutensornetStateProjectionMPSPrepare = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateProjectionMPSPrepare')
        if __cutensornetStateProjectionMPSPrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateProjectionMPSPrepare = _cyb_dlsym(handle, 'cutensornetStateProjectionMPSPrepare')

        global __cutensornetStateProjectionMPSComputeTensorEnv
        __cutensornetStateProjectionMPSComputeTensorEnv = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateProjectionMPSComputeTensorEnv')
        if __cutensornetStateProjectionMPSComputeTensorEnv == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateProjectionMPSComputeTensorEnv = _cyb_dlsym(handle, 'cutensornetStateProjectionMPSComputeTensorEnv')

        global __cutensornetStateProjectionMPSGetTensorInfo
        __cutensornetStateProjectionMPSGetTensorInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateProjectionMPSGetTensorInfo')
        if __cutensornetStateProjectionMPSGetTensorInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateProjectionMPSGetTensorInfo = _cyb_dlsym(handle, 'cutensornetStateProjectionMPSGetTensorInfo')

        global __cutensornetStateProjectionMPSExtractTensor
        __cutensornetStateProjectionMPSExtractTensor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateProjectionMPSExtractTensor')
        if __cutensornetStateProjectionMPSExtractTensor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateProjectionMPSExtractTensor = _cyb_dlsym(handle, 'cutensornetStateProjectionMPSExtractTensor')

        global __cutensornetStateProjectionMPSInsertTensor
        __cutensornetStateProjectionMPSInsertTensor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateProjectionMPSInsertTensor')
        if __cutensornetStateProjectionMPSInsertTensor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateProjectionMPSInsertTensor = _cyb_dlsym(handle, 'cutensornetStateProjectionMPSInsertTensor')

        global __cutensornetDestroyStateProjectionMPS
        __cutensornetDestroyStateProjectionMPS = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyStateProjectionMPS')
        if __cutensornetDestroyStateProjectionMPS == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyStateProjectionMPS = _cyb_dlsym(handle, 'cutensornetDestroyStateProjectionMPS')

        global __cutensornetCreateNetwork
        __cutensornetCreateNetwork = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateNetwork')
        if __cutensornetCreateNetwork == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateNetwork = _cyb_dlsym(handle, 'cutensornetCreateNetwork')

        global __cutensornetDestroyNetwork
        __cutensornetDestroyNetwork = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyNetwork')
        if __cutensornetDestroyNetwork == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyNetwork = _cyb_dlsym(handle, 'cutensornetDestroyNetwork')

        global __cutensornetNetworkAppendTensor
        __cutensornetNetworkAppendTensor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkAppendTensor')
        if __cutensornetNetworkAppendTensor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkAppendTensor = _cyb_dlsym(handle, 'cutensornetNetworkAppendTensor')

        global __cutensornetNetworkSetOutputTensor
        __cutensornetNetworkSetOutputTensor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkSetOutputTensor')
        if __cutensornetNetworkSetOutputTensor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkSetOutputTensor = _cyb_dlsym(handle, 'cutensornetNetworkSetOutputTensor')

        global __cutensornetNetworkSetOptimizerInfo
        __cutensornetNetworkSetOptimizerInfo = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkSetOptimizerInfo')
        if __cutensornetNetworkSetOptimizerInfo == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkSetOptimizerInfo = _cyb_dlsym(handle, 'cutensornetNetworkSetOptimizerInfo')

        global __cutensornetNetworkPrepareContraction
        __cutensornetNetworkPrepareContraction = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkPrepareContraction')
        if __cutensornetNetworkPrepareContraction == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkPrepareContraction = _cyb_dlsym(handle, 'cutensornetNetworkPrepareContraction')

        global __cutensornetNetworkAutotuneContraction
        __cutensornetNetworkAutotuneContraction = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkAutotuneContraction')
        if __cutensornetNetworkAutotuneContraction == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkAutotuneContraction = _cyb_dlsym(handle, 'cutensornetNetworkAutotuneContraction')

        global __cutensornetCreateNetworkAutotunePreference
        __cutensornetCreateNetworkAutotunePreference = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateNetworkAutotunePreference')
        if __cutensornetCreateNetworkAutotunePreference == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateNetworkAutotunePreference = _cyb_dlsym(handle, 'cutensornetCreateNetworkAutotunePreference')

        global __cutensornetNetworkAutotunePreferenceGetAttribute
        __cutensornetNetworkAutotunePreferenceGetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkAutotunePreferenceGetAttribute')
        if __cutensornetNetworkAutotunePreferenceGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkAutotunePreferenceGetAttribute = _cyb_dlsym(handle, 'cutensornetNetworkAutotunePreferenceGetAttribute')

        global __cutensornetNetworkAutotunePreferenceSetAttribute
        __cutensornetNetworkAutotunePreferenceSetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkAutotunePreferenceSetAttribute')
        if __cutensornetNetworkAutotunePreferenceSetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkAutotunePreferenceSetAttribute = _cyb_dlsym(handle, 'cutensornetNetworkAutotunePreferenceSetAttribute')

        global __cutensornetDestroyNetworkAutotunePreference
        __cutensornetDestroyNetworkAutotunePreference = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyNetworkAutotunePreference')
        if __cutensornetDestroyNetworkAutotunePreference == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyNetworkAutotunePreference = _cyb_dlsym(handle, 'cutensornetDestroyNetworkAutotunePreference')

        global __cutensornetNetworkSetInputTensorMemory
        __cutensornetNetworkSetInputTensorMemory = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkSetInputTensorMemory')
        if __cutensornetNetworkSetInputTensorMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkSetInputTensorMemory = _cyb_dlsym(handle, 'cutensornetNetworkSetInputTensorMemory')

        global __cutensornetNetworkSetOutputTensorMemory
        __cutensornetNetworkSetOutputTensorMemory = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkSetOutputTensorMemory')
        if __cutensornetNetworkSetOutputTensorMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkSetOutputTensorMemory = _cyb_dlsym(handle, 'cutensornetNetworkSetOutputTensorMemory')

        global __cutensornetNetworkSetGradientTensorMemory
        __cutensornetNetworkSetGradientTensorMemory = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkSetGradientTensorMemory')
        if __cutensornetNetworkSetGradientTensorMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkSetGradientTensorMemory = _cyb_dlsym(handle, 'cutensornetNetworkSetGradientTensorMemory')

        global __cutensornetNetworkSetAdjointTensorMemory
        __cutensornetNetworkSetAdjointTensorMemory = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkSetAdjointTensorMemory')
        if __cutensornetNetworkSetAdjointTensorMemory == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkSetAdjointTensorMemory = _cyb_dlsym(handle, 'cutensornetNetworkSetAdjointTensorMemory')

        global __cutensornetNetworkContract
        __cutensornetNetworkContract = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkContract')
        if __cutensornetNetworkContract == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkContract = _cyb_dlsym(handle, 'cutensornetNetworkContract')

        global __cutensornetNetworkPrepareGradientsBackward
        __cutensornetNetworkPrepareGradientsBackward = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkPrepareGradientsBackward')
        if __cutensornetNetworkPrepareGradientsBackward == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkPrepareGradientsBackward = _cyb_dlsym(handle, 'cutensornetNetworkPrepareGradientsBackward')

        global __cutensornetNetworkComputeGradientsBackward
        __cutensornetNetworkComputeGradientsBackward = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetNetworkComputeGradientsBackward')
        if __cutensornetNetworkComputeGradientsBackward == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetNetworkComputeGradientsBackward = _cyb_dlsym(handle, 'cutensornetNetworkComputeGradientsBackward')

        global __cutensornetStateApplyDiagonalTensorOperator
        __cutensornetStateApplyDiagonalTensorOperator = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateApplyDiagonalTensorOperator')
        if __cutensornetStateApplyDiagonalTensorOperator == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateApplyDiagonalTensorOperator = _cyb_dlsym(handle, 'cutensornetStateApplyDiagonalTensorOperator')

        global __cutensornetStateApplyTensorOperatorWithGradient
        __cutensornetStateApplyTensorOperatorWithGradient = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateApplyTensorOperatorWithGradient')
        if __cutensornetStateApplyTensorOperatorWithGradient == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateApplyTensorOperatorWithGradient = _cyb_dlsym(handle, 'cutensornetStateApplyTensorOperatorWithGradient')

        global __cutensornetStateUpdateTensorOperatorGradient
        __cutensornetStateUpdateTensorOperatorGradient = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateUpdateTensorOperatorGradient')
        if __cutensornetStateUpdateTensorOperatorGradient == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateUpdateTensorOperatorGradient = _cyb_dlsym(handle, 'cutensornetStateUpdateTensorOperatorGradient')

        global __cutensornetExpectationComputeWithGradientsBackward
        __cutensornetExpectationComputeWithGradientsBackward = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetExpectationComputeWithGradientsBackward')
        if __cutensornetExpectationComputeWithGradientsBackward == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetExpectationComputeWithGradientsBackward = _cyb_dlsym(handle, 'cutensornetExpectationComputeWithGradientsBackward')

        global __cutensornetStateProjectionMPSUpdateCoefficients
        __cutensornetStateProjectionMPSUpdateCoefficients = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateProjectionMPSUpdateCoefficients')
        if __cutensornetStateProjectionMPSUpdateCoefficients == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateProjectionMPSUpdateCoefficients = _cyb_dlsym(handle, 'cutensornetStateProjectionMPSUpdateCoefficients')

        global __cutensornetStateProjectionMPSUpdateDualTensors
        __cutensornetStateProjectionMPSUpdateDualTensors = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetStateProjectionMPSUpdateDualTensors')
        if __cutensornetStateProjectionMPSUpdateDualTensors == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetStateProjectionMPSUpdateDualTensors = _cyb_dlsym(handle, 'cutensornetStateProjectionMPSUpdateDualTensors')

        global __cutensornetGetLastError
        __cutensornetGetLastError = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetGetLastError')
        if __cutensornetGetLastError == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetGetLastError = _cyb_dlsym(handle, 'cutensornetGetLastError')

        global __cutensornetCreateMarginalDiagonal
        __cutensornetCreateMarginalDiagonal = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateMarginalDiagonal')
        if __cutensornetCreateMarginalDiagonal == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateMarginalDiagonal = _cyb_dlsym(handle, 'cutensornetCreateMarginalDiagonal')

        global __cutensornetCreateDistributedTensorDescriptor
        __cutensornetCreateDistributedTensorDescriptor = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateDistributedTensorDescriptor')
        if __cutensornetCreateDistributedTensorDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateDistributedTensorDescriptor = _cyb_dlsym(handle, 'cutensornetCreateDistributedTensorDescriptor')

        global __cutensornetCreateBinaryTensorContraction
        __cutensornetCreateBinaryTensorContraction = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetCreateBinaryTensorContraction')
        if __cutensornetCreateBinaryTensorContraction == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetCreateBinaryTensorContraction = _cyb_dlsym(handle, 'cutensornetCreateBinaryTensorContraction')

        global __cutensornetBinaryTensorContractionPrepare
        __cutensornetBinaryTensorContractionPrepare = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetBinaryTensorContractionPrepare')
        if __cutensornetBinaryTensorContractionPrepare == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetBinaryTensorContractionPrepare = _cyb_dlsym(handle, 'cutensornetBinaryTensorContractionPrepare')

        global __cutensornetBinaryTensorContractionCompute
        __cutensornetBinaryTensorContractionCompute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetBinaryTensorContractionCompute')
        if __cutensornetBinaryTensorContractionCompute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetBinaryTensorContractionCompute = _cyb_dlsym(handle, 'cutensornetBinaryTensorContractionCompute')

        global __cutensornetDestroyBinaryTensorContraction
        __cutensornetDestroyBinaryTensorContraction = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetDestroyBinaryTensorContraction')
        if __cutensornetDestroyBinaryTensorContraction == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetDestroyBinaryTensorContraction = _cyb_dlsym(handle, 'cutensornetDestroyBinaryTensorContraction')

        global __cutensornetTensorDescriptorGetAttribute
        __cutensornetTensorDescriptorGetAttribute = _cyb_dlsym(_cyb_RTLD_DEFAULT, 'cutensornetTensorDescriptorGetAttribute')
        if __cutensornetTensorDescriptorGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensornetTensorDescriptorGetAttribute = _cyb_dlsym(handle, 'cutensornetTensorDescriptorGetAttribute')

        _cyb_atomic_int_store(<int *>&_cyb___py_cutensornet_init, 1)
        return 0

cdef inline int _check_or_init_cutensornet() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_cutensornet_init):
        return 0

    return _init_cutensornet()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_cutensornet()
    cdef dict data = {}
    global __cutensornetCreate
    data["__cutensornetCreate"] = <intptr_t>__cutensornetCreate

    global __cutensornetDestroy
    data["__cutensornetDestroy"] = <intptr_t>__cutensornetDestroy

    global __cutensornetCreateNetworkDescriptor
    data["__cutensornetCreateNetworkDescriptor"] = <intptr_t>__cutensornetCreateNetworkDescriptor

    global __cutensornetDestroyNetworkDescriptor
    data["__cutensornetDestroyNetworkDescriptor"] = <intptr_t>__cutensornetDestroyNetworkDescriptor

    global __cutensornetGetOutputTensorDescriptor
    data["__cutensornetGetOutputTensorDescriptor"] = <intptr_t>__cutensornetGetOutputTensorDescriptor

    global __cutensornetGetTensorDetails
    data["__cutensornetGetTensorDetails"] = <intptr_t>__cutensornetGetTensorDetails

    global __cutensornetCreateWorkspaceDescriptor
    data["__cutensornetCreateWorkspaceDescriptor"] = <intptr_t>__cutensornetCreateWorkspaceDescriptor

    global __cutensornetWorkspaceComputeContractionSizes
    data["__cutensornetWorkspaceComputeContractionSizes"] = <intptr_t>__cutensornetWorkspaceComputeContractionSizes

    global __cutensornetWorkspaceGetMemorySize
    data["__cutensornetWorkspaceGetMemorySize"] = <intptr_t>__cutensornetWorkspaceGetMemorySize

    global __cutensornetWorkspaceSetMemory
    data["__cutensornetWorkspaceSetMemory"] = <intptr_t>__cutensornetWorkspaceSetMemory

    global __cutensornetWorkspaceGetMemory
    data["__cutensornetWorkspaceGetMemory"] = <intptr_t>__cutensornetWorkspaceGetMemory

    global __cutensornetDestroyWorkspaceDescriptor
    data["__cutensornetDestroyWorkspaceDescriptor"] = <intptr_t>__cutensornetDestroyWorkspaceDescriptor

    global __cutensornetCreateContractionOptimizerConfig
    data["__cutensornetCreateContractionOptimizerConfig"] = <intptr_t>__cutensornetCreateContractionOptimizerConfig

    global __cutensornetDestroyContractionOptimizerConfig
    data["__cutensornetDestroyContractionOptimizerConfig"] = <intptr_t>__cutensornetDestroyContractionOptimizerConfig

    global __cutensornetContractionOptimizerConfigGetAttribute
    data["__cutensornetContractionOptimizerConfigGetAttribute"] = <intptr_t>__cutensornetContractionOptimizerConfigGetAttribute

    global __cutensornetContractionOptimizerConfigSetAttribute
    data["__cutensornetContractionOptimizerConfigSetAttribute"] = <intptr_t>__cutensornetContractionOptimizerConfigSetAttribute

    global __cutensornetDestroyContractionOptimizerInfo
    data["__cutensornetDestroyContractionOptimizerInfo"] = <intptr_t>__cutensornetDestroyContractionOptimizerInfo

    global __cutensornetCreateContractionOptimizerInfo
    data["__cutensornetCreateContractionOptimizerInfo"] = <intptr_t>__cutensornetCreateContractionOptimizerInfo

    global __cutensornetContractionOptimize
    data["__cutensornetContractionOptimize"] = <intptr_t>__cutensornetContractionOptimize

    global __cutensornetContractionOptimizerInfoGetAttribute
    data["__cutensornetContractionOptimizerInfoGetAttribute"] = <intptr_t>__cutensornetContractionOptimizerInfoGetAttribute

    global __cutensornetContractionOptimizerInfoSetAttribute
    data["__cutensornetContractionOptimizerInfoSetAttribute"] = <intptr_t>__cutensornetContractionOptimizerInfoSetAttribute

    global __cutensornetContractionOptimizerInfoGetPackedSize
    data["__cutensornetContractionOptimizerInfoGetPackedSize"] = <intptr_t>__cutensornetContractionOptimizerInfoGetPackedSize

    global __cutensornetContractionOptimizerInfoPackData
    data["__cutensornetContractionOptimizerInfoPackData"] = <intptr_t>__cutensornetContractionOptimizerInfoPackData

    global __cutensornetCreateContractionOptimizerInfoFromPackedData
    data["__cutensornetCreateContractionOptimizerInfoFromPackedData"] = <intptr_t>__cutensornetCreateContractionOptimizerInfoFromPackedData

    global __cutensornetUpdateContractionOptimizerInfoFromPackedData
    data["__cutensornetUpdateContractionOptimizerInfoFromPackedData"] = <intptr_t>__cutensornetUpdateContractionOptimizerInfoFromPackedData

    global __cutensornetCreateContractionPlan
    data["__cutensornetCreateContractionPlan"] = <intptr_t>__cutensornetCreateContractionPlan

    global __cutensornetDestroyContractionPlan
    data["__cutensornetDestroyContractionPlan"] = <intptr_t>__cutensornetDestroyContractionPlan

    global __cutensornetContractionAutotune
    data["__cutensornetContractionAutotune"] = <intptr_t>__cutensornetContractionAutotune

    global __cutensornetCreateContractionAutotunePreference
    data["__cutensornetCreateContractionAutotunePreference"] = <intptr_t>__cutensornetCreateContractionAutotunePreference

    global __cutensornetContractionAutotunePreferenceGetAttribute
    data["__cutensornetContractionAutotunePreferenceGetAttribute"] = <intptr_t>__cutensornetContractionAutotunePreferenceGetAttribute

    global __cutensornetContractionAutotunePreferenceSetAttribute
    data["__cutensornetContractionAutotunePreferenceSetAttribute"] = <intptr_t>__cutensornetContractionAutotunePreferenceSetAttribute

    global __cutensornetDestroyContractionAutotunePreference
    data["__cutensornetDestroyContractionAutotunePreference"] = <intptr_t>__cutensornetDestroyContractionAutotunePreference

    global __cutensornetCreateSliceGroupFromIDRange
    data["__cutensornetCreateSliceGroupFromIDRange"] = <intptr_t>__cutensornetCreateSliceGroupFromIDRange

    global __cutensornetCreateSliceGroupFromIDs
    data["__cutensornetCreateSliceGroupFromIDs"] = <intptr_t>__cutensornetCreateSliceGroupFromIDs

    global __cutensornetDestroySliceGroup
    data["__cutensornetDestroySliceGroup"] = <intptr_t>__cutensornetDestroySliceGroup

    global __cutensornetContractSlices
    data["__cutensornetContractSlices"] = <intptr_t>__cutensornetContractSlices

    global __cutensornetCreateTensorDescriptor
    data["__cutensornetCreateTensorDescriptor"] = <intptr_t>__cutensornetCreateTensorDescriptor

    global __cutensornetDestroyTensorDescriptor
    data["__cutensornetDestroyTensorDescriptor"] = <intptr_t>__cutensornetDestroyTensorDescriptor

    global __cutensornetCreateTensorSVDConfig
    data["__cutensornetCreateTensorSVDConfig"] = <intptr_t>__cutensornetCreateTensorSVDConfig

    global __cutensornetDestroyTensorSVDConfig
    data["__cutensornetDestroyTensorSVDConfig"] = <intptr_t>__cutensornetDestroyTensorSVDConfig

    global __cutensornetTensorSVDConfigGetAttribute
    data["__cutensornetTensorSVDConfigGetAttribute"] = <intptr_t>__cutensornetTensorSVDConfigGetAttribute

    global __cutensornetTensorSVDConfigSetAttribute
    data["__cutensornetTensorSVDConfigSetAttribute"] = <intptr_t>__cutensornetTensorSVDConfigSetAttribute

    global __cutensornetWorkspaceComputeSVDSizes
    data["__cutensornetWorkspaceComputeSVDSizes"] = <intptr_t>__cutensornetWorkspaceComputeSVDSizes

    global __cutensornetWorkspaceComputeQRSizes
    data["__cutensornetWorkspaceComputeQRSizes"] = <intptr_t>__cutensornetWorkspaceComputeQRSizes

    global __cutensornetCreateTensorSVDInfo
    data["__cutensornetCreateTensorSVDInfo"] = <intptr_t>__cutensornetCreateTensorSVDInfo

    global __cutensornetTensorSVDInfoGetAttribute
    data["__cutensornetTensorSVDInfoGetAttribute"] = <intptr_t>__cutensornetTensorSVDInfoGetAttribute

    global __cutensornetDestroyTensorSVDInfo
    data["__cutensornetDestroyTensorSVDInfo"] = <intptr_t>__cutensornetDestroyTensorSVDInfo

    global __cutensornetTensorSVD
    data["__cutensornetTensorSVD"] = <intptr_t>__cutensornetTensorSVD

    global __cutensornetTensorQR
    data["__cutensornetTensorQR"] = <intptr_t>__cutensornetTensorQR

    global __cutensornetWorkspaceComputeGateSplitSizes
    data["__cutensornetWorkspaceComputeGateSplitSizes"] = <intptr_t>__cutensornetWorkspaceComputeGateSplitSizes

    global __cutensornetGateSplit
    data["__cutensornetGateSplit"] = <intptr_t>__cutensornetGateSplit

    global __cutensornetGetDeviceMemHandler
    data["__cutensornetGetDeviceMemHandler"] = <intptr_t>__cutensornetGetDeviceMemHandler

    global __cutensornetSetDeviceMemHandler
    data["__cutensornetSetDeviceMemHandler"] = <intptr_t>__cutensornetSetDeviceMemHandler

    global __cutensornetLoggerSetCallback
    data["__cutensornetLoggerSetCallback"] = <intptr_t>__cutensornetLoggerSetCallback

    global __cutensornetLoggerSetCallbackData
    data["__cutensornetLoggerSetCallbackData"] = <intptr_t>__cutensornetLoggerSetCallbackData

    global __cutensornetLoggerSetFile
    data["__cutensornetLoggerSetFile"] = <intptr_t>__cutensornetLoggerSetFile

    global __cutensornetLoggerOpenFile
    data["__cutensornetLoggerOpenFile"] = <intptr_t>__cutensornetLoggerOpenFile

    global __cutensornetLoggerSetLevel
    data["__cutensornetLoggerSetLevel"] = <intptr_t>__cutensornetLoggerSetLevel

    global __cutensornetLoggerSetMask
    data["__cutensornetLoggerSetMask"] = <intptr_t>__cutensornetLoggerSetMask

    global __cutensornetLoggerForceDisable
    data["__cutensornetLoggerForceDisable"] = <intptr_t>__cutensornetLoggerForceDisable

    global __cutensornetGetVersion
    data["__cutensornetGetVersion"] = <intptr_t>__cutensornetGetVersion

    global __cutensornetGetCudartVersion
    data["__cutensornetGetCudartVersion"] = <intptr_t>__cutensornetGetCudartVersion

    global __cutensornetGetErrorString
    data["__cutensornetGetErrorString"] = <intptr_t>__cutensornetGetErrorString

    global __cutensornetDistributedResetConfiguration
    data["__cutensornetDistributedResetConfiguration"] = <intptr_t>__cutensornetDistributedResetConfiguration

    global __cutensornetDistributedGetNumRanks
    data["__cutensornetDistributedGetNumRanks"] = <intptr_t>__cutensornetDistributedGetNumRanks

    global __cutensornetDistributedGetProcRank
    data["__cutensornetDistributedGetProcRank"] = <intptr_t>__cutensornetDistributedGetProcRank

    global __cutensornetDistributedSynchronize
    data["__cutensornetDistributedSynchronize"] = <intptr_t>__cutensornetDistributedSynchronize

    global __cutensornetNetworkGetAttribute
    data["__cutensornetNetworkGetAttribute"] = <intptr_t>__cutensornetNetworkGetAttribute

    global __cutensornetNetworkSetAttribute
    data["__cutensornetNetworkSetAttribute"] = <intptr_t>__cutensornetNetworkSetAttribute

    global __cutensornetWorkspacePurgeCache
    data["__cutensornetWorkspacePurgeCache"] = <intptr_t>__cutensornetWorkspacePurgeCache

    global __cutensornetCreateState
    data["__cutensornetCreateState"] = <intptr_t>__cutensornetCreateState

    global __cutensornetStateApplyTensor
    data["__cutensornetStateApplyTensor"] = <intptr_t>__cutensornetStateApplyTensor

    global __cutensornetStateUpdateTensor
    data["__cutensornetStateUpdateTensor"] = <intptr_t>__cutensornetStateUpdateTensor

    global __cutensornetDestroyState
    data["__cutensornetDestroyState"] = <intptr_t>__cutensornetDestroyState

    global __cutensornetCreateMarginal
    data["__cutensornetCreateMarginal"] = <intptr_t>__cutensornetCreateMarginal

    global __cutensornetMarginalConfigure
    data["__cutensornetMarginalConfigure"] = <intptr_t>__cutensornetMarginalConfigure

    global __cutensornetMarginalPrepare
    data["__cutensornetMarginalPrepare"] = <intptr_t>__cutensornetMarginalPrepare

    global __cutensornetMarginalCompute
    data["__cutensornetMarginalCompute"] = <intptr_t>__cutensornetMarginalCompute

    global __cutensornetDestroyMarginal
    data["__cutensornetDestroyMarginal"] = <intptr_t>__cutensornetDestroyMarginal

    global __cutensornetCreateSampler
    data["__cutensornetCreateSampler"] = <intptr_t>__cutensornetCreateSampler

    global __cutensornetSamplerConfigure
    data["__cutensornetSamplerConfigure"] = <intptr_t>__cutensornetSamplerConfigure

    global __cutensornetSamplerPrepare
    data["__cutensornetSamplerPrepare"] = <intptr_t>__cutensornetSamplerPrepare

    global __cutensornetSamplerSample
    data["__cutensornetSamplerSample"] = <intptr_t>__cutensornetSamplerSample

    global __cutensornetDestroySampler
    data["__cutensornetDestroySampler"] = <intptr_t>__cutensornetDestroySampler

    global __cutensornetStateFinalizeMPS
    data["__cutensornetStateFinalizeMPS"] = <intptr_t>__cutensornetStateFinalizeMPS

    global __cutensornetStateConfigure
    data["__cutensornetStateConfigure"] = <intptr_t>__cutensornetStateConfigure

    global __cutensornetStatePrepare
    data["__cutensornetStatePrepare"] = <intptr_t>__cutensornetStatePrepare

    global __cutensornetStateCompute
    data["__cutensornetStateCompute"] = <intptr_t>__cutensornetStateCompute

    global __cutensornetGetOutputStateDetails
    data["__cutensornetGetOutputStateDetails"] = <intptr_t>__cutensornetGetOutputStateDetails

    global __cutensornetCreateNetworkOperator
    data["__cutensornetCreateNetworkOperator"] = <intptr_t>__cutensornetCreateNetworkOperator

    global __cutensornetNetworkOperatorAppendProduct
    data["__cutensornetNetworkOperatorAppendProduct"] = <intptr_t>__cutensornetNetworkOperatorAppendProduct

    global __cutensornetDestroyNetworkOperator
    data["__cutensornetDestroyNetworkOperator"] = <intptr_t>__cutensornetDestroyNetworkOperator

    global __cutensornetCreateAccessor
    data["__cutensornetCreateAccessor"] = <intptr_t>__cutensornetCreateAccessor

    global __cutensornetAccessorConfigure
    data["__cutensornetAccessorConfigure"] = <intptr_t>__cutensornetAccessorConfigure

    global __cutensornetAccessorPrepare
    data["__cutensornetAccessorPrepare"] = <intptr_t>__cutensornetAccessorPrepare

    global __cutensornetAccessorCompute
    data["__cutensornetAccessorCompute"] = <intptr_t>__cutensornetAccessorCompute

    global __cutensornetDestroyAccessor
    data["__cutensornetDestroyAccessor"] = <intptr_t>__cutensornetDestroyAccessor

    global __cutensornetCreateExpectation
    data["__cutensornetCreateExpectation"] = <intptr_t>__cutensornetCreateExpectation

    global __cutensornetExpectationConfigure
    data["__cutensornetExpectationConfigure"] = <intptr_t>__cutensornetExpectationConfigure

    global __cutensornetExpectationPrepare
    data["__cutensornetExpectationPrepare"] = <intptr_t>__cutensornetExpectationPrepare

    global __cutensornetExpectationCompute
    data["__cutensornetExpectationCompute"] = <intptr_t>__cutensornetExpectationCompute

    global __cutensornetDestroyExpectation
    data["__cutensornetDestroyExpectation"] = <intptr_t>__cutensornetDestroyExpectation

    global __cutensornetStateApplyTensorOperator
    data["__cutensornetStateApplyTensorOperator"] = <intptr_t>__cutensornetStateApplyTensorOperator

    global __cutensornetStateApplyControlledTensorOperator
    data["__cutensornetStateApplyControlledTensorOperator"] = <intptr_t>__cutensornetStateApplyControlledTensorOperator

    global __cutensornetStateUpdateTensorOperator
    data["__cutensornetStateUpdateTensorOperator"] = <intptr_t>__cutensornetStateUpdateTensorOperator

    global __cutensornetStateApplyNetworkOperator
    data["__cutensornetStateApplyNetworkOperator"] = <intptr_t>__cutensornetStateApplyNetworkOperator

    global __cutensornetStateInitializeMPS
    data["__cutensornetStateInitializeMPS"] = <intptr_t>__cutensornetStateInitializeMPS

    global __cutensornetStateGetInfo
    data["__cutensornetStateGetInfo"] = <intptr_t>__cutensornetStateGetInfo

    global __cutensornetNetworkOperatorAppendMPO
    data["__cutensornetNetworkOperatorAppendMPO"] = <intptr_t>__cutensornetNetworkOperatorAppendMPO

    global __cutensornetAccessorGetInfo
    data["__cutensornetAccessorGetInfo"] = <intptr_t>__cutensornetAccessorGetInfo

    global __cutensornetExpectationGetInfo
    data["__cutensornetExpectationGetInfo"] = <intptr_t>__cutensornetExpectationGetInfo

    global __cutensornetMarginalGetInfo
    data["__cutensornetMarginalGetInfo"] = <intptr_t>__cutensornetMarginalGetInfo

    global __cutensornetSamplerGetInfo
    data["__cutensornetSamplerGetInfo"] = <intptr_t>__cutensornetSamplerGetInfo

    global __cutensornetStateApplyUnitaryChannel
    data["__cutensornetStateApplyUnitaryChannel"] = <intptr_t>__cutensornetStateApplyUnitaryChannel

    global __cutensornetStateCaptureMPS
    data["__cutensornetStateCaptureMPS"] = <intptr_t>__cutensornetStateCaptureMPS

    global __cutensornetStateApplyGeneralChannel
    data["__cutensornetStateApplyGeneralChannel"] = <intptr_t>__cutensornetStateApplyGeneralChannel

    global __cutensornetCreateStateProjectionMPS
    data["__cutensornetCreateStateProjectionMPS"] = <intptr_t>__cutensornetCreateStateProjectionMPS

    global __cutensornetStateProjectionMPSConfigure
    data["__cutensornetStateProjectionMPSConfigure"] = <intptr_t>__cutensornetStateProjectionMPSConfigure

    global __cutensornetStateProjectionMPSPrepare
    data["__cutensornetStateProjectionMPSPrepare"] = <intptr_t>__cutensornetStateProjectionMPSPrepare

    global __cutensornetStateProjectionMPSComputeTensorEnv
    data["__cutensornetStateProjectionMPSComputeTensorEnv"] = <intptr_t>__cutensornetStateProjectionMPSComputeTensorEnv

    global __cutensornetStateProjectionMPSGetTensorInfo
    data["__cutensornetStateProjectionMPSGetTensorInfo"] = <intptr_t>__cutensornetStateProjectionMPSGetTensorInfo

    global __cutensornetStateProjectionMPSExtractTensor
    data["__cutensornetStateProjectionMPSExtractTensor"] = <intptr_t>__cutensornetStateProjectionMPSExtractTensor

    global __cutensornetStateProjectionMPSInsertTensor
    data["__cutensornetStateProjectionMPSInsertTensor"] = <intptr_t>__cutensornetStateProjectionMPSInsertTensor

    global __cutensornetDestroyStateProjectionMPS
    data["__cutensornetDestroyStateProjectionMPS"] = <intptr_t>__cutensornetDestroyStateProjectionMPS

    global __cutensornetCreateNetwork
    data["__cutensornetCreateNetwork"] = <intptr_t>__cutensornetCreateNetwork

    global __cutensornetDestroyNetwork
    data["__cutensornetDestroyNetwork"] = <intptr_t>__cutensornetDestroyNetwork

    global __cutensornetNetworkAppendTensor
    data["__cutensornetNetworkAppendTensor"] = <intptr_t>__cutensornetNetworkAppendTensor

    global __cutensornetNetworkSetOutputTensor
    data["__cutensornetNetworkSetOutputTensor"] = <intptr_t>__cutensornetNetworkSetOutputTensor

    global __cutensornetNetworkSetOptimizerInfo
    data["__cutensornetNetworkSetOptimizerInfo"] = <intptr_t>__cutensornetNetworkSetOptimizerInfo

    global __cutensornetNetworkPrepareContraction
    data["__cutensornetNetworkPrepareContraction"] = <intptr_t>__cutensornetNetworkPrepareContraction

    global __cutensornetNetworkAutotuneContraction
    data["__cutensornetNetworkAutotuneContraction"] = <intptr_t>__cutensornetNetworkAutotuneContraction

    global __cutensornetCreateNetworkAutotunePreference
    data["__cutensornetCreateNetworkAutotunePreference"] = <intptr_t>__cutensornetCreateNetworkAutotunePreference

    global __cutensornetNetworkAutotunePreferenceGetAttribute
    data["__cutensornetNetworkAutotunePreferenceGetAttribute"] = <intptr_t>__cutensornetNetworkAutotunePreferenceGetAttribute

    global __cutensornetNetworkAutotunePreferenceSetAttribute
    data["__cutensornetNetworkAutotunePreferenceSetAttribute"] = <intptr_t>__cutensornetNetworkAutotunePreferenceSetAttribute

    global __cutensornetDestroyNetworkAutotunePreference
    data["__cutensornetDestroyNetworkAutotunePreference"] = <intptr_t>__cutensornetDestroyNetworkAutotunePreference

    global __cutensornetNetworkSetInputTensorMemory
    data["__cutensornetNetworkSetInputTensorMemory"] = <intptr_t>__cutensornetNetworkSetInputTensorMemory

    global __cutensornetNetworkSetOutputTensorMemory
    data["__cutensornetNetworkSetOutputTensorMemory"] = <intptr_t>__cutensornetNetworkSetOutputTensorMemory

    global __cutensornetNetworkSetGradientTensorMemory
    data["__cutensornetNetworkSetGradientTensorMemory"] = <intptr_t>__cutensornetNetworkSetGradientTensorMemory

    global __cutensornetNetworkSetAdjointTensorMemory
    data["__cutensornetNetworkSetAdjointTensorMemory"] = <intptr_t>__cutensornetNetworkSetAdjointTensorMemory

    global __cutensornetNetworkContract
    data["__cutensornetNetworkContract"] = <intptr_t>__cutensornetNetworkContract

    global __cutensornetNetworkPrepareGradientsBackward
    data["__cutensornetNetworkPrepareGradientsBackward"] = <intptr_t>__cutensornetNetworkPrepareGradientsBackward

    global __cutensornetNetworkComputeGradientsBackward
    data["__cutensornetNetworkComputeGradientsBackward"] = <intptr_t>__cutensornetNetworkComputeGradientsBackward

    global __cutensornetStateApplyDiagonalTensorOperator
    data["__cutensornetStateApplyDiagonalTensorOperator"] = <intptr_t>__cutensornetStateApplyDiagonalTensorOperator

    global __cutensornetStateApplyTensorOperatorWithGradient
    data["__cutensornetStateApplyTensorOperatorWithGradient"] = <intptr_t>__cutensornetStateApplyTensorOperatorWithGradient

    global __cutensornetStateUpdateTensorOperatorGradient
    data["__cutensornetStateUpdateTensorOperatorGradient"] = <intptr_t>__cutensornetStateUpdateTensorOperatorGradient

    global __cutensornetExpectationComputeWithGradientsBackward
    data["__cutensornetExpectationComputeWithGradientsBackward"] = <intptr_t>__cutensornetExpectationComputeWithGradientsBackward

    global __cutensornetStateProjectionMPSUpdateCoefficients
    data["__cutensornetStateProjectionMPSUpdateCoefficients"] = <intptr_t>__cutensornetStateProjectionMPSUpdateCoefficients

    global __cutensornetStateProjectionMPSUpdateDualTensors
    data["__cutensornetStateProjectionMPSUpdateDualTensors"] = <intptr_t>__cutensornetStateProjectionMPSUpdateDualTensors

    global __cutensornetGetLastError
    data["__cutensornetGetLastError"] = <intptr_t>__cutensornetGetLastError

    global __cutensornetCreateMarginalDiagonal
    data["__cutensornetCreateMarginalDiagonal"] = <intptr_t>__cutensornetCreateMarginalDiagonal

    global __cutensornetCreateDistributedTensorDescriptor
    data["__cutensornetCreateDistributedTensorDescriptor"] = <intptr_t>__cutensornetCreateDistributedTensorDescriptor

    global __cutensornetCreateBinaryTensorContraction
    data["__cutensornetCreateBinaryTensorContraction"] = <intptr_t>__cutensornetCreateBinaryTensorContraction

    global __cutensornetBinaryTensorContractionPrepare
    data["__cutensornetBinaryTensorContractionPrepare"] = <intptr_t>__cutensornetBinaryTensorContractionPrepare

    global __cutensornetBinaryTensorContractionCompute
    data["__cutensornetBinaryTensorContractionCompute"] = <intptr_t>__cutensornetBinaryTensorContractionCompute

    global __cutensornetDestroyBinaryTensorContraction
    data["__cutensornetDestroyBinaryTensorContraction"] = <intptr_t>__cutensornetDestroyBinaryTensorContraction

    global __cutensornetTensorDescriptorGetAttribute
    data["__cutensornetTensorDescriptorGetAttribute"] = <intptr_t>__cutensornetTensorDescriptorGetAttribute
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cutensornet")._handle_uint
    return <void*>handle


###############################################################################
# Wrapper functions
###############################################################################

cdef cutensornetStatus_t _cutensornetCreate(cutensornetHandle_t* handle) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreate
    _check_or_init_cutensornet()
    if __cutensornetCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreate is not found")
    return (<cutensornetStatus_t (*)(cutensornetHandle_t*) noexcept nogil>__cutensornetCreate)(
        handle)


cdef cutensornetStatus_t _cutensornetDestroy(cutensornetHandle_t handle) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroy
    _check_or_init_cutensornet()
    if __cutensornetDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroy is not found")
    return (<cutensornetStatus_t (*)(cutensornetHandle_t) noexcept nogil>__cutensornetDestroy)(
        handle)


cdef cutensornetStatus_t _cutensornetCreateNetworkDescriptor(const cutensornetHandle_t handle, int32_t numInputs, const int32_t numModesIn[], const int64_t* const extentsIn[], const int64_t* const stridesIn[], const int32_t* const modesIn[], const cutensornetTensorQualifiers_t qualifiersIn[], int32_t numModesOut, const int64_t extentsOut[], const int64_t stridesOut[], const int32_t modesOut[], cudaDataType_t dataType, cutensornetComputeType_t computeType, cutensornetNetworkDescriptor_t* networkDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateNetworkDescriptor
    _check_or_init_cutensornet()
    if __cutensornetCreateNetworkDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateNetworkDescriptor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t, const int32_t*, const int64_t* const*, const int64_t* const*, const int32_t* const*, const cutensornetTensorQualifiers_t*, int32_t, const int64_t*, const int64_t*, const int32_t*, cudaDataType_t, cutensornetComputeType_t, cutensornetNetworkDescriptor_t*) noexcept nogil>__cutensornetCreateNetworkDescriptor)(
        handle, numInputs, numModesIn, extentsIn, stridesIn, modesIn, qualifiersIn, numModesOut, extentsOut, stridesOut, modesOut, dataType, computeType, networkDesc)


cdef cutensornetStatus_t _cutensornetDestroyNetworkDescriptor(cutensornetNetworkDescriptor_t networkDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyNetworkDescriptor
    _check_or_init_cutensornet()
    if __cutensornetDestroyNetworkDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyNetworkDescriptor is not found")
    return (<cutensornetStatus_t (*)(cutensornetNetworkDescriptor_t) noexcept nogil>__cutensornetDestroyNetworkDescriptor)(
        networkDesc)


cdef cutensornetStatus_t _cutensornetGetOutputTensorDescriptor(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, cutensornetTensorDescriptor_t* outputTensorDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetGetOutputTensorDescriptor
    _check_or_init_cutensornet()
    if __cutensornetGetOutputTensorDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetOutputTensorDescriptor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, cutensornetTensorDescriptor_t*) noexcept nogil>__cutensornetGetOutputTensorDescriptor)(
        handle, networkDesc, outputTensorDesc)


cdef cutensornetStatus_t _cutensornetGetTensorDetails(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t tensorDesc, int32_t* numModes, size_t* dataSize, int32_t* modeLabels, int64_t* extents, int64_t* strides) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetGetTensorDetails
    _check_or_init_cutensornet()
    if __cutensornetGetTensorDetails == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetTensorDetails is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, int32_t*, size_t*, int32_t*, int64_t*, int64_t*) noexcept nogil>__cutensornetGetTensorDetails)(
        handle, tensorDesc, numModes, dataSize, modeLabels, extents, strides)


cdef cutensornetStatus_t _cutensornetCreateWorkspaceDescriptor(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t* workDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateWorkspaceDescriptor
    _check_or_init_cutensornet()
    if __cutensornetCreateWorkspaceDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateWorkspaceDescriptor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetWorkspaceDescriptor_t*) noexcept nogil>__cutensornetCreateWorkspaceDescriptor)(
        handle, workDesc)


cdef cutensornetStatus_t _cutensornetWorkspaceComputeContractionSizes(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, const cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetWorkspaceDescriptor_t workDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetWorkspaceComputeContractionSizes
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceComputeContractionSizes == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceComputeContractionSizes is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, const cutensornetContractionOptimizerInfo_t, cutensornetWorkspaceDescriptor_t) noexcept nogil>__cutensornetWorkspaceComputeContractionSizes)(
        handle, networkDesc, optimizerInfo, workDesc)


cdef cutensornetStatus_t _cutensornetWorkspaceGetMemorySize(const cutensornetHandle_t handle, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetWorksizePref_t workPref, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, int64_t* memorySize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetWorkspaceGetMemorySize
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceGetMemorySize == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceGetMemorySize is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetWorkspaceDescriptor_t, cutensornetWorksizePref_t, cutensornetMemspace_t, cutensornetWorkspaceKind_t, int64_t*) noexcept nogil>__cutensornetWorkspaceGetMemorySize)(
        handle, workDesc, workPref, memSpace, workKind, memorySize)


cdef cutensornetStatus_t _cutensornetWorkspaceSetMemory(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, void* const memoryPtr, int64_t memorySize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetWorkspaceSetMemory
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceSetMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceSetMemory is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetWorkspaceDescriptor_t, cutensornetMemspace_t, cutensornetWorkspaceKind_t, void* const, int64_t) noexcept nogil>__cutensornetWorkspaceSetMemory)(
        handle, workDesc, memSpace, workKind, memoryPtr, memorySize)


cdef cutensornetStatus_t _cutensornetWorkspaceGetMemory(const cutensornetHandle_t handle, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, void** memoryPtr, int64_t* memorySize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetWorkspaceGetMemory
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceGetMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceGetMemory is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetWorkspaceDescriptor_t, cutensornetMemspace_t, cutensornetWorkspaceKind_t, void**, int64_t*) noexcept nogil>__cutensornetWorkspaceGetMemory)(
        handle, workDesc, memSpace, workKind, memoryPtr, memorySize)


cdef cutensornetStatus_t _cutensornetDestroyWorkspaceDescriptor(cutensornetWorkspaceDescriptor_t workDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyWorkspaceDescriptor
    _check_or_init_cutensornet()
    if __cutensornetDestroyWorkspaceDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyWorkspaceDescriptor is not found")
    return (<cutensornetStatus_t (*)(cutensornetWorkspaceDescriptor_t) noexcept nogil>__cutensornetDestroyWorkspaceDescriptor)(
        workDesc)


cdef cutensornetStatus_t _cutensornetCreateContractionOptimizerConfig(const cutensornetHandle_t handle, cutensornetContractionOptimizerConfig_t* optimizerConfig) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateContractionOptimizerConfig
    _check_or_init_cutensornet()
    if __cutensornetCreateContractionOptimizerConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateContractionOptimizerConfig is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionOptimizerConfig_t*) noexcept nogil>__cutensornetCreateContractionOptimizerConfig)(
        handle, optimizerConfig)


cdef cutensornetStatus_t _cutensornetDestroyContractionOptimizerConfig(cutensornetContractionOptimizerConfig_t optimizerConfig) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyContractionOptimizerConfig
    _check_or_init_cutensornet()
    if __cutensornetDestroyContractionOptimizerConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyContractionOptimizerConfig is not found")
    return (<cutensornetStatus_t (*)(cutensornetContractionOptimizerConfig_t) noexcept nogil>__cutensornetDestroyContractionOptimizerConfig)(
        optimizerConfig)


cdef cutensornetStatus_t _cutensornetContractionOptimizerConfigGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionOptimizerConfig_t optimizerConfig, cutensornetContractionOptimizerConfigAttributes_t attr, void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetContractionOptimizerConfigGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerConfigGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerConfigGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetContractionOptimizerConfig_t, cutensornetContractionOptimizerConfigAttributes_t, void*, size_t) noexcept nogil>__cutensornetContractionOptimizerConfigGetAttribute)(
        handle, optimizerConfig, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetContractionOptimizerConfigSetAttribute(const cutensornetHandle_t handle, cutensornetContractionOptimizerConfig_t optimizerConfig, cutensornetContractionOptimizerConfigAttributes_t attr, const void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetContractionOptimizerConfigSetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerConfigSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerConfigSetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionOptimizerConfig_t, cutensornetContractionOptimizerConfigAttributes_t, const void*, size_t) noexcept nogil>__cutensornetContractionOptimizerConfigSetAttribute)(
        handle, optimizerConfig, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetDestroyContractionOptimizerInfo(cutensornetContractionOptimizerInfo_t optimizerInfo) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyContractionOptimizerInfo
    _check_or_init_cutensornet()
    if __cutensornetDestroyContractionOptimizerInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyContractionOptimizerInfo is not found")
    return (<cutensornetStatus_t (*)(cutensornetContractionOptimizerInfo_t) noexcept nogil>__cutensornetDestroyContractionOptimizerInfo)(
        optimizerInfo)


cdef cutensornetStatus_t _cutensornetCreateContractionOptimizerInfo(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, cutensornetContractionOptimizerInfo_t* optimizerInfo) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateContractionOptimizerInfo
    _check_or_init_cutensornet()
    if __cutensornetCreateContractionOptimizerInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateContractionOptimizerInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, cutensornetContractionOptimizerInfo_t*) noexcept nogil>__cutensornetCreateContractionOptimizerInfo)(
        handle, networkDesc, optimizerInfo)


cdef cutensornetStatus_t _cutensornetContractionOptimize(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, const cutensornetContractionOptimizerConfig_t optimizerConfig, uint64_t workspaceSizeConstraint, cutensornetContractionOptimizerInfo_t optimizerInfo) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetContractionOptimize
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimize == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimize is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, const cutensornetContractionOptimizerConfig_t, uint64_t, cutensornetContractionOptimizerInfo_t) noexcept nogil>__cutensornetContractionOptimize)(
        handle, networkDesc, optimizerConfig, workspaceSizeConstraint, optimizerInfo)


cdef cutensornetStatus_t _cutensornetContractionOptimizerInfoGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetContractionOptimizerInfoAttributes_t attr, void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetContractionOptimizerInfoGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerInfoGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerInfoGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetContractionOptimizerInfo_t, cutensornetContractionOptimizerInfoAttributes_t, void*, size_t) noexcept nogil>__cutensornetContractionOptimizerInfoGetAttribute)(
        handle, optimizerInfo, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetContractionOptimizerInfoSetAttribute(const cutensornetHandle_t handle, cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetContractionOptimizerInfoAttributes_t attr, const void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetContractionOptimizerInfoSetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerInfoSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerInfoSetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionOptimizerInfo_t, cutensornetContractionOptimizerInfoAttributes_t, const void*, size_t) noexcept nogil>__cutensornetContractionOptimizerInfoSetAttribute)(
        handle, optimizerInfo, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetContractionOptimizerInfoGetPackedSize(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, size_t* sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetContractionOptimizerInfoGetPackedSize
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerInfoGetPackedSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerInfoGetPackedSize is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetContractionOptimizerInfo_t, size_t*) noexcept nogil>__cutensornetContractionOptimizerInfoGetPackedSize)(
        handle, optimizerInfo, sizeInBytes)


cdef cutensornetStatus_t _cutensornetContractionOptimizerInfoPackData(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetContractionOptimizerInfoPackData
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerInfoPackData == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerInfoPackData is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetContractionOptimizerInfo_t, void*, size_t) noexcept nogil>__cutensornetContractionOptimizerInfoPackData)(
        handle, optimizerInfo, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetCreateContractionOptimizerInfoFromPackedData(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, const void* buffer, size_t sizeInBytes, cutensornetContractionOptimizerInfo_t* optimizerInfo) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateContractionOptimizerInfoFromPackedData
    _check_or_init_cutensornet()
    if __cutensornetCreateContractionOptimizerInfoFromPackedData == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateContractionOptimizerInfoFromPackedData is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, const void*, size_t, cutensornetContractionOptimizerInfo_t*) noexcept nogil>__cutensornetCreateContractionOptimizerInfoFromPackedData)(
        handle, networkDesc, buffer, sizeInBytes, optimizerInfo)


cdef cutensornetStatus_t _cutensornetUpdateContractionOptimizerInfoFromPackedData(const cutensornetHandle_t handle, const void* buffer, size_t sizeInBytes, cutensornetContractionOptimizerInfo_t optimizerInfo) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetUpdateContractionOptimizerInfoFromPackedData
    _check_or_init_cutensornet()
    if __cutensornetUpdateContractionOptimizerInfoFromPackedData == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetUpdateContractionOptimizerInfoFromPackedData is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const void*, size_t, cutensornetContractionOptimizerInfo_t) noexcept nogil>__cutensornetUpdateContractionOptimizerInfoFromPackedData)(
        handle, buffer, sizeInBytes, optimizerInfo)


cdef cutensornetStatus_t _cutensornetCreateContractionPlan(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, const cutensornetContractionOptimizerInfo_t optimizerInfo, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetContractionPlan_t* plan) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateContractionPlan
    _check_or_init_cutensornet()
    if __cutensornetCreateContractionPlan == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateContractionPlan is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, const cutensornetContractionOptimizerInfo_t, const cutensornetWorkspaceDescriptor_t, cutensornetContractionPlan_t*) noexcept nogil>__cutensornetCreateContractionPlan)(
        handle, networkDesc, optimizerInfo, workDesc, plan)


cdef cutensornetStatus_t _cutensornetDestroyContractionPlan(cutensornetContractionPlan_t plan) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyContractionPlan
    _check_or_init_cutensornet()
    if __cutensornetDestroyContractionPlan == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyContractionPlan is not found")
    return (<cutensornetStatus_t (*)(cutensornetContractionPlan_t) noexcept nogil>__cutensornetDestroyContractionPlan)(
        plan)


cdef cutensornetStatus_t _cutensornetContractionAutotune(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan, const void* const rawDataIn[], void* rawDataOut, cutensornetWorkspaceDescriptor_t workDesc, const cutensornetContractionAutotunePreference_t pref, cudaStream_t stream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetContractionAutotune
    _check_or_init_cutensornet()
    if __cutensornetContractionAutotune == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionAutotune is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionPlan_t, const void* const*, void*, cutensornetWorkspaceDescriptor_t, const cutensornetContractionAutotunePreference_t, cudaStream_t) noexcept nogil>__cutensornetContractionAutotune)(
        handle, plan, rawDataIn, rawDataOut, workDesc, pref, stream)


cdef cutensornetStatus_t _cutensornetCreateContractionAutotunePreference(const cutensornetHandle_t handle, cutensornetContractionAutotunePreference_t* autotunePreference) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateContractionAutotunePreference
    _check_or_init_cutensornet()
    if __cutensornetCreateContractionAutotunePreference == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateContractionAutotunePreference is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionAutotunePreference_t*) noexcept nogil>__cutensornetCreateContractionAutotunePreference)(
        handle, autotunePreference)


cdef cutensornetStatus_t _cutensornetContractionAutotunePreferenceGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionAutotunePreference_t autotunePreference, cutensornetContractionAutotunePreferenceAttributes_t attr, void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetContractionAutotunePreferenceGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionAutotunePreferenceGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionAutotunePreferenceGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetContractionAutotunePreference_t, cutensornetContractionAutotunePreferenceAttributes_t, void*, size_t) noexcept nogil>__cutensornetContractionAutotunePreferenceGetAttribute)(
        handle, autotunePreference, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetContractionAutotunePreferenceSetAttribute(const cutensornetHandle_t handle, cutensornetContractionAutotunePreference_t autotunePreference, cutensornetContractionAutotunePreferenceAttributes_t attr, const void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetContractionAutotunePreferenceSetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionAutotunePreferenceSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionAutotunePreferenceSetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionAutotunePreference_t, cutensornetContractionAutotunePreferenceAttributes_t, const void*, size_t) noexcept nogil>__cutensornetContractionAutotunePreferenceSetAttribute)(
        handle, autotunePreference, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetDestroyContractionAutotunePreference(cutensornetContractionAutotunePreference_t autotunePreference) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyContractionAutotunePreference
    _check_or_init_cutensornet()
    if __cutensornetDestroyContractionAutotunePreference == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyContractionAutotunePreference is not found")
    return (<cutensornetStatus_t (*)(cutensornetContractionAutotunePreference_t) noexcept nogil>__cutensornetDestroyContractionAutotunePreference)(
        autotunePreference)


cdef cutensornetStatus_t _cutensornetCreateSliceGroupFromIDRange(const cutensornetHandle_t handle, int64_t sliceIdStart, int64_t sliceIdStop, int64_t sliceIdStep, cutensornetSliceGroup_t* sliceGroup) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateSliceGroupFromIDRange
    _check_or_init_cutensornet()
    if __cutensornetCreateSliceGroupFromIDRange == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateSliceGroupFromIDRange is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int64_t, int64_t, int64_t, cutensornetSliceGroup_t*) noexcept nogil>__cutensornetCreateSliceGroupFromIDRange)(
        handle, sliceIdStart, sliceIdStop, sliceIdStep, sliceGroup)


cdef cutensornetStatus_t _cutensornetCreateSliceGroupFromIDs(const cutensornetHandle_t handle, const int64_t* beginIDSequence, const int64_t* endIDSequence, cutensornetSliceGroup_t* sliceGroup) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateSliceGroupFromIDs
    _check_or_init_cutensornet()
    if __cutensornetCreateSliceGroupFromIDs == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateSliceGroupFromIDs is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const int64_t*, const int64_t*, cutensornetSliceGroup_t*) noexcept nogil>__cutensornetCreateSliceGroupFromIDs)(
        handle, beginIDSequence, endIDSequence, sliceGroup)


cdef cutensornetStatus_t _cutensornetDestroySliceGroup(cutensornetSliceGroup_t sliceGroup) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroySliceGroup
    _check_or_init_cutensornet()
    if __cutensornetDestroySliceGroup == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroySliceGroup is not found")
    return (<cutensornetStatus_t (*)(cutensornetSliceGroup_t) noexcept nogil>__cutensornetDestroySliceGroup)(
        sliceGroup)


cdef cutensornetStatus_t _cutensornetContractSlices(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan, const void* const rawDataIn[], void* rawDataOut, int32_t accumulateOutput, cutensornetWorkspaceDescriptor_t workDesc, const cutensornetSliceGroup_t sliceGroup, cudaStream_t stream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetContractSlices
    _check_or_init_cutensornet()
    if __cutensornetContractSlices == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractSlices is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionPlan_t, const void* const*, void*, int32_t, cutensornetWorkspaceDescriptor_t, const cutensornetSliceGroup_t, cudaStream_t) noexcept nogil>__cutensornetContractSlices)(
        handle, plan, rawDataIn, rawDataOut, accumulateOutput, workDesc, sliceGroup, stream)


cdef cutensornetStatus_t _cutensornetCreateTensorDescriptor(const cutensornetHandle_t handle, int32_t numModes, const int64_t extents[], const int64_t strides[], const int32_t modeLabels[], cudaDataType_t dataType, cutensornetTensorDescriptor_t* tensorDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateTensorDescriptor
    _check_or_init_cutensornet()
    if __cutensornetCreateTensorDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateTensorDescriptor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t, const int64_t*, const int64_t*, const int32_t*, cudaDataType_t, cutensornetTensorDescriptor_t*) noexcept nogil>__cutensornetCreateTensorDescriptor)(
        handle, numModes, extents, strides, modeLabels, dataType, tensorDesc)


cdef cutensornetStatus_t _cutensornetDestroyTensorDescriptor(cutensornetTensorDescriptor_t tensorDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyTensorDescriptor
    _check_or_init_cutensornet()
    if __cutensornetDestroyTensorDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyTensorDescriptor is not found")
    return (<cutensornetStatus_t (*)(cutensornetTensorDescriptor_t) noexcept nogil>__cutensornetDestroyTensorDescriptor)(
        tensorDesc)


cdef cutensornetStatus_t _cutensornetCreateTensorSVDConfig(const cutensornetHandle_t handle, cutensornetTensorSVDConfig_t* svdConfig) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateTensorSVDConfig
    _check_or_init_cutensornet()
    if __cutensornetCreateTensorSVDConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateTensorSVDConfig is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetTensorSVDConfig_t*) noexcept nogil>__cutensornetCreateTensorSVDConfig)(
        handle, svdConfig)


cdef cutensornetStatus_t _cutensornetDestroyTensorSVDConfig(cutensornetTensorSVDConfig_t svdConfig) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyTensorSVDConfig
    _check_or_init_cutensornet()
    if __cutensornetDestroyTensorSVDConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyTensorSVDConfig is not found")
    return (<cutensornetStatus_t (*)(cutensornetTensorSVDConfig_t) noexcept nogil>__cutensornetDestroyTensorSVDConfig)(
        svdConfig)


cdef cutensornetStatus_t _cutensornetTensorSVDConfigGetAttribute(const cutensornetHandle_t handle, const cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDConfigAttributes_t attr, void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetTensorSVDConfigGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetTensorSVDConfigGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetTensorSVDConfigGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorSVDConfig_t, cutensornetTensorSVDConfigAttributes_t, void*, size_t) noexcept nogil>__cutensornetTensorSVDConfigGetAttribute)(
        handle, svdConfig, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetTensorSVDConfigSetAttribute(const cutensornetHandle_t handle, cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDConfigAttributes_t attr, const void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetTensorSVDConfigSetAttribute
    _check_or_init_cutensornet()
    if __cutensornetTensorSVDConfigSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetTensorSVDConfigSetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetTensorSVDConfig_t, cutensornetTensorSVDConfigAttributes_t, const void*, size_t) noexcept nogil>__cutensornetTensorSVDConfigSetAttribute)(
        handle, svdConfig, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetWorkspaceComputeSVDSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const cutensornetTensorDescriptor_t descTensorU, const cutensornetTensorDescriptor_t descTensorV, const cutensornetTensorSVDConfig_t svdConfig, cutensornetWorkspaceDescriptor_t workDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetWorkspaceComputeSVDSizes
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceComputeSVDSizes == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceComputeSVDSizes is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorSVDConfig_t, cutensornetWorkspaceDescriptor_t) noexcept nogil>__cutensornetWorkspaceComputeSVDSizes)(
        handle, descTensorIn, descTensorU, descTensorV, svdConfig, workDesc)


cdef cutensornetStatus_t _cutensornetWorkspaceComputeQRSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const cutensornetTensorDescriptor_t descTensorQ, const cutensornetTensorDescriptor_t descTensorR, cutensornetWorkspaceDescriptor_t workDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetWorkspaceComputeQRSizes
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceComputeQRSizes == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceComputeQRSizes is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, cutensornetWorkspaceDescriptor_t) noexcept nogil>__cutensornetWorkspaceComputeQRSizes)(
        handle, descTensorIn, descTensorQ, descTensorR, workDesc)


cdef cutensornetStatus_t _cutensornetCreateTensorSVDInfo(const cutensornetHandle_t handle, cutensornetTensorSVDInfo_t* svdInfo) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateTensorSVDInfo
    _check_or_init_cutensornet()
    if __cutensornetCreateTensorSVDInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateTensorSVDInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetTensorSVDInfo_t*) noexcept nogil>__cutensornetCreateTensorSVDInfo)(
        handle, svdInfo)


cdef cutensornetStatus_t _cutensornetTensorSVDInfoGetAttribute(const cutensornetHandle_t handle, const cutensornetTensorSVDInfo_t svdInfo, cutensornetTensorSVDInfoAttributes_t attr, void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetTensorSVDInfoGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetTensorSVDInfoGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetTensorSVDInfoGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorSVDInfo_t, cutensornetTensorSVDInfoAttributes_t, void*, size_t) noexcept nogil>__cutensornetTensorSVDInfoGetAttribute)(
        handle, svdInfo, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetDestroyTensorSVDInfo(cutensornetTensorSVDInfo_t svdInfo) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyTensorSVDInfo
    _check_or_init_cutensornet()
    if __cutensornetDestroyTensorSVDInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyTensorSVDInfo is not found")
    return (<cutensornetStatus_t (*)(cutensornetTensorSVDInfo_t) noexcept nogil>__cutensornetDestroyTensorSVDInfo)(
        svdInfo)


cdef cutensornetStatus_t _cutensornetTensorSVD(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const void* const rawDataIn, cutensornetTensorDescriptor_t descTensorU, void* u, void* s, cutensornetTensorDescriptor_t descTensorV, void* v, const cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDInfo_t svdInfo, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetTensorSVD
    _check_or_init_cutensornet()
    if __cutensornetTensorSVD == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetTensorSVD is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const void* const, cutensornetTensorDescriptor_t, void*, void*, cutensornetTensorDescriptor_t, void*, const cutensornetTensorSVDConfig_t, cutensornetTensorSVDInfo_t, const cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetTensorSVD)(
        handle, descTensorIn, rawDataIn, descTensorU, u, s, descTensorV, v, svdConfig, svdInfo, workDesc, stream)


cdef cutensornetStatus_t _cutensornetTensorQR(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const void* const rawDataIn, const cutensornetTensorDescriptor_t descTensorQ, void* q, const cutensornetTensorDescriptor_t descTensorR, void* r, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetTensorQR
    _check_or_init_cutensornet()
    if __cutensornetTensorQR == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetTensorQR is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const void* const, const cutensornetTensorDescriptor_t, void*, const cutensornetTensorDescriptor_t, void*, const cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetTensorQR)(
        handle, descTensorIn, rawDataIn, descTensorQ, q, descTensorR, r, workDesc, stream)


cdef cutensornetStatus_t _cutensornetWorkspaceComputeGateSplitSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA, const cutensornetTensorDescriptor_t descTensorInB, const cutensornetTensorDescriptor_t descTensorInG, const cutensornetTensorDescriptor_t descTensorU, const cutensornetTensorDescriptor_t descTensorV, const cutensornetGateSplitAlgo_t gateAlgo, const cutensornetTensorSVDConfig_t svdConfig, cutensornetComputeType_t computeType, cutensornetWorkspaceDescriptor_t workDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetWorkspaceComputeGateSplitSizes
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceComputeGateSplitSizes == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceComputeGateSplitSizes is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetGateSplitAlgo_t, const cutensornetTensorSVDConfig_t, cutensornetComputeType_t, cutensornetWorkspaceDescriptor_t) noexcept nogil>__cutensornetWorkspaceComputeGateSplitSizes)(
        handle, descTensorInA, descTensorInB, descTensorInG, descTensorU, descTensorV, gateAlgo, svdConfig, computeType, workDesc)


cdef cutensornetStatus_t _cutensornetGateSplit(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA, const void* rawDataInA, const cutensornetTensorDescriptor_t descTensorInB, const void* rawDataInB, const cutensornetTensorDescriptor_t descTensorInG, const void* rawDataInG, cutensornetTensorDescriptor_t descTensorU, void* u, void* s, cutensornetTensorDescriptor_t descTensorV, void* v, const cutensornetGateSplitAlgo_t gateAlgo, const cutensornetTensorSVDConfig_t svdConfig, cutensornetComputeType_t computeType, cutensornetTensorSVDInfo_t svdInfo, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetGateSplit
    _check_or_init_cutensornet()
    if __cutensornetGateSplit == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGateSplit is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const void*, const cutensornetTensorDescriptor_t, const void*, const cutensornetTensorDescriptor_t, const void*, cutensornetTensorDescriptor_t, void*, void*, cutensornetTensorDescriptor_t, void*, const cutensornetGateSplitAlgo_t, const cutensornetTensorSVDConfig_t, cutensornetComputeType_t, cutensornetTensorSVDInfo_t, const cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetGateSplit)(
        handle, descTensorInA, rawDataInA, descTensorInB, rawDataInB, descTensorInG, rawDataInG, descTensorU, u, s, descTensorV, v, gateAlgo, svdConfig, computeType, svdInfo, workDesc, stream)


cdef cutensornetStatus_t _cutensornetGetDeviceMemHandler(const cutensornetHandle_t handle, cutensornetDeviceMemHandler_t* devMemHandler) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetGetDeviceMemHandler
    _check_or_init_cutensornet()
    if __cutensornetGetDeviceMemHandler == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetDeviceMemHandler is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetDeviceMemHandler_t*) noexcept nogil>__cutensornetGetDeviceMemHandler)(
        handle, devMemHandler)


cdef cutensornetStatus_t _cutensornetSetDeviceMemHandler(cutensornetHandle_t handle, const cutensornetDeviceMemHandler_t* devMemHandler) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetSetDeviceMemHandler
    _check_or_init_cutensornet()
    if __cutensornetSetDeviceMemHandler == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetSetDeviceMemHandler is not found")
    return (<cutensornetStatus_t (*)(cutensornetHandle_t, const cutensornetDeviceMemHandler_t*) noexcept nogil>__cutensornetSetDeviceMemHandler)(
        handle, devMemHandler)


cdef cutensornetStatus_t _cutensornetLoggerSetCallback(cutensornetLoggerCallback_t callback) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetLoggerSetCallback
    _check_or_init_cutensornet()
    if __cutensornetLoggerSetCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerSetCallback is not found")
    return (<cutensornetStatus_t (*)(cutensornetLoggerCallback_t) noexcept nogil>__cutensornetLoggerSetCallback)(
        callback)


cdef cutensornetStatus_t _cutensornetLoggerSetCallbackData(cutensornetLoggerCallbackData_t callback, void* userData) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetLoggerSetCallbackData
    _check_or_init_cutensornet()
    if __cutensornetLoggerSetCallbackData == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerSetCallbackData is not found")
    return (<cutensornetStatus_t (*)(cutensornetLoggerCallbackData_t, void*) noexcept nogil>__cutensornetLoggerSetCallbackData)(
        callback, userData)


cdef cutensornetStatus_t _cutensornetLoggerSetFile(FILE* file) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetLoggerSetFile
    _check_or_init_cutensornet()
    if __cutensornetLoggerSetFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerSetFile is not found")
    return (<cutensornetStatus_t (*)(FILE*) noexcept nogil>__cutensornetLoggerSetFile)(
        file)


cdef cutensornetStatus_t _cutensornetLoggerOpenFile(const char* logFile) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetLoggerOpenFile
    _check_or_init_cutensornet()
    if __cutensornetLoggerOpenFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerOpenFile is not found")
    return (<cutensornetStatus_t (*)(const char*) noexcept nogil>__cutensornetLoggerOpenFile)(
        logFile)


cdef cutensornetStatus_t _cutensornetLoggerSetLevel(int32_t level) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetLoggerSetLevel
    _check_or_init_cutensornet()
    if __cutensornetLoggerSetLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerSetLevel is not found")
    return (<cutensornetStatus_t (*)(int32_t) noexcept nogil>__cutensornetLoggerSetLevel)(
        level)


cdef cutensornetStatus_t _cutensornetLoggerSetMask(int32_t mask) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetLoggerSetMask
    _check_or_init_cutensornet()
    if __cutensornetLoggerSetMask == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerSetMask is not found")
    return (<cutensornetStatus_t (*)(int32_t) noexcept nogil>__cutensornetLoggerSetMask)(
        mask)


cdef cutensornetStatus_t _cutensornetLoggerForceDisable() except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetLoggerForceDisable
    _check_or_init_cutensornet()
    if __cutensornetLoggerForceDisable == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerForceDisable is not found")
    return (<cutensornetStatus_t (*)() noexcept nogil>__cutensornetLoggerForceDisable)(
        )


cdef size_t _cutensornetGetVersion() except?0 nogil:
    global __cutensornetGetVersion
    _check_or_init_cutensornet()
    if __cutensornetGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetVersion is not found")
    return (<size_t (*)() noexcept nogil>__cutensornetGetVersion)(
        )


cdef size_t _cutensornetGetCudartVersion() except?0 nogil:
    global __cutensornetGetCudartVersion
    _check_or_init_cutensornet()
    if __cutensornetGetCudartVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetCudartVersion is not found")
    return (<size_t (*)() noexcept nogil>__cutensornetGetCudartVersion)(
        )


cdef const char* _cutensornetGetErrorString(cutensornetStatus_t error) except?NULL nogil:
    global __cutensornetGetErrorString
    _check_or_init_cutensornet()
    if __cutensornetGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetErrorString is not found")
    return (<const char* (*)(cutensornetStatus_t) noexcept nogil>__cutensornetGetErrorString)(
        error)


cdef cutensornetStatus_t _cutensornetDistributedResetConfiguration(cutensornetHandle_t handle, const void* commPtr, size_t commSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDistributedResetConfiguration
    _check_or_init_cutensornet()
    if __cutensornetDistributedResetConfiguration == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDistributedResetConfiguration is not found")
    return (<cutensornetStatus_t (*)(cutensornetHandle_t, const void*, size_t) noexcept nogil>__cutensornetDistributedResetConfiguration)(
        handle, commPtr, commSize)


cdef cutensornetStatus_t _cutensornetDistributedGetNumRanks(const cutensornetHandle_t handle, int32_t* numRanks) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDistributedGetNumRanks
    _check_or_init_cutensornet()
    if __cutensornetDistributedGetNumRanks == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDistributedGetNumRanks is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t*) noexcept nogil>__cutensornetDistributedGetNumRanks)(
        handle, numRanks)


cdef cutensornetStatus_t _cutensornetDistributedGetProcRank(const cutensornetHandle_t handle, int32_t* procRank) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDistributedGetProcRank
    _check_or_init_cutensornet()
    if __cutensornetDistributedGetProcRank == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDistributedGetProcRank is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t*) noexcept nogil>__cutensornetDistributedGetProcRank)(
        handle, procRank)


cdef cutensornetStatus_t _cutensornetDistributedSynchronize(const cutensornetHandle_t handle) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDistributedSynchronize
    _check_or_init_cutensornet()
    if __cutensornetDistributedSynchronize == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDistributedSynchronize is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t) noexcept nogil>__cutensornetDistributedSynchronize)(
        handle)


cdef cutensornetStatus_t _cutensornetNetworkGetAttribute(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, cutensornetNetworkAttributes_t attr, void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetNetworkGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, cutensornetNetworkAttributes_t, void*, size_t) noexcept nogil>__cutensornetNetworkGetAttribute)(
        handle, networkDesc, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetNetworkSetAttribute(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, cutensornetNetworkAttributes_t attr, const void* const buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkSetAttribute
    _check_or_init_cutensornet()
    if __cutensornetNetworkSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkSetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, cutensornetNetworkAttributes_t, const void* const, size_t) noexcept nogil>__cutensornetNetworkSetAttribute)(
        handle, networkDesc, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetWorkspacePurgeCache(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetWorkspacePurgeCache
    _check_or_init_cutensornet()
    if __cutensornetWorkspacePurgeCache == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspacePurgeCache is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetWorkspaceDescriptor_t, cutensornetMemspace_t) noexcept nogil>__cutensornetWorkspacePurgeCache)(
        handle, workDesc, memSpace)


cdef cutensornetStatus_t _cutensornetCreateState(const cutensornetHandle_t handle, cutensornetStatePurity_t purity, int32_t numStateModes, const int64_t* stateModeExtents, cudaDataType_t dataType, cutensornetState_t* tensorNetworkState) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateState
    _check_or_init_cutensornet()
    if __cutensornetCreateState == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateState is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStatePurity_t, int32_t, const int64_t*, cudaDataType_t, cutensornetState_t*) noexcept nogil>__cutensornetCreateState)(
        handle, purity, numStateModes, stateModeExtents, dataType, tensorNetworkState)


cdef cutensornetStatus_t _cutensornetStateApplyTensor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateApplyTensor
    _check_or_init_cutensornet()
    if __cutensornetStateApplyTensor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyTensor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, void*, const int64_t*, const int32_t, const int32_t, const int32_t, int64_t*) noexcept nogil>__cutensornetStateApplyTensor)(
        handle, tensorNetworkState, numStateModes, stateModes, tensorData, tensorModeStrides, immutable, adjoint, unitary, tensorId)


cdef cutensornetStatus_t _cutensornetStateUpdateTensor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int64_t tensorId, void* tensorData, int32_t unitary) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateUpdateTensor
    _check_or_init_cutensornet()
    if __cutensornetStateUpdateTensor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateUpdateTensor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int64_t, void*, int32_t) noexcept nogil>__cutensornetStateUpdateTensor)(
        handle, tensorNetworkState, tensorId, tensorData, unitary)


cdef cutensornetStatus_t _cutensornetDestroyState(cutensornetState_t tensorNetworkState) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyState
    _check_or_init_cutensornet()
    if __cutensornetDestroyState == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyState is not found")
    return (<cutensornetStatus_t (*)(cutensornetState_t) noexcept nogil>__cutensornetDestroyState)(
        tensorNetworkState)


cdef cutensornetStatus_t _cutensornetCreateMarginal(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numMarginalModes, const int32_t* marginalModes, int32_t numProjectedModes, const int32_t* projectedModes, const int64_t* marginalTensorStrides, cutensornetStateMarginal_t* tensorNetworkMarginal) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateMarginal
    _check_or_init_cutensornet()
    if __cutensornetCreateMarginal == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateMarginal is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, int32_t, const int32_t*, const int64_t*, cutensornetStateMarginal_t*) noexcept nogil>__cutensornetCreateMarginal)(
        handle, tensorNetworkState, numMarginalModes, marginalModes, numProjectedModes, projectedModes, marginalTensorStrides, tensorNetworkMarginal)


cdef cutensornetStatus_t _cutensornetMarginalConfigure(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, cutensornetMarginalAttributes_t attribute, const void* attributeValue, size_t attributeSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetMarginalConfigure
    _check_or_init_cutensornet()
    if __cutensornetMarginalConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetMarginalConfigure is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateMarginal_t, cutensornetMarginalAttributes_t, const void*, size_t) noexcept nogil>__cutensornetMarginalConfigure)(
        handle, tensorNetworkMarginal, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetMarginalPrepare(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetMarginalPrepare
    _check_or_init_cutensornet()
    if __cutensornetMarginalPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetMarginalPrepare is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateMarginal_t, size_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetMarginalPrepare)(
        handle, tensorNetworkMarginal, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetMarginalCompute(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, const int64_t* projectedModeValues, cutensornetWorkspaceDescriptor_t workDesc, void* marginalTensor, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetMarginalCompute
    _check_or_init_cutensornet()
    if __cutensornetMarginalCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetMarginalCompute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateMarginal_t, const int64_t*, cutensornetWorkspaceDescriptor_t, void*, cudaStream_t) noexcept nogil>__cutensornetMarginalCompute)(
        handle, tensorNetworkMarginal, projectedModeValues, workDesc, marginalTensor, cudaStream)


cdef cutensornetStatus_t _cutensornetDestroyMarginal(cutensornetStateMarginal_t tensorNetworkMarginal) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyMarginal
    _check_or_init_cutensornet()
    if __cutensornetDestroyMarginal == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyMarginal is not found")
    return (<cutensornetStatus_t (*)(cutensornetStateMarginal_t) noexcept nogil>__cutensornetDestroyMarginal)(
        tensorNetworkMarginal)


cdef cutensornetStatus_t _cutensornetCreateSampler(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numModesToSample, const int32_t* modesToSample, cutensornetStateSampler_t* tensorNetworkSampler) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateSampler
    _check_or_init_cutensornet()
    if __cutensornetCreateSampler == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateSampler is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, cutensornetStateSampler_t*) noexcept nogil>__cutensornetCreateSampler)(
        handle, tensorNetworkState, numModesToSample, modesToSample, tensorNetworkSampler)


cdef cutensornetStatus_t _cutensornetSamplerConfigure(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, cutensornetSamplerAttributes_t attribute, const void* attributeValue, size_t attributeSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetSamplerConfigure
    _check_or_init_cutensornet()
    if __cutensornetSamplerConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetSamplerConfigure is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateSampler_t, cutensornetSamplerAttributes_t, const void*, size_t) noexcept nogil>__cutensornetSamplerConfigure)(
        handle, tensorNetworkSampler, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetSamplerPrepare(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetSamplerPrepare
    _check_or_init_cutensornet()
    if __cutensornetSamplerPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetSamplerPrepare is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateSampler_t, size_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetSamplerPrepare)(
        handle, tensorNetworkSampler, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetSamplerSample(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, int64_t numShots, cutensornetWorkspaceDescriptor_t workDesc, int64_t* samples, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetSamplerSample
    _check_or_init_cutensornet()
    if __cutensornetSamplerSample == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetSamplerSample is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateSampler_t, int64_t, cutensornetWorkspaceDescriptor_t, int64_t*, cudaStream_t) noexcept nogil>__cutensornetSamplerSample)(
        handle, tensorNetworkSampler, numShots, workDesc, samples, cudaStream)


cdef cutensornetStatus_t _cutensornetDestroySampler(cutensornetStateSampler_t tensorNetworkSampler) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroySampler
    _check_or_init_cutensornet()
    if __cutensornetDestroySampler == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroySampler is not found")
    return (<cutensornetStatus_t (*)(cutensornetStateSampler_t) noexcept nogil>__cutensornetDestroySampler)(
        tensorNetworkSampler)


cdef cutensornetStatus_t _cutensornetStateFinalizeMPS(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetBoundaryCondition_t boundaryCondition, const int64_t* const extentsOut[], const int64_t* const stridesOut[]) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateFinalizeMPS
    _check_or_init_cutensornet()
    if __cutensornetStateFinalizeMPS == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateFinalizeMPS is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, cutensornetBoundaryCondition_t, const int64_t* const*, const int64_t* const*) noexcept nogil>__cutensornetStateFinalizeMPS)(
        handle, tensorNetworkState, boundaryCondition, extentsOut, stridesOut)


cdef cutensornetStatus_t _cutensornetStateConfigure(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetStateAttributes_t attribute, const void* attributeValue, size_t attributeSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateConfigure
    _check_or_init_cutensornet()
    if __cutensornetStateConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateConfigure is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, cutensornetStateAttributes_t, const void*, size_t) noexcept nogil>__cutensornetStateConfigure)(
        handle, tensorNetworkState, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetStatePrepare(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStatePrepare
    _check_or_init_cutensornet()
    if __cutensornetStatePrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStatePrepare is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, size_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetStatePrepare)(
        handle, tensorNetworkState, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetStateCompute(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetWorkspaceDescriptor_t workDesc, int64_t* extentsOut[], int64_t* stridesOut[], void* stateTensorsOut[], cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateCompute
    _check_or_init_cutensornet()
    if __cutensornetStateCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateCompute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, cutensornetWorkspaceDescriptor_t, int64_t**, int64_t**, void**, cudaStream_t) noexcept nogil>__cutensornetStateCompute)(
        handle, tensorNetworkState, workDesc, extentsOut, stridesOut, stateTensorsOut, cudaStream)


cdef cutensornetStatus_t _cutensornetGetOutputStateDetails(const cutensornetHandle_t handle, const cutensornetState_t tensorNetworkState, int32_t* numTensorsOut, int32_t numModesOut[], int64_t* extentsOut[], int64_t* stridesOut[]) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetGetOutputStateDetails
    _check_or_init_cutensornet()
    if __cutensornetGetOutputStateDetails == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetOutputStateDetails is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetState_t, int32_t*, int32_t*, int64_t**, int64_t**) noexcept nogil>__cutensornetGetOutputStateDetails)(
        handle, tensorNetworkState, numTensorsOut, numModesOut, extentsOut, stridesOut)


cdef cutensornetStatus_t _cutensornetCreateNetworkOperator(const cutensornetHandle_t handle, int32_t numStateModes, const int64_t stateModeExtents[], cudaDataType_t dataType, cutensornetNetworkOperator_t* tensorNetworkOperator) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateNetworkOperator
    _check_or_init_cutensornet()
    if __cutensornetCreateNetworkOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateNetworkOperator is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t, const int64_t*, cudaDataType_t, cutensornetNetworkOperator_t*) noexcept nogil>__cutensornetCreateNetworkOperator)(
        handle, numStateModes, stateModeExtents, dataType, tensorNetworkOperator)


cdef cutensornetStatus_t _cutensornetNetworkOperatorAppendProduct(const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator, cuDoubleComplex coefficient, int32_t numTensors, const int32_t numStateModes[], const int32_t* stateModes[], const int64_t* tensorModeStrides[], const void* tensorData[], int64_t* componentId) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkOperatorAppendProduct
    _check_or_init_cutensornet()
    if __cutensornetNetworkOperatorAppendProduct == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkOperatorAppendProduct is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkOperator_t, cuDoubleComplex, int32_t, const int32_t*, const int32_t**, const int64_t**, const void**, int64_t*) noexcept nogil>__cutensornetNetworkOperatorAppendProduct)(
        handle, tensorNetworkOperator, coefficient, numTensors, numStateModes, stateModes, tensorModeStrides, tensorData, componentId)


cdef cutensornetStatus_t _cutensornetDestroyNetworkOperator(cutensornetNetworkOperator_t tensorNetworkOperator) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyNetworkOperator
    _check_or_init_cutensornet()
    if __cutensornetDestroyNetworkOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyNetworkOperator is not found")
    return (<cutensornetStatus_t (*)(cutensornetNetworkOperator_t) noexcept nogil>__cutensornetDestroyNetworkOperator)(
        tensorNetworkOperator)


cdef cutensornetStatus_t _cutensornetCreateAccessor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numProjectedModes, const int32_t* projectedModes, const int64_t* amplitudesTensorStrides, cutensornetStateAccessor_t* tensorNetworkAccessor) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateAccessor
    _check_or_init_cutensornet()
    if __cutensornetCreateAccessor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateAccessor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, const int64_t*, cutensornetStateAccessor_t*) noexcept nogil>__cutensornetCreateAccessor)(
        handle, tensorNetworkState, numProjectedModes, projectedModes, amplitudesTensorStrides, tensorNetworkAccessor)


cdef cutensornetStatus_t _cutensornetAccessorConfigure(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, cutensornetAccessorAttributes_t attribute, const void* attributeValue, size_t attributeSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetAccessorConfigure
    _check_or_init_cutensornet()
    if __cutensornetAccessorConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetAccessorConfigure is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateAccessor_t, cutensornetAccessorAttributes_t, const void*, size_t) noexcept nogil>__cutensornetAccessorConfigure)(
        handle, tensorNetworkAccessor, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetAccessorPrepare(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetAccessorPrepare
    _check_or_init_cutensornet()
    if __cutensornetAccessorPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetAccessorPrepare is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateAccessor_t, size_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetAccessorPrepare)(
        handle, tensorNetworkAccessor, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetAccessorCompute(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, const int64_t* projectedModeValues, cutensornetWorkspaceDescriptor_t workDesc, void* amplitudesTensor, void* stateNorm, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetAccessorCompute
    _check_or_init_cutensornet()
    if __cutensornetAccessorCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetAccessorCompute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateAccessor_t, const int64_t*, cutensornetWorkspaceDescriptor_t, void*, void*, cudaStream_t) noexcept nogil>__cutensornetAccessorCompute)(
        handle, tensorNetworkAccessor, projectedModeValues, workDesc, amplitudesTensor, stateNorm, cudaStream)


cdef cutensornetStatus_t _cutensornetDestroyAccessor(cutensornetStateAccessor_t tensorNetworkAccessor) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyAccessor
    _check_or_init_cutensornet()
    if __cutensornetDestroyAccessor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyAccessor is not found")
    return (<cutensornetStatus_t (*)(cutensornetStateAccessor_t) noexcept nogil>__cutensornetDestroyAccessor)(
        tensorNetworkAccessor)


cdef cutensornetStatus_t _cutensornetCreateExpectation(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetNetworkOperator_t tensorNetworkOperator, cutensornetStateExpectation_t* tensorNetworkExpectation) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateExpectation
    _check_or_init_cutensornet()
    if __cutensornetCreateExpectation == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateExpectation is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, cutensornetNetworkOperator_t, cutensornetStateExpectation_t*) noexcept nogil>__cutensornetCreateExpectation)(
        handle, tensorNetworkState, tensorNetworkOperator, tensorNetworkExpectation)


cdef cutensornetStatus_t _cutensornetExpectationConfigure(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetExpectationAttributes_t attribute, const void* attributeValue, size_t attributeSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetExpectationConfigure
    _check_or_init_cutensornet()
    if __cutensornetExpectationConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetExpectationConfigure is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateExpectation_t, cutensornetExpectationAttributes_t, const void*, size_t) noexcept nogil>__cutensornetExpectationConfigure)(
        handle, tensorNetworkExpectation, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetExpectationPrepare(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetExpectationPrepare
    _check_or_init_cutensornet()
    if __cutensornetExpectationPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetExpectationPrepare is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateExpectation_t, size_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetExpectationPrepare)(
        handle, tensorNetworkExpectation, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetExpectationCompute(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetWorkspaceDescriptor_t workDesc, void* expectationValue, void* stateNorm, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetExpectationCompute
    _check_or_init_cutensornet()
    if __cutensornetExpectationCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetExpectationCompute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateExpectation_t, cutensornetWorkspaceDescriptor_t, void*, void*, cudaStream_t) noexcept nogil>__cutensornetExpectationCompute)(
        handle, tensorNetworkExpectation, workDesc, expectationValue, stateNorm, cudaStream)


cdef cutensornetStatus_t _cutensornetDestroyExpectation(cutensornetStateExpectation_t tensorNetworkExpectation) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyExpectation
    _check_or_init_cutensornet()
    if __cutensornetDestroyExpectation == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyExpectation is not found")
    return (<cutensornetStatus_t (*)(cutensornetStateExpectation_t) noexcept nogil>__cutensornetDestroyExpectation)(
        tensorNetworkExpectation)


cdef cutensornetStatus_t _cutensornetStateApplyTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateApplyTensorOperator
    _check_or_init_cutensornet()
    if __cutensornetStateApplyTensorOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyTensorOperator is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, void*, const int64_t*, const int32_t, const int32_t, const int32_t, int64_t*) noexcept nogil>__cutensornetStateApplyTensorOperator)(
        handle, tensorNetworkState, numStateModes, stateModes, tensorData, tensorModeStrides, immutable, adjoint, unitary, tensorId)


cdef cutensornetStatus_t _cutensornetStateApplyControlledTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numControlModes, const int32_t* stateControlModes, const int64_t* stateControlValues, int32_t numTargetModes, const int32_t* stateTargetModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateApplyControlledTensorOperator
    _check_or_init_cutensornet()
    if __cutensornetStateApplyControlledTensorOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyControlledTensorOperator is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, const int64_t*, int32_t, const int32_t*, void*, const int64_t*, const int32_t, const int32_t, const int32_t, int64_t*) noexcept nogil>__cutensornetStateApplyControlledTensorOperator)(
        handle, tensorNetworkState, numControlModes, stateControlModes, stateControlValues, numTargetModes, stateTargetModes, tensorData, tensorModeStrides, immutable, adjoint, unitary, tensorId)


cdef cutensornetStatus_t _cutensornetStateUpdateTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int64_t tensorId, void* tensorData, int32_t unitary) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateUpdateTensorOperator
    _check_or_init_cutensornet()
    if __cutensornetStateUpdateTensorOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateUpdateTensorOperator is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int64_t, void*, int32_t) noexcept nogil>__cutensornetStateUpdateTensorOperator)(
        handle, tensorNetworkState, tensorId, tensorData, unitary)


cdef cutensornetStatus_t _cutensornetStateApplyNetworkOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, const cutensornetNetworkOperator_t tensorNetworkOperator, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* operatorId) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateApplyNetworkOperator
    _check_or_init_cutensornet()
    if __cutensornetStateApplyNetworkOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyNetworkOperator is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, const cutensornetNetworkOperator_t, const int32_t, const int32_t, const int32_t, int64_t*) noexcept nogil>__cutensornetStateApplyNetworkOperator)(
        handle, tensorNetworkState, tensorNetworkOperator, immutable, adjoint, unitary, operatorId)


cdef cutensornetStatus_t _cutensornetStateInitializeMPS(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetBoundaryCondition_t boundaryCondition, const int64_t* const extentsIn[], const int64_t* const stridesIn[], void* stateTensorsIn[]) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateInitializeMPS
    _check_or_init_cutensornet()
    if __cutensornetStateInitializeMPS == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateInitializeMPS is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, cutensornetBoundaryCondition_t, const int64_t* const*, const int64_t* const*, void**) noexcept nogil>__cutensornetStateInitializeMPS)(
        handle, tensorNetworkState, boundaryCondition, extentsIn, stridesIn, stateTensorsIn)


cdef cutensornetStatus_t _cutensornetStateGetInfo(const cutensornetHandle_t handle, const cutensornetState_t tensorNetworkState, cutensornetStateAttributes_t attribute, void* attributeValue, size_t attributeSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateGetInfo
    _check_or_init_cutensornet()
    if __cutensornetStateGetInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateGetInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetState_t, cutensornetStateAttributes_t, void*, size_t) noexcept nogil>__cutensornetStateGetInfo)(
        handle, tensorNetworkState, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetNetworkOperatorAppendMPO(const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator, cuDoubleComplex coefficient, int32_t numStateModes, const int32_t stateModes[], const int64_t* tensorModeExtents[], const int64_t* tensorModeStrides[], const void* tensorData[], cutensornetBoundaryCondition_t boundaryCondition, int64_t* componentId) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkOperatorAppendMPO
    _check_or_init_cutensornet()
    if __cutensornetNetworkOperatorAppendMPO == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkOperatorAppendMPO is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkOperator_t, cuDoubleComplex, int32_t, const int32_t*, const int64_t**, const int64_t**, const void**, cutensornetBoundaryCondition_t, int64_t*) noexcept nogil>__cutensornetNetworkOperatorAppendMPO)(
        handle, tensorNetworkOperator, coefficient, numStateModes, stateModes, tensorModeExtents, tensorModeStrides, tensorData, boundaryCondition, componentId)


cdef cutensornetStatus_t _cutensornetAccessorGetInfo(const cutensornetHandle_t handle, const cutensornetStateAccessor_t tensorNetworkAccessor, cutensornetAccessorAttributes_t attribute, void* attributeValue, size_t attributeSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetAccessorGetInfo
    _check_or_init_cutensornet()
    if __cutensornetAccessorGetInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetAccessorGetInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetStateAccessor_t, cutensornetAccessorAttributes_t, void*, size_t) noexcept nogil>__cutensornetAccessorGetInfo)(
        handle, tensorNetworkAccessor, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetExpectationGetInfo(const cutensornetHandle_t handle, const cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetExpectationAttributes_t attribute, void* attributeValue, size_t attributeSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetExpectationGetInfo
    _check_or_init_cutensornet()
    if __cutensornetExpectationGetInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetExpectationGetInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetStateExpectation_t, cutensornetExpectationAttributes_t, void*, size_t) noexcept nogil>__cutensornetExpectationGetInfo)(
        handle, tensorNetworkExpectation, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetMarginalGetInfo(const cutensornetHandle_t handle, const cutensornetStateMarginal_t tensorNetworkMarginal, cutensornetMarginalAttributes_t attribute, void* attributeValue, size_t attributeSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetMarginalGetInfo
    _check_or_init_cutensornet()
    if __cutensornetMarginalGetInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetMarginalGetInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetStateMarginal_t, cutensornetMarginalAttributes_t, void*, size_t) noexcept nogil>__cutensornetMarginalGetInfo)(
        handle, tensorNetworkMarginal, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetSamplerGetInfo(const cutensornetHandle_t handle, const cutensornetStateSampler_t tensorNetworkSampler, cutensornetSamplerAttributes_t attribute, void* attributeValue, size_t attributeSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetSamplerGetInfo
    _check_or_init_cutensornet()
    if __cutensornetSamplerGetInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetSamplerGetInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetStateSampler_t, cutensornetSamplerAttributes_t, void*, size_t) noexcept nogil>__cutensornetSamplerGetInfo)(
        handle, tensorNetworkSampler, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetStateApplyUnitaryChannel(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, int32_t numTensors, void* tensorData[], const int64_t* tensorModeStrides, const double probabilities[], int64_t* channelId) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateApplyUnitaryChannel
    _check_or_init_cutensornet()
    if __cutensornetStateApplyUnitaryChannel == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyUnitaryChannel is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, int32_t, void**, const int64_t*, const double*, int64_t*) noexcept nogil>__cutensornetStateApplyUnitaryChannel)(
        handle, tensorNetworkState, numStateModes, stateModes, numTensors, tensorData, tensorModeStrides, probabilities, channelId)


cdef cutensornetStatus_t _cutensornetStateCaptureMPS(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateCaptureMPS
    _check_or_init_cutensornet()
    if __cutensornetStateCaptureMPS == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateCaptureMPS is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t) noexcept nogil>__cutensornetStateCaptureMPS)(
        handle, tensorNetworkState)


cdef cutensornetStatus_t _cutensornetStateApplyGeneralChannel(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, int32_t numTensors, void* tensorData[], const int64_t* tensorModeStrides, int64_t* channelId) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateApplyGeneralChannel
    _check_or_init_cutensornet()
    if __cutensornetStateApplyGeneralChannel == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyGeneralChannel is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, int32_t, void**, const int64_t*, int64_t*) noexcept nogil>__cutensornetStateApplyGeneralChannel)(
        handle, tensorNetworkState, numStateModes, stateModes, numTensors, tensorData, tensorModeStrides, channelId)


cdef cutensornetStatus_t _cutensornetCreateStateProjectionMPS(const cutensornetHandle_t handle, int32_t numStates, const cutensornetState_t tensorNetworkStates[], const cuDoubleComplex coeffs[], int32_t symmetric, int32_t numEnvs, const cutensornetMPSEnvBounds_t specEnvs[], cutensornetBoundaryCondition_t boundaryCondition, int32_t numTensors, const int32_t quditsPerTensor[], const int64_t* extentsOut[], const int64_t* stridesOut[], void* dualTensorsDataOut[], const cutensornetMPSEnvBounds_t* orthoSpec, cutensornetStateProjectionMPS_t* tensorNetworkProjection) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateStateProjectionMPS
    _check_or_init_cutensornet()
    if __cutensornetCreateStateProjectionMPS == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateStateProjectionMPS is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t, const cutensornetState_t*, const cuDoubleComplex*, int32_t, int32_t, const cutensornetMPSEnvBounds_t*, cutensornetBoundaryCondition_t, int32_t, const int32_t*, const int64_t**, const int64_t**, void**, const cutensornetMPSEnvBounds_t*, cutensornetStateProjectionMPS_t*) noexcept nogil>__cutensornetCreateStateProjectionMPS)(
        handle, numStates, tensorNetworkStates, coeffs, symmetric, numEnvs, specEnvs, boundaryCondition, numTensors, quditsPerTensor, extentsOut, stridesOut, dualTensorsDataOut, orthoSpec, tensorNetworkProjection)


cdef cutensornetStatus_t _cutensornetStateProjectionMPSConfigure(const cutensornetHandle_t handle, cutensornetStateProjectionMPS_t tensorNetworkProjection, cutensornetStateProjectionMPSAttributes_t attribute, const void* attributeValue, size_t attributeSize) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateProjectionMPSConfigure
    _check_or_init_cutensornet()
    if __cutensornetStateProjectionMPSConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateProjectionMPSConfigure is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateProjectionMPS_t, cutensornetStateProjectionMPSAttributes_t, const void*, size_t) noexcept nogil>__cutensornetStateProjectionMPSConfigure)(
        handle, tensorNetworkProjection, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetStateProjectionMPSPrepare(const cutensornetHandle_t handle, cutensornetStateProjectionMPS_t tensorNetworkProjection, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateProjectionMPSPrepare
    _check_or_init_cutensornet()
    if __cutensornetStateProjectionMPSPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateProjectionMPSPrepare is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateProjectionMPS_t, size_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetStateProjectionMPSPrepare)(
        handle, tensorNetworkProjection, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetStateProjectionMPSComputeTensorEnv(const cutensornetHandle_t handle, cutensornetStateProjectionMPS_t tensorNetworkProjection, const cutensornetMPSEnvBounds_t* envSpec, const int64_t stridesIn[], const void* envTensorDataIn, const int64_t stridesOut[], void* envTensorDataOut, int32_t applyInvMetric, int32_t reResolveChannels, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateProjectionMPSComputeTensorEnv
    _check_or_init_cutensornet()
    if __cutensornetStateProjectionMPSComputeTensorEnv == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateProjectionMPSComputeTensorEnv is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateProjectionMPS_t, const cutensornetMPSEnvBounds_t*, const int64_t*, const void*, const int64_t*, void*, int32_t, int32_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetStateProjectionMPSComputeTensorEnv)(
        handle, tensorNetworkProjection, envSpec, stridesIn, envTensorDataIn, stridesOut, envTensorDataOut, applyInvMetric, reResolveChannels, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetStateProjectionMPSGetTensorInfo(const cutensornetHandle_t handle, const cutensornetStateProjectionMPS_t tensorNetworkProjection, const cutensornetMPSEnvBounds_t* envSpec, int64_t extents[], int64_t recommendedStrides[]) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateProjectionMPSGetTensorInfo
    _check_or_init_cutensornet()
    if __cutensornetStateProjectionMPSGetTensorInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateProjectionMPSGetTensorInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetStateProjectionMPS_t, const cutensornetMPSEnvBounds_t*, int64_t*, int64_t*) noexcept nogil>__cutensornetStateProjectionMPSGetTensorInfo)(
        handle, tensorNetworkProjection, envSpec, extents, recommendedStrides)


cdef cutensornetStatus_t _cutensornetStateProjectionMPSExtractTensor(const cutensornetHandle_t handle, cutensornetStateProjectionMPS_t tensorNetworkProjection, const cutensornetMPSEnvBounds_t* envSpec, const int64_t strides[], void* envTensorData, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateProjectionMPSExtractTensor
    _check_or_init_cutensornet()
    if __cutensornetStateProjectionMPSExtractTensor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateProjectionMPSExtractTensor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateProjectionMPS_t, const cutensornetMPSEnvBounds_t*, const int64_t*, void*, cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetStateProjectionMPSExtractTensor)(
        handle, tensorNetworkProjection, envSpec, strides, envTensorData, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetStateProjectionMPSInsertTensor(const cutensornetHandle_t handle, cutensornetStateProjectionMPS_t tensorNetworkProjection, const cutensornetMPSEnvBounds_t* envSpec, const cutensornetMPSEnvBounds_t* orthoSpec, const int64_t strides[], const void* envTensorData, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateProjectionMPSInsertTensor
    _check_or_init_cutensornet()
    if __cutensornetStateProjectionMPSInsertTensor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateProjectionMPSInsertTensor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateProjectionMPS_t, const cutensornetMPSEnvBounds_t*, const cutensornetMPSEnvBounds_t*, const int64_t*, const void*, cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetStateProjectionMPSInsertTensor)(
        handle, tensorNetworkProjection, envSpec, orthoSpec, strides, envTensorData, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetDestroyStateProjectionMPS(cutensornetStateProjectionMPS_t tensorNetworkProjection) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyStateProjectionMPS
    _check_or_init_cutensornet()
    if __cutensornetDestroyStateProjectionMPS == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyStateProjectionMPS is not found")
    return (<cutensornetStatus_t (*)(cutensornetStateProjectionMPS_t) noexcept nogil>__cutensornetDestroyStateProjectionMPS)(
        tensorNetworkProjection)


cdef cutensornetStatus_t _cutensornetCreateNetwork(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t* networkDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateNetwork
    _check_or_init_cutensornet()
    if __cutensornetCreateNetwork == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateNetwork is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t*) noexcept nogil>__cutensornetCreateNetwork)(
        handle, networkDesc)


cdef cutensornetStatus_t _cutensornetDestroyNetwork(cutensornetNetworkDescriptor_t networkDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyNetwork
    _check_or_init_cutensornet()
    if __cutensornetDestroyNetwork == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyNetwork is not found")
    return (<cutensornetStatus_t (*)(cutensornetNetworkDescriptor_t) noexcept nogil>__cutensornetDestroyNetwork)(
        networkDesc)


cdef cutensornetStatus_t _cutensornetNetworkAppendTensor(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, int32_t numModes, const int64_t extents[], const int32_t modeLabels[], const cutensornetTensorQualifiers_t* const qualifiers, cudaDataType_t dataType, int64_t* tensorId) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkAppendTensor
    _check_or_init_cutensornet()
    if __cutensornetNetworkAppendTensor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkAppendTensor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, int32_t, const int64_t*, const int32_t*, const cutensornetTensorQualifiers_t* const, cudaDataType_t, int64_t*) noexcept nogil>__cutensornetNetworkAppendTensor)(
        handle, networkDesc, numModes, extents, modeLabels, qualifiers, dataType, tensorId)


cdef cutensornetStatus_t _cutensornetNetworkSetOutputTensor(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, int32_t numModes, const int32_t modeLabels[], cudaDataType_t dataType) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkSetOutputTensor
    _check_or_init_cutensornet()
    if __cutensornetNetworkSetOutputTensor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkSetOutputTensor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, int32_t, const int32_t*, cudaDataType_t) noexcept nogil>__cutensornetNetworkSetOutputTensor)(
        handle, networkDesc, numModes, modeLabels, dataType)


cdef cutensornetStatus_t _cutensornetNetworkSetOptimizerInfo(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, const cutensornetContractionOptimizerInfo_t optimizerInfo) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkSetOptimizerInfo
    _check_or_init_cutensornet()
    if __cutensornetNetworkSetOptimizerInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkSetOptimizerInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, const cutensornetContractionOptimizerInfo_t) noexcept nogil>__cutensornetNetworkSetOptimizerInfo)(
        handle, networkDesc, optimizerInfo)


cdef cutensornetStatus_t _cutensornetNetworkPrepareContraction(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, const cutensornetWorkspaceDescriptor_t workDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkPrepareContraction
    _check_or_init_cutensornet()
    if __cutensornetNetworkPrepareContraction == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkPrepareContraction is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, const cutensornetWorkspaceDescriptor_t) noexcept nogil>__cutensornetNetworkPrepareContraction)(
        handle, networkDesc, workDesc)


cdef cutensornetStatus_t _cutensornetNetworkAutotuneContraction(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, const cutensornetWorkspaceDescriptor_t workDesc, const cutensornetNetworkAutotunePreference_t pref, cudaStream_t stream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkAutotuneContraction
    _check_or_init_cutensornet()
    if __cutensornetNetworkAutotuneContraction == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkAutotuneContraction is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, const cutensornetWorkspaceDescriptor_t, const cutensornetNetworkAutotunePreference_t, cudaStream_t) noexcept nogil>__cutensornetNetworkAutotuneContraction)(
        handle, networkDesc, workDesc, pref, stream)


cdef cutensornetStatus_t _cutensornetCreateNetworkAutotunePreference(const cutensornetHandle_t handle, cutensornetNetworkAutotunePreference_t* autotunePreference) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateNetworkAutotunePreference
    _check_or_init_cutensornet()
    if __cutensornetCreateNetworkAutotunePreference == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateNetworkAutotunePreference is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkAutotunePreference_t*) noexcept nogil>__cutensornetCreateNetworkAutotunePreference)(
        handle, autotunePreference)


cdef cutensornetStatus_t _cutensornetNetworkAutotunePreferenceGetAttribute(const cutensornetHandle_t handle, const cutensornetNetworkAutotunePreference_t autotunePreference, cutensornetNetworkAutotunePreferenceAttributes_t attr, void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkAutotunePreferenceGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetNetworkAutotunePreferenceGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkAutotunePreferenceGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkAutotunePreference_t, cutensornetNetworkAutotunePreferenceAttributes_t, void*, size_t) noexcept nogil>__cutensornetNetworkAutotunePreferenceGetAttribute)(
        handle, autotunePreference, attr, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetNetworkAutotunePreferenceSetAttribute(const cutensornetHandle_t handle, cutensornetNetworkAutotunePreference_t autotunePreference, cutensornetNetworkAutotunePreferenceAttributes_t attr, const void* buf, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkAutotunePreferenceSetAttribute
    _check_or_init_cutensornet()
    if __cutensornetNetworkAutotunePreferenceSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkAutotunePreferenceSetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkAutotunePreference_t, cutensornetNetworkAutotunePreferenceAttributes_t, const void*, size_t) noexcept nogil>__cutensornetNetworkAutotunePreferenceSetAttribute)(
        handle, autotunePreference, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetDestroyNetworkAutotunePreference(cutensornetNetworkAutotunePreference_t autotunePreference) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyNetworkAutotunePreference
    _check_or_init_cutensornet()
    if __cutensornetDestroyNetworkAutotunePreference == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyNetworkAutotunePreference is not found")
    return (<cutensornetStatus_t (*)(cutensornetNetworkAutotunePreference_t) noexcept nogil>__cutensornetDestroyNetworkAutotunePreference)(
        autotunePreference)


cdef cutensornetStatus_t _cutensornetNetworkSetInputTensorMemory(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, int64_t tensorId, const void* const buffer, const int64_t strides[]) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkSetInputTensorMemory
    _check_or_init_cutensornet()
    if __cutensornetNetworkSetInputTensorMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkSetInputTensorMemory is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, int64_t, const void* const, const int64_t*) noexcept nogil>__cutensornetNetworkSetInputTensorMemory)(
        handle, networkDesc, tensorId, buffer, strides)


cdef cutensornetStatus_t _cutensornetNetworkSetOutputTensorMemory(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, void* const buffer, const int64_t strides[]) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkSetOutputTensorMemory
    _check_or_init_cutensornet()
    if __cutensornetNetworkSetOutputTensorMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkSetOutputTensorMemory is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, void* const, const int64_t*) noexcept nogil>__cutensornetNetworkSetOutputTensorMemory)(
        handle, networkDesc, buffer, strides)


cdef cutensornetStatus_t _cutensornetNetworkSetGradientTensorMemory(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, int64_t correspondingTensorId, void* const buffer, const int64_t strides[]) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkSetGradientTensorMemory
    _check_or_init_cutensornet()
    if __cutensornetNetworkSetGradientTensorMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkSetGradientTensorMemory is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, int64_t, void* const, const int64_t*) noexcept nogil>__cutensornetNetworkSetGradientTensorMemory)(
        handle, networkDesc, correspondingTensorId, buffer, strides)


cdef cutensornetStatus_t _cutensornetNetworkSetAdjointTensorMemory(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, const void* const buffer, const int64_t strides[]) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkSetAdjointTensorMemory
    _check_or_init_cutensornet()
    if __cutensornetNetworkSetAdjointTensorMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkSetAdjointTensorMemory is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, const void* const, const int64_t*) noexcept nogil>__cutensornetNetworkSetAdjointTensorMemory)(
        handle, networkDesc, buffer, strides)


cdef cutensornetStatus_t _cutensornetNetworkContract(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, int32_t accumulateOutput, const cutensornetWorkspaceDescriptor_t workDesc, const cutensornetSliceGroup_t sliceGroup, cudaStream_t stream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkContract
    _check_or_init_cutensornet()
    if __cutensornetNetworkContract == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkContract is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, int32_t, const cutensornetWorkspaceDescriptor_t, const cutensornetSliceGroup_t, cudaStream_t) noexcept nogil>__cutensornetNetworkContract)(
        handle, networkDesc, accumulateOutput, workDesc, sliceGroup, stream)


cdef cutensornetStatus_t _cutensornetNetworkPrepareGradientsBackward(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, const cutensornetWorkspaceDescriptor_t workDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkPrepareGradientsBackward
    _check_or_init_cutensornet()
    if __cutensornetNetworkPrepareGradientsBackward == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkPrepareGradientsBackward is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, const cutensornetWorkspaceDescriptor_t) noexcept nogil>__cutensornetNetworkPrepareGradientsBackward)(
        handle, networkDesc, workDesc)


cdef cutensornetStatus_t _cutensornetNetworkComputeGradientsBackward(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, int32_t accumulateOutput, const cutensornetWorkspaceDescriptor_t workDesc, const cutensornetSliceGroup_t sliceGroup, cudaStream_t stream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetNetworkComputeGradientsBackward
    _check_or_init_cutensornet()
    if __cutensornetNetworkComputeGradientsBackward == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkComputeGradientsBackward is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, int32_t, const cutensornetWorkspaceDescriptor_t, const cutensornetSliceGroup_t, cudaStream_t) noexcept nogil>__cutensornetNetworkComputeGradientsBackward)(
        handle, networkDesc, accumulateOutput, workDesc, sliceGroup, stream)


cdef cutensornetStatus_t _cutensornetStateApplyDiagonalTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateApplyDiagonalTensorOperator
    _check_or_init_cutensornet()
    if __cutensornetStateApplyDiagonalTensorOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyDiagonalTensorOperator is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, void*, const int64_t*, const int32_t, const int32_t, const int32_t, int64_t*) noexcept nogil>__cutensornetStateApplyDiagonalTensorOperator)(
        handle, tensorNetworkState, numStateModes, stateModes, tensorData, tensorModeStrides, immutable, adjoint, unitary, tensorId)


cdef cutensornetStatus_t _cutensornetStateApplyTensorOperatorWithGradient(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, void* gradientData, const int64_t* gradientModeStrides, int64_t* tensorId) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateApplyTensorOperatorWithGradient
    _check_or_init_cutensornet()
    if __cutensornetStateApplyTensorOperatorWithGradient == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyTensorOperatorWithGradient is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, void*, const int64_t*, const int32_t, const int32_t, const int32_t, void*, const int64_t*, int64_t*) noexcept nogil>__cutensornetStateApplyTensorOperatorWithGradient)(
        handle, tensorNetworkState, numStateModes, stateModes, tensorData, tensorModeStrides, immutable, adjoint, unitary, gradientData, gradientModeStrides, tensorId)


cdef cutensornetStatus_t _cutensornetStateUpdateTensorOperatorGradient(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int64_t tensorId, void* gradientData) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateUpdateTensorOperatorGradient
    _check_or_init_cutensornet()
    if __cutensornetStateUpdateTensorOperatorGradient == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateUpdateTensorOperatorGradient is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int64_t, void*) noexcept nogil>__cutensornetStateUpdateTensorOperatorGradient)(
        handle, tensorNetworkState, tensorId, gradientData)


cdef cutensornetStatus_t _cutensornetExpectationComputeWithGradientsBackward(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, int32_t accumulateGradients, const void* expectationValueAdjoint, const void* stateNormAdjoint, cutensornetWorkspaceDescriptor_t workDesc, void* expectationValue, void* stateNorm, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetExpectationComputeWithGradientsBackward
    _check_or_init_cutensornet()
    if __cutensornetExpectationComputeWithGradientsBackward == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetExpectationComputeWithGradientsBackward is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateExpectation_t, int32_t, const void*, const void*, cutensornetWorkspaceDescriptor_t, void*, void*, cudaStream_t) noexcept nogil>__cutensornetExpectationComputeWithGradientsBackward)(
        handle, tensorNetworkExpectation, accumulateGradients, expectationValueAdjoint, stateNormAdjoint, workDesc, expectationValue, stateNorm, cudaStream)


cdef cutensornetStatus_t _cutensornetStateProjectionMPSUpdateCoefficients(const cutensornetHandle_t handle, cutensornetStateProjectionMPS_t tensorNetworkProjection, int32_t numCoeffs, const cuDoubleComplex coeffs[]) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateProjectionMPSUpdateCoefficients
    _check_or_init_cutensornet()
    if __cutensornetStateProjectionMPSUpdateCoefficients == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateProjectionMPSUpdateCoefficients is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateProjectionMPS_t, int32_t, const cuDoubleComplex*) noexcept nogil>__cutensornetStateProjectionMPSUpdateCoefficients)(
        handle, tensorNetworkProjection, numCoeffs, coeffs)


cdef cutensornetStatus_t _cutensornetStateProjectionMPSUpdateDualTensors(const cutensornetHandle_t handle, cutensornetStateProjectionMPS_t tensorNetworkProjection, const int64_t* maxExtents[], const int64_t* validExtents[], const int64_t* strides[], void* dualTensorsData[], const cutensornetMPSEnvBounds_t* orthoSpec, cudaStream_t cudaStream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetStateProjectionMPSUpdateDualTensors
    _check_or_init_cutensornet()
    if __cutensornetStateProjectionMPSUpdateDualTensors == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateProjectionMPSUpdateDualTensors is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateProjectionMPS_t, const int64_t**, const int64_t**, const int64_t**, void**, const cutensornetMPSEnvBounds_t*, cudaStream_t) noexcept nogil>__cutensornetStateProjectionMPSUpdateDualTensors)(
        handle, tensorNetworkProjection, maxExtents, validExtents, strides, dualTensorsData, orthoSpec, cudaStream)


cdef const char* _cutensornetGetLastError() except?NULL nogil:
    global __cutensornetGetLastError
    _check_or_init_cutensornet()
    if __cutensornetGetLastError == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetLastError is not found")
    return (<const char* (*)() noexcept nogil>__cutensornetGetLastError)(
        )


cdef cutensornetStatus_t _cutensornetCreateMarginalDiagonal(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numMarginalModes, const int32_t* marginalModes, int32_t numProjectedModes, const int32_t* projectedModes, const int64_t* marginalDiagonalTensorStrides, cutensornetStateMarginal_t* tensorNetworkMarginal) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateMarginalDiagonal
    _check_or_init_cutensornet()
    if __cutensornetCreateMarginalDiagonal == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateMarginalDiagonal is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, int32_t, const int32_t*, const int64_t*, cutensornetStateMarginal_t*) noexcept nogil>__cutensornetCreateMarginalDiagonal)(
        handle, tensorNetworkState, numMarginalModes, marginalModes, numProjectedModes, projectedModes, marginalDiagonalTensorStrides, tensorNetworkMarginal)


cdef cutensornetStatus_t _cutensornetCreateDistributedTensorDescriptor(const cutensornetHandle_t handle, int32_t numModes, const int64_t extents[], const int64_t elementStrides[], const int64_t blockSizes[], const int64_t blockStrides[], const int64_t nranksPerMode[], const int32_t modeLabels[], cudaDataType_t dataType, cutensornetTensorDescriptor_t* tensorDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateDistributedTensorDescriptor
    _check_or_init_cutensornet()
    if __cutensornetCreateDistributedTensorDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateDistributedTensorDescriptor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t, const int64_t*, const int64_t*, const int64_t*, const int64_t*, const int64_t*, const int32_t*, cudaDataType_t, cutensornetTensorDescriptor_t*) noexcept nogil>__cutensornetCreateDistributedTensorDescriptor)(
        handle, numModes, extents, elementStrides, blockSizes, blockStrides, nranksPerMode, modeLabels, dataType, tensorDesc)


cdef cutensornetStatus_t _cutensornetCreateBinaryTensorContraction(cutensornetHandle_t handle, cutensornetTensorDescriptor_t descA, cutensornetTensorDescriptor_t descB, cutensornetTensorDescriptor_t descC, cutensornetTensorDescriptor_t descD, cutensornetComputeType_t computeType, cutensornetBinaryTensorContraction_t* contraction) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetCreateBinaryTensorContraction
    _check_or_init_cutensornet()
    if __cutensornetCreateBinaryTensorContraction == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateBinaryTensorContraction is not found")
    return (<cutensornetStatus_t (*)(cutensornetHandle_t, cutensornetTensorDescriptor_t, cutensornetTensorDescriptor_t, cutensornetTensorDescriptor_t, cutensornetTensorDescriptor_t, cutensornetComputeType_t, cutensornetBinaryTensorContraction_t*) noexcept nogil>__cutensornetCreateBinaryTensorContraction)(
        handle, descA, descB, descC, descD, computeType, contraction)


cdef cutensornetStatus_t _cutensornetBinaryTensorContractionPrepare(cutensornetHandle_t handle, cutensornetBinaryTensorContraction_t contraction, cutensornetWorkspaceDescriptor_t workDesc) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetBinaryTensorContractionPrepare
    _check_or_init_cutensornet()
    if __cutensornetBinaryTensorContractionPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetBinaryTensorContractionPrepare is not found")
    return (<cutensornetStatus_t (*)(cutensornetHandle_t, cutensornetBinaryTensorContraction_t, cutensornetWorkspaceDescriptor_t) noexcept nogil>__cutensornetBinaryTensorContractionPrepare)(
        handle, contraction, workDesc)


cdef cutensornetStatus_t _cutensornetBinaryTensorContractionCompute(cutensornetHandle_t handle, cutensornetBinaryTensorContraction_t contraction, const void* alpha, const void* A, const void* B, const void* beta, const void* C, void* D, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetBinaryTensorContractionCompute
    _check_or_init_cutensornet()
    if __cutensornetBinaryTensorContractionCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetBinaryTensorContractionCompute is not found")
    return (<cutensornetStatus_t (*)(cutensornetHandle_t, cutensornetBinaryTensorContraction_t, const void*, const void*, const void*, const void*, const void*, void*, cutensornetWorkspaceDescriptor_t, cudaStream_t) noexcept nogil>__cutensornetBinaryTensorContractionCompute)(
        handle, contraction, alpha, A, B, beta, C, D, workDesc, stream)


cdef cutensornetStatus_t _cutensornetDestroyBinaryTensorContraction(cutensornetBinaryTensorContraction_t contraction) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetDestroyBinaryTensorContraction
    _check_or_init_cutensornet()
    if __cutensornetDestroyBinaryTensorContraction == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyBinaryTensorContraction is not found")
    return (<cutensornetStatus_t (*)(cutensornetBinaryTensorContraction_t) noexcept nogil>__cutensornetDestroyBinaryTensorContraction)(
        contraction)


cdef cutensornetStatus_t _cutensornetTensorDescriptorGetAttribute(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t tensorDesc, cutensornetTensorDescriptorAttributes_t attr, void* buffer, size_t sizeInBytes) except?_CUTENSORNETSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensornetTensorDescriptorGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetTensorDescriptorGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetTensorDescriptorGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, cutensornetTensorDescriptorAttributes_t, void*, size_t) noexcept nogil>__cutensornetTensorDescriptorGetAttribute)(
        handle, tensorDesc, attr, buffer, sizeInBytes)
