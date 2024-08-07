# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 24.08.0. Do not modify it directly.

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

cdef bint __py_cutensornet_init = False

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
cdef void* __cutensornetComputeGradientsBackward = NULL
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


cdef void* load_library() except* nogil:
    cdef void* handle
    handle = dlopen("libcutensornet.so.2", RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        with gil:
            err_msg = dlerror()
            raise RuntimeError(f'Failed to dlopen libcutensornet ({err_msg.decode()})')
    return handle


cdef int _check_or_init_cutensornet() except -1 nogil:
    global __py_cutensornet_init
    if __py_cutensornet_init:
        return 0

    # Load function
    cdef void* handle = NULL
    global __cutensornetCreate
    __cutensornetCreate = dlsym(RTLD_DEFAULT, 'cutensornetCreate')
    if __cutensornetCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreate = dlsym(handle, 'cutensornetCreate')
    
    global __cutensornetDestroy
    __cutensornetDestroy = dlsym(RTLD_DEFAULT, 'cutensornetDestroy')
    if __cutensornetDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroy = dlsym(handle, 'cutensornetDestroy')
    
    global __cutensornetCreateNetworkDescriptor
    __cutensornetCreateNetworkDescriptor = dlsym(RTLD_DEFAULT, 'cutensornetCreateNetworkDescriptor')
    if __cutensornetCreateNetworkDescriptor == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateNetworkDescriptor = dlsym(handle, 'cutensornetCreateNetworkDescriptor')
    
    global __cutensornetDestroyNetworkDescriptor
    __cutensornetDestroyNetworkDescriptor = dlsym(RTLD_DEFAULT, 'cutensornetDestroyNetworkDescriptor')
    if __cutensornetDestroyNetworkDescriptor == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyNetworkDescriptor = dlsym(handle, 'cutensornetDestroyNetworkDescriptor')
    
    global __cutensornetGetOutputTensorDescriptor
    __cutensornetGetOutputTensorDescriptor = dlsym(RTLD_DEFAULT, 'cutensornetGetOutputTensorDescriptor')
    if __cutensornetGetOutputTensorDescriptor == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetGetOutputTensorDescriptor = dlsym(handle, 'cutensornetGetOutputTensorDescriptor')
    
    global __cutensornetGetTensorDetails
    __cutensornetGetTensorDetails = dlsym(RTLD_DEFAULT, 'cutensornetGetTensorDetails')
    if __cutensornetGetTensorDetails == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetGetTensorDetails = dlsym(handle, 'cutensornetGetTensorDetails')
    
    global __cutensornetCreateWorkspaceDescriptor
    __cutensornetCreateWorkspaceDescriptor = dlsym(RTLD_DEFAULT, 'cutensornetCreateWorkspaceDescriptor')
    if __cutensornetCreateWorkspaceDescriptor == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateWorkspaceDescriptor = dlsym(handle, 'cutensornetCreateWorkspaceDescriptor')
    
    global __cutensornetWorkspaceComputeContractionSizes
    __cutensornetWorkspaceComputeContractionSizes = dlsym(RTLD_DEFAULT, 'cutensornetWorkspaceComputeContractionSizes')
    if __cutensornetWorkspaceComputeContractionSizes == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetWorkspaceComputeContractionSizes = dlsym(handle, 'cutensornetWorkspaceComputeContractionSizes')
    
    global __cutensornetWorkspaceGetMemorySize
    __cutensornetWorkspaceGetMemorySize = dlsym(RTLD_DEFAULT, 'cutensornetWorkspaceGetMemorySize')
    if __cutensornetWorkspaceGetMemorySize == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetWorkspaceGetMemorySize = dlsym(handle, 'cutensornetWorkspaceGetMemorySize')
    
    global __cutensornetWorkspaceSetMemory
    __cutensornetWorkspaceSetMemory = dlsym(RTLD_DEFAULT, 'cutensornetWorkspaceSetMemory')
    if __cutensornetWorkspaceSetMemory == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetWorkspaceSetMemory = dlsym(handle, 'cutensornetWorkspaceSetMemory')
    
    global __cutensornetWorkspaceGetMemory
    __cutensornetWorkspaceGetMemory = dlsym(RTLD_DEFAULT, 'cutensornetWorkspaceGetMemory')
    if __cutensornetWorkspaceGetMemory == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetWorkspaceGetMemory = dlsym(handle, 'cutensornetWorkspaceGetMemory')
    
    global __cutensornetDestroyWorkspaceDescriptor
    __cutensornetDestroyWorkspaceDescriptor = dlsym(RTLD_DEFAULT, 'cutensornetDestroyWorkspaceDescriptor')
    if __cutensornetDestroyWorkspaceDescriptor == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyWorkspaceDescriptor = dlsym(handle, 'cutensornetDestroyWorkspaceDescriptor')
    
    global __cutensornetCreateContractionOptimizerConfig
    __cutensornetCreateContractionOptimizerConfig = dlsym(RTLD_DEFAULT, 'cutensornetCreateContractionOptimizerConfig')
    if __cutensornetCreateContractionOptimizerConfig == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateContractionOptimizerConfig = dlsym(handle, 'cutensornetCreateContractionOptimizerConfig')
    
    global __cutensornetDestroyContractionOptimizerConfig
    __cutensornetDestroyContractionOptimizerConfig = dlsym(RTLD_DEFAULT, 'cutensornetDestroyContractionOptimizerConfig')
    if __cutensornetDestroyContractionOptimizerConfig == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyContractionOptimizerConfig = dlsym(handle, 'cutensornetDestroyContractionOptimizerConfig')
    
    global __cutensornetContractionOptimizerConfigGetAttribute
    __cutensornetContractionOptimizerConfigGetAttribute = dlsym(RTLD_DEFAULT, 'cutensornetContractionOptimizerConfigGetAttribute')
    if __cutensornetContractionOptimizerConfigGetAttribute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetContractionOptimizerConfigGetAttribute = dlsym(handle, 'cutensornetContractionOptimizerConfigGetAttribute')
    
    global __cutensornetContractionOptimizerConfigSetAttribute
    __cutensornetContractionOptimizerConfigSetAttribute = dlsym(RTLD_DEFAULT, 'cutensornetContractionOptimizerConfigSetAttribute')
    if __cutensornetContractionOptimizerConfigSetAttribute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetContractionOptimizerConfigSetAttribute = dlsym(handle, 'cutensornetContractionOptimizerConfigSetAttribute')
    
    global __cutensornetDestroyContractionOptimizerInfo
    __cutensornetDestroyContractionOptimizerInfo = dlsym(RTLD_DEFAULT, 'cutensornetDestroyContractionOptimizerInfo')
    if __cutensornetDestroyContractionOptimizerInfo == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyContractionOptimizerInfo = dlsym(handle, 'cutensornetDestroyContractionOptimizerInfo')
    
    global __cutensornetCreateContractionOptimizerInfo
    __cutensornetCreateContractionOptimizerInfo = dlsym(RTLD_DEFAULT, 'cutensornetCreateContractionOptimizerInfo')
    if __cutensornetCreateContractionOptimizerInfo == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateContractionOptimizerInfo = dlsym(handle, 'cutensornetCreateContractionOptimizerInfo')
    
    global __cutensornetContractionOptimize
    __cutensornetContractionOptimize = dlsym(RTLD_DEFAULT, 'cutensornetContractionOptimize')
    if __cutensornetContractionOptimize == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetContractionOptimize = dlsym(handle, 'cutensornetContractionOptimize')
    
    global __cutensornetContractionOptimizerInfoGetAttribute
    __cutensornetContractionOptimizerInfoGetAttribute = dlsym(RTLD_DEFAULT, 'cutensornetContractionOptimizerInfoGetAttribute')
    if __cutensornetContractionOptimizerInfoGetAttribute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetContractionOptimizerInfoGetAttribute = dlsym(handle, 'cutensornetContractionOptimizerInfoGetAttribute')
    
    global __cutensornetContractionOptimizerInfoSetAttribute
    __cutensornetContractionOptimizerInfoSetAttribute = dlsym(RTLD_DEFAULT, 'cutensornetContractionOptimizerInfoSetAttribute')
    if __cutensornetContractionOptimizerInfoSetAttribute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetContractionOptimizerInfoSetAttribute = dlsym(handle, 'cutensornetContractionOptimizerInfoSetAttribute')
    
    global __cutensornetContractionOptimizerInfoGetPackedSize
    __cutensornetContractionOptimizerInfoGetPackedSize = dlsym(RTLD_DEFAULT, 'cutensornetContractionOptimizerInfoGetPackedSize')
    if __cutensornetContractionOptimizerInfoGetPackedSize == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetContractionOptimizerInfoGetPackedSize = dlsym(handle, 'cutensornetContractionOptimizerInfoGetPackedSize')
    
    global __cutensornetContractionOptimizerInfoPackData
    __cutensornetContractionOptimizerInfoPackData = dlsym(RTLD_DEFAULT, 'cutensornetContractionOptimizerInfoPackData')
    if __cutensornetContractionOptimizerInfoPackData == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetContractionOptimizerInfoPackData = dlsym(handle, 'cutensornetContractionOptimizerInfoPackData')
    
    global __cutensornetCreateContractionOptimizerInfoFromPackedData
    __cutensornetCreateContractionOptimizerInfoFromPackedData = dlsym(RTLD_DEFAULT, 'cutensornetCreateContractionOptimizerInfoFromPackedData')
    if __cutensornetCreateContractionOptimizerInfoFromPackedData == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateContractionOptimizerInfoFromPackedData = dlsym(handle, 'cutensornetCreateContractionOptimizerInfoFromPackedData')
    
    global __cutensornetUpdateContractionOptimizerInfoFromPackedData
    __cutensornetUpdateContractionOptimizerInfoFromPackedData = dlsym(RTLD_DEFAULT, 'cutensornetUpdateContractionOptimizerInfoFromPackedData')
    if __cutensornetUpdateContractionOptimizerInfoFromPackedData == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetUpdateContractionOptimizerInfoFromPackedData = dlsym(handle, 'cutensornetUpdateContractionOptimizerInfoFromPackedData')
    
    global __cutensornetCreateContractionPlan
    __cutensornetCreateContractionPlan = dlsym(RTLD_DEFAULT, 'cutensornetCreateContractionPlan')
    if __cutensornetCreateContractionPlan == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateContractionPlan = dlsym(handle, 'cutensornetCreateContractionPlan')
    
    global __cutensornetDestroyContractionPlan
    __cutensornetDestroyContractionPlan = dlsym(RTLD_DEFAULT, 'cutensornetDestroyContractionPlan')
    if __cutensornetDestroyContractionPlan == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyContractionPlan = dlsym(handle, 'cutensornetDestroyContractionPlan')
    
    global __cutensornetContractionAutotune
    __cutensornetContractionAutotune = dlsym(RTLD_DEFAULT, 'cutensornetContractionAutotune')
    if __cutensornetContractionAutotune == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetContractionAutotune = dlsym(handle, 'cutensornetContractionAutotune')
    
    global __cutensornetCreateContractionAutotunePreference
    __cutensornetCreateContractionAutotunePreference = dlsym(RTLD_DEFAULT, 'cutensornetCreateContractionAutotunePreference')
    if __cutensornetCreateContractionAutotunePreference == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateContractionAutotunePreference = dlsym(handle, 'cutensornetCreateContractionAutotunePreference')
    
    global __cutensornetContractionAutotunePreferenceGetAttribute
    __cutensornetContractionAutotunePreferenceGetAttribute = dlsym(RTLD_DEFAULT, 'cutensornetContractionAutotunePreferenceGetAttribute')
    if __cutensornetContractionAutotunePreferenceGetAttribute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetContractionAutotunePreferenceGetAttribute = dlsym(handle, 'cutensornetContractionAutotunePreferenceGetAttribute')
    
    global __cutensornetContractionAutotunePreferenceSetAttribute
    __cutensornetContractionAutotunePreferenceSetAttribute = dlsym(RTLD_DEFAULT, 'cutensornetContractionAutotunePreferenceSetAttribute')
    if __cutensornetContractionAutotunePreferenceSetAttribute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetContractionAutotunePreferenceSetAttribute = dlsym(handle, 'cutensornetContractionAutotunePreferenceSetAttribute')
    
    global __cutensornetDestroyContractionAutotunePreference
    __cutensornetDestroyContractionAutotunePreference = dlsym(RTLD_DEFAULT, 'cutensornetDestroyContractionAutotunePreference')
    if __cutensornetDestroyContractionAutotunePreference == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyContractionAutotunePreference = dlsym(handle, 'cutensornetDestroyContractionAutotunePreference')
    
    global __cutensornetCreateSliceGroupFromIDRange
    __cutensornetCreateSliceGroupFromIDRange = dlsym(RTLD_DEFAULT, 'cutensornetCreateSliceGroupFromIDRange')
    if __cutensornetCreateSliceGroupFromIDRange == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateSliceGroupFromIDRange = dlsym(handle, 'cutensornetCreateSliceGroupFromIDRange')
    
    global __cutensornetCreateSliceGroupFromIDs
    __cutensornetCreateSliceGroupFromIDs = dlsym(RTLD_DEFAULT, 'cutensornetCreateSliceGroupFromIDs')
    if __cutensornetCreateSliceGroupFromIDs == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateSliceGroupFromIDs = dlsym(handle, 'cutensornetCreateSliceGroupFromIDs')
    
    global __cutensornetDestroySliceGroup
    __cutensornetDestroySliceGroup = dlsym(RTLD_DEFAULT, 'cutensornetDestroySliceGroup')
    if __cutensornetDestroySliceGroup == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroySliceGroup = dlsym(handle, 'cutensornetDestroySliceGroup')
    
    global __cutensornetContractSlices
    __cutensornetContractSlices = dlsym(RTLD_DEFAULT, 'cutensornetContractSlices')
    if __cutensornetContractSlices == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetContractSlices = dlsym(handle, 'cutensornetContractSlices')
    
    global __cutensornetCreateTensorDescriptor
    __cutensornetCreateTensorDescriptor = dlsym(RTLD_DEFAULT, 'cutensornetCreateTensorDescriptor')
    if __cutensornetCreateTensorDescriptor == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateTensorDescriptor = dlsym(handle, 'cutensornetCreateTensorDescriptor')
    
    global __cutensornetDestroyTensorDescriptor
    __cutensornetDestroyTensorDescriptor = dlsym(RTLD_DEFAULT, 'cutensornetDestroyTensorDescriptor')
    if __cutensornetDestroyTensorDescriptor == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyTensorDescriptor = dlsym(handle, 'cutensornetDestroyTensorDescriptor')
    
    global __cutensornetCreateTensorSVDConfig
    __cutensornetCreateTensorSVDConfig = dlsym(RTLD_DEFAULT, 'cutensornetCreateTensorSVDConfig')
    if __cutensornetCreateTensorSVDConfig == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateTensorSVDConfig = dlsym(handle, 'cutensornetCreateTensorSVDConfig')
    
    global __cutensornetDestroyTensorSVDConfig
    __cutensornetDestroyTensorSVDConfig = dlsym(RTLD_DEFAULT, 'cutensornetDestroyTensorSVDConfig')
    if __cutensornetDestroyTensorSVDConfig == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyTensorSVDConfig = dlsym(handle, 'cutensornetDestroyTensorSVDConfig')
    
    global __cutensornetTensorSVDConfigGetAttribute
    __cutensornetTensorSVDConfigGetAttribute = dlsym(RTLD_DEFAULT, 'cutensornetTensorSVDConfigGetAttribute')
    if __cutensornetTensorSVDConfigGetAttribute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetTensorSVDConfigGetAttribute = dlsym(handle, 'cutensornetTensorSVDConfigGetAttribute')
    
    global __cutensornetTensorSVDConfigSetAttribute
    __cutensornetTensorSVDConfigSetAttribute = dlsym(RTLD_DEFAULT, 'cutensornetTensorSVDConfigSetAttribute')
    if __cutensornetTensorSVDConfigSetAttribute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetTensorSVDConfigSetAttribute = dlsym(handle, 'cutensornetTensorSVDConfigSetAttribute')
    
    global __cutensornetWorkspaceComputeSVDSizes
    __cutensornetWorkspaceComputeSVDSizes = dlsym(RTLD_DEFAULT, 'cutensornetWorkspaceComputeSVDSizes')
    if __cutensornetWorkspaceComputeSVDSizes == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetWorkspaceComputeSVDSizes = dlsym(handle, 'cutensornetWorkspaceComputeSVDSizes')
    
    global __cutensornetWorkspaceComputeQRSizes
    __cutensornetWorkspaceComputeQRSizes = dlsym(RTLD_DEFAULT, 'cutensornetWorkspaceComputeQRSizes')
    if __cutensornetWorkspaceComputeQRSizes == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetWorkspaceComputeQRSizes = dlsym(handle, 'cutensornetWorkspaceComputeQRSizes')
    
    global __cutensornetCreateTensorSVDInfo
    __cutensornetCreateTensorSVDInfo = dlsym(RTLD_DEFAULT, 'cutensornetCreateTensorSVDInfo')
    if __cutensornetCreateTensorSVDInfo == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateTensorSVDInfo = dlsym(handle, 'cutensornetCreateTensorSVDInfo')
    
    global __cutensornetTensorSVDInfoGetAttribute
    __cutensornetTensorSVDInfoGetAttribute = dlsym(RTLD_DEFAULT, 'cutensornetTensorSVDInfoGetAttribute')
    if __cutensornetTensorSVDInfoGetAttribute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetTensorSVDInfoGetAttribute = dlsym(handle, 'cutensornetTensorSVDInfoGetAttribute')
    
    global __cutensornetDestroyTensorSVDInfo
    __cutensornetDestroyTensorSVDInfo = dlsym(RTLD_DEFAULT, 'cutensornetDestroyTensorSVDInfo')
    if __cutensornetDestroyTensorSVDInfo == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyTensorSVDInfo = dlsym(handle, 'cutensornetDestroyTensorSVDInfo')
    
    global __cutensornetTensorSVD
    __cutensornetTensorSVD = dlsym(RTLD_DEFAULT, 'cutensornetTensorSVD')
    if __cutensornetTensorSVD == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetTensorSVD = dlsym(handle, 'cutensornetTensorSVD')
    
    global __cutensornetTensorQR
    __cutensornetTensorQR = dlsym(RTLD_DEFAULT, 'cutensornetTensorQR')
    if __cutensornetTensorQR == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetTensorQR = dlsym(handle, 'cutensornetTensorQR')
    
    global __cutensornetWorkspaceComputeGateSplitSizes
    __cutensornetWorkspaceComputeGateSplitSizes = dlsym(RTLD_DEFAULT, 'cutensornetWorkspaceComputeGateSplitSizes')
    if __cutensornetWorkspaceComputeGateSplitSizes == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetWorkspaceComputeGateSplitSizes = dlsym(handle, 'cutensornetWorkspaceComputeGateSplitSizes')
    
    global __cutensornetGateSplit
    __cutensornetGateSplit = dlsym(RTLD_DEFAULT, 'cutensornetGateSplit')
    if __cutensornetGateSplit == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetGateSplit = dlsym(handle, 'cutensornetGateSplit')
    
    global __cutensornetGetDeviceMemHandler
    __cutensornetGetDeviceMemHandler = dlsym(RTLD_DEFAULT, 'cutensornetGetDeviceMemHandler')
    if __cutensornetGetDeviceMemHandler == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetGetDeviceMemHandler = dlsym(handle, 'cutensornetGetDeviceMemHandler')
    
    global __cutensornetSetDeviceMemHandler
    __cutensornetSetDeviceMemHandler = dlsym(RTLD_DEFAULT, 'cutensornetSetDeviceMemHandler')
    if __cutensornetSetDeviceMemHandler == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetSetDeviceMemHandler = dlsym(handle, 'cutensornetSetDeviceMemHandler')
    
    global __cutensornetLoggerSetCallback
    __cutensornetLoggerSetCallback = dlsym(RTLD_DEFAULT, 'cutensornetLoggerSetCallback')
    if __cutensornetLoggerSetCallback == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetLoggerSetCallback = dlsym(handle, 'cutensornetLoggerSetCallback')
    
    global __cutensornetLoggerSetCallbackData
    __cutensornetLoggerSetCallbackData = dlsym(RTLD_DEFAULT, 'cutensornetLoggerSetCallbackData')
    if __cutensornetLoggerSetCallbackData == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetLoggerSetCallbackData = dlsym(handle, 'cutensornetLoggerSetCallbackData')
    
    global __cutensornetLoggerSetFile
    __cutensornetLoggerSetFile = dlsym(RTLD_DEFAULT, 'cutensornetLoggerSetFile')
    if __cutensornetLoggerSetFile == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetLoggerSetFile = dlsym(handle, 'cutensornetLoggerSetFile')
    
    global __cutensornetLoggerOpenFile
    __cutensornetLoggerOpenFile = dlsym(RTLD_DEFAULT, 'cutensornetLoggerOpenFile')
    if __cutensornetLoggerOpenFile == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetLoggerOpenFile = dlsym(handle, 'cutensornetLoggerOpenFile')
    
    global __cutensornetLoggerSetLevel
    __cutensornetLoggerSetLevel = dlsym(RTLD_DEFAULT, 'cutensornetLoggerSetLevel')
    if __cutensornetLoggerSetLevel == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetLoggerSetLevel = dlsym(handle, 'cutensornetLoggerSetLevel')
    
    global __cutensornetLoggerSetMask
    __cutensornetLoggerSetMask = dlsym(RTLD_DEFAULT, 'cutensornetLoggerSetMask')
    if __cutensornetLoggerSetMask == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetLoggerSetMask = dlsym(handle, 'cutensornetLoggerSetMask')
    
    global __cutensornetLoggerForceDisable
    __cutensornetLoggerForceDisable = dlsym(RTLD_DEFAULT, 'cutensornetLoggerForceDisable')
    if __cutensornetLoggerForceDisable == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetLoggerForceDisable = dlsym(handle, 'cutensornetLoggerForceDisable')
    
    global __cutensornetGetVersion
    __cutensornetGetVersion = dlsym(RTLD_DEFAULT, 'cutensornetGetVersion')
    if __cutensornetGetVersion == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetGetVersion = dlsym(handle, 'cutensornetGetVersion')
    
    global __cutensornetGetCudartVersion
    __cutensornetGetCudartVersion = dlsym(RTLD_DEFAULT, 'cutensornetGetCudartVersion')
    if __cutensornetGetCudartVersion == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetGetCudartVersion = dlsym(handle, 'cutensornetGetCudartVersion')
    
    global __cutensornetGetErrorString
    __cutensornetGetErrorString = dlsym(RTLD_DEFAULT, 'cutensornetGetErrorString')
    if __cutensornetGetErrorString == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetGetErrorString = dlsym(handle, 'cutensornetGetErrorString')
    
    global __cutensornetDistributedResetConfiguration
    __cutensornetDistributedResetConfiguration = dlsym(RTLD_DEFAULT, 'cutensornetDistributedResetConfiguration')
    if __cutensornetDistributedResetConfiguration == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDistributedResetConfiguration = dlsym(handle, 'cutensornetDistributedResetConfiguration')
    
    global __cutensornetDistributedGetNumRanks
    __cutensornetDistributedGetNumRanks = dlsym(RTLD_DEFAULT, 'cutensornetDistributedGetNumRanks')
    if __cutensornetDistributedGetNumRanks == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDistributedGetNumRanks = dlsym(handle, 'cutensornetDistributedGetNumRanks')
    
    global __cutensornetDistributedGetProcRank
    __cutensornetDistributedGetProcRank = dlsym(RTLD_DEFAULT, 'cutensornetDistributedGetProcRank')
    if __cutensornetDistributedGetProcRank == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDistributedGetProcRank = dlsym(handle, 'cutensornetDistributedGetProcRank')
    
    global __cutensornetDistributedSynchronize
    __cutensornetDistributedSynchronize = dlsym(RTLD_DEFAULT, 'cutensornetDistributedSynchronize')
    if __cutensornetDistributedSynchronize == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDistributedSynchronize = dlsym(handle, 'cutensornetDistributedSynchronize')
    
    global __cutensornetNetworkGetAttribute
    __cutensornetNetworkGetAttribute = dlsym(RTLD_DEFAULT, 'cutensornetNetworkGetAttribute')
    if __cutensornetNetworkGetAttribute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetNetworkGetAttribute = dlsym(handle, 'cutensornetNetworkGetAttribute')
    
    global __cutensornetNetworkSetAttribute
    __cutensornetNetworkSetAttribute = dlsym(RTLD_DEFAULT, 'cutensornetNetworkSetAttribute')
    if __cutensornetNetworkSetAttribute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetNetworkSetAttribute = dlsym(handle, 'cutensornetNetworkSetAttribute')
    
    global __cutensornetWorkspacePurgeCache
    __cutensornetWorkspacePurgeCache = dlsym(RTLD_DEFAULT, 'cutensornetWorkspacePurgeCache')
    if __cutensornetWorkspacePurgeCache == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetWorkspacePurgeCache = dlsym(handle, 'cutensornetWorkspacePurgeCache')
    
    global __cutensornetComputeGradientsBackward
    __cutensornetComputeGradientsBackward = dlsym(RTLD_DEFAULT, 'cutensornetComputeGradientsBackward')
    if __cutensornetComputeGradientsBackward == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetComputeGradientsBackward = dlsym(handle, 'cutensornetComputeGradientsBackward')
    
    global __cutensornetCreateState
    __cutensornetCreateState = dlsym(RTLD_DEFAULT, 'cutensornetCreateState')
    if __cutensornetCreateState == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateState = dlsym(handle, 'cutensornetCreateState')
    
    global __cutensornetStateApplyTensor
    __cutensornetStateApplyTensor = dlsym(RTLD_DEFAULT, 'cutensornetStateApplyTensor')
    if __cutensornetStateApplyTensor == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStateApplyTensor = dlsym(handle, 'cutensornetStateApplyTensor')
    
    global __cutensornetStateUpdateTensor
    __cutensornetStateUpdateTensor = dlsym(RTLD_DEFAULT, 'cutensornetStateUpdateTensor')
    if __cutensornetStateUpdateTensor == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStateUpdateTensor = dlsym(handle, 'cutensornetStateUpdateTensor')
    
    global __cutensornetDestroyState
    __cutensornetDestroyState = dlsym(RTLD_DEFAULT, 'cutensornetDestroyState')
    if __cutensornetDestroyState == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyState = dlsym(handle, 'cutensornetDestroyState')
    
    global __cutensornetCreateMarginal
    __cutensornetCreateMarginal = dlsym(RTLD_DEFAULT, 'cutensornetCreateMarginal')
    if __cutensornetCreateMarginal == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateMarginal = dlsym(handle, 'cutensornetCreateMarginal')
    
    global __cutensornetMarginalConfigure
    __cutensornetMarginalConfigure = dlsym(RTLD_DEFAULT, 'cutensornetMarginalConfigure')
    if __cutensornetMarginalConfigure == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetMarginalConfigure = dlsym(handle, 'cutensornetMarginalConfigure')
    
    global __cutensornetMarginalPrepare
    __cutensornetMarginalPrepare = dlsym(RTLD_DEFAULT, 'cutensornetMarginalPrepare')
    if __cutensornetMarginalPrepare == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetMarginalPrepare = dlsym(handle, 'cutensornetMarginalPrepare')
    
    global __cutensornetMarginalCompute
    __cutensornetMarginalCompute = dlsym(RTLD_DEFAULT, 'cutensornetMarginalCompute')
    if __cutensornetMarginalCompute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetMarginalCompute = dlsym(handle, 'cutensornetMarginalCompute')
    
    global __cutensornetDestroyMarginal
    __cutensornetDestroyMarginal = dlsym(RTLD_DEFAULT, 'cutensornetDestroyMarginal')
    if __cutensornetDestroyMarginal == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyMarginal = dlsym(handle, 'cutensornetDestroyMarginal')
    
    global __cutensornetCreateSampler
    __cutensornetCreateSampler = dlsym(RTLD_DEFAULT, 'cutensornetCreateSampler')
    if __cutensornetCreateSampler == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateSampler = dlsym(handle, 'cutensornetCreateSampler')
    
    global __cutensornetSamplerConfigure
    __cutensornetSamplerConfigure = dlsym(RTLD_DEFAULT, 'cutensornetSamplerConfigure')
    if __cutensornetSamplerConfigure == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetSamplerConfigure = dlsym(handle, 'cutensornetSamplerConfigure')
    
    global __cutensornetSamplerPrepare
    __cutensornetSamplerPrepare = dlsym(RTLD_DEFAULT, 'cutensornetSamplerPrepare')
    if __cutensornetSamplerPrepare == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetSamplerPrepare = dlsym(handle, 'cutensornetSamplerPrepare')
    
    global __cutensornetSamplerSample
    __cutensornetSamplerSample = dlsym(RTLD_DEFAULT, 'cutensornetSamplerSample')
    if __cutensornetSamplerSample == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetSamplerSample = dlsym(handle, 'cutensornetSamplerSample')
    
    global __cutensornetDestroySampler
    __cutensornetDestroySampler = dlsym(RTLD_DEFAULT, 'cutensornetDestroySampler')
    if __cutensornetDestroySampler == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroySampler = dlsym(handle, 'cutensornetDestroySampler')
    
    global __cutensornetStateFinalizeMPS
    __cutensornetStateFinalizeMPS = dlsym(RTLD_DEFAULT, 'cutensornetStateFinalizeMPS')
    if __cutensornetStateFinalizeMPS == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStateFinalizeMPS = dlsym(handle, 'cutensornetStateFinalizeMPS')
    
    global __cutensornetStateConfigure
    __cutensornetStateConfigure = dlsym(RTLD_DEFAULT, 'cutensornetStateConfigure')
    if __cutensornetStateConfigure == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStateConfigure = dlsym(handle, 'cutensornetStateConfigure')
    
    global __cutensornetStatePrepare
    __cutensornetStatePrepare = dlsym(RTLD_DEFAULT, 'cutensornetStatePrepare')
    if __cutensornetStatePrepare == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStatePrepare = dlsym(handle, 'cutensornetStatePrepare')
    
    global __cutensornetStateCompute
    __cutensornetStateCompute = dlsym(RTLD_DEFAULT, 'cutensornetStateCompute')
    if __cutensornetStateCompute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStateCompute = dlsym(handle, 'cutensornetStateCompute')
    
    global __cutensornetGetOutputStateDetails
    __cutensornetGetOutputStateDetails = dlsym(RTLD_DEFAULT, 'cutensornetGetOutputStateDetails')
    if __cutensornetGetOutputStateDetails == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetGetOutputStateDetails = dlsym(handle, 'cutensornetGetOutputStateDetails')
    
    global __cutensornetCreateNetworkOperator
    __cutensornetCreateNetworkOperator = dlsym(RTLD_DEFAULT, 'cutensornetCreateNetworkOperator')
    if __cutensornetCreateNetworkOperator == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateNetworkOperator = dlsym(handle, 'cutensornetCreateNetworkOperator')
    
    global __cutensornetNetworkOperatorAppendProduct
    __cutensornetNetworkOperatorAppendProduct = dlsym(RTLD_DEFAULT, 'cutensornetNetworkOperatorAppendProduct')
    if __cutensornetNetworkOperatorAppendProduct == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetNetworkOperatorAppendProduct = dlsym(handle, 'cutensornetNetworkOperatorAppendProduct')
    
    global __cutensornetDestroyNetworkOperator
    __cutensornetDestroyNetworkOperator = dlsym(RTLD_DEFAULT, 'cutensornetDestroyNetworkOperator')
    if __cutensornetDestroyNetworkOperator == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyNetworkOperator = dlsym(handle, 'cutensornetDestroyNetworkOperator')
    
    global __cutensornetCreateAccessor
    __cutensornetCreateAccessor = dlsym(RTLD_DEFAULT, 'cutensornetCreateAccessor')
    if __cutensornetCreateAccessor == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateAccessor = dlsym(handle, 'cutensornetCreateAccessor')
    
    global __cutensornetAccessorConfigure
    __cutensornetAccessorConfigure = dlsym(RTLD_DEFAULT, 'cutensornetAccessorConfigure')
    if __cutensornetAccessorConfigure == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetAccessorConfigure = dlsym(handle, 'cutensornetAccessorConfigure')
    
    global __cutensornetAccessorPrepare
    __cutensornetAccessorPrepare = dlsym(RTLD_DEFAULT, 'cutensornetAccessorPrepare')
    if __cutensornetAccessorPrepare == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetAccessorPrepare = dlsym(handle, 'cutensornetAccessorPrepare')
    
    global __cutensornetAccessorCompute
    __cutensornetAccessorCompute = dlsym(RTLD_DEFAULT, 'cutensornetAccessorCompute')
    if __cutensornetAccessorCompute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetAccessorCompute = dlsym(handle, 'cutensornetAccessorCompute')
    
    global __cutensornetDestroyAccessor
    __cutensornetDestroyAccessor = dlsym(RTLD_DEFAULT, 'cutensornetDestroyAccessor')
    if __cutensornetDestroyAccessor == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyAccessor = dlsym(handle, 'cutensornetDestroyAccessor')
    
    global __cutensornetCreateExpectation
    __cutensornetCreateExpectation = dlsym(RTLD_DEFAULT, 'cutensornetCreateExpectation')
    if __cutensornetCreateExpectation == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetCreateExpectation = dlsym(handle, 'cutensornetCreateExpectation')
    
    global __cutensornetExpectationConfigure
    __cutensornetExpectationConfigure = dlsym(RTLD_DEFAULT, 'cutensornetExpectationConfigure')
    if __cutensornetExpectationConfigure == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetExpectationConfigure = dlsym(handle, 'cutensornetExpectationConfigure')
    
    global __cutensornetExpectationPrepare
    __cutensornetExpectationPrepare = dlsym(RTLD_DEFAULT, 'cutensornetExpectationPrepare')
    if __cutensornetExpectationPrepare == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetExpectationPrepare = dlsym(handle, 'cutensornetExpectationPrepare')
    
    global __cutensornetExpectationCompute
    __cutensornetExpectationCompute = dlsym(RTLD_DEFAULT, 'cutensornetExpectationCompute')
    if __cutensornetExpectationCompute == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetExpectationCompute = dlsym(handle, 'cutensornetExpectationCompute')
    
    global __cutensornetDestroyExpectation
    __cutensornetDestroyExpectation = dlsym(RTLD_DEFAULT, 'cutensornetDestroyExpectation')
    if __cutensornetDestroyExpectation == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetDestroyExpectation = dlsym(handle, 'cutensornetDestroyExpectation')
    
    global __cutensornetStateApplyTensorOperator
    __cutensornetStateApplyTensorOperator = dlsym(RTLD_DEFAULT, 'cutensornetStateApplyTensorOperator')
    if __cutensornetStateApplyTensorOperator == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStateApplyTensorOperator = dlsym(handle, 'cutensornetStateApplyTensorOperator')
    
    global __cutensornetStateApplyControlledTensorOperator
    __cutensornetStateApplyControlledTensorOperator = dlsym(RTLD_DEFAULT, 'cutensornetStateApplyControlledTensorOperator')
    if __cutensornetStateApplyControlledTensorOperator == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStateApplyControlledTensorOperator = dlsym(handle, 'cutensornetStateApplyControlledTensorOperator')
    
    global __cutensornetStateUpdateTensorOperator
    __cutensornetStateUpdateTensorOperator = dlsym(RTLD_DEFAULT, 'cutensornetStateUpdateTensorOperator')
    if __cutensornetStateUpdateTensorOperator == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStateUpdateTensorOperator = dlsym(handle, 'cutensornetStateUpdateTensorOperator')
    
    global __cutensornetStateApplyNetworkOperator
    __cutensornetStateApplyNetworkOperator = dlsym(RTLD_DEFAULT, 'cutensornetStateApplyNetworkOperator')
    if __cutensornetStateApplyNetworkOperator == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStateApplyNetworkOperator = dlsym(handle, 'cutensornetStateApplyNetworkOperator')
    
    global __cutensornetStateInitializeMPS
    __cutensornetStateInitializeMPS = dlsym(RTLD_DEFAULT, 'cutensornetStateInitializeMPS')
    if __cutensornetStateInitializeMPS == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStateInitializeMPS = dlsym(handle, 'cutensornetStateInitializeMPS')
    
    global __cutensornetStateGetInfo
    __cutensornetStateGetInfo = dlsym(RTLD_DEFAULT, 'cutensornetStateGetInfo')
    if __cutensornetStateGetInfo == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetStateGetInfo = dlsym(handle, 'cutensornetStateGetInfo')
    
    global __cutensornetNetworkOperatorAppendMPO
    __cutensornetNetworkOperatorAppendMPO = dlsym(RTLD_DEFAULT, 'cutensornetNetworkOperatorAppendMPO')
    if __cutensornetNetworkOperatorAppendMPO == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetNetworkOperatorAppendMPO = dlsym(handle, 'cutensornetNetworkOperatorAppendMPO')
    
    global __cutensornetAccessorGetInfo
    __cutensornetAccessorGetInfo = dlsym(RTLD_DEFAULT, 'cutensornetAccessorGetInfo')
    if __cutensornetAccessorGetInfo == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetAccessorGetInfo = dlsym(handle, 'cutensornetAccessorGetInfo')
    
    global __cutensornetExpectationGetInfo
    __cutensornetExpectationGetInfo = dlsym(RTLD_DEFAULT, 'cutensornetExpectationGetInfo')
    if __cutensornetExpectationGetInfo == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetExpectationGetInfo = dlsym(handle, 'cutensornetExpectationGetInfo')
    
    global __cutensornetMarginalGetInfo
    __cutensornetMarginalGetInfo = dlsym(RTLD_DEFAULT, 'cutensornetMarginalGetInfo')
    if __cutensornetMarginalGetInfo == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetMarginalGetInfo = dlsym(handle, 'cutensornetMarginalGetInfo')
    
    global __cutensornetSamplerGetInfo
    __cutensornetSamplerGetInfo = dlsym(RTLD_DEFAULT, 'cutensornetSamplerGetInfo')
    if __cutensornetSamplerGetInfo == NULL:
        if handle == NULL:
            handle = load_library()
        __cutensornetSamplerGetInfo = dlsym(handle, 'cutensornetSamplerGetInfo')

    __py_cutensornet_init = True
    return 0


cpdef dict _inspect_function_pointers():
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
    
    global __cutensornetComputeGradientsBackward
    data["__cutensornetComputeGradientsBackward"] = <intptr_t>__cutensornetComputeGradientsBackward
    
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

    return data


###############################################################################
# Wrapper functions
###############################################################################

cdef cutensornetStatus_t _cutensornetCreate(cutensornetHandle_t* handle) except* nogil:
    global __cutensornetCreate
    _check_or_init_cutensornet()
    if __cutensornetCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreate is not found")
    return (<cutensornetStatus_t (*)(cutensornetHandle_t*) nogil>__cutensornetCreate)(
        handle)


cdef cutensornetStatus_t _cutensornetDestroy(cutensornetHandle_t handle) except* nogil:
    global __cutensornetDestroy
    _check_or_init_cutensornet()
    if __cutensornetDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroy is not found")
    return (<cutensornetStatus_t (*)(cutensornetHandle_t) nogil>__cutensornetDestroy)(
        handle)


cdef cutensornetStatus_t _cutensornetCreateNetworkDescriptor(const cutensornetHandle_t handle, int32_t numInputs, const int32_t numModesIn[], const int64_t* const extentsIn[], const int64_t* const stridesIn[], const int32_t* const modesIn[], const cutensornetTensorQualifiers_t qualifiersIn[], int32_t numModesOut, const int64_t extentsOut[], const int64_t stridesOut[], const int32_t modesOut[], cudaDataType_t dataType, cutensornetComputeType_t computeType, cutensornetNetworkDescriptor_t* descNet) except* nogil:
    global __cutensornetCreateNetworkDescriptor
    _check_or_init_cutensornet()
    if __cutensornetCreateNetworkDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateNetworkDescriptor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t, const int32_t*, const int64_t* const*, const int64_t* const*, const int32_t* const*, const cutensornetTensorQualifiers_t*, int32_t, const int64_t*, const int64_t*, const int32_t*, cudaDataType_t, cutensornetComputeType_t, cutensornetNetworkDescriptor_t*) nogil>__cutensornetCreateNetworkDescriptor)(
        handle, numInputs, numModesIn, extentsIn, stridesIn, modesIn, qualifiersIn, numModesOut, extentsOut, stridesOut, modesOut, dataType, computeType, descNet)


cdef cutensornetStatus_t _cutensornetDestroyNetworkDescriptor(cutensornetNetworkDescriptor_t desc) except* nogil:
    global __cutensornetDestroyNetworkDescriptor
    _check_or_init_cutensornet()
    if __cutensornetDestroyNetworkDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyNetworkDescriptor is not found")
    return (<cutensornetStatus_t (*)(cutensornetNetworkDescriptor_t) nogil>__cutensornetDestroyNetworkDescriptor)(
        desc)


cdef cutensornetStatus_t _cutensornetGetOutputTensorDescriptor(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, cutensornetTensorDescriptor_t* outputTensorDesc) except* nogil:
    global __cutensornetGetOutputTensorDescriptor
    _check_or_init_cutensornet()
    if __cutensornetGetOutputTensorDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetOutputTensorDescriptor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, cutensornetTensorDescriptor_t*) nogil>__cutensornetGetOutputTensorDescriptor)(
        handle, descNet, outputTensorDesc)


cdef cutensornetStatus_t _cutensornetGetTensorDetails(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t tensorDesc, int32_t* numModes, size_t* dataSize, int32_t* modeLabels, int64_t* extents, int64_t* strides) except* nogil:
    global __cutensornetGetTensorDetails
    _check_or_init_cutensornet()
    if __cutensornetGetTensorDetails == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetTensorDetails is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, int32_t*, size_t*, int32_t*, int64_t*, int64_t*) nogil>__cutensornetGetTensorDetails)(
        handle, tensorDesc, numModes, dataSize, modeLabels, extents, strides)


cdef cutensornetStatus_t _cutensornetCreateWorkspaceDescriptor(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t* workDesc) except* nogil:
    global __cutensornetCreateWorkspaceDescriptor
    _check_or_init_cutensornet()
    if __cutensornetCreateWorkspaceDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateWorkspaceDescriptor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetWorkspaceDescriptor_t*) nogil>__cutensornetCreateWorkspaceDescriptor)(
        handle, workDesc)


cdef cutensornetStatus_t _cutensornetWorkspaceComputeContractionSizes(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetWorkspaceDescriptor_t workDesc) except* nogil:
    global __cutensornetWorkspaceComputeContractionSizes
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceComputeContractionSizes == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceComputeContractionSizes is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, const cutensornetContractionOptimizerInfo_t, cutensornetWorkspaceDescriptor_t) nogil>__cutensornetWorkspaceComputeContractionSizes)(
        handle, descNet, optimizerInfo, workDesc)


cdef cutensornetStatus_t _cutensornetWorkspaceGetMemorySize(const cutensornetHandle_t handle, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetWorksizePref_t workPref, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, int64_t* memorySize) except* nogil:
    global __cutensornetWorkspaceGetMemorySize
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceGetMemorySize == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceGetMemorySize is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetWorkspaceDescriptor_t, cutensornetWorksizePref_t, cutensornetMemspace_t, cutensornetWorkspaceKind_t, int64_t*) nogil>__cutensornetWorkspaceGetMemorySize)(
        handle, workDesc, workPref, memSpace, workKind, memorySize)


cdef cutensornetStatus_t _cutensornetWorkspaceSetMemory(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, void* const memoryPtr, int64_t memorySize) except* nogil:
    global __cutensornetWorkspaceSetMemory
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceSetMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceSetMemory is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetWorkspaceDescriptor_t, cutensornetMemspace_t, cutensornetWorkspaceKind_t, void* const, int64_t) nogil>__cutensornetWorkspaceSetMemory)(
        handle, workDesc, memSpace, workKind, memoryPtr, memorySize)


cdef cutensornetStatus_t _cutensornetWorkspaceGetMemory(const cutensornetHandle_t handle, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace, cutensornetWorkspaceKind_t workKind, void** memoryPtr, int64_t* memorySize) except* nogil:
    global __cutensornetWorkspaceGetMemory
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceGetMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceGetMemory is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetWorkspaceDescriptor_t, cutensornetMemspace_t, cutensornetWorkspaceKind_t, void**, int64_t*) nogil>__cutensornetWorkspaceGetMemory)(
        handle, workDesc, memSpace, workKind, memoryPtr, memorySize)


cdef cutensornetStatus_t _cutensornetDestroyWorkspaceDescriptor(cutensornetWorkspaceDescriptor_t desc) except* nogil:
    global __cutensornetDestroyWorkspaceDescriptor
    _check_or_init_cutensornet()
    if __cutensornetDestroyWorkspaceDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyWorkspaceDescriptor is not found")
    return (<cutensornetStatus_t (*)(cutensornetWorkspaceDescriptor_t) nogil>__cutensornetDestroyWorkspaceDescriptor)(
        desc)


cdef cutensornetStatus_t _cutensornetCreateContractionOptimizerConfig(const cutensornetHandle_t handle, cutensornetContractionOptimizerConfig_t* optimizerConfig) except* nogil:
    global __cutensornetCreateContractionOptimizerConfig
    _check_or_init_cutensornet()
    if __cutensornetCreateContractionOptimizerConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateContractionOptimizerConfig is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionOptimizerConfig_t*) nogil>__cutensornetCreateContractionOptimizerConfig)(
        handle, optimizerConfig)


cdef cutensornetStatus_t _cutensornetDestroyContractionOptimizerConfig(cutensornetContractionOptimizerConfig_t optimizerConfig) except* nogil:
    global __cutensornetDestroyContractionOptimizerConfig
    _check_or_init_cutensornet()
    if __cutensornetDestroyContractionOptimizerConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyContractionOptimizerConfig is not found")
    return (<cutensornetStatus_t (*)(cutensornetContractionOptimizerConfig_t) nogil>__cutensornetDestroyContractionOptimizerConfig)(
        optimizerConfig)


cdef cutensornetStatus_t _cutensornetContractionOptimizerConfigGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionOptimizerConfig_t optimizerConfig, cutensornetContractionOptimizerConfigAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    global __cutensornetContractionOptimizerConfigGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerConfigGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerConfigGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetContractionOptimizerConfig_t, cutensornetContractionOptimizerConfigAttributes_t, void*, size_t) nogil>__cutensornetContractionOptimizerConfigGetAttribute)(
        handle, optimizerConfig, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetContractionOptimizerConfigSetAttribute(const cutensornetHandle_t handle, cutensornetContractionOptimizerConfig_t optimizerConfig, cutensornetContractionOptimizerConfigAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    global __cutensornetContractionOptimizerConfigSetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerConfigSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerConfigSetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionOptimizerConfig_t, cutensornetContractionOptimizerConfigAttributes_t, const void*, size_t) nogil>__cutensornetContractionOptimizerConfigSetAttribute)(
        handle, optimizerConfig, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetDestroyContractionOptimizerInfo(cutensornetContractionOptimizerInfo_t optimizerInfo) except* nogil:
    global __cutensornetDestroyContractionOptimizerInfo
    _check_or_init_cutensornet()
    if __cutensornetDestroyContractionOptimizerInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyContractionOptimizerInfo is not found")
    return (<cutensornetStatus_t (*)(cutensornetContractionOptimizerInfo_t) nogil>__cutensornetDestroyContractionOptimizerInfo)(
        optimizerInfo)


cdef cutensornetStatus_t _cutensornetCreateContractionOptimizerInfo(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, cutensornetContractionOptimizerInfo_t* optimizerInfo) except* nogil:
    global __cutensornetCreateContractionOptimizerInfo
    _check_or_init_cutensornet()
    if __cutensornetCreateContractionOptimizerInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateContractionOptimizerInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, cutensornetContractionOptimizerInfo_t*) nogil>__cutensornetCreateContractionOptimizerInfo)(
        handle, descNet, optimizerInfo)


cdef cutensornetStatus_t _cutensornetContractionOptimize(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const cutensornetContractionOptimizerConfig_t optimizerConfig, uint64_t workspaceSizeConstraint, cutensornetContractionOptimizerInfo_t optimizerInfo) except* nogil:
    global __cutensornetContractionOptimize
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimize == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimize is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, const cutensornetContractionOptimizerConfig_t, uint64_t, cutensornetContractionOptimizerInfo_t) nogil>__cutensornetContractionOptimize)(
        handle, descNet, optimizerConfig, workspaceSizeConstraint, optimizerInfo)


cdef cutensornetStatus_t _cutensornetContractionOptimizerInfoGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetContractionOptimizerInfoAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    global __cutensornetContractionOptimizerInfoGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerInfoGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerInfoGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetContractionOptimizerInfo_t, cutensornetContractionOptimizerInfoAttributes_t, void*, size_t) nogil>__cutensornetContractionOptimizerInfoGetAttribute)(
        handle, optimizerInfo, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetContractionOptimizerInfoSetAttribute(const cutensornetHandle_t handle, cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetContractionOptimizerInfoAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    global __cutensornetContractionOptimizerInfoSetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerInfoSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerInfoSetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionOptimizerInfo_t, cutensornetContractionOptimizerInfoAttributes_t, const void*, size_t) nogil>__cutensornetContractionOptimizerInfoSetAttribute)(
        handle, optimizerInfo, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetContractionOptimizerInfoGetPackedSize(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, size_t* sizeInBytes) except* nogil:
    global __cutensornetContractionOptimizerInfoGetPackedSize
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerInfoGetPackedSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerInfoGetPackedSize is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetContractionOptimizerInfo_t, size_t*) nogil>__cutensornetContractionOptimizerInfoGetPackedSize)(
        handle, optimizerInfo, sizeInBytes)


cdef cutensornetStatus_t _cutensornetContractionOptimizerInfoPackData(const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, void* buffer, size_t sizeInBytes) except* nogil:
    global __cutensornetContractionOptimizerInfoPackData
    _check_or_init_cutensornet()
    if __cutensornetContractionOptimizerInfoPackData == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionOptimizerInfoPackData is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetContractionOptimizerInfo_t, void*, size_t) nogil>__cutensornetContractionOptimizerInfoPackData)(
        handle, optimizerInfo, buffer, sizeInBytes)


cdef cutensornetStatus_t _cutensornetCreateContractionOptimizerInfoFromPackedData(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const void* buffer, size_t sizeInBytes, cutensornetContractionOptimizerInfo_t* optimizerInfo) except* nogil:
    global __cutensornetCreateContractionOptimizerInfoFromPackedData
    _check_or_init_cutensornet()
    if __cutensornetCreateContractionOptimizerInfoFromPackedData == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateContractionOptimizerInfoFromPackedData is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, const void*, size_t, cutensornetContractionOptimizerInfo_t*) nogil>__cutensornetCreateContractionOptimizerInfoFromPackedData)(
        handle, descNet, buffer, sizeInBytes, optimizerInfo)


cdef cutensornetStatus_t _cutensornetUpdateContractionOptimizerInfoFromPackedData(const cutensornetHandle_t handle, const void* buffer, size_t sizeInBytes, cutensornetContractionOptimizerInfo_t optimizerInfo) except* nogil:
    global __cutensornetUpdateContractionOptimizerInfoFromPackedData
    _check_or_init_cutensornet()
    if __cutensornetUpdateContractionOptimizerInfoFromPackedData == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetUpdateContractionOptimizerInfoFromPackedData is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const void*, size_t, cutensornetContractionOptimizerInfo_t) nogil>__cutensornetUpdateContractionOptimizerInfoFromPackedData)(
        handle, buffer, sizeInBytes, optimizerInfo)


cdef cutensornetStatus_t _cutensornetCreateContractionPlan(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t descNet, const cutensornetContractionOptimizerInfo_t optimizerInfo, const cutensornetWorkspaceDescriptor_t workDesc, cutensornetContractionPlan_t* plan) except* nogil:
    global __cutensornetCreateContractionPlan
    _check_or_init_cutensornet()
    if __cutensornetCreateContractionPlan == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateContractionPlan is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, const cutensornetContractionOptimizerInfo_t, const cutensornetWorkspaceDescriptor_t, cutensornetContractionPlan_t*) nogil>__cutensornetCreateContractionPlan)(
        handle, descNet, optimizerInfo, workDesc, plan)


cdef cutensornetStatus_t _cutensornetDestroyContractionPlan(cutensornetContractionPlan_t plan) except* nogil:
    global __cutensornetDestroyContractionPlan
    _check_or_init_cutensornet()
    if __cutensornetDestroyContractionPlan == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyContractionPlan is not found")
    return (<cutensornetStatus_t (*)(cutensornetContractionPlan_t) nogil>__cutensornetDestroyContractionPlan)(
        plan)


cdef cutensornetStatus_t _cutensornetContractionAutotune(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan, const void* const rawDataIn[], void* rawDataOut, cutensornetWorkspaceDescriptor_t workDesc, const cutensornetContractionAutotunePreference_t pref, cudaStream_t stream) except* nogil:
    global __cutensornetContractionAutotune
    _check_or_init_cutensornet()
    if __cutensornetContractionAutotune == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionAutotune is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionPlan_t, const void* const*, void*, cutensornetWorkspaceDescriptor_t, const cutensornetContractionAutotunePreference_t, cudaStream_t) nogil>__cutensornetContractionAutotune)(
        handle, plan, rawDataIn, rawDataOut, workDesc, pref, stream)


cdef cutensornetStatus_t _cutensornetCreateContractionAutotunePreference(const cutensornetHandle_t handle, cutensornetContractionAutotunePreference_t* autotunePreference) except* nogil:
    global __cutensornetCreateContractionAutotunePreference
    _check_or_init_cutensornet()
    if __cutensornetCreateContractionAutotunePreference == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateContractionAutotunePreference is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionAutotunePreference_t*) nogil>__cutensornetCreateContractionAutotunePreference)(
        handle, autotunePreference)


cdef cutensornetStatus_t _cutensornetContractionAutotunePreferenceGetAttribute(const cutensornetHandle_t handle, const cutensornetContractionAutotunePreference_t autotunePreference, cutensornetContractionAutotunePreferenceAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    global __cutensornetContractionAutotunePreferenceGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionAutotunePreferenceGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionAutotunePreferenceGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetContractionAutotunePreference_t, cutensornetContractionAutotunePreferenceAttributes_t, void*, size_t) nogil>__cutensornetContractionAutotunePreferenceGetAttribute)(
        handle, autotunePreference, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetContractionAutotunePreferenceSetAttribute(const cutensornetHandle_t handle, cutensornetContractionAutotunePreference_t autotunePreference, cutensornetContractionAutotunePreferenceAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    global __cutensornetContractionAutotunePreferenceSetAttribute
    _check_or_init_cutensornet()
    if __cutensornetContractionAutotunePreferenceSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractionAutotunePreferenceSetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionAutotunePreference_t, cutensornetContractionAutotunePreferenceAttributes_t, const void*, size_t) nogil>__cutensornetContractionAutotunePreferenceSetAttribute)(
        handle, autotunePreference, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetDestroyContractionAutotunePreference(cutensornetContractionAutotunePreference_t autotunePreference) except* nogil:
    global __cutensornetDestroyContractionAutotunePreference
    _check_or_init_cutensornet()
    if __cutensornetDestroyContractionAutotunePreference == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyContractionAutotunePreference is not found")
    return (<cutensornetStatus_t (*)(cutensornetContractionAutotunePreference_t) nogil>__cutensornetDestroyContractionAutotunePreference)(
        autotunePreference)


cdef cutensornetStatus_t _cutensornetCreateSliceGroupFromIDRange(const cutensornetHandle_t handle, int64_t sliceIdStart, int64_t sliceIdStop, int64_t sliceIdStep, cutensornetSliceGroup_t* sliceGroup) except* nogil:
    global __cutensornetCreateSliceGroupFromIDRange
    _check_or_init_cutensornet()
    if __cutensornetCreateSliceGroupFromIDRange == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateSliceGroupFromIDRange is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int64_t, int64_t, int64_t, cutensornetSliceGroup_t*) nogil>__cutensornetCreateSliceGroupFromIDRange)(
        handle, sliceIdStart, sliceIdStop, sliceIdStep, sliceGroup)


cdef cutensornetStatus_t _cutensornetCreateSliceGroupFromIDs(const cutensornetHandle_t handle, const int64_t* beginIDSequence, const int64_t* endIDSequence, cutensornetSliceGroup_t* sliceGroup) except* nogil:
    global __cutensornetCreateSliceGroupFromIDs
    _check_or_init_cutensornet()
    if __cutensornetCreateSliceGroupFromIDs == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateSliceGroupFromIDs is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const int64_t*, const int64_t*, cutensornetSliceGroup_t*) nogil>__cutensornetCreateSliceGroupFromIDs)(
        handle, beginIDSequence, endIDSequence, sliceGroup)


cdef cutensornetStatus_t _cutensornetDestroySliceGroup(cutensornetSliceGroup_t sliceGroup) except* nogil:
    global __cutensornetDestroySliceGroup
    _check_or_init_cutensornet()
    if __cutensornetDestroySliceGroup == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroySliceGroup is not found")
    return (<cutensornetStatus_t (*)(cutensornetSliceGroup_t) nogil>__cutensornetDestroySliceGroup)(
        sliceGroup)


cdef cutensornetStatus_t _cutensornetContractSlices(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan, const void* const rawDataIn[], void* rawDataOut, int32_t accumulateOutput, cutensornetWorkspaceDescriptor_t workDesc, const cutensornetSliceGroup_t sliceGroup, cudaStream_t stream) except* nogil:
    global __cutensornetContractSlices
    _check_or_init_cutensornet()
    if __cutensornetContractSlices == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetContractSlices is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionPlan_t, const void* const*, void*, int32_t, cutensornetWorkspaceDescriptor_t, const cutensornetSliceGroup_t, cudaStream_t) nogil>__cutensornetContractSlices)(
        handle, plan, rawDataIn, rawDataOut, accumulateOutput, workDesc, sliceGroup, stream)


cdef cutensornetStatus_t _cutensornetCreateTensorDescriptor(const cutensornetHandle_t handle, int32_t numModes, const int64_t extents[], const int64_t strides[], const int32_t modes[], cudaDataType_t dataType, cutensornetTensorDescriptor_t* descTensor) except* nogil:
    global __cutensornetCreateTensorDescriptor
    _check_or_init_cutensornet()
    if __cutensornetCreateTensorDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateTensorDescriptor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t, const int64_t*, const int64_t*, const int32_t*, cudaDataType_t, cutensornetTensorDescriptor_t*) nogil>__cutensornetCreateTensorDescriptor)(
        handle, numModes, extents, strides, modes, dataType, descTensor)


cdef cutensornetStatus_t _cutensornetDestroyTensorDescriptor(cutensornetTensorDescriptor_t descTensor) except* nogil:
    global __cutensornetDestroyTensorDescriptor
    _check_or_init_cutensornet()
    if __cutensornetDestroyTensorDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyTensorDescriptor is not found")
    return (<cutensornetStatus_t (*)(cutensornetTensorDescriptor_t) nogil>__cutensornetDestroyTensorDescriptor)(
        descTensor)


cdef cutensornetStatus_t _cutensornetCreateTensorSVDConfig(const cutensornetHandle_t handle, cutensornetTensorSVDConfig_t* svdConfig) except* nogil:
    global __cutensornetCreateTensorSVDConfig
    _check_or_init_cutensornet()
    if __cutensornetCreateTensorSVDConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateTensorSVDConfig is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetTensorSVDConfig_t*) nogil>__cutensornetCreateTensorSVDConfig)(
        handle, svdConfig)


cdef cutensornetStatus_t _cutensornetDestroyTensorSVDConfig(cutensornetTensorSVDConfig_t svdConfig) except* nogil:
    global __cutensornetDestroyTensorSVDConfig
    _check_or_init_cutensornet()
    if __cutensornetDestroyTensorSVDConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyTensorSVDConfig is not found")
    return (<cutensornetStatus_t (*)(cutensornetTensorSVDConfig_t) nogil>__cutensornetDestroyTensorSVDConfig)(
        svdConfig)


cdef cutensornetStatus_t _cutensornetTensorSVDConfigGetAttribute(const cutensornetHandle_t handle, const cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDConfigAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    global __cutensornetTensorSVDConfigGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetTensorSVDConfigGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetTensorSVDConfigGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorSVDConfig_t, cutensornetTensorSVDConfigAttributes_t, void*, size_t) nogil>__cutensornetTensorSVDConfigGetAttribute)(
        handle, svdConfig, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetTensorSVDConfigSetAttribute(const cutensornetHandle_t handle, cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDConfigAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    global __cutensornetTensorSVDConfigSetAttribute
    _check_or_init_cutensornet()
    if __cutensornetTensorSVDConfigSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetTensorSVDConfigSetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetTensorSVDConfig_t, cutensornetTensorSVDConfigAttributes_t, const void*, size_t) nogil>__cutensornetTensorSVDConfigSetAttribute)(
        handle, svdConfig, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetWorkspaceComputeSVDSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const cutensornetTensorDescriptor_t descTensorU, const cutensornetTensorDescriptor_t descTensorV, const cutensornetTensorSVDConfig_t svdConfig, cutensornetWorkspaceDescriptor_t workDesc) except* nogil:
    global __cutensornetWorkspaceComputeSVDSizes
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceComputeSVDSizes == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceComputeSVDSizes is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorSVDConfig_t, cutensornetWorkspaceDescriptor_t) nogil>__cutensornetWorkspaceComputeSVDSizes)(
        handle, descTensorIn, descTensorU, descTensorV, svdConfig, workDesc)


cdef cutensornetStatus_t _cutensornetWorkspaceComputeQRSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const cutensornetTensorDescriptor_t descTensorQ, const cutensornetTensorDescriptor_t descTensorR, cutensornetWorkspaceDescriptor_t workDesc) except* nogil:
    global __cutensornetWorkspaceComputeQRSizes
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceComputeQRSizes == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceComputeQRSizes is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, cutensornetWorkspaceDescriptor_t) nogil>__cutensornetWorkspaceComputeQRSizes)(
        handle, descTensorIn, descTensorQ, descTensorR, workDesc)


cdef cutensornetStatus_t _cutensornetCreateTensorSVDInfo(const cutensornetHandle_t handle, cutensornetTensorSVDInfo_t* svdInfo) except* nogil:
    global __cutensornetCreateTensorSVDInfo
    _check_or_init_cutensornet()
    if __cutensornetCreateTensorSVDInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateTensorSVDInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetTensorSVDInfo_t*) nogil>__cutensornetCreateTensorSVDInfo)(
        handle, svdInfo)


cdef cutensornetStatus_t _cutensornetTensorSVDInfoGetAttribute(const cutensornetHandle_t handle, const cutensornetTensorSVDInfo_t svdInfo, cutensornetTensorSVDInfoAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    global __cutensornetTensorSVDInfoGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetTensorSVDInfoGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetTensorSVDInfoGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorSVDInfo_t, cutensornetTensorSVDInfoAttributes_t, void*, size_t) nogil>__cutensornetTensorSVDInfoGetAttribute)(
        handle, svdInfo, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetDestroyTensorSVDInfo(cutensornetTensorSVDInfo_t svdInfo) except* nogil:
    global __cutensornetDestroyTensorSVDInfo
    _check_or_init_cutensornet()
    if __cutensornetDestroyTensorSVDInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyTensorSVDInfo is not found")
    return (<cutensornetStatus_t (*)(cutensornetTensorSVDInfo_t) nogil>__cutensornetDestroyTensorSVDInfo)(
        svdInfo)


cdef cutensornetStatus_t _cutensornetTensorSVD(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const void* const rawDataIn, cutensornetTensorDescriptor_t descTensorU, void* u, void* s, cutensornetTensorDescriptor_t descTensorV, void* v, const cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDInfo_t svdInfo, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil:
    global __cutensornetTensorSVD
    _check_or_init_cutensornet()
    if __cutensornetTensorSVD == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetTensorSVD is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const void* const, cutensornetTensorDescriptor_t, void*, void*, cutensornetTensorDescriptor_t, void*, const cutensornetTensorSVDConfig_t, cutensornetTensorSVDInfo_t, const cutensornetWorkspaceDescriptor_t, cudaStream_t) nogil>__cutensornetTensorSVD)(
        handle, descTensorIn, rawDataIn, descTensorU, u, s, descTensorV, v, svdConfig, svdInfo, workDesc, stream)


cdef cutensornetStatus_t _cutensornetTensorQR(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn, const void* const rawDataIn, const cutensornetTensorDescriptor_t descTensorQ, void* q, const cutensornetTensorDescriptor_t descTensorR, void* r, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil:
    global __cutensornetTensorQR
    _check_or_init_cutensornet()
    if __cutensornetTensorQR == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetTensorQR is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const void* const, const cutensornetTensorDescriptor_t, void*, const cutensornetTensorDescriptor_t, void*, const cutensornetWorkspaceDescriptor_t, cudaStream_t) nogil>__cutensornetTensorQR)(
        handle, descTensorIn, rawDataIn, descTensorQ, q, descTensorR, r, workDesc, stream)


cdef cutensornetStatus_t _cutensornetWorkspaceComputeGateSplitSizes(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA, const cutensornetTensorDescriptor_t descTensorInB, const cutensornetTensorDescriptor_t descTensorInG, const cutensornetTensorDescriptor_t descTensorU, const cutensornetTensorDescriptor_t descTensorV, const cutensornetGateSplitAlgo_t gateAlgo, const cutensornetTensorSVDConfig_t svdConfig, cutensornetComputeType_t computeType, cutensornetWorkspaceDescriptor_t workDesc) except* nogil:
    global __cutensornetWorkspaceComputeGateSplitSizes
    _check_or_init_cutensornet()
    if __cutensornetWorkspaceComputeGateSplitSizes == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspaceComputeGateSplitSizes is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetTensorDescriptor_t, const cutensornetGateSplitAlgo_t, const cutensornetTensorSVDConfig_t, cutensornetComputeType_t, cutensornetWorkspaceDescriptor_t) nogil>__cutensornetWorkspaceComputeGateSplitSizes)(
        handle, descTensorInA, descTensorInB, descTensorInG, descTensorU, descTensorV, gateAlgo, svdConfig, computeType, workDesc)


cdef cutensornetStatus_t _cutensornetGateSplit(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA, const void* rawDataInA, const cutensornetTensorDescriptor_t descTensorInB, const void* rawDataInB, const cutensornetTensorDescriptor_t descTensorInG, const void* rawDataInG, cutensornetTensorDescriptor_t descTensorU, void* u, void* s, cutensornetTensorDescriptor_t descTensorV, void* v, const cutensornetGateSplitAlgo_t gateAlgo, const cutensornetTensorSVDConfig_t svdConfig, cutensornetComputeType_t computeType, cutensornetTensorSVDInfo_t svdInfo, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil:
    global __cutensornetGateSplit
    _check_or_init_cutensornet()
    if __cutensornetGateSplit == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGateSplit is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetTensorDescriptor_t, const void*, const cutensornetTensorDescriptor_t, const void*, const cutensornetTensorDescriptor_t, const void*, cutensornetTensorDescriptor_t, void*, void*, cutensornetTensorDescriptor_t, void*, const cutensornetGateSplitAlgo_t, const cutensornetTensorSVDConfig_t, cutensornetComputeType_t, cutensornetTensorSVDInfo_t, const cutensornetWorkspaceDescriptor_t, cudaStream_t) nogil>__cutensornetGateSplit)(
        handle, descTensorInA, rawDataInA, descTensorInB, rawDataInB, descTensorInG, rawDataInG, descTensorU, u, s, descTensorV, v, gateAlgo, svdConfig, computeType, svdInfo, workDesc, stream)


cdef cutensornetStatus_t _cutensornetGetDeviceMemHandler(const cutensornetHandle_t handle, cutensornetDeviceMemHandler_t* devMemHandler) except* nogil:
    global __cutensornetGetDeviceMemHandler
    _check_or_init_cutensornet()
    if __cutensornetGetDeviceMemHandler == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetDeviceMemHandler is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetDeviceMemHandler_t*) nogil>__cutensornetGetDeviceMemHandler)(
        handle, devMemHandler)


cdef cutensornetStatus_t _cutensornetSetDeviceMemHandler(cutensornetHandle_t handle, const cutensornetDeviceMemHandler_t* devMemHandler) except* nogil:
    global __cutensornetSetDeviceMemHandler
    _check_or_init_cutensornet()
    if __cutensornetSetDeviceMemHandler == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetSetDeviceMemHandler is not found")
    return (<cutensornetStatus_t (*)(cutensornetHandle_t, const cutensornetDeviceMemHandler_t*) nogil>__cutensornetSetDeviceMemHandler)(
        handle, devMemHandler)


cdef cutensornetStatus_t _cutensornetLoggerSetCallback(cutensornetLoggerCallback_t callback) except* nogil:
    global __cutensornetLoggerSetCallback
    _check_or_init_cutensornet()
    if __cutensornetLoggerSetCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerSetCallback is not found")
    return (<cutensornetStatus_t (*)(cutensornetLoggerCallback_t) nogil>__cutensornetLoggerSetCallback)(
        callback)


cdef cutensornetStatus_t _cutensornetLoggerSetCallbackData(cutensornetLoggerCallbackData_t callback, void* userData) except* nogil:
    global __cutensornetLoggerSetCallbackData
    _check_or_init_cutensornet()
    if __cutensornetLoggerSetCallbackData == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerSetCallbackData is not found")
    return (<cutensornetStatus_t (*)(cutensornetLoggerCallbackData_t, void*) nogil>__cutensornetLoggerSetCallbackData)(
        callback, userData)


cdef cutensornetStatus_t _cutensornetLoggerSetFile(FILE* file) except* nogil:
    global __cutensornetLoggerSetFile
    _check_or_init_cutensornet()
    if __cutensornetLoggerSetFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerSetFile is not found")
    return (<cutensornetStatus_t (*)(FILE*) nogil>__cutensornetLoggerSetFile)(
        file)


cdef cutensornetStatus_t _cutensornetLoggerOpenFile(const char* logFile) except* nogil:
    global __cutensornetLoggerOpenFile
    _check_or_init_cutensornet()
    if __cutensornetLoggerOpenFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerOpenFile is not found")
    return (<cutensornetStatus_t (*)(const char*) nogil>__cutensornetLoggerOpenFile)(
        logFile)


cdef cutensornetStatus_t _cutensornetLoggerSetLevel(int32_t level) except* nogil:
    global __cutensornetLoggerSetLevel
    _check_or_init_cutensornet()
    if __cutensornetLoggerSetLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerSetLevel is not found")
    return (<cutensornetStatus_t (*)(int32_t) nogil>__cutensornetLoggerSetLevel)(
        level)


cdef cutensornetStatus_t _cutensornetLoggerSetMask(int32_t mask) except* nogil:
    global __cutensornetLoggerSetMask
    _check_or_init_cutensornet()
    if __cutensornetLoggerSetMask == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerSetMask is not found")
    return (<cutensornetStatus_t (*)(int32_t) nogil>__cutensornetLoggerSetMask)(
        mask)


cdef cutensornetStatus_t _cutensornetLoggerForceDisable() except* nogil:
    global __cutensornetLoggerForceDisable
    _check_or_init_cutensornet()
    if __cutensornetLoggerForceDisable == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetLoggerForceDisable is not found")
    return (<cutensornetStatus_t (*)() nogil>__cutensornetLoggerForceDisable)(
        )


cdef size_t _cutensornetGetVersion() except* nogil:
    global __cutensornetGetVersion
    _check_or_init_cutensornet()
    if __cutensornetGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetVersion is not found")
    return (<size_t (*)() nogil>__cutensornetGetVersion)(
        )


cdef size_t _cutensornetGetCudartVersion() except* nogil:
    global __cutensornetGetCudartVersion
    _check_or_init_cutensornet()
    if __cutensornetGetCudartVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetCudartVersion is not found")
    return (<size_t (*)() nogil>__cutensornetGetCudartVersion)(
        )


cdef const char* _cutensornetGetErrorString(cutensornetStatus_t error) except* nogil:
    global __cutensornetGetErrorString
    _check_or_init_cutensornet()
    if __cutensornetGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetErrorString is not found")
    return (<const char* (*)(cutensornetStatus_t) nogil>__cutensornetGetErrorString)(
        error)


cdef cutensornetStatus_t _cutensornetDistributedResetConfiguration(cutensornetHandle_t handle, const void* commPtr, size_t commSize) except* nogil:
    global __cutensornetDistributedResetConfiguration
    _check_or_init_cutensornet()
    if __cutensornetDistributedResetConfiguration == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDistributedResetConfiguration is not found")
    return (<cutensornetStatus_t (*)(cutensornetHandle_t, const void*, size_t) nogil>__cutensornetDistributedResetConfiguration)(
        handle, commPtr, commSize)


cdef cutensornetStatus_t _cutensornetDistributedGetNumRanks(const cutensornetHandle_t handle, int32_t* numRanks) except* nogil:
    global __cutensornetDistributedGetNumRanks
    _check_or_init_cutensornet()
    if __cutensornetDistributedGetNumRanks == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDistributedGetNumRanks is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t*) nogil>__cutensornetDistributedGetNumRanks)(
        handle, numRanks)


cdef cutensornetStatus_t _cutensornetDistributedGetProcRank(const cutensornetHandle_t handle, int32_t* procRank) except* nogil:
    global __cutensornetDistributedGetProcRank
    _check_or_init_cutensornet()
    if __cutensornetDistributedGetProcRank == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDistributedGetProcRank is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t*) nogil>__cutensornetDistributedGetProcRank)(
        handle, procRank)


cdef cutensornetStatus_t _cutensornetDistributedSynchronize(const cutensornetHandle_t handle) except* nogil:
    global __cutensornetDistributedSynchronize
    _check_or_init_cutensornet()
    if __cutensornetDistributedSynchronize == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDistributedSynchronize is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t) nogil>__cutensornetDistributedSynchronize)(
        handle)


cdef cutensornetStatus_t _cutensornetNetworkGetAttribute(const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, cutensornetNetworkAttributes_t attr, void* buf, size_t sizeInBytes) except* nogil:
    global __cutensornetNetworkGetAttribute
    _check_or_init_cutensornet()
    if __cutensornetNetworkGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkGetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetNetworkDescriptor_t, cutensornetNetworkAttributes_t, void*, size_t) nogil>__cutensornetNetworkGetAttribute)(
        handle, networkDesc, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetNetworkSetAttribute(const cutensornetHandle_t handle, cutensornetNetworkDescriptor_t networkDesc, cutensornetNetworkAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    global __cutensornetNetworkSetAttribute
    _check_or_init_cutensornet()
    if __cutensornetNetworkSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkSetAttribute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkDescriptor_t, cutensornetNetworkAttributes_t, const void*, size_t) nogil>__cutensornetNetworkSetAttribute)(
        handle, networkDesc, attr, buf, sizeInBytes)


cdef cutensornetStatus_t _cutensornetWorkspacePurgeCache(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t workDesc, cutensornetMemspace_t memSpace) except* nogil:
    global __cutensornetWorkspacePurgeCache
    _check_or_init_cutensornet()
    if __cutensornetWorkspacePurgeCache == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetWorkspacePurgeCache is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetWorkspaceDescriptor_t, cutensornetMemspace_t) nogil>__cutensornetWorkspacePurgeCache)(
        handle, workDesc, memSpace)


cdef cutensornetStatus_t _cutensornetComputeGradientsBackward(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan, const void* const rawDataIn[], const void* outputGradient, void* const gradients[], int32_t accumulateOutput, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) except* nogil:
    global __cutensornetComputeGradientsBackward
    _check_or_init_cutensornet()
    if __cutensornetComputeGradientsBackward == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetComputeGradientsBackward is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetContractionPlan_t, const void* const*, const void*, void* const*, int32_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) nogil>__cutensornetComputeGradientsBackward)(
        handle, plan, rawDataIn, outputGradient, gradients, accumulateOutput, workDesc, stream)


cdef cutensornetStatus_t _cutensornetCreateState(const cutensornetHandle_t handle, cutensornetStatePurity_t purity, int32_t numStateModes, const int64_t* stateModeExtents, cudaDataType_t dataType, cutensornetState_t* tensorNetworkState) except* nogil:
    global __cutensornetCreateState
    _check_or_init_cutensornet()
    if __cutensornetCreateState == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateState is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStatePurity_t, int32_t, const int64_t*, cudaDataType_t, cutensornetState_t*) nogil>__cutensornetCreateState)(
        handle, purity, numStateModes, stateModeExtents, dataType, tensorNetworkState)


cdef cutensornetStatus_t _cutensornetStateApplyTensor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except* nogil:
    global __cutensornetStateApplyTensor
    _check_or_init_cutensornet()
    if __cutensornetStateApplyTensor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyTensor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, void*, const int64_t*, const int32_t, const int32_t, const int32_t, int64_t*) nogil>__cutensornetStateApplyTensor)(
        handle, tensorNetworkState, numStateModes, stateModes, tensorData, tensorModeStrides, immutable, adjoint, unitary, tensorId)


cdef cutensornetStatus_t _cutensornetStateUpdateTensor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int64_t tensorId, void* tensorData, int32_t unitary) except* nogil:
    global __cutensornetStateUpdateTensor
    _check_or_init_cutensornet()
    if __cutensornetStateUpdateTensor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateUpdateTensor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int64_t, void*, int32_t) nogil>__cutensornetStateUpdateTensor)(
        handle, tensorNetworkState, tensorId, tensorData, unitary)


cdef cutensornetStatus_t _cutensornetDestroyState(cutensornetState_t tensorNetworkState) except* nogil:
    global __cutensornetDestroyState
    _check_or_init_cutensornet()
    if __cutensornetDestroyState == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyState is not found")
    return (<cutensornetStatus_t (*)(cutensornetState_t) nogil>__cutensornetDestroyState)(
        tensorNetworkState)


cdef cutensornetStatus_t _cutensornetCreateMarginal(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numMarginalModes, const int32_t* marginalModes, int32_t numProjectedModes, const int32_t* projectedModes, const int64_t* marginalTensorStrides, cutensornetStateMarginal_t* tensorNetworkMarginal) except* nogil:
    global __cutensornetCreateMarginal
    _check_or_init_cutensornet()
    if __cutensornetCreateMarginal == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateMarginal is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, int32_t, const int32_t*, const int64_t*, cutensornetStateMarginal_t*) nogil>__cutensornetCreateMarginal)(
        handle, tensorNetworkState, numMarginalModes, marginalModes, numProjectedModes, projectedModes, marginalTensorStrides, tensorNetworkMarginal)


cdef cutensornetStatus_t _cutensornetMarginalConfigure(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, cutensornetMarginalAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil:
    global __cutensornetMarginalConfigure
    _check_or_init_cutensornet()
    if __cutensornetMarginalConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetMarginalConfigure is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateMarginal_t, cutensornetMarginalAttributes_t, const void*, size_t) nogil>__cutensornetMarginalConfigure)(
        handle, tensorNetworkMarginal, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetMarginalPrepare(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil:
    global __cutensornetMarginalPrepare
    _check_or_init_cutensornet()
    if __cutensornetMarginalPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetMarginalPrepare is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateMarginal_t, size_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) nogil>__cutensornetMarginalPrepare)(
        handle, tensorNetworkMarginal, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetMarginalCompute(const cutensornetHandle_t handle, cutensornetStateMarginal_t tensorNetworkMarginal, const int64_t* projectedModeValues, cutensornetWorkspaceDescriptor_t workDesc, void* marginalTensor, cudaStream_t cudaStream) except* nogil:
    global __cutensornetMarginalCompute
    _check_or_init_cutensornet()
    if __cutensornetMarginalCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetMarginalCompute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateMarginal_t, const int64_t*, cutensornetWorkspaceDescriptor_t, void*, cudaStream_t) nogil>__cutensornetMarginalCompute)(
        handle, tensorNetworkMarginal, projectedModeValues, workDesc, marginalTensor, cudaStream)


cdef cutensornetStatus_t _cutensornetDestroyMarginal(cutensornetStateMarginal_t tensorNetworkMarginal) except* nogil:
    global __cutensornetDestroyMarginal
    _check_or_init_cutensornet()
    if __cutensornetDestroyMarginal == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyMarginal is not found")
    return (<cutensornetStatus_t (*)(cutensornetStateMarginal_t) nogil>__cutensornetDestroyMarginal)(
        tensorNetworkMarginal)


cdef cutensornetStatus_t _cutensornetCreateSampler(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numModesToSample, const int32_t* modesToSample, cutensornetStateSampler_t* tensorNetworkSampler) except* nogil:
    global __cutensornetCreateSampler
    _check_or_init_cutensornet()
    if __cutensornetCreateSampler == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateSampler is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, cutensornetStateSampler_t*) nogil>__cutensornetCreateSampler)(
        handle, tensorNetworkState, numModesToSample, modesToSample, tensorNetworkSampler)


cdef cutensornetStatus_t _cutensornetSamplerConfigure(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, cutensornetSamplerAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil:
    global __cutensornetSamplerConfigure
    _check_or_init_cutensornet()
    if __cutensornetSamplerConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetSamplerConfigure is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateSampler_t, cutensornetSamplerAttributes_t, const void*, size_t) nogil>__cutensornetSamplerConfigure)(
        handle, tensorNetworkSampler, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetSamplerPrepare(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil:
    global __cutensornetSamplerPrepare
    _check_or_init_cutensornet()
    if __cutensornetSamplerPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetSamplerPrepare is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateSampler_t, size_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) nogil>__cutensornetSamplerPrepare)(
        handle, tensorNetworkSampler, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetSamplerSample(const cutensornetHandle_t handle, cutensornetStateSampler_t tensorNetworkSampler, int64_t numShots, cutensornetWorkspaceDescriptor_t workDesc, int64_t* samples, cudaStream_t cudaStream) except* nogil:
    global __cutensornetSamplerSample
    _check_or_init_cutensornet()
    if __cutensornetSamplerSample == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetSamplerSample is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateSampler_t, int64_t, cutensornetWorkspaceDescriptor_t, int64_t*, cudaStream_t) nogil>__cutensornetSamplerSample)(
        handle, tensorNetworkSampler, numShots, workDesc, samples, cudaStream)


cdef cutensornetStatus_t _cutensornetDestroySampler(cutensornetStateSampler_t tensorNetworkSampler) except* nogil:
    global __cutensornetDestroySampler
    _check_or_init_cutensornet()
    if __cutensornetDestroySampler == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroySampler is not found")
    return (<cutensornetStatus_t (*)(cutensornetStateSampler_t) nogil>__cutensornetDestroySampler)(
        tensorNetworkSampler)


cdef cutensornetStatus_t _cutensornetStateFinalizeMPS(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetBoundaryCondition_t boundaryCondition, const int64_t* const extentsOut[], const int64_t* const stridesOut[]) except* nogil:
    global __cutensornetStateFinalizeMPS
    _check_or_init_cutensornet()
    if __cutensornetStateFinalizeMPS == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateFinalizeMPS is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, cutensornetBoundaryCondition_t, const int64_t* const*, const int64_t* const*) nogil>__cutensornetStateFinalizeMPS)(
        handle, tensorNetworkState, boundaryCondition, extentsOut, stridesOut)


cdef cutensornetStatus_t _cutensornetStateConfigure(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetStateAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil:
    global __cutensornetStateConfigure
    _check_or_init_cutensornet()
    if __cutensornetStateConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateConfigure is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, cutensornetStateAttributes_t, const void*, size_t) nogil>__cutensornetStateConfigure)(
        handle, tensorNetworkState, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetStatePrepare(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil:
    global __cutensornetStatePrepare
    _check_or_init_cutensornet()
    if __cutensornetStatePrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStatePrepare is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, size_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) nogil>__cutensornetStatePrepare)(
        handle, tensorNetworkState, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetStateCompute(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetWorkspaceDescriptor_t workDesc, int64_t* extentsOut[], int64_t* stridesOut[], void* stateTensorsOut[], cudaStream_t cudaStream) except* nogil:
    global __cutensornetStateCompute
    _check_or_init_cutensornet()
    if __cutensornetStateCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateCompute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, cutensornetWorkspaceDescriptor_t, int64_t**, int64_t**, void**, cudaStream_t) nogil>__cutensornetStateCompute)(
        handle, tensorNetworkState, workDesc, extentsOut, stridesOut, stateTensorsOut, cudaStream)


cdef cutensornetStatus_t _cutensornetGetOutputStateDetails(const cutensornetHandle_t handle, const cutensornetState_t tensorNetworkState, int32_t* numTensorsOut, int32_t numModesOut[], int64_t* extentsOut[], int64_t* stridesOut[]) except* nogil:
    global __cutensornetGetOutputStateDetails
    _check_or_init_cutensornet()
    if __cutensornetGetOutputStateDetails == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetGetOutputStateDetails is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetState_t, int32_t*, int32_t*, int64_t**, int64_t**) nogil>__cutensornetGetOutputStateDetails)(
        handle, tensorNetworkState, numTensorsOut, numModesOut, extentsOut, stridesOut)


cdef cutensornetStatus_t _cutensornetCreateNetworkOperator(const cutensornetHandle_t handle, int32_t numStateModes, const int64_t stateModeExtents[], cudaDataType_t dataType, cutensornetNetworkOperator_t* tensorNetworkOperator) except* nogil:
    global __cutensornetCreateNetworkOperator
    _check_or_init_cutensornet()
    if __cutensornetCreateNetworkOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateNetworkOperator is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, int32_t, const int64_t*, cudaDataType_t, cutensornetNetworkOperator_t*) nogil>__cutensornetCreateNetworkOperator)(
        handle, numStateModes, stateModeExtents, dataType, tensorNetworkOperator)


cdef cutensornetStatus_t _cutensornetNetworkOperatorAppendProduct(const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator, cuDoubleComplex coefficient, int32_t numTensors, const int32_t numStateModes[], const int32_t* stateModes[], const int64_t* tensorModeStrides[], const void* tensorData[], int64_t* componentId) except* nogil:
    global __cutensornetNetworkOperatorAppendProduct
    _check_or_init_cutensornet()
    if __cutensornetNetworkOperatorAppendProduct == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkOperatorAppendProduct is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkOperator_t, cuDoubleComplex, int32_t, const int32_t*, const int32_t**, const int64_t**, const void**, int64_t*) nogil>__cutensornetNetworkOperatorAppendProduct)(
        handle, tensorNetworkOperator, coefficient, numTensors, numStateModes, stateModes, tensorModeStrides, tensorData, componentId)


cdef cutensornetStatus_t _cutensornetDestroyNetworkOperator(cutensornetNetworkOperator_t tensorNetworkOperator) except* nogil:
    global __cutensornetDestroyNetworkOperator
    _check_or_init_cutensornet()
    if __cutensornetDestroyNetworkOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyNetworkOperator is not found")
    return (<cutensornetStatus_t (*)(cutensornetNetworkOperator_t) nogil>__cutensornetDestroyNetworkOperator)(
        tensorNetworkOperator)


cdef cutensornetStatus_t _cutensornetCreateAccessor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numProjectedModes, const int32_t* projectedModes, const int64_t* amplitudesTensorStrides, cutensornetStateAccessor_t* tensorNetworkAccessor) except* nogil:
    global __cutensornetCreateAccessor
    _check_or_init_cutensornet()
    if __cutensornetCreateAccessor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateAccessor is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, const int64_t*, cutensornetStateAccessor_t*) nogil>__cutensornetCreateAccessor)(
        handle, tensorNetworkState, numProjectedModes, projectedModes, amplitudesTensorStrides, tensorNetworkAccessor)


cdef cutensornetStatus_t _cutensornetAccessorConfigure(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, cutensornetAccessorAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil:
    global __cutensornetAccessorConfigure
    _check_or_init_cutensornet()
    if __cutensornetAccessorConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetAccessorConfigure is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateAccessor_t, cutensornetAccessorAttributes_t, const void*, size_t) nogil>__cutensornetAccessorConfigure)(
        handle, tensorNetworkAccessor, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetAccessorPrepare(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil:
    global __cutensornetAccessorPrepare
    _check_or_init_cutensornet()
    if __cutensornetAccessorPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetAccessorPrepare is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateAccessor_t, size_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) nogil>__cutensornetAccessorPrepare)(
        handle, tensorNetworkAccessor, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetAccessorCompute(const cutensornetHandle_t handle, cutensornetStateAccessor_t tensorNetworkAccessor, const int64_t* projectedModeValues, cutensornetWorkspaceDescriptor_t workDesc, void* amplitudesTensor, void* stateNorm, cudaStream_t cudaStream) except* nogil:
    global __cutensornetAccessorCompute
    _check_or_init_cutensornet()
    if __cutensornetAccessorCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetAccessorCompute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateAccessor_t, const int64_t*, cutensornetWorkspaceDescriptor_t, void*, void*, cudaStream_t) nogil>__cutensornetAccessorCompute)(
        handle, tensorNetworkAccessor, projectedModeValues, workDesc, amplitudesTensor, stateNorm, cudaStream)


cdef cutensornetStatus_t _cutensornetDestroyAccessor(cutensornetStateAccessor_t tensorNetworkAccessor) except* nogil:
    global __cutensornetDestroyAccessor
    _check_or_init_cutensornet()
    if __cutensornetDestroyAccessor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyAccessor is not found")
    return (<cutensornetStatus_t (*)(cutensornetStateAccessor_t) nogil>__cutensornetDestroyAccessor)(
        tensorNetworkAccessor)


cdef cutensornetStatus_t _cutensornetCreateExpectation(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetNetworkOperator_t tensorNetworkOperator, cutensornetStateExpectation_t* tensorNetworkExpectation) except* nogil:
    global __cutensornetCreateExpectation
    _check_or_init_cutensornet()
    if __cutensornetCreateExpectation == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetCreateExpectation is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, cutensornetNetworkOperator_t, cutensornetStateExpectation_t*) nogil>__cutensornetCreateExpectation)(
        handle, tensorNetworkState, tensorNetworkOperator, tensorNetworkExpectation)


cdef cutensornetStatus_t _cutensornetExpectationConfigure(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetExpectationAttributes_t attribute, const void* attributeValue, size_t attributeSize) except* nogil:
    global __cutensornetExpectationConfigure
    _check_or_init_cutensornet()
    if __cutensornetExpectationConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetExpectationConfigure is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateExpectation_t, cutensornetExpectationAttributes_t, const void*, size_t) nogil>__cutensornetExpectationConfigure)(
        handle, tensorNetworkExpectation, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetExpectationPrepare(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) except* nogil:
    global __cutensornetExpectationPrepare
    _check_or_init_cutensornet()
    if __cutensornetExpectationPrepare == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetExpectationPrepare is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateExpectation_t, size_t, cutensornetWorkspaceDescriptor_t, cudaStream_t) nogil>__cutensornetExpectationPrepare)(
        handle, tensorNetworkExpectation, maxWorkspaceSizeDevice, workDesc, cudaStream)


cdef cutensornetStatus_t _cutensornetExpectationCompute(const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetWorkspaceDescriptor_t workDesc, void* expectationValue, void* stateNorm, cudaStream_t cudaStream) except* nogil:
    global __cutensornetExpectationCompute
    _check_or_init_cutensornet()
    if __cutensornetExpectationCompute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetExpectationCompute is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetStateExpectation_t, cutensornetWorkspaceDescriptor_t, void*, void*, cudaStream_t) nogil>__cutensornetExpectationCompute)(
        handle, tensorNetworkExpectation, workDesc, expectationValue, stateNorm, cudaStream)


cdef cutensornetStatus_t _cutensornetDestroyExpectation(cutensornetStateExpectation_t tensorNetworkExpectation) except* nogil:
    global __cutensornetDestroyExpectation
    _check_or_init_cutensornet()
    if __cutensornetDestroyExpectation == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetDestroyExpectation is not found")
    return (<cutensornetStatus_t (*)(cutensornetStateExpectation_t) nogil>__cutensornetDestroyExpectation)(
        tensorNetworkExpectation)


cdef cutensornetStatus_t _cutensornetStateApplyTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes, const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except* nogil:
    global __cutensornetStateApplyTensorOperator
    _check_or_init_cutensornet()
    if __cutensornetStateApplyTensorOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyTensorOperator is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, void*, const int64_t*, const int32_t, const int32_t, const int32_t, int64_t*) nogil>__cutensornetStateApplyTensorOperator)(
        handle, tensorNetworkState, numStateModes, stateModes, tensorData, tensorModeStrides, immutable, adjoint, unitary, tensorId)


cdef cutensornetStatus_t _cutensornetStateApplyControlledTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numControlModes, const int32_t* stateControlModes, const int64_t* stateControlValues, int32_t numTargetModes, const int32_t* stateTargetModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* tensorId) except* nogil:
    global __cutensornetStateApplyControlledTensorOperator
    _check_or_init_cutensornet()
    if __cutensornetStateApplyControlledTensorOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyControlledTensorOperator is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int32_t, const int32_t*, const int64_t*, int32_t, const int32_t*, void*, const int64_t*, const int32_t, const int32_t, const int32_t, int64_t*) nogil>__cutensornetStateApplyControlledTensorOperator)(
        handle, tensorNetworkState, numControlModes, stateControlModes, stateControlValues, numTargetModes, stateTargetModes, tensorData, tensorModeStrides, immutable, adjoint, unitary, tensorId)


cdef cutensornetStatus_t _cutensornetStateUpdateTensorOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int64_t tensorId, void* tensorData, int32_t unitary) except* nogil:
    global __cutensornetStateUpdateTensorOperator
    _check_or_init_cutensornet()
    if __cutensornetStateUpdateTensorOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateUpdateTensorOperator is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, int64_t, void*, int32_t) nogil>__cutensornetStateUpdateTensorOperator)(
        handle, tensorNetworkState, tensorId, tensorData, unitary)


cdef cutensornetStatus_t _cutensornetStateApplyNetworkOperator(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, const cutensornetNetworkOperator_t tensorNetworkOperator, const int32_t immutable, const int32_t adjoint, const int32_t unitary, int64_t* operatorId) except* nogil:
    global __cutensornetStateApplyNetworkOperator
    _check_or_init_cutensornet()
    if __cutensornetStateApplyNetworkOperator == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateApplyNetworkOperator is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, const cutensornetNetworkOperator_t, const int32_t, const int32_t, const int32_t, int64_t*) nogil>__cutensornetStateApplyNetworkOperator)(
        handle, tensorNetworkState, tensorNetworkOperator, immutable, adjoint, unitary, operatorId)


cdef cutensornetStatus_t _cutensornetStateInitializeMPS(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, cutensornetBoundaryCondition_t boundaryCondition, const int64_t* const extentsIn[], const int64_t* const stridesIn[], void* stateTensorsIn[]) except* nogil:
    global __cutensornetStateInitializeMPS
    _check_or_init_cutensornet()
    if __cutensornetStateInitializeMPS == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateInitializeMPS is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetState_t, cutensornetBoundaryCondition_t, const int64_t* const*, const int64_t* const*, void**) nogil>__cutensornetStateInitializeMPS)(
        handle, tensorNetworkState, boundaryCondition, extentsIn, stridesIn, stateTensorsIn)


cdef cutensornetStatus_t _cutensornetStateGetInfo(const cutensornetHandle_t handle, const cutensornetState_t tensorNetworkState, cutensornetStateAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil:
    global __cutensornetStateGetInfo
    _check_or_init_cutensornet()
    if __cutensornetStateGetInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetStateGetInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetState_t, cutensornetStateAttributes_t, void*, size_t) nogil>__cutensornetStateGetInfo)(
        handle, tensorNetworkState, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetNetworkOperatorAppendMPO(const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator, cuDoubleComplex coefficient, int32_t numStateModes, const int32_t stateModes[], const int64_t* tensorModeExtents[], const int64_t* tensorModeStrides[], const void* tensorData[], cutensornetBoundaryCondition_t boundaryCondition, int64_t* componentId) except* nogil:
    global __cutensornetNetworkOperatorAppendMPO
    _check_or_init_cutensornet()
    if __cutensornetNetworkOperatorAppendMPO == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetNetworkOperatorAppendMPO is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, cutensornetNetworkOperator_t, cuDoubleComplex, int32_t, const int32_t*, const int64_t**, const int64_t**, const void**, cutensornetBoundaryCondition_t, int64_t*) nogil>__cutensornetNetworkOperatorAppendMPO)(
        handle, tensorNetworkOperator, coefficient, numStateModes, stateModes, tensorModeExtents, tensorModeStrides, tensorData, boundaryCondition, componentId)


cdef cutensornetStatus_t _cutensornetAccessorGetInfo(const cutensornetHandle_t handle, const cutensornetStateAccessor_t tensorNetworkAccessor, cutensornetAccessorAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil:
    global __cutensornetAccessorGetInfo
    _check_or_init_cutensornet()
    if __cutensornetAccessorGetInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetAccessorGetInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetStateAccessor_t, cutensornetAccessorAttributes_t, void*, size_t) nogil>__cutensornetAccessorGetInfo)(
        handle, tensorNetworkAccessor, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetExpectationGetInfo(const cutensornetHandle_t handle, const cutensornetStateExpectation_t tensorNetworkExpectation, cutensornetExpectationAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil:
    global __cutensornetExpectationGetInfo
    _check_or_init_cutensornet()
    if __cutensornetExpectationGetInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetExpectationGetInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetStateExpectation_t, cutensornetExpectationAttributes_t, void*, size_t) nogil>__cutensornetExpectationGetInfo)(
        handle, tensorNetworkExpectation, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetMarginalGetInfo(const cutensornetHandle_t handle, const cutensornetStateMarginal_t tensorNetworkMarginal, cutensornetMarginalAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil:
    global __cutensornetMarginalGetInfo
    _check_or_init_cutensornet()
    if __cutensornetMarginalGetInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetMarginalGetInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetStateMarginal_t, cutensornetMarginalAttributes_t, void*, size_t) nogil>__cutensornetMarginalGetInfo)(
        handle, tensorNetworkMarginal, attribute, attributeValue, attributeSize)


cdef cutensornetStatus_t _cutensornetSamplerGetInfo(const cutensornetHandle_t handle, const cutensornetStateSampler_t tensorNetworkSampler, cutensornetSamplerAttributes_t attribute, void* attributeValue, size_t attributeSize) except* nogil:
    global __cutensornetSamplerGetInfo
    _check_or_init_cutensornet()
    if __cutensornetSamplerGetInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensornetSamplerGetInfo is not found")
    return (<cutensornetStatus_t (*)(const cutensornetHandle_t, const cutensornetStateSampler_t, cutensornetSamplerAttributes_t, void*, size_t) nogil>__cutensornetSamplerGetInfo)(
        handle, tensorNetworkSampler, attribute, attributeValue, attributeSize)
