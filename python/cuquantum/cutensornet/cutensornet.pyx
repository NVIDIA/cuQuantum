# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# distutils: language = c++

cimport cpython
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport intptr_t, int32_t, uint32_t, int64_t, uint64_t, uintptr_t
from libcpp.vector cimport vector

from cuquantum.utils cimport is_nested_sequence
from cuquantum.utils cimport cuqnt_alloc_wrapper
from cuquantum.utils cimport cuqnt_free_wrapper
from cuquantum.utils cimport get_buffer_pointer
from cuquantum.utils cimport logger_callback_with_data
from cuquantum.utils cimport cuDoubleComplex

from enum import IntEnum
import warnings

import numpy as _numpy


cdef extern from * nogil:
    # cuTensorNet functions
    # library
    int cutensornetCreate(_Handle*)
    int cutensornetDestroy(_Handle)
    size_t cutensornetGetVersion()
    size_t cutensornetGetCudartVersion()
    const char* cutensornetGetErrorString(_Status)

    # network descriptor
    int cutensornetCreateNetworkDescriptor(
        _Handle, int32_t, const int32_t[], const int64_t* const[],
        const int64_t* const[], const int32_t* const[], const _TensorQualifiers[],
        int32_t, const int64_t[], const int64_t[], const int32_t[],
        DataType, _ComputeType, _NetworkDescriptor*)
    int cutensornetDestroyNetworkDescriptor(_NetworkDescriptor)
    int cutensornetNetworkGetAttribute(
        _Handle, _NetworkDescriptor, _NetworkAttribute, void*, size_t)
    int cutensornetNetworkSetAttribute(
        _Handle, _NetworkDescriptor, _NetworkAttribute, void*, size_t)
    int cutensornetGetOutputTensorDetails(
        const _Handle, const _NetworkDescriptor,
        int32_t*, size_t*, int32_t*, int64_t*, int64_t*)
    int cutensornetGetOutputTensorDescriptor(
        const _Handle, const _NetworkDescriptor,
        _TensorDescriptor*)
    int cutensornetGetTensorDetails(
        const _Handle, const _TensorDescriptor,
        int32_t*, size_t*, int32_t*, int64_t*, int64_t*)

    # workspace descriptor
    int cutensornetCreateWorkspaceDescriptor(
        const _Handle, _WorkspaceDescriptor*)
    int cutensornetWorkspaceComputeSizes(
        const _Handle, const _NetworkDescriptor,
        const _ContractionOptimizerInfo, _WorkspaceDescriptor)
    int cutensornetWorkspaceComputeContractionSizes(
        const _Handle, const _NetworkDescriptor,
        const _ContractionOptimizerInfo, _WorkspaceDescriptor)
    int cutensornetWorkspaceGetSize(
        const _Handle, const _WorkspaceDescriptor,
        _WorksizePref, _Memspace, uint64_t*)
    int cutensornetWorkspaceGetMemorySize(
        const _Handle, const _WorkspaceDescriptor,
        _WorksizePref, _Memspace, _WorkspaceKind, int64_t*)
    int cutensornetWorkspaceSet(
        const _Handle, _WorkspaceDescriptor, _Memspace,
        void* const, uint64_t)
    int cutensornetWorkspaceSetMemory(
        const _Handle, _WorkspaceDescriptor, _Memspace,
        _WorkspaceKind, void* const, int64_t)
    int cutensornetWorkspaceGet(
        const _Handle, const _WorkspaceDescriptor, _Memspace,
        void**, uint64_t*)
    int cutensornetWorkspaceGetMemory(
        const _Handle, const _WorkspaceDescriptor, _Memspace,
        _WorkspaceKind, void**, int64_t*)
    int cutensornetWorkspacePurgeCache(
        _Handle, _WorkspaceDescriptor, _Memspace)
    int cutensornetDestroyWorkspaceDescriptor(_WorkspaceDescriptor)

    # optimizer info
    int cutensornetCreateContractionOptimizerInfo(
        const _Handle, const _NetworkDescriptor,
        _ContractionOptimizerInfo*)
    int cutensornetCreateContractionOptimizerInfoFromPackedData(
        const _Handle, const _NetworkDescriptor,
        const void*, size_t, _ContractionOptimizerInfo*)
    int cutensornetUpdateContractionOptimizerInfoFromPackedData(
        const _Handle, const void*, size_t, _ContractionOptimizerInfo)
    int cutensornetDestroyContractionOptimizerInfo(
        _ContractionOptimizerInfo)
    int cutensornetContractionOptimizerInfoGetAttribute(
        const _Handle, const _ContractionOptimizerInfo,
        _ContractionOptimizerInfoAttribute, void*, size_t)
    int cutensornetContractionOptimizerInfoSetAttribute(
        const _Handle, _ContractionOptimizerInfo,
        _ContractionOptimizerInfoAttribute, const void*, size_t)
    int cutensornetContractionOptimizerInfoGetPackedSize(
        const _Handle, const _ContractionOptimizerInfo, size_t*)
    int cutensornetContractionOptimizerInfoPackData(
        const _Handle, const _ContractionOptimizerInfo, void*, size_t)

    # optimizer config
    int cutensornetCreateContractionOptimizerConfig(
        const _Handle, _ContractionOptimizerConfig*)
    int cutensornetDestroyContractionOptimizerConfig(
        _ContractionOptimizerConfig)
    int cutensornetContractionOptimizerConfigGetAttribute(
        const _Handle, _ContractionOptimizerConfig,
        _ContractionOptimizerConfigAttribute, void*, size_t)
    int cutensornetContractionOptimizerConfigSetAttribute(
        const _Handle, _ContractionOptimizerConfig,
        _ContractionOptimizerConfigAttribute, const void*, size_t)

    # pathfinder
    int cutensornetContractionOptimize(
        const _Handle, const _NetworkDescriptor,
        const _ContractionOptimizerConfig,
        uint64_t, _ContractionOptimizerInfo)

    # contraction plan
    int cutensornetCreateContractionPlan(
        const _Handle, const _NetworkDescriptor,
        const _ContractionOptimizerInfo,
        const _WorkspaceDescriptor, _ContractionPlan)
    int cutensornetDestroyContractionPlan(_ContractionPlan)
    int cutensornetContractionAutotune(
        const _Handle, _ContractionPlan, const void* const[],
        void*, const _WorkspaceDescriptor,
        _ContractionAutotunePreference, Stream)
    int cutensornetContraction(
        const _Handle, const _ContractionPlan, const void* const[],
        void*, const _WorkspaceDescriptor,
        int64_t, Stream)
    int cutensornetContractSlices(
        const _Handle, const _ContractionPlan, const void* const[],
        void*, int32_t, const _WorkspaceDescriptor, const _SliceGroup,
        Stream)

    # slice group
    int cutensornetCreateSliceGroupFromIDRange(
        const _Handle, int64_t, int64_t, int64_t, _SliceGroup*)
    int cutensornetCreateSliceGroupFromIDs(
        const _Handle, int64_t*, int64_t*, _SliceGroup*)
    int cutensornetDestroySliceGroup(_SliceGroup)

    # autotune pref
    int cutensornetCreateContractionAutotunePreference(
        const _Handle, _ContractionAutotunePreference*)
    int cutensornetDestroyContractionAutotunePreference(
        _ContractionAutotunePreference)
    int cutensornetContractionAutotunePreferenceGetAttribute(
        const _Handle, _ContractionAutotunePreference,
        _ContractionAutotunePreferenceAttribute, void*, size_t)
    int cutensornetContractionAutotunePreferenceSetAttribute(
        const _Handle, _ContractionAutotunePreference,
        _ContractionAutotunePreferenceAttribute, const void*, size_t)

    # memory handlers
    int cutensornetGetDeviceMemHandler(const _Handle, _DeviceMemHandler*)
    int cutensornetSetDeviceMemHandler(_Handle, const _DeviceMemHandler*)

    # logger
    #int cutensornetLoggerSetCallback(LoggerCallback)
    int cutensornetLoggerSetCallbackData(LoggerCallbackData, void*)
    #int cutensornetLoggerSetFile(FILE*)
    int cutensornetLoggerOpenFile(const char*)
    int cutensornetLoggerSetLevel(int32_t)
    int cutensornetLoggerSetMask(int32_t)
    int cutensornetLoggerForceDisable()

    # tensor descriptor
    int cutensornetCreateTensorDescriptor(
        _Handle, int32_t, const int64_t[], const int64_t[], const int32_t[],
        DataType, _TensorDescriptor*)
    int cutensornetDestroyTensorDescriptor(_TensorDescriptor)

    # svdConfig
    int cutensornetCreateTensorSVDConfig(_Handle, _TensorSVDConfig*)
    int cutensornetDestroyTensorSVDConfig(_TensorSVDConfig)
    int cutensornetTensorSVDConfigGetAttribute(
        _Handle, _TensorSVDConfig, _TensorSVDConfigAttribute, void*, size_t)
    int cutensornetTensorSVDConfigSetAttribute(
        _Handle, _TensorSVDConfig, _TensorSVDConfigAttribute, void*, size_t)

    # svdInfo
    int cutensornetCreateTensorSVDInfo(_Handle, _TensorSVDInfo*)
    int cutensornetDestroyTensorSVDInfo(_TensorSVDInfo)
    int cutensornetTensorSVDInfoGetAttribute(
        _Handle, _TensorSVDInfo, _TensorSVDInfoAttribute, void*, size_t)

    # tensorSVD
    int cutensornetWorkspaceComputeSVDSizes(
        _Handle, _TensorDescriptor, _TensorDescriptor, _TensorDescriptor,
        _TensorSVDConfig, _WorkspaceDescriptor)
    int cutensornetTensorSVD(
        _Handle, _TensorDescriptor, void*, _TensorDescriptor, void*, void*,
        _TensorDescriptor, void*, _TensorSVDConfig, _TensorSVDInfo,
        _WorkspaceDescriptor, Stream)

    # tensorQR
    int cutensornetWorkspaceComputeQRSizes(
        _Handle, _TensorDescriptor, _TensorDescriptor, _TensorDescriptor,
        _WorkspaceDescriptor)
    int cutensornetTensorQR(
        _Handle, _TensorDescriptor, void*, _TensorDescriptor, void*,
        _TensorDescriptor, void*, _WorkspaceDescriptor, Stream)

    # gate split
    int cutensornetWorkspaceComputeGateSplitSizes(
        _Handle, _TensorDescriptor, _TensorDescriptor, _TensorDescriptor,
        _TensorDescriptor, _TensorDescriptor, _GateSplitAlgo,
        _TensorSVDConfig, _ComputeType, _WorkspaceDescriptor)
    int cutensornetGateSplit(
        _Handle, _TensorDescriptor, void*, _TensorDescriptor, void*,
        _TensorDescriptor, void*, _TensorDescriptor, void*, void*,
        _TensorDescriptor, void*, _GateSplitAlgo, _TensorSVDConfig,
        _ComputeType, _TensorSVDInfo, _WorkspaceDescriptor, Stream)

    # distributed
    int cutensornetDistributedResetConfiguration(_Handle, void*, size_t)
    int cutensornetDistributedGetNumRanks(_Handle, int*)
    int cutensornetDistributedGetProcRank(_Handle, int*)
    int cutensornetDistributedSynchronize(_Handle)

    # gradients
    int cutensornetComputeGradientsBackward(
        const _Handle, _ContractionPlan, const void* const[],
        void*, void* const[], int32_t, _WorkspaceDescriptor, Stream)

    # high level API
    # state preparation
    int cutensornetCreateState(
        const _Handle, _StatePurity, int32_t, const int64_t*, 
        DataType, _State*)
    int cutensornetDestroyState(_State)
    int cutensornetStateApplyTensor(
        const _Handle, _State, int32_t, const int32_t*, void*, 
        const int64_t*, const int32_t, const int32_t, const int32_t, int64_t*)
    int cutensornetStateUpdateTensor(
        const _Handle, _State, int64_t, void*, int32_t)
    int cutensornetStateConfigure(
        const _Handle, _State, _StateAttribute, const void*, size_t)
    int cutensornetStatePrepare(
        const _Handle, _State, size_t, _WorkspaceDescriptor, Stream)
    int cutensornetStateCompute(
        const _Handle, _State, _WorkspaceDescriptor, int64_t*[], int64_t*[], void*[], Stream)
    int cutensornetGetOutputStateDetails(
        const _Handle, const _State, int32_t*, int32_t*, int64_t*[], int64_t*[])
    
    # expectation value
    int cutensornetCreateExpectation(
        const _Handle, _State, _NetworkOperator, _StateExpectation*)
    int cutensornetExpectationConfigure(
        const _Handle, _StateExpectation, _ExpectationAttribute, const void*, size_t)
    int cutensornetExpectationPrepare(
        const _Handle, _StateExpectation, size_t, _WorkspaceDescriptor, Stream)
    int cutensornetExpectationCompute(
        const _Handle, _StateExpectation, _WorkspaceDescriptor, void*, void*, Stream)
    int cutensornetDestroyExpectation(_StateExpectation)
    # accessor
    int cutensornetCreateAccessor(
        const _Handle, _State, int32_t, const int32_t*, 
        const int64_t*, _StateAccessor*)
    int cutensornetAccessorConfigure(
        const _Handle, _StateAccessor, _AccessorAttribute, const void*, size_t)
    int cutensornetAccessorPrepare(
        const _Handle, _StateAccessor, size_t, _WorkspaceDescriptor, Stream)
    int cutensornetAccessorCompute(
        const _Handle, _StateAccessor, const int64_t*, _WorkspaceDescriptor, void*, void*, Stream)
    int cutensornetDestroyAccessor(_StateAccessor)

    # marginals
    int cutensornetCreateMarginal(
        const _Handle, _State, int32_t, const int32_t*,
        int32_t, const int32_t*, const int64_t*, _StateMarginal*)
    int cutensornetMarginalConfigure(
        const _Handle, _StateMarginal, _MarginalAttribute, const void*, size_t)
    int cutensornetMarginalPrepare(
        const _Handle, _StateMarginal, size_t, _WorkspaceDescriptor, Stream)
    int cutensornetMarginalCompute(
        const _Handle, _StateMarginal, const int64_t*, _WorkspaceDescriptor, void*, Stream)
    int cutensornetDestroyMarginal(_StateMarginal)

    # sampling
    int cutensornetCreateSampler(
        const _Handle, _State, int32_t, const int32_t*, _StateSampler*)
    int cutensornetSamplerConfigure(
        const _Handle, _StateSampler, _SamplerAttribute, const void*, size_t)
    int cutensornetSamplerPrepare(
        const _Handle, _StateSampler, size_t, _WorkspaceDescriptor, Stream)
    int cutensornetSamplerSample(
        const _Handle, _StateSampler, int64_t,
        _WorkspaceDescriptor, int64_t*, Stream)
    int cutensornetDestroySampler(_StateSampler)

    # mps-specific
    int cutensornetStateFinalizeMPS(
        const _Handle handle, _State, _BoundaryCondition, const int64_t* const[], const int64_t* const[])

    # network operator
    int cutensornetCreateNetworkOperator(
        const _Handle, int32_t, const int64_t[], DataType, _NetworkOperator*)
    int cutensornetNetworkOperatorAppendProduct(
        const _Handle, _NetworkOperator, cuDoubleComplex, int32_t, const int32_t[], const int32_t* const[], const int64_t* const[], const void* const[], int64_t*)
    int cutensornetDestroyNetworkOperator(_NetworkOperator)

class cuTensorNetError(RuntimeError):
    def __init__(self, status):
        self.status = status
        cdef str err = cutensornetGetErrorString(status).decode()
        super().__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


cdef inline check_status(int status):
    if status != 0:
        raise cuTensorNetError(status)


cpdef intptr_t create() except*:
    """Create a cuTensorNet handle.

    Returns:
        intptr_t: the opaque library handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreate`
    """
    cdef _Handle handle
    cdef int status
    with nogil:
        status = cutensornetCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """Destroy a cuTensorNet handle.

    .. seealso:: `cutensornetDestroy`
    """
    # reduce the ref counts of user-provided Python objects:
    # if Python callables are attached to the handle as the handler,
    # we need to decrease the ref count to avoid leaking
    if handle in owner_pyobj:
        del owner_pyobj[handle]

    with nogil:
        status = cutensornetDestroy(<_Handle>handle)
    check_status(status)


cpdef size_t get_version() except*:
    """Query the version of the cuTensorNet library.

    Returns:
        size_t: the library version.

    .. seealso:: `cutensornetGetVersion`
    """
    cdef size_t ver = cutensornetGetVersion()
    return ver


cpdef size_t get_cudart_version() except*:
    """Query the version of the CUDA runtime used to build cuTensorNet.

    Returns:
        size_t: the CUDA runtime version (ex: 11040 for CUDA 11.4).

    .. seealso:: `cutensornetGetCudartVersion`
    """
    cdef size_t ver = cutensornetGetCudartVersion()
    return ver


cpdef intptr_t create_network_descriptor(
        intptr_t handle,
        int32_t n_inputs, n_modes_in, extents_in,
        strides_in, modes_in, qualifiers_in,
        int32_t n_modes_out, extents_out,
        strides_out, modes_out,
        int data_type, int compute_type) except*:
    """Create a tensor network descriptor.

    Args:
        handle (intptr_t): The library handle.
        n_inputs (int): The number of input tensors.
        n_modes_in: A host array of the number of modes for each input tensor.
            It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        extents_in: A host array of extents for each input tensor. It can be

            - an :class:`int` as the pointer address to the nested sequence
            - a Python sequence of :class:`int`, each of which is a pointer address
              to the corresponding tensor's extents
            - a nested Python sequence of :class:`int`

        strides_in: A host array of strides for each input tensor. It can be

            - an :class:`int` as the pointer address to the nested sequence
            - a Python sequence of :class:`int`, each of which is a pointer address
              to the corresponding tensor's strides
            - a nested Python sequence of :class:`int`

        modes_in: A host array of modes for each input tensor. It can be

            - an :class:`int` as the pointer address to the nested sequence
            - a Python sequence of :class:`int`, each of which is a pointer address
              to the corresponding tensor's modes
            - a nested Python sequence of :class:`int`

        qualifiers_in: A host array of qualifiers for each input tensor. It can
            be

            - an :class:`int` as the pointer address to the numpy array with dtype `tensor_qualifiers_dtype`
            - a numpy array with dtype `tensor_qualifiers_dtype`

        n_modes_out (int32_t): The number of modes of the output tensor. If
            this is set to -1 and ``modes_out`` is set to 0 (not provided),
            the output modes will be inferred. If this is set to 0, the
            network is force reduced.
        extents_out: The extents of the output tensor (on host). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        strides_out: The strides of the output tensor (on host). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        modes_out: The modes of the output tensor (on host). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        data_type (cuquantum.cudaDataType): The data type of the input and
            output tensors.
        compute_type (cuquantum.ComputeType): The compute type of the tensor
            contraction.

    Returns:
        intptr_t: An opaque descriptor handle (as Python :class:`int`).

    .. note::
        If ``strides_in`` (``strides_out``) is set to 0 (`NULL`), it means
        the input tensors (output tensor) are in the Fortran layout (F-contiguous).

    .. seealso:: `cutensornetCreateNetworkDescriptor`
    """
    # n_modes_in can be a pointer address, or a Python sequence
    cdef vector[int32_t] numModesInData
    cdef int32_t* numModesInPtr
    if cpython.PySequence_Check(n_modes_in):
        numModesInData = n_modes_in
        numModesInPtr = numModesInData.data()
    else:  # a pointer address
        numModesInPtr = <int32_t*><intptr_t>n_modes_in

    # extents_in can be:
    #   - a plain pointer address
    #   - a Python sequence (of pointer addresses)
    #   - a nested Python sequence (of int64_t)
    # Note: it cannot be a mix of sequences and ints.
    cdef vector[intptr_t] extentsInCData
    cdef int64_t** extentsInPtr
    if is_nested_sequence(extents_in):
        # flatten the 2D sequence
        extentsInPyData = []
        for i in extents_in:
            # too bad a Python list can't hold C++ vectors, so we use NumPy
            # arrays as the container here to keep data alive
            data = _numpy.asarray(i, dtype=_numpy.int64)
            assert data.ndim == 1
            extentsInPyData.append(data)
            extentsInCData.push_back(<intptr_t>data.ctypes.data)
        extentsInPtr = <int64_t**>(extentsInCData.data())
    elif cpython.PySequence_Check(extents_in):
        # handle 1D sequence
        extentsInCData = extents_in
        extentsInPtr = <int64_t**>(extentsInCData.data())
    else:
        # a pointer address, take it as is
        extentsInPtr = <int64_t**><intptr_t>extents_in

    # strides_in can be:
    #   - a plain pointer address
    #   - a Python sequence (of pointer addresses)
    #   - a nested Python sequence (of int64_t)
    # Note: it cannot be a mix of sequences and ints.
    cdef vector[intptr_t] stridesInCData
    cdef int64_t** stridesInPtr
    if is_nested_sequence(strides_in):
        # flatten the 2D sequence
        stridesInPyData = []
        for i in strides_in:
            # too bad a Python list can't hold C++ vectors, so we use NumPy
            # arrays as the container here to keep data alive
            data = _numpy.asarray(i, dtype=_numpy.int64)
            assert data.ndim == 1
            stridesInPyData.append(data)
            stridesInCData.push_back(<intptr_t>data.ctypes.data)
        stridesInPtr = <int64_t**>(stridesInCData.data())
    elif cpython.PySequence_Check(strides_in):
        # handle 1D sequence
        stridesInCData = strides_in
        stridesInPtr = <int64_t**>(stridesInCData.data())
    else:
        # a pointer address, take it as is
        stridesInPtr = <int64_t**><intptr_t>strides_in

    # modes_in can be:
    #   - a plain pointer address
    #   - a Python sequence (of pointer addresses)
    #   - a nested Python sequence (of int32_t)
    # Note: it cannot be a mix of sequences and ints.
    cdef vector[intptr_t] modesInCData
    cdef int32_t** modesInPtr
    if is_nested_sequence(modes_in):
        # flatten the 2D sequence
        modesInPyData = []
        for i in modes_in:
            # too bad a Python list can't hold C++ vectors, so we use NumPy
            # arrays as the container here to keep data alive
            data = _numpy.asarray(i, dtype=_numpy.int32)
            assert data.ndim == 1
            modesInPyData.append(data)
            modesInCData.push_back(<intptr_t>data.ctypes.data)
        modesInPtr = <int32_t**>(modesInCData.data())
    elif cpython.PySequence_Check(modes_in):
        # handle 1D sequence
        modesInCData = modes_in
        modesInPtr = <int32_t**>(modesInCData.data())
    else:
        # a pointer address, take it as is
        modesInPtr = <int32_t**><intptr_t>modes_in

    # qualifiers_in can be a pointer address or a numpy array
    cdef _TensorQualifiers* qualifiersInPtr
    if isinstance(qualifiers_in, _numpy.ndarray):
        assert qualifiers_in.dtype == tensor_qualifiers_dtype
        qualifiersInPtr = <_TensorQualifiers*><intptr_t>qualifiers_in.ctypes.data
    else:
        qualifiersInPtr = <_TensorQualifiers*><intptr_t> qualifiers_in 

    # extents_out can be a pointer address, or a Python sequence
    cdef vector[int64_t] extentsOutData
    cdef int64_t* extentsOutPtr
    if cpython.PySequence_Check(extents_out):
        extentsOutData = extents_out
        extentsOutPtr = extentsOutData.data()
    else:  # a pointer address
        extentsOutPtr = <int64_t*><intptr_t>extents_out

    # strides_out can be a pointer address, or a Python sequence
    cdef vector[int64_t] stridesOutData
    cdef int64_t* stridesOutPtr
    if cpython.PySequence_Check(strides_out):
        stridesOutData = strides_out
        stridesOutPtr = stridesOutData.data()
    else:  # a pointer address
        stridesOutPtr = <int64_t*><intptr_t>strides_out

    # modes_out can be a pointer address, or a Python sequence
    cdef vector[int32_t] modesOutData
    cdef int32_t* modesOutPtr
    if cpython.PySequence_Check(modes_out):
        modesOutData = modes_out
        modesOutPtr = modesOutData.data()
    else:  # a pointer address
        modesOutPtr = <int32_t*><intptr_t>modes_out

    cdef _NetworkDescriptor tn_desc
    with nogil:
        status = cutensornetCreateNetworkDescriptor(<_Handle>handle,
            n_inputs, numModesInPtr, extentsInPtr, stridesInPtr, modesInPtr, qualifiersInPtr,
            n_modes_out, extentsOutPtr, stridesOutPtr, modesOutPtr,
            <DataType>data_type, <_ComputeType>compute_type, &tn_desc)
    check_status(status)
    return <intptr_t>tn_desc


cpdef destroy_network_descriptor(intptr_t tn_desc):
    """Destroy a tensor network descriptor.

    Args:
        tn_desc (intptr_t): The tensor network descriptor.

    .. seealso:: `cutensornetDestroyNetworkDescriptor`
    """
    with nogil:
        status = cutensornetDestroyNetworkDescriptor(<_NetworkDescriptor>tn_desc)
    check_status(status)


######################### Python specific utility #########################

tensor_id_list_dtype = _numpy.dtype(
    {'names':['num_tensors','data'],
     'formats': (_numpy.int32, _numpy.intp),
     'itemsize': sizeof(_TensorIDList),
    }, align=True
)

cdef dict network_sizes = {
    CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONSTANT: _numpy.int32,
    CUTENSORNET_NETWORK_INPUT_TENSORS_CONSTANT: tensor_id_list_dtype,
    CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONJUGATED: _numpy.int32,
    CUTENSORNET_NETWORK_INPUT_TENSORS_CONJUGATED: tensor_id_list_dtype,
    CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_REQUIRE_GRAD: _numpy.int32,
    CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD: tensor_id_list_dtype,
}

cpdef network_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding network descriptor
    attribute.

    Args:
        attr (NetworkAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`network_get_attribute` and
        :func:`network_set_attribute`.
    """
    return network_sizes[attr]

###########################################################################


cpdef network_get_attribute(
        intptr_t handle, intptr_t tn_desc, int attr,
        intptr_t buf, size_t size):
    """Get the network descriptor attribute.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): The tensor network descriptor.
        attr (NetworkAttribute): The attribute to query.
        buf (intptr_t): The pointer address (as Python :class:`int`) for storing
            the returned attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`network_get_attribute_dtype`.

    .. seealso:: `cutensornetNetworkGetAttribute`
    """
    with nogil:
        status = cutensornetNetworkGetAttribute(
            <_Handle>handle, <_NetworkDescriptor>tn_desc,
            <_NetworkAttribute>attr, <void*>buf, size)
    check_status(status)


cpdef network_set_attribute(
        intptr_t handle, intptr_t tn_desc, int attr,
        intptr_t buf, size_t size):
    """Set the network descriptor attribute.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): The tensor network descriptor.
        attr (NetworkAttribute): The attribute to set.
        buf (intptr_t): The pointer address (as Python :class:`int`) to the
            attribute data.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`network_get_attribute_dtype`.

    .. seealso:: `cutensornetNetworkSetAttribute`
    """
    with nogil:
        status = cutensornetNetworkSetAttribute(
            <_Handle>handle, <_NetworkDescriptor>tn_desc,
            <_NetworkAttribute>attr, <void*>buf, size)
    check_status(status)


cpdef tuple get_output_tensor_details(intptr_t handle, intptr_t tn_desc):
    """Get the output tensor's metadata.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): The tensor network descriptor.

    Returns:
        tuple:
            The metadata of the output tensor: ``(num_modes, modes, extents,
            strides)``.

    .. seealso:: `cutensornetGetOutputTensorDetails`
    """
    warnings.warn("cuquantum.cutensornet.get_output_tensor_details() is "
                  "deprecated and will be removed in a future release; please "
                  "switch to cuquantum.cutensornet.get_output_tensor_descriptor() "
                  "instead", DeprecationWarning, 2)
    cdef int32_t numModesOut = 0
    with nogil:
        status = cutensornetGetOutputTensorDetails(
            <_Handle>handle, <_NetworkDescriptor>tn_desc,
            &numModesOut, NULL, NULL, NULL, NULL)
    check_status(status)
    modes = _numpy.empty(numModesOut, dtype=_numpy.int32)
    extents = _numpy.empty(numModesOut, dtype=_numpy.int64)
    strides = _numpy.empty(numModesOut, dtype=_numpy.int64)
    cdef int32_t* mPtr = <int32_t*><intptr_t>modes.ctypes.data
    cdef int64_t* ePtr = <int64_t*><intptr_t>extents.ctypes.data
    cdef int64_t* sPtr = <int64_t*><intptr_t>strides.ctypes.data
    with nogil:
        status = cutensornetGetOutputTensorDetails(
            <_Handle>handle, <_NetworkDescriptor>tn_desc,
            &numModesOut, NULL, mPtr, ePtr, sPtr)
    check_status(status)
    return (numModesOut, modes, extents, strides)

cpdef intptr_t get_output_tensor_descriptor(
        intptr_t handle, intptr_t tn_desc) except*:
    """Get the networks output tensor descriptor.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): The tensor network descriptor.

    Returns:
        intptr_t: An opaque descriptor handle (as Python :class:`int`).
            Users are responsible to call :func:`destroy_tensor_descriptor` to
            clean it up. 

    .. seealso:: `cutensornetGetOutputTensorDescriptor`
    """
    cdef _TensorDescriptor desc
    with nogil:
        status = cutensornetGetOutputTensorDescriptor(
            <_Handle>handle, <_NetworkDescriptor>tn_desc, &desc)
    check_status(status)
    return <intptr_t>desc


cpdef tuple get_tensor_details(intptr_t handle, intptr_t desc):
    """Get the tensor's metadata.

    Args:
        handle (intptr_t): The library handle.
        desc (intptr_t): A tensor descriptor.

    Returns:
        tuple:
            The metadata of the tensor: ``(num_modes, modes, extents,
            strides)``.

    .. seealso:: `cutensornetGetTensorDetails`

    """
    cdef int32_t numModesOut = 0
    with nogil:
        status = cutensornetGetTensorDetails(
            <_Handle>handle, <_TensorDescriptor>desc,
            &numModesOut, NULL, NULL, NULL, NULL)
    check_status(status)
    modes = _numpy.empty(numModesOut, dtype=_numpy.int32)
    extents = _numpy.empty(numModesOut, dtype=_numpy.int64)
    strides = _numpy.empty(numModesOut, dtype=_numpy.int64)
    cdef int32_t* mPtr = <int32_t*><intptr_t>modes.ctypes.data
    cdef int64_t* ePtr = <int64_t*><intptr_t>extents.ctypes.data
    cdef int64_t* sPtr = <int64_t*><intptr_t>strides.ctypes.data
    with nogil:
        status = cutensornetGetTensorDetails(
            <_Handle>handle, <_TensorDescriptor>desc,
            &numModesOut, NULL, mPtr, ePtr, sPtr)
    check_status(status)
    return (numModesOut, modes, extents, strides)


cpdef intptr_t create_workspace_descriptor(intptr_t handle) except*:
    """Create a workspace descriptor.

    Args:
        handle (intptr_t): The library handle.

    Returns:
        intptr_t: An opaque workspace descriptor (as Python :class:`int`).

    .. seealso:: `cutensornetCreateWorkspaceDescriptor`
    """
    cdef _WorkspaceDescriptor workspace
    with nogil:
        status = cutensornetCreateWorkspaceDescriptor(
            <_Handle>handle, <_WorkspaceDescriptor*>&workspace)
    check_status(status)
    return <intptr_t>workspace


cpdef destroy_workspace_descriptor(intptr_t workspace):
    """Destroy a workspace descriptor.

    Args:
        workspace (intptr_t): The workspace descriptor.

    .. seealso:: `cutensornetDestroyWorkspaceDescriptor`
    """
    with nogil:
        status = cutensornetDestroyWorkspaceDescriptor(
            <_WorkspaceDescriptor>workspace)
    check_status(status)


cpdef workspace_compute_sizes(
        intptr_t handle, intptr_t tn_desc, intptr_t info, intptr_t workspace):
    """Compute the required workspace sizes.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): The tensor network descriptor.
        info (intptr_t): The optimizer info handle.
        workspace (intptr_t): The workspace descriptor.
    
    .. warning::

       This function is deprecated and will be removed in a future release.
       Use :func:`workspace_compute_contraction_sizes` instead.

    .. seealso:: `cutensornetWorkspaceComputeSizes`
    """
    warnings.warn("cuquantum.cutensornet.workspace_compute_sizes() is deprecated and will "
                  "be removed in the future; please switch to "
                  "cuquantum.cutensornet.workspace_compute_contraction_sizes() instead",
                  DeprecationWarning, 2)
    with nogil:
        status = cutensornetWorkspaceComputeSizes(
            <_Handle>handle, <_NetworkDescriptor>tn_desc,
            <_ContractionOptimizerInfo>info,
            <_WorkspaceDescriptor>workspace)
    check_status(status)


cpdef workspace_compute_contraction_sizes(
        intptr_t handle, intptr_t tn_desc, intptr_t info, intptr_t workspace):
    """Compute the required workspace sizes for tensor network contraction.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): The tensor network descriptor.
        info (intptr_t): The optimizer info handle.
        workspace (intptr_t): The workspace descriptor.
    
    .. seealso:: `cutensornetWorkspaceComputeContractionSizes`
    """
    with nogil:
        status = cutensornetWorkspaceComputeContractionSizes(
            <_Handle>handle, <_NetworkDescriptor>tn_desc,
            <_ContractionOptimizerInfo>info,
            <_WorkspaceDescriptor>workspace)
    check_status(status)


cpdef uint64_t workspace_get_size(
        intptr_t handle, intptr_t workspace, int pref, int mem_space) except*:
    """Get the workspace size for the corresponding preference and memory
    space. Must be called after :func:`workspace_compute_sizes`.

    Args:
        handle (intptr_t): The library handle.
        workspace (intptr_t): The workspace descriptor.
        pref (WorksizePref): The preference for the workspace size.
        mem_space (Memspace): The memory space for the workspace being
            queried.
 
    Returns:
        uint64_t: The computed workspace size.

    .. seealso:: `cutensornetWorkspaceGetSize`.
    """
    warnings.warn("cuquantum.cutensornet.workspace_get_size() is "
                  "deprecated and will be removed in a future release; please "
                  "switch to cuquantum.cutensornet.workspace_get_memory_size() "
                  "instead", DeprecationWarning, 2)
    cdef uint64_t workspaceSize
    with nogil:
        status = cutensornetWorkspaceGetSize(
            <_Handle>handle, <_WorkspaceDescriptor>workspace,
            <_WorksizePref>pref, <_Memspace>mem_space,
            &workspaceSize)
    check_status(status)
    return workspaceSize

cpdef int64_t workspace_get_memory_size(
        intptr_t handle, intptr_t workspace, int pref, int mem_space, int kind) except*:
    """Get the workspace size for the corresponding preference and memory
    space. Must be called after :func:`workspace_compute_sizes`.

    Args:
        handle (intptr_t): The library handle.
        workspace (intptr_t): The workspace descriptor.
        pref (WorksizePref): The preference for the workspace size.
        mem_space (Memspace): The memory space for the workspace being
            queried.
        kind (WorkspaceKind): The kind of the workspace being queried
 
    Returns:
        int64_t: The computed workspace size.

    .. seealso:: `cutensornetWorkspaceGetMemorySize`.
    """
    cdef int64_t workspaceSize
    with nogil:
        status = cutensornetWorkspaceGetMemorySize(
            <_Handle>handle, <_WorkspaceDescriptor>workspace,
            <_WorksizePref>pref, <_Memspace>mem_space,
            <_WorkspaceKind>kind,
            &workspaceSize)
    check_status(status)
    return workspaceSize


cpdef workspace_set(
        intptr_t handle, intptr_t workspace, int mem_space,
        intptr_t workspace_ptr, uint64_t workspace_size):
    """Set the workspace pointer and size for the corresponding memory space
    in the workspace descriptor for later use.

    Args:
        handle (intptr_t): The library handle.
        workspace (intptr_t): The workspace descriptor.
        mem_space (Memspace): The memory space for the workspace being
            queried.
        workspace_ptr (intptr_t): The pointer address to the workspace.
        workspace_size (uint64_t): The size of the workspace.

    .. seealso:: `cutensornetWorkspaceSet`
    """
    warnings.warn("cuquantum.cutensornet.workspace_set() is "
                  "deprecated and will be removed in a future release; please "
                  "switch to cuquantum.cutensornet.workspace_set_memory() "
                  "instead", DeprecationWarning, 2)
    with nogil:
        status = cutensornetWorkspaceSet(
            <_Handle>handle, <_WorkspaceDescriptor>workspace,
            <_Memspace>mem_space,
            <void*>workspace_ptr, <uint64_t>workspace_size)
    check_status(status)


cpdef workspace_set_memory(
        intptr_t handle, intptr_t workspace, int mem_space, int kind,
        intptr_t workspace_ptr, int64_t workspace_size):
    """Set the workspace pointer and size for the corresponding 
    memory space and workspace kind
    in the workspace descriptor for later use.

    Args:
        handle (intptr_t): The library handle.
        workspace (intptr_t): The workspace descriptor.
        mem_space (Memspace): The memory space for the workspace being
            queried.
        kind (WorkspaceKind): The kind of the workspace being queried
        workspace_ptr (intptr_t): The pointer address to the workspace.
        workspace_size (int64_t): The size of the workspace.

    .. seealso:: `cutensornetWorkspaceSetMemory`
    """
    with nogil:
        status = cutensornetWorkspaceSetMemory(
            <_Handle>handle, <_WorkspaceDescriptor>workspace,
            <_Memspace>mem_space, <_WorkspaceKind>kind,
            <void*>workspace_ptr, <int64_t>workspace_size)
    check_status(status)


cpdef tuple workspace_get(
        intptr_t handle, intptr_t workspace, int mem_space):
    """Get the workspace pointer and size for the corresponding memory space
    that are set in a workspace descriptor.

    Args:
        handle (intptr_t): The library handle.
        workspace (intptr_t): The workspace descriptor.
        mem_space (Memspace): The memory space for the workspace being
            queried.

    Returns:
        tuple:
            A 2-tuple ``(workspace_ptr, workspace_size)`` for the pointer
            address to the workspace and the size of it.

    .. seealso:: `cutensornetWorkspaceGet`
    """
    warnings.warn("cuquantum.cutensornet.workspace_get() is "
                  "deprecated and will be removed in a future release; please "
                  "switch to cuquantum.cutensornet.workspace_get_memory() "
                  "instead", DeprecationWarning, 2)
    cdef void* workspace_ptr
    cdef uint64_t workspace_size

    with nogil:
        status = cutensornetWorkspaceGet(
            <_Handle>handle, <_WorkspaceDescriptor>workspace,
            <_Memspace>mem_space,
            &workspace_ptr, &workspace_size)
    check_status(status)
    return (<intptr_t>workspace_ptr, workspace_size)


cpdef tuple workspace_get_memory(
        intptr_t handle, intptr_t workspace, int mem_space, int kind):
    """Get the workspace pointer and size for the corresponding 
    memory space and workspace kind
    that are set in a workspace descriptor.

    Args:
        handle (intptr_t): The library handle.
        workspace (intptr_t): The workspace descriptor.
        mem_space (Memspace): The memory space for the workspace being
            queried.
        kind (WorkspaceKind): The kind of the workspace being queried

    Returns:
        tuple:
            A 2-tuple ``(workspace_ptr, workspace_size)`` for the pointer
            address to the workspace and the size of it.

    .. seealso:: `cutensornetWorkspaceGetMemory`
    """
    cdef void* workspace_ptr
    cdef int64_t workspace_size

    with nogil:
        status = cutensornetWorkspaceGetMemory(
            <_Handle>handle, <_WorkspaceDescriptor>workspace,
            <_Memspace>mem_space, <_WorkspaceKind>kind,
            &workspace_ptr, &workspace_size)
    check_status(status)
    return (<intptr_t>workspace_ptr, workspace_size)


cpdef workspace_purge_cache(
        intptr_t handle, intptr_t workspace, int mem_space):
    """Purge the cached data in the specified memory space.

    Args:
        handle (intptr_t): The library handle.
        workspace (intptr_t): The workspace descriptor.
        mem_space (Memspace): The memory space for the workspace being
            queried.

    .. seealso:: `cutensornetWorkspacePurgeCache`
    """
    with nogil:
        status = cutensornetWorkspacePurgeCache(
            <_Handle>handle, <_WorkspaceDescriptor>workspace,
            <_Memspace>mem_space)
    check_status(status)


cpdef intptr_t create_contraction_optimizer_info(
        intptr_t handle, intptr_t tn_desc) except*:
    """Create a contraction optimizer info object.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): The tensor network descriptor.

    Returns:
        intptr_t: An opaque optimizer info handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateContractionOptimizerInfo`
    """
    cdef _ContractionOptimizerInfo info
    with nogil:
        status = cutensornetCreateContractionOptimizerInfo(
            <_Handle>handle, <_NetworkDescriptor>tn_desc, &info)
    check_status(status)
    return <intptr_t>info


cpdef intptr_t create_contraction_optimizer_info_from_packed_data(
        intptr_t handle, intptr_t tn_desc, buf, size_t size) except*:
    """Create a contraction optimizer info object from the packed data.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): The tensor network descriptor.
        buf: A contiguous host buffer holding the packed optimizer info. It can
            be

            - an :class:`int` as the pointer address to the buffer
            - any Python object supporting the Python buffer protocol. In
              this case, it must be 1D and writable; the format would be
              ignored.

        size (size_t): The buffer size in bytes.

    Returns:
        intptr_t: An opaque optimizer info handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateContractionOptimizerInfoFromPackedData`
    """
    cdef _ContractionOptimizerInfo info
    cdef void* bufPtr = get_buffer_pointer(buf, size)
    with nogil:
        status = cutensornetCreateContractionOptimizerInfoFromPackedData(
            <_Handle>handle, <_NetworkDescriptor>tn_desc, bufPtr, size, &info)
    check_status(status)
    return <intptr_t>info


cpdef update_contraction_optimizer_info_from_packed_data(
        intptr_t handle, buf, size_t size, intptr_t info):
    """Update an existing contraction optimizer info object from the packed
    data.

    Args:
        handle (intptr_t): The library handle.
        buf: A contiguous host buffer holding the packed optimizer info. It can
            be

            - an :class:`int` as the pointer address to the buffer
            - any Python object supporting the Python buffer protocol. In
              this case, it must be 1D and writable; the format would be
              ignored.

        size (size_t): The buffer size in bytes.
        info (intptr_t): An opaque optimizer info handle (as Python
            :class:`int`). It must already be in a valid state (i.e.,
            initialized). Upon completion of this function, the ``info``
            is updated (in place).

    .. seealso:: `cutensornetUpdateContractionOptimizerInfoFromPackedData`
    """
    cdef void* bufPtr = get_buffer_pointer(buf, size)
    with nogil:
        status = cutensornetUpdateContractionOptimizerInfoFromPackedData(
            <_Handle>handle, bufPtr, size, <_ContractionOptimizerInfo>info)
    check_status(status)


cpdef destroy_contraction_optimizer_info(intptr_t info):
    """Destroy a contraction optimizer info object.

    Args:
        info (intptr_t): The optimizer info handle.

    .. seealso:: `cutensornetDestroyContractionOptimizerInfo`
    """
    with nogil:
        status = cutensornetDestroyContractionOptimizerInfo(
            <_ContractionOptimizerInfo>info)
    check_status(status)


######################### Python specific utility #########################

contraction_path_dtype = _numpy.dtype(
    {'names':['num_contractions','data'],
     'formats': (_numpy.uint32, _numpy.intp),
     'itemsize': sizeof(_ContractionPath),
    }, align=True
)

# We need this dtype because its members are not of the same type...
slice_info_pair_dtype = _numpy.dtype(
    {'names': ('sliced_mode','sliced_extent'),
     'formats': (_numpy.int32, _numpy.int64),
     'itemsize': sizeof(_SliceInfoPair),
    }, align=True
)

slicing_config_dtype = _numpy.dtype(
    {'names': ('num_sliced_modes','data'),
     'formats': (_numpy.uint32, _numpy.intp),
     'itemsize': sizeof(_SlicingConfig),
    }, align=True
)

gesvdj_params_dtype = _numpy.dtype(
    {'names': ('tol','max_sweeps'),
     'formats': (_numpy.float64, _numpy.int32),
     'itemsize': sizeof(_GesvdjParams),
    }, align=True
)

gesvdr_params_dtype = _numpy.dtype(
    {'names': ('oversampling','niters'),
     'formats': (_numpy.int64, _numpy.int64),
     'itemsize': sizeof(_GesvdrParams),
    }, align=True
)

gesvdj_status_dtype = _numpy.dtype(
    {'names': ('residual', 'sweeps'),
     'formats': (_numpy.float64, _numpy.int32),
     'itemsize': sizeof(_GesvdjStatus),
    }, align=True
)

gesvdp_status_dtype = _numpy.dtype(
    {'names': ('err_sigma', ),
     'formats': (_numpy.float64, ),
     'itemsize': sizeof(_GesvdpStatus),
    }, align=True
)

cdef dict contract_opti_info_sizes = {
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES: _numpy.int64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT: _numpy.int64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH: contraction_path_dtype,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PHASE1_FLOP_COUNT: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_OVERHEAD: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_INTERMEDIATE_MODES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS_EST: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_RUNTIME_EST: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG: slicing_config_dtype,
}

cpdef contraction_optimizer_info_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding optimizer info attribute.

    Args:
        attr (ContractionOptimizerInfoAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`contraction_optimizer_info_get_attribute`
        and :func:`contraction_optimizer_info_set_attribute`.

    .. note:: Unlike other enum values, for :data:`ContractionOptimizerInfoAttribute.PATH`
        the following usage pattern is expected:

        .. code-block:: python

            val = ContractionOptimizerInfoAttribute.PATH
            dtype = contraction_optimizer_info_get_attribute_dtype(val)

            # for setting a path
            path = np.asarray([(1, 3), (1, 2), (0, 1)], dtype=np.int32)
            # ... or for getting a path; note that num_contractions is the number of
            # input tensors minus one
            path = np.empty(2*num_contractions, dtype=np.int32)

            path_obj = np.zeros((1,), dtype=dtype)
            path_obj["num_contractions"] = path.size // 2
            path_obj["data"] = path.ctypes.data

            # for setting a path
            contraction_optimizer_info_set_attribute(
                handle, info, val, path_obj.ctypes.data, path_obj.dtype.itemsize)

            # for getting a path
            contraction_optimizer_info_get_attribute(
                handle, info, val, path_obj.ctypes.data, path_obj.dtype.itemsize)
            # now path is filled
            print(path)

    """
    return contract_opti_info_sizes[attr]

###########################################################################


cpdef contraction_optimizer_info_get_attribute(
        intptr_t handle, intptr_t info, int attr,
        intptr_t buf, size_t size):
    """Get the optimizer info attribute.

    Args:
        handle (intptr_t): The library handle.
        info (intptr_t): The optimizer info handle.
        attr (ContractionOptimizerInfoAttribute): The attribute to query.
        buf (intptr_t): The pointer address (as Python :class:`int`) for storing
            the returned attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`contraction_optimizer_info_get_attribute_dtype`.

    .. seealso:: `cutensornetContractionOptimizerInfoGetAttribute`
    """
    with nogil:
        status = cutensornetContractionOptimizerInfoGetAttribute(
            <_Handle>handle, <_ContractionOptimizerInfo>info,
            <_ContractionOptimizerInfoAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef contraction_optimizer_info_set_attribute(
        intptr_t handle, intptr_t info, int attr,
        intptr_t buf, size_t size):
    """Set the optimizer info attribute.

    Args:
        handle (intptr_t): The library handle.
        info (intptr_t): The optimizer info handle.
        attr (ContractionOptimizerInfoAttribute): The attribute to set.
        buf (intptr_t): The pointer address (as Python :class:`int`) to the attribute data.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`contraction_optimizer_info_get_attribute_dtype`.

    .. seealso:: `cutensornetContractionOptimizerInfoSetAttribute`
    """
    with nogil:
        status = cutensornetContractionOptimizerInfoSetAttribute(
            <_Handle>handle, <_ContractionOptimizerInfo>info,
            <_ContractionOptimizerInfoAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef size_t contraction_optimizer_info_get_packed_size(
        intptr_t handle, intptr_t info) except*:
    """Get the required buffer size for packing the optimizer info object.

    Args:
        handle (intptr_t): The library handle.
        info (intptr_t): An opaque optimizer info handle (as Python
            :class:`int`). It must already be in a valid state (i.e.,
            initialized).

    Returns:
        size_t: The buffer size.

    .. seealso:: `cutensornetContractionOptimizerInfoGetPackedSize`
    """
    cdef size_t size
    with nogil:
        status = cutensornetContractionOptimizerInfoGetPackedSize(
            <_Handle>handle, <_ContractionOptimizerInfo>info, &size)
    check_status(status)
    return size


cpdef contraction_optimizer_info_pack_data(
        intptr_t handle, intptr_t info, buf, size_t size):
    """Pack the contraction optimizer info data into a contiguous buffer.

    Args:
        handle (intptr_t): The library handle.
        info (intptr_t): An opaque optimizer info handle (as Python
            :class:`int`). It must already be in a valid state (i.e.,
            initialized).
        buf: A contiguous host buffer of ``size`` bytes for holding the packed
            optimizer info. It can be

            - an :class:`int` as the pointer address to the buffer
            - any Python object supporting the Python buffer protocol. In
              this case, it must be 1D and writable; the format would be
              ignored.

        size (size_t): The buffer size in bytes as returned by
            :func:`contraction_optimizer_info_get_packed_size`.

    .. seealso:: `cutensornetContractionOptimizerInfoPackData`
    """
    cdef void* bufPtr = get_buffer_pointer(buf, size)

    with nogil:
        status = cutensornetContractionOptimizerInfoPackData(
            <_Handle>handle, <_ContractionOptimizerInfo>info,
            bufPtr, size)
    check_status(status)


cpdef intptr_t create_contraction_optimizer_config(
        intptr_t handle) except*:
    """Create a contraction optimizer config object.

    Args:
        handle (intptr_t): The library handle.

    Returns:
        intptr_t: An opaque optimizer config handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateContractionOptimizerConfig`
    """
    cdef _ContractionOptimizerConfig config
    with nogil:
        status = cutensornetCreateContractionOptimizerConfig(
            <_Handle>handle, &config)
    check_status(status)
    return <intptr_t>config


cpdef destroy_contraction_optimizer_config(intptr_t config):
    """Destroy a contraction optimizer config object.

    Args:
        config (intptr_t): The optimizer config handle.

    .. seealso:: `cutensornetDestroyContractionOptimizerConfig`
    """
    with nogil:
        status = cutensornetDestroyContractionOptimizerConfig(
            <_ContractionOptimizerConfig>config)
    check_status(status)


######################### Python specific utility #########################

cdef dict contract_opti_cfg_sizes = {
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_PARTITIONS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_CUTOFF_SIZE: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM: _numpy.int32,  # = sizeof(enum value)
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_ITERATIONS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL: _numpy.int32,  # = sizeof(enum value)
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_SLICE_FACTOR: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE: _numpy.int32,  # = sizeof(enum value)
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION: _numpy.int32,  # = sizeof(enum value)
}

cpdef contraction_optimizer_config_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding optimizer config attribute.

    Args:
        attr (ContractionOptimizerConfigAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`contraction_optimizer_config_get_attribute`
        and :func:`contraction_optimizer_config_set_attribute`.
    """
    dtype = contract_opti_cfg_sizes[attr]
    if attr == CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM:
        if _numpy.dtype(dtype).itemsize != sizeof(_GraphAlgo):
            warnings.warn("binary size may be incompatible")
    elif attr == CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL:
        if _numpy.dtype(dtype).itemsize != sizeof(_MemoryModel):
            warnings.warn("binary size may be incompatible")
    elif attr == CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE:
        if _numpy.dtype(dtype).itemsize != sizeof(_OptimizerCost):
            warnings.warn("binary size may be incompatible")
    elif attr == CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION:
        if _numpy.dtype(dtype).itemsize != sizeof(_SmartOption):
            warnings.warn("binary size may be incompatible")
    return dtype

###########################################################################


cpdef contraction_optimizer_config_get_attribute(
        intptr_t handle, intptr_t config, int attr,
        intptr_t buf, size_t size):
    """Get the optimizer config attribute.

    Args:
        handle (intptr_t): The library handle.
        config (intptr_t): The optimizer config handle.
        attr (ContractionOptimizerConfigAttribute): The attribute to set.
        buf (intptr_t): The pointer address (as Python :class:`int`) for storing
            the returned attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`contraction_optimizer_config_get_attribute_dtype`.

    .. seealso:: `cutensornetContractionOptimizerConfigGetAttribute`
    """
    with nogil:
        status = cutensornetContractionOptimizerConfigGetAttribute(
            <_Handle>handle, <_ContractionOptimizerConfig>config,
            <_ContractionOptimizerConfigAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef contraction_optimizer_config_set_attribute(
        intptr_t handle, intptr_t config, int attr,
        intptr_t buf, size_t size):
    """Set the optimizer config attribute.

    Args:
        handle (intptr_t): The library handle.
        config (intptr_t): The optimizer config handle.
        attr (ContractionOptimizerConfigAttribute): The attribute to set.
        buf (intptr_t): The pointer address (as Python :class:`int`) to the attribute data.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`contraction_optimizer_config_get_attribute_dtype`.

    .. seealso:: `cutensornetContractionOptimizerConfigSetAttribute`
    """
    with nogil:
        status = cutensornetContractionOptimizerConfigSetAttribute(
            <_Handle>handle, <_ContractionOptimizerConfig>config,
            <_ContractionOptimizerConfigAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef contraction_optimize(
        intptr_t handle, intptr_t tn_desc, intptr_t config,
        uint64_t size_limit, intptr_t info):
    """Optimize the contraction path, slicing, etc, for the given tensor network.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): The tensor network descriptor.
        config (intptr_t): The optimizer config handle.
        size_limit (uint64_t): Maximal device memory that is available to the
            user.
        info (intptr_t): The optimizer info handle.

    .. note:: The ``size_limit`` argument here should not be confused with the
        workspace size returned by :func:`contraction_get_workspace_size`. The
        former is an upper bound for the available memory, whereas the latter
        is the needed size to perform the actual contraction.

    .. seealso:: `cutensornetContractionOptimize`
    """
    with nogil:
        status = cutensornetContractionOptimize(
            <_Handle>handle, <_NetworkDescriptor>tn_desc,
            <_ContractionOptimizerConfig>config,
            size_limit, <_ContractionOptimizerInfo>info)
    check_status(status)


cpdef intptr_t create_contraction_plan(
        intptr_t handle, intptr_t tn_desc, intptr_t info,
        intptr_t workspace) except*:
    """Create a contraction plan for the given tensor network and the
    associated path.

    When this function is called, the optimizer info object should already
    contain a contraction path.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): The tensor network descriptor.
        info (intptr_t): The optimizer info handle.
        workspace (intptr_t): The workspace descriptor.

    Returns:
        intptr_t: An opaque contraction plan handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateContractionPlan`
    """
    cdef _ContractionPlan plan
    # we always release gil here, because we don't need to allocate
    # memory at this point yet
    with nogil:
        status = cutensornetCreateContractionPlan(
            <_Handle>handle, <_NetworkDescriptor>tn_desc,
            <_ContractionOptimizerInfo>info,
            <_WorkspaceDescriptor>workspace, &plan)
    check_status(status)
    return <intptr_t>plan


cpdef destroy_contraction_plan(intptr_t plan):
    """Destroy a contraction plan.

    Args:
        plan (intptr_t): The contraction plan handle.

    .. seealso:: `cutensornetDestroyContractionPlan`
    """
    with nogil:
        status = cutensornetDestroyContractionPlan(<_ContractionPlan>plan)
    check_status(status)


cpdef contraction_autotune(
        intptr_t handle, intptr_t plan,
        raw_data_in, intptr_t raw_data_out, intptr_t workspace,
        intptr_t pref, intptr_t stream):
    """Autotune the contraction plan to find the best kernels for each pairwise
    tensor contraction.

    The input tensors should form a tensor network that is prescribed by the
    tensor network descriptor that was used to create the contraction plan.

    Args:
        handle (intptr_t): The library handle.
        plan (intptr_t): The contraction plan handle.
        raw_data_in: A host array of pointer addresses (as Python :class:`int`) for
            each input tensor (on device). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        raw_data_out (intptr_t): The pointer address (as Python :class:`int`) to the
            output tensor (on device).
        workspace (intptr_t): The workspace descriptor.
        pref (intptr_t): The autotune preference handle.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetContractionAutotune`
    """
    # raw_data_in can be a pointer address, or a Python sequence
    cdef vector[intptr_t] rawDataInData
    cdef void** rawDataInPtr
    if cpython.PySequence_Check(raw_data_in):
        rawDataInData = raw_data_in
        rawDataInPtr = <void**>(rawDataInData.data())
    else:  # a pointer address
        rawDataInPtr = <void**><intptr_t>raw_data_in

    with nogil:
        status = cutensornetContractionAutotune(
            <_Handle>handle, <_ContractionPlan>plan,
            rawDataInPtr, <void*>raw_data_out,
            <_WorkspaceDescriptor>workspace,
            <_ContractionAutotunePreference>pref,
            <Stream>stream)
    check_status(status)


cpdef intptr_t create_contraction_autotune_preference(intptr_t handle) except*:
    """Create a handle to hold all autotune parameters.

    Args:
        handle (intptr_t): The library handle.

    Returns:
        intptr_t: An opaque autotune preference handle.

    .. seealso:: `cutensornetCreateContractionAutotunePreference`
    """
    cdef _ContractionAutotunePreference pref
    with nogil:
        status = cutensornetCreateContractionAutotunePreference(
            <_Handle>handle, &pref)
    check_status(status)
    return <intptr_t>pref


cpdef destroy_contraction_autotune_preference(intptr_t pref):
    """Destroy the autotue preference handle.

    Args:
        pref (intptr_t): The opaque autotune preference handle.

    .. seealso:: `cutensornetDestroyContractionAutotunePreference`
    """
    with nogil:
        status = cutensornetDestroyContractionAutotunePreference(
            <_ContractionAutotunePreference>pref)
    check_status(status)


######################### Python specific utility #########################

cdef dict contract_autotune_pref_sizes = {
    CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS: _numpy.int32,
    CUTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES: _numpy.int32,
}

cpdef contraction_autotune_preference_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding autotune preference
    attribute.

    Args:
        attr (ContractionAutotunePreferenceAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`contraction_autotune_preference_get_attribute`
        and :func:`contraction_autotune_preference_set_attribute`.
    """
    return contract_autotune_pref_sizes[attr]

###########################################################################


cpdef contraction_autotune_preference_get_attribute(
        intptr_t handle, intptr_t autotune_preference, int attr,
        intptr_t buf, size_t size):
    """Get the autotue preference attributes.

    Args:
        handle (intptr_t): The library handle.
        autotune_preference (intptr_t): The autotune preference handle.
        attr (ContractionAutotunePreferenceAttribute): The attribute to query.
        buf (intptr_t): The pointer address (as Python :class:`int`) for storing
            the returned attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. seealso:: `cutensornetContractionAutotunePreferenceGetAttribute`
    """
    with nogil:
        status = cutensornetContractionAutotunePreferenceGetAttribute(
            <_Handle>handle,
            <_ContractionAutotunePreference>autotune_preference,
            <_ContractionAutotunePreferenceAttribute>attr,
            <void*>buf, size)


cpdef contraction_autotune_preference_set_attribute(
        intptr_t handle, intptr_t autotune_preference, int attr,
        intptr_t buf, size_t size):
    """Set the autotue preference attributes.

    Args:
        handle (intptr_t): The library handle.
        autotune_preference (intptr_t): The autotune preference handle.
        attr (ContractionAutotunePreferenceAttribute): The attribute to query.
        buf (intptr_t): The pointer address (as Python :class:`int`) to the attribute data.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`contraction_autotune_preference_get_attribute_dtype`.

    .. seealso:: `cutensornetContractionAutotunePreferenceSetAttribute`
    """
    with nogil:
        status = cutensornetContractionAutotunePreferenceSetAttribute(
            <_Handle>handle,
            <_ContractionAutotunePreference>autotune_preference,
            <_ContractionAutotunePreferenceAttribute>attr,
            <void*>buf, size)


cpdef contraction(
        intptr_t handle, intptr_t plan,
        raw_data_in, intptr_t raw_data_out, intptr_t workspace,
        int64_t slice_id, intptr_t stream):
    """Perform the contraction of the input tensors.

    The input tensors should form a tensor network that is prescribed by the
    tensor network descriptor that was used to create the contraction plan.

    Args:
        handle (intptr_t): The library handle.
        plan (intptr_t): The contraction plan handle.
        raw_data_in: A host array of pointer addresses (as Python :class:`int`) for
            each input tensor (on device). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        raw_data_out (intptr_t): The pointer address (as Python :class:`int`) to the
            output tensor (on device).
        workspace (intptr_t): The workspace descriptor.
        slice_id (int64_t): The slice ID.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. warning::

       This function is deprecated and will be removed in a future release.
       Use :func:`contract_slices` instead.

    .. note:: The number of slices can be queried by :func:`contraction_optimizer_info_get_attribute`.

    .. seealso:: `cutensornetContraction`
    """
    # raw_data_in can be a pointer address, or a Python sequence
    cdef vector[intptr_t] rawDataInData
    cdef void** rawDataInPtr
    if cpython.PySequence_Check(raw_data_in):
        rawDataInData = raw_data_in
        rawDataInPtr = <void**>(rawDataInData.data())
    else:  # a pointer address
        rawDataInPtr = <void**><intptr_t>raw_data_in
    warnings.warn("cuquantum.cutensornet.contraction() is deprecated and will "
                  "be removed in the future; please switch to "
                  "cuquantum.cutensornet.contract_slices() instead",
                  DeprecationWarning, 2)

    with nogil:
        status = cutensornetContraction(
            <_Handle>handle, <_ContractionPlan>plan,
            rawDataInPtr, <void*>raw_data_out,
            <_WorkspaceDescriptor>workspace,
            slice_id, <Stream>stream)
    check_status(status)


cpdef contract_slices(
        intptr_t handle, intptr_t plan,
        raw_data_in, intptr_t raw_data_out, bint accumulate_output,
        intptr_t workspace, intptr_t slice_group, intptr_t stream):
    """Perform the contraction of the input tensors.

    The input tensors should form a tensor network that is prescribed by the
    tensor network descriptor that was used to create the contraction plan.

    This version of the contraction API takes a group of slices instead of
    a single slice.

    Args:
        handle (intptr_t): The library handle.
        plan (intptr_t): The contraction plan handle.
        raw_data_in: A host array of pointer addresses (as Python :class:`int`) for
            each input tensor (on device). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        raw_data_out (intptr_t): The pointer address (as Python :class:`int`)
            to the output tensor (on device).
        accumulate_output (bool): Whether to accumulate the data in
            ``raw_data_out``.
        workspace (intptr_t): The workspace descriptor.
        slice_group (intptr_t): The slice group descriptor.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetContractSlices`
    """
    # raw_data_in can be a pointer address, or a Python sequence
    cdef vector[intptr_t] rawDataInData
    cdef void** rawDataInPtr
    if cpython.PySequence_Check(raw_data_in):
        rawDataInData = raw_data_in
        rawDataInPtr = <void**>(rawDataInData.data())
    else:  # a pointer address
        rawDataInPtr = <void**><intptr_t>raw_data_in

    with nogil:
        status = cutensornetContractSlices(
            <_Handle>handle, <_ContractionPlan>plan,
            rawDataInPtr, <void*>raw_data_out, <int32_t>accumulate_output,
            <_WorkspaceDescriptor>workspace,
            <_SliceGroup>slice_group,
            <Stream>stream)
    check_status(status)


cpdef intptr_t create_slice_group_from_id_range(
        intptr_t handle, int64_t slice_start, int64_t slice_stop,
        int64_t slice_step) except*:
    """Create a slice group that contains a sequence of slice IDs from
    ``slice_start`` (inclusive) to ``slice_stop`` (exclusive) by
    ``slice_step``.

    This version of the slice group constructor works similarly to the
    Python :class:`range`.

    Args:
        handle (intptr_t): The library handle.
        slice_start (int64_t): The start of the ID sequence (inclusive).
        slice_stop (int64_t): The stop value of the ID sequence (exclusive).
        slice_step (int64_t): The step size of the ID sequence.

    Returns:
        intptr_t: An opaque slice group descriptor.

    .. seealso:: `cutensornetCreateSliceGroupFromIDRange`
    """
    cdef _SliceGroup slice_group
    with nogil:
        status = cutensornetCreateSliceGroupFromIDRange(
            <_Handle>handle, slice_start, slice_stop, slice_step, &slice_group)
    check_status(status)
    return <intptr_t>slice_group


cpdef intptr_t create_slice_group_from_ids(
        intptr_t handle, ids, size_t ids_size) except*:
    """Create a slice group from a sequence of slice IDs.

    Args:
        handle (intptr_t): The library handle.
        ids: A host sequence of slice IDs (as Python :class:`int`). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        ids_size (size_t): the length of the ID sequence.

    Returns:
        intptr_t: An opaque slice group descriptor.

    .. seealso:: `cutensornetCreateSliceGroupFromIDs`
    """
    # ids can be a pointer address, or a Python sequence
    cdef vector[int64_t] IDsData
    cdef int64_t* IdsPtr
    cdef size_t size
    if cpython.PySequence_Check(ids):
        IDsData = ids
        IDsPtr = <int64_t*>(IDsData.data())
        size = IDsData.size()
        assert size == ids_size
    else:  # a pointer address
        IDsPtr = <int64_t*><intptr_t>ids
        size = ids_size

    cdef _SliceGroup slice_group
    with nogil:
        status = cutensornetCreateSliceGroupFromIDs(
            <_Handle>handle, IDsPtr, IDsPtr+size, &slice_group)
    check_status(status)
    return <intptr_t>slice_group


cpdef destroy_slice_group(intptr_t slice_group):
    """Destroy the slice group.

    Args:
        slice_group (intptr_t): An opaque slice group descriptor.

    .. seealso:: `cutensornetDestroySliceGroup`
    """
    with nogil:
        status = cutensornetDestroySliceGroup(<_SliceGroup>slice_group)
    check_status(status)


cpdef set_device_mem_handler(intptr_t handle, handler):
    """ Set the device memory handler for cuTensorNet.

    The ``handler`` object can be passed in multiple ways:

      - If ``handler`` is an :class:`int`, it refers to the address of a fully
        initialized `cutensornetDeviceMemHandler_t` struct.
      - If ``handler`` is a Python sequence:

        - If ``handler`` is a sequence of length 4, it is interpreted as ``(ctx, device_alloc,
          device_free, name)``, where the first three elements are the pointer
          addresses (:class:`int`) of the corresponding members. ``name`` is a
          :class:`str` as the name of the handler.
        - If ``handler`` is a sequence of length 3, it is interpreted as ``(malloc, free,
          name)``, where the first two objects are Python *callables* with the
          following calling convention:

            - ``ptr = malloc(size, stream)``
            - ``free(ptr, size, stream)``

          with all arguments and return value (``ptr``) being Python :class:`int`.
          ``name`` is the same as above.

    .. note:: Only when ``handler`` is a length-3 sequence will the GIL be
        held whenever a routine requires memory allocation and deallocation,
        so for all other cases be sure your ``handler`` does not manipulate
        any Python objects.

    Args:
        handle (intptr_t): The library handle.
        handler: The memory handler object, see above.

    .. seealso:: `cutensornetSetDeviceMemHandler`
    """
    cdef bytes name
    cdef _DeviceMemHandler our_handler
    cdef _DeviceMemHandler* handlerPtr = &our_handler

    if isinstance(handler, int):
        handlerPtr = <_DeviceMemHandler*><intptr_t>handler
    elif cpython.PySequence_Check(handler):
        name = handler[-1].encode('ascii')
        if len(name) > CUTENSORNET_ALLOCATOR_NAME_LEN:
            raise ValueError("the handler name is too long")
        our_handler.name[:len(name)] = name
        our_handler.name[len(name)] = 0

        if len(handler) == 4:
            # handler = (ctx_ptr, malloc_ptr, free_ptr, name)
            assert (isinstance(handler[1], int) and isinstance(handler[2], int))
            our_handler.ctx = <void*><intptr_t>(handler[0])
            our_handler.device_alloc = <DeviceAllocType><intptr_t>(handler[1])
            our_handler.device_free = <DeviceFreeType><intptr_t>(handler[2])
        elif len(handler) == 3:
            # handler = (malloc, free, name)
            assert (callable(handler[0]) and callable(handler[1]))
            ctx = (handler[0], handler[1])
            owner_pyobj[handle] = ctx  # keep it alive
            our_handler.ctx = <void*>ctx
            our_handler.device_alloc = cuqnt_alloc_wrapper
            our_handler.device_free = cuqnt_free_wrapper
        else:
            raise ValueError("handler must be a sequence of length 3 or 4, "
                             "see the documentation for detail")
    else:
        raise NotImplementedError("handler format not recognized")

    with nogil:
        status = cutensornetSetDeviceMemHandler(<_Handle>handle, handlerPtr)
    check_status(status)


cpdef tuple get_device_mem_handler(intptr_t handle):
    """ Get the device memory handler for cuTensorNet.

    Args:
        handle (intptr_t): The library handle.

    Returns:
        tuple:
            The ``handler`` object, which has two forms:

              - If ``handler`` is a 3-tuple, it is interpreted as ``(malloc, free,
                name)``, where the first two objects are Python *callables*, and ``name``
                is the name of the handler. This 3-tuple handler would be compared equal
                (elementwisely) to the one previously passed to :func:`set_device_mem_handler`.
              - If ``handler`` is a 4-tuple, it is interpreted as ``(ctx, device_alloc,
                device_free, name)``, where the first three elements are the pointer
                addresses (:class:`int`) of the corresponding members. ``name`` is the
                same as above.

    .. seealso:: `cutensornetGetDeviceMemHandler`
    """
    cdef _DeviceMemHandler handler
    with nogil:
        status = cutensornetGetDeviceMemHandler(<_Handle>handle, &handler)
    check_status(status)

    cdef tuple ctx
    cdef bytes name = handler.name
    if (handler.device_alloc == cuqnt_alloc_wrapper and
            handler.device_free == cuqnt_free_wrapper):
        ctx = <object>(handler.ctx)
        return (ctx[0], ctx[1], name.decode('ascii'))
    else:
        # TODO: consider other possibilities?
        return (<intptr_t>handler.ctx,
                <intptr_t>handler.device_alloc,
                <intptr_t>handler.device_free,
                name.decode('ascii'))


# can't be cpdef because args & kwargs can't be handled in a C signature
def logger_set_callback_data(callback, *args, **kwargs):
    """Set the logger callback along with arguments.

    Args:
        callback: A Python callable with the following signature (no return):

          - ``callback(log_level, func_name, message, *args, **kwargs)``

          where ``log_level`` (:py:`int`), ``func_name`` (`str`), and
          ``message`` (`str`) are provided by the logger API.

    .. seealso:: `cutensornetLoggerSetCallbackData`
    """
    func_arg = (callback, args, kwargs)
    # if only set once, the callback lifetime should be as long as this module,
    # because we don't know when the logger is done using it
    global logger_callback_holder
    logger_callback_holder = func_arg
    with nogil:
        status = cutensornetLoggerSetCallbackData(
            <LoggerCallbackData>logger_callback_with_data, <void*>(func_arg))
    check_status(status)


cpdef logger_open_file(filename):
    """Set the filename for the logger to write to.

    Args:
        filename (str): The log filename.

    .. seealso:: `cutensornetLoggerOpenFile`
    """
    cdef bytes name = filename.encode()
    cdef char* name_ptr = name
    with nogil:
        status = cutensornetLoggerOpenFile(name_ptr)
    check_status(status)


cpdef logger_set_level(int level):
    """Set the logging level.

    Args:
        level (int): The logging level.

    .. seealso:: `cutensornetLoggerSetLevel`
    """
    with nogil:
        status = cutensornetLoggerSetLevel(level)
    check_status(status)


cpdef logger_set_mask(int mask):
    """Set the logging mask.

    Args:
        level (int): The logging mask.

    .. seealso:: `cutensornetLoggerSetMask`
    """
    with nogil:
        status = cutensornetLoggerSetMask(mask)
    check_status(status)


cpdef logger_force_disable():
    """Disable the logger.

    .. seealso:: `cutensornetLoggerForceDisable`
    """
    with nogil:
        status = cutensornetLoggerForceDisable()
    check_status(status)


cpdef intptr_t create_tensor_descriptor(
        intptr_t handle, int32_t n_modes, extents, strides, modes,
        int data_type) except*:
    """Create a tensor descriptor.

    Args:
        handle (intptr_t): The library handle.
        n_modes (int32_t): The number of modes of the tensor.
        extents: The extents of the tensor (on host). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        strides: The strides of the tensor (on host). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        modes: The modes of the tensor (on host). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        data_type (cuquantum.cudaDataType): The data type of the tensor.

    Returns:
        intptr_t: An opaque descriptor handle (as Python :class:`int`).

    .. note::
        If ``strides`` is set to 0 (``NULL``), it means the tensor is in
        the Fortran layout (F-contiguous).

    .. seealso:: `cutensornetCreateTensorDescriptor`
    """
    # extents can be a pointer address, or a Python sequence
    cdef vector[int64_t] extentsData
    cdef int64_t* extentsPtr
    if cpython.PySequence_Check(extents):
        extentsData = extents
        extentsPtr = extentsData.data()
    else:  # a pointer address
        extentsPtr = <int64_t*><intptr_t>extents

    # strides can be a pointer address, or a Python sequence
    cdef vector[int64_t] stridesData
    cdef int64_t* stridesPtr
    if cpython.PySequence_Check(strides):
        stridesData = strides
        stridesPtr = stridesData.data()
    else:  # a pointer address
        stridesPtr = <int64_t*><intptr_t>strides

    # modes can be a pointer address, or a Python sequence
    cdef vector[int32_t] modesData
    cdef int32_t* modesPtr
    if cpython.PySequence_Check(modes):
        modesData = modes
        modesPtr = modesData.data()
    else:  # a pointer address
        modesPtr = <int32_t*><intptr_t>modes

    cdef _TensorDescriptor desc
    with nogil:
        status = cutensornetCreateTensorDescriptor(
            <_Handle>handle, n_modes, extentsPtr, stridesPtr, modesPtr,
            <DataType>data_type, &desc)
    check_status(status)
    return <intptr_t>desc


cpdef destroy_tensor_descriptor(intptr_t desc):
    """Destroy a tensor descriptor.

    Args:
        desc (intptr_t): The tensor descriptor.

    .. seealso:: `cutensornetDestroyTensorDescriptor`
    """
    with nogil:
        status = cutensornetDestroyTensorDescriptor(<_TensorDescriptor>desc)
    check_status(status)


cpdef intptr_t create_tensor_svd_config(
        intptr_t handle) except*:
    """Create a tensor SVD config object.

    Args:
        handle (intptr_t): The library handle.

    Returns:
        intptr_t: An opaque tensor SVD config handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateTensorSVDConfig`
    """
    cdef _TensorSVDConfig config
    with nogil:
        status = cutensornetCreateTensorSVDConfig(
            <_Handle>handle, &config)
    check_status(status)
    return <intptr_t>config


cpdef destroy_tensor_svd_config(intptr_t config):
    """Destroy a tensor SVD config object.

    Args:
        config (intptr_t): The tensor SVD config handle.

    .. seealso:: `cutensornetDestroyTensorSVDConfig`
    """
    with nogil:
        status = cutensornetDestroyTensorSVDConfig(
            <_TensorSVDConfig>config)
    check_status(status)


######################### Python specific utility #########################

cdef dict tensor_svd_cfg_sizes = {
    CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF: _numpy.float64,
    CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF: _numpy.float64,
    CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION: _numpy.int32,  # = sizeof(enum value)
    CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION: _numpy.int32,  # = sizeof(enum value)
    CUTENSORNET_TENSOR_SVD_CONFIG_ALGO: _numpy.int32, # = sizeof(enum value)
    CUTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF: _numpy.float64,
}

cdef dict svd_algo_params_sizes = {
    CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ: gesvdj_params_dtype,
    CUTENSORNET_TENSOR_SVD_ALGO_GESVDR: gesvdr_params_dtype
}

cpdef tensor_svd_config_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding tensor SVD config attribute.

    Args:
        attr (TensorSVDConfigAttribute): The attribute to query. 
            The enum CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS is not supported, 
            the dtype of which can be queried by :func:`tensor_svd_algo_params_get_dtype`.


    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`tensor_svd_config_get_attribute`
        and :func:`tensor_svd_config_set_attribute`.
    """
    if attr == CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS:
        raise ValueError("For CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS, use `tensor_svd_algo_params_get_dtype` to get the dtype")
    dtype = tensor_svd_cfg_sizes[attr]
    if attr == CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION:
        if _numpy.dtype(dtype).itemsize != sizeof(_TensorSVDNormalization):
            warnings.warn("binary size may be incompatible")
    elif attr == CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION:
        if _numpy.dtype(dtype).itemsize != sizeof(_TensorSVDPartition):
            warnings.warn("binary size may be incompatible")
    elif attr == CUTENSORNET_TENSOR_SVD_CONFIG_ALGO:
        if _numpy.dtype(dtype).itemsize != sizeof(_TensorSVDAlgo):
            warnings.warn("binary size may be incompatible")
    return dtype

cpdef tensor_svd_algo_params_get_dtype(int svd_algo):
    """Get the Python data type of the corresponding tensor SVD parameters attribute.
    
    Args:
        svd_algo (TensorSVDAlgo): The SVD algorithm to query.
    
    Returns:
        The data type of algorithm parameters for the queried SVD algorithm. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for `CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS`.
    """
    if svd_algo not in svd_algo_params_sizes:
        raise ValueError(f"Algorithm {svd_algo} does not support tunable parameters.")
    return svd_algo_params_sizes[svd_algo]

###########################################################################


cpdef tensor_svd_config_get_attribute(
        intptr_t handle, intptr_t config, int attr,
        intptr_t buf, size_t size):
    """Get the tensor SVD config attribute.

    Args:
        handle (intptr_t): The library handle.
        config (intptr_t): The tensor SVD config handle.
        attr (TensorSVDConfigAttribute): The attribute to set.
        buf (intptr_t): The pointer address (as Python :class:`int`) for storing
            the returned attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`tensor_svd_config_get_attribute_dtype`.

    .. seealso:: `cutensornetTensorSVDConfigGetAttribute`
    """
    with nogil:
        status = cutensornetTensorSVDConfigGetAttribute(
            <_Handle>handle, <_TensorSVDConfig>config,
            <_TensorSVDConfigAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef tensor_svd_config_set_attribute(
        intptr_t handle, intptr_t config, int attr,
        intptr_t buf, size_t size):
    """Set the tensor SVD config attribute.

    Args:
        handle (intptr_t): The library handle.
        config (intptr_t): The tensor SVD config handle.
        attr (TensorSVDConfigAttribute): The attribute to set.
        buf (intptr_t): The pointer address (as Python :class:`int`) to the attribute data.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`tensor_svd_config_get_attribute_dtype`.

    .. seealso:: `cutensornetTensorSVDConfigSetAttribute`
    """
    with nogil:
        status = cutensornetTensorSVDConfigSetAttribute(
            <_Handle>handle, <_TensorSVDConfig>config,
            <_TensorSVDConfigAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef intptr_t create_tensor_svd_info(intptr_t handle) except*:
    """Create a tensor SVD info object.

    Args:
        handle (intptr_t): The library handle.

    Returns:
        intptr_t: An opaque tensor SVD info handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateTensorSVDInfo`
    """
    cdef _TensorSVDInfo info
    with nogil:
        status = cutensornetCreateTensorSVDInfo(
            <_Handle>handle, &info)
    check_status(status)
    return <intptr_t>info


cpdef destroy_tensor_svd_info(intptr_t info):
    """Destroy a tensor SVD info object.

    Args:
        info (intptr_t): The tensor SVD info handle.

    .. seealso:: `cutensornetDestroyTensorSVDInfo`
    """
    with nogil:
        status = cutensornetDestroyTensorSVDInfo(
            <_TensorSVDInfo>info)
    check_status(status)


######################### Python specific utility #########################

cdef dict tensor_svd_info_sizes = {
    CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT: _numpy.int64,
    CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT: _numpy.int64,
    CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT: _numpy.float64,
    CUTENSORNET_TENSOR_SVD_INFO_ALGO: _numpy.int32, # = sizeof(enum value)
}

cdef dict svd_algo_status_sizes = {
    CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ: gesvdj_status_dtype,
    CUTENSORNET_TENSOR_SVD_ALGO_GESVDP: gesvdp_status_dtype
}

cpdef tensor_svd_info_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding tensor SVD info attribute.

    Args:
        attr (TensorSVDInfoAttribute): The attribute to query.
            The enum CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS is not supported, 
            the dtype of which can be queried by :func:`tensor_svd_algo_status_get_dtype`.

    Returns:
        The data type of the queried attribute. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`tensor_svd_info_get_attribute`.

    """
    if attr == CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS:
        raise ValueError("For CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS, use `tensor_svd_algo_status_get_dtype` to get the dtype")
    return tensor_svd_info_sizes[attr]

cpdef tensor_svd_algo_status_get_dtype(int svd_algo):
    """Get the Python data type of the corresponding tensor SVD status attribute.
    
    Args:
        svd_algo (TensorSVDAlgo): The SVD algorithm to query.
    
    Returns:
        The data type of algorithm status for the queried SVD algorithm. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for `CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS`.
    """
    if svd_algo not in svd_algo_status_sizes:
        raise ValueError(f"Algorithm {svd_algo} does not support tunable parameters.")
    return svd_algo_status_sizes[svd_algo]

###########################################################################


cpdef tensor_svd_info_get_attribute(
        intptr_t handle, intptr_t info, int attr,
        intptr_t buf, size_t size):
    """Get the tensor SVD info attribute.

    Args:
        handle (intptr_t): The library handle.
        info (intptr_t): The tensor SVD info handle.
        attr (TensorSVDInfoAttribute): The attribute to query.
        buf (intptr_t): The pointer address (as Python :class:`int`) for storing
            the returned attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`tensor_svd_info_get_attribute_dtype`.

    .. seealso:: `cutensornetTensorSVDInfoGetAttribute`
    """
    with nogil:
        status = cutensornetTensorSVDInfoGetAttribute(
            <_Handle>handle, <_TensorSVDInfo>info,
            <_TensorSVDInfoAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef workspace_compute_svd_sizes(
        intptr_t handle, intptr_t tensor_in, intptr_t tensor_u,
        intptr_t tensor_v, intptr_t config, intptr_t workspace):
    """Compute the required workspace sizes for :func:`tensor_svd`.

    Args:
        handle (intptr_t): The library handle.
        tensor_in (intptr_t): The input tensor descriptor.
        tensor_u (intptr_t): The tensor descriptor for the output U.
        tensor_v (intptr_t): The tensor descriptor for the output V.
        config (intptr_t): The tensor SVD config handle.
        workspace (intptr_t): The workspace descriptor.

    .. seealso:: `cutensornetWorkspaceComputeSVDSizes`
    """
    with nogil:
        status = cutensornetWorkspaceComputeSVDSizes(
            <_Handle>handle, <_TensorDescriptor>tensor_in,
            <_TensorDescriptor>tensor_u, <_TensorDescriptor>tensor_v,
            <_TensorSVDConfig>config,
            <_WorkspaceDescriptor>workspace)
    check_status(status)


cpdef tensor_svd(
        intptr_t handle, intptr_t tensor_in, intptr_t raw_data_in,
        intptr_t tensor_u, intptr_t u,
        intptr_t s,
        intptr_t tensor_v, intptr_t v,
        intptr_t config, intptr_t info,
        intptr_t workspace, intptr_t stream):
    """Perform SVD decomposition of a tensor.

    Args:
        handle (intptr_t): The library handle.
        tensor_in (intptr_t): The input tensor descriptor.
        raw_data_in (intptr_t): The pointer address (as Python :class:`int`) to the
            input tensor (on device).
        tensor_u (intptr_t): The tensor descriptor for the output U.
        u (intptr_t): The pointer address (as Python :class:`int`) to the output
            tensor U (on device).
        s (intptr_t): The pointer address (as Python :class:`int`) to the output
            array S (on device).
        tensor_v (intptr_t): The tensor descriptor for the output V.
        v (intptr_t): The pointer address (as Python :class:`int`) to the output
            tensor V (on device).
        config (intptr_t): The tensor SVD config handle.
        info (intptr_t): The tensor SVD info handle.
        workspace (intptr_t): The workspace descriptor.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. note::

        After this function call, the output tensor descriptors ``tensor_u`` and
        ``tensor_v`` may have their shapes and strides changed. See the documentation
        for further information.

    .. seealso:: `cutensornetTensorSVD`
    """
    with nogil:
        status = cutensornetTensorSVD(
            <_Handle>handle, <_TensorDescriptor>tensor_in, <void*>raw_data_in,
            <_TensorDescriptor>tensor_u, <void*>u,
            <void*>s,
            <_TensorDescriptor>tensor_v, <void*>v,
            <_TensorSVDConfig>config, <_TensorSVDInfo>info,
            <_WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef workspace_compute_qr_sizes(
        intptr_t handle, intptr_t tensor_in, intptr_t tensor_q,
        intptr_t tensor_r, intptr_t workspace):
    """Compute the required workspace sizes for :func:`tensor_qr`.

    Args:
        handle (intptr_t): The library handle.
        tensor_in (intptr_t): The input tensor descriptor.
        tensor_q (intptr_t): The tensor descriptor for the output Q.
        tensor_r (intptr_t): The tensor descriptor for the output R.
        workspace (intptr_t): The workspace descriptor.

    .. seealso:: `cutensornetWorkspaceComputeQRSizes`
    """
    with nogil:
        status = cutensornetWorkspaceComputeQRSizes(
            <_Handle>handle, <_TensorDescriptor>tensor_in,
            <_TensorDescriptor>tensor_q, <_TensorDescriptor>tensor_r,
            <_WorkspaceDescriptor>workspace)
    check_status(status)


cpdef tensor_qr(
        intptr_t handle, intptr_t tensor_in, intptr_t raw_data_in,
        intptr_t tensor_q, intptr_t q,
        intptr_t tensor_r, intptr_t r,
        intptr_t workspace, intptr_t stream):
    """Perform QR decomposition of a tensor.

    Args:
        handle (intptr_t): The library handle.
        tensor_in (intptr_t): The input tensor descriptor.
        raw_data_in (intptr_t): The pointer address (as Python :class:`int`) to the
            input tensor (on device).
        tensor_q (intptr_t): The tensor descriptor for the output Q.
        q (intptr_t): The pointer address (as Python :class:`int`) to the output
            tensor Q (on device).
        tensor_r (intptr_t): The tensor descriptor for the output R.
        r (intptr_t): The pointer address (as Python :class:`int`) to the output
            tensor R (on device).
        workspace (intptr_t): The workspace descriptor.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetTensorQR`
    """
    with nogil:
        status = cutensornetTensorQR(
            <_Handle>handle, <_TensorDescriptor>tensor_in, <void*>raw_data_in,
            <_TensorDescriptor>tensor_q, <void*>q,
            <_TensorDescriptor>tensor_r, <void*>r,
            <_WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef workspace_compute_gate_split_sizes(
        intptr_t handle, intptr_t tensor_a, intptr_t tensor_b,
        intptr_t tensor_g, intptr_t tensor_u, intptr_t tensor_v,
        int algo, intptr_t svd_config, int compute_type,
        intptr_t workspace):
    """Compute the required workspace sizes for :func:`gate_split`.

    Args:
        handle (intptr_t): The library handle.
        tensor_a (intptr_t): The tensor descriptor for the input A.
        tensor_b (intptr_t): The tensor descriptor for the input B.
        tensor_g (intptr_t): The tensor descriptor for the input G (the gate).
        tensor_u (intptr_t): The tensor descriptor for the output U.
        tensor_v (intptr_t): The tensor descriptor for the output V.
        algo (cuquantum.cutensornet.GateSplitAlgo): The gate splitting algorithm.
        svd_config (intptr_t): The tensor SVD config handle.
        compute_type (cuquantum.ComputeType): The compute type of the
            computation.
        workspace (intptr_t): The workspace descriptor.

    .. seealso:: `cutensornetWorkspaceComputeGateSplitSizes`
    """
    with nogil:
        status = cutensornetWorkspaceComputeGateSplitSizes(
            <_Handle>handle, <_TensorDescriptor>tensor_a,
            <_TensorDescriptor>tensor_b, <_TensorDescriptor>tensor_g,
            <_TensorDescriptor>tensor_u, <_TensorDescriptor>tensor_v,
            <_GateSplitAlgo>algo, <_TensorSVDConfig>svd_config,
            <_ComputeType>compute_type, <_WorkspaceDescriptor>workspace)
    check_status(status)


cpdef gate_split(
        intptr_t handle, intptr_t tensor_a, intptr_t raw_data_a,
        intptr_t tensor_b, intptr_t raw_data_b,
        intptr_t tensor_g, intptr_t raw_data_g,
        intptr_t tensor_u, intptr_t u,
        intptr_t s,
        intptr_t tensor_v, intptr_t v,
        int algo, intptr_t svd_config, int compute_type,
        intptr_t svd_info, intptr_t workspace, intptr_t stream):
    """Perform gate split operation.

    Args:
        handle (intptr_t): The library handle.
        tensor_a (intptr_t): The tensor descriptor for the input A.
        raw_data_a (intptr_t): The pointer address (as Python :class:`int`) to the
            input tensor A (on device).
        tensor_b (intptr_t): The tensor descriptor for the input B.
        raw_data_b (intptr_t): The pointer address (as Python :class:`int`) to the
            input tensor B (on device).
        tensor_g (intptr_t): The tensor descriptor for the input G (the gate).
        raw_data_g (intptr_t): The pointer address (as Python :class:`int`) to the
            gate tensor G (on device).
        tensor_u (intptr_t): The tensor descriptor for the output U.
        u (intptr_t): The pointer address (as Python :class:`int`) to the output
            tensor U (on device).
        s (intptr_t): The pointer address (as Python :class:`int`) to the output
            array S (on device).
        tensor_v (intptr_t): The tensor descriptor for the output V.
        v (intptr_t): The pointer address (as Python :class:`int`) to the output
            tensor V (on device).
        algo (cuquantum.cutensornet.GateSplitAlgo): The gate splitting algorithm.
        svd_config (intptr_t): The tensor SVD config handle.
        compute_type (cuquantum.ComputeType): The compute type of the
            computation.
        svd_info (intptr_t): The tensor SVD info handle.
        workspace (intptr_t): The workspace descriptor.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. note::

        After this function call, the output tensor descriptors ``tensor_u`` and
        ``tensor_v`` may have their shapes and strides changed. See the documentation
        for further information.

    .. seealso:: `cutensornetGateSplit`
    """
    with nogil:
        status = cutensornetGateSplit(
            <_Handle>handle,
            <_TensorDescriptor>tensor_a, <void*>raw_data_a,
            <_TensorDescriptor>tensor_b, <void*>raw_data_b,
            <_TensorDescriptor>tensor_g, <void*>raw_data_g,
            <_TensorDescriptor>tensor_u, <void*>u,
            <void*>s,
            <_TensorDescriptor>tensor_v, <void*>v,
            <_GateSplitAlgo>algo, <_TensorSVDConfig>svd_config,
            <_ComputeType>compute_type, <_TensorSVDInfo>svd_info,
            <_WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef distributed_reset_configuration(
        intptr_t handle, intptr_t comm_ptr, size_t comm_size):
    """Reset the distributed communicator.

    Args:
        handle (intptr_t): The library handle.
        comm_ptr (intptr_t): The pointer to the provided communicator.
        comm_size (size_t): The size of the provided communicator
            (``sizeof(comm)``).

    .. note:: For using MPI communicators from mpi4py, the helper function
        :func:`~cuquantum.cutensornet.get_mpi_comm_pointer` can be used:

        .. code-block:: python

            cutn.distributed_reset_configuration(handle, *get_mpi_comm_pointer(comm))

    .. seealso:: `cutensornetDistributedResetConfiguration`
    """
    with nogil:
        status = cutensornetDistributedResetConfiguration(
            <_Handle>handle, <void*>comm_ptr, comm_size)
    check_status(status)


cpdef int distributed_get_num_ranks(intptr_t handle) except -1:
    """Get the number of distributed ranks.

    Args:
        handle (intptr_t): The library handle.

    .. seealso:: `cutensornetDistributedGetNumRanks`
    """
    cdef int rank
    with nogil:
        status = cutensornetDistributedGetNumRanks(
            <_Handle>handle, &rank)
    check_status(status)
    return rank


cpdef int distributed_get_proc_rank(intptr_t handle) except -1:
    """Get the current process rank.

    Args:
        handle (intptr_t): The library handle.

    .. seealso:: `cutensornetDistributedGetProcRank`
    """
    cdef int rank
    with nogil:
        status = cutensornetDistributedGetProcRank(
            <_Handle>handle, &rank)
    check_status(status)
    return rank


cpdef distributed_synchronize(intptr_t handle):
    """Synchronize the distributed communicator.

    Args:
        handle (intptr_t): The library handle.

    .. seealso:: `cutensornetDistributedSynchronize`
    """
    with nogil:
        status = cutensornetDistributedSynchronize(<_Handle>handle)
    check_status(status)


cpdef compute_gradients_backward(
        intptr_t handle, intptr_t plan,
        raw_data_in, intptr_t output_gradient, gradients, bint accumulate_output,
        intptr_t workspace, intptr_t stream):
    """Compute the gradients of the network w.r.t. the input tensors whose
    gradients are required.

    The input tensors should form a tensor network that is prescribed by the
    tensor network descriptor that was used to create the contraction plan.

    .. warning::

        This function is experimental and is subject to change in future
        releases.

    Args:
        handle (intptr_t): The library handle.
        plan (intptr_t): The contraction plan handle.
        raw_data_in: A host array of pointer addresses (as Python :class:`int`) for
            each input tensor (on device). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        output_gradient (intptr_t): The pointer address (as Python :class:`int`)
            to the gradient w.r.t. the output tensor (on device).
        gradients: A host array of pointer addresses (as Python :class:`int`) for
            each gradient tensor (on device). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        accumulate_output (bool): Whether to accumulate the data in
            ``gradients``.
        workspace (intptr_t): The workspace descriptor.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetComputeGradientsBackward`
    """
    warnings.warn("compute_gradients_backward() is an experimental API and "
                  "subject to future changes", stacklevel=2)

    # raw_data_in can be a pointer address, or a Python sequence
    cdef vector[intptr_t] rawDataInData
    cdef void** rawDataInPtr
    if cpython.PySequence_Check(raw_data_in):
        rawDataInData = raw_data_in
        rawDataInPtr = <void**>(rawDataInData.data())
    else:  # a pointer address
        rawDataInPtr = <void**><intptr_t>raw_data_in

    # gradients can be a pointer address, or a Python sequence
    cdef vector[intptr_t] gradientsData
    cdef void** gradientsPtr
    if cpython.PySequence_Check(gradients):
        gradientsData = gradients
        gradientsPtr = <void**>(gradientsData.data())
    else:  # a pointer address
        gradientsPtr = <void**><intptr_t>gradients

    with nogil:
        status = cutensornetComputeGradientsBackward(
            <_Handle>handle, <_ContractionPlan>plan,
            rawDataInPtr, <void*>output_gradient, gradientsPtr,
            <int32_t>accumulate_output,
            <_WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef intptr_t create_state(
        intptr_t handle, 
        int purity, int32_t n_state_modes, 
        state_mode_extents, int data_type) except*:
    """Create a tensor network state.

    Args:
        handle (intptr_t): The library handle.
        purity (cuquantum.cutensornet.StatePurity): The tensor network state purity.
        n_state_modes (int32_t): The number of modes of the tensor network states. 
        state_mode_extents: A host array of extents for each state mode. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`
        
        data_type (cuquantum.cudaDataType): The data type of the tensor network state.

    Returns:
        intptr_t: An opaque tensor network state handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateState`
    """
    # state_mode_extents can be a pointer address, or a Python sequence
    cdef vector[int64_t] stateModesExtentsData
    cdef int64_t* stateModesExtentsPtr
    if cpython.PySequence_Check(state_mode_extents):
        if len(state_mode_extents) != n_state_modes:
            raise ValueError("size of state_mode_extents not matching n_state_modes")
        stateModesExtentsData = state_mode_extents
        stateModesExtentsPtr = stateModesExtentsData.data()
    else:  # a pointer address
        stateModesExtentsPtr = <int64_t*><intptr_t>state_mode_extents
    
    cdef _State state
    with nogil:
        status = cutensornetCreateState(
            <_Handle>handle, <_StatePurity>purity, n_state_modes, 
            stateModesExtentsPtr, <DataType>data_type, &state)
    check_status(status)
    return <intptr_t>state


cpdef destroy_state(intptr_t state):
    """Destroy a tensor network state.

    Args:
        state (intptr_t): The tensor network state.

    .. seealso:: `cutensornetDestroyState`
    """
    with nogil:
        status = cutensornetDestroyState(<_State>state)
    check_status(status)


cpdef int64_t state_apply_tensor(
        intptr_t handle, intptr_t state, int32_t n_state_modes, 
        state_modes, intptr_t tensor_data, tensor_mode_strides, 
        int32_t immutable, int32_t adjoint, int32_t unitary):
    """Apply a tensor operator to the tensor network state.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.
        n_state_modes (int32_t): The number of state modes that the tensor applies on.
        state_modes: A host array of modes to specify where the tensor is applied to. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`
         
        tensor_data (intptr_t): The tensor data.
        tensor_mode_strides: A host array of strides for each mode. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        immutable (int32_t): Whether the tensor is immutable
        adjoint (int32_t): Whether the tensor should be considered as adjoint.
        unitary (int32_t): Whether the tensor represents a unitary operation.
    
    Returns:
        tensor_id (int64_t): The id that is assigned to the tensor.

    .. seealso:: `cutensornetStateApplyTensor`
    """
    # state_modes can be a pointer address, or a Python sequence
    cdef int64_t tensor_id
    cdef vector[int32_t] stateModesData
    cdef int32_t* stateModesPtr
    if cpython.PySequence_Check(state_modes):
        if len(state_modes) != n_state_modes:
            raise ValueError("size of state_modes not matching n_state_modes")
        stateModesData = state_modes
        stateModesPtr = stateModesData.data()
    else:  # a pointer address
        stateModesPtr = <int32_t*><intptr_t>state_modes
    
    # tensor_mode_strides can be a pointer address, or a Python sequence
    cdef vector[int64_t] tensorModesStridesData
    cdef int64_t* tensorModesStridesPtr
    if cpython.PySequence_Check(tensor_mode_strides):
        tensorModesStridesData = tensor_mode_strides
        tensorModesStridesPtr = tensorModesStridesData.data()
    else:  # a pointer address
        tensorModesStridesPtr = <int64_t*><intptr_t>tensor_mode_strides
    
    with nogil:
        status = cutensornetStateApplyTensor(
            <_Handle>handle, <_State>state, n_state_modes, stateModesPtr, <void*>tensor_data, 
            tensorModesStridesPtr, immutable, adjoint, unitary, &tensor_id)
    check_status(status)
    return tensor_id


cpdef state_update_tensor(
        intptr_t handle, intptr_t state, 
        int64_t tensor_id, intptr_t tensor_data, int32_t unitary):
    """Update a tensor operand that has been applied to the tensor network state.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.
        tensor_id (int64_t): The id that is assigned to the tensor.        
        tensor_data (intptr_t): The tensor data.
        adjoint (int32_t): Whether the tensor should be considered as adjoint.
        unitary (int32_t): Whether the tensor represents a unitary operation.

    .. seealso:: `cutensornetStateUpdateTensor`
    """
    with nogil:
        status = cutensornetStateUpdateTensor(
            <_Handle>handle, <_State>state, tensor_id, <void*>tensor_data, unitary)
    check_status(status)


cpdef intptr_t create_marginal(
        intptr_t handle, intptr_t state, 
        int32_t n_marginal_modes, marginal_modes, 
        int32_t n_projected_modes, projected_modes, marginal_tensor_strides) except*:
    """Create a representation for the tensor network state marginal distribution.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.
        n_marginal_modes (int32_t): The number of modes for the marginal.
        marginal_modes: A host array of modes for the marginal. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`
        
        n_projected_modes (int32_t): The number of modes that are projected out for the marginal.
        projected_modes: A host array of projected modes for the marginal. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`
        
        marginal_tensor_strides: A host array of strides for the marginal modes. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

    Returns:
        intptr_t: An opaque tensor network state marginal handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateMarginal`
    """
    # marginal_modes can be a pointer address, or a Python sequence
    cdef vector[int32_t] marginalModesData
    cdef int32_t* marginalModesPtr
    if cpython.PySequence_Check(marginal_modes):
        if len(marginal_modes) != n_marginal_modes:
            raise ValueError("size of marginal_modes not matching n_marginal_modes")
        marginalModesData = marginal_modes
        marginalModesPtr = marginalModesData.data()
    else:  # a pointer address
        marginalModesPtr = <int32_t*><intptr_t>marginal_modes
    
    # projected_modes can be a pointer address, or a Python sequence
    cdef vector[int32_t] projectedModesData
    cdef int32_t* projectedModesPtr
    if cpython.PySequence_Check(projected_modes):
        if len(projected_modes) != n_projected_modes:
            raise ValueError("size of projected_modes not matching n_projected_modes")
        projectedModesData = projected_modes
        projectedModesPtr = projectedModesData.data()
    else:  # a pointer address
        projectedModesPtr = <int32_t*><intptr_t>projected_modes
    
    # marginal_tensor_strides can be a pointer address, or a Python sequence
    cdef vector[int64_t] marginalTensorStridesData
    cdef int64_t* marginalTensorStridesPtr
    if cpython.PySequence_Check(marginal_tensor_strides):
        marginalTensorStridesData = marginal_tensor_strides
        marginalTensorStridesPtr = marginalTensorStridesData.data()
    else:  # a pointer address
        marginalTensorStridesPtr = <int64_t*><intptr_t>marginal_tensor_strides
    
    cdef _StateMarginal marginal
    with nogil:
        status = cutensornetCreateMarginal(
            <_Handle>handle, <_State>state, 
            n_marginal_modes, marginalModesPtr,
            n_projected_modes, projectedModesPtr, 
            marginalTensorStridesPtr, &marginal)
    check_status(status)
    return <intptr_t>marginal


cdef dict marginal_attribute_sizes = {
    CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES: _numpy.int32
}


cpdef marginal_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding marginal attribute.

    Args:
        attr (MarginalAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`marginal_configure`.
    """
    return marginal_attribute_sizes[attr]
    

cpdef marginal_configure(intptr_t handle, intptr_t marginal, int attr, intptr_t buf, size_t size):
    """Configures computation of the tensor network state marginal distribution.

    Args:
        handle (intptr_t): The library handle.
        marginal (intptr_t): The tensor network marginal computation handle.
        attr (MarginalAttribute): The attribute to configure.
        buf (intptr_t): The pointer address (as Python :class:`int`) of the attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`marginal_get_attribute_dtype`.

    .. seealso:: `cutensornetMarginalConfigure`
    """
    with nogil:
        status = cutensornetMarginalConfigure(
            <_Handle>handle, <_StateMarginal>marginal,
            <_MarginalAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef marginal_prepare(
        intptr_t handle, intptr_t marginal, 
        size_t max_workspace_size_device, intptr_t workspace, intptr_t stream):
    """Prepares computation of the tensor network state marginal distribution.

    Args:
        handle (intptr_t): The library handle.
        marginal (intptr_t): The tensor network marginal computation handle.
        max_workspace_size_device (size_t): The maximal device workspace size (in bytes) allowed 
            for the mariginal computation.
        workspace (intptr_t): The workspace descriptor.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetMarginalPrepare`
    """
    with nogil:
        status = cutensornetMarginalPrepare(
            <_Handle>handle, <_StateMarginal>marginal,
            max_workspace_size_device, <_WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef marginal_compute(
        intptr_t handle, intptr_t marginal, projected_mode_values, 
        intptr_t workspace, intptr_t marginal_tensor, intptr_t stream):
    """Computes the tensor network state marginal distribution.

    Args:
        handle (intptr_t): The library handle.
        marginal (intptr_t): The tensor network marginal computation handle.
        projected_mode_values: A host array of values for the projected modes. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`
        
        workspace (intptr_t): The workspace descriptor.
        marginal_tensor (intptr_t): The pointer address (as Python :class:`int`) for storing
            the computed marginals.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetMarginalCompute`
    """
    # projected_mode_values can be a pointer address, or a Python sequence
    cdef vector[int64_t] projectedModeValuesData
    cdef int64_t* projectedModeValuesPtr
    if cpython.PySequence_Check(projected_mode_values):
        projectedModeValuesData = projected_mode_values
        projectedModeValuesPtr = projectedModeValuesData.data()
    else:  # a pointer address
        projectedModeValuesPtr = <int64_t*><intptr_t>projected_mode_values
    
    with nogil:
        status = cutensornetMarginalCompute(
            <_Handle>handle, <_StateMarginal>marginal,
            projectedModeValuesPtr, <_WorkspaceDescriptor>workspace, 
            <void*>marginal_tensor, <Stream>stream)
    check_status(status)


cpdef destroy_marginal(intptr_t marginal):
    """Destroy a tensor network marginal representation.

    Args:
        marginal (intptr_t): The tensor network marginal distribution.

    .. seealso:: `cutensornetDestroyMarginal`
    """
    with nogil:
        status = cutensornetDestroyMarginal(<_StateMarginal>marginal)
    check_status(status)


cpdef intptr_t create_sampler(
        intptr_t handle, intptr_t state, 
        int32_t n_modes_to_sample, modes_to_sample) except*:
    """Creates a tensor network state sampler.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.
        n_modes_to_sample (int32_t): The number of modes to sample for the sampler.
        modes_to_sample: A host array of modes for the sampler. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`
        
    Returns:
        intptr_t: An opaque tensor network state sampler handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateSampler`
    """
    # modes_to_sample can be a pointer address, or a Python sequence
    cdef vector[int32_t] modesData
    cdef int32_t* modesPtr
    if cpython.PySequence_Check(modes_to_sample):
        if len(modes_to_sample) != n_modes_to_sample:
            raise ValueError("size of modes_to_sample not matching n_modes_to_sample")
        modesData = modes_to_sample
        modesPtr = modesData.data()
    else:  # a pointer address
        modesPtr = <int32_t*><intptr_t>modes_to_sample
    
    cdef _StateSampler sampler
    with nogil:
        status = cutensornetCreateSampler(
            <_Handle>handle, <_State>state, 
            n_modes_to_sample, modesPtr, &sampler)
    check_status(status)
    return <intptr_t> sampler


cdef dict sampler_attribute_sizes = {
    CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES: _numpy.int32
}


cpdef sampler_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding sampler attribute.

    Args:
        attr (SamplerAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`sampler_configure`.
    """
    return sampler_attribute_sizes[attr]


cpdef sampler_configure(
        intptr_t handle, intptr_t sampler, int attr, intptr_t buf, size_t size):
    """Configures the tensor network state sampler.

    Args:
        handle (intptr_t): The library handle.
        sampler (intptr_t): The tensor network sampler handle.
        attr (SamplerAttribute): The attribute to configure.
        buf (intptr_t): The pointer address (as Python :class:`int`) for storing
            the returned attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`sampler_get_attribute_dtype`.

    .. seealso:: `cutensornetSamplerConfigure`
    """
    with nogil:
        status = cutensornetSamplerConfigure(
            <_Handle>handle, <_StateSampler>sampler,
            <_SamplerAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef sampler_prepare(
        intptr_t handle, intptr_t sampler, size_t max_workspace_size_device, intptr_t workspace, intptr_t stream):
    """Prepares computation of the tensor network state marginal distribution.

    Args:
        handle (intptr_t): The library handle.
        sampler (intptr_t): The tensor network sampler.
        max_workspace_size_device (size_t): The maximal device workspace size (in bytes) allowed 
            for the sampling computation.
        workspace (intptr_t): The workspace descriptor.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetSamplerPrepare`
    """
    with nogil:
        status = cutensornetSamplerPrepare(
            <_Handle>handle, <_StateSampler>sampler,
            max_workspace_size_device, <_WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef sampler_sample(
        intptr_t handle, intptr_t sampler, int64_t n_shots,
        intptr_t workspace, intptr_t samples, intptr_t stream):
    """Computes the tensor network state marginal distribution.

    Args:
        handle (intptr_t): The library handle.
        sampler (intptr_t): The tensor network sampler.
        n_shots (int64_t): The number of shots.
        workspace (intptr_t): The workspace descriptor.
        samples (intptr_t): The pointer address (as Python :class:`int`) for storing
            the computed samples.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetSamplerSample`
    """
    with nogil:
        status = cutensornetSamplerSample(
            <_Handle>handle, <_StateSampler>sampler, n_shots, 
            <_WorkspaceDescriptor>workspace, 
            <int64_t*>samples, <Stream>stream)
    check_status(status)


cpdef destroy_sampler(intptr_t sampler):
    """Destroy a tensor network state sampler.

    Args:
        sampler (intptr_t): The tensor network state sampler.

    .. seealso:: `cutensornetDestroySampler`
    """
    with nogil:
        status = cutensornetDestroySampler(<_StateSampler>sampler)
    check_status(status)


cdef dict state_attribute_sizes = {
    CUTENSORNET_STATE_MPS_CANONICAL_CENTER: _numpy.int32,
    CUTENSORNET_STATE_MPS_SVD_CONFIG_ABS_CUTOFF: _numpy.float64,
    CUTENSORNET_STATE_MPS_SVD_CONFIG_REL_CUTOFF: _numpy.float64,
    CUTENSORNET_STATE_MPS_SVD_CONFIG_S_NORMALIZATION: _numpy.int32, # = sizeof(enum value)
    CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO: _numpy.int32, # = sizeof(enum value)
    CUTENSORNET_STATE_MPS_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF: _numpy.float64,
    CUTENSORNET_STATE_NUM_HYPER_SAMPLES: _numpy.int32
}


cpdef state_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding state attribute.

    Args:
        attr (StateAttribute): The attribute to query. The enum CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO is not supported,
            the dtype of which can be queried by :func:`tensor_svd_algo_params_get_dtype`.

    Returns:
        The data type of the queried attribute. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`state_configure`.
    """
    if attr == CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO_PARAMS:
        raise ValueError("For CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO_PARAMS, use `tensor_svd_algo_params_get_dtype` to get the dtype")
    dtype = state_attribute_sizes[attr]
    if attr == CUTENSORNET_STATE_MPS_SVD_CONFIG_S_NORMALIZATION:
        if _numpy.dtype(dtype).itemsize != sizeof(_TensorSVDNormalization):
            warnings.warn("binary size may be incompatible")
    elif attr == CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO:
        if _numpy.dtype(dtype).itemsize != sizeof(_TensorSVDAlgo):
            warnings.warn("binary size may be incompatible")
    return dtype


cpdef state_configure(intptr_t handle, intptr_t state, int attr, intptr_t buf, size_t size):
    """Configures computation of the tensor network state.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.
        attr (StateAttribute): The attribute to configure.
        buf (intptr_t): The pointer address (as Python :class:`int`) for storing
            the returned attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`state_get_attribute_dtype`.

    .. seealso:: `cutensornetStateConfigure`
    """
    with nogil:
        status = cutensornetStateConfigure(
            <_Handle>handle, <_State>state,
            <_StateAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef state_prepare(
        intptr_t handle, intptr_t state,
        size_t max_workspace_size_device, intptr_t workspace, intptr_t stream):
    """Prepares computation of the tensor network state representation.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state handle.
        max_workspace_size_device (size_t): The maximal device workspace size (in bytes) allowed
            for the mariginal computation.
        workspace (intptr_t): The workspace descriptor.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetStatePrepare`
    """
    with nogil:
        status = cutensornetStatePrepare(
            <_Handle>handle, <_State>state,
            max_workspace_size_device, <_WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef tuple state_compute(
        intptr_t handle, intptr_t state, intptr_t workspace,
        state_tensors_out, intptr_t stream):
    """Computes the tensor network state representation.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.
        workspace (intptr_t): The workspace descriptor.
        state_tensors_out: A host array of pointer addresses (as Python :class:`int`) for
            each output tensor (on device). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    Returns:
        tuple:
            The metadata of the output tensors: ``(extents_out, strides_out)``.

    .. seealso:: `cutensornetStateCompute`
    """
    cdef int32_t num_tensors = 0
    with nogil:
        status = cutensornetGetOutputStateDetails(
            <_Handle>handle, <_State>state,
            &num_tensors, NULL, NULL, NULL)
    check_status(status)

    num_modes = _numpy.empty(num_tensors, dtype=_numpy.int32)
    cdef int32_t* numModesPtr = <int32_t*><intptr_t>num_modes.ctypes.data
    with nogil:
        status = cutensornetGetOutputStateDetails(
            <_Handle>handle, <_State>state,
            &num_tensors, numModesPtr, NULL, NULL)
    check_status(status)

    extents_out_py = [_numpy.empty(num_modes[i], dtype=_numpy.int64) for i in range(num_tensors)]
    strides_out_py = [_numpy.empty(num_modes[i], dtype=_numpy.int64) for i in range(num_tensors)]

    cdef vector[intptr_t] extentsOut
    cdef vector[intptr_t] stridesOut
    for i in range(num_tensors):
        extentsOut.push_back(<intptr_t>extents_out_py[i].ctypes.data)
        stridesOut.push_back(<intptr_t>strides_out_py[i].ctypes.data)

    cdef int64_t** extentsOutPtr = <int64_t**>(extentsOut.data())
    cdef int64_t** stridesOutPtr = <int64_t**>(stridesOut.data())

    cdef vector[intptr_t] stateTensorsOutData
    cdef void** stateTensorsOutPtr
    if cpython.PySequence_Check(state_tensors_out):
        stateTensorsOutData = state_tensors_out
        stateTensorsOutPtr = <void**>(stateTensorsOutData.data())
    else:  # a pointer address
        stateTensorsOutPtr = <void**><intptr_t>state_tensors_out

    with nogil:
        status = cutensornetStateCompute(
            <_Handle>handle, <_State>state, <_WorkspaceDescriptor>workspace,
            extentsOutPtr, stridesOutPtr, stateTensorsOutPtr, <Stream>stream)
    check_status(status)
    return (extents_out_py, strides_out_py)


cpdef tuple get_output_state_details(intptr_t handle, intptr_t state):
    """Get the output state tensors' metadata.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.

    Returns:
        tuple:
            The metadata of the output tensor: ``(num_tensors, num_modes, extents,
            strides)``.

    .. seealso:: `cutensornetGetOutputStateDetails`
    """
    cdef int32_t num_tensors = 0
    with nogil:
        status = cutensornetGetOutputStateDetails(
            <_Handle>handle, <_State>state,
            &num_tensors, NULL, NULL, NULL)
    check_status(status)

    num_modes = _numpy.empty(num_tensors, dtype=_numpy.int32)
    cdef int32_t* numModesPtr = <int32_t*><intptr_t>num_modes.ctypes.data
    with nogil:
        status = cutensornetGetOutputStateDetails(
            <_Handle>handle, <_State>state,
            &num_tensors, numModesPtr, NULL, NULL)
    check_status(status)
    extents_out_py = [_numpy.empty(num_modes[i], dtype=_numpy.int64) for i in range(num_tensors)]
    strides_out_py = [_numpy.empty(num_modes[i], dtype=_numpy.int64) for i in range(num_tensors)]

    cdef vector[intptr_t] extentsOut
    cdef vector[intptr_t] stridesOut
    for i in range(num_tensors):
        extentsOut.push_back(<intptr_t>extents_out_py[i].ctypes.data)
        stridesOut.push_back(<intptr_t>strides_out_py[i].ctypes.data)

    cdef int64_t** extentsOutPtr = <int64_t**>(extentsOut.data())
    cdef int64_t** stridesOutPtr = <int64_t**>(stridesOut.data())
    with nogil:
        status = cutensornetGetOutputStateDetails(
            <_Handle>handle, <_State>state,
            &num_tensors, NULL, extentsOutPtr, stridesOutPtr)
    check_status(status)
    return (num_tensors, num_modes, extents_out_py, strides_out_py)

cpdef state_finalize_mps(
        intptr_t handle, intptr_t state, int boundary_condition, extents_out, strides_out):
    """Set the target MPS representation.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.
        boundary_condition (BoundaryCondition): The boundary condition of the initial MPS state.
        extents_out: A host array of extents for all target MPS tensors. It can be

            - an :class:`int` as the pointer address to the nested sequence
            - a Python sequence of :class:`int`, each of which is a pointer address
              to the corresponding tensor's extents
            - a nested Python sequence of :class:`int`

        strides_out: A host array of strides for all target MPS tensors. It can be

            - an :class:`int` as the pointer address to the nested sequence
            - a Python sequence of :class:`int`, each of which is a pointer address
              to the corresponding tensor's strides
            - a nested Python sequence of :class:`int`


    .. seealso:: `cutensornetStateFinalizeMPS`
    """
    # extents_out can be:
    #   - a plain pointer address
    #   - a Python sequence (of pointer addresses)
    #   - a nested Python sequence (of int64_t)
    # Note: it cannot be a mix of sequences and ints.
    cdef vector[intptr_t] extentsOutCData
    cdef int64_t** extentsOutPtr
    if is_nested_sequence(extents_out):
        # flatten the 2D sequence
        extentsOutPyData = []
        for i in extents_out:
            # too bad a Python list can't hold C++ vectors, so we use NumPy
            # arrays as the container here to keep data alive
            data = _numpy.asarray(i, dtype=_numpy.int64)
            assert data.ndim == 1
            extentsOutPyData.append(data)
            extentsOutCData.push_back(<intptr_t>data.ctypes.data)
        extentsOutPtr = <int64_t**>(extentsOutCData.data())
    elif cpython.PySequence_Check(extents_out):
        # handle 1D sequence
        extentsOutCData = extents_out
        extentsOutPtr = <int64_t**>(extentsOutCData.data())
    else:
        # a pointer address, take it as is
        extentsOutPtr = <int64_t**><intptr_t>extents_out

    # strides_out can be:
    #   - a plain pointer address
    #   - a Python sequence (of pointer addresses)
    #   - a nested Python sequence (of int64_t)
    # Note: it cannot be a mix of sequences and ints.
    cdef vector[intptr_t] stridesOutCData
    cdef int64_t** stridesOutPtr
    if is_nested_sequence(strides_out):
        # flatten the 2D sequence
        stridesOutPyData = []
        for i in strides_out:
            # too bad a Python list can't hold C++ vectors, so we use NumPy
            # arrays as the container here to keep data alive
            data = _numpy.asarray(i, dtype=_numpy.int64)
            assert data.ndim == 1
            stridesOutPyData.append(data)
            stridesOutCData.push_back(<intptr_t>data.ctypes.data)
        stridesOutPtr = <int64_t**>(stridesOutCData.data())
    elif cpython.PySequence_Check(strides_out):
        # handle 1D sequence
        stridesOutCData = strides_out
        stridesOutPtr = <int64_t**>(stridesOutCData.data())
    else:
        # a pointer address, take it as is
        stridesOutPtr = <int64_t**><intptr_t>strides_out

    with nogil:
        status = cutensornetStateFinalizeMPS(
            <_Handle>handle, <_State>state, <_BoundaryCondition>boundary_condition,
            extentsOutPtr, stridesOutPtr)
    check_status(status)


cpdef intptr_t create_network_operator(
        intptr_t handle, int32_t n_state_modes, state_mode_extents, int data_type) except*:
    """Create a tensor network operator of a given shape.

    Args:
        handle (intptr_t): The library handle.
        n_state_modes (int32_t): The total number of state modes the operator will act on.
        state_mode_extents:  A host array of extents of each state mode. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        data_type (cuquantum.cudaDataType): The data type of the operator.
    Returns:
        intptr_t: An opaque tensor network operator handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateNetworkOperator`
    """
    # state_mode_extents can be a pointer address, or a Python sequence
    cdef vector[int64_t] stateModeExtentsData
    cdef int64_t* stateModeExtentsPtr
    if cpython.PySequence_Check(state_mode_extents):
        if len(state_mode_extents) != n_state_modes:
            raise ValueError("size of state_mode_extents not matching num_state_modes")
        stateModeExtentsData = state_mode_extents
        stateModeExtentsPtr = stateModeExtentsData.data()
    else:  # a pointer address
        stateModeExtentsPtr = <int64_t*><intptr_t>state_mode_extents

    cdef _NetworkOperator operator
    with nogil:
        status = cutensornetCreateNetworkOperator(
            <_Handle>handle, n_state_modes, stateModeExtentsPtr, <DataType>data_type
            , &operator)
    check_status(status)
    return <intptr_t>operator


cpdef int64_t network_operator_append_product(
        intptr_t handle, intptr_t network_operator, coefficient,
        int32_t num_tensors, num_modes, state_modes, tensor_mode_strides,
        tensor_data) except*:
    """Appends a tensor product component to the tensor network operator.

    Args:
        handle (intptr_t): The library handle.
        network_operator (intptr_t): The tensor network operator the product will be appended to.
        coefficient: Complex coefficient associated with the appended operator component.
        num_tensors: Number of tensor factors in the tensor product.
        num_modes: A host array of number of state modes each appended tensor factor acts on. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        state_modes: A host array of modes each appended tensor factor acts on (length = nModes). It can be

            - an :class:`int` as the pointer address to the nested sequence
            - a Python sequence of :class:`int`, each of which is a pointer address
              to the corresponding tensor's modes
            - a nested Python sequence of :class:`int`

        tensor_modes_strides: Tensor mode strides for each tensor factor (length = nModes * 2). It can be

            - an :class:`int` as the pointer address to the nested sequence
            - a Python sequence of :class:`int`, each of which is a pointer address
              to the corresponding tensor's strides
            - a nested Python sequence of :class:`int`

        tensor_data: A host array of pointer addresses (as Python :class:`int`) for
            each tensor data (on device). It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

    Returns:
        int64_t: A unique sequential integer identifier of the appended tensor network operator component.

    .. seealso:: `cutensornetNetworkOperatorAppendProduct`
    """
    # num_modes can be a pointer address, or a Python sequence
    cdef vector[int32_t] numModesData
    cdef const int32_t* numModesPtr
    if cpython.PySequence_Check(num_modes):
        numModesData = num_modes
        numModesPtr = numModesData.data()
    else:  # a pointer address
        numModesPtr = <const int32_t*><intptr_t>num_modes

    # state_modes can be:
    #   - a plain pointer address
    #   - a Python sequence (of pointer addresses)
    #   - a nested Python sequence (of int32_t)
    # Note: it cannot be a mix of sequences and ints.
    cdef vector[intptr_t] stateModesCData
    cdef const int32_t** stateModesPtr
    if is_nested_sequence(state_modes):
        # flatten the 2D sequence
        stateModesPyData = []
        for i in state_modes:
            # too bad a Python list can't hold C++ vectors, so we use NumPy
            # arrays as the container here to keep data alive
            data = _numpy.asarray(i, dtype=_numpy.int32)
            assert data.ndim == 1
            stateModesPyData.append(data)
            stateModesCData.push_back(<intptr_t>data.ctypes.data)
        stateModesPtr = <const int32_t**>(stateModesCData.data())
    elif cpython.PySequence_Check(state_modes):
        # handle 1D sequence
        stateModesCData = state_modes
        stateModesPtr = <const int32_t**>(stateModesCData.data())
    else:
        # a pointer address, take it as is
        stateModesPtr = <const int32_t**><intptr_t>state_modes

    # tensor_mode_strides can be:
    #   - a plain pointer address
    #   - a Python sequence (of pointer addresses)
    #   - a nested Python sequence (of int64_t)
    # Note: it cannot be a mix of sequences and ints.
    cdef vector[intptr_t] tensorModeStridesCData
    cdef const int64_t** tensorModeStridesPtr
    if is_nested_sequence(tensor_mode_strides):
        # flatten the 2D sequence
        tensorModeStridesPyData = []
        for i in tensor_mode_strides:
            # too bad a Python list can't hold C++ vectors, so we use NumPy
            # arrays as the container here to keep data alive
            data = _numpy.asarray(i, dtype=_numpy.int64)
            assert data.ndim == 1
            tensorModeStridesPyData.append(data)
            tensorModeStridesCData.push_back(<intptr_t>data.ctypes.data)
        tensorModeStridesPtr = <const int64_t**>(tensorModeStridesCData.data())
    elif cpython.PySequence_Check(tensor_mode_strides):
        # handle 1D sequence
        tensorModeStridesCData = tensor_mode_strides
        tensorModeStridesPtr = <const int64_t**>(tensorModeStridesCData.data())
    else:
        # a pointer address, take it as is
        tensorModeStridesPtr = <const int64_t**><intptr_t>tensor_mode_strides

    # tensor_data can be a pointer address, or a Python sequence
    cdef vector[intptr_t] tensorDataData
    cdef const void** tensorDataPtr
    if cpython.PySequence_Check(tensor_data):
        tensorDataData = tensor_data
        tensorDataPtr = <const void**>(tensorDataData.data())
    else:  # a pointer address
        tensorDataPtr = <const void**><intptr_t>tensor_data

    cdef cuDoubleComplex coeff
    coeff.x = coefficient.real
    coeff.y = coefficient.imag

    cdef int64_t componentId = 0
    with nogil:
        status = cutensornetNetworkOperatorAppendProduct(
                <_Handle>handle, <_NetworkOperator>network_operator,
                coeff, num_tensors, numModesPtr, stateModesPtr,
                tensorModeStridesPtr, tensorDataPtr
                , &componentId)
    check_status(status)
    return componentId


cpdef intptr_t create_accessor(
        intptr_t handle, intptr_t state,
        int32_t n_projected_modes, projected_modes, amplitudes_tensor_strides) except*:
    """Create a representation for the tensor network state accessor.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.
        n_projected_modes (int32_t): The number of modes that are projected out for the state.
        projected_modes: A host array of projected modes for the marginal. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        amplitudes_tensor_strides: A host array of strides for the amplitudes tensor. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

    Returns:
        intptr_t: An opaque tensor network state accessor handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateAccessor`
    """

    # projected_modes can be a pointer address, or a Python sequence
    cdef vector[int32_t] projectedModesData
    cdef int32_t* projectedModesPtr
    if cpython.PySequence_Check(projected_modes):
        if len(projected_modes) != n_projected_modes:
            raise ValueError("size of projected_modes not matching n_projected_modes")
        projectedModesData = projected_modes
        projectedModesPtr = projectedModesData.data()
    else:  # a pointer address
        projectedModesPtr = <int32_t*><intptr_t>projected_modes

    # amplitudes_tensor_strides can be a pointer address, or a Python sequence
    cdef vector[int64_t] amplitudesTensorStridesData
    cdef int64_t* amplitudesTensorStridesPtr
    if cpython.PySequence_Check(amplitudes_tensor_strides):
        amplitudesTensorStridesData = amplitudes_tensor_strides
        amplitudesTensorStridesPtr = amplitudesTensorStridesData.data()
    else:  # a pointer address
        amplitudesTensorStridesPtr = <int64_t*><intptr_t>amplitudes_tensor_strides

    cdef _StateAccessor accessor
    with nogil:
        status = cutensornetCreateAccessor(
            <_Handle>handle, <_State>state,
            n_projected_modes, projectedModesPtr,
            amplitudesTensorStridesPtr, &accessor)
    check_status(status)
    return <intptr_t>accessor


cdef dict accessor_attribute_sizes = {
    CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES: _numpy.int32
}


cpdef accessor_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding accessor attribute.

    Args:
        attr (AccessorAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`accessor_configure`.
    """
    return accessor_attribute_sizes[attr]


cpdef accessor_configure(intptr_t handle, intptr_t accessor, int attr, intptr_t buf, size_t size):
    """Configures computation of the tensor network state accessor.

    Args:
        handle (intptr_t): The library handle.
        accessor (intptr_t): The tensor network state accessor computation handle.
        attr (AccessorAttribute): The attribute to configure.
        buf (intptr_t): The pointer address (as Python :class:`int`) for storing
            the returned attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`accessor_get_attribute_dtype`.

    .. seealso:: `cutensornetAccessorConfigure`
    """
    with nogil:
        status = cutensornetAccessorConfigure(
            <_Handle>handle, <_StateAccessor>accessor,
            <_AccessorAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef accessor_prepare(
        intptr_t handle, intptr_t accessor,
        size_t max_workspace_size_device, intptr_t workspace, intptr_t stream):
    """Prepares computation of the tensor network state accessor.

    Args:
        handle (intptr_t): The library handle.
        accessor (intptr_t): The tensor network state accessor handle.
        max_workspace_size_device (size_t): The maximal device workspace size (in bytes) allowed
            for the accessor computation.
        workspace (intptr_t): The workspace descriptor.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetAccessorPrepare`
    """
    with nogil:
        status = cutensornetAccessorPrepare(
            <_Handle>handle, <_StateAccessor>accessor,
            max_workspace_size_device, <_WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef accessor_compute(
        intptr_t handle, intptr_t accessor, projected_mode_values,
        intptr_t workspace, intptr_t amplitudes_tensor, intptr_t state_norm, intptr_t stream):
    """Computes the tensor network state amplitudes.

    Args:
        handle (intptr_t): The library handle.
        accessor (intptr_t): The tensor network state accessor handle.
        projected_mode_values: A host array of values for the projected modes. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`

        workspace (intptr_t): The workspace descriptor.
        amplitudes_tensor (intptr_t): The pointer address (as Python :class:`int`) for storing
            the computed amplitudes.
        state_norm (intptr_t): The pointer address (as Python :class:`int`) for storing
            the 2-norm of the underlying state. If set to 0 (`NULL` pointer), the norm calculation will be ignored.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetAccessorCompute`
    """
    # projected_mode_values can be a pointer address, or a Python sequence
    cdef vector[int64_t] projectedModeValuesData
    cdef int64_t* projectedModeValuesPtr
    if cpython.PySequence_Check(projected_mode_values):
        projectedModeValuesData = projected_mode_values
        projectedModeValuesPtr = projectedModeValuesData.data()
    else:  # a pointer address
        projectedModeValuesPtr = <int64_t*><intptr_t>projected_mode_values

    with nogil:
        status = cutensornetAccessorCompute(
            <_Handle>handle, <_StateAccessor>accessor,
            projectedModeValuesPtr, <_WorkspaceDescriptor>workspace,
            <void*>amplitudes_tensor, <void*>state_norm, <Stream>stream)
    check_status(status)


cpdef destroy_accessor(intptr_t accessor):
    """Destroy a tensor network state accessor handle.

    Args:
        marginal (intptr_t): The tensor network state accessor handle.

    .. seealso:: `cutensornetDestroyAccessor`
    """
    with nogil:
        status = cutensornetDestroyAccessor(<_StateAccessor>accessor)
    check_status(status)


cpdef destroy_network_operator(intptr_t network_operator):
    """Destroy a tensor network operator.

    Args:
        network_operator (intptr_t): The tensor network operator.

    .. seealso:: `cutensornetDestroyNetworkOperator`
    """
    with nogil:
        status = cutensornetDestroyNetworkOperator(<_NetworkOperator>network_operator)
    check_status(status)


cpdef intptr_t create_expectation(
        intptr_t handle, intptr_t state, intptr_t operator) except*:
    """Create a representation for the tensor network state expectation value.

    Args:
        handle (intptr_t): The library handle.
        state (intptr_t): The tensor network state.

    Returns:
        intptr_t: An opaque tensor network state expectation handle (as Python :class:`int`).

    .. seealso:: `cutensornetCreateExpectation`
    """
    cdef _StateExpectation expectation
    with nogil:
        status = cutensornetCreateExpectation(
            <_Handle>handle, <_State>state, <_NetworkOperator>operator
            , &expectation)
    check_status(status)
    return <intptr_t>expectation


cdef dict expectation_attribute_sizes = {
    CUTENSORNET_EXPECTATION_OPT_NUM_HYPER_SAMPLES: _numpy.int32
}


cpdef expectation_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding expectation attribute.

    Args:
        attr (ExpectationAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`expectation_configure`.
    """
    return expectation_attribute_sizes[attr]


cpdef expectation_configure(intptr_t handle, intptr_t expectation, int attr, intptr_t buf, size_t size):
    """Configures computation of the tensor network state expectation value.

    Args:
        handle (intptr_t): The library handle.
        expectation (intptr_t): The tensor network expectation computation handle.
        attr (ExpectationAttribute): The attribute to configure.
        buf (intptr_t): The pointer address (as Python :class:`int`) of the attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`expectation_get_attribute_dtype`.

    .. seealso:: `cutensornetExpectationConfigure`
    """
    with nogil:
        status = cutensornetExpectationConfigure(
            <_Handle>handle, <_StateExpectation>expectation,
            <_ExpectationAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef expectation_prepare(
        intptr_t handle, intptr_t expectation,
        size_t max_workspace_size_device, intptr_t workspace, intptr_t stream):
    """Prepares computation of the tensor network state expectation.

    Args:
        handle (intptr_t): The library handle.
        expectation (intptr_t): The tensor network expectation computation handle.
        max_workspace_size_device (size_t): The maximal device workspace size (in bytes) allowed
            for the expectation value computation.
        workspace (intptr_t): The workspace descriptor.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetExpectationPrepare`
    """
    with nogil:
        status = cutensornetExpectationPrepare(
            <_Handle>handle, <_StateExpectation>expectation,
            max_workspace_size_device, <_WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef expectation_compute(
        intptr_t handle, intptr_t expectation,
        intptr_t workspace, intptr_t expectation_value, intptr_t state_norm, intptr_t stream):
    """Computes the tensor network state expectation value.

    Args:
        handle (intptr_t): The library handle.
        expectation (intptr_t): The tensor network expectation computation handle.
        workspace (intptr_t): The workspace descriptor.
        expectation_value (intptr_t): The pointer address (as Python :class:`int`) for storing
            the computed expectation_value (stored on host).
        state_norm (intptr_t): The pointer address (as Python :class:`int`) for storing
            the 2-norm of the underlying state. If set to 0 (`NULL` pointer), the norm calculation will be ignored.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            :class:`int`).

    .. seealso:: `cutensornetExpectationCompute`
    """
    with nogil:
        status = cutensornetExpectationCompute(
            <_Handle>handle, <_StateExpectation>expectation,
            <_WorkspaceDescriptor>workspace, <void*>expectation_value, <void*>state_norm, <Stream>stream)
    check_status(status)


cpdef destroy_expectation(intptr_t expectation):
    """Destroy a tensor network expectation value representation.

    Args:
        expectation (intptr_t): The tensor network expectation value representation.

    .. seealso:: `cutensornetDestroyExpectation`
    """
    with nogil:
        status = cutensornetDestroyExpectation(<_StateExpectation>expectation)
    check_status(status)


class NetworkAttribute(IntEnum):
    """See `cutensornetNetworkAttributes_t`."""
    INPUT_TENSORS_NUM_CONSTANT = CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONSTANT
    INPUT_TENSORS_CONSTANT = CUTENSORNET_NETWORK_INPUT_TENSORS_CONSTANT
    INPUT_TENSORS_NUM_CONJUGATED = CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONJUGATED
    INPUT_TENSORS_CONJUGATED = CUTENSORNET_NETWORK_INPUT_TENSORS_CONJUGATED
    INPUT_TENSORS_NUM_REQUIRE_GRAD = CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_REQUIRE_GRAD
    INPUT_TENSORS_REQUIRE_GRAD = CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD

class GraphAlgo(IntEnum):
    """See `cutensornetGraphAlgo_t`."""
    RB = CUTENSORNET_GRAPH_ALGO_RB
    KWAY = CUTENSORNET_GRAPH_ALGO_KWAY

class MemoryModel(IntEnum):
    """See `cutensornetMemoryModel_t`."""
    HEURISTIC = CUTENSORNET_MEMORY_MODEL_HEURISTIC
    CUTENSOR = CUTENSORNET_MEMORY_MODEL_CUTENSOR

class OptimizerCost(IntEnum):
    """See `cutensornetOptimizerCost_t`."""
    FLOPS = CUTENSORNET_OPTIMIZER_COST_FLOPS
    TIME = CUTENSORNET_OPTIMIZER_COST_TIME
    TIME_TUNED = CUTENSORNET_OPTIMIZER_COST_TIME_TUNED

class ContractionOptimizerConfigAttribute(IntEnum):
    """See `cutensornetContractionOptimizerConfigAttributes_t`."""
    GRAPH_NUM_PARTITIONS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_PARTITIONS
    GRAPH_CUTOFF_SIZE = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_CUTOFF_SIZE
    GRAPH_ALGORITHM = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM
    GRAPH_IMBALANCE_FACTOR = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR
    GRAPH_NUM_ITERATIONS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_ITERATIONS
    GRAPH_NUM_CUTS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS
    RECONFIG_NUM_ITERATIONS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS
    RECONFIG_NUM_LEAVES = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES
    SLICER_DISABLE_SLICING = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING
    SLICER_MEMORY_MODEL = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL
    SLICER_MEMORY_FACTOR = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR
    SLICER_MIN_SLICES = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES
    SLICER_SLICE_FACTOR = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_SLICE_FACTOR
    HYPER_NUM_SAMPLES = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES
    HYPER_NUM_THREADS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS
    SIMPLIFICATION_DISABLE_DR = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR
    SEED = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED
    COST_FUNCTION_OBJECTIVE = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE
    CACHE_REUSE_NRUNS = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS
    SMART_OPTION = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION

class ContractionOptimizerInfoAttribute(IntEnum):
    """See `cutensornetContractionOptimizerInfoAttributes_t`."""
    NUM_SLICES = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES
    NUM_SLICED_MODES = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES
    SLICED_MODE = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE
    SLICED_EXTENT = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT
    PATH = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH
    PHASE1_FLOP_COUNT = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PHASE1_FLOP_COUNT
    FLOP_COUNT = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT
    LARGEST_TENSOR = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR
    SLICING_OVERHEAD = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_OVERHEAD
    INTERMEDIATE_MODES = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES
    NUM_INTERMEDIATE_MODES = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_INTERMEDIATE_MODES
    EFFECTIVE_FLOPS_EST = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS_EST
    RUNTIME_EST = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_RUNTIME_EST
    SLICING_CONFIG = CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG

class ContractionAutotunePreferenceAttribute(IntEnum):
    """See `cutensornetContractionAutotunePreferenceAttributes_t`."""
    MAX_ITERATIONS = CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS
    INTERMEDIATE_MODES = CUTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES

class WorksizePref(IntEnum):
    """See `cutensornetWorksizePref_t`."""
    MIN = CUTENSORNET_WORKSIZE_PREF_MIN
    RECOMMENDED = CUTENSORNET_WORKSIZE_PREF_RECOMMENDED
    MAX = CUTENSORNET_WORKSIZE_PREF_MAX

class Memspace(IntEnum):
    """See `cutensornetMemspace_t`."""
    DEVICE = CUTENSORNET_MEMSPACE_DEVICE
    HOST = CUTENSORNET_MEMSPACE_HOST

class WorkspaceKind(IntEnum):
    """See `cutensornetWorkspaceKind_t`."""
    SCRATCH = CUTENSORNET_WORKSPACE_SCRATCH
    CACHE = CUTENSORNET_WORKSPACE_CACHE

class TensorSVDConfigAttribute(IntEnum):
    """See `cutensornetTensorSVDConfigAttributes_t`."""
    ABS_CUTOFF = CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF
    REL_CUTOFF = CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF
    S_NORMALIZATION = CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION
    S_PARTITION = CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION
    ALGO = CUTENSORNET_TENSOR_SVD_CONFIG_ALGO
    ALGO_PARAMS = CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS
    DISCARDED_WEIGHT_CUTOFF = CUTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF

class TensorSVDNormalization(IntEnum):
    """See `cutensornetTensorSVDNormalization_t`."""
    NONE = CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE
    L1 = CUTENSORNET_TENSOR_SVD_NORMALIZATION_L1
    L2 = CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2
    LINF = CUTENSORNET_TENSOR_SVD_NORMALIZATION_LINF

class TensorSVDPartition(IntEnum):
    """See `cutensornetTensorSVDPartition_t`."""
    NONE = CUTENSORNET_TENSOR_SVD_PARTITION_NONE
    US = CUTENSORNET_TENSOR_SVD_PARTITION_US
    SV = CUTENSORNET_TENSOR_SVD_PARTITION_SV
    UV_EQUAL = CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL

class TensorSVDInfoAttribute(IntEnum):
    """See `cutensornetTensorSVDInfoAttributes_t`."""
    FULL_EXTENT = CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT
    REDUCED_EXTENT = CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT
    DISCARDED_WEIGHT = CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT
    ALGO = CUTENSORNET_TENSOR_SVD_INFO_ALGO
    ALGO_STATUS = CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS

class TensorSVDAlgo(IntEnum):
    """See `cutensornetTensorSVDAlgo_t`."""
    GESVD = CUTENSORNET_TENSOR_SVD_ALGO_GESVD
    GESVDJ = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ
    GESVDP = CUTENSORNET_TENSOR_SVD_ALGO_GESVDP
    GESVDR = CUTENSORNET_TENSOR_SVD_ALGO_GESVDR

class GateSplitAlgo(IntEnum):
    """See `cutensornetGateSplitAlgo_t`."""
    DIRECT = CUTENSORNET_GATE_SPLIT_ALGO_DIRECT
    REDUCED = CUTENSORNET_GATE_SPLIT_ALGO_REDUCED

class StatePurity(IntEnum):
    """See `cutensornetStatePurity_t`."""
    PURE = CUTENSORNET_STATE_PURITY_PURE

class MarginalAttribute(IntEnum):
    """See `cutensornetMarginalAttributes_t`."""
    OPT_NUM_HYPER_SAMPLES = CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES

class SamplerAttribute(IntEnum):
    """See `cutensornetSamplerAttributes_t`."""
    OPT_NUM_HYPER_SAMPLES = CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES

class AccessorAttribute(IntEnum):
    """See `cutensornetAccessorAttributes_t`."""
    OPT_NUM_HYPER_SAMPLES = CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES

class ExpectationAttribute(IntEnum):
    """See `cutensornetExpectationAttributes_t`."""
    OPT_NUM_HYPER_SAMPLES = CUTENSORNET_EXPECTATION_OPT_NUM_HYPER_SAMPLES

class BoundaryCondition(IntEnum):
    """See `cutensornetBoundaryCondition_t`."""
    OPEN = CUTENSORNET_BOUNDARY_CONDITION_OPEN

class StateAttribute(IntEnum):
    """See `cutensornetStateAttributes_t`."""
    MPS_CANONICAL_CENTER = CUTENSORNET_STATE_MPS_CANONICAL_CENTER
    MPS_SVD_CONFIG_ABS_CUTOFF = CUTENSORNET_STATE_MPS_SVD_CONFIG_ABS_CUTOFF
    MPS_SVD_CONFIG_REL_CUTOFF = CUTENSORNET_STATE_MPS_SVD_CONFIG_REL_CUTOFF
    MPS_SVD_CONFIG_S_NORMALIZATION = CUTENSORNET_STATE_MPS_SVD_CONFIG_S_NORMALIZATION
    MPS_SVD_CONFIG_ALGO = CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO
    MPS_SVD_CONFIG_ALGO_PARAMS = CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO_PARAMS
    MPS_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF = CUTENSORNET_STATE_MPS_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF
    NUM_HYPER_SAMPLES = CUTENSORNET_STATE_NUM_HYPER_SAMPLES

del IntEnum


# expose them to Python
MAJOR_VER = CUTENSORNET_MAJOR
MINOR_VER = CUTENSORNET_MINOR
PATCH_VER = CUTENSORNET_PATCH
VERSION = CUTENSORNET_VERSION

# numpy dtypes 
tensor_qualifiers_dtype = _numpy.dtype(
    {'names':('is_conjugate', 'is_constant', 'requires_gradient'),
     'formats': (_numpy.int32, _numpy.int32, _numpy.int32, ),
     'itemsize': sizeof(_TensorQualifiers),
    }, align=True
)

# who owns a reference to user-provided Python objects (k: owner, v: object)
cdef dict owner_pyobj = {}
