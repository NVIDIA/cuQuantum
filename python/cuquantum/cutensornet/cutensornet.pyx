# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
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
    int cutensornetWorkspaceSet(
        const _Handle, _WorkspaceDescriptor, _Memspace,
        void* const, uint64_t)
    int cutensornetWorkspaceGet(
        const _Handle, const _WorkspaceDescriptor, _Memspace,
        void**, uint64_t*)
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
    cdef uint64_t workspaceSize
    with nogil:
        status = cutensornetWorkspaceGetSize(
            <_Handle>handle, <_WorkspaceDescriptor>workspace,
            <_WorksizePref>pref, <_Memspace>mem_space,
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
    with nogil:
        status = cutensornetWorkspaceSet(
            <_Handle>handle, <_WorkspaceDescriptor>workspace,
            <_Memspace>mem_space,
            <void*>workspace_ptr, <uint64_t>workspace_size)
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

    .. seealso:: `cutensornetWorkspaceSet`
    """
    cdef void* workspace_ptr
    cdef uint64_t workspace_size

    with nogil:
        status = cutensornetWorkspaceGet(
            <_Handle>handle, <_WorkspaceDescriptor>workspace,
            <_Memspace>mem_space,
            &workspace_ptr, &workspace_size)
    check_status(status)
    return (<intptr_t>workspace_ptr, workspace_size)


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
            path_obj["node_pair"] = path.ctypes.ptr

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
}

cpdef tensor_svd_config_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding tensor SVD config attribute.

    Args:
        attr (TensorSVDConfigAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`tensor_svd_config_get_attribute`
        and :func:`tensor_svd_config_set_attribute`.
    """
    dtype = tensor_svd_cfg_sizes[attr]
    if attr == CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION:
        if _numpy.dtype(dtype).itemsize != sizeof(_TensorSVDNormalization):
            warnings.warn("binary size may be incompatible")
    elif attr == CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION:
        if _numpy.dtype(dtype).itemsize != sizeof(_TensorSVDPartition):
            warnings.warn("binary size may be incompatible")
    return dtype

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
}

cpdef tensor_svd_info_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding tensor SVD info attribute.

    Args:
        attr (TensorSVDInfoAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute. The returned dtype is always
        a valid NumPy dtype object.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`tensor_svd_info_get_attribute`.

    """
    return tensor_svd_info_sizes[attr]

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

class TensorSVDConfigAttribute(IntEnum):
    """See `cutensornetTensorSVDConfigAttributes_t`."""
    ABS_CUTOFF = CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF
    REL_CUTOFF = CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF
    S_NORMALIZATION = CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION
    S_PARTITION = CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION

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

class GateSplitAlgo(IntEnum):
    """See `cutensornetGateSplitAlgo_t`."""
    DIRECT = CUTENSORNET_GATE_SPLIT_ALGO_DIRECT
    REDUCED = CUTENSORNET_GATE_SPLIT_ALGO_REDUCED

del IntEnum


# expose them to Python
MAJOR_VER = CUTENSORNET_MAJOR
MINOR_VER = CUTENSORNET_MINOR
PATCH_VER = CUTENSORNET_PATCH
VERSION = CUTENSORNET_VERSION

# numpy dtypes 
tensor_qualifiers_dtype = _numpy.dtype(
    {'names':('is_conjugate', ),
     'formats': (_numpy.int32, ),
     'itemsize': sizeof(_TensorQualifiers),
    }, align=True
)

# who owns a reference to user-provided Python objects (k: owner, v: object)
cdef dict owner_pyobj = {}
