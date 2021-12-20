# distutils: language = c++

cimport cpython
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport intptr_t, int32_t, uint32_t, int64_t, uint64_t, uintptr_t
from libcpp.vector cimport vector

from cuquantum.utils cimport is_nested_sequence

from enum import IntEnum
import warnings

import numpy as _numpy


cdef extern from * nogil:
    # from CUDA
    ctypedef int Stream 'cudaStream_t'
    ctypedef enum DataType 'cudaDataType_t':
        pass

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
        const int64_t* const[], const int32_t* const[], const uint32_t[],
        int32_t, const int64_t[], const int64_t[], const int32_t[],
        uint32_t, DataType, _ComputeType, _NetworkDescriptor*)
    int cutensornetDestroyNetworkDescriptor(_NetworkDescriptor)

    # optimizer info
    int cutensornetCreateContractionOptimizerInfo(
        const _Handle, const _NetworkDescriptor,
        _ContractionOptimizerInfo*)
    int cutensornetDestroyContractionOptimizerInfo(
        _ContractionOptimizerInfo)
    int cutensornetContractionOptimizerInfoGetAttribute(
        const _Handle, const _ContractionOptimizerInfo,
        _ContractionOptimizerInfoAttribute, void*, size_t)
    int cutensornetContractionOptimizerInfoSetAttribute(
        const _Handle, _ContractionOptimizerInfo,
        _ContractionOptimizerInfoAttribute, const void*, size_t)

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

    # contraction
    int cutensornetContractionGetWorkspaceSize(
        const _Handle, const _NetworkDescriptor,
        const _ContractionOptimizerInfo,
        uint64_t* workspaceSize)
    int cutensornetContractionOptimize(
        const _Handle, const _NetworkDescriptor,
        const _ContractionOptimizerConfig,
        uint64_t, _ContractionOptimizerInfo)
    int cutensornetCreateContractionPlan(
        const _Handle, const _NetworkDescriptor,
        const _ContractionOptimizerInfo,
        const uint64_t, _ContractionPlan)
    int cutensornetDestroyContractionPlan(_ContractionPlan)
    int cutensornetContractionAutotune(
        const _Handle, _ContractionPlan, const void* const[],
        void*, void*, uint64_t, _ContractionAutotunePreference, Stream)
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
    int cutensornetContraction(
        const _Handle, const _ContractionPlan, const void* const[],
        void*, void*, uint64_t, int64_t, Stream)


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
        intptr_t: the opaque library handle (as Python `int`).

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
    """Query the version of the CUDA runtime.

    Returns:
        size_t: the CUDA runtime version (ex: 11040 for CUDA 11.4).

    .. seealso:: `cutensornetGetCudartVersion`
    """
    cdef size_t ver = cutensornetGetCudartVersion()
    return ver


cpdef intptr_t create_network_descriptor(
        intptr_t handle,
        int32_t n_inputs, n_modes_in, extents_in,
        strides_in, modes_in, alignments_in,
        int32_t n_modes_out, extents_out,
        strides_out, modes_out, uint32_t alignment_out,
        int data_type, int compute_type) except*:
    """Create a tensor network descriptor.

    Args:
        handle (intptr_t): The library handle.
        n_inputs (int): The number of input tensors.
        n_modes_in: A host array of the number of modes for each input tensor.
            It can be

            - an `int` as the pointer address to the array
            - a Python sequence of `int`

        extents_in: A host array of extents for each input tensor. It can be

            - an `int` as the pointer address to the nested sequence
            - a Python sequence of `int`, each of which is a pointer address
              to the corresponding tensor's extents
            - a nested Python sequence of `int`

        strides_in: A host array of strides for each input tensor. It can be

            - an `int` as the pointer address to the nested sequence
            - a Python sequence of `int`, each of which is a pointer address
              to the corresponding tensor's strides
            - a nested Python sequence of `int`

        modes_in: A host array of modes for each input tensor. It can be

            - an `int` as the pointer address to the nested sequence
            - a Python sequence of `int`, each of which is a pointer address
              to the corresponding tensor's modes
            - a nested Python sequence of `int`

        alignments_in: A host array of alignments for each input tensor. It can
            be

            - an `int` as the pointer address to the array
            - a Python sequence of `int`

        n_modes_out (int32_t): The number of modes of the output tensor.
        extents_out: The extents of the output tensor (on host). It can be

            - an `int` as the pointer address to the array
            - a Python sequence of `int`

        strides_out: The strides of the output tensor (on host). It can be

            - an `int` as the pointer address to the array
            - a Python sequence of `int`

        modes_out: The modes of the output tensor (on host). It can be

            - an `int` as the pointer address to the array
            - a Python sequence of `int`

        alignment_out (uint32_t): The alignment for the output tensor.
        data_type (cuquantum.cudaDataType): The data type of the input and
            output tensors.
        compute_type (cuquantum.ComputeType): The compute type of the tensor
            contraction.

    Returns:
        intptr_t: An opaque descriptor handle (as Python `int`).

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

    # alignments_in can be a pointer address, or a Python sequence
    cdef vector[uint32_t] alignmentsInData
    cdef uint32_t* alignmentsInPtr
    if cpython.PySequence_Check(alignments_in):
        alignmentsInData = alignments_in
        alignmentsInPtr = alignmentsInData.data()
    else:  # a pointer address
        alignmentsInPtr = <uint32_t*><intptr_t>alignments_in

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
            n_inputs, numModesInPtr, extentsInPtr, stridesInPtr, modesInPtr, alignmentsInPtr,
            n_modes_out, extentsOutPtr, stridesOutPtr, modesOutPtr, alignment_out,
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


cpdef intptr_t create_contraction_optimizer_info(
        intptr_t handle, intptr_t tn_desc) except*:
    """Create a contraction optimizer info object.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): the tensor network descriptor.

    Returns:
        intptr_t: An opaque optimizer info handle (as Python `int`).

    .. seealso:: `cutensornetCreateContractionOptimizerInfo`
    """
    cdef _ContractionOptimizerInfo info
    with nogil:
        status = cutensornetCreateContractionOptimizerInfo(
            <_Handle>handle, <_NetworkDescriptor>tn_desc, &info)
    check_status(status)
    return <intptr_t>info


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

cdef dict contract_opti_info_sizes = {
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES: _numpy.int64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT: _numpy.int64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH: ContractionPath,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PHASE1_FLOP_COUNT: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR: _numpy.float64,
    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_OVERHEAD: _numpy.float64,
}

cpdef contraction_optimizer_info_get_attribute_dtype(int attr):
    """Get the Python data type of the corresponding optimizer info attribute.

    Args:
        attr (ContractionOptimizerInfoAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`contraction_optimizer_info_get_attribute`
        and :func:`contraction_optimizer_info_set_attribute`.

    .. note:: Unlike other enum values, for :data:`ContractionOptimizerInfoAttribute.PATH`
        the following usage pattern is expected:

        .. code-block:: python

            val = ContractionOptimizerInfoAttribute.PATH
            dtype = contraction_optimizer_info_get_attribute_dtype(val)

            # setter
            path = np.asarray([(1, 3), (1, 2), (0, 1)], dtype=np.int32)
            path_obj = dtype(path.size//2, path.ctypes.data)
            contraction_optimizer_info_set_attribute(
                handle, info, val, path_obj.get_data(), path_obj.get_size())

            # getter
            # num_contractions is the number of input tensors minus one
            path = np.empty(2*num_contractions, dtype=np.int32)
            path_obj = dtype(num_contractions, path.ctypes.data)
            contraction_optimizer_info_get_attribute(
                handle, info, val, path_obj.get_data(), path_obj.get_size())
            # now path is filled
            print(path)

        See also the documentation of :class:`ContractionPath`. This design is subject
        to change in a future release.
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
        buf (intptr_t): The pointer address (as Python `int`) for storing
            the returned attribute value.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`contraction_optimizer_info_get_attribute_dtype`.

    .. note:: For getting the :data:`ContractionOptimizerInfoAttribute.PATH` attribute
        please see :func:`contraction_optimizer_info_get_attribute_dtype`.

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
        buf (intptr_t): The pointer address (as Python `int`) to the attribute data.
        size (size_t): The size of ``buf`` (in bytes).

    .. note:: To compute ``size``, use the itemsize of the corresponding data
        type, which can be queried using :func:`contraction_optimizer_info_get_attribute_dtype`.

    .. note:: For setting the :data:`ContractionOptimizerInfoAttribute.PATH` attribute
        please see :func:`contraction_optimizer_info_get_attribute_dtype`.

    .. seealso:: `cutensornetContractionOptimizerInfoSetAttribute`
    """
    with nogil:
        status = cutensornetContractionOptimizerInfoSetAttribute(
            <_Handle>handle, <_ContractionOptimizerInfo>info,
            <_ContractionOptimizerInfoAttribute>attr,
            <void*>buf, size)
    check_status(status)


cpdef intptr_t create_contraction_optimizer_config(
        intptr_t handle) except*:
    """Create a contraction optimizer config object.

    Args:
        handle (intptr_t): The library handle.

    Returns:
        intptr_t: An opaque optimizer config handle (as Python `int`).

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
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR: _numpy.int32,
    CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED: _numpy.int32,
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
        buf (intptr_t): The pointer address (as Python `int`) for storing
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
        buf (intptr_t): The pointer address (as Python `int`) to the attribute data.
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


cpdef uint64_t contraction_get_workspace_size(
        intptr_t handle, intptr_t tn_desc, intptr_t info) except*:
    """Compute the required workspace size for contracting the input tensor
    network.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): the tensor network descriptor.
        info (intptr_t): The optimizer info handle.

    Returns:
        uint64_t: The workspace size (in bytes).

    .. note:: This function should be called either after a contraction path
        is manually set, or after :func:`contraction_optimize` is called.

    .. seealso:: `cutensornetContractionGetWorkspaceSize`
    """
    # TODO(leofang): note in the docstring that the API name deviates
    # from its C counterpart in beta 2
    cdef uint64_t workspaceSize
    with nogil:
        status = cutensornetContractionGetWorkspaceSize(
            <_Handle>handle, <_NetworkDescriptor>tn_desc,
            <_ContractionOptimizerInfo>info, &workspaceSize)
    check_status(status)
    return workspaceSize


cpdef contraction_optimize(
        intptr_t handle, intptr_t tn_desc, intptr_t config,
        uint64_t size_limit, intptr_t info):
    """Optimize the contraction path, slicing, etc, for the given tensor network.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): the tensor network descriptor.
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
        uint64_t workspace_size) except*:
    """Create a contraction plan for the given tensor network and the
    associated path.

    When this function is called, the optimizer info object should already
    contain a contraction path.

    Args:
        handle (intptr_t): The library handle.
        tn_desc (intptr_t): the tensor network descriptor.
        info (intptr_t): The optimizer info handle.
        workspace_size (uint64_t): The workspace size (in bytes).

    Returns:
        intptr_t: An opaque contraction plan handle (as Python `int`).

    .. seealso:: `cutensornetCreateContractionPlan`
    """
    cdef _ContractionPlan plan
    with nogil:
        status = cutensornetCreateContractionPlan(
            <_Handle>handle, <_NetworkDescriptor>tn_desc,
            <_ContractionOptimizerInfo>info,
            workspace_size, &plan)
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
        uint64_t workspace_size, intptr_t pref, intptr_t stream):
    """Autotune the contraction plan to find the best kernels for each pairwise
    tensor contraction.

    The input tensors should form a tensor network that is prescribed by the
    tensor network descriptor that was used to create the contraction plan.

    Args:
        handle (intptr_t): The library handle.
        plan (intptr_t): The contraction plan handle.
        raw_data_in: A host array of pointer addresses (as Python `int`) for
            each input tensor (on device). It can be

            - an `int` as the pointer address to the array
            - a Python sequence of `int`

        raw_data_out (intptr_t): The pointer address (as Python `int`) to the
            output tensor (on device).
        workspace (intptr_t): The pointer address (as Python `int`) to the
            workspace (on device).
        workspace_size (uint64_t): The workspace size (in bytes).
        pref (intptr_t): The autotune preference handle.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            `int`).

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
            rawDataInPtr, <void*>raw_data_out, <void*>workspace,
            workspace_size, <_ContractionAutotunePreference>pref,
            <Stream>stream)
    check_status(status)


cpdef intptr_t create_contraction_autotune_preference(intptr_t handle):
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


cpdef intptr_t destroy_contraction_autotune_preference(intptr_t pref):
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
        buf (intptr_t): The pointer address (as Python `int`) for storing
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
        buf (intptr_t): The pointer address (as Python `int`) to the attribute data.
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
        uint64_t workspace_size, int64_t slice_id, intptr_t stream):
    """Perform the contraction of the input tensors.

    The input tensors should form a tensor network that is prescribed by the
    tensor network descriptor that was used to create the contraction plan.

    Args:
        handle (intptr_t): The library handle.
        plan (intptr_t): The contraction plan handle.
        raw_data_in: A host array of pointer addresses (as Python `int`) for
            each input tensor (on device). It can be

            - an `int` as the pointer address to the array
            - a Python sequence of `int`

        raw_data_out (intptr_t): The pointer address (as Python `int`) to the
            output tensor (on device).
        workspace (intptr_t): The pointer address (as Python `int`) to the
            workspace (on device).
        workspace_size (uint64_t): The workspace size (in bytes).
        slice_id (int64_t): The slice ID.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            `int`).

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

    with nogil:
        status = cutensornetContraction(
            <_Handle>handle, <_ContractionPlan>plan,
            rawDataInPtr, <void*>raw_data_out, <void*>workspace,
            workspace_size, slice_id, <Stream>stream)
    check_status(status)


cdef class ContractionPath:
    """A proxy object to hold a `cutensornetContractionPath_t` struct.

    Users provide the number of contractions and a pointer address to the actual
    contraction path, and this object creates an `cutensornetContractionPath_t`
    instance and fills in the provided information.

    Example:

        .. code-block:: python

            # the pairwise contraction order is stored as C int
            path = np.asarray([(1, 3), (1, 2), (0, 1)], dtype=np.int32)
            path_obj = ContractionPath(path.size//2, path.ctypes.data)

            # get the pointer address to the underlying `cutensornetContractionPath_t`
            my_func(..., path_obj.get_data(), ...)

            # path must outlive path_obj!
            del path_obj
            del path

    Args:
        num_contractions (int): The number of contractions in the provided path.
        data (uintptr_t): The pointer address (as Python `int`) to the provided path.

    .. note::
        Users are responsible for managing the lifetime of the underlying path data
        (i.e. the validity of the ``data`` pointer).

    .. warning::
        The design of how `cutensornetContractionPath_t` is handled in Python is
        experimental and subject to change in a future release.
    """
    cdef _ContractionPath* path

    def __cinit__(self, int num_contractions, uintptr_t data):
        self.path = <_ContractionPath*>PyMem_Malloc(sizeof(_ContractionPath))

    def __dealloc__(self):
        PyMem_Free(<void*>self.path)

    def __init__(self, int num_contractions, uintptr_t data):
        """
        __init__(self, int num_contractions, uintptr_t data)
        """
        self.path.numContractions = num_contractions
        self.path.data = <_NodePair*>data

    def get_path(self):
        """Get the pointer address to the underlying `cutensornetContractionPath_t` struct.

        Returns:
            uintptr_t: The pointer address.
        """
        return <uintptr_t>self.path

    def get_size(self):
        """Get the size of the `cutensornetContractionPath_t` struct.

        Returns:
            size_t: ``sizeof(cutensornetContractionPath_t)``.
        """
        return sizeof(_ContractionPath)


class GraphAlgorithm(IntEnum):
    """See `cutensornetGraphAlgo_t`."""
    RB = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM_RB
    KWAY = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM_KWAY

class MemoryModel(IntEnum):
    """See `cutensornetMemoryModel_t`."""
    SLICER_HEURISTIC = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL_HEURISTIC
    SLICER_CUTENSOR = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL_CUTENSOR

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
    SIMPLIFICATION_DISABLE_DR = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR
    SEED = CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED

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

class ContractionAutotunePreferenceAttribute(IntEnum):
    """See `cutensornetContractionAutotunePreferenceAttributes_t`."""
    MAX_ITERATIONS = CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS

del IntEnum


# expose them to Python
MAJOR_VER = CUTENSORNET_MAJOR
MINOR_VER = CUTENSORNET_MINOR
PATCH_VER = CUTENSORNET_PATCH
VERSION = CUTENSORNET_VERSION
