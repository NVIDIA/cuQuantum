# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 23.03.0 to 25.03.0. Do not modify it directly.

cimport cython  # NOQA
cimport cpython
from libcpp.vector cimport vector

from ._utils cimport (get_resource_ptr, get_nested_resource_ptr, nested_resource, nullable_unique_ptr,
                      DeviceAllocType, DeviceFreeType, cuqnt_alloc_wrapper, cuqnt_free_wrapper,
                      is_nested_sequence, logger_callback_with_data)

from enum import IntEnum as _IntEnum

import numpy as _numpy


###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """See `custatevecStatus_t`."""
    SUCCESS = CUSTATEVEC_STATUS_SUCCESS
    NOT_INITIALIZED = CUSTATEVEC_STATUS_NOT_INITIALIZED
    ALLOC_FAILED = CUSTATEVEC_STATUS_ALLOC_FAILED
    INVALID_VALUE = CUSTATEVEC_STATUS_INVALID_VALUE
    ARCH_MISMATCH = CUSTATEVEC_STATUS_ARCH_MISMATCH
    EXECUTION_FAILED = CUSTATEVEC_STATUS_EXECUTION_FAILED
    INTERNAL_ERROR = CUSTATEVEC_STATUS_INTERNAL_ERROR
    NOT_SUPPORTED = CUSTATEVEC_STATUS_NOT_SUPPORTED
    INSUFFICIENT_WORKSPACE = CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE
    SAMPLER_NOT_PREPROCESSED = CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED
    NO_DEVICE_ALLOCATOR = CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR
    DEVICE_ALLOCATOR_ERROR = CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR
    COMMUNICATOR_ERROR = CUSTATEVEC_STATUS_COMMUNICATOR_ERROR
    LOADING_LIBRARY_FAILED = CUSTATEVEC_STATUS_LOADING_LIBRARY_FAILED
    MAX_VALUE = CUSTATEVEC_STATUS_MAX_VALUE

class Pauli(_IntEnum):
    """See `custatevecPauli_t`."""
    I = CUSTATEVEC_PAULI_I
    X = CUSTATEVEC_PAULI_X
    Y = CUSTATEVEC_PAULI_Y
    Z = CUSTATEVEC_PAULI_Z

class MatrixLayout(_IntEnum):
    """See `custatevecMatrixLayout_t`."""
    COL = CUSTATEVEC_MATRIX_LAYOUT_COL
    ROW = CUSTATEVEC_MATRIX_LAYOUT_ROW

class MatrixType(_IntEnum):
    """See `custatevecMatrixType_t`."""
    GENERAL = CUSTATEVEC_MATRIX_TYPE_GENERAL
    UNITARY = CUSTATEVEC_MATRIX_TYPE_UNITARY
    HERMITIAN = CUSTATEVEC_MATRIX_TYPE_HERMITIAN

class CollapseOp(_IntEnum):
    """See `custatevecCollapseOp_t`."""
    NONE = CUSTATEVEC_COLLAPSE_NONE
    NORMALIZE_AND_ZERO = CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO

class ComputeType(_IntEnum):
    """See `custatevecComputeType_t`."""
    COMPUTE_DEFAULT = CUSTATEVEC_COMPUTE_DEFAULT
    COMPUTE_32F = CUSTATEVEC_COMPUTE_32F
    COMPUTE_64F = CUSTATEVEC_COMPUTE_64F
    COMPUTE_TF32 = CUSTATEVEC_COMPUTE_TF32

class SamplerOutput(_IntEnum):
    """See `custatevecSamplerOutput_t`."""
    RANDNUM_ORDER = CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER
    ASCENDING_ORDER = CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER

class DeviceNetworkType(_IntEnum):
    """See `custatevecDeviceNetworkType_t`."""
    SWITCH = CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH
    FULLMESH = CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH

class CommunicatorType(_IntEnum):
    """See `custatevecCommunicatorType_t`."""
    EXTERNAL = CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL
    OPENMPI = CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI
    MPICH = CUSTATEVEC_COMMUNICATOR_TYPE_MPICH

class DataTransferType(_IntEnum):
    """See `custatevecDataTransferType_t`."""
    NONE = CUSTATEVEC_DATA_TRANSFER_TYPE_NONE
    SEND = CUSTATEVEC_DATA_TRANSFER_TYPE_SEND
    RECV = CUSTATEVEC_DATA_TRANSFER_TYPE_RECV
    SEND_RECV = CUSTATEVEC_DATA_TRANSFER_TYPE_SEND_RECV

class MatrixMapType(_IntEnum):
    """See `custatevecMatrixMapType_t`."""
    BROADCAST = CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST
    MATRIX_INDEXED = CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED

class StateVectorType(_IntEnum):
    """See `custatevecStateVectorType_t`."""
    ZERO = CUSTATEVEC_STATE_VECTOR_TYPE_ZERO
    UNIFORM = CUSTATEVEC_STATE_VECTOR_TYPE_UNIFORM
    GHZ = CUSTATEVEC_STATE_VECTOR_TYPE_GHZ
    W = CUSTATEVEC_STATE_VECTOR_TYPE_W

class MathMode(_IntEnum):
    """See `custatevecMathMode_t`."""
    DEFAULT = CUSTATEVEC_MATH_MODE_DEFAULT
    ALLOW_FP32_EMULATED_BF16X9 = CUSTATEVEC_MATH_MODE_ALLOW_FP32_EMULATED_BF16X9
    DISALLOW_FP32_EMULATED_BF16X9 = CUSTATEVEC_MATH_MODE_DISALLOW_FP32_EMULATED_BF16X9

###############################################################################
# Error handling
###############################################################################

class cuStateVecError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value}): {get_error_string(status)}"
        super(cuStateVecError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuStateVecError(status)


###############################################################################
# Special dtypes
###############################################################################

_mask_dtype = _numpy.dtype(
    (_numpy.int32, CUSTATEVEC_MAX_SEGMENT_MASK_SIZE),
    align=True
)


cdef object _init_sv_swap_parameters_dtype():
    # offsetof is not exposed to Cython (it's not possible), but luckily we
    # only need to know this at runtime.
    cdef _SVSwapParameters param

    sv_swap_parameters_dtype = _numpy.dtype(
        {'names': ('swap_batch_index', 'org_sub_sv_index', 'dst_sub_sv_index',
                   'org_segment_mask_string', 'dst_segment_mask_string',
                   'segment_mask_ordering', 'segment_mask_len', 'n_segment_bits',
                   'data_transfer_type', 'transfer_size'),
         'formats': (_numpy.int32, _numpy.int32, _numpy.int32,
                     _mask_dtype, _mask_dtype,
                     _mask_dtype, _numpy.uint32, _numpy.uint32,
                     _numpy.int32, _numpy.int64),
         'offsets': (<intptr_t>&param.swapBatchIndex       - <intptr_t>&param,
                     <intptr_t>&param.orgSubSVIndex        - <intptr_t>&param,
                     <intptr_t>&param.dstSubSVIndex        - <intptr_t>&param,
                     <intptr_t>&param.orgSegmentMaskString - <intptr_t>&param,
                     <intptr_t>&param.dstSegmentMaskString - <intptr_t>&param,
                     <intptr_t>&param.segmentMaskOrdering  - <intptr_t>&param,
                     <intptr_t>&param.segmentMaskLen       - <intptr_t>&param,
                     <intptr_t>&param.nSegmentBits         - <intptr_t>&param,
                     <intptr_t>&param.dataTransferType     - <intptr_t>&param,
                     <intptr_t>&param.transferSize         - <intptr_t>&param,
                    ),
         'itemsize': sizeof(_SVSwapParameters),
        }, align=True
    )

    return sv_swap_parameters_dtype


sv_swap_parameters_dtype = _init_sv_swap_parameters_dtype()


cdef inline void _check_for_sv_swap_parameters(data) except*:
    if not isinstance(data, _numpy.ndarray) or data.size != 1:
        raise ValueError("data must be size-1 NumPy ndarray")
    if data.dtype != sv_swap_parameters_dtype:
        raise ValueError("data must be of dtype sv_swap_parameters_dtype")


cdef class SVSwapParameters:

    """A wrapper class holding a set of data transfer parameters.

    A instance of this cass can be constructed manually (either without any
    argument, or using the :meth:`from_data` factory method). The parameters
    can be retrieved/set via the instance attributes' getters/setters.

    Attributes:
        swap_batch_index (int32_t): See
            `custatevecSVSwapParameters_t::swapBatchIndex`.
        org_sub_sv_index (int32_t): See
            `custatevecSVSwapParameters_t::orgSubSVIndex`.
        dst_sub_sv_index (int32_t): See
            `custatevecSVSwapParameters_t::dstSubSVIndex`.
        org_segment_mask_string (numpy.ndarray): Should be a 1D array of dtype
            :obj:`numpy.int32` and of size ``custatevec.MAX_SEGMENT_MASK_SIZE``.
            See `custatevecSVSwapParameters_t::orgSegmentMaskString`.
        dst_segment_mask_string (numpy.ndarray): Should be a 1D array of dtype
            :obj:`numpy.int32` and of size ``custatevec.MAX_SEGMENT_MASK_SIZE``.
            See `custatevecSVSwapParameters_t::dstSegmentMaskString`.
        segment_mask_ordering (numpy.ndarray): Should be a 1D array of dtype
            :obj:`numpy.int32` and of size ``custatevec.MAX_SEGMENT_MASK_SIZE``.
            See `custatevecSVSwapParameters_t::segmentMaskOrdering`.
        segment_mask_len (uint32_t): See
            `custatevecSVSwapParameters_t::segmentMaskLen`.
        n_segment_bits (uint32_t): See
            `custatevecSVSwapParameters_t::nSegmentBits`.
        data_transfer_type (DataTransferType): See
            `custatevecSVSwapParameters_t::dataTransferType`.
        transfer_size (int64_t): See
            `custatevecSVSwapParameters_t::transferSize`.

    .. seealso:: `custatevecSVSwapParameters_t`
    """

    cdef:
        readonly object data
        """data (numpy.ndarray): The underlying storage."""

        readonly intptr_t ptr
        """ptr (intptr_t): The pointer address (as Python :class:`int`) to the
            underlying storage.
        """

    def __init__(self):
        self.data = _numpy.empty((1,), dtype=sv_swap_parameters_dtype)
        self.ptr = self.data.ctypes.data

    def __getattr__(self, attr):
        return self.data[attr]

    def __setattr__(self, attr, val):
        if attr in ('data', 'ptr'):
            # because we redirect to internal storage, we need to hardwire
            # Cython's err msg for readonly attrs
            raise AttributeError(f"attribute '{attr}' of SVSwapParameters "
                                  "objects is not writable")
        else:
            self.data[attr] = val

    def __repr__(self):
        return repr(self.data)

    def __eq__(self, other):
        return self.data == other.data

    # has to be cdef so as to access cdef attributes
    cdef inline _data_setter(self, data):
        _check_for_sv_swap_parameters(data)
        self.data = data
        self.ptr = data.ctypes.data

    @staticmethod
    def from_data(data):
        """Construct an :class:`SVSwapParameters` instance from an existing
        NumPy ndarray.

        Args:
            data (numpy.ndarray): Must be a size-1 NumPy ndarray of dtype
                :obj:`sv_swap_parameters_dtype`.
        """
        cdef SVSwapParameters param = SVSwapParameters.__new__(SVSwapParameters)
        param._data_setter(data)
        return param

    # This works, but is really not very useful. If users manage to create the
    # struct from within Python, they either do it with np.ndarray already (which
    # would be silly that we create a view over), or they are smarter / more
    # creative than we do, and in that case they just don't need this.
    # @staticmethod
    # def from_ptr(ptr):
    #     # No check could be done.
    #     cdef SVSwapParameters param = SVSwapParameters.__new__(SVSwapParameters)
    #     # create a legit view over the memory
    #     cdef object buf = PyMemoryView_FromMemory(
    #         <char*><intptr_t>ptr, sizeof(_SVSwapParameters), cpython.PyBUF_WRITE)
    #     data = _numpy.ndarray((1,), buffer=buf,
    #                           dtype=sv_swap_parameters_dtype)
    #     param.data = data
    #     param.ptr = ptr
    #     return param


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t create() except? 0:
    """This function initializes the cuStateVec library and creates a handle on the cuStateVec context. It must be called prior to any other cuStateVec API functions. If the device has unsupported compute capability, this function could return ``CUSTATEVEC_STATUS_ARCH_MISMATCH``.

    Returns:
        intptr_t: the pointer to the handle to the cuStateVec context.

    .. seealso:: `custatevecCreate`
    """
    cdef Handle handle
    with nogil:
        status = custatevecCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """This function releases resources used by the cuStateVec library.

    Args:
        handle (intptr_t): the handle to the cuStateVec context.

    .. seealso:: `custatevecDestroy`
    """
    with nogil:
        status = custatevecDestroy(<Handle>handle)
    check_status(status)


cpdef size_t get_default_workspace_size(intptr_t handle) except? 0:
    """This function returns the default workspace size defined by the cuStateVec library.

    Args:
        handle (intptr_t): the handle to the cuStateVec context.

    Returns:
        size_t: default workspace size.

    .. seealso:: `custatevecGetDefaultWorkspaceSize`
    """
    cdef size_t workspace_size_in_bytes
    with nogil:
        status = custatevecGetDefaultWorkspaceSize(<Handle>handle, &workspace_size_in_bytes)
    check_status(status)
    return workspace_size_in_bytes


cpdef set_workspace(intptr_t handle, intptr_t workspace, size_t workspace_size_in_bytes):
    """This function sets the workspace used by the cuStateVec library.

    Args:
        handle (intptr_t): the handle to the cuStateVec context.
        workspace (intptr_t): device pointer to workspace.
        workspace_size_in_bytes (size_t): workspace size.

    .. seealso:: `custatevecSetWorkspace`
    """
    with nogil:
        status = custatevecSetWorkspace(<Handle>handle, <void*>workspace, workspace_size_in_bytes)
    check_status(status)


cpdef str get_error_name(int status):
    """This function returns the name string for the input error code. If the error code is not recognized, "unrecognized error code" is returned.

    Args:
        status (Status): Error code to convert to string.

    .. seealso:: `custatevecGetErrorName`
    """
    cdef bytes _output_
    _output_ = custatevecGetErrorName(<_Status>status)
    return _output_.decode()


cpdef str get_error_string(int status):
    """This function returns the description string for an error code. If the error code is not recognized, "unrecognized error code" is returned.

    Args:
        status (Status): Error code to convert to string.

    .. seealso:: `custatevecGetErrorString`
    """
    cdef bytes _output_
    _output_ = custatevecGetErrorString(<_Status>status)
    return _output_.decode()


cpdef int32_t get_property(int type) except? -1:
    """This function returns the version information of the cuStateVec library.

    Args:
        type (int): requested property (``MAJOR_VERSION``, ``MINOR_VERSION``, or ``PATCH_LEVEL``).

    Returns:
        int32_t: value of the requested property.

    .. seealso:: `custatevecGetProperty`
    """
    cdef int32_t value
    with nogil:
        status = custatevecGetProperty(<LibraryPropertyType>type, &value)
    check_status(status)
    return value


cpdef size_t get_version() except? 0:
    """This function returns the version information of the cuStateVec library.

    .. seealso:: `custatevecGetVersion`
    """
    return custatevecGetVersion()


cpdef set_stream(intptr_t handle, intptr_t stream_id):
    """This function sets the stream to be used by the cuStateVec library to execute its routine.

    Args:
        handle (intptr_t): the handle to the cuStateVec context.
        stream_id (intptr_t): the stream to be used by the library.

    .. seealso:: `custatevecSetStream`
    """
    with nogil:
        status = custatevecSetStream(<Handle>handle, <Stream>stream_id)
    check_status(status)


cpdef intptr_t get_stream(intptr_t handle) except? 0:
    """This function gets the cuStateVec library stream used to execute all calls from the cuStateVec library functions.

    Args:
        handle (intptr_t): the handle to the cuStateVec context.

    Returns:
        intptr_t: the stream to be used by the library.

    .. seealso:: `custatevecGetStream`
    """
    cdef Stream stream_id
    with nogil:
        status = custatevecGetStream(<Handle>handle, &stream_id)
    check_status(status)
    return <intptr_t>stream_id


cpdef logger_open_file(log_file):
    """Experimental: This function opens a logging output file in the given path.

    Args:
        log_file (str): Path of the logging output file.

    .. seealso:: `custatevecLoggerOpenFile`
    """
    if not isinstance(log_file, str):
        raise TypeError("log_file must be a Python str")
    cdef bytes _temp_log_file_ = (<str>log_file).encode()
    cdef char* _log_file_ = _temp_log_file_
    with nogil:
        status = custatevecLoggerOpenFile(<const char*>_log_file_)
    check_status(status)


cpdef logger_set_level(int32_t level):
    """Experimental: This function sets the value of the logging level.

    Args:
        level (int32_t): Value of the logging level.

    .. seealso:: `custatevecLoggerSetLevel`
    """
    with nogil:
        status = custatevecLoggerSetLevel(level)
    check_status(status)


cpdef logger_set_mask(int32_t mask):
    """Experimental: This function sets the value of the logging mask. Masks are defined as a combination of the following masks:.

    Args:
        mask (int32_t): Value of the logging mask.

    .. seealso:: `custatevecLoggerSetMask`
    """
    with nogil:
        status = custatevecLoggerSetMask(mask)
    check_status(status)


cpdef logger_force_disable():
    """Experimental: This function disables logging for the entire run.

    .. seealso:: `custatevecLoggerForceDisable`
    """
    with nogil:
        status = custatevecLoggerForceDisable()
    check_status(status)


cpdef abs2sum_array(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t abs2sum, bit_ordering, uint32_t bit_ordering_len, mask_bit_string, mask_ordering, uint32_t mask_len):
    """Calculate abs2sum array for a given set of index bits.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): data type of state vector.
        n_index_bits (uint32_t): the number of index bits.
        abs2sum (intptr_t): pointer to a host or device array of sums of squared absolute values.
        bit_ordering (object): pointer to a host array of index bit ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        bit_ordering_len (uint32_t): the length of bit_ordering.
        mask_bit_string (object): pointer to a host array for a bit string to specify mask. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mask_ordering (object): pointer to a host array for the mask ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mask_len (uint32_t): the length of mask.

    .. seealso:: `custatevecAbs2SumArray`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _bit_ordering_
    get_resource_ptr[int32_t](_bit_ordering_, bit_ordering, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_bit_string_
    get_resource_ptr[int32_t](_mask_bit_string_, mask_bit_string, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_ordering_
    get_resource_ptr[int32_t](_mask_ordering_, mask_ordering, <int32_t*>NULL)
    with nogil:
        status = custatevecAbs2SumArray(<Handle>handle, <const void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <double*>abs2sum, <const int32_t*>(_bit_ordering_.data()), <const uint32_t>bit_ordering_len, <const int32_t*>(_mask_bit_string_.data()), <const int32_t*>(_mask_ordering_.data()), <const uint32_t>mask_len)
    check_status(status)


cpdef collapse_on_z_basis(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, int32_t parity, basis_bits, uint32_t n_basis_bits, double norm):
    """Collapse state vector on a given Z product basis.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): data type of state vector.
        n_index_bits (uint32_t): the number of index bits.
        parity (int32_t): parity, 0 or 1.
        basis_bits (object): pointer to a host array of Z-basis index bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_basis_bits (uint32_t): the number of Z basis bits.
        norm (double): normalization factor.

    .. seealso:: `custatevecCollapseOnZBasis`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _basis_bits_
    get_resource_ptr[int32_t](_basis_bits_, basis_bits, <int32_t*>NULL)
    with nogil:
        status = custatevecCollapseOnZBasis(<Handle>handle, <void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const int32_t>parity, <const int32_t*>(_basis_bits_.data()), <const uint32_t>n_basis_bits, norm)
    check_status(status)


cpdef collapse_by_bit_string(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, bit_string, bit_ordering, uint32_t bit_string_len, double norm):
    """Collapse state vector to the state specified by a given bit string.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): data type of state vector.
        n_index_bits (uint32_t): the number of index bits.
        bit_string (object): pointer to a host array of bit string. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        bit_ordering (object): pointer to a host array of bit string ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        bit_string_len (uint32_t): length of bit string.
        norm (double): normalization constant.

    .. seealso:: `custatevecCollapseByBitString`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _bit_string_
    get_resource_ptr[int32_t](_bit_string_, bit_string, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _bit_ordering_
    get_resource_ptr[int32_t](_bit_ordering_, bit_ordering, <int32_t*>NULL)
    with nogil:
        status = custatevecCollapseByBitString(<Handle>handle, <void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const int32_t*>(_bit_string_.data()), <const int32_t*>(_bit_ordering_.data()), <const uint32_t>bit_string_len, norm)
    check_status(status)


cpdef int32_t measure_on_z_basis(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, basis_bits, uint32_t n_basis_bits, double randnum, int collapse) except? -1:
    """Measurement on a given Z-product basis.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): data type of state vector.
        n_index_bits (uint32_t): the number of index bits.
        basis_bits (object): pointer to a host array of Z basis bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_basis_bits (uint32_t): the number of Z basis bits.
        randnum (double): random number, [0, 1).
        collapse (CollapseOp): Collapse operation.

    Returns:
        int32_t: parity, 0 or 1.

    .. seealso:: `custatevecMeasureOnZBasis`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _basis_bits_
    get_resource_ptr[int32_t](_basis_bits_, basis_bits, <int32_t*>NULL)
    cdef int32_t parity
    with nogil:
        status = custatevecMeasureOnZBasis(<Handle>handle, <void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, &parity, <const int32_t*>(_basis_bits_.data()), <const uint32_t>n_basis_bits, <const double>randnum, <_CollapseOp>collapse)
    check_status(status)
    return parity


cpdef batch_measure(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t bit_string, bit_ordering, uint32_t bit_string_len, double randnum, int collapse):
    """Batched single qubit measurement.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): data type of the state vector.
        n_index_bits (uint32_t): the number of index bits.
        bit_string (intptr_t): pointer to a host array of measured bit string.
        bit_ordering (object): pointer to a host array of bit string ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        bit_string_len (uint32_t): length of bit_string.
        randnum (double): random number, [0, 1).
        collapse (CollapseOp): Collapse operation.

    .. seealso:: `custatevecBatchMeasure`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _bit_ordering_
    get_resource_ptr[int32_t](_bit_ordering_, bit_ordering, <int32_t*>NULL)
    with nogil:
        status = custatevecBatchMeasure(<Handle>handle, <void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <int32_t*>bit_string, <const int32_t*>(_bit_ordering_.data()), <const uint32_t>bit_string_len, <const double>randnum, <_CollapseOp>collapse)
    check_status(status)


cpdef batch_measure_with_offset(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t bit_string, bit_ordering, uint32_t bit_string_len, double randnum, int collapse, double offset, double abs2sum):
    """Batched single qubit measurement for partial vector.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): partial state vector.
        sv_data_type (int): data type of the state vector.
        n_index_bits (uint32_t): the number of index bits.
        bit_string (intptr_t): pointer to a host array of measured bit string.
        bit_ordering (object): pointer to a host array of bit string ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        bit_string_len (uint32_t): length of bit_string.
        randnum (double): random number, [0, 1).
        collapse (CollapseOp): Collapse operation.
        offset (double): partial sum of squared absolute values.
        abs2sum (double): sum of squared absolute values for the entire state vector.

    .. seealso:: `custatevecBatchMeasureWithOffset`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _bit_ordering_
    get_resource_ptr[int32_t](_bit_ordering_, bit_ordering, <int32_t*>NULL)
    with nogil:
        status = custatevecBatchMeasureWithOffset(<Handle>handle, <void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <int32_t*>bit_string, <const int32_t*>(_bit_ordering_.data()), <const uint32_t>bit_string_len, <const double>randnum, <_CollapseOp>collapse, <const double>offset, <const double>abs2sum)
    check_status(status)


cpdef apply_pauli_rotation(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, double theta, paulis, targets, uint32_t n_targets, controls, control_bit_values, uint32_t n_controls):
    """Apply the exponential of a multi-qubit Pauli operator.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): data type of the state vector.
        n_index_bits (uint32_t): the number of bits in the state vector index.
        theta (double): theta.
        paulis (object): host pointer to custatevecPauli_t array. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``_Pauli``.

        targets (object): pointer to a host array of target bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_targets (uint32_t): the number of target bits.
        controls (object): pointer to a host array of control bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        control_bit_values (object): pointer to a host array of control bit values. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_controls (uint32_t): the number of control bits.

    .. seealso:: `custatevecApplyPauliRotation`
    """
    cdef nullable_unique_ptr[ vector[int] ] _paulis_
    get_resource_ptr[int](_paulis_, paulis, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _targets_
    get_resource_ptr[int32_t](_targets_, targets, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _controls_
    get_resource_ptr[int32_t](_controls_, controls, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _control_bit_values_
    get_resource_ptr[int32_t](_control_bit_values_, control_bit_values, <int32_t*>NULL)
    with nogil:
        status = custatevecApplyPauliRotation(<Handle>handle, <void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, theta, <const _Pauli*>(_paulis_.data()), <const int32_t*>(_targets_.data()), <const uint32_t>n_targets, <const int32_t*>(_controls_.data()), <const int32_t*>(_control_bit_values_.data()), <const uint32_t>n_controls)
    check_status(status)


cpdef size_t apply_matrix_get_workspace_size(intptr_t handle, int sv_data_type, uint32_t n_index_bits, intptr_t matrix, int matrix_data_type, int layout, int32_t adjoint, uint32_t n_targets, uint32_t n_controls, int compute_type) except? 0:
    """This function gets the required workspace size for :func:`apply_matrix`.

    Args:
        handle (intptr_t): the handle to the cuStateVec context.
        sv_data_type (int): Data type of the state vector.
        n_index_bits (uint32_t): the number of index bits of the state vector.
        matrix (intptr_t): host or device pointer to a matrix.
        matrix_data_type (int): data type of matrix.
        layout (MatrixLayout): enumerator specifying the memory layout of matrix.
        adjoint (int32_t): apply adjoint of matrix.
        n_targets (uint32_t): the number of target bits.
        n_controls (uint32_t): the number of control bits.
        compute_type (ComputeType): compute_type of matrix multiplication.

    Returns:
        size_t: workspace size.

    .. seealso:: `custatevecApplyMatrixGetWorkspaceSize`
    """
    cdef size_t extra_workspace_size_in_bytes
    with nogil:
        status = custatevecApplyMatrixGetWorkspaceSize(<Handle>handle, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const void*>matrix, <DataType>matrix_data_type, <_MatrixLayout>layout, <const int32_t>adjoint, <const uint32_t>n_targets, <const uint32_t>n_controls, <_ComputeType>compute_type, &extra_workspace_size_in_bytes)
    check_status(status)
    return extra_workspace_size_in_bytes


cpdef apply_matrix(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t matrix, int matrix_data_type, int layout, int32_t adjoint, targets, uint32_t n_targets, controls, control_bit_values, uint32_t n_controls, int compute_type, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes):
    """Apply gate matrix.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): data type of the state vector.
        n_index_bits (uint32_t): the number of index bits of the state vector.
        matrix (intptr_t): host or device pointer to a square matrix.
        matrix_data_type (int): data type of matrix.
        layout (MatrixLayout): enumerator specifying the memory layout of matrix.
        adjoint (int32_t): apply adjoint of matrix.
        targets (object): pointer to a host array of target bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_targets (uint32_t): the number of target bits.
        controls (object): pointer to a host array of control bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        control_bit_values (object): pointer to a host array of control bit values. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_controls (uint32_t): the number of control bits.
        compute_type (ComputeType): compute_type of matrix multiplication.
        extra_workspace (intptr_t): extra workspace.
        extra_workspace_size_in_bytes (size_t): extra workspace size.

    .. seealso:: `custatevecApplyMatrix`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _targets_
    get_resource_ptr[int32_t](_targets_, targets, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _controls_
    get_resource_ptr[int32_t](_controls_, controls, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _control_bit_values_
    get_resource_ptr[int32_t](_control_bit_values_, control_bit_values, <int32_t*>NULL)
    with nogil:
        status = custatevecApplyMatrix(<Handle>handle, <void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const void*>matrix, <DataType>matrix_data_type, <_MatrixLayout>layout, <const int32_t>adjoint, <const int32_t*>(_targets_.data()), <const uint32_t>n_targets, <const int32_t*>(_controls_.data()), <const int32_t*>(_control_bit_values_.data()), <const uint32_t>n_controls, <_ComputeType>compute_type, <void*>extra_workspace, extra_workspace_size_in_bytes)
    check_status(status)


cpdef size_t compute_expectation_get_workspace_size(intptr_t handle, int sv_data_type, uint32_t n_index_bits, intptr_t matrix, int matrix_data_type, int layout, uint32_t n_basis_bits, int compute_type) except? 0:
    """This function gets the required workspace size for :func:`compute_expectation`.

    Args:
        handle (intptr_t): the handle to the cuStateVec context.
        sv_data_type (int): Data type of the state vector.
        n_index_bits (uint32_t): the number of index bits of the state vector.
        matrix (intptr_t): host or device pointer to a matrix.
        matrix_data_type (int): data type of matrix.
        layout (MatrixLayout): enumerator specifying the memory layout of matrix.
        n_basis_bits (uint32_t): the number of target bits.
        compute_type (ComputeType): compute_type of matrix multiplication.

    Returns:
        size_t: size of the extra workspace.

    .. seealso:: `custatevecComputeExpectationGetWorkspaceSize`
    """
    cdef size_t extra_workspace_size_in_bytes
    with nogil:
        status = custatevecComputeExpectationGetWorkspaceSize(<Handle>handle, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const void*>matrix, <DataType>matrix_data_type, <_MatrixLayout>layout, <const uint32_t>n_basis_bits, <_ComputeType>compute_type, &extra_workspace_size_in_bytes)
    check_status(status)
    return extra_workspace_size_in_bytes


cpdef double compute_expectation(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t expectation_value, int expectation_data_type, intptr_t matrix, int matrix_data_type, int layout, basis_bits, uint32_t n_basis_bits, int compute_type, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes) except? 0:
    """Compute expectation of matrix observable.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): data type of the state vector.
        n_index_bits (uint32_t): the number of index bits of the state vector.
        expectation_value (intptr_t): host pointer to a variable to store an expectation value.
        expectation_data_type (int): data type of expect.
        matrix (intptr_t): observable as matrix.
        matrix_data_type (int): data type of matrix.
        layout (MatrixLayout): matrix memory layout.
        basis_bits (object): pointer to a host array of basis index bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_basis_bits (uint32_t): the number of basis bits.
        compute_type (ComputeType): compute_type of matrix multiplication.
        extra_workspace (intptr_t): pointer to an extra workspace.
        extra_workspace_size_in_bytes (size_t): the size of extra workspace.

    Returns:
        double: result of matrix type test.

    .. seealso:: `custatevecComputeExpectation`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _basis_bits_
    get_resource_ptr[int32_t](_basis_bits_, basis_bits, <int32_t*>NULL)
    cdef double residual_norm
    with nogil:
        status = custatevecComputeExpectation(<Handle>handle, <const void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <void*>expectation_value, <DataType>expectation_data_type, &residual_norm, <const void*>matrix, <DataType>matrix_data_type, <_MatrixLayout>layout, <const int32_t*>(_basis_bits_.data()), <const uint32_t>n_basis_bits, <_ComputeType>compute_type, <void*>extra_workspace, extra_workspace_size_in_bytes)
    check_status(status)
    return residual_norm


cpdef tuple sampler_create(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_max_shots):
    """Create sampler descriptor.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): pointer to state vector.
        sv_data_type (int): data type of state vector.
        n_index_bits (uint32_t): the number of index bits of the state vector.
        n_max_shots (uint32_t): the max number of shots used for this sampler context.

    Returns:
        A 2-tuple containing:

        - intptr_t: pointer to a new sampler descriptor.
        - size_t: workspace size.

    .. seealso:: `custatevecSamplerCreate`
    """
    cdef SamplerDescriptor sampler
    cdef size_t extra_workspace_size_in_bytes
    with nogil:
        status = custatevecSamplerCreate(<Handle>handle, <const void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, &sampler, n_max_shots, &extra_workspace_size_in_bytes)
    check_status(status)
    return (<intptr_t>sampler, extra_workspace_size_in_bytes)


cpdef sampler_destroy(intptr_t sampler):
    """This function releases resources used by the sampler.

    Args:
        sampler (intptr_t): the sampler descriptor.

    .. seealso:: `custatevecSamplerDestroy`
    """
    with nogil:
        status = custatevecSamplerDestroy(<SamplerDescriptor>sampler)
    check_status(status)


cpdef sampler_preprocess(intptr_t handle, intptr_t sampler, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes):
    """Preprocess the state vector for preparation of sampling.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sampler (intptr_t): the sampler descriptor.
        extra_workspace (intptr_t): extra workspace.
        extra_workspace_size_in_bytes (size_t): size of the extra workspace.

    .. seealso:: `custatevecSamplerPreprocess`
    """
    with nogil:
        status = custatevecSamplerPreprocess(<Handle>handle, <SamplerDescriptor>sampler, <void*>extra_workspace, <const size_t>extra_workspace_size_in_bytes)
    check_status(status)


cpdef double sampler_get_squared_norm(intptr_t handle, intptr_t sampler) except? -1:
    """Get the squared norm of the state vector.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sampler (intptr_t): the sampler descriptor.

    Returns:
        double: the norm of the state vector.

    .. seealso:: `custatevecSamplerGetSquaredNorm`
    """
    cdef double norm
    with nogil:
        status = custatevecSamplerGetSquaredNorm(<Handle>handle, <SamplerDescriptor>sampler, &norm)
    check_status(status)
    return norm


cpdef sampler_apply_sub_sv_offset(intptr_t handle, intptr_t sampler, int32_t sub_sv_ord, uint32_t n_sub_svs, double offset, double norm):
    """Apply the partial norm and norm to the state vector to the sample descriptor.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sampler (intptr_t): the sampler descriptor.
        sub_sv_ord (int32_t): sub state vector ordinal.
        n_sub_svs (uint32_t): the number of sub state vectors.
        offset (double): cumulative sum offset for the sub state vector.
        norm (double): norm for all sub vectors.

    .. seealso:: `custatevecSamplerApplySubSVOffset`
    """
    with nogil:
        status = custatevecSamplerApplySubSVOffset(<Handle>handle, <SamplerDescriptor>sampler, sub_sv_ord, n_sub_svs, offset, norm)
    check_status(status)


cpdef sampler_sample(intptr_t handle, intptr_t sampler, intptr_t bit_strings, bit_ordering, uint32_t bit_string_len, randnums, uint32_t n_shots, int output):
    """Sample bit strings from the state vector.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sampler (intptr_t): the sampler descriptor.
        bit_strings (intptr_t): pointer to a host array to store sampled bit strings.
        bit_ordering (object): pointer to a host array of bit ordering for sampling. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        bit_string_len (uint32_t): the number of bits in bit_ordering.
        randnums (object): pointer to an array of random numbers. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``float``.

        n_shots (uint32_t): the number of shots.
        output (SamplerOutput): the order of sampled bit strings.

    .. seealso:: `custatevecSamplerSample`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _bit_ordering_
    get_resource_ptr[int32_t](_bit_ordering_, bit_ordering, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[double] ] _randnums_
    get_resource_ptr[double](_randnums_, randnums, <double*>NULL)
    with nogil:
        status = custatevecSamplerSample(<Handle>handle, <SamplerDescriptor>sampler, <custatevecIndex_t*>bit_strings, <const int32_t*>(_bit_ordering_.data()), <const uint32_t>bit_string_len, <const double*>(_randnums_.data()), <const uint32_t>n_shots, <_SamplerOutput>output)
    check_status(status)


cpdef size_t apply_generalized_permutation_matrix_get_workspace_size(intptr_t handle, int sv_data_type, uint32_t n_index_bits, permutation, intptr_t diagonals, int diagonals_data_type, targets, uint32_t n_targets, uint32_t n_controls) except? 0:
    """Get the extra workspace size required by :func:`apply_generalized_permutation_matrix`.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv_data_type (int): data type of the state vector.
        n_index_bits (uint32_t): the number of index bits of the state vector.
        permutation (object): host or device pointer to a permutation table. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``custatevecIndex_t``.

        diagonals (intptr_t): host or device pointer to diagonal elements.
        diagonals_data_type (int): data type of diagonals.
        targets (object): pointer to a host array of target bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_targets (uint32_t): the number of target bits.
        n_controls (uint32_t): the number of control bits.

    Returns:
        size_t: extra workspace size.

    .. seealso:: `custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _permutation_
    get_resource_ptr[int64_t](_permutation_, permutation, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _targets_
    get_resource_ptr[int32_t](_targets_, targets, <int32_t*>NULL)
    cdef size_t extra_workspace_size_in_bytes
    with nogil:
        status = custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(<Handle>handle, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const custatevecIndex_t*>(_permutation_.data()), <const void*>diagonals, <DataType>diagonals_data_type, <const int32_t*>(_targets_.data()), <const uint32_t>n_targets, <const uint32_t>n_controls, &extra_workspace_size_in_bytes)
    check_status(status)
    return extra_workspace_size_in_bytes


cpdef apply_generalized_permutation_matrix(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, permutation, intptr_t diagonals, int diagonals_data_type, int32_t adjoint, targets, uint32_t n_targets, controls, control_bit_values, uint32_t n_controls, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes):
    """Apply generalized permutation matrix.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): data type of the state vector.
        n_index_bits (uint32_t): the number of index bits of the state vector.
        permutation (object): host or device pointer to a permutation table. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``custatevecIndex_t``.

        diagonals (intptr_t): host or device pointer to diagonal elements.
        diagonals_data_type (int): data type of diagonals.
        adjoint (int32_t): apply adjoint of generalized permutation matrix.
        targets (object): pointer to a host array of target bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_targets (uint32_t): the number of target bits.
        controls (object): pointer to a host array of control bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        control_bit_values (object): pointer to a host array of control bit values. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_controls (uint32_t): the number of control bits.
        extra_workspace (intptr_t): extra workspace.
        extra_workspace_size_in_bytes (size_t): extra workspace size.

    .. seealso:: `custatevecApplyGeneralizedPermutationMatrix`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _permutation_
    get_resource_ptr[int64_t](_permutation_, permutation, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _targets_
    get_resource_ptr[int32_t](_targets_, targets, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _controls_
    get_resource_ptr[int32_t](_controls_, controls, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _control_bit_values_
    get_resource_ptr[int32_t](_control_bit_values_, control_bit_values, <int32_t*>NULL)
    with nogil:
        status = custatevecApplyGeneralizedPermutationMatrix(<Handle>handle, <void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <custatevecIndex_t*>(_permutation_.data()), <const void*>diagonals, <DataType>diagonals_data_type, <const int32_t>adjoint, <const int32_t*>(_targets_.data()), <const uint32_t>n_targets, <const int32_t*>(_controls_.data()), <const int32_t*>(_control_bit_values_.data()), <const uint32_t>n_controls, <void*>extra_workspace, extra_workspace_size_in_bytes)
    check_status(status)


cpdef compute_expectations_on_pauli_basis(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, intptr_t expectation_values, pauli_operators_array, uint32_t n_pauli_operator_arrays, basis_bits_array, n_basis_bits_array):
    """Calculate expectation values for a batch of (multi-qubit) Pauli operators.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): data type of the state vector.
        n_index_bits (uint32_t): the number of index bits of the state vector.
        expectation_values (intptr_t): pointer to a host array to store expectation values.
        pauli_operators_array (object): pointer to a host array of Pauli operator arrays. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of '_Pauli', or
            - a nested Python sequence of ``_Pauli``.

        n_pauli_operator_arrays (uint32_t): the number of Pauli operator arrays.
        basis_bits_array (object): host array of basis bit arrays. It can be:

            - an :class:`int` as the pointer address to the nested sequence, or
            - a Python sequence of :class:`int`\s, each of which is a pointer address
              to a valid sequence of 'int32_t', or
            - a nested Python sequence of ``int32_t``.

        n_basis_bits_array (object): host array of the number of basis bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``uint32_t``.


    .. seealso:: `custatevecComputeExpectationsOnPauliBasis`
    """
    cdef nested_resource[ int ] _pauli_operators_array_
    get_nested_resource_ptr[int](_pauli_operators_array_, pauli_operators_array, <int*>NULL)
    cdef nested_resource[ int32_t ] _basis_bits_array_
    get_nested_resource_ptr[int32_t](_basis_bits_array_, basis_bits_array, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[uint32_t] ] _n_basis_bits_array_
    get_resource_ptr[uint32_t](_n_basis_bits_array_, n_basis_bits_array, <uint32_t*>NULL)
    with nogil:
        status = custatevecComputeExpectationsOnPauliBasis(<Handle>handle, <const void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <double*>expectation_values, <const _Pauli**>(_pauli_operators_array_.ptrs.data()), <const uint32_t>n_pauli_operator_arrays, <const int32_t**>(_basis_bits_array_.ptrs.data()), <const uint32_t*>(_n_basis_bits_array_.data()))
    check_status(status)


cpdef tuple accessor_create(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, bit_ordering, uint32_t bit_ordering_len, mask_bit_string, mask_ordering, uint32_t mask_len):
    """Create accessor to copy elements between the state vector and an external buffer.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): Data type of state vector.
        n_index_bits (uint32_t): the number of index bits of state vector.
        bit_ordering (object): pointer to a host array to specify the basis bits of the external buffer. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        bit_ordering_len (uint32_t): the length of bit_ordering.
        mask_bit_string (object): pointer to a host array to specify the mask values to limit access. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mask_ordering (object): pointer to a host array for the mask ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mask_len (uint32_t): the length of mask.

    Returns:
        A 2-tuple containing:

        - intptr_t: pointer to an accessor descriptor.
        - size_t: the required size of extra workspace.

    .. seealso:: `custatevecAccessorCreate`
    """
    cdef AccessorDescriptor accessor
    cdef size_t extra_workspace_size_in_bytes
    cdef nullable_unique_ptr[ vector[int32_t] ] _bit_ordering_
    get_resource_ptr[int32_t](_bit_ordering_, bit_ordering, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_bit_string_
    get_resource_ptr[int32_t](_mask_bit_string_, mask_bit_string, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_ordering_
    get_resource_ptr[int32_t](_mask_ordering_, mask_ordering, <int32_t*>NULL)
    with nogil:
        status = custatevecAccessorCreate(<Handle>handle, <void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, &accessor, <const int32_t*>(_bit_ordering_.data()), <const uint32_t>bit_ordering_len, <const int32_t*>(_mask_bit_string_.data()), <const int32_t*>(_mask_ordering_.data()), <const uint32_t>mask_len, &extra_workspace_size_in_bytes)
    check_status(status)
    return (<intptr_t>accessor, extra_workspace_size_in_bytes)


cpdef tuple accessor_create_view(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, bit_ordering, uint32_t bit_ordering_len, mask_bit_string, mask_ordering, uint32_t mask_len):
    """Create accessor for the constant state vector.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): Data type of state vector.
        n_index_bits (uint32_t): the number of index bits of state vector.
        bit_ordering (object): pointer to a host array to specify the basis bits of the external buffer. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        bit_ordering_len (uint32_t): the length of bit_ordering.
        mask_bit_string (object): pointer to a host array to specify the mask values to limit access. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mask_ordering (object): pointer to a host array for the mask ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mask_len (uint32_t): the length of mask.

    Returns:
        A 2-tuple containing:

        - intptr_t: pointer to an accessor descriptor.
        - size_t: the required size of extra workspace.

    .. seealso:: `custatevecAccessorCreateView`
    """
    cdef AccessorDescriptor accessor
    cdef size_t extra_workspace_size_in_bytes
    cdef nullable_unique_ptr[ vector[int32_t] ] _bit_ordering_
    get_resource_ptr[int32_t](_bit_ordering_, bit_ordering, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_bit_string_
    get_resource_ptr[int32_t](_mask_bit_string_, mask_bit_string, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_ordering_
    get_resource_ptr[int32_t](_mask_ordering_, mask_ordering, <int32_t*>NULL)
    with nogil:
        status = custatevecAccessorCreateView(<Handle>handle, <const void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, &accessor, <const int32_t*>(_bit_ordering_.data()), <const uint32_t>bit_ordering_len, <const int32_t*>(_mask_bit_string_.data()), <const int32_t*>(_mask_ordering_.data()), <const uint32_t>mask_len, &extra_workspace_size_in_bytes)
    check_status(status)
    return (<intptr_t>accessor, extra_workspace_size_in_bytes)


cpdef accessor_destroy(intptr_t accessor):
    """This function releases resources used by the accessor.

    Args:
        accessor (intptr_t): the accessor descriptor.

    .. seealso:: `custatevecAccessorDestroy`
    """
    with nogil:
        status = custatevecAccessorDestroy(<AccessorDescriptor>accessor)
    check_status(status)


cpdef accessor_set_extra_workspace(intptr_t handle, intptr_t accessor, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes):
    """Set the external workspace to the accessor.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        accessor (intptr_t): the accessor descriptor.
        extra_workspace (intptr_t): extra workspace.
        extra_workspace_size_in_bytes (size_t): extra workspace size.

    .. seealso:: `custatevecAccessorSetExtraWorkspace`
    """
    with nogil:
        status = custatevecAccessorSetExtraWorkspace(<Handle>handle, <AccessorDescriptor>accessor, <void*>extra_workspace, extra_workspace_size_in_bytes)
    check_status(status)


cpdef accessor_get(intptr_t handle, intptr_t accessor, intptr_t external_buffer, int64_t begin, int64_t end):
    """Copy state vector elements to an external buffer.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        accessor (intptr_t): the accessor descriptor.
        external_buffer (intptr_t): pointer to a host or device buffer to receive copied elements.
        begin (int64_t): index in the permuted bit ordering for the first elements being copied to the state vector.
        end (int64_t): index in the permuted bit ordering for the last elements being copied to the state vector (non-inclusive).

    .. seealso:: `custatevecAccessorGet`
    """
    with nogil:
        status = custatevecAccessorGet(<Handle>handle, <AccessorDescriptor>accessor, <void*>external_buffer, <const custatevecIndex_t>begin, <const custatevecIndex_t>end)
    check_status(status)


cpdef accessor_set(intptr_t handle, intptr_t accessor, intptr_t external_buffer, int64_t begin, int64_t end):
    """Set state vector elements from an external buffer.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        accessor (intptr_t): the accessor descriptor.
        external_buffer (intptr_t): pointer to a host or device buffer of complex values being copied to the state vector.
        begin (int64_t): index in the permuted bit ordering for the first elements being copied from the state vector.
        end (int64_t): index in the permuted bit ordering for the last elements being copied from the state vector (non-inclusive).

    .. seealso:: `custatevecAccessorSet`
    """
    with nogil:
        status = custatevecAccessorSet(<Handle>handle, <AccessorDescriptor>accessor, <const void*>external_buffer, <const custatevecIndex_t>begin, <const custatevecIndex_t>end)
    check_status(status)


cpdef size_t test_matrix_type_get_workspace_size(intptr_t handle, int matrix_type, intptr_t matrix, int matrix_data_type, int layout, uint32_t n_targets, int32_t adjoint, int compute_type) except? 0:
    """Get extra workspace size for :func:`test_matrix_type`.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        matrix_type (MatrixType): matrix type.
        matrix (intptr_t): host or device pointer to a matrix.
        matrix_data_type (int): data type of matrix.
        layout (MatrixLayout): enumerator specifying the memory layout of matrix.
        n_targets (uint32_t): the number of target bits, up to 15.
        adjoint (int32_t): flag to control whether the adjoint of matrix is tested.
        compute_type (ComputeType): compute type.

    Returns:
        size_t: workspace size.

    .. seealso:: `custatevecTestMatrixTypeGetWorkspaceSize`
    """
    cdef size_t extra_workspace_size_in_bytes
    with nogil:
        status = custatevecTestMatrixTypeGetWorkspaceSize(<Handle>handle, <_MatrixType>matrix_type, <const void*>matrix, <DataType>matrix_data_type, <_MatrixLayout>layout, <const uint32_t>n_targets, <const int32_t>adjoint, <_ComputeType>compute_type, &extra_workspace_size_in_bytes)
    check_status(status)
    return extra_workspace_size_in_bytes


cpdef double test_matrix_type(intptr_t handle, int matrix_type, intptr_t matrix, int matrix_data_type, int layout, uint32_t n_targets, int32_t adjoint, int compute_type, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes) except? -1:
    """Test the deviation of a given matrix from a Hermitian (or Unitary) matrix.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        matrix_type (MatrixType): matrix type.
        matrix (intptr_t): host or device pointer to a matrix.
        matrix_data_type (int): data type of matrix.
        layout (MatrixLayout): enumerator specifying the memory layout of matrix.
        n_targets (uint32_t): the number of target bits, up to 15.
        adjoint (int32_t): flag to control whether the adjoint of matrix is tested.
        compute_type (ComputeType): compute type.
        extra_workspace (intptr_t): extra workspace.
        extra_workspace_size_in_bytes (size_t): extra workspace size.

    Returns:
        double: host pointer, to store the deviation from certain matrix type.

    .. seealso:: `custatevecTestMatrixType`
    """
    cdef double residual_norm
    with nogil:
        status = custatevecTestMatrixType(<Handle>handle, &residual_norm, <_MatrixType>matrix_type, <const void*>matrix, <DataType>matrix_data_type, <_MatrixLayout>layout, <const uint32_t>n_targets, <const int32_t>adjoint, <_ComputeType>compute_type, <void*>extra_workspace, extra_workspace_size_in_bytes)
    check_status(status)
    return residual_norm


cpdef intptr_t communicator_create(intptr_t handle, int communicator_type, soname) except? 0:
    """Create communicator.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        communicator_type (CommunicatorType): the communicator type.
        soname (str): the shared object name.

    Returns:
        intptr_t: a pointer to the communicator.

    .. seealso:: `custatevecCommunicatorCreate`
    """
    if not isinstance(soname, str):
        raise TypeError("soname must be a Python str")
    cdef bytes _temp_soname_ = (<str>soname).encode()
    cdef char* _soname_ = _temp_soname_
    cdef CommunicatorDescriptor communicator
    with nogil:
        status = custatevecCommunicatorCreate(<Handle>handle, &communicator, <_CommunicatorType>communicator_type, <const char*>_soname_)
    check_status(status)
    return <intptr_t>communicator


cpdef communicator_destroy(intptr_t handle, intptr_t communicator):
    """This function releases communicator.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        communicator (intptr_t): the communicator descriptor.

    .. seealso:: `custatevecCommunicatorDestroy`
    """
    with nogil:
        status = custatevecCommunicatorDestroy(<Handle>handle, <CommunicatorDescriptor>communicator)
    check_status(status)


cpdef intptr_t dist_index_bit_swap_scheduler_create(intptr_t handle, uint32_t n_global_index_bits, uint32_t n_local_index_bits) except? 0:
    """Create distributed index bit swap scheduler.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        n_global_index_bits (uint32_t): the number of global index bits.
        n_local_index_bits (uint32_t): the number of local index bits.

    Returns:
        intptr_t: a pointer to a batch swap scheduler.

    .. seealso:: `custatevecDistIndexBitSwapSchedulerCreate`
    """
    cdef DistIndexBitSwapSchedulerDescriptor scheduler
    with nogil:
        status = custatevecDistIndexBitSwapSchedulerCreate(<Handle>handle, &scheduler, <const uint32_t>n_global_index_bits, <const uint32_t>n_local_index_bits)
    check_status(status)
    return <intptr_t>scheduler


cpdef dist_index_bit_swap_scheduler_destroy(intptr_t handle, intptr_t scheduler):
    """This function releases distributed index bit swap scheduler.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        scheduler (intptr_t): a pointer to the batch swap scheduler to destroy.

    .. seealso:: `custatevecDistIndexBitSwapSchedulerDestroy`
    """
    with nogil:
        status = custatevecDistIndexBitSwapSchedulerDestroy(<Handle>handle, <DistIndexBitSwapSchedulerDescriptor>scheduler)
    check_status(status)


cpdef tuple sv_swap_worker_create(intptr_t handle, intptr_t communicator, intptr_t org_sub_sv, int32_t org_sub_sv_ind_ex, intptr_t org_event, int sv_data_type, intptr_t stream):
    """Create state vector swap worker.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        communicator (intptr_t): a pointer to the MPI communicator.
        org_sub_sv (intptr_t): a pointer to a sub state vector.
        org_sub_sv_ind_ex (int32_t): the index of the sub state vector specified by the org_sub_sv argument.
        org_event (intptr_t): the event for synchronization with the peer worker.
        sv_data_type (int): data type used by the state vector representation.
        stream (intptr_t): a stream that is used to locally execute kernels during data transfers.

    Returns:
        A 3-tuple containing:

        - intptr_t: state vector swap worker.
        - size_t: the size of the extra workspace needed.
        - size_t: the minimum-required size of the transfer workspace.

    .. seealso:: `custatevecSVSwapWorkerCreate`
    """
    cdef SVSwapWorkerDescriptor sv_swap_worker
    cdef size_t extra_workspace_size_in_bytes
    cdef size_t min_transfer_workspace_size_in_bytes
    with nogil:
        status = custatevecSVSwapWorkerCreate(<Handle>handle, &sv_swap_worker, <CommunicatorDescriptor>communicator, <void*>org_sub_sv, org_sub_sv_ind_ex, <Event>org_event, <DataType>sv_data_type, <Stream>stream, &extra_workspace_size_in_bytes, &min_transfer_workspace_size_in_bytes)
    check_status(status)
    return (<intptr_t>sv_swap_worker, extra_workspace_size_in_bytes, min_transfer_workspace_size_in_bytes)


cpdef sv_swap_worker_destroy(intptr_t handle, intptr_t sv_swap_worker):
    """This function releases the state vector swap worker.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        sv_swap_worker (intptr_t): state vector swap worker.

    .. seealso:: `custatevecSVSwapWorkerDestroy`
    """
    with nogil:
        status = custatevecSVSwapWorkerDestroy(<Handle>handle, <SVSwapWorkerDescriptor>sv_swap_worker)
    check_status(status)


cpdef sv_swap_worker_set_extra_workspace(intptr_t handle, intptr_t sv_swap_worker, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes):
    """Set extra workspace.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        sv_swap_worker (intptr_t): state vector swap worker.
        extra_workspace (intptr_t): pointer to the user-owned workspace.
        extra_workspace_size_in_bytes (size_t): size of the user-provided workspace.

    .. seealso:: `custatevecSVSwapWorkerSetExtraWorkspace`
    """
    with nogil:
        status = custatevecSVSwapWorkerSetExtraWorkspace(<Handle>handle, <SVSwapWorkerDescriptor>sv_swap_worker, <void*>extra_workspace, extra_workspace_size_in_bytes)
    check_status(status)


cpdef sv_swap_worker_set_transfer_workspace(intptr_t handle, intptr_t sv_swap_worker, intptr_t transfer_workspace, size_t transfer_workspace_size_in_bytes):
    """Set transfer workspace.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        sv_swap_worker (intptr_t): state vector swap worker.
        transfer_workspace (intptr_t): pointer to the user-owned workspace.
        transfer_workspace_size_in_bytes (size_t): size of the user-provided workspace.

    .. seealso:: `custatevecSVSwapWorkerSetTransferWorkspace`
    """
    with nogil:
        status = custatevecSVSwapWorkerSetTransferWorkspace(<Handle>handle, <SVSwapWorkerDescriptor>sv_swap_worker, <void*>transfer_workspace, transfer_workspace_size_in_bytes)
    check_status(status)


cpdef sv_swap_worker_set_sub_svs_p2p(intptr_t handle, intptr_t sv_swap_worker, dst_sub_svs_p2p, dst_sub_sv_indices_p2p, dst_events, uint32_t n_dst_sub_svs_p2p):
    """Set sub state vector pointers accessible via GPUDirect P2P with CUDA IPC events.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        sv_swap_worker (intptr_t): state vector swap worker.
        dst_sub_svs_p2p (object): an array of pointers to sub state vectors that are accessed by GPUDirect P2P. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``intptr_t``.

        dst_sub_sv_indices_p2p (object): the sub state vector indices of sub state vector pointers specified by the dst_sub_svs_p2p argument. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        dst_events (object): events used to create peer workers. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``intptr_t``.

        n_dst_sub_svs_p2p (uint32_t): the number of sub state vector pointers specified by the dst_sub_svs_p2p argument.

    .. seealso:: `custatevecSVSwapWorkerSetSubSVsP2P`
    """
    cdef nullable_unique_ptr[ vector[intptr_t] ] _dst_sub_svs_p2p_
    get_resource_ptr[intptr_t](_dst_sub_svs_p2p_, dst_sub_svs_p2p, <intptr_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _dst_sub_sv_indices_p2p_
    get_resource_ptr[int32_t](_dst_sub_sv_indices_p2p_, dst_sub_sv_indices_p2p, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[intptr_t] ] _dst_events_
    get_resource_ptr[intptr_t](_dst_events_, dst_events, <intptr_t*>NULL)
    with nogil:
        status = custatevecSVSwapWorkerSetSubSVsP2P(<Handle>handle, <SVSwapWorkerDescriptor>sv_swap_worker, <void**>(_dst_sub_svs_p2p_.data()), <const int32_t*>(_dst_sub_sv_indices_p2p_.data()), <Event*>(_dst_events_.data()), <const uint32_t>n_dst_sub_svs_p2p)
    check_status(status)


cpdef sv_swap_worker_execute(intptr_t handle, intptr_t sv_swap_worker, int64_t begin, int64_t end):
    """Execute the data transfer.

    Args:
        handle (intptr_t): the handle to cuStateVec library.
        sv_swap_worker (intptr_t): state vector swap worker.
        begin (int64_t): the index to start transfer.
        end (int64_t): the index to end transfer.

    .. seealso:: `custatevecSVSwapWorkerExecute`
    """
    with nogil:
        status = custatevecSVSwapWorkerExecute(<Handle>handle, <SVSwapWorkerDescriptor>sv_swap_worker, <custatevecIndex_t>begin, <custatevecIndex_t>end)
    check_status(status)


cpdef initialize_state_vector(intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits, int sv_type):
    """Initialize the state vector to a certain form.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        sv (intptr_t): state vector.
        sv_data_type (int): data type of state vector.
        n_index_bits (uint32_t): the number of index bits.
        sv_type (StateVectorType): the target quantum state.

    .. seealso:: `custatevecInitializeStateVector`
    """
    with nogil:
        status = custatevecInitializeStateVector(<Handle>handle, <void*>sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <_StateVectorType>sv_type)
    check_status(status)


cpdef size_t apply_matrix_batched_get_workspace_size(intptr_t handle, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, int map_type, matrix_indices, intptr_t matrices, int matrix_data_type, int layout, int32_t adjoint, uint32_t n_matrices, uint32_t n_targets, uint32_t n_controls, int compute_type) except? 0:
    """This function gets the required workspace size for :func:`apply_matrix_batched`.

    Args:
        handle (intptr_t): the handle to the cuStateVec context.
        sv_data_type (int): Data type of the state vector.
        n_index_bits (uint32_t): the number of index bits of the state vector.
        n_svs (uint32_t): the number of state vectors.
        sv_stride (int64_t): distance of two consecutive state vectors.
        map_type (MatrixMapType): enumerator specifying the way to assign matrices.
        matrix_indices (object): pointer to a host or device array of matrix indices. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        matrices (intptr_t): pointer to allocated matrices in one contiguous memory chunk on host or device.
        matrix_data_type (int): data type of matrix.
        layout (MatrixLayout): enumerator specifying the memory layout of matrix.
        adjoint (int32_t): apply adjoint of matrix.
        n_matrices (uint32_t): the number of matrices.
        n_targets (uint32_t): the number of target bits.
        n_controls (uint32_t): the number of control bits.
        compute_type (ComputeType): compute_type of matrix multiplication.

    Returns:
        size_t: workspace size.

    .. seealso:: `custatevecApplyMatrixBatchedGetWorkspaceSize`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _matrix_indices_
    get_resource_ptr[int32_t](_matrix_indices_, matrix_indices, <int32_t*>NULL)
    cdef size_t extra_workspace_size_in_bytes
    with nogil:
        status = custatevecApplyMatrixBatchedGetWorkspaceSize(<Handle>handle, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const uint32_t>n_svs, <const custatevecIndex_t>sv_stride, <_MatrixMapType>map_type, <const int32_t*>(_matrix_indices_.data()), <const void*>matrices, <DataType>matrix_data_type, <_MatrixLayout>layout, <const int32_t>adjoint, <const uint32_t>n_matrices, <const uint32_t>n_targets, <const uint32_t>n_controls, <_ComputeType>compute_type, &extra_workspace_size_in_bytes)
    check_status(status)
    return extra_workspace_size_in_bytes


cpdef apply_matrix_batched(intptr_t handle, intptr_t batched_sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, int map_type, matrix_indices, intptr_t matrices, int matrix_data_type, int layout, int32_t adjoint, uint32_t n_matrices, targets, uint32_t n_targets, controls, control_bit_values, uint32_t n_controls, int compute_type, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes):
    """This function applies one gate matrix to each one of a set of batched state vectors.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        batched_sv (intptr_t): batched state vector allocated in one continuous memory chunk on device.
        sv_data_type (int): data type of the state vectors.
        n_index_bits (uint32_t): the number of index bits of the state vectors.
        n_svs (uint32_t): the number of state vectors.
        sv_stride (int64_t): distance of two consecutive state vectors.
        map_type (MatrixMapType): enumerator specifying the way to assign matrices.
        matrix_indices (object): pointer to a host or device array of matrix indices. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        matrices (intptr_t): pointer to allocated matrices in one contiguous memory chunk on host or device.
        matrix_data_type (int): data type of matrices.
        layout (MatrixLayout): enumerator specifying the memory layout of matrix.
        adjoint (int32_t): apply adjoint of matrix.
        n_matrices (uint32_t): the number of matrices.
        targets (object): pointer to a host array of target bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_targets (uint32_t): the number of target bits.
        controls (object): pointer to a host array of control bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        control_bit_values (object): pointer to a host array of control bit values. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_controls (uint32_t): the number of control bits.
        compute_type (ComputeType): compute_type of matrix multiplication.
        extra_workspace (intptr_t): extra workspace.
        extra_workspace_size_in_bytes (size_t): extra workspace size.

    .. seealso:: `custatevecApplyMatrixBatched`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _matrix_indices_
    get_resource_ptr[int32_t](_matrix_indices_, matrix_indices, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _targets_
    get_resource_ptr[int32_t](_targets_, targets, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _controls_
    get_resource_ptr[int32_t](_controls_, controls, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _control_bit_values_
    get_resource_ptr[int32_t](_control_bit_values_, control_bit_values, <int32_t*>NULL)
    with nogil:
        status = custatevecApplyMatrixBatched(<Handle>handle, <void*>batched_sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const uint32_t>n_svs, <custatevecIndex_t>sv_stride, <_MatrixMapType>map_type, <const int32_t*>(_matrix_indices_.data()), <const void*>matrices, <DataType>matrix_data_type, <_MatrixLayout>layout, <const int32_t>adjoint, <const uint32_t>n_matrices, <const int32_t*>(_targets_.data()), <const uint32_t>n_targets, <const int32_t*>(_controls_.data()), <const int32_t*>(_control_bit_values_.data()), <const uint32_t>n_controls, <_ComputeType>compute_type, <void*>extra_workspace, extra_workspace_size_in_bytes)
    check_status(status)


cpdef abs2sum_array_batched(intptr_t handle, intptr_t batched_sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, intptr_t abs2sum_arrays, int64_t abs2sum_array_stride, bit_ordering, uint32_t bit_ordering_len, mask_bit_strings, mask_ordering, uint32_t mask_len):
    """Calculate batched abs2sum array for a given set of index bits.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        batched_sv (intptr_t): batch of state vectors.
        sv_data_type (int): data type of state vector.
        n_index_bits (uint32_t): the number of index bits.
        n_svs (uint32_t): the number of state vectors in a batch.
        sv_stride (int64_t): the stride of state vector.
        abs2sum_arrays (intptr_t): pointer to a host or device array of sums of squared absolute values.
        abs2sum_array_stride (int64_t): the distance between consequence abs2sum_arrays.
        bit_ordering (object): pointer to a host array of index bit ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        bit_ordering_len (uint32_t): the length of bit_ordering.
        mask_bit_strings (object): pointer to a host or device array of mask bit strings. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``custatevecIndex_t``.

        mask_ordering (object): pointer to a host array for the mask ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        mask_len (uint32_t): the length of mask.

    .. seealso:: `custatevecAbs2SumArrayBatched`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _bit_ordering_
    get_resource_ptr[int32_t](_bit_ordering_, bit_ordering, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _mask_bit_strings_
    get_resource_ptr[int64_t](_mask_bit_strings_, mask_bit_strings, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_ordering_
    get_resource_ptr[int32_t](_mask_ordering_, mask_ordering, <int32_t*>NULL)
    with nogil:
        status = custatevecAbs2SumArrayBatched(<Handle>handle, <const void*>batched_sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const uint32_t>n_svs, <const custatevecIndex_t>sv_stride, <double*>abs2sum_arrays, <const custatevecIndex_t>abs2sum_array_stride, <const int32_t*>(_bit_ordering_.data()), <const uint32_t>bit_ordering_len, <const custatevecIndex_t*>(_mask_bit_strings_.data()), <const int32_t*>(_mask_ordering_.data()), <const uint32_t>mask_len)
    check_status(status)


cpdef size_t collapse_by_bit_string_batched_get_workspace_size(intptr_t handle, uint32_t n_svs, bit_strings, norms) except? 0:
    """This function gets the required workspace size for :func:`collapse_by_bit_string_batched`.

    Args:
        handle (intptr_t): the handle to the cuStateVec context.
        n_svs (uint32_t): the number of batched state vectors.
        bit_strings (object): pointer to an array of bit strings, on either host or device. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``custatevecIndex_t``.

        norms (object): pointer to an array of normalization constants, on either host or device. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``float``.


    Returns:
        size_t: workspace size.

    .. seealso:: `custatevecCollapseByBitStringBatchedGetWorkspaceSize`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _bit_strings_
    get_resource_ptr[int64_t](_bit_strings_, bit_strings, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[double] ] _norms_
    get_resource_ptr[double](_norms_, norms, <double*>NULL)
    cdef size_t extra_workspace_size_in_bytes
    with nogil:
        status = custatevecCollapseByBitStringBatchedGetWorkspaceSize(<Handle>handle, <const uint32_t>n_svs, <const custatevecIndex_t*>(_bit_strings_.data()), <const double*>(_norms_.data()), &extra_workspace_size_in_bytes)
    check_status(status)
    return extra_workspace_size_in_bytes


cpdef collapse_by_bit_string_batched(intptr_t handle, intptr_t batched_sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, bit_strings, bit_ordering, uint32_t bit_string_len, norms, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes):
    """Collapse the batched state vectors to the state specified by a given bit string.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        batched_sv (intptr_t): batched state vector allocated in one continuous memory chunk on device.
        sv_data_type (int): data type of the state vectors.
        n_index_bits (uint32_t): the number of index bits of the state vectors.
        n_svs (uint32_t): the number of batched state vectors.
        sv_stride (int64_t): distance of two consecutive state vectors.
        bit_strings (object): pointer to an array of bit strings, on either host or device. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``custatevecIndex_t``.

        bit_ordering (object): pointer to a host array of bit string ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        bit_string_len (uint32_t): length of bit string.
        norms (object): pointer to an array of normalization constants on either host or device. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``float``.

        extra_workspace (intptr_t): extra workspace.
        extra_workspace_size_in_bytes (size_t): size of the extra workspace.

    .. seealso:: `custatevecCollapseByBitStringBatched`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _bit_strings_
    get_resource_ptr[int64_t](_bit_strings_, bit_strings, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _bit_ordering_
    get_resource_ptr[int32_t](_bit_ordering_, bit_ordering, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[double] ] _norms_
    get_resource_ptr[double](_norms_, norms, <double*>NULL)
    with nogil:
        status = custatevecCollapseByBitStringBatched(<Handle>handle, <void*>batched_sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const uint32_t>n_svs, <const custatevecIndex_t>sv_stride, <const custatevecIndex_t*>(_bit_strings_.data()), <const int32_t*>(_bit_ordering_.data()), <const uint32_t>bit_string_len, <const double*>(_norms_.data()), <void*>extra_workspace, extra_workspace_size_in_bytes)
    check_status(status)


cpdef measure_batched(intptr_t handle, intptr_t batched_sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, intptr_t bit_strings, bit_ordering, uint32_t bit_string_len, randnums, int collapse):
    """Single qubit measurements for batched state vectors.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        batched_sv (intptr_t): batched state vectors.
        sv_data_type (int): data type of the state vector.
        n_index_bits (uint32_t): the number of index bits.
        n_svs (uint32_t): the number of state vectors in the batched state vector.
        sv_stride (int64_t): the distance between state vectors in the batch.
        bit_strings (intptr_t): pointer to a host or device array of measured bit strings.
        bit_ordering (object): pointer to a host array of bit string ordering. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        bit_string_len (uint32_t): length of bitString.
        randnums (object): pointer to a host or device array of random numbers. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``float``.

        collapse (CollapseOp): Collapse operation.

    .. seealso:: `custatevecMeasureBatched`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _bit_ordering_
    get_resource_ptr[int32_t](_bit_ordering_, bit_ordering, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[double] ] _randnums_
    get_resource_ptr[double](_randnums_, randnums, <double*>NULL)
    with nogil:
        status = custatevecMeasureBatched(<Handle>handle, <void*>batched_sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const uint32_t>n_svs, <const custatevecIndex_t>sv_stride, <custatevecIndex_t*>bit_strings, <const int32_t*>(_bit_ordering_.data()), <const uint32_t>bit_string_len, <const double*>(_randnums_.data()), <_CollapseOp>collapse)
    check_status(status)


cpdef intptr_t sub_sv_migrator_create(intptr_t handle, intptr_t device_slots, int sv_data_type, int n_device_slots, int n_local_index_bits) except? 0:
    """Create sub state vector migrator descriptor.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        device_slots (intptr_t): pointer to sub state vectors on device.
        sv_data_type (int): data type of state vector.
        n_device_slots (int): the number of sub state vectors in device_slots.
        n_local_index_bits (int): the number of index bits of sub state vectors.

    Returns:
        intptr_t: pointer to a new migrator descriptor.

    .. seealso:: `custatevecSubSVMigratorCreate`
    """
    cdef SubSVMigratorDescriptor migrator
    with nogil:
        status = custatevecSubSVMigratorCreate(<Handle>handle, &migrator, <void*>device_slots, <DataType>sv_data_type, n_device_slots, n_local_index_bits)
    check_status(status)
    return <intptr_t>migrator


cpdef sub_sv_migrator_destroy(intptr_t handle, intptr_t migrator):
    """Destroy sub state vector migrator descriptor.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        migrator (intptr_t): the migrator descriptor.

    .. seealso:: `custatevecSubSVMigratorDestroy`
    """
    with nogil:
        status = custatevecSubSVMigratorDestroy(<Handle>handle, <SubSVMigratorDescriptor>migrator)
    check_status(status)


cpdef sub_sv_migrator_migrate(intptr_t handle, intptr_t migrator, int device_slot_ind_ex, intptr_t src_sub_sv, intptr_t dst_sub_sv, int64_t begin, int64_t end):
    """Sub state vector migration.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        migrator (intptr_t): the migrator descriptor.
        device_slot_ind_ex (int): the index to specify sub state vector to migrate.
        src_sub_sv (intptr_t): a pointer to a sub state vector that is migrated to deviceSlots.
        dst_sub_sv (intptr_t): a pointer to a sub state vector that is migrated from deviceSlots.
        begin (int64_t): the index to start migration.
        end (int64_t): the index to end migration.

    .. seealso:: `custatevecSubSVMigratorMigrate`
    """
    with nogil:
        status = custatevecSubSVMigratorMigrate(<Handle>handle, <SubSVMigratorDescriptor>migrator, device_slot_ind_ex, <const void*>src_sub_sv, <void*>dst_sub_sv, <custatevecIndex_t>begin, <custatevecIndex_t>end)
    check_status(status)


cpdef size_t compute_expectation_batched_get_workspace_size(intptr_t handle, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, intptr_t matrices, int matrix_data_type, int layout, uint32_t n_matrices, uint32_t n_basis_bits, int compute_type) except? 0:
    """This function gets the required workspace size for :func:`compute_expectation_batched`.

    Args:
        handle (intptr_t): the handle to the cuStateVec context.
        sv_data_type (int): Data type of the state vector.
        n_index_bits (uint32_t): the number of index bits of the state vector.
        n_svs (uint32_t): the number of state vectors.
        sv_stride (int64_t): distance of two consecutive state vectors.
        matrices (intptr_t): pointer to allocated matrices in one contiguous memory chunk on host or device.
        matrix_data_type (int): data type of matrices.
        layout (MatrixLayout): enumerator specifying the memory layout of matrix.
        n_matrices (uint32_t): the number of matrices.
        n_basis_bits (uint32_t): the number of basis bits.
        compute_type (ComputeType): compute_type of matrix multiplication.

    Returns:
        size_t: size of the extra workspace.

    .. seealso:: `custatevecComputeExpectationBatchedGetWorkspaceSize`
    """
    cdef size_t extra_workspace_size_in_bytes
    with nogil:
        status = custatevecComputeExpectationBatchedGetWorkspaceSize(<Handle>handle, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const uint32_t>n_svs, <const custatevecIndex_t>sv_stride, <const void*>matrices, <DataType>matrix_data_type, <_MatrixLayout>layout, <const uint32_t>n_matrices, <const uint32_t>n_basis_bits, <_ComputeType>compute_type, &extra_workspace_size_in_bytes)
    check_status(status)
    return extra_workspace_size_in_bytes


cpdef compute_expectation_batched(intptr_t handle, intptr_t batched_sv, int sv_data_type, uint32_t n_index_bits, uint32_t n_svs, int64_t sv_stride, intptr_t expectation_values, intptr_t matrices, int matrix_data_type, int layout, uint32_t n_matrices, basis_bits, uint32_t n_basis_bits, int compute_type, intptr_t extra_workspace, size_t extra_workspace_size_in_bytes):
    """Compute the expectation values of matrix observables for each of the batched state vectors.

    Args:
        handle (intptr_t): the handle to the cuStateVec library.
        batched_sv (intptr_t): batched state vector allocated in one continuous memory chunk on device.
        sv_data_type (int): data type of the state vector.
        n_index_bits (uint32_t): the number of index bits of the state vector.
        n_svs (uint32_t): the number of state vectors.
        sv_stride (int64_t): distance of two consecutive state vectors.
        expectation_values (intptr_t): pointer to a host or device array to store expectation values.
        matrices (intptr_t): pointer to allocated matrices in one contiguous memory chunk on host or device.
        matrix_data_type (int): data type of matrices.
        layout (MatrixLayout): matrix memory layout.
        n_matrices (uint32_t): the number of matrices.
        basis_bits (object): pointer to a host array of basis index bits. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        n_basis_bits (uint32_t): the number of basis bits.
        compute_type (ComputeType): compute_type of matrix multiplication.
        extra_workspace (intptr_t): pointer to an extra workspace.
        extra_workspace_size_in_bytes (size_t): the size of extra workspace.

    .. seealso:: `custatevecComputeExpectationBatched`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _basis_bits_
    get_resource_ptr[int32_t](_basis_bits_, basis_bits, <int32_t*>NULL)
    with nogil:
        status = custatevecComputeExpectationBatched(<Handle>handle, <const void*>batched_sv, <DataType>sv_data_type, <const uint32_t>n_index_bits, <const uint32_t>n_svs, <custatevecIndex_t>sv_stride, <double2*>expectation_values, <const void*>matrices, <DataType>matrix_data_type, <_MatrixLayout>layout, <const uint32_t>n_matrices, <const int32_t*>(_basis_bits_.data()), <const uint32_t>n_basis_bits, <_ComputeType>compute_type, <void*>extra_workspace, extra_workspace_size_in_bytes)
    check_status(status)


cpdef set_math_mode(intptr_t handle, int mode):
    """Set the compute precision mode.

    Args:
        handle (intptr_t): Opaque handle holding cuStateVec's library context.
        mode (MathMode): the compute precision mode.

    .. seealso:: `custatevecSetMathMode`
    """
    with nogil:
        status = custatevecSetMathMode(<Handle>handle, <_MathMode>mode)
    check_status(status)


cpdef int get_math_mode(intptr_t handle) except? -1:
    """Get the current compute precision mode.

    Args:
        handle (intptr_t): Opaque handle holding cuStateVec's library context.

    Returns:
        int: the compute precision mode.

    .. seealso:: `custatevecGetMathMode`
    """
    cdef _MathMode mode
    with nogil:
        status = custatevecGetMathMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


# for backward compat
collapse_by_bitstring = collapse_by_bit_string
collapse_by_bitstring_batched_get_workspace_size = collapse_by_bit_string_batched_get_workspace_size
collapse_by_bitstring_batched = collapse_by_bit_string_batched
Collapse = CollapseOp
MAJOR_VER = CUSTATEVEC_VER_MAJOR
MINOR_VER = CUSTATEVEC_VER_MINOR
PATCH_VER = CUSTATEVEC_VER_PATCH
VERSION = CUSTATEVEC_VERSION
ALLOCATOR_NAME_LEN = CUSTATEVEC_ALLOCATOR_NAME_LEN
MAX_SEGMENT_MASK_SIZE = CUSTATEVEC_MAX_SEGMENT_MASK_SIZE


cpdef tuple abs2sum_on_z_basis(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        bint get_parity0, bint get_parity1,
        basis_bits, uint32_t n_basis_bits):
    """Calculates the sum of squared absolute values on a given Z product basis.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python :class:`int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        get_parity0 (bool): Whether to compute the sum of squared absolute values
            for parity 0.
        get_parity1 (bool): Whether to compute the sum of squared absolute values
            for parity 1.
        basis_bits: A host array of Z-basis index bits. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of index bits

        n_basis_bits (uint32_t): the number of basis bits.

    Returns:
        tuple:
            A 2-tuple of the calculated sums for partiy 0 and 1, respectively.
            If the corresponding bool is set to `False`, `None` is returned.

    .. seealso:: `custatevecAbs2SumOnZBasis`
    """
    if not get_parity0 and not get_parity1:
        raise ValueError("no target to compute")
    cdef double abs2sum0, abs2sum1
    cdef double* abs2sum0_ptr
    cdef double* abs2sum1_ptr
    abs2sum0_ptr = &abs2sum0 if get_parity0 else NULL
    abs2sum1_ptr = &abs2sum1 if get_parity1 else NULL

    # basis_bits can be a pointer address, or a Python sequence
    cdef nullable_unique_ptr[ vector[int32_t] ] _basis_bits_
    get_resource_ptr[int32_t](_basis_bits_, basis_bits, <int32_t*>NULL)

    with nogil:
        status = custatevecAbs2SumOnZBasis(
            <Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            abs2sum0_ptr, abs2sum1_ptr,
            <const int32_t*>(_basis_bits_.data()), n_basis_bits)
    check_status(status)
    if get_parity0 and get_parity1:
        return (abs2sum0, abs2sum1)
    elif get_parity0:
        return (abs2sum0, None)
    elif get_parity1:
        return (None, abs2sum1)


cpdef swap_index_bits(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        swapped_bits, uint32_t n_swapped_bits,
        mask_bit_string, mask_ordering, uint32_t mask_len):
    """Swap index bits and reorder statevector elements on the device.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python :class:`int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        swapped_bits: A host array of pairs of swapped index bits. It can be

            - an :class:`int` as the pointer address to the nested sequence
            - a nested Python sequence of swapped index bits

        n_swapped_bits (uint32_t): The number of pairs of swapped index bits.
        mask_bit_string: A host array for specifying mask values. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of mask values

        mask_ordering: A host array of mask ordering. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of index bit ordering

        mask_len (uint32_t): The length of ``mask_ordering``.

    .. seealso:: `custatevecSwapIndexBits`
    """
    # swapped_bits can be:
    #   - a plain pointer address
    #   - a nested Python sequence (ex: a list of 2-tuples)
    # Note: it cannot be a mix of sequences and ints. It also cannot be a
    # 1D sequence (of ints), because it's inefficient.
    cdef vector[intptr_t] swappedBitsCData
    cdef int2* swappedBitsPtr
    if is_nested_sequence(swapped_bits):
        try:
            # direct conversion
            data = _numpy.asarray(swapped_bits, dtype=_numpy.int32)
            data = data.reshape(-1)
        except:
            # unlikely, but let's do it in the stupid way
            data = _numpy.empty(2*n_swapped_bits, dtype=_numpy.int32)
            for i, (first, second) in enumerate(swapped_bits):
                data[2*i] = first
                data[2*i+1] = second
        assert data.size == 2*n_swapped_bits
        swappedBitsPtr = <int2*>(<intptr_t>data.ctypes.data)
    elif isinstance(swapped_bits, int):
        # a pointer address, take it as is
        swappedBitsPtr = <int2*><intptr_t>swapped_bits
    else:
        raise ValueError("swapped_bits is provided in an "
                         "un-recognized format")

    # mask_bit_string can be a pointer address, or a Python sequence
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_bit_string_
    get_resource_ptr[int32_t](_mask_bit_string_, mask_bit_string, <int32_t*>NULL)

    # mask_ordering can be a pointer address, or a Python sequence
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_ordering_
    get_resource_ptr[int32_t](_mask_ordering_, mask_ordering, <int32_t*>NULL)

    with nogil:
        status = custatevecSwapIndexBits(
            <Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            swappedBitsPtr, n_swapped_bits,
            <const int32_t*>(_mask_bit_string_.data()), <const int32_t*>(_mask_ordering_.data()), mask_len)
    check_status(status)


cpdef multi_device_swap_index_bits(
        handles, uint32_t n_handles, sub_svs, int sv_data_type,
        uint32_t n_global_index_bits, uint32_t n_local_index_bits,
        swapped_bits, uint32_t n_swapped_bits,
        mask_bit_string, mask_ordering, uint32_t mask_len,
        int device_network_type):
    """Swap index bits and reorder statevector elements on multiple devices.

    Args:
        handles: A host array of the library handles. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`, each of which is a valid
              library handle

        n_handles (uint32_t): The number of handles.
        sub_svs: A host array of the sub-statevector pointers. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of :class:`int`, each of which is a valid
              sub-statevector pointer (on device)

        sv_data_type (cuquantum.cudaDataType): The data type of the statevectors.
        n_global_index_bits (uint32_t): The number of the global index bits.
        n_local_index_bits (uint32_t): The number of the local index bits.
        swapped_bits: A host array of pairs of swapped index bits. It can be

            - an :class:`int` as the pointer address to the nested sequence
            - a nested Python sequence of swapped index bits

        n_swapped_bits (uint32_t): The number of pairs of swapped index bits.
        mask_bit_string: A host array for specifying mask values. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of mask values

        mask_ordering: A host array of mask ordering. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of index bit ordering

        mask_len (uint32_t): The length of ``mask_ordering``.
        device_network_type (DeviceNetworkType): The device network topology.

    .. seealso:: `custatevecMultiDeviceSwapIndexBits`
    """
    # handles can be a pointer address, or a Python sequence
    cdef nullable_unique_ptr[ vector[intptr_t] ] _handles_
    get_resource_ptr[intptr_t](_handles_, handles, <intptr_t*>NULL)

    # sub_svs can be a pointer address, or a Python sequence
    cdef nullable_unique_ptr[ vector[intptr_t] ] _sub_svs_
    get_resource_ptr[intptr_t](_sub_svs_, sub_svs, <intptr_t*>NULL)

    # swapped_bits can be:
    #   - a plain pointer address
    #   - a nested Python sequence (ex: a list of 2-tuples)
    # Note: it cannot be a mix of sequences and ints. It also cannot be a
    # 1D sequence (of ints), because it's inefficient.
    cdef vector[intptr_t] swappedBitsCData
    cdef int2* swappedBitsPtr
    if is_nested_sequence(swapped_bits):
        try:
            # direct conversion
            data = _numpy.asarray(swapped_bits, dtype=_numpy.int32)
            data = data.reshape(-1)
        except:
            # unlikely, but let's do it in the stupid way
            data = _numpy.empty(2*n_swapped_bits, dtype=_numpy.int32)
            for i, (first, second) in enumerate(swapped_bits):
                data[2*i] = first
                data[2*i+1] = second
        assert data.size == 2*n_swapped_bits
        swappedBitsPtr = <int2*>(<intptr_t>data.ctypes.data)
    elif isinstance(swapped_bits, int):
        # a pointer address, take it as is
        swappedBitsPtr = <int2*><intptr_t>swapped_bits
    else:
        raise ValueError("swapped_bits is provided in an "
                         "un-recognized format")

    # mask_bit_string can be a pointer address, or a Python sequence
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_bit_string_
    get_resource_ptr[int32_t](_mask_bit_string_, mask_bit_string, <int32_t*>NULL)

    # mask_ordering can be a pointer address, or a Python sequence
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_ordering_
    get_resource_ptr[int32_t](_mask_ordering_, mask_ordering, <int32_t*>NULL)

    with nogil:
        status = custatevecMultiDeviceSwapIndexBits(
            <Handle*>(_handles_.data()), n_handles, <void**>(_sub_svs_.data()), <DataType>sv_data_type,
            n_global_index_bits, n_local_index_bits,
            swappedBitsPtr, n_swapped_bits,
            <const int32_t*>(_mask_bit_string_.data()), <const int32_t*>(_mask_ordering_.data()), mask_len,
            <_DeviceNetworkType>device_network_type)
    check_status(status)


cpdef sv_swap_worker_set_parameters(
        intptr_t handle, intptr_t worker, params, int peer):
    """Set data transfer parameters for the distributed sub statevector
    swap workers.

    Args:
        handle (intptr_t): The library handle.
        worker (intptr_t): The worker descriptor.
        params: The data transfer parameters. It can be:

            - an :class:`int` as the pointer address to the struct
            - a :class:`numpy.ndarray` of dtype :obj:`sv_swap_parameters_dtype`
            - a :class:`SVSwapParameters`
        peer (int): The peer process identifier of the data transfer.

    .. seealso:: `custatevecSVSwapWorkerSetParameters`
    """
    cdef _SVSwapParameters* paramPtr
    if isinstance(params, SVSwapParameters):
        paramPtr = <_SVSwapParameters*><intptr_t>params.ptr
    elif isinstance(params, _numpy.ndarray):
        _check_for_sv_swap_parameters(params)
        paramPtr = <_SVSwapParameters*><intptr_t>params.ctypes.data
    elif isinstance(params, int):
        paramPtr = <_SVSwapParameters*><intptr_t>params
    else:
        raise ValueError("params must be of type SVSwapParameters, "
                         "numpy.ndarray, or int")

    with nogil:
        status = custatevecSVSwapWorkerSetParameters(
            <Handle>handle, <SVSwapWorkerDescriptor>worker, paramPtr, peer)
    check_status(status)


cpdef SVSwapParameters dist_index_bit_swap_scheduler_get_parameters(
        intptr_t handle, intptr_t scheduler, int32_t swap_batch_index,
        int32_t org_sub_sv_index, params=None):
    """Get the data transfer parameters from the scheduler.

    Args:
        handle (intptr_t): The library handle.
        scheduler (intptr_t): The scheduler descriptor.
        swap_batch_index (int32_t): The swap batch index for statevector
            swap parameters.
        org_sub_sv_index (int32_t): The index of the origin sub statevector.
        params: Optional. If set, it should be

            - an :class:`int` as the pointer address to the struct
            - a :class:`numpy.ndarray` of dtype :obj:`sv_swap_parameters_dtype`
            - a :class:`SVSwapParameters`

            and the result would be written in-place. Additionally, if an
            :class:`int` is passed, there is no return value.

    Returns:
        SVSwapParameters:
            the data transfer parameters that can be consumed later by a data
            transfer worker.

    .. seealso:: `custatevecDistIndexBitSwapSchedulerGetParameters`
    """
    cdef SVSwapParameters param = None  # placeholder
    cdef _SVSwapParameters* paramPtr = NULL  # placeholder
    cdef bint to_return = True

    if params is None:
        param = SVSwapParameters()
    else:
        if isinstance(params, SVSwapParameters):
            param = params
        elif isinstance(params, _numpy.ndarray):
            param = SVSwapParameters.from_data(params)  # also check validity
        elif isinstance(params, int):
            #param = SVSwapParameters.from_ptr(params)
            # no check, user is responsible
            # don't even create an SVSwapParameters instance, we want it to
            # be blazingly fast
            paramPtr = <_SVSwapParameters*><intptr_t>params
            to_return = False
        else:
            raise ValueError("params must be of type SVSwapParameters or "
                             "of dtype sv_swap_parameters_dtype")
    if paramPtr == NULL:
        paramPtr = <_SVSwapParameters*><intptr_t>param.ptr

    with nogil:
        status = custatevecDistIndexBitSwapSchedulerGetParameters(
            <Handle>handle, <DistIndexBitSwapSchedulerDescriptor>scheduler,
            swap_batch_index, org_sub_sv_index, paramPtr)
    check_status(status)
    return param if to_return else None


cpdef uint32_t dist_index_bit_swap_scheduler_set_index_bit_swaps(
        intptr_t handle, intptr_t scheduler,
        swapped_bits, uint32_t n_swapped_bits,
        mask_bit_string, mask_ordering, uint32_t mask_len) except*:
    """Schedule the index bits to be swapped across processes.

    Args:
        handle (intptr_t): The library handle.
        scheduler (intptr_t): The scheduler descriptor.
        swapped_bits: A host array of pairs of swapped index bits. It can be

            - an :class:`int` as the pointer address to the nested sequence
            - a nested Python sequence of swapped index bits

        n_swapped_bits (uint32_t): The number of pairs of swapped index bits.
        mask_bit_string: A host array for specifying mask values. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of mask values

        mask_ordering: A host array of mask ordering. It can be

            - an :class:`int` as the pointer address to the array
            - a Python sequence of index bit ordering

        mask_len (uint32_t): The length of ``mask_ordering``.

    .. seealso:: `custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps`
    """
    # swapped_bits can be:
    #   - a plain pointer address
    #   - a nested Python sequence (ex: a list of 2-tuples)
    # Note: it cannot be a mix of sequences and ints. It also cannot be a
    # 1D sequence (of ints), because it's inefficient.
    cdef vector[intptr_t] swappedBitsCData
    cdef int2* swappedBitsPtr
    if is_nested_sequence(swapped_bits):
        try:
            # direct conversion
            data = _numpy.asarray(swapped_bits, dtype=_numpy.int32)
            data = data.reshape(-1)
        except:
            # unlikely, but let's do it in the stupid way
            data = _numpy.empty(2*n_swapped_bits, dtype=_numpy.int32)
            for i, (first, second) in enumerate(swapped_bits):
                data[2*i] = first
                data[2*i+1] = second
        assert data.size == 2*n_swapped_bits
        swappedBitsPtr = <int2*>(<intptr_t>data.ctypes.data)
    elif isinstance(swapped_bits, int):
        # a pointer address, take it as is
        swappedBitsPtr = <int2*><intptr_t>swapped_bits
    else:
        raise ValueError("swapped_bits is provided in an "
                         "un-recognized format")

    # mask_bit_string can be a pointer address, or a Python sequence
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_bit_string_
    get_resource_ptr[int32_t](_mask_bit_string_, mask_bit_string, <int32_t*>NULL)

    # mask_ordering can be a pointer address, or a Python sequence
    cdef nullable_unique_ptr[ vector[int32_t] ] _mask_ordering_
    get_resource_ptr[int32_t](_mask_ordering_, mask_ordering, <int32_t*>NULL)

    cdef uint32_t n_swap_batches
    with nogil:
        status = custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(
            <Handle>handle, <DistIndexBitSwapSchedulerDescriptor>scheduler,
            swappedBitsPtr, n_swapped_bits,
            <const int32_t*>(_mask_bit_string_.data()), <const int32_t*>(_mask_ordering_.data()), mask_len,
            &n_swap_batches)
    check_status(status)

    return n_swap_batches


cpdef set_device_mem_handler(intptr_t handle, handler):
    """ Set the device memory handler for cuStateVec.

    The ``handler`` object can be passed in multiple ways:

      - If ``handler`` is an :class:`int`, it refers to the address of a fully
        initialized `custatevecDeviceMemHandler_t` struct.
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

    .. seealso:: `custatevecSetDeviceMemHandler`
    """
    cdef bytes name
    cdef _DeviceMemHandler our_handler
    cdef _DeviceMemHandler* handlerPtr = &our_handler

    if isinstance(handler, int):
        handlerPtr = <_DeviceMemHandler*><intptr_t>handler
    elif cpython.PySequence_Check(handler):
        name = handler[-1].encode('ascii')
        if len(name) > CUSTATEVEC_ALLOCATOR_NAME_LEN:
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
        status = custatevecSetDeviceMemHandler(<Handle>handle, handlerPtr)
    check_status(status)


cpdef tuple get_device_mem_handler(intptr_t handle):
    """ Get the device memory handler for cuStateVec.

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

    .. seealso:: `custatevecGetDeviceMemHandler`
    """
    cdef _DeviceMemHandler handler
    with nogil:
        status = custatevecGetDeviceMemHandler(<Handle>handle, &handler)
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

          where ``log_level`` (:class:`int`), ``func_name`` (`str`), and
          ``message`` (`str`) are provided by the logger API.

    .. seealso:: `custatevecLoggerSetCallbackData`
    """
    func_arg = (callback, args, kwargs)
    # if only set once, the callback lifetime should be as long as this module,
    # because we don't know when the logger is done using it
    owner_pyobj['callback'] = func_arg
    with nogil:
        status = custatevecLoggerSetCallbackData(
            <LoggerCallbackData>logger_callback_with_data, <void*>(func_arg))
    check_status(status)


# who owns a reference to user-provided Python objects (k: owner, v: object)
cdef dict owner_pyobj = {}
