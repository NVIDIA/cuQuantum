# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.11.0. Do not modify it directly.

cimport cython
cimport cpython
from cpython.memoryview cimport PyMemoryView_FromMemory
from cpython cimport buffer as _buffer
from libcpp.vector cimport vector

from ._utils cimport (get_resource_ptr, get_nested_resource_ptr, nested_resource, nullable_unique_ptr,
                      get_buffer_pointer, get_resource_ptrs, DeviceAllocType, DeviceFreeType,
                      cuqnt_alloc_wrapper, cuqnt_free_wrapper, logger_callback_with_data)

from enum import IntEnum as _IntEnum
import warnings as _warnings

import numpy as _numpy


###############################################################################
# POD
###############################################################################

pauli_term_dtype = _numpy.dtype([
    ("xzbits", _numpy.intp, ),
    ("coef", _numpy.intp, ),
    ], align=True)


cdef class PauliTerm:
    """Empty-initialize an instance of `cupaulipropPauliTerm_t`.


    .. seealso:: `cupaulipropPauliTerm_t`
    """
    cdef:
        readonly object _data

    def __init__(self):
        arr = _numpy.empty(1, dtype=pauli_term_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(cupaulipropPauliTerm_t), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(cupaulipropPauliTerm_t)}"

    def __repr__(self):
        return f"<{__name__}.PauliTerm object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        return self._data.ctypes.data

    def __eq__(self, other):
        if not isinstance(other, PauliTerm):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    @property
    def xzbits(self):
        """int: non-owning pointer to the packed X and Z bits arranged as an array of `cupaulipropPackedIntegerType_t`"""
        return int(self._data.xzbits[0])

    @xzbits.setter
    def xzbits(self, val):
        self._data.xzbits = val

    @property
    def coef(self):
        """int: non-owning pointer to the coefficient (real/complex float/double)"""
        return int(self._data.coef[0])

    @coef.setter
    def coef(self, val):
        self._data.coef = val

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an PauliTerm instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `pauli_term_dtype` holding the data.
        """
        cdef PauliTerm obj = PauliTerm.__new__(PauliTerm)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != pauli_term_dtype:
            raise ValueError("data array must be of dtype pauli_term_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False):
        """Create an PauliTerm instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef PauliTerm obj = PauliTerm.__new__(PauliTerm)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(cupaulipropPauliTerm_t), flag)
        data = _numpy.ndarray((1,), buffer=buf,
                              dtype=pauli_term_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


truncation_strategy_dtype = _numpy.dtype([
    ("strategy", _numpy.int32, ),
    ("param_struct", _numpy.intp, ),
    ], align=True)


cdef class TruncationStrategy:
    """Empty-initialize an instance of `cupaulipropTruncationStrategy_t`.


    .. seealso:: `cupaulipropTruncationStrategy_t`
    """
    cdef:
        readonly object _data

    def __init__(self):
        arr = _numpy.empty(1, dtype=truncation_strategy_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(cupaulipropTruncationStrategy_t), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(cupaulipropTruncationStrategy_t)}"

    def __repr__(self):
        return f"<{__name__}.TruncationStrategy object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        return self._data.ctypes.data

    def __eq__(self, other):
        if not isinstance(other, TruncationStrategy):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    @property
    def strategy(self):
        """int: which kind of truncation strategy to apply"""
        return int(self._data.strategy[0])

    @strategy.setter
    def strategy(self, val):
        self._data.strategy = val

    @property
    def param_struct(self):
        """int: pointer to the parameter structure for the truncation strategy"""
        return int(self._data.param_struct[0])

    @param_struct.setter
    def param_struct(self, val):
        self._data.param_struct = val

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an TruncationStrategy instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `truncation_strategy_dtype` holding the data.
        """
        cdef TruncationStrategy obj = TruncationStrategy.__new__(TruncationStrategy)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != truncation_strategy_dtype:
            raise ValueError("data array must be of dtype truncation_strategy_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False):
        """Create an TruncationStrategy instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef TruncationStrategy obj = TruncationStrategy.__new__(TruncationStrategy)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(cupaulipropTruncationStrategy_t), flag)
        data = _numpy.ndarray((1,), buffer=buf,
                              dtype=truncation_strategy_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


coefficient_truncation_params_dtype = _numpy.dtype([
    ("cutoff", _numpy.float64, ),
    ], align=True)


cdef class CoefficientTruncationParams:
    """Empty-initialize an instance of `cupaulipropCoefficientTruncationParams_t`.


    .. seealso:: `cupaulipropCoefficientTruncationParams_t`
    """
    cdef:
        readonly object _data

    def __init__(self):
        arr = _numpy.empty(1, dtype=coefficient_truncation_params_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(cupaulipropCoefficientTruncationParams_t), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(cupaulipropCoefficientTruncationParams_t)}"

    def __repr__(self):
        return f"<{__name__}.CoefficientTruncationParams object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        return self._data.ctypes.data

    def __eq__(self, other):
        if not isinstance(other, CoefficientTruncationParams):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    @property
    def cutoff(self):
        """float: cutoff value for the magnitude of the Pauli term's coefficient"""
        return float(self._data.cutoff[0])

    @cutoff.setter
    def cutoff(self, val):
        self._data.cutoff = val

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an CoefficientTruncationParams instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `coefficient_truncation_params_dtype` holding the data.
        """
        cdef CoefficientTruncationParams obj = CoefficientTruncationParams.__new__(CoefficientTruncationParams)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != coefficient_truncation_params_dtype:
            raise ValueError("data array must be of dtype coefficient_truncation_params_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False):
        """Create an CoefficientTruncationParams instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef CoefficientTruncationParams obj = CoefficientTruncationParams.__new__(CoefficientTruncationParams)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(cupaulipropCoefficientTruncationParams_t), flag)
        data = _numpy.ndarray((1,), buffer=buf,
                              dtype=coefficient_truncation_params_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


pauli_weight_truncation_params_dtype = _numpy.dtype([
    ("cutoff", _numpy.int32, ),
    ], align=True)


cdef class PauliWeightTruncationParams:
    """Empty-initialize an instance of `cupaulipropPauliWeightTruncationParams_t`.


    .. seealso:: `cupaulipropPauliWeightTruncationParams_t`
    """
    cdef:
        readonly object _data

    def __init__(self):
        arr = _numpy.empty(1, dtype=pauli_weight_truncation_params_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(cupaulipropPauliWeightTruncationParams_t), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(cupaulipropPauliWeightTruncationParams_t)}"

    def __repr__(self):
        return f"<{__name__}.PauliWeightTruncationParams object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        return self._data.ctypes.data

    def __eq__(self, other):
        if not isinstance(other, PauliWeightTruncationParams):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    @property
    def cutoff(self):
        """int: cutoff value for the number of non-identity Paulis in the Pauli string"""
        return int(self._data.cutoff[0])

    @cutoff.setter
    def cutoff(self, val):
        self._data.cutoff = val

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an PauliWeightTruncationParams instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `pauli_weight_truncation_params_dtype` holding the data.
        """
        cdef PauliWeightTruncationParams obj = PauliWeightTruncationParams.__new__(PauliWeightTruncationParams)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != pauli_weight_truncation_params_dtype:
            raise ValueError("data array must be of dtype pauli_weight_truncation_params_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False):
        """Create an PauliWeightTruncationParams instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef PauliWeightTruncationParams obj = PauliWeightTruncationParams.__new__(PauliWeightTruncationParams)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(cupaulipropPauliWeightTruncationParams_t), flag)
        data = _numpy.ndarray((1,), buffer=buf,
                              dtype=pauli_weight_truncation_params_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj



###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """See `cupaulipropStatus_t`."""
    SUCCESS = CUPAULIPROP_STATUS_SUCCESS
    NOT_INITIALIZED = CUPAULIPROP_STATUS_NOT_INITIALIZED
    INVALID_VALUE = CUPAULIPROP_STATUS_INVALID_VALUE
    INTERNAL_ERROR = CUPAULIPROP_STATUS_INTERNAL_ERROR
    NOT_SUPPORTED = CUPAULIPROP_STATUS_NOT_SUPPORTED
    CUDA_ERROR = CUPAULIPROP_STATUS_CUDA_ERROR
    DISTRIBUTED_FAILURE = CUPAULIPROP_STATUS_DISTRIBUTED_FAILURE

class ComputeType(_IntEnum):
    """See `cupaulipropComputeType_t`."""
    COMPUTE_32F = CUPAULIPROP_COMPUTE_32F
    COMPUTE_64F = CUPAULIPROP_COMPUTE_64F

class Memspace(_IntEnum):
    """See `cupaulipropMemspace_t`."""
    DEVICE = CUPAULIPROP_MEMSPACE_DEVICE
    HOST = CUPAULIPROP_MEMSPACE_HOST

class WorkspaceKind(_IntEnum):
    """See `cupaulipropWorkspaceKind_t`."""
    WORKSPACE_SCRATCH = CUPAULIPROP_WORKSPACE_SCRATCH

class TruncationStrategyKind(_IntEnum):
    """See `cupaulipropTruncationStrategyKind_t`."""
    TRUNCATION_STRATEGY_COEFFICIENT_BASED = CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED
    TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED = CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED

class PauliKind(_IntEnum):
    """See `cupaulipropPauliKind_t`."""
    PAULI_I = CUPAULIPROP_PAULI_I
    PAULI_X = CUPAULIPROP_PAULI_X
    PAULI_Z = CUPAULIPROP_PAULI_Z
    PAULI_Y = CUPAULIPROP_PAULI_Y

class CliffordGateKind(_IntEnum):
    """See `cupaulipropCliffordGateKind_t`."""
    CLIFFORD_GATE_I = CUPAULIPROP_CLIFFORD_GATE_I
    CLIFFORD_GATE_X = CUPAULIPROP_CLIFFORD_GATE_X
    CLIFFORD_GATE_Z = CUPAULIPROP_CLIFFORD_GATE_Z
    CLIFFORD_GATE_Y = CUPAULIPROP_CLIFFORD_GATE_Y
    CLIFFORD_GATE_H = CUPAULIPROP_CLIFFORD_GATE_H
    CLIFFORD_GATE_S = CUPAULIPROP_CLIFFORD_GATE_S
    CLIFFORD_GATE_CX = CUPAULIPROP_CLIFFORD_GATE_CX
    CLIFFORD_GATE_CZ = CUPAULIPROP_CLIFFORD_GATE_CZ
    CLIFFORD_GATE_CY = CUPAULIPROP_CLIFFORD_GATE_CY
    CLIFFORD_GATE_SWAP = CUPAULIPROP_CLIFFORD_GATE_SWAP
    CLIFFORD_GATE_ISWAP = CUPAULIPROP_CLIFFORD_GATE_ISWAP
    CLIFFORD_GATE_SQRTX = CUPAULIPROP_CLIFFORD_GATE_SQRTX
    CLIFFORD_GATE_SQRTZ = CUPAULIPROP_CLIFFORD_GATE_SQRTZ
    CLIFFORD_GATE_SQRTY = CUPAULIPROP_CLIFFORD_GATE_SQRTY

class QuantumOperatorKind(_IntEnum):
    """See `cupaulipropQuantumOperatorKind_t`."""
    EXPANSION_KIND_PAULI_ROTATION_GATE = CUPAULIPROP_EXPANSION_KIND_PAULI_ROTATION_GATE
    EXPANSION_KIND_CLIFFORD_GATE = CUPAULIPROP_EXPANSION_KIND_CLIFFORD_GATE
    EXPANSION_KIND_PAULI_NOISE_CHANNEL = CUPAULIPROP_EXPANSION_KIND_PAULI_NOISE_CHANNEL


###############################################################################
# Error handling
###############################################################################

class cuPauliPropError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value}): {get_error_string(status)}"
        super(cuPauliPropError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuPauliPropError(status)


###############################################################################
# Special dtypes
###############################################################################

###############################################################################
# Wrapper functions
###############################################################################

cpdef size_t get_version() except? 0:
    """Returns the semantic version number of the cuPauliProp library.

    .. seealso:: `cupaulipropGetVersion`
    """
    return cupaulipropGetVersion()


cpdef str get_error_string(int error):
    """Returns the description string for an error code.

    Args:
        error (Status): Error code to get the description string for.

    .. seealso:: `cupaulipropGetErrorString`
    """
    cdef bytes _output_
    _output_ = cupaulipropGetErrorString(<_Status>error)
    return _output_.decode()


cpdef int32_t get_num_packed_integers(int32_t num_qubits) except? -1:
    """Returns the number of packed integers of cupaulipropPackedIntegerType_t needed to represent the X bits (or equivalently, the Z bits) of a single Pauli string.

    Args:
        num_qubits (int32_t): Number of qubits.

    Returns:
        int32_t: Number of uint64 integers needed to store X bits (or Z bits) for one Pauli string. To get the total storage requirement, multiply this value by 2.

    .. seealso:: `cupaulipropGetNumPackedIntegers`
    """
    cdef int32_t num_packed_integers
    with nogil:
        status = cupaulipropGetNumPackedIntegers(num_qubits, &num_packed_integers)
    check_status(status)
    return num_packed_integers


cpdef intptr_t create() except? 0:
    """Creates and initializes the library context.

    Returns:
        intptr_t: Library handle.

    .. seealso:: `cupaulipropCreate`
    """
    cdef Handle handle
    with nogil:
        status = cupaulipropCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """Destroys the library context.

    Args:
        handle (intptr_t): Library handle.

    .. seealso:: `cupaulipropDestroy`
    """
    with nogil:
        status = cupaulipropDestroy(<Handle>handle)
    check_status(status)


cpdef set_stream(intptr_t handle, stream):
    """Sets the CUDA stream to be used for library operations.

    Args:
        handle (intptr_t): Library handle.
        stream (object): CUDA stream to be used for library operations. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``intptr_t``.


    .. seealso:: `cupaulipropSetStream`
    """
    cdef nullable_unique_ptr[ vector[intptr_t] ] _stream_
    get_resource_ptr[intptr_t](_stream_, stream, <intptr_t*>NULL)
    with nogil:
        status = cupaulipropSetStream(<Handle>handle, <Stream>(_stream_.data()))
    check_status(status)


cpdef intptr_t create_workspace_descriptor(intptr_t handle) except? 0:
    """Creates a workspace descriptor.

    Args:
        handle (intptr_t): Library handle.

    Returns:
        intptr_t: Workspace descriptor.

    .. seealso:: `cupaulipropCreateWorkspaceDescriptor`
    """
    cdef WorkspaceDescriptor workspace_desc
    with nogil:
        status = cupaulipropCreateWorkspaceDescriptor(<Handle>handle, &workspace_desc)
    check_status(status)
    return <intptr_t>workspace_desc


cpdef destroy_workspace_descriptor(intptr_t workspace_desc):
    """Destroys a workspace descriptor.

    Args:
        workspace_desc (intptr_t): Workspace descriptor.

    .. seealso:: `cupaulipropDestroyWorkspaceDescriptor`
    """
    with nogil:
        status = cupaulipropDestroyWorkspaceDescriptor(<WorkspaceDescriptor>workspace_desc)
    check_status(status)


cpdef int64_t workspace_get_memory_size(intptr_t handle, intptr_t workspace_desc, int mem_space, int workspace_kind) except? -1:
    """Queries the required workspace buffer size.

    Args:
        handle (intptr_t): Library handle.
        workspace_desc (intptr_t): Workspace descriptor.
        mem_space (Memspace): Memory space.
        workspace_kind (WorkspaceKind): Workspace kind.

    Returns:
        int64_t: Required workspace buffer size in bytes.

    .. seealso:: `cupaulipropWorkspaceGetMemorySize`
    """
    cdef int64_t memory_buffer_size
    with nogil:
        status = cupaulipropWorkspaceGetMemorySize(<const Handle>handle, <const WorkspaceDescriptor>workspace_desc, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, &memory_buffer_size)
    check_status(status)
    return memory_buffer_size


cpdef workspace_set_memory(intptr_t handle, intptr_t workspace_desc, int mem_space, int workspace_kind, intptr_t memory_buffer, int64_t memory_buffer_size):
    """Attaches memory to a workspace buffer.

    Args:
        handle (intptr_t): Library handle.
        workspace_desc (intptr_t): Workspace descriptor.
        mem_space (Memspace): Memory space.
        workspace_kind (WorkspaceKind): Workspace kind.
        memory_buffer (intptr_t): Pointer to a user-owned memory buffer to be used by the specified workspace.
        memory_buffer_size (int64_t): Size of the provided memory buffer in bytes.

    .. seealso:: `cupaulipropWorkspaceSetMemory`
    """
    with nogil:
        status = cupaulipropWorkspaceSetMemory(<const Handle>handle, <WorkspaceDescriptor>workspace_desc, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, <void*>memory_buffer, memory_buffer_size)
    check_status(status)


cpdef tuple workspace_get_memory(intptr_t handle, intptr_t workspace_descr, int mem_space, int workspace_kind):
    """Retrieves a workspace buffer.

    Args:
        handle (intptr_t): Library handle.
        workspace_descr (intptr_t): Workspace descriptor.
        mem_space (Memspace): Memory space.
        workspace_kind (WorkspaceKind): Workspace kind.

    Returns:
        A 2-tuple containing:

        - intptr_t: Pointer to a user-owned memory buffer used by the specified workspace.
        - int64_t: Size of the memory buffer in bytes.

    .. seealso:: `cupaulipropWorkspaceGetMemory`
    """
    cdef void* memory_buffer
    cdef int64_t memory_buffer_size
    with nogil:
        status = cupaulipropWorkspaceGetMemory(<const Handle>handle, <const WorkspaceDescriptor>workspace_descr, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, &memory_buffer, &memory_buffer_size)
    check_status(status)
    return (<intptr_t>memory_buffer, memory_buffer_size)


cpdef intptr_t create_pauli_expansion(intptr_t handle, int32_t num_qubits, intptr_t xz_bits_buffer, int64_t xz_bits_buffer_size, intptr_t coef_buffer, int64_t coef_buffer_size, int data_type, int64_t num_terms, int32_t is_sorted, int32_t has_duplicates) except? 0:
    """Creates a Pauli operator expansion.

    Args:
        handle (intptr_t): Library handle.
        num_qubits (int32_t): Number of qubits.
        xz_bits_buffer (intptr_t): Pointer to a user-owned memory buffer to be used by the created Pauli operator expansion for storing the X and Z bits for each Pauli operator term. The first ``num_terms`` Pauli operator terms will define the current Pauli operator expansion.
        xz_bits_buffer_size (int64_t): Size (in bytes) of the provided memory buffer for storing the X and Z bits.
        coef_buffer (intptr_t): Pointer to a user-owned memory buffer to be used by the created Pauli operator expansion for storing the coefficients for all Pauli operator terms. The first ``num_terms`` Pauli operator terms will define the current Pauli operator expansion.
        coef_buffer_size (int64_t): Size (in bytes) of the provided memory buffer for storing the coefficients.
        data_type (int): Data type of the coefficients in the Pauli operator expansion.
        num_terms (int64_t): Number of the Pauli operator terms stored in the provided memory buffer (the first ``num_terms`` components define the current Pauli operator expansion).
        is_sorted (int32_t): Whether or not the Pauli expansion is sorted. A sorted expansion has its Pauli operator terms sorted by the X and Z bits in ascending order (interpreted as big integers in little-endian representation).
        has_duplicates (int32_t): Whether or not there are duplicates in the expansion, i.e. several terms with identical X and Z bits.

    Returns:
        intptr_t: Pauli operator expansion.

    .. seealso:: `cupaulipropCreatePauliExpansion`
    """
    cdef PauliExpansion pauli_expansion
    with nogil:
        status = cupaulipropCreatePauliExpansion(<const Handle>handle, num_qubits, <void*>xz_bits_buffer, xz_bits_buffer_size, <void*>coef_buffer, coef_buffer_size, <DataType>data_type, num_terms, is_sorted, has_duplicates, &pauli_expansion)
    check_status(status)
    return <intptr_t>pauli_expansion


cpdef destroy_pauli_expansion(intptr_t pauli_expansion):
    """Destroys a Pauli operator expansion.

    Args:
        pauli_expansion (intptr_t): Pauli operator expansion.

    .. seealso:: `cupaulipropDestroyPauliExpansion`
    """
    with nogil:
        status = cupaulipropDestroyPauliExpansion(<PauliExpansion>pauli_expansion)
    check_status(status)


cpdef tuple pauli_expansion_get_storage_buffer(intptr_t handle, intptr_t pauli_expansion):
    """Gets access to the storage of a Pauli operator expansion.

    Args:
        handle (intptr_t): Library handle.
        pauli_expansion (intptr_t): Pauli operator expansion.

    Returns:
        A 6-tuple containing:

        - intptr_t: Pointer to a user-owned memory buffer used by the Pauli operator expansion for storing the X and Z bits for each Pauli operator term.
        - int64_t: Size (in bytes) of the memory buffer for X and Z bits.
        - intptr_t: Pointer to a user-owned memory buffer used by the Pauli operator expansion for storing the coefficients for each Pauli operator term.
        - int64_t: Size (in bytes) of the memory buffer for storing the coefficients.
        - int64_t: Current number of Pauli operator terms in the Pauli operator expansion (first ``numTerms`` terms define the current Pauli operator expansion).
        - int: Storage location of the Pauli operator expansion (whether it is on the host or device).

    .. seealso:: `cupaulipropPauliExpansionGetStorageBuffer`
    """
    cdef void* xz_bits_buffer
    cdef int64_t xz_bits_buffer_size
    cdef void* coef_buffer
    cdef int64_t coef_buffer_size
    cdef int64_t num_terms
    cdef _Memspace location
    with nogil:
        status = cupaulipropPauliExpansionGetStorageBuffer(<const Handle>handle, <const PauliExpansion>pauli_expansion, &xz_bits_buffer, &xz_bits_buffer_size, &coef_buffer, &coef_buffer_size, &num_terms, &location)
    check_status(status)
    return (<intptr_t>xz_bits_buffer, xz_bits_buffer_size, <intptr_t>coef_buffer, coef_buffer_size, num_terms, <int>location)


cpdef int32_t pauli_expansion_get_num_qubits(intptr_t handle, intptr_t pauli_expansion) except? -1:
    """Gets the number of qubits of a Pauli operator expansion.

    Args:
        handle (intptr_t): Library handle.
        pauli_expansion (intptr_t): Pauli operator expansion.

    Returns:
        int32_t: Number of qubits.

    .. seealso:: `cupaulipropPauliExpansionGetNumQubits`
    """
    cdef int32_t num_qubits
    with nogil:
        status = cupaulipropPauliExpansionGetNumQubits(<const Handle>handle, <const PauliExpansion>pauli_expansion, &num_qubits)
    check_status(status)
    return num_qubits


cpdef int64_t pauli_expansion_get_num_terms(intptr_t handle, intptr_t pauli_expansion) except? -1:
    """Gets the number of terms in the Pauli operator expansion.

    Args:
        handle (intptr_t): Library handle.
        pauli_expansion (intptr_t): Pauli operator expansion.

    Returns:
        int64_t: Number of terms.

    .. seealso:: `cupaulipropPauliExpansionGetNumTerms`
    """
    cdef int64_t num_terms
    with nogil:
        status = cupaulipropPauliExpansionGetNumTerms(<const Handle>handle, <const PauliExpansion>pauli_expansion, &num_terms)
    check_status(status)
    return num_terms


cpdef int pauli_expansion_get_data_type(intptr_t handle, intptr_t pauli_expansion) except? -1:
    """Gets the data type of the coefficients in a Pauli operator expansion.

    Args:
        handle (intptr_t): Library handle.
        pauli_expansion (intptr_t): Pauli operator expansion.

    Returns:
        int: Data type.

    .. seealso:: `cupaulipropPauliExpansionGetDataType`
    """
    cdef DataType data_type
    with nogil:
        status = cupaulipropPauliExpansionGetDataType(<const Handle>handle, <const PauliExpansion>pauli_expansion, &data_type)
    check_status(status)
    return <int>data_type


cpdef int32_t pauli_expansion_is_sorted(intptr_t handle, intptr_t pauli_expansion) except? -1:
    """Queries whether a Pauli operator expansion is sorted or not.

    Args:
        handle (intptr_t): Library handle.
        pauli_expansion (intptr_t): Pauli operator expansion.

    Returns:
        int32_t: Indicating whether the Pauli operator expansion is sorted. Sortedness is defined as the Pauli strings being sorted in ascending order according to the little-endian representation of the big integers formed by the X and Z bits. True (!= 0) if the Pauli operator expansion is sorted, false (0) otherwise.

    .. seealso:: `cupaulipropPauliExpansionIsSorted`
    """
    cdef int32_t is_sorted
    with nogil:
        status = cupaulipropPauliExpansionIsSorted(<const Handle>handle, <const PauliExpansion>pauli_expansion, &is_sorted)
    check_status(status)
    return is_sorted


cpdef int32_t pauli_expansion_is_deduplicated(intptr_t handle, intptr_t pauli_expansion) except? -1:
    """Queries whether a Pauli operator expansion is deduplicated. i.e. guaranteed to not contain duplicate Pauli strings or may otherwise potentially contain duplicates Pauli strings.

    Args:
        handle (intptr_t): Library handle.
        pauli_expansion (intptr_t): Pauli operator expansion.

    Returns:
        int32_t: Indicating whether the Pauli operator expansion is deduplicated. True (!= 0) if the Pauli operator expansion is guaranteed to not contain duplicate Pauli strings, false (0) if no such guarantee can be made (though it may be incidentally the case).

    .. seealso:: `cupaulipropPauliExpansionIsDeduplicated`
    """
    cdef int32_t is_deduplicated
    with nogil:
        status = cupaulipropPauliExpansionIsDeduplicated(<const Handle>handle, <const PauliExpansion>pauli_expansion, &is_deduplicated)
    check_status(status)
    return is_deduplicated


cpdef intptr_t pauli_expansion_get_contiguous_range(intptr_t handle, intptr_t pauli_expansion, int64_t start_ind_ex, int64_t end_ind_ex) except? 0:
    """Creates a non-owning view of a contiguous range of Pauli operator terms inside a Pauli operator expansion.

    Args:
        handle (intptr_t): Library handle.
        pauli_expansion (intptr_t): Pauli operator expansion.
        start_ind_ex (int64_t): Start index of the range (inclusive, first element in the range).
        end_ind_ex (int64_t): End index of the range (exclusive, one past the last element).

    Returns:
        intptr_t: View to a range of Pauli terms inside the Pauli operator expansion.

    .. seealso:: `cupaulipropPauliExpansionGetContiguousRange`
    """
    cdef PauliExpansionView view
    with nogil:
        status = cupaulipropPauliExpansionGetContiguousRange(<const Handle>handle, <const PauliExpansion>pauli_expansion, start_ind_ex, end_ind_ex, &view)
    check_status(status)
    return <intptr_t>view


cpdef destroy_pauli_expansion_view(intptr_t view):
    """Destroys a Pauli expansion view.

    Args:
        view (intptr_t): Pauli expansion view.

    .. seealso:: `cupaulipropDestroyPauliExpansionView`
    """
    with nogil:
        status = cupaulipropDestroyPauliExpansionView(<PauliExpansionView>view)
    check_status(status)


cpdef int64_t pauli_expansion_view_get_num_terms(intptr_t handle, intptr_t view) except? -1:
    """Returns the number of Pauli terms in a Pauli expansion view.

    Args:
        handle (intptr_t): Library handle.
        view (intptr_t): Pauli expansion view.

    Returns:
        int64_t: Number of terms.

    .. seealso:: `cupaulipropPauliExpansionViewGetNumTerms`
    """
    cdef int64_t num_terms
    with nogil:
        status = cupaulipropPauliExpansionViewGetNumTerms(<const Handle>handle, <const PauliExpansionView>view, &num_terms)
    check_status(status)
    return num_terms


cpdef int pauli_expansion_view_get_location(intptr_t view) except? -1:
    """Gets the storage location of a Pauli expansion view (whether its elements are stored on the host or device).

    Args:
        view (intptr_t): Pauli expansion view.

    Returns:
        int: Location.

    .. seealso:: `cupaulipropPauliExpansionViewGetLocation`
    """
    cdef _Memspace location
    with nogil:
        status = cupaulipropPauliExpansionViewGetLocation(<const PauliExpansionView>view, &location)
    check_status(status)
    return <int>location


cpdef pauli_expansion_view_prepare_deduplication(intptr_t handle, intptr_t view_in, int32_t make_sorted, int64_t max_workspace_size, intptr_t workspace):
    """Prepares a Pauli expansion view for deduplication.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to be deduplicated.
        make_sorted (int32_t): Whether or not the output expansion is required to be sorted.
        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor with the required workspace buffer size.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareDeduplication`
    """
    with nogil:
        status = cupaulipropPauliExpansionViewPrepareDeduplication(<const Handle>handle, <const PauliExpansionView>view_in, make_sorted, max_workspace_size, <WorkspaceDescriptor>workspace)
    check_status(status)


cpdef pauli_expansion_view_execute_deduplication(intptr_t handle, intptr_t view_in, intptr_t expansion_out, int32_t make_sorted, intptr_t workspace):
    """Deduplicates a Pauli expansion view.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to be deduplicated.
        expansion_out (intptr_t): Pauli expansion to be populated with the deduplicated view.
        make_sorted (int32_t): Whether or not the output expansion is required to be sorted.
        workspace (intptr_t): Allocated workspace descriptor.

    .. seealso:: `cupaulipropPauliExpansionViewExecuteDeduplication`
    """
    with nogil:
        status = cupaulipropPauliExpansionViewExecuteDeduplication(<const Handle>handle, <const PauliExpansionView>view_in, <PauliExpansion>expansion_out, make_sorted, <WorkspaceDescriptor>workspace)
    check_status(status)


cpdef pauli_expansion_view_prepare_canonical_sort(intptr_t handle, intptr_t view_in, int64_t max_workspace_size, intptr_t workspace):
    """Prepares a Pauli expansion view for canonical sorting.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to be sorted.
        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor with the required workspace buffer size.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareCanonicalSort`
    """
    with nogil:
        status = cupaulipropPauliExpansionViewPrepareCanonicalSort(<const Handle>handle, <const PauliExpansionView>view_in, max_workspace_size, <WorkspaceDescriptor>workspace)
    check_status(status)


cpdef pauli_expansion_view_execute_canonical_sort(intptr_t handle, intptr_t view_in, intptr_t expansion_out, intptr_t workspace):
    """Sorts a Pauli expansion view canonically.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to be sorted.
        expansion_out (intptr_t): Pauli expansion to be populated with the sorted view.
        workspace (intptr_t): Allocated workspace descriptor.

    .. seealso:: `cupaulipropPauliExpansionViewExecuteCanonicalSort`
    """
    with nogil:
        status = cupaulipropPauliExpansionViewExecuteCanonicalSort(<const Handle>handle, <const PauliExpansionView>view_in, <PauliExpansion>expansion_out, <WorkspaceDescriptor>workspace)
    check_status(status)


cpdef pauli_expansion_populate_from_view(intptr_t handle, intptr_t view_in, intptr_t expansion_out):
    """Populates a Pauli operator expansion from a Pauli expansion view.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Input Pauli expansion view.
        expansion_out (intptr_t): Populated Pauli operator expansion.

    .. seealso:: `cupaulipropPauliExpansionPopulateFromView`
    """
    with nogil:
        status = cupaulipropPauliExpansionPopulateFromView(<const Handle>handle, <const PauliExpansionView>view_in, <PauliExpansion>expansion_out)
    check_status(status)


cpdef pauli_expansion_view_prepare_trace_with_expansion_view(intptr_t handle, intptr_t view1, intptr_t view2, int64_t max_workspace_size, intptr_t workspace):
    """Prepares a Pauli expansion view for computing the product trace of two Pauli expansion views.

    Args:
        handle (intptr_t): Library handle.
        view1 (intptr_t): First Pauli expansion view to be traced.
        view2 (intptr_t): Second Pauli expansion view to be traced.
        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor with the required workspace buffer size.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareTraceWithExpansionView`
    """
    with nogil:
        status = cupaulipropPauliExpansionViewPrepareTraceWithExpansionView(<const Handle>handle, <const PauliExpansionView>view1, <const PauliExpansionView>view2, max_workspace_size, <WorkspaceDescriptor>workspace)
    check_status(status)


cpdef pauli_expansion_view_compute_trace_with_expansion_view(intptr_t handle, intptr_t view1, intptr_t view2, int32_t take_adjoint1, intptr_t trace, intptr_t workspace):
    """Computes the trace of two Pauli expansion views.

    Args:
        handle (intptr_t): Library handle.
        view1 (intptr_t): First Pauli expansion view.
        view2 (intptr_t): Second Pauli expansion view.
        take_adjoint1 (int32_t): Whether or not the adjoint of the first view is taken. True ``(!= 0)`` if the adjoint is taken, false ``(0)`` otherwise.
        trace (intptr_t): Pointer to CPU-accessible memory where the trace value will be written. The numerical type must match the data type of the views' coefficients.
        workspace (intptr_t): Allocated workspace descriptor.

    .. seealso:: `cupaulipropPauliExpansionViewComputeTraceWithExpansionView`
    """
    with nogil:
        status = cupaulipropPauliExpansionViewComputeTraceWithExpansionView(<const Handle>handle, <const PauliExpansionView>view1, <const PauliExpansionView>view2, take_adjoint1, <void*>trace, <WorkspaceDescriptor>workspace)
    check_status(status)


cpdef pauli_expansion_view_prepare_trace_with_zero_state(intptr_t handle, intptr_t view, int64_t max_workspace_size, intptr_t workspace):
    """Prepares a Pauli expansion view for tracing with the zero state, i.e. computing ``Tr(view * |0...0⟩)`` .

    Args:
        handle (intptr_t): Library handle.
        view (intptr_t): Pauli expansion view to be traced.
        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor with the required workspace buffer size.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareTraceWithZeroState`
    """
    with nogil:
        status = cupaulipropPauliExpansionViewPrepareTraceWithZeroState(<const Handle>handle, <const PauliExpansionView>view, max_workspace_size, <WorkspaceDescriptor>workspace)
    check_status(status)


cpdef pauli_expansion_view_compute_trace_with_zero_state(intptr_t handle, intptr_t view, intptr_t trace, intptr_t workspace):
    """Traces a Pauli expansion view with the zero state, i.e. computes ``Tr(view * |0...0⟩)`` .

    Args:
        handle (intptr_t): Library handle.
        view (intptr_t): Pauli expansion view to be traced.
        trace (intptr_t): Pointer to CPU-accessible memory where the trace value will be written. The numerical type must match the data type of the views' coefficients.
        workspace (intptr_t): Allocated workspace descriptor.

    .. seealso:: `cupaulipropPauliExpansionViewComputeTraceWithZeroState`
    """
    with nogil:
        status = cupaulipropPauliExpansionViewComputeTraceWithZeroState(<const Handle>handle, <const PauliExpansionView>view, <void*>trace, <WorkspaceDescriptor>workspace)
    check_status(status)


cpdef intptr_t create_clifford_gate_operator(intptr_t handle, int clifford_gate_kind, qubit_indices) except? 0:
    """Creates a Clifford gate.

    Args:
        handle (intptr_t): Library handle.
        clifford_gate_kind (CliffordGateKind): Clifford gate kind.
        qubit_indices (object): Qubit indices. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.


    Returns:
        intptr_t: Quantum operator associated with the Clifford gate.

    .. seealso:: `cupaulipropCreateCliffordGateOperator`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _qubit_indices_
    get_resource_ptr[int32_t](_qubit_indices_, qubit_indices, <int32_t*>NULL)
    cdef QuantumOperator oper
    with nogil:
        status = cupaulipropCreateCliffordGateOperator(<const Handle>handle, <_CliffordGateKind>clifford_gate_kind, <const int32_t*>(_qubit_indices_.data()), &oper)
    check_status(status)
    return <intptr_t>oper


cpdef intptr_t create_pauli_rotation_gate_operator(intptr_t handle, double angle, int32_t num_qubits, qubit_indices, paulis) except? 0:
    """Creates a Pauli rotation gate, ``exp(-i * angle/2 * P)``, for a rotation of ``angle`` around the Pauli string ``P``.

    Args:
        handle (intptr_t): Library handle.
        angle (double): Rotation angle in radians.
        num_qubits (int32_t): Number of qubits.
        qubit_indices (object): Qubit indices. If NULL, the qubit indices are assumed to be [0, 1, 2, ..., num_qubits-1]. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        paulis (object): Pauli operators for each qubit index. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``_PauliKind``.


    Returns:
        intptr_t: Quantum operator associated with the Pauli rotation gate.

    .. seealso:: `cupaulipropCreatePauliRotationGateOperator`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _qubit_indices_
    get_resource_ptr[int32_t](_qubit_indices_, qubit_indices, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _paulis_
    get_resource_ptr[int](_paulis_, paulis, <int*>NULL)
    cdef QuantumOperator oper
    with nogil:
        status = cupaulipropCreatePauliRotationGateOperator(<const Handle>handle, angle, num_qubits, <const int32_t*>(_qubit_indices_.data()), <const _PauliKind*>(_paulis_.data()), &oper)
    check_status(status)
    return <intptr_t>oper


cpdef intptr_t create_pauli_noise_channel_operator(intptr_t handle, int32_t num_qubits, qubit_indices, probabilities) except? 0:
    """Creates a Pauli noise channel.

    Args:
        handle (intptr_t): Library handle.
        num_qubits (int32_t): Number of qubits. Only 1 and 2 qubits are supported.
        qubit_indices (object): Qubit indices. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        probabilities (object): Probabilities for each Pauli channel. For a single qubit Pauli Channel, the probabilities are an array of length 4: ``PauliKind((i)%4)`` (i.e. ``[p_I, p_X, p_Y, p_Z]``). For a two qubit Pauli Channel, probabibilities is an array of length 16. The i-th element of the probabilities is associated with the i-th element of the 2-qubit Pauli strings in lexographic order. E.g. prob[i] corresponds to the Pauli string ``PauliKind((i)%4), PauliKind_t((i)/4)``. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``float``.


    Returns:
        intptr_t: Quantum operator associated with the Pauli channel.

    .. seealso:: `cupaulipropCreatePauliNoiseChannelOperator`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _qubit_indices_
    get_resource_ptr[int32_t](_qubit_indices_, qubit_indices, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[double] ] _probabilities_
    get_resource_ptr[double](_probabilities_, probabilities, <double*>NULL)
    cdef QuantumOperator oper
    with nogil:
        status = cupaulipropCreatePauliNoiseChannelOperator(<const Handle>handle, num_qubits, <const int32_t*>(_qubit_indices_.data()), <const double*>(_probabilities_.data()), &oper)
    check_status(status)
    return <intptr_t>oper


cpdef int quantum_operator_get_kind(intptr_t handle, intptr_t oper) except? -1:
    """Queries what kind of gate or channel a quantum operator represents.

    Args:
        handle (intptr_t): Library handle.
        oper (intptr_t): Quantum operator.

    Returns:
        int: Kind of the quantum operator.

    .. seealso:: `cupaulipropQuantumOperatorGetKind`
    """
    cdef _QuantumOperatorKind kind
    with nogil:
        status = cupaulipropQuantumOperatorGetKind(<const Handle>handle, <const QuantumOperator>oper, &kind)
    check_status(status)
    return <int>kind


cpdef destroy_operator(intptr_t oper):
    """Destroys a quantum operator.

    Args:
        oper (intptr_t): Quantum operator.

    .. seealso:: `cupaulipropDestroyOperator`
    """
    with nogil:
        status = cupaulipropDestroyOperator(<QuantumOperator>oper)
    check_status(status)


# Custom implementations for truncation strategy functions (not auto-generated)
cpdef tuple pauli_expansion_view_prepare_operator_application(intptr_t handle, intptr_t view_in, intptr_t quantum_operator, int32_t make_sorted, int32_t keep_duplicates, int32_t num_truncation_strategies, truncation_strategies, int64_t max_workspace_size, intptr_t workspace):
    """Prepares a Pauli expansion view for quantum operator application.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to apply a quantum operator to.
        quantum_operator (intptr_t): Quantum operator to be applied.
        make_sorted (int32_t): Whether or not the output expansion is required to be sorted.
        keep_duplicates (int32_t): Whether or not the output expansion is allowed to contain duplicates.
        num_truncation_strategies (int32_t): Number of Pauli expansion truncation strategies.
        truncation_strategies (object): Pauli expansion truncation strategies. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cupaulipropTruncationStrategy_t``.

        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor with the required workspace buffer size.

    Returns:
        A 2-tuple containing:

        - int64_t: Required size (in bytes) of the X and Z bits output buffer.
        - int64_t: Required size (in bytes) of the coefficients output buffer.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareOperatorApplication`
    """
    cdef int64_t required_xz_bits_buffer_size
    cdef int64_t required_coef_buffer_size
    cdef vector[cupaulipropTruncationStrategy_t] _trunc_vec
    cdef const cupaulipropTruncationStrategy_t* _trunc_ptr = NULL
    cdef intptr_t ptr_val
    
    # Handle truncation_strategies: Python sequence of TruncationStrategy objects
    if truncation_strategies is None or num_truncation_strategies == 0:
        _trunc_ptr = NULL
    elif cpython.PySequence_Check(truncation_strategies):
        # Build vector of structs from Python sequence of TruncationStrategy objects
        for i in range(len(truncation_strategies)):
            # Get pointer from TruncationStrategy object and dereference to copy the struct
            ptr_val = <intptr_t><size_t>int(truncation_strategies[i].ptr)
            _trunc_vec.push_back((<cupaulipropTruncationStrategy_t*>ptr_val)[0])
        _trunc_ptr = <const cupaulipropTruncationStrategy_t*>(_trunc_vec.data())
    else:
        # For advanced users: accept a raw pointer to a pre-built array of structs
        _trunc_ptr = <const cupaulipropTruncationStrategy_t*><intptr_t>truncation_strategies
    
    with nogil:
        status = cupaulipropPauliExpansionViewPrepareOperatorApplication(<const Handle>handle, <const PauliExpansionView>view_in, <const QuantumOperator>quantum_operator, make_sorted, keep_duplicates, num_truncation_strategies, _trunc_ptr, max_workspace_size, &required_xz_bits_buffer_size, &required_coef_buffer_size, <WorkspaceDescriptor>workspace)
    check_status(status)
    return (required_xz_bits_buffer_size, required_coef_buffer_size)


cpdef pauli_expansion_view_compute_operator_application(intptr_t handle, intptr_t view_in, intptr_t expansion_out, intptr_t quantum_operator, int32_t adjoint, int32_t make_sorted, int32_t keep_duplicates, int32_t num_truncation_strategies, truncation_strategies, intptr_t workspace):
    """Computes the application of a quantum operator to a Pauli expansion view.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to apply a quantum operator to.
        expansion_out (intptr_t): Pauli expansion to be overwritten with the result. The output expansion will satisfy sortedness and uniqueness respectively if these flags have been set to true when creating or resetting the output expansion. Otherwise they may or may not be satisfied. Their state is queryable on the output expansion after this function call via ``cupaulipropPauliExpansionGetSortedness()`` and ``cupaulipropPauliExpansionGetUniqueness()``.
        quantum_operator (intptr_t): Quantum operator to be applied.
        adjoint (int32_t): Whether or not the adjoint of the quantum operator is applied. True (!= 0) if the adjoint is applied, false (0) otherwise.
        make_sorted (int32_t): Whether or not the output expansion is required to be sorted.
        keep_duplicates (int32_t): Whether or not the output expansion is allowed to contain duplicates.
        num_truncation_strategies (int32_t): Number of Pauli expansion truncation strategies.
        truncation_strategies (object): Pauli expansion truncation strategies. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cupaulipropTruncationStrategy_t``.

        workspace (intptr_t): Allocated workspace descriptor.

    .. seealso:: `cupaulipropPauliExpansionViewComputeOperatorApplication`
    """
    cdef vector[cupaulipropTruncationStrategy_t] _trunc_vec
    cdef const cupaulipropTruncationStrategy_t* _trunc_ptr = NULL
    cdef intptr_t ptr_val
    
    # Handle truncation_strategies: Python sequence of TruncationStrategy objects
    if truncation_strategies is None or num_truncation_strategies == 0:
        _trunc_ptr = NULL
    elif cpython.PySequence_Check(truncation_strategies):
        # Build vector of structs from Python sequence of TruncationStrategy objects
        for i in range(len(truncation_strategies)):
            # Get pointer from TruncationStrategy object and dereference to copy the struct
            ptr_val = <intptr_t><size_t>int(truncation_strategies[i].ptr)
            _trunc_vec.push_back((<cupaulipropTruncationStrategy_t*>ptr_val)[0])
        _trunc_ptr = <const cupaulipropTruncationStrategy_t*>(_trunc_vec.data())
    else:
        # For advanced users: accept a raw pointer to a pre-built array of structs
        _trunc_ptr = <const cupaulipropTruncationStrategy_t*><intptr_t>truncation_strategies
    
    with nogil:
        status = cupaulipropPauliExpansionViewComputeOperatorApplication(<const Handle>handle, <const PauliExpansionView>view_in, <PauliExpansion>expansion_out, <const QuantumOperator>quantum_operator, adjoint, make_sorted, keep_duplicates, num_truncation_strategies, _trunc_ptr, <WorkspaceDescriptor>workspace)
    check_status(status)


cpdef pauli_expansion_view_prepare_truncation(intptr_t handle, intptr_t view_in, int32_t num_truncation_strategies, truncation_strategies, int64_t max_workspace_size, intptr_t workspace):
    """Prepares a Pauli expansion view for truncation.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to be truncated.
        num_truncation_strategies (int32_t): Number of Pauli expansion truncation strategies.
        truncation_strategies (object): Pauli expansion truncation strategies. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cupaulipropTruncationStrategy_t``.

        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor with the required workspace buffer size.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareTruncation`
    """
    cdef vector[cupaulipropTruncationStrategy_t] _trunc_vec
    cdef const cupaulipropTruncationStrategy_t* _trunc_ptr = NULL
    cdef intptr_t ptr_val
    
    # Handle truncation_strategies: Python sequence of TruncationStrategy objects
    if truncation_strategies is None or num_truncation_strategies == 0:
        _trunc_ptr = NULL
    elif cpython.PySequence_Check(truncation_strategies):
        # Build vector of structs from Python sequence of TruncationStrategy objects
        for i in range(len(truncation_strategies)):
            # Get pointer from TruncationStrategy object and dereference to copy the struct
            ptr_val = <intptr_t><size_t>int(truncation_strategies[i].ptr)
            _trunc_vec.push_back((<cupaulipropTruncationStrategy_t*>ptr_val)[0])
        _trunc_ptr = <const cupaulipropTruncationStrategy_t*>(_trunc_vec.data())
    else:
        # For advanced users: accept a raw pointer to a pre-built array of structs
        _trunc_ptr = <const cupaulipropTruncationStrategy_t*><intptr_t>truncation_strategies
    
    with nogil:
        status = cupaulipropPauliExpansionViewPrepareTruncation(<const Handle>handle, <const PauliExpansionView>view_in, num_truncation_strategies, _trunc_ptr, max_workspace_size, <WorkspaceDescriptor>workspace)
    check_status(status)


cpdef pauli_expansion_view_execute_truncation(intptr_t handle, intptr_t view_in, intptr_t expansion_out, int32_t num_truncation_strategies, truncation_strategies, intptr_t workspace):
    """Truncates a Pauli expansion view.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Input Pauli expansion view to be truncated.
        expansion_out (intptr_t): Output Pauli operator expansion.
        num_truncation_strategies (int32_t): Number of Pauli expansion truncation strategies.
        truncation_strategies (object): Pauli expansion truncation strategies. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cupaulipropTruncationStrategy_t``.

        workspace (intptr_t): Allocated workspace descriptor.

    .. seealso:: `cupaulipropPauliExpansionViewExecuteTruncation`
    """
    cdef vector[cupaulipropTruncationStrategy_t] _trunc_vec
    cdef const cupaulipropTruncationStrategy_t* _trunc_ptr = NULL
    cdef intptr_t ptr_val
    
    # Handle truncation_strategies: Python sequence of TruncationStrategy objects
    if truncation_strategies is None or num_truncation_strategies == 0:
        _trunc_ptr = NULL
    elif cpython.PySequence_Check(truncation_strategies):
        # Build vector of structs from Python sequence of TruncationStrategy objects
        for i in range(len(truncation_strategies)):
            # Get pointer from TruncationStrategy object and dereference to copy the struct
            ptr_val = <intptr_t><size_t>int(truncation_strategies[i].ptr)
            _trunc_vec.push_back((<cupaulipropTruncationStrategy_t*>ptr_val)[0])
        _trunc_ptr = <const cupaulipropTruncationStrategy_t*>(_trunc_vec.data())
    else:
        # For advanced users: accept a raw pointer to a pre-built array of structs
        _trunc_ptr = <const cupaulipropTruncationStrategy_t*><intptr_t>truncation_strategies
    
    with nogil:
        status = cupaulipropPauliExpansionViewExecuteTruncation(<const Handle>handle, <const PauliExpansionView>view_in, <PauliExpansion>expansion_out, num_truncation_strategies, _trunc_ptr, <WorkspaceDescriptor>workspace)
    check_status(status)


cpdef PauliTerm pauli_expansion_get_term(intptr_t handle, intptr_t pauli_expansion, uint64_t term_index):
    """Gets access to a specific term of a Pauli operator expansion.

    Args:
        handle (intptr_t): Library handle.
        pauli_expansion (intptr_t): Pauli operator expansion.
        term_index (uint64_t): Index of the term.

    Returns:
        PauliTerm: Pauli operator term.

    .. seealso:: `cupaulipropPauliExpansionGetTerm`
    """
    cdef PauliTerm term = PauliTerm()
    cdef intptr_t term_ptr = term.ptr
    with nogil:
        status = cupaulipropPauliExpansionGetTerm(<const Handle>handle, <const PauliExpansion>pauli_expansion, term_index, <cupaulipropPauliTerm_t*>term_ptr)
    check_status(status)
    return term


cpdef PauliTerm pauli_expansion_view_get_term(intptr_t handle, intptr_t view, uint64_t term_index):
    """Gets a specific term of a Pauli expansion view.

    Args:
        handle (intptr_t): Library handle.
        view (intptr_t): Pauli expansion view.
        term_index (uint64_t): Index of the term in the Pauli expansion view.

    Returns:
        PauliTerm: Pauli operator term.

    .. seealso:: `cupaulipropPauliExpansionViewGetTerm`
    """
    cdef PauliTerm term = PauliTerm()
    cdef intptr_t term_ptr = term.ptr
    with nogil:
        status = cupaulipropPauliExpansionViewGetTerm(<const Handle>handle, <const PauliExpansionView>view, term_index, <cupaulipropPauliTerm_t*>term_ptr)
    check_status(status)
    return term

ALLOCATOR_NAME_LEN = CUPAULIPROP_ALLOCATOR_NAME_LEN

######################### Python specific utility #########################

# who owns a reference to user-provided Python objects (k: owner, v: object)
cdef dict owner_pyobj = {}
