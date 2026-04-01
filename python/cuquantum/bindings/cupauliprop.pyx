# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 25.11.0 to 26.03.0, generator version 0.3.1.dev1375+gca9bf77db.d20260310. Do not modify it directly.

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

from libc.stdlib cimport calloc, free, malloc
from cython cimport view
cimport cpython.buffer
cimport cpython.memoryview
cimport cpython
from libc.string cimport memcmp, memcpy
import numpy as _numpy


cdef __from_data(data, dtype_name, expected_dtype, lowpp_type):
    # _numpy.recarray is a subclass of _numpy.ndarray, so implicitly handled here.
    if isinstance(data, lowpp_type):
        return data
    if not isinstance(data, _numpy.ndarray):
        raise TypeError("data argument must be a NumPy ndarray")
    if data.size != 1:
        raise ValueError("data array must have a size of 1")
    if data.dtype != expected_dtype:
        raise ValueError(f"data array must be of dtype {dtype_name}")
    return lowpp_type.from_ptr(data.ctypes.data, not data.flags.writeable, data)


cdef __from_buffer(buffer, size, lowpp_type):
    cdef Py_buffer view
    if cpython.PyObject_GetBuffer(buffer, &view, cpython.PyBUF_SIMPLE) != 0:
        raise TypeError("buffer argument does not support the buffer protocol")
    try:
        if view.itemsize != 1:
            raise ValueError("buffer itemsize must be 1 byte")
        if view.len != size:
            raise ValueError(f"buffer length must be {size} bytes")
        return lowpp_type.from_ptr(<intptr_t><void *>view.buf, not view.readonly, buffer)
    finally:
        cpython.PyBuffer_Release(&view)


cdef __getbuffer(object self, cpython.Py_buffer *buffer, void *ptr, int size, bint readonly):
    buffer.buf = <char *>ptr
    buffer.format = 'b'
    buffer.internal = NULL
    buffer.itemsize = 1
    buffer.len = size
    buffer.ndim = 1
    buffer.obj = self
    buffer.readonly = readonly
    buffer.shape = &buffer.len
    buffer.strides = &buffer.itemsize
    buffer.suboffsets = NULL


###############################################################################
# POD
###############################################################################

cdef _get_pauli_term_dtype_offsets():
    cdef cupaulipropPauliTerm_t pod = cupaulipropPauliTerm_t()
    return _numpy.dtype({
        'names': ['xzbits', 'coef'],
        'formats': [_numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.xzbits)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.coef)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(cupaulipropPauliTerm_t),
    })

pauli_term_dtype = _get_pauli_term_dtype_offsets()

cdef class PauliTerm:
    """Empty-initialize an instance of `cupaulipropPauliTerm_t`.


    .. seealso:: `cupaulipropPauliTerm_t`
    """
    cdef:
        cupaulipropPauliTerm_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <cupaulipropPauliTerm_t *>calloc(1, sizeof(cupaulipropPauliTerm_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating PauliTerm")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef cupaulipropPauliTerm_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.PauliTerm object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef PauliTerm other_
        if not isinstance(other, PauliTerm):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cupaulipropPauliTerm_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cupaulipropPauliTerm_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cupaulipropPauliTerm_t *>malloc(sizeof(cupaulipropPauliTerm_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating PauliTerm")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cupaulipropPauliTerm_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def xzbits(self):
        """int: non-owning pointer to the packed X and Z bits arranged as an array of `cupaulipropPackedIntegerType_t`"""
        return <intptr_t>(self._ptr[0].xzbits)

    @xzbits.setter
    def xzbits(self, val):
        if self._readonly:
            raise ValueError("This PauliTerm instance is read-only")
        self._ptr[0].xzbits = <cupaulipropPackedIntegerType_t*><intptr_t>val

    @property
    def coef(self):
        """int: non-owning pointer to the coefficient (real/complex float/double)"""
        return <intptr_t>(self._ptr[0].coef)

    @coef.setter
    def coef(self, val):
        if self._readonly:
            raise ValueError("This PauliTerm instance is read-only")
        self._ptr[0].coef = <void *><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an PauliTerm instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cupaulipropPauliTerm_t), PauliTerm)

    @staticmethod
    def from_data(data):
        """Create an PauliTerm instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `pauli_term_dtype` holding the data.
        """
        return __from_data(data, "pauli_term_dtype", pauli_term_dtype, PauliTerm)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an PauliTerm instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef PauliTerm obj = PauliTerm.__new__(PauliTerm)
        if owner is None:
            obj._ptr = <cupaulipropPauliTerm_t *>malloc(sizeof(cupaulipropPauliTerm_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating PauliTerm")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cupaulipropPauliTerm_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cupaulipropPauliTerm_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_truncation_strategy_dtype_offsets():
    cdef cupaulipropTruncationStrategy_t pod = cupaulipropTruncationStrategy_t()
    return _numpy.dtype({
        'names': ['strategy', 'param_struct'],
        'formats': [_numpy.int32, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.strategy)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.paramStruct)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(cupaulipropTruncationStrategy_t),
    })

truncation_strategy_dtype = _get_truncation_strategy_dtype_offsets()

cdef class TruncationStrategy:
    """Empty-initialize an instance of `cupaulipropTruncationStrategy_t`.


    .. seealso:: `cupaulipropTruncationStrategy_t`
    """
    cdef:
        cupaulipropTruncationStrategy_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <cupaulipropTruncationStrategy_t *>calloc(1, sizeof(cupaulipropTruncationStrategy_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating TruncationStrategy")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef cupaulipropTruncationStrategy_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.TruncationStrategy object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef TruncationStrategy other_
        if not isinstance(other, TruncationStrategy):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cupaulipropTruncationStrategy_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cupaulipropTruncationStrategy_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cupaulipropTruncationStrategy_t *>malloc(sizeof(cupaulipropTruncationStrategy_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating TruncationStrategy")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cupaulipropTruncationStrategy_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def strategy(self):
        """int: which kind of truncation strategy to apply"""
        return <int>(self._ptr[0].strategy)

    @strategy.setter
    def strategy(self, val):
        if self._readonly:
            raise ValueError("This TruncationStrategy instance is read-only")
        self._ptr[0].strategy = <cupaulipropTruncationStrategyKind_t><int>val

    @property
    def param_struct(self):
        """int: pointer to the parameter structure for the truncation strategy"""
        return <intptr_t>(self._ptr[0].paramStruct)

    @param_struct.setter
    def param_struct(self, val):
        if self._readonly:
            raise ValueError("This TruncationStrategy instance is read-only")
        self._ptr[0].paramStruct = <void *><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an TruncationStrategy instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cupaulipropTruncationStrategy_t), TruncationStrategy)

    @staticmethod
    def from_data(data):
        """Create an TruncationStrategy instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `truncation_strategy_dtype` holding the data.
        """
        return __from_data(data, "truncation_strategy_dtype", truncation_strategy_dtype, TruncationStrategy)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an TruncationStrategy instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef TruncationStrategy obj = TruncationStrategy.__new__(TruncationStrategy)
        if owner is None:
            obj._ptr = <cupaulipropTruncationStrategy_t *>malloc(sizeof(cupaulipropTruncationStrategy_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating TruncationStrategy")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cupaulipropTruncationStrategy_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cupaulipropTruncationStrategy_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_coefficient_truncation_params_dtype_offsets():
    cdef cupaulipropCoefficientTruncationParams_t pod = cupaulipropCoefficientTruncationParams_t()
    return _numpy.dtype({
        'names': ['cutoff'],
        'formats': [_numpy.float64],
        'offsets': [
            (<intptr_t>&(pod.cutoff)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(cupaulipropCoefficientTruncationParams_t),
    })

coefficient_truncation_params_dtype = _get_coefficient_truncation_params_dtype_offsets()

cdef class CoefficientTruncationParams:
    """Empty-initialize an instance of `cupaulipropCoefficientTruncationParams_t`.


    .. seealso:: `cupaulipropCoefficientTruncationParams_t`
    """
    cdef:
        cupaulipropCoefficientTruncationParams_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <cupaulipropCoefficientTruncationParams_t *>calloc(1, sizeof(cupaulipropCoefficientTruncationParams_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating CoefficientTruncationParams")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef cupaulipropCoefficientTruncationParams_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.CoefficientTruncationParams object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef CoefficientTruncationParams other_
        if not isinstance(other, CoefficientTruncationParams):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cupaulipropCoefficientTruncationParams_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cupaulipropCoefficientTruncationParams_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cupaulipropCoefficientTruncationParams_t *>malloc(sizeof(cupaulipropCoefficientTruncationParams_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating CoefficientTruncationParams")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cupaulipropCoefficientTruncationParams_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def cutoff(self):
        """float: cutoff value for the magnitude of the Pauli term's coefficient"""
        return self._ptr[0].cutoff

    @cutoff.setter
    def cutoff(self, val):
        if self._readonly:
            raise ValueError("This CoefficientTruncationParams instance is read-only")
        self._ptr[0].cutoff = val

    @staticmethod
    def from_buffer(buffer):
        """Create an CoefficientTruncationParams instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cupaulipropCoefficientTruncationParams_t), CoefficientTruncationParams)

    @staticmethod
    def from_data(data):
        """Create an CoefficientTruncationParams instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `coefficient_truncation_params_dtype` holding the data.
        """
        return __from_data(data, "coefficient_truncation_params_dtype", coefficient_truncation_params_dtype, CoefficientTruncationParams)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an CoefficientTruncationParams instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef CoefficientTruncationParams obj = CoefficientTruncationParams.__new__(CoefficientTruncationParams)
        if owner is None:
            obj._ptr = <cupaulipropCoefficientTruncationParams_t *>malloc(sizeof(cupaulipropCoefficientTruncationParams_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating CoefficientTruncationParams")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cupaulipropCoefficientTruncationParams_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cupaulipropCoefficientTruncationParams_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_pauli_weight_truncation_params_dtype_offsets():
    cdef cupaulipropPauliWeightTruncationParams_t pod = cupaulipropPauliWeightTruncationParams_t()
    return _numpy.dtype({
        'names': ['cutoff'],
        'formats': [_numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.cutoff)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(cupaulipropPauliWeightTruncationParams_t),
    })

pauli_weight_truncation_params_dtype = _get_pauli_weight_truncation_params_dtype_offsets()

cdef class PauliWeightTruncationParams:
    """Empty-initialize an instance of `cupaulipropPauliWeightTruncationParams_t`.


    .. seealso:: `cupaulipropPauliWeightTruncationParams_t`
    """
    cdef:
        cupaulipropPauliWeightTruncationParams_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <cupaulipropPauliWeightTruncationParams_t *>calloc(1, sizeof(cupaulipropPauliWeightTruncationParams_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating PauliWeightTruncationParams")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef cupaulipropPauliWeightTruncationParams_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.PauliWeightTruncationParams object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef PauliWeightTruncationParams other_
        if not isinstance(other, PauliWeightTruncationParams):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cupaulipropPauliWeightTruncationParams_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cupaulipropPauliWeightTruncationParams_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cupaulipropPauliWeightTruncationParams_t *>malloc(sizeof(cupaulipropPauliWeightTruncationParams_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating PauliWeightTruncationParams")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cupaulipropPauliWeightTruncationParams_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def cutoff(self):
        """int: cutoff value for the number of non-identity Paulis in the Pauli string"""
        return self._ptr[0].cutoff

    @cutoff.setter
    def cutoff(self, val):
        if self._readonly:
            raise ValueError("This PauliWeightTruncationParams instance is read-only")
        self._ptr[0].cutoff = val

    @staticmethod
    def from_buffer(buffer):
        """Create an PauliWeightTruncationParams instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cupaulipropPauliWeightTruncationParams_t), PauliWeightTruncationParams)

    @staticmethod
    def from_data(data):
        """Create an PauliWeightTruncationParams instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `pauli_weight_truncation_params_dtype` holding the data.
        """
        return __from_data(data, "pauli_weight_truncation_params_dtype", pauli_weight_truncation_params_dtype, PauliWeightTruncationParams)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an PauliWeightTruncationParams instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef PauliWeightTruncationParams obj = PauliWeightTruncationParams.__new__(PauliWeightTruncationParams)
        if owner is None:
            obj._ptr = <cupaulipropPauliWeightTruncationParams_t *>malloc(sizeof(cupaulipropPauliWeightTruncationParams_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating PauliWeightTruncationParams")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cupaulipropPauliWeightTruncationParams_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cupaulipropPauliWeightTruncationParams_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj



###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """
    Return status of the library API functions.All library API functions
    return a status which can take one of the following values.

    See `cupaulipropStatus_t`.
    """
    SUCCESS = CUPAULIPROP_STATUS_SUCCESS
    NOT_INITIALIZED = CUPAULIPROP_STATUS_NOT_INITIALIZED
    INVALID_VALUE = CUPAULIPROP_STATUS_INVALID_VALUE
    INTERNAL_ERROR = CUPAULIPROP_STATUS_INTERNAL_ERROR
    NOT_SUPPORTED = CUPAULIPROP_STATUS_NOT_SUPPORTED
    CUDA_ERROR = CUPAULIPROP_STATUS_CUDA_ERROR
    DISTRIBUTED_FAILURE = CUPAULIPROP_STATUS_DISTRIBUTED_FAILURE

class ComputeType(_IntEnum):
    """
    Supported compute types.

    See `cupaulipropComputeType_t`.
    """
    COMPUTE_32F = CUPAULIPROP_COMPUTE_32F
    COMPUTE_64F = CUPAULIPROP_COMPUTE_64F

class Memspace(_IntEnum):
    """
    Memory spaces for workspace buffer allocation.

    See `cupaulipropMemspace_t`.
    """
    DEVICE = CUPAULIPROP_MEMSPACE_DEVICE
    HOST = CUPAULIPROP_MEMSPACE_HOST

class WorkspaceKind(_IntEnum):
    """
    Kinds of workspace memory buffers.

    See `cupaulipropWorkspaceKind_t`.
    """
    WORKSPACE_SCRATCH = CUPAULIPROP_WORKSPACE_SCRATCH

class TruncationStrategyKind(_IntEnum):
    """
    Pauli operator expansion truncation strategies.

    See `cupaulipropTruncationStrategyKind_t`.
    """
    TRUNCATION_STRATEGY_COEFFICIENT_BASED = CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED
    TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED = CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED

class PauliKind(_IntEnum):
    """
    Pauli operators.

    See `cupaulipropPauliKind_t`.
    """
    PAULI_I = CUPAULIPROP_PAULI_I
    PAULI_X = CUPAULIPROP_PAULI_X
    PAULI_Y = CUPAULIPROP_PAULI_Y
    PAULI_Z = CUPAULIPROP_PAULI_Z

class CliffordGateKind(_IntEnum):
    """
    Clifford gates.

    See `cupaulipropCliffordGateKind_t`.
    """
    CLIFFORD_GATE_I = CUPAULIPROP_CLIFFORD_GATE_I
    CLIFFORD_GATE_X = CUPAULIPROP_CLIFFORD_GATE_X
    CLIFFORD_GATE_Y = CUPAULIPROP_CLIFFORD_GATE_Y
    CLIFFORD_GATE_Z = CUPAULIPROP_CLIFFORD_GATE_Z
    CLIFFORD_GATE_H = CUPAULIPROP_CLIFFORD_GATE_H
    CLIFFORD_GATE_S = CUPAULIPROP_CLIFFORD_GATE_S
    CLIFFORD_GATE_CX = CUPAULIPROP_CLIFFORD_GATE_CX
    CLIFFORD_GATE_CY = CUPAULIPROP_CLIFFORD_GATE_CY
    CLIFFORD_GATE_CZ = CUPAULIPROP_CLIFFORD_GATE_CZ
    CLIFFORD_GATE_SWAP = CUPAULIPROP_CLIFFORD_GATE_SWAP
    CLIFFORD_GATE_ISWAP = CUPAULIPROP_CLIFFORD_GATE_ISWAP
    CLIFFORD_GATE_SQRTX = CUPAULIPROP_CLIFFORD_GATE_SQRTX
    CLIFFORD_GATE_SQRTY = CUPAULIPROP_CLIFFORD_GATE_SQRTY
    CLIFFORD_GATE_SQRTZ = CUPAULIPROP_CLIFFORD_GATE_SQRTZ

class QuantumOperatorKind(_IntEnum):
    """
    Kinds of quantum operators.

    See `cupaulipropQuantumOperatorKind_t`.
    """
    EXPANSION_KIND_PAULI_ROTATION_GATE = CUPAULIPROP_EXPANSION_KIND_PAULI_ROTATION_GATE
    EXPANSION_KIND_CLIFFORD_GATE = CUPAULIPROP_EXPANSION_KIND_CLIFFORD_GATE
    EXPANSION_KIND_PAULI_NOISE_CHANNEL = CUPAULIPROP_EXPANSION_KIND_PAULI_NOISE_CHANNEL

class SortOrder(_IntEnum):
    """
    Sort order for Pauli expansions.

    See `cupaulipropSortOrder_t`.
    """
    NONE = CUPAULIPROP_SORT_ORDER_NONE
    INTERNAL = CUPAULIPROP_SORT_ORDER_INTERNAL
    LITTLE_ENDIAN_BITWISE = CUPAULIPROP_SORT_ORDER_LITTLE_ENDIAN_BITWISE


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
        __status__ = cupaulipropGetNumPackedIntegers(num_qubits, &num_packed_integers)
    check_status(__status__)
    return num_packed_integers


cpdef intptr_t create() except? 0:
    """Creates and initializes the library context.

    Returns:
        intptr_t: Library handle.

    .. seealso:: `cupaulipropCreate`
    """
    cdef Handle handle
    with nogil:
        __status__ = cupaulipropCreate(&handle)
    check_status(__status__)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """Destroys the library context.

    Args:
        handle (intptr_t): Library handle.

    .. seealso:: `cupaulipropDestroy`
    """
    with nogil:
        __status__ = cupaulipropDestroy(<Handle>handle)
    check_status(__status__)


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
        __status__ = cupaulipropCreateWorkspaceDescriptor(<Handle>handle, &workspace_desc)
    check_status(__status__)
    return <intptr_t>workspace_desc


cpdef destroy_workspace_descriptor(intptr_t workspace_desc):
    """Destroys a workspace descriptor.

    Args:
        workspace_desc (intptr_t): Workspace descriptor.

    .. seealso:: `cupaulipropDestroyWorkspaceDescriptor`
    """
    with nogil:
        __status__ = cupaulipropDestroyWorkspaceDescriptor(<WorkspaceDescriptor>workspace_desc)
    check_status(__status__)


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
        __status__ = cupaulipropWorkspaceGetMemorySize(<const Handle>handle, <const WorkspaceDescriptor>workspace_desc, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, &memory_buffer_size)
    check_status(__status__)
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
        __status__ = cupaulipropWorkspaceSetMemory(<const Handle>handle, <WorkspaceDescriptor>workspace_desc, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, <void*>memory_buffer, memory_buffer_size)
    check_status(__status__)


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
        __status__ = cupaulipropWorkspaceGetMemory(<const Handle>handle, <const WorkspaceDescriptor>workspace_descr, <_Memspace>mem_space, <_WorkspaceKind>workspace_kind, &memory_buffer, &memory_buffer_size)
    check_status(__status__)
    return (<intptr_t>memory_buffer, memory_buffer_size)


cpdef intptr_t create_pauli_expansion(intptr_t handle, int32_t num_qubits, intptr_t xz_bits_buffer, int64_t xz_bits_buffer_size, intptr_t coef_buffer, int64_t coef_buffer_size, int data_type, int64_t num_terms, int sort_order, int32_t has_duplicates) except? 0:
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
        sort_order (SortOrder): Sort order of the expansion. Use ``CUPAULIPROP_SORT_ORDER_NONE`` for unsorted expansions.
        has_duplicates (int32_t): Whether or not there are duplicates in the expansion, i.e. several terms with identical X and Z bits.

    Returns:
        intptr_t: Pauli operator expansion.

    .. seealso:: `cupaulipropCreatePauliExpansion`
    """
    cdef PauliExpansion pauli_expansion
    with nogil:
        __status__ = cupaulipropCreatePauliExpansion(<const Handle>handle, num_qubits, <void*>xz_bits_buffer, xz_bits_buffer_size, <void*>coef_buffer, coef_buffer_size, <DataType>data_type, num_terms, <_SortOrder>sort_order, has_duplicates, &pauli_expansion)
    check_status(__status__)
    return <intptr_t>pauli_expansion


cpdef destroy_pauli_expansion(intptr_t pauli_expansion):
    """Destroys a Pauli operator expansion.

    Args:
        pauli_expansion (intptr_t): Pauli operator expansion.

    .. seealso:: `cupaulipropDestroyPauliExpansion`
    """
    with nogil:
        __status__ = cupaulipropDestroyPauliExpansion(<PauliExpansion>pauli_expansion)
    check_status(__status__)


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
        __status__ = cupaulipropPauliExpansionGetStorageBuffer(<const Handle>handle, <const PauliExpansion>pauli_expansion, &xz_bits_buffer, &xz_bits_buffer_size, &coef_buffer, &coef_buffer_size, &num_terms, &location)
    check_status(__status__)
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
        __status__ = cupaulipropPauliExpansionGetNumQubits(<const Handle>handle, <const PauliExpansion>pauli_expansion, &num_qubits)
    check_status(__status__)
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
        __status__ = cupaulipropPauliExpansionGetNumTerms(<const Handle>handle, <const PauliExpansion>pauli_expansion, &num_terms)
    check_status(__status__)
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
        __status__ = cupaulipropPauliExpansionGetDataType(<const Handle>handle, <const PauliExpansion>pauli_expansion, &data_type)
    check_status(__status__)
    return <int>data_type


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
        __status__ = cupaulipropPauliExpansionIsDeduplicated(<const Handle>handle, <const PauliExpansion>pauli_expansion, &is_deduplicated)
    check_status(__status__)
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
        __status__ = cupaulipropPauliExpansionGetContiguousRange(<const Handle>handle, <const PauliExpansion>pauli_expansion, start_ind_ex, end_ind_ex, &view)
    check_status(__status__)
    return <intptr_t>view


cpdef destroy_pauli_expansion_view(intptr_t view):
    """Destroys a Pauli expansion view.

    Args:
        view (intptr_t): Pauli expansion view.

    .. seealso:: `cupaulipropDestroyPauliExpansionView`
    """
    with nogil:
        __status__ = cupaulipropDestroyPauliExpansionView(<PauliExpansionView>view)
    check_status(__status__)


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
        __status__ = cupaulipropPauliExpansionViewGetNumTerms(<const Handle>handle, <const PauliExpansionView>view, &num_terms)
    check_status(__status__)
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
        __status__ = cupaulipropPauliExpansionViewGetLocation(<const PauliExpansionView>view, &location)
    check_status(__status__)
    return <int>location


cpdef pauli_expansion_view_prepare_deduplication(intptr_t handle, intptr_t view_in, int sort_order, int64_t max_workspace_size, intptr_t workspace):
    """Updates the given workspace descriptor in preparation for deduplication of the given view.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to be deduplicated.
        sort_order (SortOrder): Sort order to apply to the output expansion. Use ``CUPAULIPROP_SORT_ORDER_NONE`` if sorting is not required. Currently, only ``CUPAULIPROP_SORT_ORDER_INTERNAL`` and ``CUPAULIPROP_SORT_ORDER_NONE`` are supported.
        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor to be updated with the required workspace buffer size.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareDeduplication`
    """
    with nogil:
        __status__ = cupaulipropPauliExpansionViewPrepareDeduplication(<const Handle>handle, <const PauliExpansionView>view_in, <_SortOrder>sort_order, max_workspace_size, <WorkspaceDescriptor>workspace)
    check_status(__status__)


cpdef pauli_expansion_view_execute_deduplication(intptr_t handle, intptr_t view_in, intptr_t expansion_out, int sort_order, intptr_t workspace, intptr_t stream):
    """Deduplicates a Pauli expansion view.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to be deduplicated.
        expansion_out (intptr_t): Pauli expansion to be populated with the deduplicated view.
        sort_order (SortOrder): Sort order to apply to the output expansion. Use ``CUPAULIPROP_SORT_ORDER_NONE`` if sorting is not required. Currently, only ``CUPAULIPROP_SORT_ORDER_INTERNAL`` and ``CUPAULIPROP_SORT_ORDER_NONE`` are supported.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream to be used for the operation.

    .. seealso:: `cupaulipropPauliExpansionViewExecuteDeduplication`
    """
    with nogil:
        __status__ = cupaulipropPauliExpansionViewExecuteDeduplication(<const Handle>handle, <const PauliExpansionView>view_in, <PauliExpansion>expansion_out, <_SortOrder>sort_order, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef pauli_expansion_populate_from_view(intptr_t handle, intptr_t view_in, intptr_t expansion_out, intptr_t stream):
    """Populates a Pauli operator expansion from a Pauli expansion view.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Input Pauli expansion view.
        expansion_out (intptr_t): Populated Pauli operator expansion.
        stream (intptr_t): CUDA stream to be used for the operation.

    .. seealso:: `cupaulipropPauliExpansionPopulateFromView`
    """
    with nogil:
        __status__ = cupaulipropPauliExpansionPopulateFromView(<const Handle>handle, <const PauliExpansionView>view_in, <PauliExpansion>expansion_out, <Stream>stream)
    check_status(__status__)


cpdef pauli_expansion_view_prepare_trace_with_expansion_view(intptr_t handle, intptr_t view1, intptr_t view2, int64_t max_workspace_size, intptr_t workspace):
    """Updates the given workspace descriptor in preparation for computing the trace of the product of two Pauli expansion views.

    Args:
        handle (intptr_t): Library handle.
        view1 (intptr_t): First Pauli expansion view to be traced.
        view2 (intptr_t): Second Pauli expansion view to be traced.
        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor to be updated with the required workspace buffer size.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareTraceWithExpansionView`
    """
    with nogil:
        __status__ = cupaulipropPauliExpansionViewPrepareTraceWithExpansionView(<const Handle>handle, <const PauliExpansionView>view1, <const PauliExpansionView>view2, max_workspace_size, <WorkspaceDescriptor>workspace)
    check_status(__status__)


cpdef pauli_expansion_view_compute_trace_with_expansion_view(intptr_t handle, intptr_t view1, intptr_t view2, int32_t take_adjoint1, intptr_t trace_significand, intptr_t trace_exponent, intptr_t workspace, intptr_t stream):
    """Computes the trace of the product of two Pauli expansion views.

    Args:
        handle (intptr_t): Library handle.
        view1 (intptr_t): First Pauli expansion view.
        view2 (intptr_t): Second Pauli expansion view.
        take_adjoint1 (int32_t): Whether or not the adjoint of the first view is taken. True ``(!= 0)`` if the adjoint is taken, false ``(0)`` otherwise.
        trace_significand (intptr_t): Pointer to CPU-accessible memory where the trace's significand will be written. The numerical type must match the data type of the views' coefficients.
        trace_exponent (intptr_t): Pointer to CPU-accessible memory where the trace's exponent will be stored. The numerical type is always ``double``.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream to be used for the operation.

    .. seealso:: `cupaulipropPauliExpansionViewComputeTraceWithExpansionView`
    """
    with nogil:
        __status__ = cupaulipropPauliExpansionViewComputeTraceWithExpansionView(<const Handle>handle, <const PauliExpansionView>view1, <const PauliExpansionView>view2, take_adjoint1, <void*>trace_significand, <double*>trace_exponent, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef pauli_expansion_view_prepare_trace_with_zero_state(intptr_t handle, intptr_t view, int64_t max_workspace_size, intptr_t workspace):
    """Updates the given workspace descriptor in preparation for computing the trace of the given Pauli expansion view with the zero state, i.e. computing ``Tr(view * |0...0><0...0|)``.

    Args:
        handle (intptr_t): Library handle.
        view (intptr_t): Pauli expansion view to be traced.
        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor to be updated with the required workspace buffer size.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareTraceWithZeroState`
    """
    with nogil:
        __status__ = cupaulipropPauliExpansionViewPrepareTraceWithZeroState(<const Handle>handle, <const PauliExpansionView>view, max_workspace_size, <WorkspaceDescriptor>workspace)
    check_status(__status__)


cpdef pauli_expansion_view_compute_trace_with_zero_state(intptr_t handle, intptr_t view, intptr_t trace_significand, intptr_t trace_exponent, intptr_t workspace, intptr_t stream):
    """Computes the trace of the Pauli expansion view with the zero state, i.e. ``Tr(view * |0...0><0...0|)``.

    Args:
        handle (intptr_t): Library handle.
        view (intptr_t): Pauli expansion view to be traced.
        trace_significand (intptr_t): Pointer to CPU-accessible memory where the trace's significand will be written. The numerical type must match the data type of the views' coefficients.
        trace_exponent (intptr_t): Pointer to CPU-accessible memory where the trace's exponent will be stored. The numerical type is always ``double``.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream to be used for the operation.

    .. seealso:: `cupaulipropPauliExpansionViewComputeTraceWithZeroState`
    """
    with nogil:
        __status__ = cupaulipropPauliExpansionViewComputeTraceWithZeroState(<const Handle>handle, <const PauliExpansionView>view, <void*>trace_significand, <double*>trace_exponent, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


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
        __status__ = cupaulipropCreateCliffordGateOperator(<const Handle>handle, <_CliffordGateKind>clifford_gate_kind, <const int32_t*>(_qubit_indices_.data()), &oper)
    check_status(__status__)
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
        __status__ = cupaulipropCreatePauliRotationGateOperator(<const Handle>handle, angle, num_qubits, <const int32_t*>(_qubit_indices_.data()), <const _PauliKind*>(_paulis_.data()), &oper)
    check_status(__status__)
    return <intptr_t>oper


cpdef intptr_t create_pauli_noise_channel_operator(intptr_t handle, int32_t num_qubits, qubit_indices, probabilities) except? 0:
    """Creates a Pauli noise channel.

    Args:
        handle (intptr_t): Library handle.
        num_qubits (int32_t): Number of qubits. Only 1 and 2 qubits are supported.
        qubit_indices (object): Qubit indices. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        probabilities (object): Probabilities for each Pauli channel. For a single qubit Pauli Channel, the probabilities are an array of length 4: ``PauliKind((i)%4)`` (i.e. ``[p_I, p_X, p_Y, p_Z]``). For a two qubit Pauli Channel, probabilities is an array of length 16. The i-th element of the probabilities is associated with the i-th element of the 2-qubit Pauli strings in lexicographic order. E.g. prob[i] corresponds to the Pauli string ``PauliKind((i)%4), PauliKind_t((i)/4)``. It can be:

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
        __status__ = cupaulipropCreatePauliNoiseChannelOperator(<const Handle>handle, num_qubits, <const int32_t*>(_qubit_indices_.data()), <const double*>(_probabilities_.data()), &oper)
    check_status(__status__)
    return <intptr_t>oper


cpdef destroy_operator(intptr_t oper):
    """Destroys a quantum operator.

    Args:
        oper (intptr_t): Quantum operator.

    .. seealso:: `cupaulipropDestroyOperator`
    """
    with nogil:
        __status__ = cupaulipropDestroyOperator(<QuantumOperator>oper)
    check_status(__status__)


cpdef int pauli_expansion_get_sort_order(intptr_t handle, intptr_t pauli_expansion) except? -1:
    """Queries the sort order of a Pauli operator expansion.

    Args:
        handle (intptr_t): Library handle.
        pauli_expansion (intptr_t): Pauli operator expansion.

    Returns:
        int: Sort order of the Pauli operator expansion. ``CUPAULIPROP_SORT_ORDER_NONE`` indicates the expansion is unsorted.

    .. seealso:: `cupaulipropPauliExpansionGetSortOrder`
    """
    cdef _SortOrder sort_order
    with nogil:
        __status__ = cupaulipropPauliExpansionGetSortOrder(<const Handle>handle, <const PauliExpansion>pauli_expansion, &sort_order)
    check_status(__status__)
    return <int>sort_order


cpdef pauli_expansion_view_prepare_sort(intptr_t handle, intptr_t view_in, int sort_order, int64_t max_workspace_size, intptr_t workspace):
    """Updates the given workspace descriptor in preparation for sorting of the given view.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to be sorted.
        sort_order (SortOrder): Sort order to apply.
        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor to be updated with the required workspace buffer size.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareSort`
    """
    with nogil:
        __status__ = cupaulipropPauliExpansionViewPrepareSort(<const Handle>handle, <const PauliExpansionView>view_in, <_SortOrder>sort_order, max_workspace_size, <WorkspaceDescriptor>workspace)
    check_status(__status__)


cpdef pauli_expansion_view_execute_sort(intptr_t handle, intptr_t view_in, intptr_t expansion_out, int sort_order, intptr_t workspace, intptr_t stream):
    """Sorts a Pauli expansion view.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to be sorted.
        expansion_out (intptr_t): Pauli expansion to be populated with the sorted view.
        sort_order (SortOrder): Sort order to apply.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream to be used for the operation.

    .. seealso:: `cupaulipropPauliExpansionViewExecuteSort`
    """
    with nogil:
        __status__ = cupaulipropPauliExpansionViewExecuteSort(<const Handle>handle, <const PauliExpansionView>view_in, <PauliExpansion>expansion_out, <_SortOrder>sort_order, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef intptr_t create_amplitude_damping_channel_operator(intptr_t handle, int32_t qubit_ind_ex, double damping_prob, double excite_prob) except? 0:
    """Creates a generalized amplitude damping channel.

    Args:
        handle (intptr_t): Library handle.
        qubit_ind_ex (int32_t): Index of qubit upon which to operate.
        damping_prob (double): Probability that the qubit is damped, i.e. decohered into a classical state.
        excite_prob (double): Probability that damping results in excitation (driving to the one state) rather than dissipation (driving to the zero state). Set to zero for conventional, dissipative amplitude damping.

    Returns:
        intptr_t: Quantum operator associated with the channel.

    .. seealso:: `cupaulipropCreateAmplitudeDampingChannelOperator`
    """
    cdef QuantumOperator oper
    with nogil:
        __status__ = cupaulipropCreateAmplitudeDampingChannelOperator(<const Handle>handle, qubit_ind_ex, damping_prob, excite_prob, &oper)
    check_status(__status__)
    return <intptr_t>oper


cpdef tuple pauli_expansion_view_prepare_trace_with_expansion_view_backward_diff(intptr_t handle, intptr_t view1, intptr_t view2, int64_t max_workspace_size, intptr_t workspace):
    """Updates the given workspace descriptor in preparation for backward differentiation of the trace of the product of two Pauli expansion views.

    Args:
        handle (intptr_t): Library handle.
        view1 (intptr_t): First Pauli expansion view to be traced.
        view2 (intptr_t): Second Pauli expansion view to be traced.
        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor to be updated with the required workspace buffer size.

    Returns:
        A 4-tuple containing:

        - int64_t: Required size (in bytes) of the X and Z bits output buffer for ``cotangentExpansion1``.
        - int64_t: Required size (in bytes) of the coefficients output buffer for ``cotangentExpansion1``.
        - int64_t: Required size (in bytes) of the X and Z bits output buffer for ``cotangentExpansion2``.
        - int64_t: Required size (in bytes) of the coefficients output buffer for ``cotangentExpansion2``.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff`
    """
    cdef int64_t required_xz_bits_buffer_size1
    cdef int64_t required_coef_buffer_size1
    cdef int64_t required_xz_bits_buffer_size2
    cdef int64_t required_coef_buffer_size2
    with nogil:
        __status__ = cupaulipropPauliExpansionViewPrepareTraceWithExpansionViewBackwardDiff(<const Handle>handle, <const PauliExpansionView>view1, <const PauliExpansionView>view2, max_workspace_size, &required_xz_bits_buffer_size1, &required_coef_buffer_size1, &required_xz_bits_buffer_size2, &required_coef_buffer_size2, <WorkspaceDescriptor>workspace)
    check_status(__status__)
    return (required_xz_bits_buffer_size1, required_coef_buffer_size1, required_xz_bits_buffer_size2, required_coef_buffer_size2)


cpdef pauli_expansion_view_compute_trace_with_expansion_view_backward_diff(intptr_t handle, intptr_t view1, intptr_t view2, int32_t take_adjoint1, intptr_t cotangent_trace_significand, intptr_t cotangent_trace_exponent, intptr_t cotangent_expansion1, intptr_t cotangent_expansion2, intptr_t workspace, intptr_t stream):
    """Computes the backward differentiation of the trace of two Pauli expansion views.

    Args:
        handle (intptr_t): Library handle.
        view1 (intptr_t): First Pauli expansion view.
        view2 (intptr_t): Second Pauli expansion view.
        take_adjoint1 (int32_t): Whether or not the adjoint of the first view is taken when forming the trace. True ``(!= 0)`` if the adjoint is taken, false ``(0)`` otherwise.
        cotangent_trace_significand (intptr_t): Pointer to host-accessible memory holding the scalar cotangent  of ``traceSignificand``. The numerical type must match the data type of the views' coefficients.
        cotangent_trace_exponent (intptr_t): Pointer to host-accessible memory holding the scalar cotangent  of ``traceExponent``. This argument is currently a dead branch for coefficient cotangents, since ``traceExponent`` presently carries no parameter dependence, though beware that this may change in the future. The numerical type is always ``double``.
        cotangent_expansion1 (intptr_t): Output Pauli expansion populated with coefficient cotangents corresponding to ``view1``. The numerical type must match the data type of the views' coefficients.
        cotangent_expansion2 (intptr_t): Output Pauli expansion populated with coefficient cotangents corresponding to ``view2``. The numerical type must match the data type of the views' coefficients.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream to be used for the operation.

    .. seealso:: `cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff`
    """
    with nogil:
        __status__ = cupaulipropPauliExpansionViewComputeTraceWithExpansionViewBackwardDiff(<const Handle>handle, <const PauliExpansionView>view1, <const PauliExpansionView>view2, take_adjoint1, <const void*>cotangent_trace_significand, <const double*>cotangent_trace_exponent, <PauliExpansion>cotangent_expansion1, <PauliExpansion>cotangent_expansion2, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef tuple pauli_expansion_view_prepare_trace_with_zero_state_backward_diff(intptr_t handle, intptr_t view, int64_t max_workspace_size, intptr_t workspace):
    """Updates the given workspace descriptor in preparation for backward differentiation of the trace with the zero state.

    Args:
        handle (intptr_t): Library handle.
        view (intptr_t): Pauli expansion view to be traced.
        max_workspace_size (int64_t): Maximum workspace size limit in bytes.
        workspace (intptr_t): Workspace descriptor to be updated with the required workspace buffer size.

    Returns:
        A 2-tuple containing:

        - int64_t: Required size (in bytes) of the X and Z bits output buffer for the cotangent expansion.
        - int64_t: Required size (in bytes) of the coefficients output buffer for the cotangent expansion.

    .. seealso:: `cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff`
    """
    cdef int64_t required_xz_bits_buffer_size
    cdef int64_t required_coef_buffer_size
    with nogil:
        __status__ = cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff(<const Handle>handle, <const PauliExpansionView>view, max_workspace_size, &required_xz_bits_buffer_size, &required_coef_buffer_size, <WorkspaceDescriptor>workspace)
    check_status(__status__)
    return (required_xz_bits_buffer_size, required_coef_buffer_size)


cpdef pauli_expansion_view_compute_trace_with_zero_state_backward_diff(intptr_t handle, intptr_t view, intptr_t cotangent_trace_significand, intptr_t cotangent_trace_exponent, intptr_t cotangent_expansion, intptr_t workspace, intptr_t stream):
    """Computes the backward differentiation of the trace of a Pauli expansion view with the zero state.

    Args:
        handle (intptr_t): Library handle.
        view (intptr_t): Pauli expansion view to be traced.
        cotangent_trace_significand (intptr_t): Pointer to host-accessible memory holding the scalar cotangent  of ``traceSignificand``. The numerical type must match the data type of the view's coefficients.
        cotangent_trace_exponent (intptr_t): Pointer to host-accessible memory holding the scalar cotangent  of ``traceExponent``. This argument is currently a dead branch for coefficient cotangents, since ``traceExponent`` presently carries no parameter dependence, though beware that this may change in the future. The numerical type is always ``double``.
        cotangent_expansion (intptr_t): Output Pauli expansion populated with coefficient cotangents corresponding to ``view``. The numerical type must match the data type of the view's coefficients.
        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream to be used for the operation.

    .. seealso:: `cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff`
    """
    with nogil:
        __status__ = cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff(<const Handle>handle, <const PauliExpansionView>view, <const void*>cotangent_trace_significand, <const double*>cotangent_trace_exponent, <PauliExpansion>cotangent_expansion, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(__status__)


cpdef quantum_operator_attach_cotangent_buffer(intptr_t handle, intptr_t oper, intptr_t cotangent_buffer, int64_t cotangent_buffer_size, int data_type, int location):
    """Attaches a cotangent buffer to a quantum operator.

    Args:
        handle (intptr_t): Library handle.
        oper (intptr_t): Quantum operator.
        cotangent_buffer (intptr_t): Pointer to user-owned cotangent buffer.
        cotangent_buffer_size (int64_t): Size of the buffer in bytes.
        data_type (int): Data type of elements in the cotangent buffer.
        location (Memspace): Memory location of the buffer (host or device).

    .. seealso:: `cupaulipropQuantumOperatorAttachCotangentBuffer`
    """
    with nogil:
        __status__ = cupaulipropQuantumOperatorAttachCotangentBuffer(<const Handle>handle, <QuantumOperator>oper, <void*>cotangent_buffer, cotangent_buffer_size, <DataType>data_type, <_Memspace>location)
    check_status(__status__)


cpdef tuple quantum_operator_get_cotangent_buffer(intptr_t handle, intptr_t oper):
    """Retrieves the cotangent buffer attached to a quantum operator.

    Args:
        handle (intptr_t): Library handle.
        oper (intptr_t): Quantum operator.

    Returns:
        A 4-tuple containing:

        - intptr_t: Pointer to the attached cotangent buffer. If no buffer is attached, this pointer is set to NULL.
        - int64_t: Required element count (number of differentiable parameters).
        - int: Data type of elements in the attached buffer. If no buffer is attached, this pointer is set to ``cudaDataType_t`` ``CUPAULIPROP_DATA_TYPE_INVALID``.
        - int: Memory location of the attached buffer. If no buffer is attached, this pointer is set to ``cupaulipropMemspace_t`` ``CUPAULIPROP_MEMSPACE_DEVICE``.

    .. seealso:: `cupaulipropQuantumOperatorGetCotangentBuffer`
    """
    cdef void* cotangent_buffer
    cdef int64_t cotangent_buffer_num_elements
    cdef DataType data_type
    cdef _Memspace location
    with nogil:
        __status__ = cupaulipropQuantumOperatorGetCotangentBuffer(<const Handle>handle, <const QuantumOperator>oper, &cotangent_buffer, &cotangent_buffer_num_elements, &data_type, &location)
    check_status(__status__)
    return (<intptr_t>cotangent_buffer, cotangent_buffer_num_elements, <int>data_type, <int>location)


# Custom implementations for truncation strategy functions (not auto-generated)
cpdef tuple pauli_expansion_view_prepare_operator_application(intptr_t handle, intptr_t view_in, intptr_t quantum_operator, int sort_order, int32_t keep_duplicates, int32_t num_truncation_strategies, truncation_strategies, int64_t max_workspace_size, intptr_t workspace):
    """Prepares a Pauli expansion view for quantum operator application.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to apply a quantum operator to.
        quantum_operator (intptr_t): Quantum operator to be applied.
        sort_order (int): Sort order to apply to the output expansion. Use ``CUPAULIPROP_SORT_ORDER_NONE`` if sorting is not required.
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
        status = cupaulipropPauliExpansionViewPrepareOperatorApplication(<const Handle>handle, <const PauliExpansionView>view_in, <const QuantumOperator>quantum_operator, <cupaulipropSortOrder_t>sort_order, keep_duplicates, num_truncation_strategies, _trunc_ptr, max_workspace_size, &required_xz_bits_buffer_size, &required_coef_buffer_size, <WorkspaceDescriptor>workspace)
    check_status(status)
    return (required_xz_bits_buffer_size, required_coef_buffer_size)


cpdef pauli_expansion_view_compute_operator_application(intptr_t handle, intptr_t view_in, intptr_t expansion_out, intptr_t quantum_operator, int32_t adjoint, int sort_order, int32_t keep_duplicates, int32_t num_truncation_strategies, truncation_strategies, intptr_t workspace, intptr_t stream):
    """Computes the application of a quantum operator to a Pauli expansion view.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Pauli expansion view to apply a quantum operator to.
        expansion_out (intptr_t): Pauli expansion to be overwritten with the result. The output expansion will satisfy sortedness and uniqueness respectively if these flags have been set to true when creating or resetting the output expansion. Otherwise they may or may not be satisfied. Their state is queryable on the output expansion after this function call via ``cupaulipropPauliExpansionGetSortedness()`` and ``cupaulipropPauliExpansionGetUniqueness()``.
        quantum_operator (intptr_t): Quantum operator to be applied.
        adjoint (int32_t): Whether or not the adjoint of the quantum operator is applied. True (!= 0) if the adjoint is applied, false (0) otherwise.
        sort_order (int): Sort order to apply to the output expansion. Use ``CUPAULIPROP_SORT_ORDER_NONE`` if sorting is not required.
        keep_duplicates (int32_t): Whether or not the output expansion is allowed to contain duplicates.
        num_truncation_strategies (int32_t): Number of Pauli expansion truncation strategies.
        truncation_strategies (object): Pauli expansion truncation strategies. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cupaulipropTruncationStrategy_t``.

        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream to be used for the operation.

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
        status = cupaulipropPauliExpansionViewComputeOperatorApplication(<const Handle>handle, <const PauliExpansionView>view_in, <PauliExpansion>expansion_out, <const QuantumOperator>quantum_operator, adjoint, <cupaulipropSortOrder_t>sort_order, keep_duplicates, num_truncation_strategies, _trunc_ptr, <WorkspaceDescriptor>workspace, <Stream>stream)
    check_status(status)


cpdef tuple pauli_expansion_view_prepare_operator_application_backward_diff(intptr_t handle, intptr_t view_in, intptr_t cotangent_out, intptr_t quantum_operator, int sort_order, int32_t keep_duplicates, int32_t num_truncation_strategies, truncation_strategies, int64_t max_workspace_size, intptr_t workspace):
    """Prepares a Pauli expansion view for backward differentiation of operator application.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Input Pauli expansion view (forward input).
        cotangent_out (intptr_t): Output cotangent represented as a Pauli expansion view.
        quantum_operator (intptr_t): Quantum operator whose adjoint buffer will be updated during computation.
        sort_order (int): Sort order to apply to the output expansion. Use ``CUPAULIPROP_SORT_ORDER_NONE`` if sorting is not required.
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

    .. seealso:: `cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff`
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
        status = cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff(<const Handle>handle, <const PauliExpansionView>view_in, <const PauliExpansionView>cotangent_out, <const QuantumOperator>quantum_operator, <cupaulipropSortOrder_t>sort_order, keep_duplicates, num_truncation_strategies, _trunc_ptr, max_workspace_size, &required_xz_bits_buffer_size, &required_coef_buffer_size, <WorkspaceDescriptor>workspace)
    check_status(status)
    return (required_xz_bits_buffer_size, required_coef_buffer_size)


cpdef pauli_expansion_view_compute_operator_application_backward_diff(intptr_t handle, intptr_t view_in, intptr_t cotangent_out, intptr_t cotangent_in, intptr_t quantum_operator, int32_t adjoint, int sort_order, int32_t keep_duplicates, int32_t num_truncation_strategies, truncation_strategies, intptr_t workspace, intptr_t stream):
    """Computes the backward differentiation of operator application on a Pauli expansion view.

    Args:
        handle (intptr_t): Library handle.
        view_in (intptr_t): Input Pauli expansion view (forward input).
        cotangent_out (intptr_t): Output cotangent represented as a Pauli expansion view.
        cotangent_in (intptr_t): Pauli expansion populated with the input cotangent.
        quantum_operator (intptr_t): Quantum operator whose adjoint buffer will be updated.
        adjoint (int32_t): Whether or not the adjoint of the quantum operator is applied. True (!= 0) if the adjoint is applied, false (0) otherwise.
        sort_order (int): Sort order to apply to the output expansion. Use ``CUPAULIPROP_SORT_ORDER_NONE`` if sorting is not required.
        keep_duplicates (int32_t): Whether or not the output expansion is allowed to contain duplicates.
        num_truncation_strategies (int32_t): Number of Pauli expansion truncation strategies.
        truncation_strategies (object): Pauli expansion truncation strategies. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cupaulipropTruncationStrategy_t``.

        workspace (intptr_t): Allocated workspace descriptor.
        stream (intptr_t): CUDA stream to be used for the operation.

    .. seealso:: `cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff`
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
        status = cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff(<const Handle>handle, <const PauliExpansionView>view_in, <const PauliExpansionView>cotangent_out, <PauliExpansion>cotangent_in, <QuantumOperator>quantum_operator, adjoint, <cupaulipropSortOrder_t>sort_order, keep_duplicates, num_truncation_strategies, _trunc_ptr, <WorkspaceDescriptor>workspace, <Stream>stream)
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


cpdef pauli_expansion_view_execute_truncation(intptr_t handle, intptr_t view_in, intptr_t expansion_out, int32_t num_truncation_strategies, truncation_strategies, intptr_t workspace, intptr_t stream):
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
        stream (intptr_t): CUDA stream to be used for the operation.

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
        status = cupaulipropPauliExpansionViewExecuteTruncation(<const Handle>handle, <const PauliExpansionView>view_in, <PauliExpansion>expansion_out, num_truncation_strategies, _trunc_ptr, <WorkspaceDescriptor>workspace, <Stream>stream)
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
