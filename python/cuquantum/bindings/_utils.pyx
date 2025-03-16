# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from libcpp.utility cimport move
from cython.operator cimport dereference as deref

from enum import IntEnum

from numpy import ndarray as _np_ndarray


cdef bint is_nested_sequence(data):
    if not cpython.PySequence_Check(data):
        return False
    else:
        for i in data:
            if not cpython.PySequence_Check(i):
                return False
        else:
            return True


cdef int cuqnt_alloc_wrapper(void* ctx, void** ptr, size_t size, Stream stream) with gil:
    """Assuming the user provides an alloc routine: ptr = alloc(size, stream).

    Note: this function holds the Python GIL.
    """
    cdef tuple pairs

    try:
        pairs = <object>(ctx)
        user_alloc = pairs[0]
        ptr[0] = <void*>(<intptr_t>user_alloc(size, <intptr_t>stream))
    except:
        # TODO: logging?
        return 1
    else:
        return 0


cdef int cuqnt_free_wrapper(void* ctx, void* ptr, size_t size, Stream stream) with gil:
    """Assuming the user provides a free routine: free(ptr, size, stream).

    Note: this function holds the Python GIL.
    """
    cdef tuple pairs

    try:
        pairs = <object>(ctx)
        user_free = pairs[1]
        user_free(<intptr_t>ptr, size, <intptr_t>stream)
    except:
        # TODO: logging?
        return 1
    else:
        return 0


cdef void logger_callback_with_data(
        int32_t log_level, const char* func_name, const char* message,
        void* func_arg) with gil:
    func, args, kwargs = <object>func_arg
    cdef bytes function_name = func_name
    cdef bytes function_message = message
    func(log_level, function_name.decode(), function_message.decode(),
         *args, **kwargs)


cdef void* get_buffer_pointer(buf, Py_ssize_t size, readonly=True) except*:
    """The caller must ensure ``buf`` is alive when the returned pointer is in use.""" 
    cdef void* bufPtr
    cdef int flags = cpython.PyBUF_ANY_CONTIGUOUS
    if not readonly:
        flags |= cpython.PyBUF_WRITABLE
    cdef int status = -1
    cdef cpython.Py_buffer view

    if isinstance(buf, int):
        bufPtr = <void*><intptr_t>buf
    else:  # try buffer protocol
        try:
            status = cpython.PyObject_GetBuffer(buf, &view, flags)
            assert view.len == size
            assert view.ndim == 1
        except Exception as e:
            adj = "writable " if not readonly else ""
            raise ValueError(
                 "buf must be either a Python int representing the pointer "
                f"address to a valid buffer, or a 1D contiguous {adj}"
                 "buffer, of size bytes") from e
        else:
            bufPtr = view.buf
        finally:
            if status == 0:
                cpython.PyBuffer_Release(&view)

    return bufPtr


# The (subset of) compute types below are shared by cuStateVec and cuTensorNet
class ComputeType(IntEnum):
    """An enumeration of CUDA compute types."""
    COMPUTE_DEFAULT = 0
    COMPUTE_16F     = 1 << 0
    COMPUTE_32F     = 1 << 2
    COMPUTE_64F     = 1 << 4
    COMPUTE_8U      = 1 << 6
    COMPUTE_8I      = 1 << 8
    COMPUTE_32U     = 1 << 7
    COMPUTE_32I     = 1 << 9
    COMPUTE_16BF    = 1 << 10
    COMPUTE_TF32    = 1 << 12
    COMPUTE_3XTF32  = 1 << 13


# TODO: use those exposed by CUDA Python instead, but before removing these
# duplicates, check if they are fixed to inherit IntEnum instead of Enum.
class cudaDataType(IntEnum):
    """An enumeration of `cudaDataType_t`."""
    CUDA_R_16F  =  2
    CUDA_C_16F  =  6
    CUDA_R_16BF = 14
    CUDA_C_16BF = 15
    CUDA_R_32F  =  0
    CUDA_C_32F  =  4
    CUDA_R_64F  =  1
    CUDA_C_64F  =  5
    CUDA_R_4I   = 16
    CUDA_C_4I   = 17
    CUDA_R_4U   = 18
    CUDA_C_4U   = 19
    CUDA_R_8I   =  3
    CUDA_C_8I   =  7
    CUDA_R_8U   =  8
    CUDA_C_8U   =  9
    CUDA_R_16I  = 20
    CUDA_C_16I  = 21
    CUDA_R_16U  = 22
    CUDA_C_16U  = 23
    CUDA_R_32I  = 10
    CUDA_C_32I  = 11
    CUDA_R_32U  = 12
    CUDA_C_32U  = 13
    CUDA_R_64I  = 24
    CUDA_C_64I  = 25
    CUDA_R_64U  = 26
    CUDA_C_64U  = 27


class libraryPropertyType(IntEnum):
    """An enumeration of library version information."""
    MAJOR_VERSION = 0
    MINOR_VERSION = 1
    PATCH_LEVEL = 2


del IntEnum


# Defined in CPython:
# https://github.com/python/cpython/blob/26bc2cc06128890ac89492eca20e83abe0789c1c/Objects/unicodetype_db.h#L6311-L6349
cdef int[29] _WHITESPACE_UNICODE_INTS = [
    0x0009,
    0x000A,
    0x000B,
    0x000C,
    0x000D,
    0x001C,
    0x001D,
    0x001E,
    0x001F,
    0x0020,
    0x0085,
    0x00A0,
    0x1680,
    0x2000,
    0x2001,
    0x2002,
    0x2003,
    0x2004,
    0x2005,
    0x2006,
    0x2007,
    0x2008,
    0x2009,
    0x200A,
    0x2028,
    0x2029,
    0x202F,
    0x205F,
    0x3000,
]


WHITESPACE_UNICODE = ''.join(chr(s) for s in _WHITESPACE_UNICODE_INTS)


# Cython can't infer the ResT overload when it is wrapped in nullable_unique_ptr,
# so we need a dummy (__unused) input argument to help it
cdef int get_resource_ptr(nullable_unique_ptr[vector[ResT]] &in_out_ptr, object obj, ResT* __unused) except 1:
    if isinstance(obj, _np_ndarray):
        # TODO: can we do "assert obj.dtype == some_dtype" here? it seems we have no
        # way to check the dtype...
        # TODO: how about buffer protocol?
        assert <size_t>(obj.dtype.itemsize) == sizeof(ResT)
        in_out_ptr.reset(<vector[ResT]*><intptr_t>(obj.ctypes.data), False)
    elif cpython.PySequence_Check(obj):
        vec = new vector[ResT](len(obj))
        # set the ownership immediately to avoid leaking the `vec` memory in
        # case of exception in the following loop
        in_out_ptr.reset(vec, True)
        for i in range(len(obj)):
            deref(vec)[i] = obj[i]
    else:
        in_out_ptr.reset(<vector[ResT]*><intptr_t>obj, False)
    return 0


cdef int get_resource_ptrs(nullable_unique_ptr[ vector[PtrT*] ] &in_out_ptr, object obj, PtrT* __unused) except 1:
    if cpython.PySequence_Check(obj):
        vec = new vector[PtrT*](len(obj))
        # set the ownership immediately to avoid leaking the `vec` memory in
        # case of exception in the following loop
        in_out_ptr.reset(vec, True)
        for i in range(len(obj)):
            deref(vec)[i] = <PtrT*><intptr_t>(obj[i])
    else:
        in_out_ptr.reset(<vector[PtrT*]*><intptr_t>obj, False)
    return 0


cdef int get_nested_resource_ptr(nested_resource[ResT] &in_out_ptr, object obj, ResT* __unused) except 1:
    cdef nullable_unique_ptr[ vector[intptr_t] ] nested_ptr
    cdef nullable_unique_ptr[ vector[vector[ResT]] ] nested_res_ptr
    cdef vector[intptr_t]* nested_vec = NULL
    cdef vector[vector[ResT]]* nested_res_vec = NULL
    cdef size_t i = 0, length = 0
    cdef intptr_t addr

    if is_nested_sequence(obj):
        length = len(obj)
        nested_res_vec = new vector[vector[ResT]](length)
        nested_vec = new vector[intptr_t](length)
        # set the ownership immediately to avoid leaking memory in case of
        # exception in the following loop
        nested_res_ptr.reset(nested_res_vec, True)
        nested_ptr.reset(nested_vec, True)
        for i, obj_i in enumerate(obj):
            deref(nested_res_vec)[i] = obj_i
            deref(nested_vec)[i] = <intptr_t>(deref(nested_res_vec)[i].data())
    elif cpython.PySequence_Check(obj):
        length = len(obj)
        nested_vec = new vector[intptr_t](length)
        nested_ptr.reset(nested_vec, True)
        for i, addr in enumerate(obj):
            deref(nested_vec)[i] = addr
        nested_res_ptr.reset(NULL, False)
    else:
        # obj is an int (ResT**)
        nested_res_ptr.reset(NULL, False)
        nested_ptr.reset(<vector[intptr_t]*><intptr_t>obj, False)

    in_out_ptr.ptrs = move(nested_ptr)
    in_out_ptr.nested_resource_ptr = move(nested_res_ptr)
    return 0


class FunctionNotFoundError(RuntimeError): pass

class NotSupportedError(RuntimeError): pass
