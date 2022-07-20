# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from enum import IntEnum


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
        ptr[0] = <void*>(<intptr_t>user_alloc(size, stream))
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
        user_free(<intptr_t>ptr, size, stream)
    except:
        # TODO: logging?
        return 1
    else:
        return 0


cdef void logger_callback_with_data(
        int log_level, const char* func_name, const char* message,
        void* func_arg) with gil:
    func, args, kwargs = <object>func_arg
    cdef bytes function_name = func_name
    cdef bytes function_message = message
    func(log_level, function_name.decode(), function_message.decode(),
         *args, **kwargs)


cdef void* get_buffer_pointer(buf, Py_ssize_t size) except*:
    """The caller must ensure ``buf`` is alive when the returned pointer is in use.""" 
    cdef void* bufPtr
    cdef int flags = cpython.PyBUF_ANY_CONTIGUOUS | cpython.PyBUF_WRITABLE
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
            raise ValueError(
                "buf must be either a Python int representing the pointer "
                "address to a valid buffer, or a 1D contiguous writable "
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
