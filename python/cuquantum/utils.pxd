# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from libc.stdint cimport intptr_t
cimport cpython


cdef extern from "driver_types.h" nogil:
    ctypedef void* Stream 'cudaStream_t'
    ctypedef void* Event 'cudaEvent_t'

cdef extern from "library_types.h" nogil:
    ctypedef enum DataType 'cudaDataType_t':
        pass
    ctypedef enum LibPropType 'libraryPropertyType':
        pass

cdef extern from "vector_types.h" nogil:
    ctypedef struct int2 'int2':
        pass


# Cython limitation: need standalone typedef if we wanna use it for casting
ctypedef int (*DeviceAllocType)(void*, void**, size_t, Stream)
ctypedef int (*DeviceFreeType)(void*, void*, size_t, Stream)


cdef bint is_nested_sequence(data)
cdef int cuqnt_alloc_wrapper(void* ctx, void** ptr, size_t size, Stream stream) with gil
cdef int cuqnt_free_wrapper(void* ctx, void* ptr, size_t size, Stream stream) with gil
cdef void logger_callback_with_data(
        int log_level, const char* func_name, const char* message,
        void* func_arg) with gil
cdef void* get_buffer_pointer(buf, Py_ssize_t size) except*
