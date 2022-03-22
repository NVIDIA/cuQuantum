from libc.stdint cimport intptr_t
cimport cpython


cdef extern from * nogil:
    # from CUDA
    ctypedef int Stream 'cudaStream_t'


cdef inline bint is_nested_sequence(data):
    if not cpython.PySequence_Check(data):
        return False
    else:
        for i in data:
            if not cpython.PySequence_Check(i):
                return False
        else:
            return True


cdef inline int cuqnt_alloc_wrapper(void* ctx, void** ptr, size_t size, Stream stream) with gil:
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


cdef inline int cuqnt_free_wrapper(void* ctx, void* ptr, size_t size, Stream stream) with gil:
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


cdef inline void logger_callback_with_data(
        int log_level, const char* func_name, const char* message,
        void* func_arg) with gil:
    func, args, kwargs = <object>func_arg
    cdef bytes function_name = func_name
    cdef bytes function_message = message
    func(log_level, function_name.decode(), function_message.decode(),
         *args, **kwargs)
