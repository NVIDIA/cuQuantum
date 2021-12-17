cimport cpython


cdef inline bint is_nested_sequence(data):
    if not cpython.PySequence_Check(data):
        return False
    else:
        for i in data:
            if not cpython.PySequence_Check(i):
                return False
        else:
            return True
