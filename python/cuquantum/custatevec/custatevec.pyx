# distutils: language = c++

cimport cython
from libc.stdio cimport FILE
from libcpp.vector cimport vector
cimport cpython
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from cuquantum.utils cimport is_nested_sequence

from enum import IntEnum

import numpy as _numpy


cdef extern from * nogil:
    # from CUDA
    ctypedef int Stream 'cudaStream_t'
    ctypedef enum DataType 'cudaDataType_t':
        pass
    ctypedef enum LibPropType 'libraryPropertyType':
        pass

    # cuStateVec functions
    int custatevecCreate(_Handle*)
    int custatevecDestroy(_Handle)
    const char* custatevecGetErrorName(_Status)
    const char* custatevecGetErrorString(_Status)
    int custatevecGetDefaultWorkspaceSize(_Handle, size_t*)
    int custatevecSetWorkspace(_Handle, void*, size_t)
    int custatevecGetProperty(LibPropType, int32_t*)
    size_t custatevecGetVersion()
    int custatevecSetStream(_Handle, Stream)
    int custatevecGetStream(_Handle, Stream*)
    # int custatevecLoggerSetCallback(LoggerCallback)
    # int custatevecLoggerSetFile(FILE*)
    # int custatevecLoggerOpenFile(const char*)
    # int custatevecLoggerSetLevel(int32_t)
    # int custatevecLoggerSetMask(int32_t)
    # int custatevecLoggerForceDisable()
    int custatevecAbs2SumOnZBasis(
        _Handle, const void*, DataType, const uint32_t, double*, double*,
        const int32_t*, const uint32_t)
    int custatevecAbs2SumArray(
        _Handle, const void*, DataType, const uint32_t, double*, const int32_t*,
        const uint32_t, const int32_t*, const int32_t*, const uint32_t)
    int custatevecCollapseOnZBasis(
        _Handle, void*, DataType, const uint32_t, const int32_t, const int32_t*,
        const uint32_t, double)
    int custatevecCollapseByBitString(
        _Handle, void*, DataType, const uint32_t, const int32_t*, const int32_t*,
        const uint32_t, double)
    int custatevecMeasureOnZBasis(
        _Handle, void*, DataType, const uint32_t, int32_t*, const int32_t*,
        const uint32_t, const double, _CollapseOp)
    int custatevecBatchMeasure(
        _Handle, void*, DataType, const uint32_t, int32_t*, const int32_t*,
        const uint32_t, const double, _CollapseOp)
    int custatevecApplyExp(
        _Handle, void*, DataType, const uint32_t, double, const _Pauli*,
        const int32_t*, const uint32_t, const int32_t*, const int32_t*,
        const uint32_t)
    int custatevecApplyMatrix_bufferSize(
        _Handle, DataType, const uint32_t, const void*, DataType,
        _MatrixLayout, const int32_t, const uint32_t, const uint32_t,
        _ComputeType, size_t*)
    int custatevecApplyMatrix(
        _Handle, void*, DataType, const uint32_t, const void*,
        DataType, _MatrixLayout, const int32_t, const int32_t*,
        const uint32_t, const int32_t*, const uint32_t, const int32_t*,
        _ComputeType, void*, size_t)
    int custatevecExpectation_bufferSize(
        _Handle, DataType, const uint32_t, const void*, DataType, _MatrixLayout,
        const uint32_t, _ComputeType, size_t*)
    int custatevecExpectation(
        _Handle, const void*, DataType, const uint32_t, void*, DataType, double*,
        const void*, DataType, _MatrixLayout, const int32_t*,
        const uint32_t, _ComputeType, void*, size_t)
    int custatevecSampler_create(
        _Handle, const void*, DataType, const uint32_t, _SamplerDescriptor*,
        uint32_t, size_t*)
    int custatevecSampler_preprocess(
        _Handle, _SamplerDescriptor*, void*, const size_t)
    int custatevecSampler_sample(
        _Handle, _SamplerDescriptor*, _Index*, const int32_t*, const uint32_t,
        const double*, const uint32_t, _SamplerOutput)
    int custatevecApplyGeneralizedPermutationMatrix_bufferSize(
        _Handle, DataType, const uint32_t, const _Index*, void*, DataType,
        const int32_t*, const uint32_t, const uint32_t, size_t*)
    int custatevecApplyGeneralizedPermutationMatrix(
        _Handle, void*, DataType, const uint32_t, _Index*, const void*,
        DataType, const int32_t, const int32_t*, const uint32_t,
        const int32_t*, const int32_t*, const uint32_t, void*, size_t)
    int custatevecExpectationsOnPauliBasis(
        _Handle, void*, DataType, const uint32_t, double*, const _Pauli**,
        const int32_t**, const uint32_t*, const uint32_t)
    int custatevecAccessor_create(
        _Handle, void*, DataType, const uint32_t,
        _AccessorDescriptor*, const int32_t*, const uint32_t, const int32_t*,
        const int32_t*, const uint32_t, size_t*)
    int custatevecAccessor_createReadOnly(
        _Handle, const void*, DataType, const uint32_t,
        _AccessorDescriptor*, const int32_t*, const uint32_t, const int32_t*,
        const int32_t*, const uint32_t, size_t*)
    int custatevecAccessor_setExtraWorkspace(
        _Handle, _AccessorDescriptor*, void*, size_t)
    int custatevecAccessor_get(
        _Handle, _AccessorDescriptor*, void*, const _Index, const _Index)
    int custatevecAccessor_set(
        _Handle, _AccessorDescriptor*, const void*, const _Index, const _Index)


class cuStateVecError(RuntimeError):

    def __init__(self, status):
        self.status = status
        cdef str err_name = custatevecGetErrorName(status).decode()
        cdef str err_desc = custatevecGetErrorString(status).decode()
        cdef str err = f"{err_name} ({err_desc})"
        super().__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cdef inline check_status(int status):
    if status != 0:
        raise cuStateVecError(status)


cpdef intptr_t create() except*:
    """Initialize the cuStateVec library and create a handle.

    Returns:
        intptr_t: The opaque library handle (as Python `int`).

    .. note:: The returned handle should be tied to the current device.

    .. seealso:: `custatevecCreate`
    """
    cdef _Handle handle
    cdef int status
    with nogil:
        status = custatevecCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """Destroy the cuStateVec library handle.

    Args:
        handle (intptr_t): The library handle.

    .. seealso:: `custatevecDestroy`
    """
    with nogil:
        status = custatevecDestroy(<_Handle>handle)
    check_status(status)


cpdef size_t get_default_workspace_size(intptr_t handle) except*:
    """Get the default workspace size defined by cuStateVec.

    Args:
        handle (intptr_t): The library handle.

    Returns:
        size_t: The workspace size (in bytes).

    .. seealso:: `custatevecGetDefaultWorkspaceSize`
    """
    cdef size_t workspaceSizeInBytes
    with nogil:
        status = custatevecGetDefaultWorkspaceSize(
            <_Handle>handle, &workspaceSizeInBytes)
    check_status(status)
    return workspaceSizeInBytes


cpdef set_workspace(intptr_t handle, intptr_t workspace, size_t workspace_size):
    """Set the workspace to be used by cuStateVec.

    Args:
        handle (intptr_t): The library handle.
        workspace (intptr_t): The pointer address (as Python `int`) to the
            workspace (on device).
        workspace_size (size_t): The workspace size (in bytes).

    .. seealso:: `custatevecSetWorkspace`
    """
    with nogil:
        status = custatevecSetWorkspace(
            <_Handle>handle, <void*>workspace, workspace_size)
    check_status(status)


cpdef int get_property(int lib_prop_type) except-1:
    """Get the version information of cuStateVec.

    Args:
        lib_prop_type (cuquantum.libraryPropertyType): The property type.

    Returns:
        int: The corresponding value of the requested property.

    .. seealso:: `custatevecGetProperty`
    """
    cdef int32_t value
    status = custatevecGetProperty(<LibPropType>lib_prop_type, &value)
    check_status(status)
    return value


cpdef size_t get_version() except*:
    """Get the version of cuStateVec.

    Returns:
        size_t: The library version.

    .. seealso:: `custatevecGetVersion`
    """
    cdef size_t version = custatevecGetVersion()
    return version


cpdef set_stream(intptr_t handle, intptr_t stream):
    """Set the stream to be used by cuStateVec.

    Args:
        handle (intptr_t): The library handle.
        stream (intptr_t): The CUDA stream handle (``cudaStream_t`` as Python
            `int`).

    .. seealso:: `custatevecSetStream`
    """
    with nogil:
        status = custatevecSetStream(
            <_Handle>handle, <Stream>stream)
    check_status(status)


cpdef intptr_t get_stream(intptr_t handle):
    """Get the stream used by cuStateVec.

    Args:
        handle (intptr_t): The library handle.

    Returns:
        intptr_t:
            The CUDA stream handle (``cudaStream_t`` as Python `int`).

    .. seealso:: `custatevecGetStream`
    """
    cdef intptr_t stream
    with nogil:
        status = custatevecGetStream(
            <_Handle>handle, <Stream*>(&stream))
    check_status(status)
    return stream


# TODO(leofang): add logger callback APIs


cpdef tuple abs2sum_on_z_basis(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        bint get_parity0, bint get_parity1,
        basis_bits, uint32_t n_basis_bits):
    """Calculates the sum of squared absolute values on a given Z product basis.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        get_parity0 (bool): Whether to compute the sum of squared absolute values
            for parity 0.
        get_parity1 (bool): Whether to compute the sum of squared absolute values
            for parity 1.
        basis_bits: A host array of Z-basis index bits. It can be

            - an `int` as the pointer address to the array
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
    cdef vector[int32_t] basisBitsData
    cdef int32_t* basisBitsPtr
    if cpython.PySequence_Check(basis_bits):
        basisBitsData = basis_bits
        basisBitsPtr = basisBitsData.data()
    else:  # a pointer address
        basisBitsPtr = <int32_t*><intptr_t>basis_bits

    with nogil:
        status = custatevecAbs2SumOnZBasis(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            abs2sum0_ptr, abs2sum1_ptr,
            basisBitsPtr, n_basis_bits)
    check_status(status)
    if get_parity0 and get_parity1:
        return (abs2sum0, abs2sum1)
    elif get_parity0:
        return (abs2sum0, None)
    elif get_parity1:
        return (None, abs2sum1)


cpdef abs2sum_array(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        intptr_t abs2sum,
        bit_ordering, uint32_t bit_ordering_len,
        mask_bit_string, mask_ordering, uint32_t mask_len):
    """Calculates the sum of squared absolute values for a given set of index
    bits.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        abs2sum (intptr_t): The pointer address (as Python `int`) to the array
            (on either host or device) that would hold the sums.
        bit_ordering: A host array of index bit ordering. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of index bit ordering

        bit_ordering_len (uint32_t): The length of ``bit_ordering``.
        mask_bit_string: A host array for a bit string to specify mask. It can
            be

            - an `int` as the pointer address to the array
            - a Python sequence of index bit ordering

        mask_ordering: A host array of mask ordering. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of index bit ordering

        mask_len (uint32_t): The length of ``mask_ordering``.


    .. seealso:: `custatevecAbs2SumArray`
    """
    # bit_ordering can be a pointer address, or a Python sequence
    cdef vector[int32_t] bitOrderingData
    cdef int32_t* bitOrderingPtr
    if cpython.PySequence_Check(bit_ordering):
        bitOrderingData = bit_ordering
        bitOrderingPtr = bitOrderingData.data()
    else:  # a pointer address
        bitOrderingPtr = <int32_t*><intptr_t>bit_ordering

    # mask_bit_string can be a pointer address, or a Python sequence
    cdef vector[int32_t] maskBitStringData
    cdef int32_t* maskBitStringPtr
    if cpython.PySequence_Check(mask_bit_string):
        maskBitStringData = mask_bit_string
        maskBitStringPtr = maskBitStringData.data()
    else:  # a pointer address
        maskBitStringPtr = <int32_t*><intptr_t>mask_bit_string

    # mask_ordering can be a pointer address, or a Python sequence
    cdef vector[int32_t] maskOrderingData
    cdef int32_t* maskOrderingPtr
    if cpython.PySequence_Check(mask_ordering):
        maskOrderingData = mask_ordering
        maskOrderingPtr = maskOrderingData.data()
    else:  # a pointer address
        maskOrderingPtr = <int32_t*><intptr_t>mask_ordering

    with nogil:
        status = custatevecAbs2SumArray(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            <double*>abs2sum, bitOrderingPtr, bit_ordering_len,
            maskBitStringPtr, maskOrderingPtr, mask_len)
    check_status(status)


cpdef collapse_on_z_basis(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        int32_t parity, basis_bits, uint32_t n_basis_bits, double norm):
    """Collapse the statevector on the given Z product basis.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        parity (int32_t): The parity, 0 or 1.
        basis_bits: A host array of Z-basis index bits. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of index bits

        n_basis_bits (uint32_t): the number of basis bits.
        norm (double): The normalization factor for the statevector after
            collapse.

    .. seealso:: `custatevecCollapseOnZBasis`
    """
    # basis_bits can be a pointer address, or a Python sequence
    cdef vector[int32_t] basisBitsData
    cdef int32_t* basisBitsPtr
    if cpython.PySequence_Check(basis_bits):
        basisBitsData = basis_bits
        basisBitsPtr = basisBitsData.data()
    else:  # a pointer address
        basisBitsPtr = <int32_t*><intptr_t>basis_bits

    with nogil:
        status = custatevecCollapseOnZBasis(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            parity, basisBitsPtr, n_basis_bits, norm)
    check_status(status)


cpdef collapse_by_bitstring(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        bit_string, bit_ordering, uint32_t bit_string_len, double norm):
    """Collapse the statevector to the state specified by the given bit string.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        bit_string: A host array of a bit string. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of bits

        bit_ordering: A host array of bit string ordering. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of bit ordering

        bit_string_len (uint32_t): The length of ``bit_string``.
        norm (double): The normalization factor for the statevector after
            collapse.

    .. seealso:: `custatevecCollapseByBitString`
    """
    # bit_string can be a pointer address, or a Python sequence
    cdef vector[int32_t] bitStringData
    cdef int32_t* bitStringPtr
    if cpython.PySequence_Check(bit_string):
        bitStringData = bit_string
        bitStringPtr = bitStringData.data()
    else:  # a pointer address
        bitStringPtr = <int32_t*><intptr_t>bit_string

    # bit_ordering can be a pointer address, or a Python sequence
    cdef vector[int32_t] bitOrderingData
    cdef int32_t* bitOrderingPtr
    if cpython.PySequence_Check(bit_ordering):
        bitOrderingData = bit_ordering
        bitOrderingPtr = bitOrderingData.data()
    else:  # a pointer address
        bitOrderingPtr = <int32_t*><intptr_t>bit_ordering

    with nogil:
        status = custatevecCollapseByBitString(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            bitStringPtr, bitOrderingPtr,
            bit_string_len, norm)
    check_status(status)


cpdef int measure_on_z_basis(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        basis_bits, const uint32_t n_basis_bits, double rand_num,
        int collapse) except -1:
    """Performs measurement on the given Z-product basis.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        basis_bits: A host array of Z-basis index bits. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of index bits

        n_basis_bits (uint32_t): The number of basis bits.
        rand_num (double): A random number in [0, 1).
        collapse (Collapse): Indicate the collapse operation.

    Returns:
        int: The parity measurement outcome.

    .. seealso:: `custatevecMeasureOnZBasis`
    """
    # basis_bits can be a pointer address, or a Python sequence
    cdef vector[int32_t] basisBitsData
    cdef int32_t* basisBitsPtr
    if cpython.PySequence_Check(basis_bits):
        basisBitsData = basis_bits
        basisBitsPtr = basisBitsData.data()
    else:  # a pointer address
        basisBitsPtr = <int32_t*><intptr_t>basis_bits

    cdef int32_t parity
    with nogil:
        status = custatevecMeasureOnZBasis(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            &parity, basisBitsPtr, n_basis_bits, rand_num,
            <_CollapseOp>collapse)
    check_status(status)
    return parity


cpdef batch_measure(
        intptr_t handle, intptr_t sv, int sv_data_type,
        uint32_t n_index_bits, intptr_t bit_string, bit_ordering,
        const uint32_t bit_string_len, double rand_num, int collapse):
    """Performs measurement of arbitrary number of single qubits.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        bit_string (intptr_t): The pointer address (as Python `int`) to a host
            array of measured bit string.
        bit_ordering: A host array of bit string ordering. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of bit ordering

        bit_string_len (uint32_t): The length of ``bit_string``.
        rand_num (double): A random number in [0, 1).
        collapse (Collapse): Indicate the collapse operation.

    .. seealso:: `custatevecBatchMeasure`
    """
    # bit_ordering can be a pointer address, or a Python sequence
    cdef vector[int32_t] bitOrderingData
    cdef int32_t* bitOrderingPtr
    if cpython.PySequence_Check(bit_ordering):
        bitOrderingData = bit_ordering
        bitOrderingPtr = bitOrderingData.data()
    else:  # a pointer address
        bitOrderingPtr = <int32_t*><intptr_t>bit_ordering

    with nogil:
        status = custatevecBatchMeasure(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            <int32_t*>bit_string, bitOrderingPtr, bit_string_len,
            rand_num, <_CollapseOp>collapse)
    check_status(status)


cpdef apply_exp(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        double theta, paulis,
        targets, uint32_t n_targets,
        controls, control_bit_values, uint32_t n_controls):
    """Apply the exponential of a multi-qubit Pauli operator.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        theta (double): The rotation angle.
        paulis: A host array of :data:`Pauli` operators. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of :data:`Pauli`

        targets: A host array of target bits. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of target bits

        n_targets (uint32_t): The length of ``targets``.
        controls: A host array of control bits. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of control bits

        control_bit_values: A host array of control bit values. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of control bit values

        n_controls (uint32_t): The length of ``controls``.

    .. seealso:: `custatevecApplyExp`
    """
    # paulis can be a pointer address, or a Python sequence
    cdef vector[_Pauli] paulisData
    cdef _Pauli* paulisPtr
    if cpython.PySequence_Check(paulis):
        paulisData = paulis
        paulisPtr = paulisData.data()
    else:  # a pointer address
        paulisPtr = <_Pauli*><intptr_t>paulis

    # targets can be a pointer address, or a Python sequence
    cdef vector[int32_t] targetsData
    cdef int32_t* targetsPtr
    if cpython.PySequence_Check(targets):
        targetsData = targets
        targetsPtr = targetsData.data()
    else:  # a pointer address
        targetsPtr = <int32_t*><intptr_t>targets

    # controls can be a pointer address, or a Python sequence
    cdef vector[int32_t] controlsData
    cdef int32_t* controlsPtr
    if cpython.PySequence_Check(controls):
        controlsData = controls
        controlsPtr = controlsData.data()
    else:  # a pointer address
        controlsPtr = <int32_t*><intptr_t>controls

    # control_bit_values can be a pointer address, or a Python sequence
    cdef vector[int32_t] controlBitValuesData
    cdef int32_t* controlBitValuesPtr
    if cpython.PySequence_Check(control_bit_values):
        controlBitValuesData = control_bit_values
        controlBitValuesPtr = controlBitValuesData.data()
    else:  # a pointer address
        controlBitValuesPtr = <int32_t*><intptr_t>control_bit_values

    with nogil:
        status = custatevecApplyExp(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            theta, paulisPtr,
            targetsPtr, n_targets,
            controlsPtr, controlBitValuesPtr, n_controls)
    check_status(status)


cpdef size_t apply_matrix_buffer_size(
        intptr_t handle, int sv_data_type, uint32_t n_index_bits, intptr_t matrix,
        int matrix_data_type, int layout, int32_t adjoint, uint32_t n_targets,
        uint32_t n_controls, int compute_type) except*:
    """Computes the required workspace size for :func:`apply_matrix`.

    Args:
        handle (intptr_t): The library handle.
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        matrix (intptr_t): The pointer address (as Python `int`) to a matrix
            (on either host or device).
        matrix_data_type (cuquantum.cudaDataType): The data type of the matrix.
        layout (MatrixLayout): The memory layout the the matrix.
        adjoint (int32_t): Whether the adjoint of the matrix would be applied.
        n_targets (uint32_t): The length of ``targets``.
        n_controls (uint32_t): The length of ``controls``.
        compute_type (cuquantum.ComputeType): The compute type of matrix
            multiplication.

    Returns:
        size_t: The required workspace size (in bytes).

    .. seealso:: `custatevecApplyMatrix_bufferSize`
    """
    cdef size_t extraWorkspaceSizeInBytes
    with nogil:
        status = custatevecApplyMatrix_bufferSize(
            <_Handle>handle, <DataType>sv_data_type, n_index_bits, <void*>matrix,
            <DataType>matrix_data_type, <_MatrixLayout>layout, adjoint, n_targets,
            n_controls, <_ComputeType>compute_type, &extraWorkspaceSizeInBytes)
    check_status(status)
    return extraWorkspaceSizeInBytes


cpdef apply_matrix(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        intptr_t matrix, int matrix_data_type, int layout, int32_t adjoint,
        targets, uint32_t n_targets,
        controls, uint32_t n_controls, control_bit_values,
        int compute_type, intptr_t workspace, size_t workspace_size):
    """Apply the specified gate matrix.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        matrix (intptr_t): The pointer address (as Python `int`) to a matrix
            (on either host or device).
        matrix_data_type (cuquantum.cudaDataType): The data type of the matrix.
        layout (MatrixLayout): The memory layout the the matrix.
        adjoint (int32_t): Whether the adjoint of the matrix would be applied.
        targets: A host array of target bits. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of target bits

        n_targets (uint32_t): The length of ``targets``.
        controls: A host array of control bits. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of control bits

        n_controls (uint32_t): The length of ``controls``.
        control_bit_values: A host array of control bit values. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of control bit values

        compute_type (cuquantum.ComputeType): The compute type of matrix
            multiplication.
        workspace (intptr_t): The pointer address (as Python `int`) to the
            workspace (on device).
        workspace_size (size_t): The workspace size (in bytes).

    .. seealso:: `custatevecApplyMatrix`
    """
    # targets can be a pointer address, or a Python sequence
    cdef vector[int32_t] targetsData
    cdef int32_t* targetsPtr
    if cpython.PySequence_Check(targets):
        targetsData = targets
        targetsPtr = targetsData.data()
    else:  # a pointer address
        targetsPtr = <int32_t*><intptr_t>targets

    # controls can be a pointer address, or a Python sequence
    cdef vector[int32_t] controlsData
    cdef int32_t* controlsPtr
    if cpython.PySequence_Check(controls):
        controlsData = controls
        controlsPtr = controlsData.data()
    else:  # a pointer address
        controlsPtr = <int32_t*><intptr_t>controls

    # control_bit_values can be a pointer address, or a Python sequence
    cdef vector[int32_t] controlBitValuesData
    cdef int32_t* controlBitValuesPtr
    if cpython.PySequence_Check(control_bit_values):
        controlBitValuesData = control_bit_values
        controlBitValuesPtr = controlBitValuesData.data()
    else:  # a pointer address
        controlBitValuesPtr = <int32_t*><intptr_t>control_bit_values

    with nogil:
        status = custatevecApplyMatrix(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            <void*>matrix, <DataType>matrix_data_type,
            <_MatrixLayout>layout, adjoint,
            targetsPtr, n_targets,
            controlsPtr, n_controls,
            controlBitValuesPtr, <_ComputeType>compute_type,
            <void*>workspace, workspace_size)
    check_status(status)


cpdef size_t expectation_buffer_size(
        intptr_t handle, int sv_data_type, uint32_t n_index_bits, intptr_t matrix,
        int matrix_data_type, int layout, uint32_t n_basis_bits, int compute_type) except*:
    """Computes the required workspace size for :func:`expectation`.

    Args:
        handle (intptr_t): The library handle.
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        matrix (intptr_t): The pointer address (as Python `int`) to a matrix
            (on either host or device).
        matrix_data_type (cuquantum.cudaDataType): The data type of the matrix.
        layout (MatrixLayout): The memory layout the the matrix.
        n_basis_bits (uint32_t): The length of ``basis_bits``.
        compute_type (cuquantum.ComputeType): The compute type of matrix
            multiplication.

    Returns:
        size_t: The required workspace size (in bytes).

    .. seealso:: `custatevecExpectation_bufferSize`
    """
    cdef size_t extraWorkspaceSizeInBytes
    with nogil:
        status = custatevecExpectation_bufferSize(
            <_Handle>handle, <DataType>sv_data_type, n_index_bits, <void*>matrix,
            <DataType>matrix_data_type, <_MatrixLayout>layout, n_basis_bits,
            <_ComputeType>compute_type, &extraWorkspaceSizeInBytes)
    check_status(status)
    return extraWorkspaceSizeInBytes


cpdef expectation(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        intptr_t expect, int expect_data_type,
        intptr_t matrix, int matrix_data_type, int layout,
        basis_bits, uint32_t n_basis_bits,
        int compute_type, intptr_t workspace, size_t workspace_size):
    """Compute the expectation value of the given matrix with respect to the
    statevector.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        expect (intptr_t): The pointer address (as Python `int`) for storing the
            expectation value (on host).
        expect_data_type (cuquantum.cudaDataType): The data type of ``expect``.
        matrix (intptr_t): The pointer address (as Python `int`) to a matrix
            (on either host or device).
        matrix_data_type (cuquantum.cudaDataType): The data type of the matrix.
        layout (MatrixLayout): The memory layout the the matrix.
        basis_bits: A host array of basis index bits. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of basis bits

        n_basis_bits (uint32_t): The length of ``basis_bits``.
        compute_type (cuquantum.ComputeType): The compute type of matrix
            multiplication.
        workspace (intptr_t): The pointer address (as Python `int`) to the
            workspace (on device).
        workspace_size (size_t): The workspace size (in bytes).

    .. seealso:: `custatevecExpectation`
    """
    # basis_bits can be a pointer address, or a Python sequence
    cdef vector[int32_t] basisBitsData
    cdef int32_t* basisBitsPtr
    if cpython.PySequence_Check(basis_bits):
        basisBitsData = basis_bits
        basisBitsPtr = basisBitsData.data()
    else:  # a pointer address
        basisBitsPtr = <int32_t*><intptr_t>basis_bits

    # Note: residualNorm is not supported in beta 1
    # TODO(leofang): check for beta 2
    cdef double residualNorm
    with nogil:
        status = custatevecExpectation(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            <void*>expect, <DataType>expect_data_type, &residualNorm,
            <void*>matrix, <DataType>matrix_data_type,
            <_MatrixLayout>layout,
            basisBitsPtr, n_basis_bits,
            <_ComputeType>compute_type,
            <void*>workspace, workspace_size)
    check_status(status)


cpdef tuple sampler_create(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        uint32_t n_max_shots):
    """Create a sampler descriptor.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        n_max_shots (uint32_t): The maximal number of shots that will be
            performed using this sampler.

    Returns:
        tuple:
            A 2-tuple. The first element is the pointer address (as Python
            `int`) to the sampler descriptor, and the second element is the
            amount of required workspace size (in bytes).

    .. note:: Unlike its C counterpart, the returned sampler descriptor must
        be explicitly cleaned up using :func:`sampler_destroy` when the work
        is done.

    .. seealso:: `custatevecSampler_create`
    """
    cdef _SamplerDescriptor* sampler = <_SamplerDescriptor*>(
        PyMem_Malloc(sizeof(_SamplerDescriptor)))
    cdef size_t extraWorkspaceSizeInBytes
    with nogil:
        status = custatevecSampler_create(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            sampler, n_max_shots, &extraWorkspaceSizeInBytes)
    check_status(status)
    return (<intptr_t>sampler, extraWorkspaceSizeInBytes)


# TODO(leofang): fix this when the beta 2 (?) APIs are up
cpdef sampler_destroy(intptr_t sampler):
    """Destroy the sampler descriptor.

    Args:
        sampler (intptr_t): The pointer address (as Python `int`) to the
            sampler descriptor.

    .. note:: This function has no C counterpart in the current release.

    .. seealso:: :func:`sampler_create`
    """
    # This API is unique in Python as we can't pass around structs
    # allocated on stack
    PyMem_Free(<void*>sampler)


cpdef sampler_preprocess(
        intptr_t handle, intptr_t sampler, intptr_t workspace,
        size_t workspace_size):
    """Preprocess the statevector to prepare for sampling.

    Args:
        handle (intptr_t): The library handle.
        sampler (intptr_t): The pointer address (as Python `int`) to the
            sampler descriptor.
        workspace (intptr_t): The pointer address (as Python `int`) to the
            workspace (on device).
        workspace_size (size_t): The workspace size (in bytes).

    .. seealso:: `custatevecSampler_preprocess`
    """
    with nogil:
        status = custatevecSampler_preprocess(
            <_Handle>handle, <_SamplerDescriptor*>sampler,
            <void*>workspace, workspace_size)
    check_status(status)


cpdef sampler_sample(
        intptr_t handle, intptr_t sampler, intptr_t bit_strings,
        bit_ordering, uint32_t bit_string_len, rand_nums,
        uint32_t n_shots, int order):
    """Sample bit strings from the statevector.

    Args:
        handle (intptr_t): The library handle.
        sampler (intptr_t): The pointer address (as Python `int`) to the
            sampler descriptor.
        bit_strings (intptr_t): The pointer address (as Python `int`) for
            storing the sampled bit strings (on host).
        bit_ordering: A host array of bit string ordering. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of bit ordering

        bit_string_len (uint32_t): The number of bits in ``bit_ordering``.
        rand_nums: A host array of random numbers in [0, 1). It can be

            - an `int` as the pointer address to the array
            - a Python sequence of random numbers

        n_shots (uint32_t): The number of shots.
        order (SamplerOutput): The order of sampled bit strings.

    .. seealso:: `custatevecSampler_sample`
    """
    # bit_ordering can be a pointer address, or a Python sequence
    cdef vector[int32_t] bitOrderingData
    cdef int32_t* bitOrderingPtr
    if cpython.PySequence_Check(bit_ordering):
        bitOrderingData = bit_ordering
        bitOrderingPtr = bitOrderingData.data()
    else:  # a pointer address
        bitOrderingPtr = <int32_t*><intptr_t>bit_ordering

    # rand_nums can be a pointer address, or a Python sequence
    cdef vector[double] randNumsData
    cdef double* randNumsPtr
    if cpython.PySequence_Check(rand_nums):
        randNumsData = rand_nums
        randNumsPtr = randNumsData.data()
    else:  # a pointer address
        randNumsPtr = <double*><intptr_t>rand_nums

    with nogil:
        status = custatevecSampler_sample(
            <_Handle>handle, <_SamplerDescriptor*>sampler, <_Index*>bit_strings,
            bitOrderingPtr, bit_string_len, randNumsPtr, n_shots,
            <_SamplerOutput>order)
    check_status(status)


cpdef size_t apply_generalized_permutation_matrix_buffer_size(
        intptr_t handle, int sv_data_type, uint32_t n_index_bits,
        permutation, intptr_t diagonals, int diagonals_data_type,
        basis_bits, uint32_t n_basis_bits, uint32_t mask_len) except*:
    """Computes the required workspace size for :func:`apply_generalized_permutation_matrix`.

    Args:
        handle (intptr_t): The library handle.
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        permutation: A host or device array for the permutation table. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of permutation elements

        diagonals (intptr_t): The pointer address (as Python `int`) to a matrix
            (on either host or device).
        diagonals_data_type (cuquantum.cudaDataType): The data type of the matrix.
        basis_bits: A host array of permutation matrix basis bits. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of basis bits

        n_basis_bits (uint32_t): The length of ``basis_bits``.
        mask_len (uint32_t): The length of ``mask_ordering``.

    Returns:
        size_t: The required workspace size (in bytes).

    .. seealso:: `custatevecApplyGeneralizedPermutationMatrix_bufferSize`
    """
    cdef size_t extraWorkspaceSize

    # permutation can be a pointer address (on host or device), or a Python
    # sequence (on host)
    cdef vector[_Index] permutationData
    cdef _Index* permutationPtr
    if cpython.PySequence_Check(permutation):
        permutationData = permutation
        permutationPtr = permutationData.data()
    else:  # a pointer address
        permutationPtr = <_Index*><intptr_t>permutation

    # basis_bits can be a pointer address, or a Python sequence
    cdef vector[int32_t] basisBitsData
    cdef int32_t* basisBitsPtr
    if cpython.PySequence_Check(basis_bits):
        basisBitsData = basis_bits
        basisBitsPtr = basisBitsData.data()
    else:  # a pointer address
        basisBitsPtr = <int32_t*><intptr_t>basis_bits

    with nogil:
        status = custatevecApplyGeneralizedPermutationMatrix_bufferSize(
            <_Handle>handle, <DataType>sv_data_type, n_index_bits,
            permutationPtr, <void*>diagonals, <DataType>diagonals_data_type,
            basisBitsPtr, n_basis_bits, mask_len, &extraWorkspaceSize)
    check_status(status)
    return extraWorkspaceSize


cpdef apply_generalized_permutation_matrix(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        permutation, intptr_t diagonals, int diagonals_data_type,
        int32_t adjoint, basis_bits, uint32_t n_basis_bits,
        mask_bit_string, mask_ordering, uint32_t mask_len,
        intptr_t workspace, size_t workspace_size):
    """Apply a generalized permutation matrix.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        permutation: A host or device array for the permutation table. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of permutation elements

        diagonals (intptr_t): The pointer address (as Python `int`) to a matrix
            (on either host or device).
        diagonals_data_type (cuquantum.cudaDataType): The data type of the matrix.
        adjoint (int32_t): Whether the adjoint of the matrix would be applied.
        basis_bits: A host array of permutation matrix basis bits. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of basis bits

        n_basis_bits (uint32_t): The length of ``basis_bits``.
        mask_bit_string: A host array for a bit string to specify mask. It can
            be

            - an `int` as the pointer address to the array
            - a Python sequence of index bit ordering

        mask_ordering: A host array of mask ordering. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of index bit ordering

        mask_len (uint32_t): The length of ``mask_ordering``.
        workspace (intptr_t): The pointer address (as Python `int`) to the
            workspace (on device).
        workspace_size (size_t): The workspace size (in bytes).

    .. seealso:: `custatevecApplyGeneralizedPermutationMatrix`
    """
    # permutation can be a pointer address (on host or device), or a Python
    # sequence (on host)
    cdef vector[_Index] permutationData
    cdef _Index* permutationPtr
    if cpython.PySequence_Check(permutation):
        permutationData = permutation
        permutationPtr = permutationData.data()
    else:  # a pointer address
        permutationPtr = <_Index*><intptr_t>permutation

    # basis_bits can be a pointer address, or a Python sequence
    cdef vector[int32_t] basisBitsData
    cdef int32_t* basisBitsPtr
    if cpython.PySequence_Check(basis_bits):
        basisBitsData = basis_bits
        basisBitsPtr = basisBitsData.data()
    else:  # a pointer address
        basisBitsPtr = <int32_t*><intptr_t>basis_bits

    # mask_bit_string can be a pointer address, or a Python sequence
    cdef vector[int32_t] maskBitStringData
    cdef int32_t* maskBitStringPtr
    if cpython.PySequence_Check(mask_bit_string):
        maskBitStringData = mask_bit_string
        maskBitStringPtr = maskBitStringData.data()
    else:  # a pointer address
        maskBitStringPtr = <int32_t*><intptr_t>mask_bit_string

    # mask_ordering can be a pointer address, or a Python sequence
    cdef vector[int32_t] maskOrderingData
    cdef int32_t* maskOrderingPtr
    if cpython.PySequence_Check(mask_ordering):
        maskOrderingData = mask_ordering
        maskOrderingPtr = maskOrderingData.data()
    else:  # a pointer address
        maskOrderingPtr = <int32_t*><intptr_t>mask_ordering

    with nogil:
        status = custatevecApplyGeneralizedPermutationMatrix(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            permutationPtr, <void*>diagonals, <DataType>diagonals_data_type,
            adjoint, basisBitsPtr, n_basis_bits,
            maskBitStringPtr, maskOrderingPtr, mask_len,
            <void*>workspace, workspace_size)
    check_status(status)


cpdef expectations_on_pauli_basis(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        intptr_t expectations, pauli_ops,
        basis_bits, n_basis_bits, uint32_t n_pauli_op_arrays):
    """Compute expectation values for multiple multi-qubit Pauli strings.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        expectations (intptr_t): The pointer address (as Python `int`) to store
            the corresponding expectation values on host. The returned values
            are stored in double (float64).
        pauli_ops: A host array of :data:`Pauli` operators. It can be

            - an `int` as the pointer address to the nested sequence
            - a Python sequence of `int`, each of which is a pointer address
              to the corresponding Pauli string
            - a nested Python sequence of :data:`Pauli`

        basis_bits: A host array of basis index bits. It can be

            - an `int` as the pointer address to the nested sequence
            - a Python sequence of `int`, each of which is a pointer address
              to the corresponding basis bits
            - a nested Python sequence of basis bits

        n_basis_bits: A host array of the length of each array in
            ``basis_bits``. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of `int`

        n_pauli_op_arrays (uint32_t): The number of Pauli operator arrays.

    .. seealso:: `custatevecExpectationsOnPauliBasis`
    """
    # pauli_ops can be:
    #   - a plain pointer address
    #   - a Python sequence (of pointer addresses)
    #   - a nested Python sequence (of _Pauli)
    # Note: it cannot be a mix of sequences and ints.
    cdef vector[intptr_t] pauliOpsCData
    cdef _Pauli** pauliOpsPtr
    if is_nested_sequence(pauli_ops):
        # flatten the 2D sequence
        pauliOpsPyData = []
        for i in pauli_ops:
            # too bad a Python list can't hold C++ vectors, so we use NumPy
            # arrays as the container here to keep data alive
            data = _numpy.asarray(i, dtype=_numpy.int32)
            assert data.ndim == 1
            pauliOpsPyData.append(data)
            pauliOpsCData.push_back(<intptr_t>data.ctypes.data)
        pauliOpsPtr = <_Pauli**>(pauliOpsCData.data())
    elif cpython.PySequence_Check(pauli_ops):
        # handle 1D sequence
        pauliOpsCData = pauli_ops
        pauliOpsPtr = <_Pauli**>(pauliOpsCData.data())
    else:
        # a pointer address, take it as is
        pauliOpsPtr = <_Pauli**><intptr_t>pauli_ops

    # basis_bits can be:
    #   - a plain pointer address
    #   - a Python sequence (of pointer addresses)
    #   - a nested Python sequence (of int32_t)
    # Note: it cannot be a mix of sequences and ints.
    cdef vector[intptr_t] basisBitsCData
    cdef int32_t** basisBitsPtr
    if is_nested_sequence(basis_bits):
        # flatten the 2D sequence
        basisBitsPyData = []
        for i in basis_bits:
            # too bad a Python list can't hold C++ vectors, so we use NumPy
            # arrays as the container here to keep data alive
            data = _numpy.asarray(i, dtype=_numpy.int32)
            assert data.ndim == 1
            basisBitsPyData.append(data)
            basisBitsCData.push_back(<intptr_t>data.ctypes.data)
        basisBitsPtr = <int32_t**>(basisBitsCData.data())
    elif cpython.PySequence_Check(basis_bits):
        # handle 1D sequence
        basisBitsCData = basis_bits
        basisBitsPtr = <int32_t**>(basisBitsCData.data())
    else:
        # a pointer address, take it as is
        basisBitsPtr = <int32_t**><intptr_t>basis_bits

    # n_basis_bits can be a pointer address, or a Python sequence
    cdef vector[uint32_t] nBasisBitsData
    cdef uint32_t* nBasisBitsPtr
    if cpython.PySequence_Check(n_basis_bits):
        nBasisBitsData = n_basis_bits
        nBasisBitsPtr = nBasisBitsData.data()
    else:  # a pointer address
        nBasisBitsPtr = <uint32_t*><intptr_t>n_basis_bits

    with nogil:
        status = custatevecExpectationsOnPauliBasis(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            <double*>expectations, <const _Pauli**>pauliOpsPtr,
            <const int32_t**>basisBitsPtr, nBasisBitsPtr, n_pauli_op_arrays)
    check_status(status)


cpdef (intptr_t, size_t) accessor_create(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        bit_ordering, uint32_t bit_ordering_len,
        mask_bit_string, mask_ordering, uint32_t mask_len):
    """Create accessor to copy elements between the statevector and external
    buffers.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device).
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        bit_ordering: A host array of basis bits for the external buffer. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of basis bits

        bit_ordering_len (uint32_t): The length of ``bit_ordering``.
        mask_bit_string: A host array for specifying mask values. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of mask values

        mask_ordering: A host array of mask ordering. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of index bit ordering

        mask_len (uint32_t): The length of ``mask_ordering``.

    Returns:
        tuple:
            A 2-tuple. The first element is the accessor descriptor (as Python
            `int`), and the second element is the required workspace size (in
            bytes).

    .. note:: Unlike its C counterpart, the returned accessor descriptor must
        be explicitly cleaned up using :func:`accessor_destroy` when the work
        is done.

    .. seealso:: `custatevecAccessor_create`
    """
    cdef _AccessorDescriptor* accessor = <_AccessorDescriptor*>(
        PyMem_Malloc(sizeof(_AccessorDescriptor)))
    cdef size_t workspace_size

    # bit_ordering can be a pointer address, or a Python sequence
    cdef vector[int32_t] bitOrderingData
    cdef int32_t* bitOrderingPtr
    if cpython.PySequence_Check(bit_ordering):
        bitOrderingData = bit_ordering
        bitOrderingPtr = bitOrderingData.data()
    else:  # a pointer address
        bitOrderingPtr = <int32_t*><intptr_t>bit_ordering

    # mask_bit_string can be a pointer address, or a Python sequence
    cdef vector[int32_t] maskBitStringData
    cdef int32_t* maskBitStringPtr
    if cpython.PySequence_Check(mask_bit_string):
        maskBitStringData = mask_bit_string
        maskBitStringPtr = maskBitStringData.data()
    else:  # a pointer address
        maskBitStringPtr = <int32_t*><intptr_t>mask_bit_string

    # mask_ordering can be a pointer address, or a Python sequence
    cdef vector[int32_t] maskOrderingData
    cdef int32_t* maskOrderingPtr
    if cpython.PySequence_Check(mask_ordering):
        maskOrderingData = mask_ordering
        maskOrderingPtr = maskOrderingData.data()
    else:  # a pointer address
        maskOrderingPtr = <int32_t*><intptr_t>mask_ordering

    with nogil:
        status = custatevecAccessor_create(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            accessor, bitOrderingPtr, bit_ordering_len,
            maskBitStringPtr, maskOrderingPtr, mask_len, &workspace_size)
    check_status(status)
    return (<intptr_t>accessor, workspace_size)


cpdef (intptr_t, size_t) accessor_create_readonly(
        intptr_t handle, intptr_t sv, int sv_data_type, uint32_t n_index_bits,
        bit_ordering, uint32_t bit_ordering_len,
        mask_bit_string, mask_ordering, uint32_t mask_len):
    """Create accessor to copy elements from the statevector to external buffers.

    Args:
        handle (intptr_t): The library handle.
        sv (intptr_t): The pointer address (as Python `int`) to the statevector
            (on device). The statevector is read-only.
        sv_data_type (cuquantum.cudaDataType): The data type of the statevector.
        n_index_bits (uint32_t): The number of index bits.
        bit_ordering: A host array of basis bits for the external buffer. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of basis bits

        bit_ordering_len (uint32_t): The length of ``bit_ordering``.
        mask_bit_string: A host array for specifying mask values. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of mask values

        mask_ordering: A host array of mask ordering. It can be

            - an `int` as the pointer address to the array
            - a Python sequence of index bit ordering

        mask_len (uint32_t): The length of ``mask_ordering``.

    Returns:
        tuple:
            A 2-tuple. The first element is the accessor descriptor (as Python
            `int`), and the second element is the required workspace size (in
            bytes).

    .. note:: Unlike its C counterpart, the returned accessor descriptor must
        be explicitly cleaned up using :func:`accessor_destroy` when the work
        is done.

    .. seealso:: `custatevecAccessor_createReadOnly`
    """
    cdef _AccessorDescriptor* accessor = <_AccessorDescriptor*>(
        PyMem_Malloc(sizeof(_AccessorDescriptor)))
    cdef size_t workspace_size

    # bit_ordering can be a pointer address, or a Python sequence
    cdef vector[int32_t] bitOrderingData
    cdef int32_t* bitOrderingPtr
    if cpython.PySequence_Check(bit_ordering):
        bitOrderingData = bit_ordering
        bitOrderingPtr = bitOrderingData.data()
    else:  # a pointer address
        bitOrderingPtr = <int32_t*><intptr_t>bit_ordering

    # mask_bit_string can be a pointer address, or a Python sequence
    cdef vector[int32_t] maskBitStringData
    cdef int32_t* maskBitStringPtr
    if cpython.PySequence_Check(mask_bit_string):
        maskBitStringData = mask_bit_string
        maskBitStringPtr = maskBitStringData.data()
    else:  # a pointer address
        maskBitStringPtr = <int32_t*><intptr_t>mask_bit_string

    # mask_ordering can be a pointer address, or a Python sequence
    cdef vector[int32_t] maskOrderingData
    cdef int32_t* maskOrderingPtr
    if cpython.PySequence_Check(mask_ordering):
        maskOrderingData = mask_ordering
        maskOrderingPtr = maskOrderingData.data()
    else:  # a pointer address
        maskOrderingPtr = <int32_t*><intptr_t>mask_ordering

    with nogil:
        status = custatevecAccessor_createReadOnly(
            <_Handle>handle, <void*>sv, <DataType>sv_data_type, n_index_bits,
            accessor, bitOrderingPtr, bit_ordering_len,
            maskBitStringPtr, maskOrderingPtr, mask_len, &workspace_size)
    check_status(status)
    return (<intptr_t>accessor, workspace_size)


cpdef accessor_destroy(intptr_t accessor):
    """Destroy the accessor descriptor.

    Args:
        accessor (intptr_t): The accessor descriptor.

    .. note:: This function has no C counterpart in the current release.

    .. seealso:: :func:`accessor_create`
    """
    # This API is unique in Python as we can't pass around structs
    # allocated on stack
    PyMem_Free(<void*>accessor)


cpdef accessor_set_extra_workspace(
        intptr_t handle, intptr_t accessor,
        intptr_t workspace, size_t workspace_size):
    """Set the external workspace to the accessor.

    Args:
        handle (intptr_t): The library handle.
        accessor (intptr_t): The accessor descriptor.
        workspace (intptr_t): The pointer address to the workspace (on device).
        workspace_size (size_t): The size of workspace (in bytes).

    .. seealso:: `custatevecAccessor_setExtraWorkspace`
    """
    with nogil:
        status = custatevecAccessor_setExtraWorkspace(
            <_Handle>handle, <_AccessorDescriptor*>accessor,
            <void*>workspace, workspace_size)
    check_status(status)


cpdef accessor_get(
        intptr_t handle, intptr_t accessor, intptr_t buf,
        _Index begin, _Index end):
    """Copy elements from the statevector to an external buffer.

    Args:
        handle (intptr_t): The library handle.
        accessor (intptr_t): The accessor descriptor.
        buf (intptr_t): The external buffer to store the copied elements.
        begin (int): The beginning index.
        end (int): The end index.

    .. seealso:: `custatevecAccessor_get`
    """
    with nogil:
        status = custatevecAccessor_get(
            <_Handle>handle, <_AccessorDescriptor*>accessor, <void*>buf,
            begin, end)
    check_status(status)


cpdef accessor_set(
        intptr_t handle, intptr_t accessor, intptr_t buf,
        _Index begin, _Index end):
    """Copy elements from an external buffer to the statevector.

    Args:
        handle (intptr_t): The library handle.
        accessor (intptr_t): The accessor descriptor.
        buf (intptr_t): The external buffer to copy elements from.
        begin (int): The beginning index.
        end (int): The end index.

    .. seealso:: `custatevecAccessor_set`
    """
    with nogil:
        status = custatevecAccessor_set(
            <_Handle>handle, <_AccessorDescriptor*>accessor, <void*>buf,
            begin, end)
    check_status(status)


class Pauli(IntEnum):
    """See `custatevecPauli_t`."""
    I = CUSTATEVEC_PAULI_I
    X = CUSTATEVEC_PAULI_X
    Y = CUSTATEVEC_PAULI_Y
    Z = CUSTATEVEC_PAULI_Z

class MatrixLayout(IntEnum):
    """See `custatevecMatrixLayout_t`."""
    COL = CUSTATEVEC_MATRIX_LAYOUT_COL
    ROW = CUSTATEVEC_MATRIX_LAYOUT_ROW

# unused in beta 1
class MatrixType(IntEnum):
    """See `custatevecMatrixType_t`."""
    GENERAL = CUSTATEVEC_MATRIX_TYPE_GENERAL
    UNITARY = CUSTATEVEC_MATRIX_TYPE_UNITARY
    HERMITIAN = CUSTATEVEC_MATRIX_TYPE_HERMITIAN

class Collapse(IntEnum):
    """See `custatevecCollapseOp_t`."""
    NONE = CUSTATEVEC_COLLAPSE_NONE
    NORMALIZE_AND_ZERO = CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO

class SamplerOutput(IntEnum):
    """See `custatevecSamplerOutput_t`."""
    RANDNUM_ORDER = CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER
    ASCENDING_ORDER = CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER


del IntEnum


# expose them to Python
MAJOR_VER = CUSTATEVEC_VER_MAJOR
MINOR_VER = CUSTATEVEC_VER_MINOR
PATCH_VER = CUSTATEVEC_VER_PATCH
VERSION = CUSTATEVEC_VERSION
