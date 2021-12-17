import cupy
from cupy import testing
import numpy
import pytest

import cuquantum
from cuquantum import ComputeType, cudaDataType
from cuquantum import custatevec


###################################################################
#
# As of beta 2, the test suite for Python bindings is kept minimal.
# The sole goal is to ensure the Python arguments are properly
# passed to the C level. We do not ensure coverage nor correctness.
# This decision will be revisited in the future.
#
###################################################################

dtype_to_data_type = {
    numpy.dtype(numpy.complex64): cudaDataType.CUDA_C_32F,
    numpy.dtype(numpy.complex128): cudaDataType.CUDA_C_64F,
}


dtype_to_compute_type = {
    numpy.dtype(numpy.complex64): ComputeType.COMPUTE_32F,
    numpy.dtype(numpy.complex128): ComputeType.COMPUTE_64F,
}


@pytest.fixture()
def handle():
    h = custatevec.create()
    yield h
    custatevec.destroy(h)


@testing.parameterize(*testing.product({
    'n_qubits': (3,),
    'dtype': (numpy.complex64, numpy.complex128),
}))
class TestSV:
    # Base class for all statevector tests

    def get_sv(self):
        arr = cupy.zeros((2**self.n_qubits,), dtype=self.dtype)
        arr[0] = 1  # initialize in |000...00>
        return arr

    def _return_data(self, data, name, dtype, return_value):
        if return_value == 'int':
            if len(data) == 0:
                # empty, give it a NULL
                return 0, 0
            else:
                # return int as void*
                data = numpy.asarray(data, dtype=dtype)
                setattr(self, name, data)  # keep data alive
                return data.ctypes.data, data.size
        elif return_value == 'seq':
            # data itself is already a flat sequence
            return data, len(data)
        else:
            assert False


class TestLibHelper:

    def test_get_version(self):
        ver = custatevec.get_version()
        assert ver == (custatevec.MAJOR_VER * 1000
            + custatevec.MINOR_VER * 100
            + custatevec.PATCH_VER)
        assert ver == custatevec.VERSION

    def test_get_property(self):
        assert custatevec.MAJOR_VER == custatevec.get_property(
            cuquantum.libraryPropertyType.MAJOR_VERSION)
        assert custatevec.MINOR_VER == custatevec.get_property(
            cuquantum.libraryPropertyType.MINOR_VERSION)
        assert custatevec.PATCH_VER == custatevec.get_property(
            cuquantum.libraryPropertyType.PATCH_LEVEL)


class TestHandle:

    def test_handle_create_destroy(self, handle):
        # simple rount-trip test
        pass

    def test_workspace(self, handle):
        default_workspace_size = custatevec.get_default_workspace_size(handle)
        # this is about 18MB as of cuQuantum beta 1
        assert default_workspace_size > 0
        # cuStateVec does not like a smaller workspace...
        size = 24*1024**2
        assert size > default_workspace_size
        memptr = cupy.cuda.alloc(size)
        custatevec.set_workspace(handle, memptr.ptr, size)  # should not fail

    def test_stream(self, handle):
        # default is on the null stream
        assert 0 == custatevec.get_stream(handle)

        # simple set/get round-trip
        stream = cupy.cuda.Stream()
        custatevec.set_stream(handle, stream.ptr)
        assert stream.ptr == custatevec.get_stream(handle)


class TestAbs2Sum(TestSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'basis_bits': (numpy.int32, 'int'),},
            {'basis_bits': (numpy.int32, 'seq'),},
        )
    )
    def test_abs2sum_on_z_basis(self, handle, input_form):
        sv = self.get_sv()
        basis_bits = list(range(self.n_qubits))
        basis_bits, basis_bits_len = self._return_data(
            basis_bits, 'basis_bits', *input_form['basis_bits'])
        data_type = dtype_to_data_type[sv.dtype]

        # case 1: both are computed
        sum0, sum1 = custatevec.abs2sum_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            True, True, basis_bits, basis_bits_len)
        assert numpy.allclose(sum0+sum1, 1)
        assert (sum0 is not None) and (sum1 is not None)

        # case 2: only sum0 is computed
        sum0, sum1 = custatevec.abs2sum_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            True, False, basis_bits, basis_bits_len)
        assert numpy.allclose(sum0, 1)
        assert (sum0 is not None) and (sum1 is None)

        # case 3: only sum1 is computed
        sum0, sum1 = custatevec.abs2sum_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            False, True, basis_bits, basis_bits_len)
        assert numpy.allclose(sum1, 0)
        assert (sum0 is None) and (sum1 is not None)

        # case 4: none is computed
        with pytest.raises(ValueError):
            sum0, sum1 = custatevec.abs2sum_on_z_basis(
                handle, sv.data.ptr, data_type, self.n_qubits,
                False, False, basis_bits, basis_bits_len)

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (numpy.int32, 'int'),},
            {'bit_ordering': (numpy.int32, 'seq'),},
        )
    )
    @pytest.mark.parametrize(
        'xp', (numpy, cupy)
     )
    def test_abs2sum_array_no_mask(self, handle, xp, input_form):
        # change sv from |000> to 1/\sqrt{2} (|001> + |100>)
        sv = self.get_sv()
        sv[0] = 0
        sv[1] = 1./numpy.sqrt(2)
        sv[4] = 1./numpy.sqrt(2)

        data_type = dtype_to_data_type[sv.dtype]
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, bit_ordering_len = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])
        # test abs2sum on both host and device
        abs2sum = xp.empty((2**bit_ordering_len,), dtype=xp.float64)
        abs2sum_ptr = abs2sum.data.ptr if xp is cupy else abs2sum.ctypes.data
        custatevec.abs2sum_array(
            handle, sv.data.ptr, data_type, self.n_qubits, abs2sum_ptr,
            bit_ordering, bit_ordering_len, 0, 0, 0)
        assert xp.allclose(abs2sum.sum(), 1)
        assert xp.allclose(abs2sum[1], 0.5)
        assert xp.allclose(abs2sum[4], 0.5)

    # TODO(leofang): add more tests for abs2sum_array, such as nontrivial masks


class TestCollapse(TestSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'basis_bits': (numpy.int32, 'int'),},
            {'basis_bits': (numpy.int32, 'seq'),},
        )
    )
    @pytest.mark.parametrize(
        'parity', (0, 1)
    )
    def test_collapse_on_z_basis(self, handle, parity, input_form):
        sv = self.get_sv()
        basis_bits = list(range(self.n_qubits))
        basis_bits, basis_bits_len = self._return_data(
            basis_bits, 'basis_bits', *input_form['basis_bits'])
        data_type = dtype_to_data_type[sv.dtype]

        custatevec.collapse_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            parity, basis_bits, basis_bits_len, 1)

        if parity == 0:
            assert cupy.allclose(sv.sum(), 1)
        elif parity == 1:
            assert cupy.allclose(sv.sum(), 0)

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (numpy.int32, 'int'), 'bitstring': (numpy.int32, 'int')},
            {'bit_ordering': (numpy.int32, 'seq'), 'bitstring': (numpy.int32, 'seq')},
        )
    )
    def test_collapse_by_bitstring(self, handle, input_form):
        # change sv to 1/\sqrt{2} (|000> + |111>)
        sv = self.get_sv()
        sv[0] = numpy.sqrt(0.5)
        sv[-1] = numpy.sqrt(0.5)

        # collapse to |111>
        bitstring = [1] * self.n_qubits
        bitstring, bitstring_len = self._return_data(
            bitstring, 'bitstring', *input_form['bitstring'])

        bit_ordering = list(range(self.n_qubits))
        bit_ordering, _ = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])
        data_type = dtype_to_data_type[sv.dtype]

        norm = 0.5
        # the sv after collapse is normalized as sv -> sv / \sqrt{norm}
        custatevec.collapse_by_bitstring(
            handle, sv.data.ptr, data_type, self.n_qubits,
            bitstring, bit_ordering, bitstring_len,
            norm)
        assert cupy.allclose(sv.sum(), 1)
        assert cupy.allclose(sv[-1], 1)


@pytest.mark.parametrize(
    'rand',
    # the choices here ensure we get either parity
    (0, numpy.nextafter(1, 0))
)
@pytest.mark.parametrize(
    'collapse',
    (custatevec.Collapse.NORMALIZE_AND_ZERO, custatevec.Collapse.NONE)
)
class TestMeasure(TestSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'basis_bits': (numpy.int32, 'int'),},
            {'basis_bits': (numpy.int32, 'seq'),},
        )
    )
    def test_measure_on_z_basis(self, handle, rand, collapse, input_form):
        # change the sv to 1/\sqrt{2} (|000> + |010>) to allow 50-50 chance
        # of getting either parity
        sv = self.get_sv()
        sv[0] = numpy.sqrt(0.5)
        sv[2] = numpy.sqrt(0.5)

        basis_bits = list(range(self.n_qubits))
        basis_bits, basis_bits_len = self._return_data(
            basis_bits, 'basis_bits', *input_form['basis_bits'])
        data_type = dtype_to_data_type[sv.dtype]
        orig_sv = sv.copy()

        parity = custatevec.measure_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            basis_bits, basis_bits_len, rand, collapse)

        if collapse == custatevec.Collapse.NORMALIZE_AND_ZERO:
            if parity == 0:
                # collapse to |000>
                assert cupy.allclose(sv[0], 1)
            elif parity == 1:
                # collapse to |111>
                assert cupy.allclose(sv[2], 1)
            # sv is collapsed
            assert not (sv == orig_sv).all()
        else:
            # sv is intact
            assert (sv == orig_sv).all()

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (numpy.int32, 'int'),},
            {'bit_ordering': (numpy.int32, 'seq'),},
        )
    )
    def test_batch_measure(self, handle, rand, collapse, input_form):
        # change sv to 1/\sqrt{2} (|000> + |111>)
        sv = self.get_sv()
        sv[0] = numpy.sqrt(0.5)
        sv[-1] = numpy.sqrt(0.5)
        orig_sv = sv.copy()

        data_type = dtype_to_data_type[sv.dtype]
        bitstring = numpy.empty(self.n_qubits, dtype=numpy.int32)
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, _ = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])

        custatevec.batch_measure(
            handle, sv.data.ptr, data_type, self.n_qubits,
            bitstring.ctypes.data, bit_ordering, bitstring.size,
            rand, collapse)

        if collapse == custatevec.Collapse.NORMALIZE_AND_ZERO:
            if bitstring.sum() == 0:
                # collapse to |000>
                assert cupy.allclose(sv[0], 1)
            elif bitstring.sum() == 3:
                # collapse to |111>
                assert cupy.allclose(sv[-1], 1)
            else:
                assert False, f"unexpected bitstring: {bitstring}"
            # sv is collapsed
            assert not (sv == orig_sv).all()
        else:
            assert bitstring.sum() in (0, 3)
            # sv is intact
            assert (sv == orig_sv).all()


class TestApply(TestSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'targets': (numpy.int32, 'int'), 'controls': (numpy.int32, 'int'),
             # sizeof(enum) == sizeof(int)
             'paulis': (numpy.int32, 'int'),},
            {'targets': (numpy.int32, 'seq'), 'controls': (numpy.int32, 'seq'),
             'paulis': (numpy.int32, 'seq'),},
        )
    )
    def test_apply_exp(self, handle, input_form):
        # change sv to |100>
        sv = self.get_sv()
        sv[0] = 0
        sv[4] = 1

        data_type = dtype_to_data_type[sv.dtype]
        targets = [0, 1]
        targets, targets_len = self._return_data(
            targets, 'targets', *input_form['targets'])
        controls = [2]
        controls, controls_len = self._return_data(
            controls, 'controls', *input_form['controls'])
        control_values = 0  # set all control bits to 1
        paulis = [custatevec.Pauli.X, custatevec.Pauli.X]
        paulis, _ = self._return_data(
            paulis, 'paulis', *input_form['paulis'])

        custatevec.apply_exp(
            handle, sv.data.ptr, data_type, self.n_qubits,
            0.5*numpy.pi, paulis,
            targets, targets_len,
            controls, control_values, controls_len)
        sv *= -1j

        # result is |111>
        assert cupy.allclose(sv[-1], 1)

    @pytest.mark.parametrize(
        'input_form', (
            {'targets': (numpy.int32, 'int'), 'controls': (numpy.int32, 'int'),
             # sizeof(enum) == sizeof(int)
             'paulis': (numpy.int32, 'int'),},
            {'targets': (numpy.int32, 'seq'), 'controls': (numpy.int32, 'seq'),
             'paulis': (numpy.int32, 'seq'),},
        )
    )
    @pytest.mark.parametrize(
        'xp', (numpy, cupy)
     )
    def test_apply_matrix(self, handle, xp, input_form):
        sv = self.get_sv()
        data_type = dtype_to_data_type[sv.dtype]
        compute_type = dtype_to_compute_type[sv.dtype]
        targets = [0, 1, 2]
        targets, targets_len = self._return_data(
            targets, 'targets', *input_form['targets'])
        controls = []
        controls, controls_len = self._return_data(
            controls, 'controls', *input_form['controls'])

        # matrix can live on host or device
        matrix = xp.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=sv.dtype)
        matrix[-1][0] = 1
        matrix_ptr = matrix.ctypes.data if xp is numpy else matrix.data.ptr

        workspace_size = custatevec.apply_matrix_buffer_size(
            handle, data_type, self.n_qubits,
            matrix_ptr, data_type, custatevec.MatrixLayout.ROW, 0,
            targets_len, controls_len, compute_type)
        if workspace_size:
            workspace = cupy.cuda.alloc(workspace_size)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        custatevec.apply_matrix(
            handle, sv.data.ptr, data_type, self.n_qubits,
            matrix_ptr, data_type, custatevec.MatrixLayout.ROW, 0,
            targets, targets_len,
            controls, controls_len, 0,
            compute_type, workspace_ptr, workspace_size)

        assert sv[-1] == 1  # output state is |111>


    @pytest.mark.parametrize(
        'input_form', (
            {'permutation': (numpy.int64, 'int'), 'basis_bits': (numpy.int32, 'int'),
             'mask_bitstring': (numpy.int32, 'int'), 'mask_ordering': (numpy.int32, 'int')},
            {'permutation': (numpy.int64, 'seq'), 'basis_bits': (numpy.int32, 'seq'),
             'mask_bitstring': (numpy.int32, 'seq'), 'mask_ordering': (numpy.int32, 'seq')},
        )
    )
    @pytest.mark.parametrize(
        'xp', (numpy, cupy)
     )
    def test_apply_generalized_permutation_matrix(self, handle, xp, input_form):
        sv = self.get_sv()
        sv[:] = 1  # invalid sv just to make math checking easier
        data_type = dtype_to_data_type[sv.dtype]
        compute_type = dtype_to_compute_type[sv.dtype]

        # TODO(leofang): test permutation on either host or device
        permutation = list(numpy.random.permutation(2**self.n_qubits))
        permutation_data = permutation
        permutation, permutation_len = self._return_data(
            permutation, 'permutation', *input_form['permutation'])

        # diagonal can live on host or device
        diagonal = 10 * xp.ones((2**self.n_qubits, ), dtype=sv.dtype)
        diagonal_ptr = diagonal.ctypes.data if xp is numpy else diagonal.data.ptr

        basis_bits = list(range(self.n_qubits))
        basis_bits, basis_bits_len = self._return_data(
            basis_bits, 'basis_bits', *input_form['basis_bits'])

        # TODO(leofang): test masks
        mask_bitstring = 0
        mask_ordering = 0
        mask_len = 0

        workspace_size = custatevec.apply_generalized_permutation_matrix_buffer_size(
            handle, data_type, self.n_qubits,
            permutation, diagonal_ptr, data_type,
            basis_bits, basis_bits_len, mask_len)

        if workspace_size:
            workspace = cupy.cuda.alloc(workspace_size)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        custatevec.apply_generalized_permutation_matrix(
            handle, sv.data.ptr, data_type, self.n_qubits,
            permutation, diagonal_ptr, data_type, 0,
            basis_bits, basis_bits_len,
            mask_bitstring, mask_ordering, mask_len,
            workspace_ptr, workspace_size)

        assert cupy.allclose(sv, diagonal[xp.asarray(permutation_data)])


class TestExpect(TestSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'basis_bits': (numpy.int32, 'int'),},
            {'basis_bits': (numpy.int32, 'seq'),},
        )
    )
    @pytest.mark.parametrize(
        'expect_dtype', (numpy.float64, numpy.complex128)
    )
    @pytest.mark.parametrize(
        'xp', (numpy, cupy)
    )
    def test_expectation(self, handle, xp, expect_dtype, input_form):
        # create a uniform sv
        sv = self.get_sv()
        sv[:] = numpy.sqrt(1/(2**self.n_qubits))

        data_type = dtype_to_data_type[sv.dtype]
        compute_type = dtype_to_compute_type[sv.dtype]
        basis_bits = list(range(self.n_qubits))
        basis_bits, basis_bits_len = self._return_data(
            basis_bits, 'basis_bits', *input_form['basis_bits'])

        # matrix can live on host or device
        matrix = xp.ones((2**self.n_qubits, 2**self.n_qubits), dtype=sv.dtype)
        matrix_ptr = matrix.ctypes.data if xp is numpy else matrix.data.ptr

        workspace_size = custatevec.expectation_buffer_size(
            handle, data_type, self.n_qubits,
            matrix_ptr, data_type, custatevec.MatrixLayout.ROW,
            basis_bits_len, compute_type)
        if workspace_size:
            workspace = cupy.cuda.alloc(workspace_size)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        expect = numpy.empty((1,), dtype=expect_dtype)
        # TODO(leofang): check if this is relaxed in beta 2
        expect_data_type = (
            cudaDataType.CUDA_R_64F if expect_dtype == numpy.float64
            else cudaDataType.CUDA_C_64F)

        custatevec.expectation(
            handle, sv.data.ptr, data_type, self.n_qubits,
            expect.ctypes.data, expect_data_type,
            matrix_ptr, data_type, custatevec.MatrixLayout.ROW,
            basis_bits, basis_bits_len,
            compute_type, workspace_ptr, workspace_size)

        assert xp.allclose(expect, 2**self.n_qubits)


class TestSampler(TestSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (numpy.int32, 'int'),},
            {'bit_ordering': (numpy.int32, 'seq'),},
        )
    )
    def test_sampling(self, handle, input_form):
        # create a uniform sv
        sv = self.get_sv()
        sv[:] = numpy.sqrt(1/(2**self.n_qubits))

        data_type = dtype_to_data_type[sv.dtype]
        compute_type = dtype_to_compute_type[sv.dtype]
        shots = 4096

        bitstrings = numpy.empty((shots,), dtype=numpy.int64)
        rand_nums = numpy.random.random((shots,)).astype(numpy.float64)
        # measure all qubits
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, _ = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])

        sampler, workspace_size = custatevec.sampler_create(
            handle, sv.data.ptr, data_type, self.n_qubits, shots)
        if workspace_size:
            workspace = cupy.cuda.alloc(workspace_size)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        try:
            custatevec.sampler_preprocess(
                handle, sampler, workspace_ptr, workspace_size)
            custatevec.sampler_sample(
                handle, sampler, bitstrings.ctypes.data,
                bit_ordering, self.n_qubits,
                rand_nums.ctypes.data, shots,
                custatevec.SamplerOutput.RANDNUM_ORDER)
        finally:
            # This is Python-only API. Need finally to ensure it's freed.
            custatevec.sampler_destroy(sampler)

        keys, counts = numpy.unique(bitstrings, return_counts=True)
        # keys are the returned bitstrings 000, 001, ..., 111
        # the sv has all components, and unique() returns a sorted array,
        # so the following should hold:
        assert (keys == numpy.arange(2**self.n_qubits)).all()

        # TODO: test counts, which should follow a uniform distribution


# TODO(leofang): test mask_bitstring & mask_ordering
@pytest.mark.parametrize(
    'input_form', (
        {'bit_ordering': (numpy.int32, 'int'), 'mask_bitstring': (numpy.int32, 'int'), 'mask_ordering': (numpy.int32, 'int')},
        {'bit_ordering': (numpy.int32, 'seq'), 'mask_bitstring': (numpy.int32, 'seq'), 'mask_ordering': (numpy.int32, 'seq')},
    )
)
@pytest.mark.parametrize(
    'readonly', (True, False)
)
class TestAccessor(TestSV):

    def test_accessor_get(self, handle, input_form, readonly):
        # create a monotonically increasing sv
        sv = self.get_sv()
        data = cupy.arange(2**self.n_qubits, dtype=sv.dtype)
        data /= cupy.sqrt(data**2)
        sv[:] = data

        data_type = dtype_to_data_type[sv.dtype]
        compute_type = dtype_to_compute_type[sv.dtype]

        # measure all qubits
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, bit_ordering_len = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])
        # TODO(leofang): test them
        mask_bitstring = 0
        mask_ordering = 0
        mask_len = 0

        if readonly:
            accessor_create = custatevec.accessor_create_readonly
        else:
            accessor_create = custatevec.accessor_create

        accessor, workspace_size = accessor_create(
            handle, sv.data.ptr, data_type, self.n_qubits,
            bit_ordering, bit_ordering_len,
            mask_bitstring, mask_ordering, mask_len)

        try:
            if workspace_size:
                workspace = cupy.cuda.alloc(workspace_size)
                custatevec.accessor_set_extra_workspace(
                    handle, accessor, workspace.ptr, workspace_size)

            buf_len = 2**2
            buf = cupy.empty(buf_len, dtype=sv.dtype)

            # copy the last buf_len elements
            custatevec.accessor_get(
                handle, accessor, buf.data.ptr, sv.size-1-buf_len, sv.size-1)
        finally:
            # This is Python-only API. Need finally to ensure it's freed.
            custatevec.accessor_destroy(accessor)

        assert (sv[sv.size-1-buf_len: sv.size-1] == buf).all()

    def test_accessor_set(self, handle, input_form, readonly):
        # create a monotonically increasing sv
        sv = self.get_sv()
        data = cupy.arange(2**self.n_qubits, dtype=sv.dtype)
        data /= cupy.sqrt(data**2)
        sv[:] = data

        data_type = dtype_to_data_type[sv.dtype]
        compute_type = dtype_to_compute_type[sv.dtype]

        # measure all qubits
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, bit_ordering_len = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])
        # TODO(leofang): test them
        mask_bitstring = 0
        mask_ordering = 0
        mask_len = 0

        if readonly:
            accessor_create = custatevec.accessor_create_readonly
        else:
            accessor_create = custatevec.accessor_create

        accessor, workspace_size = accessor_create(
            handle, sv.data.ptr, data_type, self.n_qubits,
            bit_ordering, bit_ordering_len,
            mask_bitstring, mask_ordering, mask_len)

        try:
            if workspace_size:
                workspace = cupy.cuda.alloc(workspace_size)
                custatevec.accessor_set_extra_workspace(
                    handle, accessor, workspace.ptr, workspace_size)

            buf_len = 2**2
            buf = cupy.zeros(buf_len, dtype=sv.dtype)

            if readonly:
                # copy the last buf_len elements would fail
                with pytest.raises(custatevec.cuStateVecError) as e_info:
                    custatevec.accessor_set(
                        handle, accessor, buf.data.ptr, sv.size-1-buf_len, sv.size-1)
            else:
                # copy the last buf_len elements
                custatevec.accessor_set(
                    handle, accessor, buf.data.ptr, sv.size-1-buf_len, sv.size-1)
        finally:
            # This is Python-only API. Need finally to ensure it's freed.
            custatevec.accessor_destroy(accessor)

        if readonly:
            # sv unchanged
            assert (sv[sv.size-1-buf_len: sv.size-1] == data[sv.size-1-buf_len: sv.size-1]).all()
        else:
            assert (sv[sv.size-1-buf_len: sv.size-1] == 0).all()
