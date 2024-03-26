# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import copy

import cupy as cp
from cupy import testing
import cupyx as cpx
import numpy as np
try:
    from mpi4py import MPI  # init!
except ImportError:
    MPI = None
import pytest

import cuquantum
from cuquantum import ComputeType, cudaDataType
from cuquantum import custatevec as cusv

from .. import (_can_use_cffi, dtype_to_compute_type, dtype_to_data_type,
                MemHandlerTestBase, MemoryResourceFactory, LoggerTestBase)


###################################################################
#
# As of beta 2, the test suite for Python bindings is kept minimal.
# The sole goal is to ensure the Python arguments are properly
# passed to the C level. We do not ensure coverage nor correctness.
# This decision will be revisited in the future.
#
###################################################################

@pytest.fixture()
def handle():
    h = cusv.create()
    yield h
    cusv.destroy(h)


@testing.parameterize(*testing.product({
    'n_qubits': (3,),
    'dtype': (np.complex64, np.complex128),
}))
class TestSV:
    # Base class for all statevector tests

    def get_sv(self):
        arr = cp.zeros((2**self.n_qubits,), dtype=self.dtype)
        arr[0] = 1  # initialize in |000...00>
        return arr

    # TODO: make this a static method
    def _return_data(self, data, name, dtype, return_value):
        if return_value == 'int':
            if len(data) == 0:
                # empty, give it a NULL
                return 0, 0
            else:
                # return int as void*
                data = np.asarray(data, dtype=dtype)
                setattr(self, name, data)  # keep data alive
                return data.ctypes.data, data.size
        elif return_value == 'seq':
            # data itself is already a flat sequence
            return data, len(data)
        else:
            assert False


@testing.parameterize(*testing.product({
    'n_svs': (3,),
    'n_qubits': (4,),
    'n_extra_qubits': (0, 1),  # for padding purpose
    'dtype': (np.complex64, np.complex128),
}))
class TestBatchedSV:
    # Base class for all batched statevector tests

    def get_sv(self):
        arr = cp.zeros((self.n_svs, 2**(self.n_qubits + self.n_extra_qubits)), dtype=self.dtype)
        arr[:, 0] = 1  # initialize in |000...00>
        self.sv_stride = 2 ** (self.n_qubits + self.n_extra_qubits)  # in counts, not bytes
        return arr

    # TODO: make this a static method
    # TODO: refactor this to a helper class?
    def _return_data(self, data, name, dtype, return_value):
        if return_value == 'int_d':
            if len(data) == 0:
                # empty, give it a NULL
                return 0, 0
            else:
                # return int as void*
                data = cp.asarray(data, dtype=dtype)
                setattr(self, name, data)  # keep data alive
                return data.data.ptr, data.size
        if return_value == 'int_h':
            if len(data) == 0:
                # empty, give it a NULL
                return 0, 0
            else:
                # return int as void*
                data = np.asarray(data, dtype=dtype)
                setattr(self, name, data)  # keep data alive
                return data.ctypes.data, data.size
        elif return_value == 'seq':
            # data itself is already a flat sequence
            return data, len(data)
        else:
            assert False


@pytest.fixture()
def multi_gpu_handles(request):
    # TODO: consider making this class more flexible
    # (ex: arbitrary number of qubits and/or devices, etc)
    n_devices = 2  # should be power of 2
    handles = []
    p2p_required = request.param

    for dev in range(n_devices):
        with cp.cuda.Device(dev):
            h = cusv.create()
            handles.append(h)
            if p2p_required:
                for peer in range(n_devices):
                    if dev == peer: continue
                    try:
                        cp.cuda.runtime.deviceEnablePeerAccess(peer)
                    except Exception as e:
                        if 'PeerAccessUnsupported' in str(e):
                            pytest.skip("P2P unsupported")
                        if 'PeerAccessAlreadyEnabled' not in str(e):
                            raise

    yield handles

    for dev in range(n_devices):
        with cp.cuda.Device(dev):
            h = handles.pop(0)
            cusv.destroy(h)
            if p2p_required:
                for peer in range(n_devices):
                    if dev == peer: continue
                    try:
                        cp.cuda.runtime.deviceDisablePeerAccess(peer)
                    except Exception as e:
                        if 'PeerAccessNotEnabled' not in str(e):
                            raise


def get_exponent(n):
    assert (n % 2) == 0
    exponent = 1
    while True:
        out = n >> exponent
        if out != 1:
            exponent += 1
        else:
            break
    return exponent


@testing.parameterize(*testing.product({
    'n_qubits': (4,),
    'dtype': (np.complex64, np.complex128),
}))
class TestMultiGpuSV:
    # TODO: consider making this class more flexible
    # (ex: arbitrary number of qubits and/or devices, etc)
    n_devices = 2  # should be power of 2

    def get_sv(self):
        self.n_global_bits = get_exponent(self.n_devices)
        self.n_local_bits = self.n_qubits - self.n_global_bits

        self.sub_sv = []
        for dev in range(self.n_devices):
            with cp.cuda.Device(dev):
                self.sub_sv.append(cp.zeros(
                    2**self.n_local_bits, dtype=self.dtype))
        self.sub_sv[0][0] = 1  # initialize in |000...00>
        return self.sub_sv

    # TODO: make this a static method
    def _return_data(self, data, name, dtype, return_value):
        if return_value == 'int':
            if len(data) == 0:
                # empty, give it a NULL
                return 0, 0
            else:
                # return int as void*
                data = np.asarray(data, dtype=dtype)
                setattr(self, name, data)  # keep data alive
                return data.ctypes.data, data.size
        elif return_value == 'seq':
            # data itself is already a flat sequence
            return data, len(data)
        else:
            assert False


class TestLibHelper:

    def test_get_version(self):
        ver = cusv.get_version()
        major = ver // 1000
        minor = (ver % 1000) // 100

        # run-time version must be compatible with build-time version
        assert major == cusv.MAJOR_VER
        assert minor >= cusv.MINOR_VER

        # sanity check (build-time versions should agree)
        assert cusv.VERSION == (cusv.MAJOR_VER * 1000
            + cusv.MINOR_VER * 100
            + cusv.PATCH_VER)

    def test_get_property(self):
        # run-time version must be compatible with build-time version
        assert cusv.MAJOR_VER == cusv.get_property(
            cuquantum.libraryPropertyType.MAJOR_VERSION)
        assert cusv.MINOR_VER <= cusv.get_property(
            cuquantum.libraryPropertyType.MINOR_VERSION)


class TestHandle:

    def test_handle_create_destroy(self, handle):
        # simple rount-trip test
        pass

    def test_workspace(self, handle):
        default_workspace_size = cusv.get_default_workspace_size(handle)
        # this is about 18MB as of cuQuantum beta 1
        assert default_workspace_size > 0
        # cuStateVec does not like a smaller workspace...
        size = 24*1024**2
        assert size > default_workspace_size
        memptr = cp.cuda.alloc(size)
        cusv.set_workspace(handle, memptr.ptr, size)  # should not fail

    def test_stream(self, handle):
        # default is on the null stream
        assert 0 == cusv.get_stream(handle)

        # simple set/get round-trip
        stream = cp.cuda.Stream()
        cusv.set_stream(handle, stream.ptr)
        assert stream.ptr == cusv.get_stream(handle)


class TestInitSV(TestSV):

    @pytest.mark.parametrize('sv_type', cusv.StateVectorType)
    def test_initialize_state_vector(self, handle, sv_type):
        sv = self.get_sv()
        data_type = dtype_to_data_type[self.dtype]

        if sv_type == cusv.StateVectorType.ZERO:
            sv_orig = sv.copy()  # already zero-init'd
            sv[:] = 1.  # reset to something else
        cusv.initialize_state_vector(
            handle, sv.data.ptr, data_type, self.n_qubits, sv_type)
        if sv_type == cusv.StateVectorType.ZERO:
            assert (sv == sv_orig).all()
        assert cp.allclose(cp.sum(cp.abs(sv)**2), 1.)


class TestAbs2Sum(TestSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'basis_bits': (np.int32, 'int'),},
            {'basis_bits': (np.int32, 'seq'),},
        )
    )
    def test_abs2sum_on_z_basis(self, handle, input_form):
        sv = self.get_sv()
        basis_bits = list(range(self.n_qubits))
        basis_bits, basis_bits_len = self._return_data(
            basis_bits, 'basis_bits', *input_form['basis_bits'])
        data_type = dtype_to_data_type[self.dtype]

        # case 1: both are computed
        sum0, sum1 = cusv.abs2sum_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            True, True, basis_bits, basis_bits_len)
        assert np.allclose(sum0+sum1, 1)
        assert (sum0 is not None) and (sum1 is not None)

        # case 2: only sum0 is computed
        sum0, sum1 = cusv.abs2sum_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            True, False, basis_bits, basis_bits_len)
        assert np.allclose(sum0, 1)
        assert (sum0 is not None) and (sum1 is None)

        # case 3: only sum1 is computed
        sum0, sum1 = cusv.abs2sum_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            False, True, basis_bits, basis_bits_len)
        assert np.allclose(sum1, 0)
        assert (sum0 is None) and (sum1 is not None)

        # case 4: none is computed
        with pytest.raises(ValueError):
            sum0, sum1 = cusv.abs2sum_on_z_basis(
                handle, sv.data.ptr, data_type, self.n_qubits,
                False, False, basis_bits, basis_bits_len)

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (np.int32, 'int'),},
            {'bit_ordering': (np.int32, 'seq'),},
        )
    )
    @pytest.mark.parametrize(
        'xp', (np, cp)
     )
    def test_abs2sum_array_no_mask(self, handle, xp, input_form):
        # change sv from |000> to 1/\sqrt{2} (|001> + |100>)
        sv = self.get_sv()
        sv[0] = 0
        sv[1] = 1./np.sqrt(2)
        sv[4] = 1./np.sqrt(2)

        data_type = dtype_to_data_type[self.dtype]
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, bit_ordering_len = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])
        # test abs2sum on both host and device
        abs2sum = xp.zeros((2**bit_ordering_len,), dtype=xp.float64)
        abs2sum_ptr = abs2sum.data.ptr if xp is cp else abs2sum.ctypes.data
        cusv.abs2sum_array(
            handle, sv.data.ptr, data_type, self.n_qubits, abs2sum_ptr,
            bit_ordering, bit_ordering_len, 0, 0, 0)
        assert xp.allclose(abs2sum.sum(), 1)
        assert xp.allclose(abs2sum[1], 0.5)
        assert xp.allclose(abs2sum[4], 0.5)

    # TODO(leofang): add more tests for abs2sum_array, such as nontrivial masks


class TestBatchedAbs2Sum(TestBatchedSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (np.int32, 'int_h'),},
            {'bit_ordering': (np.int32, 'seq'),},
        )
    )
    @pytest.mark.parametrize(
        'xp', (np, cp)
     )
    def test_abs2sum_array_batched_no_mask(self, handle, xp, input_form):
        # change sv from |0000> to 1/\sqrt{2} (|0001> + |1000>)
        sv = self.get_sv()
        sv[..., 0] = 0
        sv[..., 1] = 1./np.sqrt(2)
        sv[..., 8] = 1./np.sqrt(2)

        data_type = dtype_to_data_type[self.dtype]
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, bit_ordering_len = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])

        # test abs2sum on both host and device
        abs2sum = xp.zeros((self.n_svs, 2**bit_ordering_len,),
                           dtype=xp.float64)
        abs2sum_ptr = abs2sum.data.ptr if xp is cp else abs2sum.ctypes.data
        cusv.abs2sum_array_batched(
            handle, sv.data.ptr, data_type, self.n_qubits,
            self.n_svs, self.sv_stride,
            abs2sum_ptr, 2**bit_ordering_len,
            bit_ordering, bit_ordering_len, 0, 0, 0)

        assert xp.allclose(abs2sum.sum(), self.n_svs)
        assert xp.allclose(abs2sum[..., 1], 0.5)
        assert xp.allclose(abs2sum[..., 8], 0.5)

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (np.int32, 'int_h'), 'mask_bit_strings': (np.int64, 'int_h'), },
            {'bit_ordering': (np.int32, 'int_h'), 'mask_bit_strings': (np.int64, 'int_d'), },
            {'bit_ordering': (np.int32, 'seq'), 'mask_bit_strings': (np.int64, 'seq'), },
        )
    )
    @pytest.mark.parametrize(
        'xp', (np, cp)
     )
    def test_abs2sum_array_batched_masked(self, handle, xp, input_form):
        # change sv from |0000> to 1/\sqrt{2} (|0001> + |1000>)
        sv = self.get_sv()
        sv[..., 0] = 0
        sv[..., 1] = 1./np.sqrt(2)
        sv[..., 8] = 1./np.sqrt(2)

        data_type = dtype_to_data_type[self.dtype]
        bit_ordering = list(range(self.n_qubits - 1))  # exclude the last qubit
        bit_ordering, bit_ordering_len = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])

        # mask = 0b1
        mask_bit_strings = np.ones(self.n_svs, dtype=np.int64)
        mask_bit_strings, _ = self._return_data(
            mask_bit_strings, 'mask_bit_strings',
            *input_form['mask_bit_strings'])
        mask_bit_ordering = [self.n_qubits - 1]
        mask_bit_ordering, mask_len = self._return_data(
            mask_bit_ordering, 'mask_bit_ordering', *input_form['bit_ordering'])

        # test abs2sum on both host and device
        abs2sum = xp.zeros((self.n_svs, 2**bit_ordering_len,),
                           dtype=xp.float64)
        abs2sum_ptr = abs2sum.data.ptr if xp is cp else abs2sum.ctypes.data
        cusv.abs2sum_array_batched(
            handle, sv.data.ptr, data_type, self.n_qubits,
            self.n_svs, self.sv_stride,
            abs2sum_ptr, 2**bit_ordering_len,
            bit_ordering, bit_ordering_len,
            mask_bit_strings, mask_bit_ordering, mask_len)

        # we mask out half of the values
        assert xp.allclose(abs2sum.sum(), self.n_svs * 0.5)
        assert xp.allclose(abs2sum[..., 0], 0.5)


class TestCollapse(TestSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'basis_bits': (np.int32, 'int'),},
            {'basis_bits': (np.int32, 'seq'),},
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
        data_type = dtype_to_data_type[self.dtype]

        cusv.collapse_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            parity, basis_bits, basis_bits_len, 1)

        if parity == 0:
            assert cp.allclose(sv.sum(), 1)
        elif parity == 1:
            assert cp.allclose(sv.sum(), 0)

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (np.int32, 'int'), 'bitstring': (np.int32, 'int')},
            {'bit_ordering': (np.int32, 'seq'), 'bitstring': (np.int32, 'seq')},
        )
    )
    def test_collapse_by_bitstring(self, handle, input_form):
        # change sv to 1/\sqrt{2} (|000> + |111>)
        sv = self.get_sv()
        sv[0] = np.sqrt(0.5)
        sv[-1] = np.sqrt(0.5)

        # collapse to |111>
        bitstring = [1] * self.n_qubits
        bitstring, bitstring_len = self._return_data(
            bitstring, 'bitstring', *input_form['bitstring'])

        bit_ordering = list(range(self.n_qubits))
        bit_ordering, _ = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])
        data_type = dtype_to_data_type[self.dtype]

        norm = 0.5
        # the sv after collapse is normalized as sv -> sv / \sqrt{norm}
        cusv.collapse_by_bitstring(
            handle, sv.data.ptr, data_type, self.n_qubits,
            bitstring, bit_ordering, bitstring_len,
            norm)
        assert cp.allclose(sv.sum(), 1)
        assert cp.allclose(sv[-1], 1)


class TestBatchedCollapse(TestBatchedSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (np.int32, 'int_h'), 'bitstrings': (np.int64, 'int_d'), 'norms': (np.double, 'int_d')},
            {'bit_ordering': (np.int32, 'int_h'), 'bitstrings': (np.int64, 'int_h'), 'norms': (np.double, 'int_h')},
            {'bit_ordering': (np.int32, 'seq'), 'bitstrings': (np.int64, 'seq'), 'norms': (np.double, 'seq')},
        )
    )
    def test_collapse_by_bitstring_batched(self, handle, input_form):
        # change sv to 1/\sqrt{2} (|00...0> + |11...1>)
        sv = self.get_sv()
        sv[:, 0] = np.sqrt(0.5)
        sv[:, 2**self.n_qubits-1] = np.sqrt(0.5)  # Note the padding at the end

        bit_ordering = list(range(self.n_qubits))
        bit_ordering, _ = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])
        bitstrings_len = self.n_qubits
        data_type = dtype_to_data_type[self.dtype]

        # collapse to |11...1>
        bitstrings = [2**self.n_qubits-1] * self.n_svs
        bitstrings, _ = self._return_data(
            bitstrings, 'bitstrings', *input_form['bitstrings'])

        # the sv after collapse is normalized as sv -> sv / \sqrt{norm}
        norms = [0.5] * self.n_svs
        norms, _ = self._return_data(
            norms, 'norms', *input_form['norms'])

        workspace_size = cusv.collapse_by_bitstring_batched_get_workspace_size(
            handle, self.n_svs, bitstrings, norms)
        if workspace_size > 0:
            workspace = cp.cuda.alloc(workspace_size)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        cusv.collapse_by_bitstring_batched(
            handle, sv.data.ptr, data_type, self.n_qubits,
            self.n_svs, self.sv_stride,
            bitstrings, bit_ordering, bitstrings_len,
            norms,
            workspace_ptr, workspace_size)
        cp.cuda.Device().synchronize()
        assert cp.allclose(sv[:, 0:2**self.n_qubits].sum(), self.n_svs)
        assert cp.allclose(sv[:, 2**self.n_qubits-1], cp.ones(self.n_svs, dtype=self.dtype))


@pytest.mark.parametrize(
    'rand',
    # the choices here ensure we get either parity
    (0, np.nextafter(1, 0))
)
@pytest.mark.parametrize(
    'collapse',
    (cusv.Collapse.NORMALIZE_AND_ZERO, cusv.Collapse.NONE)
)
class TestMeasure(TestSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'basis_bits': (np.int32, 'int'),},
            {'basis_bits': (np.int32, 'seq'),},
        )
    )
    def test_measure_on_z_basis(self, handle, rand, collapse, input_form):
        # change the sv to 1/\sqrt{2} (|000> + |010>) to allow 50-50 chance
        # of getting either parity
        sv = self.get_sv()
        sv[0] = np.sqrt(0.5)
        sv[2] = np.sqrt(0.5)

        basis_bits = list(range(self.n_qubits))
        basis_bits, basis_bits_len = self._return_data(
            basis_bits, 'basis_bits', *input_form['basis_bits'])
        data_type = dtype_to_data_type[self.dtype]
        orig_sv = sv.copy()

        parity = cusv.measure_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            basis_bits, basis_bits_len, rand, collapse)

        if collapse == cusv.Collapse.NORMALIZE_AND_ZERO:
            if parity == 0:
                # collapse to |000>
                assert cp.allclose(sv[0], 1)
            elif parity == 1:
                # collapse to |111>
                assert cp.allclose(sv[2], 1)
            # sv is collapsed
            assert not (sv == orig_sv).all()
        else:
            # sv is intact
            assert (sv == orig_sv).all()

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (np.int32, 'int'),},
            {'bit_ordering': (np.int32, 'seq'),},
        )
    )
    def test_batch_measure(self, handle, rand, collapse, input_form):
        # change sv to 1/\sqrt{2} (|000> + |111>)
        sv = self.get_sv()
        sv[0] = np.sqrt(0.5)
        sv[-1] = np.sqrt(0.5)
        orig_sv = sv.copy()

        data_type = dtype_to_data_type[self.dtype]
        bitstring = np.empty(self.n_qubits, dtype=np.int32)
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, _ = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])

        cusv.batch_measure(
            handle, sv.data.ptr, data_type, self.n_qubits,
            bitstring.ctypes.data, bit_ordering, bitstring.size,
            rand, collapse)

        if bitstring.sum() == 0:
            assert rand == 0
            if collapse == cusv.Collapse.NORMALIZE_AND_ZERO:
                # collapse to |000>
                assert cp.allclose(sv[0], 1)
                # sv is collapsed
                assert (sv != orig_sv).any()
            else:
                # sv is intact
                assert (sv == orig_sv).all()
        elif bitstring.sum() == self.n_qubits:
            assert rand == np.nextafter(1, 0)
            if collapse == cusv.Collapse.NORMALIZE_AND_ZERO:
                # collapse to |111>
                assert cp.allclose(sv[-1], 1)
                # sv is collapsed
                assert (sv != orig_sv).any()
            else:
                # sv is intact
                assert (sv == orig_sv).all()
        else:
            assert False, f"unexpected bitstrings: {bitstrings}"


class TestMeasureBatched(TestBatchedSV):

    @pytest.mark.parametrize(
        'rand',
        # the choices here ensure we get either parity
        (0, np.nextafter(1, 0))
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'bitstrings': (np.int64, 'int_h'), 'bit_ordering': (np.int32, 'int_h'), 'rand_nums': (np.float64, 'int_h')},
            {'bitstrings': (np.int64, 'int_d'), 'bit_ordering': (np.int32, 'int_h'), 'rand_nums': (np.float64, 'int_d')},
            {'bitstrings': (np.int64, 'int_d'), 'bit_ordering': (np.int32, 'seq'), 'rand_nums': (np.float64, 'seq')},
        )
    )
    @pytest.mark.parametrize('collapse', cusv.Collapse)
    @pytest.mark.parametrize('xp', (np, cp))
    def test_measure_batched(self, handle, rand, input_form, collapse, xp):
        # change sv to 1/\sqrt{2} (|00...0> + |11...1>)
        sv = self.get_sv()
        sv[:, 0] = np.sqrt(0.5)
        sv[:, 2**self.n_qubits-1] = np.sqrt(0.5)  # Note the padding at the end
        orig_sv = sv.copy()

        data_type = dtype_to_data_type[self.dtype]
        bitstrings = np.empty(self.n_svs, dtype=np.int32)
        bitstrings, _ = self._return_data(
            bitstrings, 'bitstrings', *input_form['bitstrings'])
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, bit_ordering_len = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])
        rand_nums = [rand] * self.n_svs
        rand_nums, _ = self._return_data(
            rand_nums, 'rand_nums', *input_form['rand_nums'])

        cusv.measure_batched(
            handle, sv.data.ptr, data_type, self.n_qubits,
            self.n_svs, self.sv_stride,
            bitstrings, bit_ordering, bit_ordering_len,
            rand_nums, collapse)

        bitstrings = self.bitstrings
        if bitstrings.sum() == 0:
            assert rand == 0
            if collapse == cusv.Collapse.NORMALIZE_AND_ZERO:
                # collapse to |00...0>
                assert cp.allclose(sv[:, 0], 1)
                # sv is collapsed
                assert (sv != orig_sv).any()
            else:
                # sv is intact
                assert (sv == orig_sv).all()
        elif bitstrings.sum() == (2**self.n_qubits-1)*self.n_svs:
            assert rand == np.nextafter(1, 0)
            if collapse == cusv.Collapse.NORMALIZE_AND_ZERO:
                # collapse to |11...1>
                assert cp.allclose(sv[:, 2**self.n_qubits-1], 1)
                # sv is collapsed
                assert (sv != orig_sv).any()
            else:
                # sv is intact
                assert (sv == orig_sv).all()
        else:
            assert False, f"unexpected bitstrings: {bitstrings}"


class TestApply(TestSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'targets': (np.int32, 'int'), 'controls': (np.int32, 'int'),
             # sizeof(enum) == sizeof(int)
             'paulis': (np.int32, 'int'),},
            {'targets': (np.int32, 'seq'), 'controls': (np.int32, 'seq'),
             'paulis': (np.int32, 'seq'),},
        )
    )
    def test_apply_pauli_rotation(self, handle, input_form):
        # change sv to |100>
        sv = self.get_sv()
        sv[0] = 0
        sv[4] = 1

        data_type = dtype_to_data_type[self.dtype]
        targets = [0, 1]
        targets, targets_len = self._return_data(
            targets, 'targets', *input_form['targets'])
        controls = [2]
        controls, controls_len = self._return_data(
            controls, 'controls', *input_form['controls'])
        control_values = 0  # set all control bits to 1
        paulis = [cusv.Pauli.X, cusv.Pauli.X]
        paulis, _ = self._return_data(
            paulis, 'paulis', *input_form['paulis'])

        cusv.apply_pauli_rotation(
            handle, sv.data.ptr, data_type, self.n_qubits,
            0.5*np.pi, paulis,
            targets, targets_len,
            controls, control_values, controls_len)
        sv *= -1j

        # result is |111>
        assert cp.allclose(sv[-1], 1)

    @pytest.mark.parametrize(
        'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'targets': (np.int32, 'int'), 'controls': (np.int32, 'int')},
            {'targets': (np.int32, 'seq'), 'controls': (np.int32, 'seq')},
        )
    )
    @pytest.mark.parametrize(
        'xp', (np, cp)
     )
    def test_apply_matrix(self, handle, xp, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        sv = self.get_sv()
        data_type = dtype_to_data_type[self.dtype]
        compute_type = dtype_to_compute_type[self.dtype]
        targets = [0, 1, 2]
        targets, targets_len = self._return_data(
            targets, 'targets', *input_form['targets'])
        controls = []
        controls, controls_len = self._return_data(
            controls, 'controls', *input_form['controls'])

        # matrix can live on host or device
        matrix = xp.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=sv.dtype)
        matrix[-1][0] = 1
        matrix_ptr = matrix.ctypes.data if xp is np else matrix.data.ptr

        if mempool is None:
            workspace_size = cusv.apply_matrix_get_workspace_size(
                handle, data_type, self.n_qubits,
                matrix_ptr, data_type, cusv.MatrixLayout.ROW, 0,
                targets_len, controls_len, compute_type)
            if workspace_size:
                workspace = cp.cuda.alloc(workspace_size)
                workspace_ptr = workspace.ptr
            else:
                workspace_ptr = 0
        else:
            mr = MemoryResourceFactory(mempool)
            handler = mr.get_dev_mem_handler()
            cusv.set_device_mem_handler(handle, handler)

            workspace_ptr = 0
            workspace_size = 0

        cusv.apply_matrix(
            handle, sv.data.ptr, data_type, self.n_qubits,
            matrix_ptr, data_type, cusv.MatrixLayout.ROW, 0,
            targets, targets_len,
            controls, 0, controls_len,
            compute_type, workspace_ptr, workspace_size)

        assert sv[-1] == 1  # output state is |111>

    @pytest.mark.parametrize(
        'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'permutation': (np.int64, 'int'), 'basis_bits': (np.int32, 'int'),
             'mask_bitstring': (np.int32, 'int'), 'mask_ordering': (np.int32, 'int')},
            {'permutation': (np.int64, 'seq'), 'basis_bits': (np.int32, 'seq'),
             'mask_bitstring': (np.int32, 'seq'), 'mask_ordering': (np.int32, 'seq')},
        )
    )
    @pytest.mark.parametrize(
        'xp', (np, cp)
     )
    def test_apply_generalized_permutation_matrix(
            self, handle, xp, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        sv = self.get_sv()
        sv[:] = 1  # invalid sv just to make math checking easier
        data_type = dtype_to_data_type[self.dtype]
        compute_type = dtype_to_compute_type[self.dtype]

        # TODO(leofang): test permutation on either host or device
        permutation = list(np.random.permutation(2**self.n_qubits))
        permutation_data = permutation
        permutation, permutation_len = self._return_data(
            permutation, 'permutation', *input_form['permutation'])

        # diagonal can live on host or device
        diagonal = 10 * xp.ones((2**self.n_qubits, ), dtype=sv.dtype)
        diagonal_ptr = diagonal.ctypes.data if xp is np else diagonal.data.ptr

        basis_bits = list(range(self.n_qubits))
        basis_bits, basis_bits_len = self._return_data(
            basis_bits, 'basis_bits', *input_form['basis_bits'])

        # TODO(leofang): test masks
        mask_bitstring = 0
        mask_ordering = 0
        mask_len = 0

        if mempool is None:
            workspace_size = cusv.apply_generalized_permutation_matrix_get_workspace_size(
                handle, data_type, self.n_qubits,
                permutation, diagonal_ptr, data_type,
                basis_bits, basis_bits_len, mask_len)

            if workspace_size:
                workspace = cp.cuda.alloc(workspace_size)
                workspace_ptr = workspace.ptr
            else:
                workspace_ptr = 0
        else:
            mr = MemoryResourceFactory(mempool)
            handler = mr.get_dev_mem_handler()
            cusv.set_device_mem_handler(handle, handler)

            workspace_ptr = 0
            workspace_size = 0

        cusv.apply_generalized_permutation_matrix(
            handle, sv.data.ptr, data_type, self.n_qubits,
            permutation, diagonal_ptr, data_type, 0,
            basis_bits, basis_bits_len,
            mask_bitstring, mask_ordering, mask_len,
            workspace_ptr, workspace_size)

        assert cp.allclose(sv, diagonal[xp.asarray(permutation_data)])


class TestBatchedApply(TestBatchedSV):

    @pytest.mark.parametrize(
        'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'matrix_indices': (np.int32, 'int_h'), 'targets': (np.int32, 'int_h'), 'controls': (np.int32, 'int_h')},
            {'matrix_indices': (np.int32, 'int_d'), 'targets': (np.int32, 'int_h'), 'controls': (np.int32, 'int_h')},
            {'matrix_indices': (np.int32, 'seq'), 'targets': (np.int32, 'seq'), 'controls': (np.int32, 'seq')},
        )
    )
    @pytest.mark.parametrize('xp', (np, cp))
    @pytest.mark.parametrize('map_type', cusv.MatrixMapType)
    def test_apply_matrix_batched(
            self, handle, map_type, xp, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        sv = self.get_sv()
        data_type = dtype_to_data_type[self.dtype]
        compute_type = dtype_to_compute_type[self.dtype]
        targets = list(range(self.n_qubits))
        targets, targets_len = self._return_data(
            targets, 'targets', *input_form['targets'])
        controls = []
        controls, controls_len = self._return_data(
            controls, 'controls', *input_form['controls'])

        if map_type == cusv.MatrixMapType.BROADCAST:
            n_matrices = 1
        elif map_type == cusv.MatrixMapType.MATRIX_INDEXED:
            n_matrices = self.n_svs

        # matrices and their indices can live on host or device
        matrices = xp.zeros(
            (n_matrices, 2**self.n_qubits, 2**self.n_qubits),
            dtype=sv.dtype)
        matrices[..., -1, 0] = 1
        matrices_ptr = matrices.ctypes.data if xp is np else matrices.data.ptr
        matrix_indices = list(range(n_matrices))
        if len(matrix_indices) > 1:
            np.random.shuffle(matrix_indices)
        matrix_indices, n_matrices = self._return_data(
            matrix_indices, 'matrix_indices', *input_form['matrix_indices'])

        if mempool is None:
            workspace_size = cusv.apply_matrix_batched_get_workspace_size(
                handle, data_type, self.n_qubits, self.n_svs, self.sv_stride,
                map_type, matrix_indices, matrices_ptr, data_type,
                cusv.MatrixLayout.ROW, 0, n_matrices,
                targets_len, controls_len, compute_type)
            if workspace_size:
                workspace = cp.cuda.alloc(workspace_size)
                workspace_ptr = workspace.ptr
            else:
                workspace_ptr = 0
        else:
            mr = MemoryResourceFactory(mempool)
            handler = mr.get_dev_mem_handler()
            cusv.set_device_mem_handler(handle, handler)

            workspace_ptr = 0
            workspace_size = 0

        cusv.apply_matrix_batched(
            handle, sv.data.ptr, data_type, self.n_qubits,
            self.n_svs, self.sv_stride, map_type, matrix_indices,
            matrices_ptr, data_type,
            cusv.MatrixLayout.ROW, 0, n_matrices,
            targets, targets_len,
            controls, 0, controls_len,
            compute_type, workspace_ptr, workspace_size)

        assert (sv[..., 2**self.n_qubits-1] == 1).all()  # output state is |11...1>


class TestExpect(TestSV):

    @pytest.mark.parametrize(
        'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'basis_bits': (np.int32, 'int'),},
            {'basis_bits': (np.int32, 'seq'),},
        )
    )
    @pytest.mark.parametrize(
        'expect_dtype', (np.float64, np.complex128)
    )
    @pytest.mark.parametrize(
        'xp', (np, cp)
    )
    def test_compute_expectation(self, handle, xp, expect_dtype, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        # create a uniform sv
        sv = self.get_sv()
        sv[:] = np.sqrt(1/(2**self.n_qubits))

        data_type = dtype_to_data_type[self.dtype]
        compute_type = dtype_to_compute_type[self.dtype]
        basis_bits = list(range(self.n_qubits))
        basis_bits, basis_bits_len = self._return_data(
            basis_bits, 'basis_bits', *input_form['basis_bits'])

        # matrix can live on host or device
        matrix = xp.ones((2**self.n_qubits, 2**self.n_qubits), dtype=sv.dtype)
        matrix_ptr = matrix.ctypes.data if xp is np else matrix.data.ptr

        if mempool is None:
            workspace_size = cusv.compute_expectation_get_workspace_size(
                handle, data_type, self.n_qubits,
                matrix_ptr, data_type, cusv.MatrixLayout.ROW,
                basis_bits_len, compute_type)
            if workspace_size:
                workspace = cp.cuda.alloc(workspace_size)
                workspace_ptr = workspace.ptr
            else:
                workspace_ptr = 0
        else:
            mr = MemoryResourceFactory(mempool)
            handler = mr.get_dev_mem_handler()
            cusv.set_device_mem_handler(handle, handler)

            workspace_ptr = 0
            workspace_size = 0

        expect = np.empty((1,), dtype=expect_dtype)
        # TODO(leofang): check if this is relaxed in beta 2
        expect_data_type = (
            cudaDataType.CUDA_R_64F if expect_dtype == np.float64
            else cudaDataType.CUDA_C_64F)

        cusv.compute_expectation(
            handle, sv.data.ptr, data_type, self.n_qubits,
            expect.ctypes.data, expect_data_type,
            matrix_ptr, data_type, cusv.MatrixLayout.ROW,
            basis_bits, basis_bits_len,
            compute_type, workspace_ptr, workspace_size)

        assert xp.allclose(expect, 2**self.n_qubits)

    # TODO: test other input forms?
    def test_compute_expectations_on_pauli_basis(self, handle):
        # create a uniform sv
        sv = self.get_sv()
        sv[:] = np.sqrt(1/(2**self.n_qubits))
        data_type = dtype_to_data_type[self.dtype]
        compute_type = dtype_to_compute_type[self.dtype]

        # measure XX...X, YY..Y, ZZ...Z
        paulis = [[cusv.Pauli.X for i in range(self.n_qubits)],
                  [cusv.Pauli.Y for i in range(self.n_qubits)],
                  [cusv.Pauli.Z for i in range(self.n_qubits)],]

        basis_bits = [[*range(self.n_qubits)] for i in range(len(paulis))]
        n_basis_bits = [len(basis_bits[i]) for i in range(len(paulis))]
        expect = np.empty((len(paulis),), dtype=np.float64)

        cusv.compute_expectations_on_pauli_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            expect.ctypes.data, paulis, len(paulis),
            basis_bits, n_basis_bits)

        result = np.zeros_like(expect)
        result[0] = 1  # for XX...X
        assert np.allclose(expect, result)

class TestBatchedExpect(TestBatchedSV):

    @pytest.mark.parametrize(
        'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'basis_bits': (np.int32, 'int_h')},
            {'basis_bits': (np.int32, 'seq')},
        )
    )
    @pytest.mark.parametrize('xp', (np, cp))
    def test_compute_expectation_batched(
            self, handle, xp, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        sv = self.get_sv()
        data_type = dtype_to_data_type[self.dtype]
        compute_type = dtype_to_compute_type[self.dtype]
        basis_bits = list(range(self.n_qubits))
        basis_bits, n_basis_bits = self._return_data(
            basis_bits, 'basis_bits', *input_form['basis_bits'])

        # matrices can live on host or device
        n_matrices = 2
        matrix_dim = 2**self.n_qubits
        matrices = xp.ones(
            (n_matrices, matrix_dim, matrix_dim),
            dtype=sv.dtype)
        matrices_ptr = matrices.ctypes.data if xp is np else matrices.data.ptr

        if mempool is None:
            workspace_size = cusv.compute_expectation_batched_get_workspace_size(
                handle, data_type, self.n_qubits, self.n_svs, self.sv_stride,
                matrices_ptr, data_type, cusv.MatrixLayout.ROW, n_matrices,
                n_basis_bits, compute_type)
            if workspace_size:
                workspace = cp.cuda.alloc(workspace_size)
                workspace_ptr = workspace.ptr
            else:
                workspace_ptr = 0
        else:
            mr = MemoryResourceFactory(mempool)
            handler = mr.get_dev_mem_handler()
            cusv.set_device_mem_handler(handle, handler)

            workspace_ptr = 0
            workspace_size = 0

        expect = np.empty((n_matrices * self.n_svs,), dtype=np.complex128)
        cusv.compute_expectation_batched(
            handle, sv.data.ptr, data_type, self.n_qubits, self.n_svs, self.sv_stride,
            expect.ctypes.data, matrices_ptr, data_type, cusv.MatrixLayout.ROW, n_matrices,
            basis_bits, n_basis_bits, compute_type, workspace_ptr, workspace_size)

        assert (np.allclose(expect, 1))

class TestSampler(TestSV):

    @pytest.mark.parametrize(
        'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (np.int32, 'int'),},
            {'bit_ordering': (np.int32, 'seq'),},
        )
    )
    def test_sampling(self, handle, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        # create a uniform sv
        sv = self.get_sv()
        sv[:] = np.sqrt(1/(2**self.n_qubits))

        data_type = dtype_to_data_type[self.dtype]
        compute_type = dtype_to_compute_type[self.dtype]
        shots = 4096

        bitstrings = np.empty((shots,), dtype=np.int64)
        rand_nums = np.random.random((shots,)).astype(np.float64)
        # measure all qubits
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, _ = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])

        sampler, workspace_size = cusv.sampler_create(
            handle, sv.data.ptr, data_type, self.n_qubits, shots)
        if mempool is None:
            if workspace_size:
                workspace = cp.cuda.alloc(workspace_size)
                workspace_ptr = workspace.ptr
            else:
                workspace_ptr = 0
        else:
            mr = MemoryResourceFactory(mempool)
            handler = mr.get_dev_mem_handler()
            cusv.set_device_mem_handler(handle, handler)

            workspace_ptr = 0
            workspace_size = 0

        try:
            cusv.sampler_preprocess(
                handle, sampler, workspace_ptr, workspace_size)
            cusv.sampler_sample(
                handle, sampler, bitstrings.ctypes.data,
                bit_ordering, self.n_qubits,
                rand_nums.ctypes.data, shots,
                cusv.SamplerOutput.RANDNUM_ORDER)
            norm = cusv.sampler_get_squared_norm(handle, sampler)

            # TODO: add a multi-GPU test for this API
            # We're being sloppy here by checking a trivial case, which is
            # effectively a no-op. This is just a call check.
            cusv.sampler_apply_sub_sv_offset(
                handle, sampler, 0, 1, 0, norm)
        finally:
            cusv.sampler_destroy(sampler)

        keys, counts = np.unique(bitstrings, return_counts=True)
        # keys are the returned bitstrings 000, 001, ..., 111
        # the sv has all components, and unique() returns a sorted array,
        # so the following should hold:
        assert (keys == np.arange(2**self.n_qubits)).all()

        assert np.allclose(norm, 1)

        # TODO: test counts, which should follow a uniform distribution


@pytest.mark.parametrize(
    'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
)
# TODO(leofang): test mask_bitstring & mask_ordering
@pytest.mark.parametrize(
    'input_form', (
        {'bit_ordering': (np.int32, 'int'), 'mask_bitstring': (np.int32, 'int'), 'mask_ordering': (np.int32, 'int')},
        {'bit_ordering': (np.int32, 'seq'), 'mask_bitstring': (np.int32, 'seq'), 'mask_ordering': (np.int32, 'seq')},
    )
)
@pytest.mark.parametrize(
    'readonly', (True, False)
)
class TestAccessor(TestSV):

    def test_accessor_get(self, handle, readonly, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        # create a monotonically increasing sv
        sv = self.get_sv()
        data = cp.arange(2**self.n_qubits, dtype=sv.dtype)
        data /= cp.sqrt(data**2)
        sv[:] = data

        data_type = dtype_to_data_type[self.dtype]
        compute_type = dtype_to_compute_type[self.dtype]

        # measure all qubits
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, bit_ordering_len = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])
        # TODO(leofang): test them
        mask_bitstring = 0
        mask_ordering = 0
        mask_len = 0

        if readonly:
            accessor_create = cusv.accessor_create_view
        else:
            accessor_create = cusv.accessor_create

        accessor, workspace_size = accessor_create(
            handle, sv.data.ptr, data_type, self.n_qubits,
            bit_ordering, bit_ordering_len,
            mask_bitstring, mask_ordering, mask_len)

        try:
            if mempool is None:
                if workspace_size:
                    workspace = cp.cuda.alloc(workspace_size)
                    workspace_ptr = workspace.ptr
                else:
                    workspace_ptr = 0
            else:
                mr = MemoryResourceFactory(mempool)
                handler = mr.get_dev_mem_handler()
                cusv.set_device_mem_handler(handle, handler)

                workspace_ptr = 0
                workspace_size = 0

            cusv.accessor_set_extra_workspace(
                handle, accessor, workspace_ptr, workspace_size)

            buf_len = 2**2
            buf = cp.empty(buf_len, dtype=sv.dtype)

            # copy the last buf_len elements
            cusv.accessor_get(
                handle, accessor, buf.data.ptr, sv.size-1-buf_len, sv.size-1)
        finally:
            cusv.accessor_destroy(accessor)

        assert (sv[sv.size-1-buf_len: sv.size-1] == buf).all()

    def test_accessor_set(self, handle, readonly, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        # create a monotonically increasing sv
        sv = self.get_sv()
        data = cp.arange(2**self.n_qubits, dtype=sv.dtype)
        data /= cp.sqrt(data**2)
        sv[:] = data

        data_type = dtype_to_data_type[self.dtype]
        compute_type = dtype_to_compute_type[self.dtype]

        # measure all qubits
        bit_ordering = list(range(self.n_qubits))
        bit_ordering, bit_ordering_len = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])
        # TODO(leofang): test them
        mask_bitstring = 0
        mask_ordering = 0
        mask_len = 0

        if readonly:
            accessor_create = cusv.accessor_create_view
        else:
            accessor_create = cusv.accessor_create

        accessor, workspace_size = accessor_create(
            handle, sv.data.ptr, data_type, self.n_qubits,
            bit_ordering, bit_ordering_len,
            mask_bitstring, mask_ordering, mask_len)

        try:
            if mempool is None:
                if workspace_size:
                    workspace = cp.cuda.alloc(workspace_size)
                    workspace_ptr = workspace.ptr
                else:
                    workspace_ptr = 0
            else:
                mr = MemoryResourceFactory(mempool)
                handler = mr.get_dev_mem_handler()
                cusv.set_device_mem_handler(handle, handler)

                workspace_ptr = 0
                workspace_size = 0

            cusv.accessor_set_extra_workspace(
                handle, accessor, workspace_ptr, workspace_size)

            buf_len = 2**2
            buf = cp.zeros(buf_len, dtype=sv.dtype)

            if readonly:
                # copy the last buf_len elements would fail
                with pytest.raises(cusv.cuStateVecError) as e_info:
                    cusv.accessor_set(
                        handle, accessor, buf.data.ptr, sv.size-1-buf_len, sv.size-1)
            else:
                # copy the last buf_len elements
                cusv.accessor_set(
                    handle, accessor, buf.data.ptr, sv.size-1-buf_len, sv.size-1)
        finally:
            cusv.accessor_destroy(accessor)

        if readonly:
            # sv unchanged
            assert (sv[sv.size-1-buf_len: sv.size-1] == data[sv.size-1-buf_len: sv.size-1]).all()
        else:
            assert (sv[sv.size-1-buf_len: sv.size-1] == 0).all()


class TestTestMatrixType:

    @pytest.mark.parametrize(
        'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
    )
    @pytest.mark.parametrize(
        'matrix_type', (cusv.MatrixType.UNITARY, cusv.MatrixType.HERMITIAN)
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'targets': (np.int32, 'int'), },
            {'targets': (np.int32, 'seq'), },
        )
    )
    @pytest.mark.parametrize(
        'dtype', (np.complex64, np.complex128)
    )
    @pytest.mark.parametrize(
        'xp', (np, cp)
     )
    def test_apply_matrix_type(
            self, handle, xp, dtype, input_form, matrix_type, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        data_type = dtype_to_data_type[dtype]
        compute_type = dtype_to_compute_type[dtype]
        n_targets = 4

        # matrix can live on host or device
        # choose a trivial matrix
        data = xp.ones(2**n_targets, dtype=dtype)
        matrix = xp.diag(data)
        matrix_ptr = matrix.ctypes.data if xp is np else matrix.data.ptr

        if mempool is None:
            workspace_size = cusv.test_matrix_type_get_workspace_size(
                handle, matrix_type,
                matrix_ptr, data_type, cusv.MatrixLayout.ROW, n_targets,
                0, compute_type)
            if workspace_size:
                workspace = cp.cuda.alloc(workspace_size)
                workspace_ptr = workspace.ptr
            else:
                workspace_ptr = 0
        else:
            mr = MemoryResourceFactory(mempool)
            handler = mr.get_dev_mem_handler()
            cusv.set_device_mem_handler(handle, handler)

            workspace_ptr = 0
            workspace_size = 0

        residual = cusv.test_matrix_type(
            handle, matrix_type,
            matrix_ptr, data_type, cusv.MatrixLayout.ROW, n_targets,
            0, compute_type, workspace_ptr, workspace_size)
        assert np.isclose(residual, 0)


@pytest.mark.parametrize(
    'rand',
    # the choices here ensure we get either parity
    (0, np.nextafter(1, 0))
)
@pytest.mark.parametrize(
    'collapse',
    (cusv.Collapse.NORMALIZE_AND_ZERO, cusv.Collapse.NONE)
)
@pytest.mark.skipif(
    cp.cuda.runtime.getDeviceCount() < 2, reason='not enough GPUs')
class TestBatchMeasureWithSubSV(TestMultiGpuSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (np.int32, 'int'),},
            {'bit_ordering': (np.int32, 'seq'),},
        )
    )
    @pytest.mark.parametrize(
        'multi_gpu_handles', (False,), indirect=True  # no need for P2P
    )
    def test_batch_measure_with_offset(
            self, multi_gpu_handles, rand, collapse, input_form):
        handles = multi_gpu_handles
        sub_sv = self.get_sv()
        data_type = dtype_to_data_type[self.dtype]
        bit_ordering = list(range(self.n_local_bits))
        bit_ordering, bit_ordering_len = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])

        # change sv to 1/\sqrt{2} (|0000> + |1111>), and compute abs2sum;
        # calling abs2sum_array is also OK, but we focus on testing the target API
        cumulative_array = np.zeros(self.n_devices+1, dtype=np.float64)
        for i_sv in range(self.n_devices):
            with cp.cuda.Device(i_sv):
                if i_sv == 0:
                    # |0 000> is on GPU 0
                    sub_sv[i_sv][0] = np.sqrt(0.5)
                elif i_sv == 1:
                    # |1 111> is on GPU 1
                    sub_sv[i_sv][-1] = np.sqrt(0.5)
                abs2sum = cp.asnumpy(cp.sum(cp.abs(sub_sv[i_sv])**2))
                cumulative_array[i_sv+1] = cumulative_array[i_sv] + abs2sum

        orig_sub_sv = copy.deepcopy(sub_sv)

        bitstring = np.empty(self.n_local_bits, dtype=np.int32)
        norm = cumulative_array[-1]
        for i_sv in range(self.n_devices):
            if (cumulative_array[i_sv] <= rand * norm
                    and (rand * norm < cumulative_array[i_sv+1] or i_sv == self.n_devices-1)):
                global_bits = i_sv
                offset = cumulative_array[i_sv]
                with cp.cuda.Device(i_sv) as dev:
                    cusv.batch_measure_with_offset(
                        handles[i_sv], sub_sv[i_sv].data.ptr, data_type,
                        self.n_local_bits, bitstring.ctypes.data,
                        bit_ordering, bit_ordering_len, rand,
                        collapse, offset, norm)
                    dev.synchronize()
                break
        else:
            assert False

        if global_bits == 0:
            # get |0 000>
            assert (bitstring == 0).all()
        elif global_bits == 1:
            # get |1 111>
            assert (bitstring == 1).all()
        else:
            assert False

        if collapse == cusv.Collapse.NORMALIZE_AND_ZERO:
            # the measured sub sv is collapsed (those not measured are intact!)
            if global_bits == 0:
                # collapse to |0 000>
                with cp.cuda.Device(0):
                    assert cp.allclose(sub_sv[0][0], 1)
                    assert not (sub_sv[0] == orig_sub_sv[0]).all()
                with cp.cuda.Device(1):
                    assert (sub_sv[1] == orig_sub_sv[1]).all()
            elif global_bits == 1:
                # collapse to |1 111>
                with cp.cuda.Device(0):
                    assert (sub_sv[0] == orig_sub_sv[0]).all()
                with cp.cuda.Device(1):
                    assert cp.allclose(sub_sv[1][-1], 1)
                    assert not (sub_sv[1] == orig_sub_sv[1]).all()
            else:
                assert False, f"unexpected bitstring: {bitstring}"
        else:
            # sv is intact
            with cp.cuda.Device(0):
                assert (sub_sv[0] == orig_sub_sv[0]).all()
            with cp.cuda.Device(1):
                assert (sub_sv[1] == orig_sub_sv[1]).all()


class TestSwap:

    @pytest.mark.parametrize(
        'input_form', (
            {'swapped_bits': (np.int32, 'int'),
             'mask_bitstring': (np.int32, 'int'), 'mask_ordering': (np.int32, 'int')},
            {'swapped_bits': (np.int32, 'seq'),
             'mask_bitstring': (np.int32, 'seq'), 'mask_ordering': (np.int32, 'seq')},
        )
    )
    @pytest.mark.parametrize(
        'dtype', (np.complex64, np.complex128)
    )
    def test_swap_index_bits(self, handle, dtype, input_form):
        n_qubits = 4
        sv = cp.zeros(2**n_qubits, dtype=dtype)
        data_type = dtype_to_data_type[dtype]

        # set sv to |0110>
        sv[6] = 1
        orig_sv = sv.copy()

        swapped_bits = [(0, 2), (1, 3)]
        n_swapped_bits = len(swapped_bits)
        if input_form['swapped_bits'][1] == 'int':
            swapped_bits_data = np.asarray(
                swapped_bits, dtype=input_form['swapped_bits'][0])
            swapped_bits = swapped_bits_data.ctypes.data

        # TODO: test mask
        mask_bitstring = 0
        mask_ordering = 0
        mask_len = 0

        cusv.swap_index_bits(
            handle, sv.data.ptr, data_type, n_qubits,
            swapped_bits, n_swapped_bits,
            mask_bitstring, mask_ordering, mask_len)

        # now we should get |1001>
        assert (sv != orig_sv).any()
        assert sv[6] == 0
        assert sv[9] == 1


@pytest.mark.parametrize(
    'topology', [t for t in cusv.DeviceNetworkType]
)
@pytest.mark.skipif(
    cp.cuda.runtime.getDeviceCount() < 2, reason='not enough GPUs')
class TestMultiGPUSwap(TestMultiGpuSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'handles': (np.intp, 'int'), 'sub_svs': (np.intp, 'int'),
             'swapped_bits': (np.int32, 'int'), 'mask': (np.int32, 'int')},
            {'handles': (np.intp, 'seq'), 'sub_svs': (np.intp, 'seq'),
             'swapped_bits': (np.int32, 'seq'), 'mask': (np.int32, 'seq')},
        )
    )
    @pytest.mark.parametrize(
        'multi_gpu_handles', (True,), indirect=True  # need P2P
    )
    def test_multi_device_swap_index_bits(
            self, multi_gpu_handles, input_form, topology):
        # currently the test class sets the following:
        #  - n_global_qubits = 1
        #  - n_local_qubits = 3
        handles = multi_gpu_handles
        n_handles = len(handles)
        sub_sv = self.get_sv()
        data_type = dtype_to_data_type[self.dtype]

        # set sv to |0110> (up to normalization)
        with cp.cuda.Device(0):
            sub_sv[0][0] = 0
            sub_sv[0][-2] = 1

        if input_form['handles'][1] == 'int':
            handles_data = np.asarray(
                handles, dtype=input_form['handles'][0])
            handles = handles_data.ctypes.data
        sub_sv_data = sub_sv
        sub_sv_ptr_data = [arr.data.ptr for arr in sub_sv]
        sub_sv = sub_sv_ptr_data
        if input_form['sub_svs'][1] == 'int':
            sub_sv_ptr_data = np.asarray(
                sub_sv_ptr_data, dtype=input_form['sub_svs'][0])
            sub_sv = sub_sv_ptr_data.ctypes.data
        else:
            sub_sv = sub_sv_ptr_data

        swapped_bits = [(3, 1)]
        n_swapped_bits = len(swapped_bits)
        if input_form['swapped_bits'][1] == 'int':
            swapped_bits_data = np.asarray(
                swapped_bits, dtype=input_form['swapped_bits'][0])
            swapped_bits = swapped_bits_data.ctypes.data

        # TODO: test mask
        mask_bitstring = []
        mask_ordering = []
        mask_len = 0
        if input_form['mask'][1] == 'int':
            mask_bitstring_data = np.asarray(
                mask_bitstring, dtype=input_form['mask'][0])
            mask_bitstring = mask_bitstring_data.ctypes.data
            mask_ordering_data = np.asarray(
                mask_ordering, dtype=input_form['mask'][0])
            mask_ordering = mask_ordering_data.ctypes.data

        cusv.multi_device_swap_index_bits(
            handles, n_handles, sub_sv, data_type,
            self.n_global_bits, self.n_local_bits,
            swapped_bits, n_swapped_bits,
            mask_bitstring, mask_ordering, mask_len,
            topology)

        # now we should get |1100>
        sub_sv = sub_sv_data
        with cp.cuda.Device(0):
            assert sub_sv[0][-2] == 0
        with cp.cuda.Device(1):
            assert sub_sv[1][4] == 1


@pytest.mark.skipif(MPI is None, reason="need mpi4py (& MPI)")
class TestCommunicator:

    @pytest.mark.parametrize(
        "communicator_args", (
            (cusv.CommunicatorType.MPICH, 'libmpi.so'),  # see NVIDIA/cuQuantum#31
            (cusv.CommunicatorType.OPENMPI, ''),
            # TODO: can we use cffi to generate the wrapper lib on the fly?
            (cusv.CommunicatorType.EXTERNAL, ''),
        )
    )
    def test_communicator(self, handle, communicator_args):
        if communicator_args[0] == cusv.CommunicatorType.MPICH:
            vendor = "MPICH"
        elif communicator_args[0] == cusv.CommunicatorType.OPENMPI:
            vendor = "Open MPI"
        else:
            vendor = "n/a"
        comm_name, _ = MPI.get_vendor()
        if comm_name != vendor:
            pytest.skip(f"Using {comm_name}, which mismatches with the "
                        f"requested MPI implementation ({vendor})")
        c = cusv.communicator_create(handle, *communicator_args)
        cusv.communicator_destroy(handle, c)


class TestParameters:

    def test_parameters(self):

        # test constructor
        parameters = cusv.SVSwapParameters()

        # test getter/setter
        parameters.transfer_size = 42
        assert parameters.transfer_size == 42

        # test accessing internal data (numpy.ndarray)
        parameters_arr = parameters.data
        assert parameters_arr.ctypes.data == parameters.ptr

        # test reading/writing the underlying ndarray of custom dtype
        assert parameters_arr.dtype == cusv.sv_swap_parameters_dtype
        assert parameters_arr['transfer_size'] == 42
        parameters_arr['transfer_size'] = 24
        assert parameters_arr['transfer_size'] == 24
        assert parameters.transfer_size == 24

        # test all struct members
        parameters.swap_batch_index        == parameters_arr['swap_batch_index']
        parameters.org_sub_sv_index        == parameters_arr['org_sub_sv_index']
        parameters.dst_sub_sv_index        == parameters_arr['dst_sub_sv_index']
        parameters.org_segment_mask_string == parameters_arr['org_segment_mask_string']
        parameters.dst_segment_mask_string == parameters_arr['dst_segment_mask_string']
        parameters.segment_mask_ordering   == parameters_arr['segment_mask_ordering']
        parameters.segment_mask_len        == parameters_arr['segment_mask_len']
        parameters.n_segment_bits          == parameters_arr['n_segment_bits']
        parameters.data_transfer_type      == parameters_arr['data_transfer_type']
        parameters.transfer_size           == parameters_arr['transfer_size']

        # test alternative constructor & comparison op
        new_parameters = cusv.SVSwapParameters.from_data(parameters_arr)
        assert parameters.data == new_parameters.data
        assert parameters.ptr == new_parameters.ptr
        assert parameters == new_parameters

        new_parameters_arr = np.empty(
            (1,), dtype=cusv.sv_swap_parameters_dtype)
        new_parameters_arr['segment_mask_ordering'][:] = 1
        new_parameters = cusv.SVSwapParameters.from_data(new_parameters_arr)
        assert parameters.data != new_parameters.data
        assert parameters.ptr != new_parameters.ptr
        assert parameters != new_parameters

        # negative tests
        parameters_arr = np.empty(
            (2,), dtype=cusv.sv_swap_parameters_dtype)
        with pytest.raises(ValueError) as e:  # wrong size
            parameters = cusv.SVSwapParameters.from_data(parameters_arr)
        parameters_arr = np.empty(
            (1,), dtype=np.float32)
        with pytest.raises(ValueError) as e:  # wrong dtype
            parameters = cusv.SVSwapParameters.from_data(parameters_arr)
        parameters_arr = "ABC"
        with pytest.raises(ValueError) as e:  # wrong type
            parameters = cusv.SVSwapParameters.from_data(parameters_arr)


class TestWorker:

    event = cp.cuda.Event()
    stream = cp.cuda.Stream()
    sv = cp.zeros((2**4,), dtype=cp.complex64)

    @pytest.mark.parametrize(
        "worker_args", ((sv.data.ptr, 0, event.ptr, cudaDataType.CUDA_C_32F, stream.ptr),)
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'sv': (np.intp, 'int'), 'indices': (np.int32, 'int'),
             'event': (np.intp, 'int')},
            {'sv': (np.intp, 'seq'), 'indices': (np.int32, 'seq'),
             'event': (np.intp, 'seq')},
        )
    )
    @pytest.mark.parametrize(
        'param_form', ('class', 'ndarray', 'int')
    )
    def test_worker(self, handle, worker_args, input_form, param_form):
        worker, extra_size, min_size = cusv.sv_swap_worker_create(
            handle,
            0,  # set the communicator to null, assuming single process
            *worker_args)

        extra_space = cp.cuda.alloc(extra_size)
        cusv.sv_swap_worker_set_extra_workspace(
            handle, worker, extra_space.ptr, extra_size)

        transfer_space = cp.cuda.alloc(min_size)
        cusv.sv_swap_worker_set_transfer_workspace(
            handle, worker, transfer_space.ptr, min_size)

        sv = [self.sv.data.ptr]
        if input_form['sv'][1] == 'int':
            sv_data = np.asarray(
                sv, dtype=input_form['sv'][0])
            sv = sv_data.ctypes.data

        indices = [1]
        if input_form['indices'][1] == 'int':
            indices_data = np.asarray(
                indices, dtype=input_form['indices'][0])
            indices = indices_data.ctypes.data

        dummy = cp.cuda.Event()
        event = [dummy.ptr]
        if input_form['event'][1] == 'int':
            event_data = np.asarray(
                event, dtype=input_form['event'][0])
            event = event_data.ctypes.data

        cusv.sv_swap_worker_set_sub_svs_p2p(
            handle, worker,
            sv, indices, event, 1)

        parameters_data = cusv.SVSwapParameters()
        parameters_data.swap_batch_index = 0
        parameters_data.org_sub_sv_index = 0
        parameters_data.dst_sub_sv_index = 1
        parameters_data.n_segment_bits = 0
        parameters_data.transfer_size = 1
        parameters_data.data_transfer_type = cusv.DataTransferType.NONE
        parameters_data.segment_mask_len = 0
        if param_form == "class":
            parameters = parameters_data
        elif param_form == "ndarray":
            parameters = parameters_data.data
        elif param_form == "int":
            parameters = parameters_data.ptr

        cusv.sv_swap_worker_set_parameters(
            handle, worker, parameters, 1)

        cusv.sv_swap_worker_execute(
            handle, worker, 0, 0)

        cusv.sv_swap_worker_destroy(handle, worker)


class TestScheduler:

    @pytest.mark.parametrize(
        "scheduler_args", ((1, 1),),
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'swapped_bits': (np.int32, 'int'), 'mask': (np.int32, 'int')},
            {'swapped_bits': (np.int32, 'seq'), 'mask': (np.int32, 'seq')},
        )
    )
    @pytest.mark.parametrize(
        'param_form', (None, 'class', 'ndarray', 'int')
    )
    def test_scheduler(self, handle, scheduler_args, input_form, param_form):
        scheduler = cusv.dist_index_bit_swap_scheduler_create(
            handle, *scheduler_args)

        swapped_bits = [(0, 1)]
        n_swapped_bits = len(swapped_bits)
        if input_form['swapped_bits'][1] == 'int':
            swapped_bits_data = np.asarray(
                swapped_bits, dtype=input_form['swapped_bits'][0])
            swapped_bits = swapped_bits_data.ctypes.data

        # TODO: test mask
        mask_bitstring = []
        mask_ordering = []
        mask_len = 0
        if input_form['mask'][1] == 'int':
            mask_bitstring_data = np.asarray(
                mask_bitstring, dtype=input_form['mask'][0])
            mask_bitstring = mask_bitstring_data.ctypes.data
            mask_ordering_data = np.asarray(
                mask_ordering, dtype=input_form['mask'][0])
            mask_ordering = mask_ordering_data.ctypes.data

        n_swap_batches = cusv.dist_index_bit_swap_scheduler_set_index_bit_swaps(
            handle, scheduler,
            swapped_bits, n_swapped_bits,
            mask_bitstring, mask_ordering, mask_len)
        if param_form is None:
            params_in = None
        elif param_form == "class":
            params_in = cusv.SVSwapParameters()
        elif param_form == "ndarray":
            params_in = np.empty((1,), dtype=cusv.sv_swap_parameters_dtype)
        elif param_form == "int":
            params = np.empty((1,), dtype=cusv.sv_swap_parameters_dtype)
            params_in = params.ctypes.data
        else:
            assert False

        params_out = cusv.dist_index_bit_swap_scheduler_get_parameters(
            handle, scheduler, 0, 0, params=params_in)
        cusv.dist_index_bit_swap_scheduler_destroy(handle, scheduler)
        if param_form != "int":
            assert isinstance(params_out, cusv.SVSwapParameters)
        else:
            assert params_out is None

        # params_in should be modified in-place
        if param_form == "class":
            assert id(params_out) == id(params_in)
            assert params_out.data == params_in.data
            assert params_out.ptr == params_in.ptr
        elif param_form == "ndarray":
            assert params_out.data == params_in
            assert params_out.ptr == params_in.ctypes.data
        elif param_form == "int":
            # nothing to compare against...
            pass


class TestSubSVMigrator:
    ''' This class runs random tests to check all API arguemnts
        are correctly passed to C-API
    '''
    @classmethod
    def setup_class(cls):
        np.random.seed(20231003)

    @pytest.mark.parametrize(
        'dtype', (np.complex64, np.complex128)
    )
    @pytest.mark.parametrize(
        'exec_num', range(5)
    )
    def test_sub_sv_migrator(self, handle, dtype, exec_num):
        n_local_index_bits = np.random.randint(low=1, high=22)
        n_device_slots = np.random.randint(low=2, high=16)
        device_slot_idx = np.random.randint(n_device_slots)
        check_in_out = np.random.randint(4)
        randnum_dtype = np.float32 if dtype == np.complex64 else np.float64

        data_type = dtype_to_data_type[dtype]
        sub_sv_size = 2 ** n_local_index_bits
        device_slot_size = sub_sv_size * n_device_slots

        randnums = np.random.rand(device_slot_size) + 1.j * np.random.rand(device_slot_size)
        host_slots_ref = np.asarray(randnums, dtype=dtype)
        device_slots = cp.array(host_slots_ref)
        begin = np.random.randint(low=0, high=sub_sv_size-1)
        end = np.random.randint(low=begin + 1, high=sub_sv_size)

        src_sub_sv_ptr = 0
        dst_sub_sv_ptr = 0
        if check_in_out == 0:
            # swap
            check_in = check_out = True
            randnums = np.random.rand(sub_sv_size) + 1.j * np.random.rand(sub_sv_size)
            src_sub_sv_ref = np.asarray(randnums, dtype=dtype)
            src_sub_sv = cpx.empty_pinned(sub_sv_size, dtype=dtype)
            src_sub_sv[:] = src_sub_sv_ref[:]
            # src and dst are the same memory chunk
            dst_sub_sv = src_sub_sv
            dst_sub_sv_ref = src_sub_sv_ref
            src_sub_sv_ptr = dst_sub_sv_ptr = src_sub_sv.ctypes.data
        else:
            # check-in / check-out
            check_in = (check_in_out & 1) != 0
            check_out = (check_in_out & 2) != 0
            if check_out:
                randnums = np.random.rand(sub_sv_size) + 1.j * np.random.rand(sub_sv_size)
                src_sub_sv_ref = np.asarray(randnums, dtype=dtype)
                src_sub_sv = cpx.empty_pinned(sub_sv_size, dtype=dtype)
                src_sub_sv[:] = src_sub_sv_ref[:]
                src_sub_sv_ptr = src_sub_sv.ctypes.data
            if check_in:
                randnums = np.random.rand(sub_sv_size) + 1.j * np.random.rand(sub_sv_size)
                dst_sub_sv_ref = np.asarray(randnums, dtype=dtype)
                dst_sub_sv = cpx.empty_pinned(sub_sv_size, dtype=dtype)
                dst_sub_sv[:] = dst_sub_sv_ref[:]
                dst_sub_sv_ptr = dst_sub_sv.ctypes.data

        # create SubStateVectorMigrator
        migrator = cusv.sub_sv_migrator_create(
            handle, device_slots.data.ptr, data_type, n_device_slots, n_local_index_bits)
        # migrate
        cusv.sub_sv_migrator_migrate(
            handle, migrator, device_slot_idx, src_sub_sv_ptr, dst_sub_sv_ptr, begin, end)
        # destroy
        cp.cuda.Stream().synchronize()
        cusv.sub_sv_migrator_destroy(handle, migrator)

        # reference
        offset = sub_sv_size * device_slot_idx
        if check_in:
            # copy values for swap
            tmp = host_slots_ref[offset+begin:offset+end].copy()
        if check_out:
            host_slots_ref[offset+begin:offset+end] = src_sub_sv_ref[begin:end]
        if check_in:
            dst_sub_sv_ref[begin:end] = tmp[:]

        assert cp.all(cp.asarray(host_slots_ref) == device_slots)
        if check_out:
            assert np.all(src_sub_sv == src_sub_sv_ref)
        if check_in:
            assert np.all(dst_sub_sv == dst_sub_sv_ref)


class TestMemHandler(MemHandlerTestBase):

    mod = cusv
    prefix = "custatevec"
    error = cusv.cuStateVecError

    # TODO: add more different memory sources
    @pytest.mark.parametrize(
        'source', (None, "py-callable", 'cffi', 'cffi_struct')
    )
    def test_set_get_device_mem_handler(self, source, handle):
        self._test_set_get_device_mem_handler(source, handle)


class TestLogger(LoggerTestBase):

    mod = cusv
    prefix = "custatevec"
