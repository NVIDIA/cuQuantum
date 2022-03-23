import copy
import os
import tempfile

try:
    import cffi
except ImportError:
    cffi = None
import cupy
from cupy import testing
import numpy
import pytest

import cuquantum
from cuquantum import ComputeType, cudaDataType
from cuquantum import custatevec as cusv


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
    h = cusv.create()
    yield h
    cusv.destroy(h)


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

    # TODO: make this a static method
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


@pytest.fixture()
def multi_gpu_handles():
    # TODO: consider making this class more flexible
    # (ex: arbitrary number of qubits and/or devices, etc)
    n_devices = 2  # should be power of 2
    handles = []

    for dev in range(n_devices):
        with cupy.cuda.Device(dev):
            h = cusv.create()
            handles.append(h)
    yield handles
    for dev in range(n_devices):
        with cupy.cuda.Device(dev):
            h = handles.pop(0)
            cusv.destroy(h)


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
    'dtype': (numpy.complex64, numpy.complex128),
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
            with cupy.cuda.Device(dev):
                self.sub_sv.append(cupy.zeros(
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
        ver = cusv.get_version()
        assert ver == (cusv.MAJOR_VER * 1000
            + cusv.MINOR_VER * 100
            + cusv.PATCH_VER)
        assert ver == cusv.VERSION

    def test_get_property(self):
        assert cusv.MAJOR_VER == cusv.get_property(
            cuquantum.libraryPropertyType.MAJOR_VERSION)
        assert cusv.MINOR_VER == cusv.get_property(
            cuquantum.libraryPropertyType.MINOR_VERSION)
        assert cusv.PATCH_VER == cusv.get_property(
            cuquantum.libraryPropertyType.PATCH_LEVEL)


# we don't wanna recompile for every test case...
_cffi_mod1 = None
_cffi_mod2 = None

def _can_use_cffi():
    if cffi is None or os.environ.get('CUDA_PATH') is None:
        return False
    else:
        return True


class MemoryResourceFactory:

    def __init__(self, source, name=None):
        self.source = source
        self.name = source if name is None else name

    def get_dev_mem_handler(self):
        if self.source == "py-callable":
            return (*self._get_cuda_callable(), self.name)
        elif self.source == "cffi":
            # ctx is not needed, so set to NULL
            return (0, *self._get_functor_address(), self.name)
        elif self.source == "cffi_struct":
            return self._get_handler_address()
        # TODO: add more different memory sources
        else:
            raise NotImplementedError

    def _get_cuda_callable(self):
        def alloc(size, stream):
            return cupy.cuda.runtime.mallocAsync(size, stream)

        def free(ptr, size, stream):
            cupy.cuda.runtime.freeAsync(ptr, stream)

        return alloc, free

    def _get_functor_address(self):
        if not _can_use_cffi():
            raise RuntimeError

        global _cffi_mod1
        if _cffi_mod1 is None:
            import importlib
            mod_name = f"cusv_test_{self.source}"
            ffi = cffi.FFI()
            ffi.set_source(mod_name, """
                #include <cuda_runtime.h>

                // cffi limitation: we can't use the actual type cudaStream_t because
                // it's considered an "incomplete" type and we can't get the functor
                // address by doing so...

                int my_alloc(void* ctx, void** ptr, size_t size, void* stream) {
                    return (int)cudaMallocAsync(ptr, size, stream);
                }

                int my_free(void* ctx, void* ptr, size_t size, void* stream) {
                    return (int)cudaFreeAsync(ptr, stream);
                }
                """,
                include_dirs=[os.environ['CUDA_PATH']+'/include'],
                library_dirs=[os.environ['CUDA_PATH']+'/lib64'],
                libraries=['cudart'],
            )
            ffi.cdef("""
                int my_alloc(void* ctx, void** ptr, size_t size, void* stream);
                int my_free(void* ctx, void* ptr, size_t size, void* stream);
            """)
            ffi.compile(verbose=True)
            self.ffi = ffi
            _cffi_mod1 = importlib.import_module(mod_name)
        self.ffi_mod = _cffi_mod1

        alloc_addr = self._get_address("my_alloc")
        free_addr = self._get_address("my_free")
        return alloc_addr, free_addr

    def _get_handler_address(self):
        if not _can_use_cffi():
            raise RuntimeError

        global _cffi_mod2
        if _cffi_mod2 is None:
            import importlib
            mod_name = f"cusv_test_{self.source}"
            ffi = cffi.FFI()
            ffi.set_source(mod_name, """
                #include <cuda_runtime.h>

                // cffi limitation: we can't use the actual type cudaStream_t because
                // it's considered an "incomplete" type and we can't get the functor
                // address by doing so...

                int my_alloc(void* ctx, void** ptr, size_t size, void* stream) {
                    return (int)cudaMallocAsync(ptr, size, stream);
                }

                int my_free(void* ctx, void* ptr, size_t size, void* stream) {
                    return (int)cudaFreeAsync(ptr, stream);
                }

                typedef struct {
                    void* ctx;
                    int (*device_alloc)(void* ctx, void** ptr, size_t size, void* stream);
                    int (*device_free)(void* ctx, void* ptr, size_t size, void* stream);
                    char name[64];
                } myHandler;

                myHandler* init_myHandler(myHandler* h, const char* name) {
                    h->ctx = NULL;
                    h->device_alloc = my_alloc;
                    h->device_free = my_free;
                    memcpy(h->name, name, 64);
                    return h;
                }
                """,
                include_dirs=[os.environ['CUDA_PATH']+'/include'],
                library_dirs=[os.environ['CUDA_PATH']+'/lib64'],
                libraries=['cudart'],
            )
            ffi.cdef("""
                typedef struct {
                    ...;
                } myHandler;

                myHandler* init_myHandler(myHandler* h, const char* name);
            """)
            ffi.compile(verbose=True)
            self.ffi = ffi
            _cffi_mod2 = importlib.import_module(mod_name)
        self.ffi_mod = _cffi_mod2

        h = self.handler = self.ffi_mod.ffi.new("myHandler*")
        self.ffi_mod.lib.init_myHandler(h, self.name.encode())
        return self._get_address(h)

    def _get_address(self, func_name_or_ptr):
        if isinstance(func_name_or_ptr, str):
            func_name = func_name_or_ptr
            data = str(self.ffi_mod.ffi.addressof(self.ffi_mod.lib, func_name))
        else:
            ptr = func_name_or_ptr  # ptr to struct
            data = str(self.ffi_mod.ffi.addressof(ptr[0]))
        # data has this format: "<cdata 'int(*)(void *, void * *, size_t, void *)' 0x7f6c5da37300>"
        return int(data.split()[-1][:-1], base=16)


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
        memptr = cupy.cuda.alloc(size)
        cusv.set_workspace(handle, memptr.ptr, size)  # should not fail

    def test_stream(self, handle):
        # default is on the null stream
        assert 0 == cusv.get_stream(handle)

        # simple set/get round-trip
        stream = cupy.cuda.Stream()
        cusv.set_stream(handle, stream.ptr)
        assert stream.ptr == cusv.get_stream(handle)


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
        sum0, sum1 = cusv.abs2sum_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            True, True, basis_bits, basis_bits_len)
        assert numpy.allclose(sum0+sum1, 1)
        assert (sum0 is not None) and (sum1 is not None)

        # case 2: only sum0 is computed
        sum0, sum1 = cusv.abs2sum_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            True, False, basis_bits, basis_bits_len)
        assert numpy.allclose(sum0, 1)
        assert (sum0 is not None) and (sum1 is None)

        # case 3: only sum1 is computed
        sum0, sum1 = cusv.abs2sum_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            False, True, basis_bits, basis_bits_len)
        assert numpy.allclose(sum1, 0)
        assert (sum0 is None) and (sum1 is not None)

        # case 4: none is computed
        with pytest.raises(ValueError):
            sum0, sum1 = cusv.abs2sum_on_z_basis(
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
        abs2sum = xp.zeros((2**bit_ordering_len,), dtype=xp.float64)
        abs2sum_ptr = abs2sum.data.ptr if xp is cupy else abs2sum.ctypes.data
        cusv.abs2sum_array(
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

        cusv.collapse_on_z_basis(
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
        cusv.collapse_by_bitstring(
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
    (cusv.Collapse.NORMALIZE_AND_ZERO, cusv.Collapse.NONE)
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

        parity = cusv.measure_on_z_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            basis_bits, basis_bits_len, rand, collapse)

        if collapse == cusv.Collapse.NORMALIZE_AND_ZERO:
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

        cusv.batch_measure(
            handle, sv.data.ptr, data_type, self.n_qubits,
            bitstring.ctypes.data, bit_ordering, bitstring.size,
            rand, collapse)

        if collapse == cusv.Collapse.NORMALIZE_AND_ZERO:
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
    def test_apply_pauli_rotation(self, handle, input_form):
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
        paulis = [cusv.Pauli.X, cusv.Pauli.X]
        paulis, _ = self._return_data(
            paulis, 'paulis', *input_form['paulis'])

        cusv.apply_pauli_rotation(
            handle, sv.data.ptr, data_type, self.n_qubits,
            0.5*numpy.pi, paulis,
            targets, targets_len,
            controls, control_values, controls_len)
        sv *= -1j

        # result is |111>
        assert cupy.allclose(sv[-1], 1)

    @pytest.mark.parametrize(
        'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'targets': (numpy.int32, 'int'), 'controls': (numpy.int32, 'int')},
            {'targets': (numpy.int32, 'seq'), 'controls': (numpy.int32, 'seq')},
        )
    )
    @pytest.mark.parametrize(
        'xp', (numpy, cupy)
     )
    def test_apply_matrix(self, handle, xp, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

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

        if mempool is None:
            workspace_size = cusv.apply_matrix_get_workspace_size(
                handle, data_type, self.n_qubits,
                matrix_ptr, data_type, cusv.MatrixLayout.ROW, 0,
                targets_len, controls_len, compute_type)
            if workspace_size:
                workspace = cupy.cuda.alloc(workspace_size)
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
            {'permutation': (numpy.int64, 'int'), 'basis_bits': (numpy.int32, 'int'),
             'mask_bitstring': (numpy.int32, 'int'), 'mask_ordering': (numpy.int32, 'int')},
            {'permutation': (numpy.int64, 'seq'), 'basis_bits': (numpy.int32, 'seq'),
             'mask_bitstring': (numpy.int32, 'seq'), 'mask_ordering': (numpy.int32, 'seq')},
        )
    )
    @pytest.mark.parametrize(
        'xp', (numpy, cupy)
     )
    def test_apply_generalized_permutation_matrix(
            self, handle, xp, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

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

        if mempool is None:
            workspace_size = cusv.apply_generalized_permutation_matrix_get_workspace_size(
                handle, data_type, self.n_qubits,
                permutation, diagonal_ptr, data_type,
                basis_bits, basis_bits_len, mask_len)

            if workspace_size:
                workspace = cupy.cuda.alloc(workspace_size)
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

        assert cupy.allclose(sv, diagonal[xp.asarray(permutation_data)])


class TestExpect(TestSV):

    @pytest.mark.parametrize(
        'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
    )
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
    def test_compute_expectation(self, handle, xp, expect_dtype, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

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

        if mempool is None:
            workspace_size = cusv.compute_expectation_get_workspace_size(
                handle, data_type, self.n_qubits,
                matrix_ptr, data_type, cusv.MatrixLayout.ROW,
                basis_bits_len, compute_type)
            if workspace_size:
                workspace = cupy.cuda.alloc(workspace_size)
                workspace_ptr = workspace.ptr
            else:
                workspace_ptr = 0
        else:
            mr = MemoryResourceFactory(mempool)
            handler = mr.get_dev_mem_handler()
            cusv.set_device_mem_handler(handle, handler)

            workspace_ptr = 0
            workspace_size = 0

        expect = numpy.empty((1,), dtype=expect_dtype)
        # TODO(leofang): check if this is relaxed in beta 2
        expect_data_type = (
            cudaDataType.CUDA_R_64F if expect_dtype == numpy.float64
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
        sv[:] = numpy.sqrt(1/(2**self.n_qubits))
        data_type = dtype_to_data_type[sv.dtype]
        compute_type = dtype_to_compute_type[sv.dtype]

        # measure XX...X, YY..Y, ZZ...Z
        paulis = [[cusv.Pauli.X for i in range(self.n_qubits)],
                  [cusv.Pauli.Y for i in range(self.n_qubits)],
                  [cusv.Pauli.Z for i in range(self.n_qubits)],]

        basis_bits = [[*range(self.n_qubits)] for i in range(len(paulis))]
        n_basis_bits = [len(basis_bits[i]) for i in range(len(paulis))]
        expect = numpy.empty((len(paulis),), dtype=numpy.float64)

        cusv.compute_expectations_on_pauli_basis(
            handle, sv.data.ptr, data_type, self.n_qubits,
            expect.ctypes.data, paulis, len(paulis),
            basis_bits, n_basis_bits)

        result = numpy.zeros_like(expect)
        result[0] = 1  # for XX...X
        assert numpy.allclose(expect, result)


class TestSampler(TestSV):

    @pytest.mark.parametrize(
        'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
    )
    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (numpy.int32, 'int'),},
            {'bit_ordering': (numpy.int32, 'seq'),},
        )
    )
    def test_sampling(self, handle, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

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

        sampler, workspace_size = cusv.sampler_create(
            handle, sv.data.ptr, data_type, self.n_qubits, shots)
        if mempool is None:
            if workspace_size:
                workspace = cupy.cuda.alloc(workspace_size)
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

        keys, counts = numpy.unique(bitstrings, return_counts=True)
        # keys are the returned bitstrings 000, 001, ..., 111
        # the sv has all components, and unique() returns a sorted array,
        # so the following should hold:
        assert (keys == numpy.arange(2**self.n_qubits)).all()

        assert numpy.allclose(norm, 1)

        # TODO: test counts, which should follow a uniform distribution


@pytest.mark.parametrize(
    'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
)
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

    def test_accessor_get(self, handle, readonly, input_form, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

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
                    workspace = cupy.cuda.alloc(workspace_size)
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
            buf = cupy.empty(buf_len, dtype=sv.dtype)

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
                    workspace = cupy.cuda.alloc(workspace_size)
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
            buf = cupy.zeros(buf_len, dtype=sv.dtype)

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
            {'targets': (numpy.int32, 'int'), },
            {'targets': (numpy.int32, 'seq'), },
        )
    )
    @pytest.mark.parametrize(
        'dtype', (numpy.complex64, numpy.complex128)
    )
    @pytest.mark.parametrize(
        'xp', (numpy, cupy)
     )
    def test_apply_matrix_type(
            self, handle, xp, dtype, input_form, matrix_type, mempool):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        data_type = dtype_to_data_type[xp.dtype(dtype)]
        compute_type = dtype_to_compute_type[xp.dtype(dtype)]
        n_targets = 4

        # matrix can live on host or device
        # choose a trivial matrix
        data = xp.ones(2**n_targets, dtype=dtype)
        matrix = xp.diag(data)
        matrix_ptr = matrix.ctypes.data if xp is numpy else matrix.data.ptr

        if mempool is None:
            workspace_size = cusv.test_matrix_type_get_workspace_size(
                handle, matrix_type,
                matrix_ptr, data_type, cusv.MatrixLayout.ROW, n_targets,
                0, compute_type)
            if workspace_size:
                workspace = cupy.cuda.alloc(workspace_size)
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
        assert numpy.isclose(residual, 0)


@pytest.mark.parametrize(
    'rand',
    # the choices here ensure we get either parity
    (0, numpy.nextafter(1, 0))
)
@pytest.mark.parametrize(
    'collapse',
    (cusv.Collapse.NORMALIZE_AND_ZERO, cusv.Collapse.NONE)
)
@pytest.mark.skipif(
    cupy.cuda.runtime.getDeviceCount() < 2, reason='not enough GPUs')
class TestBatchMeasureWithSubSV(TestMultiGpuSV):

    @pytest.mark.parametrize(
        'input_form', (
            {'bit_ordering': (numpy.int32, 'int'),},
            {'bit_ordering': (numpy.int32, 'seq'),},
        )
    )
    def test_batch_measure_with_offset(
            self, multi_gpu_handles, rand, collapse, input_form):
        handles = multi_gpu_handles
        sub_sv = self.get_sv()
        data_type = dtype_to_data_type[sub_sv[0].dtype]
        bit_ordering = list(range(self.n_local_bits))
        bit_ordering, bit_ordering_len = self._return_data(
            bit_ordering, 'bit_ordering', *input_form['bit_ordering'])

        # change sv to 1/\sqrt{2} (|0000> + |1111>), and compute abs2sum;
        # calling abs2sum_array is also OK, but we focus on testing the target API
        cumulative_array = numpy.zeros(self.n_devices+1, dtype=numpy.float64)
        for i_sv in range(self.n_devices):
            with cupy.cuda.Device(i_sv):
                if i_sv == 0:
                    # |0 000> is on GPU 0
                    sub_sv[i_sv][0] = numpy.sqrt(0.5)
                elif i_sv == 1:
                    # |1 111> is on GPU 1
                    sub_sv[i_sv][-1] = numpy.sqrt(0.5)
                abs2sum = cupy.asnumpy(cupy.sum(cupy.abs(sub_sv[i_sv])**2))
                cumulative_array[i_sv+1] = cumulative_array[i_sv] + abs2sum

        orig_sub_sv = copy.deepcopy(sub_sv)

        bitstring = numpy.empty(self.n_local_bits, dtype=numpy.int32)
        for i_sv in range(self.n_devices):
            if (cumulative_array[i_sv] <= rand
                    and rand < cumulative_array[i_sv+1]):
                global_bits = i_sv
                norm = cumulative_array[-1]
                offset = cumulative_array[i_sv]
                with cupy.cuda.Device(i_sv) as dev:
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
                with cupy.cuda.Device(0):
                    assert cupy.allclose(sub_sv[0][0], 1)
                    assert not (sub_sv[0] == orig_sub_sv[0]).all()
                with cupy.cuda.Device(1):
                    assert (sub_sv[1] == orig_sub_sv[1]).all()
            elif global_bits == 1:
                # collapse to |1 111>
                with cupy.cuda.Device(0):
                    assert (sub_sv[0] == orig_sub_sv[0]).all()
                with cupy.cuda.Device(1):
                    assert cupy.allclose(sub_sv[1][-1], 1)
                    assert not (sub_sv[1] == orig_sub_sv[1]).all()
            else:
                assert False, f"unexpected bitstring: {bitstring}"
        else:
            # sv is intact
            with cupy.cuda.Device(0):
                assert (sub_sv[0] == orig_sub_sv[0]).all()
            with cupy.cuda.Device(1):
                assert (sub_sv[1] == orig_sub_sv[1]).all()


class TestSwap:

    @pytest.mark.parametrize(
        'input_form', (
            {'swapped_bits': (numpy.int32, 'int'),
             'mask_bitstring': (numpy.int32, 'int'), 'mask_ordering': (numpy.int32, 'int')},
            {'swapped_bits': (numpy.int32, 'seq'),
             'mask_bitstring': (numpy.int32, 'seq'), 'mask_ordering': (numpy.int32, 'seq')},
        )
    )
    @pytest.mark.parametrize(
        'dtype', (numpy.complex64, numpy.complex128)
    )
    def test_swap_index_bits(self, handle, dtype, input_form):
        n_qubits = 4
        sv = cupy.zeros(2**n_qubits, dtype=dtype)
        data_type = dtype_to_data_type[sv.dtype]

        # set sv to |0110>
        sv[6] = 1
        orig_sv = sv.copy()

        swapped_bits = [(0, 2), (1, 3)]
        n_swapped_bits = len(swapped_bits)
        if input_form['swapped_bits'][1] == 'int':
            swapped_bits_data = numpy.asarray(
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


class TestMemHandler:

    # TODO: add more different memory sources
    @pytest.mark.parametrize(
        'source', (None, "py-callable", 'cffi', 'cffi_struct')
    )
    def test_set_get_device_mem_handler(self, handle, source):
        if (isinstance(source, str) and source.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        if source is not None:
            mr = MemoryResourceFactory(source)
            handler = mr.get_dev_mem_handler()
            cusv.set_device_mem_handler(handle, handler)
            # round-trip test
            queried_handler = cusv.get_device_mem_handler(handle)
            if source == 'cffi_struct':
                # I'm lazy, otherwise I'd also fetch the functor addresses here...
                assert queried_handler[0] == 0  # ctx is NULL
                assert queried_handler[-1] == source
            else:
                assert queried_handler == handler
        else:
            with pytest.raises(cusv.cuStateVecError) as e:
                queried_handler = cusv.get_device_mem_handler(handle)
            assert 'CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR' in str(e.value)


class TestLogger:

    def test_logger_set_level(self):
        cusv.logger_set_level(6)  # on
        cusv.logger_set_level(0)  # off

    def test_logger_set_mask(self):
        cusv.logger_set_mask(16)  # should not raise

    def test_logger_set_callback_data(self):
        # we also test logger_open_file() here to avoid polluting stdout

        def callback(level, name, message, my_data, is_ok=False):
            log = f"{level}, {name}, {message} (is_ok={is_ok}) -> logged\n"
            my_data.append(log)

        handle = None
        my_data = []
        is_ok = True

        with tempfile.TemporaryDirectory() as temp:
            file_name = os.path.join(temp, "cusv_test")
            cusv.logger_open_file(file_name)
            cusv.logger_set_callback_data(callback, my_data, is_ok=is_ok)
            cusv.logger_set_level(6)

            try:
                handle = cusv.create()
                cusv.destroy(handle)
            except:
                if handle:
                    cusv.destroy(handle)
                raise
            finally:
                cusv.logger_force_disable()  # to not affect the rest of tests

            with open(file_name) as f:
                log_from_f = f.read()

        # check the log file
        assert '[custatevecCreate]' in log_from_f
        assert '[custatevecDestroy]' in log_from_f

        # check the captured data (note we log 2 APIs)
        log = ''.join(my_data)
        assert log.count("-> logged") >= 2
        assert log.count("is_ok=True") >= 2
