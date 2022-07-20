# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import abc
import functools
import os
import sys
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
from cuquantum import cutensornet as cutn


###################################################################
#
# As of beta 2, the test suite for Python bindings is kept minimal.
# The sole goal is to ensure the Python arguments are properly
# passed to the C level. We do not ensure coverage nor correctness.
# This decision will be revisited in the future.
#
###################################################################

if cffi:
    # if the Python binding is not installed in the editable mode (pip install
    # -e .), the cffi tests would fail as the modules cannot be imported
    sys.path.append(os.getcwd())

dtype_to_data_type = {
    numpy.float16: cudaDataType.CUDA_R_16F,
    numpy.float32: cudaDataType.CUDA_R_32F,
    numpy.float64: cudaDataType.CUDA_R_64F,
    numpy.complex64: cudaDataType.CUDA_C_32F,
    numpy.complex128: cudaDataType.CUDA_C_64F,
}


dtype_to_compute_type = {
    numpy.float16: ComputeType.COMPUTE_16F,
    numpy.float32: ComputeType.COMPUTE_32F,
    numpy.float64: ComputeType.COMPUTE_64F,
    numpy.complex64: ComputeType.COMPUTE_32F,
    numpy.complex128: ComputeType.COMPUTE_64F,
}


def manage_resource(name):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kwargs):
            try:
                if name == 'handle':
                    h = cutn.create()
                elif name == 'dscr':
                    tn, dtype, input_form, output_form = self.tn, self.dtype, self.input_form, self.output_form
                    einsum, shapes = tn  # unpack
                    tn = TensorNetworkFactory(einsum, shapes, dtype)
                    i_n_inputs, i_n_modes, i_extents, i_strides, i_modes, i_alignments = \
                        tn.get_input_metadata(**input_form)
                    o_n_modes, o_extents, o_strides, o_modes, o_alignments = \
                        tn.get_output_metadata(**output_form)
                    h = cutn.create_network_descriptor(
                        self.handle,
                        i_n_inputs, i_n_modes, i_extents, i_strides, i_modes, i_alignments,
                        o_n_modes, o_extents, o_strides, o_modes, o_alignments,
                        dtype_to_data_type[dtype], dtype_to_compute_type[dtype])
                    # we also need to keep the tn data alive
                    self.tn = tn
                elif name == 'config':
                    h = cutn.create_contraction_optimizer_config(self.handle)
                elif name == 'info':
                    h = cutn.create_contraction_optimizer_info(
                        self.handle, self.dscr)
                elif name == 'autotune':
                    h = cutn.create_contraction_autotune_preference(self.handle)
                elif name == 'workspace':
                    h = cutn.create_workspace_descriptor(self.handle)
                elif name == 'slice_group':
                    # we use this version to avoid creating a sequence; another API
                    # is tested elsewhere
                    h = cutn.create_slice_group_from_id_range(self.handle, 0, 1, 1)
                else:
                    assert False, f'name "{name}" not recognized'
                setattr(self, name, h)
                impl(self, *args, **kwargs)
            except:
                print(f'managing resource {name} failed')
                raise
            finally:
                if name == 'handle' and hasattr(self, name):
                    cutn.destroy(self.handle)
                    del self.handle
                elif name == 'dscr' and hasattr(self, name):
                    cutn.destroy_network_descriptor(self.dscr)
                    del self.dscr
                elif name == 'config' and hasattr(self, name):
                    cutn.destroy_contraction_optimizer_config(self.config)
                    del self.config
                elif name == 'info' and hasattr(self, name):
                    cutn.destroy_contraction_optimizer_info(self.info)
                    del self.info
                elif name == 'autotune' and hasattr(self, name):
                    cutn.destroy_contraction_autotune_preference(self.autotune)
                    del self.autotune
                elif name == 'workspace' and hasattr(self, name):
                    h = cutn.destroy_workspace_descriptor(self.workspace)
                    del self.workspace
                elif name == 'slice_group':
                    h = cutn.destroy_slice_group(self.slice_group)
                    del self.slice_group
        return test_func
    return decorator


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
            mod_name = f"cutn_test_{self.source}"
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
            mod_name = f"cutn_test_{self.source}"
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


class TestLibHelper:

    def test_get_version(self):
        ver = cutn.get_version()
        assert ver == (cutn.MAJOR_VER * 10000
            + cutn.MINOR_VER * 100
            + cutn.PATCH_VER)
        assert ver == cutn.VERSION

    def test_get_cudart_version(self):
        # CUDA runtime is statically linked, so we can't compare
        # with the "runtime" version
        ver = cutn.get_cudart_version()
        assert isinstance(ver, int)


class TestHandle:

    @manage_resource('handle')
    def test_handle_create_destroy(self):
        # simple rount-trip test
        pass


class TensorNetworkFactory:

    # TODO(leofang): replace the utilities here by high-level private APIs

    # This factory CANNOT be reused; once a TN descriptor uses it, it must
    # be discarded.

    def __init__(self, einsum, shapes, dtype):
        self.einsum = einsum
        inputs, output = einsum.split('->') if "->" in einsum else (einsum, None)
        i_shapes, o_shape = shapes[:-1], shapes[-1]
        inputs = tuple(tuple(_input) for _input in inputs.split(","))
        assert all([len(i) == len(s) for i, s in zip(inputs, i_shapes)])
        assert len(output) == len(o_shape)

        self.input_tensors = [
            testing.shaped_random(s, cupy, dtype) for s in i_shapes]
        self.input_n_modes = [len(i) for i in inputs]
        self.input_extents = i_shapes
        self.input_strides = [arr.strides for arr in self.input_tensors]
        self.input_modes = [tuple([ord(m) for m in i]) for i in inputs]
        self.input_alignments = [256] * len(i_shapes)

        self.output_tensor = cupy.empty(o_shape, dtype=dtype)
        self.output_n_modes = len(o_shape)
        self.output_extent = o_shape
        self.output_stride = self.output_tensor.strides
        self.output_mode = tuple([ord(m) for m in output])
        self.output_alignment = 256

    def _get_data_type(self, category):
        if 'n_modes' in category:
            return numpy.int32
        elif 'extent' in category:
            return numpy.int64
        elif 'stride' in category:
            return numpy.int64
        elif 'mode' in category:
            return numpy.int32
        elif 'alignment' in category:
            return numpy.uint32
        elif 'tensor' in category:
            return None  # unused
        else:
            assert False

    def _return_data(self, category, return_value):
        data = getattr(self, category)

        if return_value == 'int':
            if len(data) == 0:
                # empty, give it a NULL
                return 0
            elif category == 'input_tensors':
                # special case for device arrays, return int as void**
                data = numpy.asarray([d.data.ptr for d in data],
                    dtype=numpy.intp)
                setattr(self, f'{category}_ptrs', data)  # keep data alive
            # some data are not nested in nature, so we peek at the first
            # element to determine
            elif isinstance(data[0], abc.Sequence):
                # return int as void**
                data = [numpy.asarray(d, dtype=self._get_data_type(category))
                    for d in data]
                setattr(self, category, data)  # keep data alive
                data = numpy.asarray([d.ctypes.data for d in data],
                    dtype=numpy.intp)
                setattr(self, f'{category}_ptrs', data)  # keep data alive
            else:
                # return int as void*
                data = numpy.asarray(data, dtype=self._get_data_type(category))
                setattr(self, category, data)  # keep data alive
            return data.ctypes.data
        elif return_value == 'seq':
            if len(data) == 0:
                # empty, leave it as is
                pass
            elif category == 'input_tensors':
                # special case for device arrays
                data = [d.data.ptr for d in data]
                setattr(self, f'{category}_ptrs', data)  # keep data alive
            # some data are not nested in nature, so we peek at the first
            # element to determine
            elif isinstance(data[0], abc.Sequence):
                data = [numpy.asarray(d, dtype=self._get_data_type(category))
                    for d in data]
                setattr(self, category, data)  # keep data alive
            else:
                # data itself is already a flat sequence
                pass
            return data
        elif return_value == 'nested_seq':
            return data
        else:
            assert False

    def get_input_metadata(self, **kwargs):
        n_inputs = len(self.input_tensors)
        n_modes = self._return_data('input_n_modes', kwargs.pop('n_modes'))
        extents = self._return_data('input_extents', kwargs.pop('extent'))
        strides = self._return_data('input_strides', kwargs.pop('stride'))
        modes = self._return_data('input_modes', kwargs.pop('mode'))
        alignments = self._return_data(
            'input_alignments', kwargs.pop('alignment'))
        return n_inputs, n_modes, extents, strides, modes, alignments

    def get_output_metadata(self, **kwargs):
        n_modes = self.output_n_modes
        extent = self._return_data('output_extent', kwargs.pop('extent'))
        stride = self._return_data('output_stride', kwargs.pop('stride'))
        mode = self._return_data('output_mode', kwargs.pop('mode'))
        alignment = self.output_alignment
        return n_modes, extent, stride, mode, alignment

    def get_input_tensors(self, **kwargs):
        data = self._return_data('input_tensors', kwargs['data'])
        return data

    def get_output_tensor(self):
        return self.output_tensor.data.ptr


@testing.parameterize(*testing.product({
    'tn': (
        ('ab,bc->ac', [(2, 3), (3, 2), (2, 2)]),
        ('ab,ba->', [(2, 3), (3, 2), ()]),
        ('abc,bca->', [(2, 3, 4), (3, 4, 2), ()]),
        ('ab,bc,cd->ad', [(2, 3), (3, 1), (1, 5), (2, 5)]),
    ),
    'dtype': (
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128
    ),
    'input_form': (
        {'n_modes': 'int', 'extent': 'int', 'stride': 'int',
         'mode': 'int', 'alignment': 'int', 'data': 'int'},
        {'n_modes': 'int', 'extent': 'seq', 'stride': 'seq',
         'mode': 'seq', 'alignment': 'int', 'data': 'seq'},
        {'n_modes': 'seq', 'extent': 'nested_seq', 'stride': 'nested_seq',
         'mode': 'seq', 'alignment': 'seq', 'data': 'seq'},
    ),
    'output_form': (
        {'extent': 'int', 'stride': 'int', 'mode': 'int'},
        {'extent': 'seq', 'stride': 'seq', 'mode': 'seq'},
    )
}))
class TestTensorNetworkBase:

    # Use this class as the base to share all common test parametrizations
    pass


class TestTensorNetworkDescriptor(TestTensorNetworkBase):

    @manage_resource('handle')
    @manage_resource('dscr')
    def test_descriptor_create_destroy(self):
        # we could just do a simple round-trip test, but let's also get
        # this helper API tested
        handle, dscr = self.handle, self.dscr
        num_modes, modes, extents, strides = cutn.get_output_tensor_details(handle, dscr)
        assert num_modes == self.tn.output_n_modes
        assert (modes == numpy.asarray(self.tn.output_mode, dtype=numpy.int32)).all()
        assert (extents == numpy.asarray(self.tn.output_extent, dtype=numpy.int64)).all()
        assert (strides == numpy.asarray(self.tn.output_stride, dtype=numpy.int64)).all()


class TestOptimizerInfo(TestTensorNetworkBase):

    def _get_path(self, handle, info):
        raise NotImplementedError

    def _set_path(self, handle, info, path):
        attr = cutn.ContractionOptimizerInfoAttribute.PATH
        if not isinstance(path, numpy.ndarray):
            path = numpy.ascontiguousarray(path, dtype=numpy.int32)
        num_contraction = path.shape[0]
        p = cutn.ContractionPath(num_contraction, path.ctypes.data)
        cutn.contraction_optimizer_info_set_attribute(
            handle, info, attr, p.get_path(), p.get_size())

    def _get_scalar_attr(self, handle, info, attr):
        dtype = cutn.contraction_optimizer_info_get_attribute_dtype(attr)
        data = numpy.empty((1,), dtype=dtype)
        cutn.contraction_optimizer_info_get_attribute(
            handle, info, attr,
            data.ctypes.data, data.dtype.itemsize)
        return data

    def _set_scalar_attr(self, handle, info, attr, data):
        dtype = cutn.contraction_optimizer_info_get_attribute_dtype(attr)
        if not isinstance(data, numpy.ndarray):
            data = numpy.ascontiguousarray(data, dtype=dtype)
        cutn.contraction_optimizer_info_set_attribute(
            handle, info, attr,
            data.ctypes.data, data.dtype.itemsize)

    @manage_resource('handle')
    @manage_resource('dscr')
    @manage_resource('info')
    def test_optimizer_info_create_destroy(self):
        # simple round-trip test
        pass

    @pytest.mark.parametrize(
        'attr', [val for val in cutn.ContractionOptimizerInfoAttribute]
    )
    @manage_resource('handle')
    @manage_resource('dscr')
    @manage_resource('info')
    def test_optimizer_info_get_set_attribute(self, attr):
        if attr in (
                cutn.ContractionOptimizerInfoAttribute.NUM_SLICES,
                cutn.ContractionOptimizerInfoAttribute.PHASE1_FLOP_COUNT,
                cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT,
                cutn.ContractionOptimizerInfoAttribute.LARGEST_TENSOR,
                cutn.ContractionOptimizerInfoAttribute.SLICING_OVERHEAD,
                cutn.ContractionOptimizerInfoAttribute.EFFECTIVE_FLOPS_EST,
                cutn.ContractionOptimizerInfoAttribute.RUNTIME_EST,
                ):
            pytest.skip("setter not supported")
        elif attr in (
                cutn.ContractionOptimizerInfoAttribute.PATH,
                cutn.ContractionOptimizerInfoAttribute.SLICED_MODE,
                cutn.ContractionOptimizerInfoAttribute.SLICED_EXTENT,
                cutn.ContractionOptimizerInfoAttribute.INTERMEDIATE_MODES,
                cutn.ContractionOptimizerInfoAttribute.NUM_INTERMEDIATE_MODES,
                ):
            pytest.skip("TODO")
        handle, info = self.handle, self.info
        # Hack: assume this is a valid value for all attrs
        factor = 30
        self._set_scalar_attr(handle, info, attr, factor)
        # do a round-trip test as a sanity check
        factor2 = self._get_scalar_attr(handle, info, attr)
        assert factor == factor2

    @pytest.mark.parametrize(
        "buffer_form", ("int", "buf")
    )
    @manage_resource('handle')
    @manage_resource('dscr')
    @manage_resource('info')
    def test_optimizer_info_packing_unpacking(self, buffer_form):
        tn, handle, dscr, info = self.tn, self.handle, self.dscr, self.info
        attr = cutn.ContractionOptimizerInfoAttribute.PATH
        dtype = cutn.contraction_optimizer_info_get_attribute_dtype(attr)

        # compute a valid path for the problem
        path, _ = numpy.einsum_path(
            tn.einsum,
            *[arr for arr in map(lambda a: numpy.broadcast_to(0, a.shape),
                                 tn.input_tensors)])

        # set the path in info (a few other attributes would be computed too)
        # and then serialize it
        self._set_path(handle, info, path[1:])
        buf_size = cutn.contraction_optimizer_info_get_packed_size(
            handle, info)
        buf_data = numpy.empty((buf_size,), dtype=numpy.int8)
        if buffer_form == "int":
            buf = buf_data.ctypes.data
        else:  # buffer_form == "buf"
            buf = buf_data
        cutn.contraction_optimizer_info_pack_data(
            handle, info, buf, buf_size)

        # sanity check: all info must give the same attribute
        attr = cutn.ContractionOptimizerInfoAttribute.LARGEST_TENSOR
        largest = self._get_scalar_attr(handle, info, attr)

        info2 = cutn.create_contraction_optimizer_info_from_packed_data(
            handle, dscr, buf, buf_size)
        largest2 = self._get_scalar_attr(handle, info2, attr)

        info3 = cutn.create_contraction_optimizer_info(handle, dscr)
        cutn.update_contraction_optimizer_info_from_packed_data(
            handle, buf, buf_size, info3)
        largest3 = self._get_scalar_attr(handle, info3, attr)

        try:
            assert largest == largest2
            assert largest == largest3
        finally:
            cutn.destroy_contraction_optimizer_info(info2)
            cutn.destroy_contraction_optimizer_info(info3)


class TestOptimizerConfig:

    @manage_resource('handle')
    @manage_resource('config')
    def test_optimizer_config_create_destroy(self):
        # simple round-trip test
        pass

    @pytest.mark.parametrize(
        'attr', [val for val in cutn.ContractionOptimizerConfigAttribute]
    )
    @manage_resource('handle')
    @manage_resource('config')
    def test_optimizer_config_get_set_attribute(self, attr):
        handle, config = self.handle, self.config
        dtype = cutn.contraction_optimizer_config_get_attribute_dtype(attr)
        # Hack: assume this is a valid value for all attrs
        if attr in (cutn.ContractionOptimizerConfigAttribute.GRAPH_ALGORITHM,
                    cutn.ContractionOptimizerConfigAttribute.SLICER_MEMORY_MODEL,
                    cutn.ContractionOptimizerConfigAttribute.SLICER_DISABLE_SLICING,
                    cutn.ContractionOptimizerConfigAttribute.SIMPLIFICATION_DISABLE_DR,
                    cutn.ContractionOptimizerConfigAttribute.COST_FUNCTION_OBJECTIVE):
            factor = numpy.asarray([1], dtype=dtype)
        else:
            factor = numpy.asarray([30], dtype=dtype)
        cutn.contraction_optimizer_config_set_attribute(
            handle, config, attr,
            factor.ctypes.data, factor.dtype.itemsize)
        # do a round-trip test as a sanity check
        factor2 = numpy.zeros_like(factor)
        cutn.contraction_optimizer_config_get_attribute(
            handle, config, attr,
            factor2.ctypes.data, factor2.dtype.itemsize)
        assert factor == factor2


class TestAutotunePreference:

    @manage_resource('handle')
    @manage_resource('autotune')
    def test_autotune_preference_create_destroy(self):
        # simple round-trip test
        pass

    @pytest.mark.parametrize(
        'attr', [val for val in cutn.ContractionAutotunePreferenceAttribute]
    )
    @manage_resource('handle')
    @manage_resource('autotune')
    def test_autotune_preference_get_set_attribute(self, attr):
        handle, pref = self.handle, self.autotune
        dtype = cutn.contraction_autotune_preference_get_attribute_dtype(attr)
        # Hack: assume this is a valid value for all attrs
        factor = numpy.asarray([2], dtype=dtype)
        cutn.contraction_autotune_preference_set_attribute(
            handle, pref, attr,
            factor.ctypes.data, factor.dtype.itemsize)
        # do a round-trip test as a sanity check
        factor2 = numpy.zeros_like(factor)
        cutn.contraction_autotune_preference_get_attribute(
            handle, pref, attr,
            factor2.ctypes.data, factor2.dtype.itemsize)
        assert factor == factor2


@pytest.mark.parametrize(
    'mempool', (None, 'py-callable', 'cffi', 'cffi_struct')
)
@pytest.mark.parametrize(
    'workspace_pref', ("min", "recommended", "max")
)
@pytest.mark.parametrize(
    'autotune', (True, False)
)
@pytest.mark.parametrize(
    'contract', (False, "legacy", "slice_group")
)
@pytest.mark.parametrize(
    'stream', (cupy.cuda.Stream.null, cupy.cuda.Stream(non_blocking=True))
)
class TestContraction(TestTensorNetworkBase):

    # There is no easy way for us to test each API independently, so we instead
    # parametrize the steps and test the whole workflow
    @manage_resource('handle')
    @manage_resource('dscr')
    @manage_resource('info')
    @manage_resource('config')
    @manage_resource('autotune')
    @manage_resource('workspace')
    @manage_resource('slice_group')
    def test_contraction_workflow(
            self, mempool, workspace_pref, autotune, contract, stream):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        # unpack
        handle, dscr, info, config, pref = self.handle, self.dscr, self.info, self.config, self.autotune
        workspace = self.workspace
        tn, input_form, output_form = self.tn, self.input_form, self.output_form

        if mempool:
            mr = MemoryResourceFactory(mempool)
            handler = mr.get_dev_mem_handler()
            cutn.set_device_mem_handler(handle, handler)

        workspace_size = 32*1024**2  # large enough for our test cases
        # we have to run this API in any case in order to create a path
        cutn.contraction_optimize(
            handle, dscr, config, workspace_size, info)

        # manage workspace
        if mempool is None:
            cutn.workspace_compute_sizes(handle, dscr, info, workspace)
            required_size = cutn.workspace_get_size(
                handle, workspace,
                getattr(cutn.WorksizePref, f"{workspace_pref.upper()}"),
                cutn.Memspace.DEVICE)  # TODO: parametrize memspace?
            if workspace_size < required_size:
                assert False, \
                    f"wrong assumption on the workspace size " \
                    f"(given: {workspace_size}, needed: {required_size})"
            workspace_ptr = cupy.cuda.alloc(workspace_size)
            cutn.workspace_set(
                handle, workspace,
                cutn.Memspace.DEVICE,
                workspace_ptr.ptr, workspace_size)
            # round-trip check
            assert (workspace_ptr.ptr, workspace_size) == cutn.workspace_get(
                handle, workspace,
                cutn.Memspace.DEVICE)
        else:
            cutn.workspace_set(
                handle, workspace,
                cutn.Memspace.DEVICE,
                0, 0)  # TODO: check custom workspace size?

        plan = None
        try:
            plan = cutn.create_contraction_plan(
                handle, dscr, info, workspace)
            if autotune:
                cutn.contraction_autotune(
                    handle, plan,
                    tn.get_input_tensors(**input_form),
                    tn.get_output_tensor(),
                    workspace, pref, stream.ptr)

            # we don't care about correctness here, so just contract 1 slice
            # TODO(leofang): check correctness?
            if contract == "legacy":
                cutn.contraction(
                    handle, plan,
                    tn.get_input_tensors(**input_form),
                    tn.get_output_tensor(),
                    workspace, 0, stream.ptr)
            elif contract == "slice_group":
                accumulate = 0
                cutn.contract_slices(
                    handle, plan,
                    tn.get_input_tensors(**input_form),
                    tn.get_output_tensor(),
                    accumulate,
                    workspace, self.slice_group, stream.ptr)
            stream.synchronize()
        finally:
            if plan is not None:
                cutn.destroy_contraction_plan(plan)


@pytest.mark.parametrize(
    'source', ('int', 'seq', 'range')
)
class TestSliceGroup:

    @manage_resource('handle')
    def test_slice_group(self, source):
        # we don't do a simple round-trip test here because there are two
        # flavors of constructors
        if source == "int":
            ids = numpy.arange(10, dtype=numpy.int64)
            slice_group = cutn.create_slice_group_from_ids(
                self.handle, ids.ctypes.data, ids.size)
        elif source == "seq":
            ids = numpy.arange(10, dtype=numpy.int64)
            slice_group = cutn.create_slice_group_from_ids(
                self.handle, ids, ids.size)
        elif source == "range":
            slice_group = cutn.create_slice_group_from_id_range(
                self.handle, 0, 10, 1)
        cutn.destroy_slice_group(slice_group)


# TODO: add more different memory sources
@pytest.mark.parametrize(
    'source', (None, "py-callable", 'cffi', 'cffi_struct')
)
class TestMemHandler:

    @manage_resource('handle')
    def test_set_get_device_mem_handler(self, source):
        if (isinstance(source, str) and source.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        handle = self.handle
        if source is not None:
            mr = MemoryResourceFactory(source)
            handler = mr.get_dev_mem_handler()
            cutn.set_device_mem_handler(handle, handler)
            # round-trip test
            queried_handler = cutn.get_device_mem_handler(handle)
            if source == 'cffi_struct':
                # I'm lazy, otherwise I'd also fetch the functor addresses here...
                assert queried_handler[0] == 0  # ctx is NULL
                assert queried_handler[-1] == source
            else:
                assert queried_handler == handler
        else:
            with pytest.raises(cutn.cuTensorNetError) as e:
                queried_handler = cutn.get_device_mem_handler(handle)
            assert 'CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR' in str(e.value)


class TestLogger:

    def test_logger_set_level(self):
        cutn.logger_set_level(6)  # on
        cutn.logger_set_level(0)  # off

    def test_logger_set_mask(self):
        cutn.logger_set_mask(16)  # should not raise

    def test_logger_set_callback_data(self):
        # we also test logger_open_file() here to avoid polluting stdout

        def callback(level, name, message, my_data, is_ok=False):
            log = f"{level}, {name}, {message} (is_ok={is_ok}) -> logged\n"
            my_data.append(log)

        handle = None
        my_data = []
        is_ok = True

        with tempfile.TemporaryDirectory() as temp:
            file_name = os.path.join(temp, "cutn_test")
            cutn.logger_open_file(file_name)
            cutn.logger_set_callback_data(callback, my_data, is_ok=is_ok)
            cutn.logger_set_level(6)

            try:
                handle = cutn.create()
                cutn.destroy(handle)
            except:
                if handle:
                    cutn.destroy(handle)
                raise
            finally:
                cutn.logger_force_disable()  # to not affect the rest of tests

            with open(file_name) as f:
                log_from_f = f.read()

        # check the log file
        assert '[cutensornetCreate]' in log_from_f
        assert '[cutensornetDestroy]' in log_from_f

        # check the captured data (note we log 2 APIs)
        log = ''.join(my_data)
        assert log.count("-> logged") >= 2
        assert log.count("is_ok=True") >= 2
