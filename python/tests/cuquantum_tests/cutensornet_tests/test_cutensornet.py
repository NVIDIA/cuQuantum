import contextlib
from collections import abc
import functools

import cupy
from cupy import testing
import numpy
import pytest

import cuquantum
from cuquantum import ComputeType, cudaDataType
from cuquantum import cutensornet


###################################################################
#
# As of beta 2, the test suite for Python bindings is kept minimal.
# The sole goal is to ensure the Python arguments are properly
# passed to the C level. We do not ensure coverage nor correctness.
# This decision will be revisited in the future.
#
###################################################################

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
                    h = cutensornet.create()
                elif name == 'dscr':
                    tn, dtype, input_form, output_form = self.tn, self.dtype, self.input_form, self.output_form
                    einsum, shapes = tn  # unpack
                    tn = TensorNetworkFactory(einsum, shapes, dtype)
                    i_n_inputs, i_n_modes, i_extents, i_strides, i_modes, i_alignments = \
                        tn.get_input_metadata(**input_form)
                    o_n_modes, o_extents, o_strides, o_modes, o_alignments = \
                        tn.get_output_metadata(**output_form)
                    h = cutensornet.create_network_descriptor(
                        self.handle,
                        i_n_inputs, i_n_modes, i_extents, i_strides, i_modes, i_alignments,
                        o_n_modes, o_extents, o_strides, o_modes, o_alignments,
                        dtype_to_data_type[dtype], dtype_to_compute_type[dtype])
                    # we also need to keep the tn data alive
                    self.tn = tn
                elif name == 'config':
                    h = cutensornet.create_contraction_optimizer_config(self.handle)
                elif name == 'info':
                    h = cutensornet.create_contraction_optimizer_info(
                        self.handle, self.dscr)
                elif name == 'autotune':
                    h = cutensornet.create_contraction_autotune_preference(self.handle)
                else:
                    assert False, f'name "{name}" not recognized'
                setattr(self, name, h)
                impl(self, *args, **kwargs)
            except:
                print(f'managing resource {name} failed')
                raise
            finally:
                if name == 'handle' and hasattr(self, name):
                    cutensornet.destroy(self.handle)
                    del self.handle
                elif name == 'dscr' and hasattr(self, name):
                    cutensornet.destroy_network_descriptor(self.dscr)
                    del self.dscr
                elif name == 'config' and hasattr(self, name):
                    cutensornet.destroy_contraction_optimizer_config(self.config)
                    del self.config
                elif name == 'info' and hasattr(self, name):
                    cutensornet.destroy_contraction_optimizer_info(self.info)
                    del self.info
                elif name == 'autotune' and hasattr(self, name):
                    cutensornet.destroy_contraction_autotune_preference(self.autotune)
                    del self.autotune
        return test_func
    return decorator


class TestLibHelper:

    def test_get_version(self):
        ver = cutensornet.get_version()
        assert ver == (cutensornet.MAJOR_VER * 10000
            + cutensornet.MINOR_VER * 100
            + cutensornet.PATCH_VER)
        assert ver == cutensornet.VERSION

    def test_get_cudart_version(self):
        ver = cutensornet.get_cudart_version()
        assert ver == cupy.cuda.runtime.runtimeGetVersion()


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
        # simple round-trip test
        pass


class TestOptimizerInfo(TestTensorNetworkBase):

    @manage_resource('handle')
    @manage_resource('dscr')
    @manage_resource('info')
    def test_optimizer_info_create_destroy(self):
        # simple round-trip test
        pass

    @pytest.mark.parametrize(
        'attr', [val for val in cutensornet.ContractionOptimizerInfoAttribute]
    )
    @manage_resource('handle')
    @manage_resource('dscr')
    @manage_resource('info')
    def test_optimizer_info_get_set_attribute(self, attr):
        if attr in (
                cutensornet.ContractionOptimizerInfoAttribute.NUM_SLICES,
                cutensornet.ContractionOptimizerInfoAttribute.PHASE1_FLOP_COUNT,
                cutensornet.ContractionOptimizerInfoAttribute.FLOP_COUNT,
                cutensornet.ContractionOptimizerInfoAttribute.LARGEST_TENSOR,
                cutensornet.ContractionOptimizerInfoAttribute.SLICING_OVERHEAD,
                ):
            pytest.skip("setter not supported")
        elif attr in (
                cutensornet.ContractionOptimizerInfoAttribute.PATH,
                cutensornet.ContractionOptimizerInfoAttribute.SLICED_MODE,
                cutensornet.ContractionOptimizerInfoAttribute.SLICED_EXTENT,
                ):
            pytest.skip("TODO")
        handle, info = self.handle, self.info
        dtype = cutensornet.contraction_optimizer_info_get_attribute_dtype(attr)
        # Hack: assume this is a valid value for all attrs
        factor = numpy.asarray([30], dtype=dtype)
        cutensornet.contraction_optimizer_info_set_attribute(
            handle, info, attr,
            factor.ctypes.data, factor.dtype.itemsize)
        # do a round-trip test as a sanity check
        factor2 = numpy.zeros_like(factor)
        cutensornet.contraction_optimizer_info_get_attribute(
            handle, info, attr,
            factor2.ctypes.data, factor2.dtype.itemsize)
        assert factor == factor2


class TestOptimizerConfig:

    @manage_resource('handle')
    @manage_resource('config')
    def test_optimizer_config_create_destroy(self):
        # simple round-trip test
        pass

    @pytest.mark.parametrize(
        # TODO(leofang): enable this when the getter bug is fixed
        'attr', [val for val in cutensornet.ContractionOptimizerConfigAttribute]
        #'attr', [cutensornet.ContractionOptimizerConfigAttribute.GRAPH_IMBALANCE_FACTOR]
    )
    @manage_resource('handle')
    @manage_resource('config')
    def test_optimizer_config_get_set_attribute(self, attr):
        if attr == cutensornet.ContractionOptimizerConfigAttribute.SIMPLIFICATION_DISABLE_DR:
            pytest.skip("pending on MR 275")
        handle, config = self.handle, self.config
        dtype = cutensornet.contraction_optimizer_config_get_attribute_dtype(attr)
        # Hack: assume this is a valid value for all attrs
        if attr in (cutensornet.ContractionOptimizerConfigAttribute.GRAPH_ALGORITHM,
                    cutensornet.ContractionOptimizerConfigAttribute.SLICER_MEMORY_MODEL,
                    cutensornet.ContractionOptimizerConfigAttribute.SLICER_DISABLE_SLICING):
            factor = numpy.asarray([1], dtype=dtype)
        else:
            factor = numpy.asarray([30], dtype=dtype)
        cutensornet.contraction_optimizer_config_set_attribute(
            handle, config, attr,
            factor.ctypes.data, factor.dtype.itemsize)
        # do a round-trip test as a sanity check
        factor2 = numpy.zeros_like(factor)
        cutensornet.contraction_optimizer_config_get_attribute(
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
        'attr', [val for val in cutensornet.ContractionAutotunePreferenceAttribute]
    )
    @manage_resource('handle')
    @manage_resource('autotune')
    def test_autotune_preference_get_set_attribute(self, attr):
        handle, pref = self.handle, self.autotune
        dtype = cutensornet.contraction_autotune_preference_get_attribute_dtype(attr)
        # Hack: assume this is a valid value for all attrs
        factor = numpy.asarray([10], dtype=dtype)
        cutensornet.contraction_autotune_preference_set_attribute(
            handle, pref, attr,
            factor.ctypes.data, factor.dtype.itemsize)
        # do a round-trip test as a sanity check
        factor2 = numpy.zeros_like(factor)
        cutensornet.contraction_autotune_preference_get_attribute(
            handle, pref, attr,
            factor2.ctypes.data, factor2.dtype.itemsize)
        assert factor == factor2


@pytest.mark.parametrize(
    'get_workspace_size', (True, False)
)
@pytest.mark.parametrize(
    'autotune', (True, False)
)
@pytest.mark.parametrize(
    'contract', (True, False)
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
    def test_contraction_workflow(
            self, get_workspace_size, autotune, contract, stream):
        # unpack
        handle, dscr, info, config, pref = self.handle, self.dscr, self.info, self.config, self.autotune
        tn, input_form, output_form = self.tn, self.input_form, self.output_form

        workspace_size = 4*1024**2  # large enough for our test cases
        # we have to run this API in any case in order to create a path
        cutensornet.contraction_optimize(
            handle, dscr, config, workspace_size, info)
        if get_workspace_size:
            workspace_size = cutensornet.contraction_get_workspace_size(
                handle, dscr, info)
        workspace = cupy.cuda.alloc(workspace_size)

        plan = None
        try:
            plan = cutensornet.create_contraction_plan(
                handle, dscr, info, workspace_size)
            if autotune:
                cutensornet.contraction_autotune(
                    handle, plan,
                    tn.get_input_tensors(**input_form),
                    tn.get_output_tensor(),
                    workspace.ptr, workspace_size, pref, stream.ptr)
            if contract:
                # assume no slicing for simple test cases!
                cutensornet.contraction(
                    handle, plan,
                    tn.get_input_tensors(**input_form),
                    tn.get_output_tensor(),
                    workspace.ptr, workspace_size, 0, stream.ptr)
                # TODO(leofang): check correctness?
            stream.synchronize()
        finally:
            if plan is not None:
                cutensornet.destroy_contraction_plan(plan)
