# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import abc
import functools
import os

import cupy
from cupy import testing
import numpy
try:
    import mpi4py
    from mpi4py import MPI  # init!
except ImportError:
    mpi4py = MPI = None
import pytest

import cuquantum
from cuquantum import ComputeType, cudaDataType
from cuquantum import cutensornet as cutn
from .test_utils import atol_mapper, rtol_mapper

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
                    i_n_inputs, i_n_modes, i_extents, i_strides, i_modes = \
                        tn.get_input_metadata(**input_form)
                    o_n_modes, o_extents, o_strides, o_modes = \
                        tn.get_output_metadata(**output_form)
                    i_qualifiers = numpy.zeros(i_n_inputs, dtype=cutn.tensor_qualifiers_dtype)
                    h = cutn.create_network_descriptor(
                        self.handle,
                        i_n_inputs, i_n_modes, i_extents, i_strides, i_modes, i_qualifiers, 
                        o_n_modes, o_extents, o_strides, o_modes,
                        dtype_to_data_type[dtype], dtype_to_compute_type[dtype])
                    # we also need to keep the tn data alive
                    self.tn = tn
                elif name == 'tensor_decom':
                    tn, dtype, tensor_form = self.tn, self.dtype, self.tensor_form
                    einsum, shapes = tn  # unpack
                    tn = TensorDecompositionFactory(einsum, shapes, dtype)
                    h = []
                    for t in tn.tensor_names:
                        t = cutn.create_tensor_descriptor(
                            self.handle,
                            *tn.get_tensor_metadata(t, **tensor_form),
                            dtype_to_data_type[dtype])
                        h.append(t)
                    # we also need to keep the tn data alive
                    self.tn = tn
                elif name == 'config':
                    h = cutn.create_contraction_optimizer_config(self.handle)
                elif name == 'info':
                    h = cutn.create_contraction_optimizer_info(
                        self.handle, self.dscr)
                elif name == 'svd_config':
                    h = cutn.create_tensor_svd_config(self.handle)
                elif name == 'svd_info':
                    h = cutn.create_tensor_svd_info(self.handle)
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
                elif name == 'tensor_decom' and hasattr(self, name):
                    for t in self.tensor_decom:
                        cutn.destroy_tensor_descriptor(t)
                    del self.tensor_decom
                elif name == 'config' and hasattr(self, name):
                    cutn.destroy_contraction_optimizer_config(self.config)
                    del self.config
                elif name == 'info' and hasattr(self, name):
                    cutn.destroy_contraction_optimizer_info(self.info)
                    del self.info
                elif name == 'svd_config' and hasattr(self, name):
                    cutn.destroy_tensor_svd_config(self.svd_config)
                    del self.svd_config
                elif name == 'svd_info' and hasattr(self, name):
                    cutn.destroy_tensor_svd_info(self.svd_info)
                    del self.svd_info
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

        # xp strides in bytes, cutn strides in counts
        itemsize = cupy.dtype(dtype).itemsize

        self.input_tensors = [
            testing.shaped_random(s, cupy, dtype) for s in i_shapes]
        self.input_n_modes = [len(i) for i in inputs]
        self.input_extents = i_shapes
        self.input_strides = [[stride // itemsize for stride in arr.strides] for arr in self.input_tensors]
        self.input_modes = [tuple([ord(m) for m in i]) for i in inputs]

        self.output_tensor = cupy.empty(o_shape, dtype=dtype)
        self.output_n_modes = len(o_shape)
        self.output_extent = o_shape
        self.output_stride = [stride // itemsize for stride in self.output_tensor.strides]
        self.output_mode = tuple([ord(m) for m in output])

    def _get_data_type(self, category):
        if 'n_modes' in category:
            return numpy.int32
        elif 'extent' in category:
            return numpy.int64
        elif 'stride' in category:
            return numpy.int64
        elif 'mode' in category:
            return numpy.int32
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
        return n_inputs, n_modes, extents, strides, modes

    def get_output_metadata(self, **kwargs):
        n_modes = self.output_n_modes
        extent = self._return_data('output_extent', kwargs.pop('extent'))
        stride = self._return_data('output_stride', kwargs.pop('stride'))
        mode = self._return_data('output_mode', kwargs.pop('mode'))
        return n_modes, extent, stride, mode

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
         'mode': 'int', 'data': 'int'},
        {'n_modes': 'int', 'extent': 'seq', 'stride': 'seq',
         'mode': 'seq', 'data': 'seq'},
        {'n_modes': 'seq', 'extent': 'nested_seq', 'stride': 'nested_seq',
         'mode': 'seq', 'data': 'seq'},
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

    @pytest.mark.parametrize(
        'API', ('old', 'new')
    )
    @manage_resource('handle')
    @manage_resource('dscr')
    def test_descriptor_create_destroy(self, API):
        # we could just do a simple round-trip test, but let's also get
        # this helper API tested
        handle, dscr = self.handle, self.dscr

        if API == 'old':
            # TODO: remove this branch
            num_modes, modes, extents, strides = cutn.get_output_tensor_details(
                handle, dscr)
        else:
            tensor_dscr = cutn.get_output_tensor_descriptor(handle, dscr)
            num_modes, modes, extents, strides = cutn.get_tensor_details(
                handle, tensor_dscr)

        assert num_modes == self.tn.output_n_modes
        assert (modes == numpy.asarray(self.tn.output_mode, dtype=numpy.int32)).all()
        assert (extents == numpy.asarray(self.tn.output_extent, dtype=numpy.int64)).all()
        assert (strides == numpy.asarray(self.tn.output_stride, dtype=numpy.int64)).all()

        if API == 'new':
            cutn.destroy_tensor_descriptor(tensor_dscr)


class TestOptimizerInfo(TestTensorNetworkBase):

    def _get_path(self, handle, info):
        raise NotImplementedError

    def _set_path(self, handle, info, path):
        attr = cutn.ContractionOptimizerInfoAttribute.PATH
        dtype = cutn.contraction_optimizer_info_get_attribute_dtype(attr)
        if not isinstance(path, numpy.ndarray):
            path = numpy.ascontiguousarray(path, dtype=numpy.int32)
        path_obj = numpy.asarray((path.shape[0], path.ctypes.data), dtype=dtype)
        self._set_scalar_attr(handle, info, attr, path_obj)

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
                cutn.ContractionOptimizerInfoAttribute.NUM_SLICED_MODES,
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
                cutn.ContractionOptimizerInfoAttribute.SLICING_CONFIG,
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
            required_size_deprecated = cutn.workspace_get_size(
                handle, workspace,
                getattr(cutn.WorksizePref, f"{workspace_pref.upper()}"),
                cutn.Memspace.DEVICE)  # TODO: parametrize memspace?
            cutn.workspace_compute_contraction_sizes(handle, dscr, info, workspace)
            required_size = cutn.workspace_get_size(
                handle, workspace,
                getattr(cutn.WorksizePref, f"{workspace_pref.upper()}"),
                cutn.Memspace.DEVICE)  # TODO: parametrize memspace?
            assert required_size == required_size_deprecated
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
class TestMemHandler(MemHandlerTestBase):

    mod = cutn
    prefix = "cutensornet"
    error = cutn.cuTensorNetError

    @manage_resource('handle')
    def test_set_get_device_mem_handler(self, source):
        self._test_set_get_device_mem_handler(source, self.handle)


class TensorDecompositionFactory:

    # QR Example: "ab->ax,xb"
    # SVD Example: "ab->ax,x,xb"
    # Gate Example: "ijk,klm,jkpq->->ipk,k,kqm" for indirect gate with singular values returned.
    #               "ijk,klm,jkpq->ipk,-,kqm" for direct gate algorithm with singular values equally partitioned onto u and v

    # self.reconstruct must be a valid einsum expr and can be used to reconstruct
    # the input tensor if no/little truncation was done

    # This factory CANNOT be reused; once a tensor descriptor uses it, it must
    # be discarded.

    svd_partitioned = ('<', '-', '>')  # reserved symbols

    def __init__(self, einsum, shapes, dtype):
        if len(shapes) == 3:
            self.tensor_names = ['input', 'left', 'right']
            self.einsum = einsum
        elif len(shapes) == 5:
            self.tensor_names = ['inputA', 'inputB', 'inputG', 'left', 'right']
            if einsum.count("->") == 1:
                self.gate_algorithm = cutn.GateSplitAlgo.DIRECT
                self.einsum = einsum
            elif einsum.count("->") == 2:
                self.gate_algorithm = cutn.GateSplitAlgo.REDUCED
                self.einsum = einsum.replace("->->", "->")
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        inputs, output = self.einsum.split('->')
        output = output.split(',')
        if len(output) == 2:  # QR
            left, right = output
            self.reconstruct = f"{left},{right}->{inputs}"
            all_modes = [inputs, left, right]
        elif len(output) == 3:  # SVD or Gate
            left, mid_mode, right = output
            common_mode = set(left).intersection(right).pop()
            assert len(common_mode) == 1
            idx_left = left.find(common_mode)
            idx_right = right.find(common_mode)
            self.mid_mode = mid_mode
            
            if len(shapes) == 3: # svd
                all_modes = [inputs, left, right]
                assert shapes[1][idx_left] == shapes[2][idx_right]
                self.mid_extent = shapes[1][idx_left]
                self.reference_einsum = None
                if mid_mode in self.svd_partitioned:
                    # s is already merged into left, both, or right
                    self.reconstruct = f"{left},{right}->{inputs}"
                else:
                    assert mid_mode == common_mode
                    self.reconstruct = f"{left},{common_mode},{right}->{inputs}"
            else: # Gate
                all_modes = list(inputs.split(","))+[left, right]
                assert shapes[3][idx_left] == shapes[4][idx_right]
                self.mid_extent = shapes[3][idx_left]
                contracted_output_modes = "".join((set(left) | set(right)) - (set(left) & set(right)))
                self.reference_einsum = f"{inputs}->{contracted_output_modes}"
                if mid_mode in self.svd_partitioned:
                    # s is already merged into left, both, or right
                    self.reconstruct = f"{left},{right}->{contracted_output_modes}"
                else:
                    assert mid_mode == common_mode
                    self.reconstruct = f"{left},{common_mode},{right}->{contracted_output_modes}"
        else:
            assert False
        del output

        # xp strides in bytes, cutn strides in counts
        dtype = cupy.dtype(dtype)
        itemsize = dtype.itemsize

        for name, shape, modes in zip(self.tensor_names, shapes, all_modes):
            real_dtype = dtype.char.lower()
            if name.startswith('input'):
                if dtype.char != real_dtype:  # complex
                    arr = (cupy.random.random(shape, dtype=real_dtype)
                           + 1j*cupy.random.random(shape, dtype=real_dtype)).astype(dtype)
                else:
                    arr = cupy.random.random(shape, dtype=dtype)
            else:
                arr = cupy.empty(shape, dtype=dtype, order='F')
            setattr(self, f'{name}_tensor', arr)
            setattr(self, f'{name}_n_modes', len(arr.shape))
            setattr(self, f'{name}_extent', arr.shape)
            setattr(self, f'{name}_stride', [stride // itemsize for stride in arr.strides])
            setattr(self, f'{name}_mode', tuple([ord(m) for m in modes]))

    def _get_data_type(self, category):
        if 'n_modes' in category:
            return numpy.int32
        elif 'extent' in category:
            return numpy.int64
        elif 'stride' in category:
            return numpy.int64
        elif 'mode' in category:
            return numpy.int32
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
            else:
                # return int as void*
                data = numpy.asarray(data, dtype=self._get_data_type(category))
                setattr(self, category, data)  # keep data alive
            return data.ctypes.data
        elif return_value == 'seq':
            return data
        else:
            assert False

    def get_tensor_metadata(self, name, **kwargs):
        assert name in self.tensor_names
        n_modes = getattr(self, f'{name}_n_modes')
        extent = self._return_data(f'{name}_extent', kwargs.pop('extent'))
        stride = self._return_data(f'{name}_stride', kwargs.pop('stride'))
        mode = self._return_data(f'{name}_mode', kwargs.pop('mode'))
        return n_modes, extent, stride, mode

    def get_tensor_ptr(self, name):
        return getattr(self, f'{name}_tensor').data.ptr


@testing.parameterize(*testing.product({
    'tn': (
        ('ab->ax,xb', [(8, 8), (8, 8), (8, 8)]),
        ('ab->ax,bx', [(8, 8), (8, 8), (8, 8)]),
        ('ab->xa,xb', [(8, 8), (8, 8), (8, 8)]),
        ('ab->xa,bx', [(8, 8), (8, 8), (8, 8)]),
        ('ab->ax,xb', [(6, 8), (6, 6), (6, 8)]),
        ('ab->ax,bx', [(6, 8), (6, 6), (8, 6)]),
        ('ab->xa,xb', [(6, 8), (6, 6), (6, 8)]),
        ('ab->xa,bx', [(6, 8), (6, 6), (8, 6)]),
        ('ab->ax,xb', [(8, 6), (8, 6), (6, 6)]),
        ('ab->ax,bx', [(8, 6), (8, 6), (6, 6)]),
        ('ab->xa,xb', [(8, 6), (6, 8), (6, 6)]),
        ('ab->xa,bx', [(8, 6), (6, 8), (6, 6)]),
    ),
    'dtype': (
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128
    ),
    'tensor_form': (
        {'extent': 'int', 'stride': 'int', 'mode': 'int'},
        {'extent': 'seq', 'stride': 'seq', 'mode': 'seq'},
    ),
}))
class TestTensorQR:

    # There is no easy way for us to test each API independently, so we instead
    # parametrize the steps and test the whole workflow
    @manage_resource('handle')
    @manage_resource('tensor_decom')
    @manage_resource('workspace')
    def test_tensor_qr(self):
        # unpack
        handle, tn, workspace = self.handle, self.tn, self.workspace
        tensor_in, tensor_q, tensor_r = self.tensor_decom
        dtype = cupy.dtype(self.dtype)

        # prepare workspace
        cutn.workspace_compute_qr_sizes(
            handle, tensor_in, tensor_q, tensor_r, workspace)
        # for now host workspace is always 0, so just query device one
        # also, it doesn't matter which one (min/recommended/max) is queried
        required_size = cutn.workspace_get_size(
            handle, workspace, cutn.WorksizePref.MIN,
            cutn.Memspace.DEVICE)  # TODO: parametrize memspace?
        if required_size > 0:
            workspace_ptr = cupy.cuda.alloc(required_size)
            cutn.workspace_set(
                handle, workspace, cutn.Memspace.DEVICE,
                workspace_ptr.ptr, required_size)
            # round-trip check
            assert (workspace_ptr.ptr, required_size) == cutn.workspace_get(
                handle, workspace, cutn.Memspace.DEVICE)

        # perform QR
        stream = cupy.cuda.get_current_stream().ptr  # TODO
        cutn.tensor_qr(
            handle, tensor_in, tn.get_tensor_ptr('input'),
            tensor_q, tn.get_tensor_ptr('left'),
            tensor_r, tn.get_tensor_ptr('right'),
            workspace, stream)

        # we add a minimal correctness check here as we are not protected by
        # any high-level API yet
        out = cupy.einsum(tn.reconstruct, tn.left_tensor, tn.right_tensor)
        assert cupy.allclose(out, tn.input_tensor,
                             rtol=rtol_mapper[dtype.name],
                             atol=atol_mapper[dtype.name])


# TODO: expand tests:
#  - add truncation
#  - use config (cutoff & normalization)
@testing.parameterize(*testing.product({
    'tn': (
        # no truncation, no partition
        ('ab->ax,x,xb', [(8, 8), (8, 8), (8, 8)]),
        ('ab->ax,x,bx', [(8, 8), (8, 8), (8, 8)]),
        ('ab->xa,x,xb', [(8, 8), (8, 8), (8, 8)]),
        ('ab->xa,x,bx', [(8, 8), (8, 8), (8, 8)]),
        ('ab->ax,x,xb', [(6, 8), (6, 6), (6, 8)]),
        ('ab->ax,x,bx', [(6, 8), (6, 6), (8, 6)]),
        ('ab->xa,x,xb', [(6, 8), (6, 6), (6, 8)]),
        ('ab->xa,x,bx', [(6, 8), (6, 6), (8, 6)]),
        ('ab->ax,x,xb', [(8, 6), (8, 6), (6, 6)]),
        ('ab->ax,x,bx', [(8, 6), (8, 6), (6, 6)]),
        ('ab->xa,x,xb', [(8, 6), (6, 8), (6, 6)]),
        ('ab->xa,x,bx', [(8, 6), (6, 8), (6, 6)]),
        # no truncation, partition to u
        ('ab->ax,<,xb', [(8, 8), (8, 8), (8, 8)]),
        ('ab->ax,<,bx', [(8, 8), (8, 8), (8, 8)]),
        ('ab->xa,<,xb', [(8, 8), (8, 8), (8, 8)]),
        ('ab->xa,<,bx', [(8, 8), (8, 8), (8, 8)]),
        # no truncation, partition to v
        ('ab->ax,>,xb', [(8, 8), (8, 8), (8, 8)]),
        ('ab->ax,>,bx', [(8, 8), (8, 8), (8, 8)]),
        ('ab->xa,>,xb', [(8, 8), (8, 8), (8, 8)]),
        ('ab->xa,>,bx', [(8, 8), (8, 8), (8, 8)]),
        # no truncation, partition to both
        ('ab->ax,-,xb', [(8, 8), (8, 8), (8, 8)]),
        ('ab->ax,-,bx', [(8, 8), (8, 8), (8, 8)]),
        ('ab->xa,-,xb', [(8, 8), (8, 8), (8, 8)]),
        ('ab->xa,-,bx', [(8, 8), (8, 8), (8, 8)]),
    ),
    'dtype': (
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128
    ),
    'tensor_form': (
        {'extent': 'int', 'stride': 'int', 'mode': 'int'},
        {'extent': 'seq', 'stride': 'seq', 'mode': 'seq'},
    ),
}))
class TestTensorSVD:

    def _get_scalar_attr(self, handle, obj_type, obj, attr):
        if obj_type == 'config':
            dtype_getter = cutn.tensor_svd_config_get_attribute_dtype
            getter = cutn.tensor_svd_config_get_attribute
        elif obj_type == 'info':
            dtype_getter = cutn.tensor_svd_info_get_attribute_dtype
            getter = cutn.tensor_svd_info_get_attribute
        else:
            assert False

        dtype = dtype_getter(attr)
        data = numpy.empty((1,), dtype=dtype)
        getter(handle, obj, attr, data.ctypes.data, data.dtype.itemsize)
        return data

    def _set_scalar_attr(self, handle, obj_type, obj, attr, data):
        assert obj_type == 'config'  # svd info has no setter
        dtype_getter = cutn.tensor_svd_config_get_attribute_dtype
        setter = cutn.tensor_svd_config_set_attribute

        dtype = dtype_getter(attr)
        if not isinstance(data, numpy.ndarray):
            data = numpy.asarray(data, dtype=dtype)
        setter(handle, obj, attr, data.ctypes.data, data.dtype.itemsize)

    # There is no easy way for us to test each API independently, so we instead
    # parametrize the steps and test the whole workflow
    @manage_resource('handle')
    @manage_resource('tensor_decom')
    @manage_resource('svd_config')
    @manage_resource('svd_info')
    @manage_resource('workspace')
    def test_tensor_svd(self):
        # unpack
        handle, tn, workspace = self.handle, self.tn, self.workspace
        tensor_in, tensor_u, tensor_v = self.tensor_decom
        svd_config, svd_info = self.svd_config, self.svd_info
        dtype = cupy.dtype(self.dtype)

        # prepare workspace
        cutn.workspace_compute_svd_sizes(
            handle, tensor_in, tensor_u, tensor_v, svd_config, workspace)
        # for now host workspace is always 0, so just query device one
        # also, it doesn't matter which one (min/recommended/max) is queried
        required_size = cutn.workspace_get_size(
            handle, workspace, cutn.WorksizePref.MIN,
            cutn.Memspace.DEVICE)  # TODO: parametrize memspace?
        if required_size > 0:
            workspace_ptr = cupy.cuda.alloc(required_size)
            cutn.workspace_set(
                handle, workspace, cutn.Memspace.DEVICE,
                workspace_ptr.ptr, required_size)
            # round-trip check
            assert (workspace_ptr.ptr, required_size) == cutn.workspace_get(
                handle, workspace, cutn.Memspace.DEVICE)

        # set singular value partitioning, if requested
        if tn.mid_mode in tn.svd_partitioned:
            if tn.mid_mode == '<':
                data = cutn.TensorSVDPartition.US
            elif tn.mid_mode == '-':
                data = cutn.TensorSVDPartition.UV_EQUAL
            else:  # = '<':
                data = cutn.TensorSVDPartition.SV
            self._set_scalar_attr(
                handle, 'config', svd_config,
                cutn.TensorSVDConfigAttribute.S_PARTITION,
                data)
            # do a round-trip test as a sanity check
            factor = self._get_scalar_attr(
                handle, 'config', svd_config,
                cutn.TensorSVDConfigAttribute.S_PARTITION)
            assert factor == data

        # perform SVD
        stream = cupy.cuda.get_current_stream().ptr  # TODO
        if tn.mid_mode in tn.svd_partitioned:
            s_ptr = 0
        else:
            s = cupy.empty(tn.mid_extent, dtype=dtype.char.lower())
            s_ptr = s.data.ptr
        cutn.tensor_svd(
            handle, tensor_in, tn.get_tensor_ptr('input'),
            tensor_u, tn.get_tensor_ptr('left'),
            s_ptr,
            tensor_v, tn.get_tensor_ptr('right'),
            svd_config, svd_info, workspace, stream)

        # sanity checks (only valid for no truncation)
        assert tn.mid_extent == self._get_scalar_attr(
            handle, 'info', svd_info,
            cutn.TensorSVDInfoAttribute.FULL_EXTENT)
        assert tn.mid_extent == self._get_scalar_attr(
            handle, 'info', svd_info,
            cutn.TensorSVDInfoAttribute.REDUCED_EXTENT)
        assert 0 == self._get_scalar_attr(
            handle, 'info', svd_info,
            cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT)

        # we add a minimal correctness check here as we are not protected by
        # any high-level API yet
        if tn.mid_mode in tn.svd_partitioned:
            out = cupy.einsum(
                tn.reconstruct, tn.left_tensor, tn.right_tensor)
        else:
            out = cupy.einsum(
                tn.reconstruct, tn.left_tensor, s, tn.right_tensor)
        assert cupy.allclose(out, tn.input_tensor,
                             rtol=rtol_mapper[dtype.name],
                             atol=atol_mapper[dtype.name])


# TODO: expand tests:
#  - add truncation
#  - use config (cutoff & normalization)
@testing.parameterize(*testing.product({
    'tn': (
        # direct algorithm, no truncation, no partition
        ('ijk,klm,jlpq->ipk,k,kqm', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (4, 2, 8), (8, 2, 4)]),
        ('ijk,klm,jlpq->kpi,k,qmk', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (8, 2, 4), (2, 4, 8)]),
        ('ijk,klm,jlpq->pki,k,mkq', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (2, 8, 4), (4, 8, 2)]),
        # direct algorithm, no truncation, partition onto u
        ('ijk,klm,jlpq->ipk,<,kqm', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (4, 2, 8), (8, 2, 4)]),
        ('ijk,klm,jlpq->kpi,<,qmk', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (8, 2, 4), (2, 4, 8)]),
        ('ijk,klm,jlpq->pki,<,mkq', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (2, 8, 4), (4, 8, 2)]),
        # direct algorithm, no truncation, partition onto v
        ('ijk,klm,jlpq->ipk,>,kqm', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (4, 2, 8), (8, 2, 4)]),
        ('ijk,klm,jlpq->kpi,>,qmk', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (8, 2, 4), (2, 4, 8)]),
        ('ijk,klm,jlpq->pki,>,mkq', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (2, 8, 4), (4, 8, 2)]),
        # direct algorithm, no truncation, partition onto u and v equally
        ('ijk,klm,jlpq->ipk,-,kqm', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (4, 2, 8), (8, 2, 4)]),
        ('ijk,klm,jlpq->kpi,-,qmk', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (8, 2, 4), (2, 4, 8)]),
        ('ijk,klm,jlpq->pki,-,mkq', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (2, 8, 4), (4, 8, 2)]),
        # reduced algorithm, no truncation, no partition
        ('ijk,klm,jlpq->->ipk,k,kqm', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (4, 2, 8), (8, 2, 4)]),
        ('ijk,klm,jlpq->->kpi,k,qmk', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (8, 2, 4), (2, 4, 8)]),
        ('ijk,klm,jlpq->->pki,k,mkq', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (2, 8, 4), (4, 8, 2)]),
        # reduced algorithm, no truncation, partition onto u
        ('ijk,klm,jlpq->->ipk,<,kqm', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (4, 2, 8), (8, 2, 4)]),
        ('ijk,klm,jlpq->->kpi,<,qmk', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (8, 2, 4), (2, 4, 8)]),
        ('ijk,klm,jlpq->->pki,<,mkq', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (2, 8, 4), (4, 8, 2)]),
        # reduced algorithm, no truncation, partition onto v
        ('ijk,klm,jlpq->->ipk,>,kqm', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (4, 2, 8), (8, 2, 4)]),
        ('ijk,klm,jlpq->->kpi,>,qmk', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (8, 2, 4), (2, 4, 8)]),
        ('ijk,klm,jlpq->->pki,>,mkq', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (2, 8, 4), (4, 8, 2)]),
        # reduced algorithm, no truncation, partition onto u and v equally
        ('ijk,klm,jlpq->->ipk,-,kqm', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (4, 2, 8), (8, 2, 4)]),
        ('ijk,klm,jlpq->->kpi,-,qmk', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (8, 2, 4), (2, 4, 8)]),
        ('ijk,klm,jlpq->->pki,-,mkq', [(4, 2, 4), (4, 2, 4), (2, 2, 2, 2), (2, 8, 4), (4, 8, 2)]),
    ),
    'dtype': (
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128
    ),
    'tensor_form': (
        {'extent': 'int', 'stride': 'int', 'mode': 'int'},
        {'extent': 'seq', 'stride': 'seq', 'mode': 'seq'},
    ),
}))
class TestTensorGate:

    def _get_scalar_attr(self, handle, obj_type, obj, attr):
        if obj_type == 'config':
            dtype_getter = cutn.tensor_svd_config_get_attribute_dtype
            getter = cutn.tensor_svd_config_get_attribute
        elif obj_type == 'info':
            dtype_getter = cutn.tensor_svd_info_get_attribute_dtype
            getter = cutn.tensor_svd_info_get_attribute
        else:
            assert False

        dtype = dtype_getter(attr)
        data = numpy.empty((1,), dtype=dtype)
        getter(handle, obj, attr, data.ctypes.data, data.dtype.itemsize)
        return data

    def _set_scalar_attr(self, handle, obj_type, obj, attr, data):
        assert obj_type == 'config'  # svd info has no setter
        dtype_getter = cutn.tensor_svd_config_get_attribute_dtype
        setter = cutn.tensor_svd_config_set_attribute

        dtype = dtype_getter(attr)
        if not isinstance(data, numpy.ndarray):
            data = numpy.asarray(data, dtype=dtype)
        setter(handle, obj, attr, data.ctypes.data, data.dtype.itemsize)

    # There is no easy way for us to test each API independently, so we instead
    # parametrize the steps and test the whole workflow
    @manage_resource('handle')
    @manage_resource('tensor_decom')
    @manage_resource('svd_config')
    @manage_resource('svd_info')
    @manage_resource('workspace')
    def test_gate_split(self):
        # unpack
        handle, tn, workspace = self.handle, self.tn, self.workspace
        tensor_in_a, tensor_in_b, tensor_in_g, tensor_u, tensor_v = self.tensor_decom
        gate_algorithm = tn.gate_algorithm
        svd_config, svd_info = self.svd_config, self.svd_info
        dtype = cupy.dtype(self.dtype)
        compute_type = dtype_to_compute_type[self.dtype]
        # prepare workspace
        cutn.workspace_compute_gate_split_sizes(handle, 
            tensor_in_a, tensor_in_b, tensor_in_g, tensor_u, tensor_v, 
            gate_algorithm, svd_config, compute_type, workspace)
        # for now host workspace is always 0, so just query device one
        # also, it doesn't matter which one (min/recommended/max) is queried
        required_size = cutn.workspace_get_size(
            handle, workspace, cutn.WorksizePref.MIN,
            cutn.Memspace.DEVICE)  # TODO: parametrize memspace?
        if required_size > 0:
            workspace_ptr = cupy.cuda.alloc(required_size)
            cutn.workspace_set(
                handle, workspace, cutn.Memspace.DEVICE,
                workspace_ptr.ptr, required_size)
            # round-trip check
            assert (workspace_ptr.ptr, required_size) == cutn.workspace_get(
                handle, workspace, cutn.Memspace.DEVICE)

        # set singular value partitioning, if requested
        if tn.mid_mode in tn.svd_partitioned:
            if tn.mid_mode == '<':
                data = cutn.TensorSVDPartition.US
            elif tn.mid_mode == '-':
                data = cutn.TensorSVDPartition.UV_EQUAL
            else:  # = '<':
                data = cutn.TensorSVDPartition.SV
            self._set_scalar_attr(
                handle, 'config', svd_config,
                cutn.TensorSVDConfigAttribute.S_PARTITION,
                data)
            # do a round-trip test as a sanity check
            factor = self._get_scalar_attr(
                handle, 'config', svd_config,
                cutn.TensorSVDConfigAttribute.S_PARTITION)
            assert factor == data

        # perform gate split
        stream = cupy.cuda.get_current_stream().ptr  # TODO
        if tn.mid_mode in tn.svd_partitioned:
            s_ptr = 0
        else:
            s = cupy.empty(tn.mid_extent, dtype=dtype.char.lower())
            s_ptr = s.data.ptr
        cutn.gate_split(handle, tensor_in_a, tn.get_tensor_ptr('inputA'),
            tensor_in_b, tn.get_tensor_ptr('inputB'),
            tensor_in_g, tn.get_tensor_ptr('inputG'),
            tensor_u, tn.get_tensor_ptr('left'), s_ptr, 
            tensor_v, tn.get_tensor_ptr('right'),
            gate_algorithm, svd_config, compute_type, 
            svd_info, workspace, stream)
      
        # sanity checks (only valid for no truncation)
        assert tn.mid_extent == self._get_scalar_attr(
            handle, 'info', svd_info,
            cutn.TensorSVDInfoAttribute.FULL_EXTENT)
        assert tn.mid_extent == self._get_scalar_attr(
            handle, 'info', svd_info,
            cutn.TensorSVDInfoAttribute.REDUCED_EXTENT)
        assert 0 == self._get_scalar_attr(
            handle, 'info', svd_info,
            cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT)

        # we add a minimal correctness check here as we are not protected by
        # any high-level API yet
        if tn.mid_mode in tn.svd_partitioned:
            out = cupy.einsum(
                tn.reconstruct, tn.left_tensor, tn.right_tensor)
        else:
            out = cupy.einsum(
                tn.reconstruct, tn.left_tensor, s, tn.right_tensor)
        reference = cupy.einsum(tn.reference_einsum, tn.inputA_tensor, tn.inputB_tensor, tn.inputG_tensor)
        error = cupy.linalg.norm(out - reference)
        assert cupy.allclose(out, reference,
                             rtol=rtol_mapper[dtype.name],
                             atol=atol_mapper[dtype.name])


class TestTensorSVDConfig:

    @manage_resource('handle')
    @manage_resource('svd_config')
    def test_tensor_svd_config_create_destroy(self):
        # simple round-trip test
        pass

    @pytest.mark.parametrize(
        'attr', [val for val in cutn.TensorSVDConfigAttribute]
    )
    @manage_resource('handle')
    @manage_resource('svd_config')
    def test_tensor_svd_config_get_set_attribute(self, attr):
        handle, svd_config = self.handle, self.svd_config
        dtype = cutn.tensor_svd_config_get_attribute_dtype(attr)
        # Hack: assume this is a valid value for all attrs
        factor = numpy.asarray([0.8], dtype=dtype)
        cutn.tensor_svd_config_set_attribute(
            handle, svd_config, attr,
            factor.ctypes.data, factor.dtype.itemsize)
        # do a round-trip test as a sanity check
        factor2 = numpy.zeros_like(factor)
        cutn.tensor_svd_config_get_attribute(
            handle, svd_config, attr,
            factor2.ctypes.data, factor2.dtype.itemsize)
        assert factor == factor2


@pytest.mark.skipif(mpi4py is None, reason="need mpi4py")
@pytest.mark.skipif(os.environ.get("CUTENSORNET_COMM_LIB") is None,
                    reason="wrapper lib not set")
class TestDistributed:

    def _get_comm(self, comm):
        if comm == 'world':
            return MPI.COMM_WORLD.Dup()
        elif comm == 'self':
            return MPI.COMM_SELF.Dup()
        else:
            assert False

    @pytest.mark.parametrize(
        'comm', ('world', 'self'),
    )
    @manage_resource('handle')
    def test_distributed(self, comm):
        handle = self.handle
        comm = self._get_comm(comm)
        cutn.distributed_reset_configuration(
            handle, *cutn.get_mpi_comm_pointer(comm))
        assert comm.Get_size() == cutn.distributed_get_num_ranks(handle)
        assert comm.Get_rank() == cutn.distributed_get_proc_rank(handle)
        cutn.distributed_synchronize(handle)
        # no need to free the comm, for world/self mpi4py does it for us...


class TestLogger(LoggerTestBase):

    mod = cutn
    prefix = "cutensornet"
