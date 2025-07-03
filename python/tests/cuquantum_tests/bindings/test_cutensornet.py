# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import abc
import functools
import os

import cupy as cp
from cupy import testing
import numpy as np
try:
    import mpi4py
    from mpi4py import MPI  # init!
except ImportError:
    mpi4py = MPI = None
import pytest
try:
    import torch
    # unlike in other test modules, we don't check torch.cuda.is_available()
    # here because we allow verifying against PyTorch CPU tensors
except:
    torch = None

from cuquantum import ComputeType
from cuquantum.bindings import cutensornet as cutn
from cuquantum.tensornet import tensor, get_mpi_comm_pointer
from cuquantum.tensornet._internal.decomposition_utils import get_svd_info_dict, parse_svd_config
from cuquantum._internal.utils import check_or_create_options

from ..tensornet.utils import approxTN_utils
from ..tensornet.utils.data import gate_decomp_expressions, tensor_decomp_expressions
from ..tensornet.utils.test_utils import get_stream_for_backend
from . import (_can_use_cffi, dtype_to_compute_type, dtype_to_data_type, 
               MemHandlerTestBase, MemoryResourceFactory, LoggerTestBase, BindingsDeprecationTestBase)


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
            # "self" refers to the test case
            try:
                if name == 'handle':
                    h = cutn.create()
                elif name == 'state':
                    self.simple_state = SimpleState(self.num_qubits, self.extents, self.strides, self.dtype, self.order, self.mps)
                    self.qubits_dims_or_ptr = self.simple_state.qubits_dims if self.extents == 'seq' else self.simple_state.qubits_dims.ctypes.data
                    h = self.state = cutn.create_state(self.handle, self.state_purity, self.num_qubits, 
                                                       self.qubits_dims_or_ptr, dtype_to_data_type[self.dtype])
                    gate_h_strides = self.simple_state.gate_h_strides if self.strides == 'seq' else self.simple_state.gate_h_strides.ctypes.data
                    gate_cx_strides = self.simple_state.gate_cx_strides if self.strides == 'seq' else self.simple_state.gate_cx_strides.ctypes.data
                    cutn.state_apply_tensor_operator(self.handle, self.state, 1, (0,), 
                                                     self.simple_state.gate_h.data.ptr, gate_h_strides, 1, 0, 1)
                    for i in range(1, self.num_qubits):
                        cutn.state_apply_tensor_operator(self.handle, self.state, 2, (i-1, i),
                                                         self.simple_state.gate_cx.data.ptr, gate_cx_strides, 1, 0, 1)    
                    if self.mps:
                        cutn.state_finalize_mps(self.handle, self.state, cutn.BoundaryCondition.OPEN, 
                                                self.simple_state.mps_tensor_extents, self.simple_state.mps_tensor_strides)
                        self.simple_state._setup_mps_state(self.handle, self.state, self.workspace, self.stream)
                elif name == 'accessor':
                    fixed_modes = ()
                    num_fixed_modes = len(fixed_modes)
                    amplitudes_shape = self.simple_state.get_amplitudes_shape(fixed_modes)
                    self.amplitudes = cp.empty(amplitudes_shape, dtype=self.dtype, order=self.order)
                    amplitudes_strides = [s // self.amplitudes.itemsize for s in self.amplitudes.strides]
                    h = cutn.create_accessor(self.handle, self.state, num_fixed_modes, fixed_modes, amplitudes_strides)
                elif name == 'sampler':
                    h = cutn.create_sampler(self.handle, self.state, self.num_qubits, 0)
                elif name == 'hamiltonian':
                    if self.dtype in [np.float32, np.float64]:
                        pytest.skip("skipping test for non-complex dtype as we use Y gate as complex")
                    h = self.hamiltonian = cutn.create_network_operator(self.handle, self.num_qubits, self.qubits_dims_or_ptr, dtype_to_data_type[self.dtype])
                    # Construct a tensor network operator: (0.5 * Z1 * Z2) + (0.25 * Y3)
                    self.simple_state.get_gate_z_y()
                    num_modes = (1, 1) # Z1 acts on 1 mode, Z2 acts on 1 mode
                    modes_Z1, modes_Z2 = (1, ), (2, ) 
                    state_modes = (modes_Z1, modes_Z2) # state modes (Z1 * Z2) acts on
                    gate_data = (self.simple_state.gate_z.data.ptr, self.simple_state.gate_z.data.ptr) 
                    cutn.network_operator_append_product(self.handle, self.hamiltonian, 0.5, 2, num_modes, state_modes, 0, gate_data)
                    num_modes = (1, ) # Y3 acts on 1 mode
                    modes_Y3 = (3, ) 
                    state_modes = (modes_Y3, ) # state modes (Y3) acts on
                    gate_data = (self.simple_state.gate_y.data.ptr, ) 
                    cutn.network_operator_append_product(self.handle, self.hamiltonian, 0.25, 1, num_modes, state_modes, 0, gate_data)
                elif name == 'expectation':
                    h = cutn.create_expectation(self.handle, self.state, self.hamiltonian)
                elif name == 'marginal':
                    marginal_modes = (0, 1) # open qubits
                    num_marginal_modes = len(marginal_modes)
                    rdm_shape = self.simple_state.get_rdm_shape(marginal_modes)
                    self.rdm = cp.empty(rdm_shape, dtype=self.dtype)
                    rdm_strides = [stride_in_bytes // self.rdm.itemsize for stride_in_bytes in self.rdm.strides]
                    h = cutn.create_marginal(self.handle, self.state, num_marginal_modes, marginal_modes, 0, 0, rdm_strides)
                elif name == 'dscr':
                    tn, dtype, input_form = self.tn, self.dtype, self.input_form
                    einsum, shapes = tn  # unpack
                    tn = TensorNetworkFactory(einsum, shapes, dtype, order=self.order)
                    i_n_inputs, i_n_modes, i_extents, i_strides, i_modes = \
                        tn.get_input_metadata(**input_form)
                    o_n_modes, o_extents, o_strides, o_modes = \
                        tn.get_output_metadata(**input_form)
                    i_qualifiers = np.zeros(i_n_inputs, dtype=cutn.tensor_qualifiers_dtype)
                    if self.qual is not None:
                        i_qualifiers['requires_gradient'][:] = True
                    h = cutn.create_network_descriptor(
                        self.handle,
                        i_n_inputs, i_n_modes, i_extents, i_strides, i_modes, i_qualifiers, 
                        o_n_modes, o_extents, o_strides, o_modes,
                        dtype_to_data_type[dtype], dtype_to_compute_type[dtype])
                    # we also need to keep the tn data alive
                    self.tn = tn
                elif name == 'tensor_decom':
                    tn, dtype, tensor_form = self.tn, self.dtype, self.tensor_form
                    options = getattr(self, "options", {})
                    max_extent = options.get("max_extent", None)
                    subscript, shapes = tn  # unpack
                    tn = TensorDecompositionFactory(subscript, shapes, dtype, max_extent=max_extent, order=self.order)
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
                elif name == 'state':
                    cutn.destroy_state(self.state)
                    del self.state
                    del self.simple_state
                elif name == 'accessor':
                    cutn.destroy_accessor(self.accessor)
                    del self.accessor
                    del self.amplitudes
                elif name == 'sampler':
                    cutn.destroy_sampler(self.sampler)
                    del self.sampler
                elif name == 'hamiltonian':
                    if hasattr(self, 'hamiltonian'):
                        cutn.destroy_network_operator(self.hamiltonian)
                        del self.hamiltonian
                elif name == 'expectation':
                    if hasattr(self, 'expectation'):
                        cutn.destroy_expectation(self.expectation)
                        del self.expectation
                elif name == 'marginal':
                    cutn.destroy_marginal(self.marginal)
                    del self.marginal
        return test_func
    return decorator


class TestLibHelper:

    def test_get_version(self):
        ver = cutn.get_version()
        assert isinstance(ver, int)

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


class SimpleState:

    def __init__(self, num_qubits, extents, strides, dtype, order, mps=False):
        self.num_qubits = num_qubits
        self.extents = extents
        self.strides = strides
        self.dtype = dtype
        self.order = order
        self.mps = mps
        self.qubits_dims = np.array([2,] * self.num_qubits, dtype=np.int64)

        gate_h = 2**-0.5 * cp.asarray([[1,1], [1,-1]],).reshape(2,2, order='F')
        self.gate_h = gate_h.astype(self.dtype, order=self.order)
        self.gate_h_strides = np.array([s // self.gate_h.itemsize for s in self.gate_h.strides], dtype=np.int64)

        gate_cx = cp.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],).reshape(2, 2, 2, 2, order='F')
        self.gate_cx = gate_cx.astype(self.dtype, order=self.order)
        self.gate_cx_strides = np.array([s // self.gate_cx.itemsize for s in self.gate_cx.strides], dtype=np.int64)

        if self.mps:
            self._setup_mps_tensors()

    def get_gate_z_y(self):
        gate_z = cp.asarray([[1, 0], [0, -1]],).reshape(2,2, order='F')
        self.gate_z = gate_z.astype(self.dtype, order=self.order)
        self.gate_z_strides = np.array([s // self.gate_z.itemsize for s in self.gate_z.strides], dtype=np.int64)
        if self.dtype in [np.complex64, np.complex128]:
            gate_y = cp.asarray([[0, -1j], [1j, 0]],).reshape(2,2, order='F')
            self.gate_y = gate_y.astype(self.dtype, order=self.order)
            self.gate_y_strides = np.array([s // self.gate_y.itemsize for s in self.gate_y.strides], dtype=np.int64)

    def get_amplitudes_shape(self, fixed_modes):
        amplitudes_shape = [self.qubits_dims[i] for i in range(self.num_qubits) if i not in fixed_modes]
        return amplitudes_shape
    
    def get_rdm_shape(self, marginal_modes):
        dims = tuple(self.qubits_dims[m] for m in marginal_modes)
        rdm_shape = dims * 2
        return rdm_shape
    
    def _setup_mps_tensors(self):
        max_extent = self.mps
        self.mps_tensor_extents = []
        self.mps_tensor_strides = []
        self.mps_tensors = []
        self.mps_tensor_ptrs = []
        for i in range(self.num_qubits):
            if i == 0:
                extents = (2, max_extent)
            elif i == self.num_qubits - 1:
                extents = (max_extent, 2)
            else:
                extents = (max_extent, 2, max_extent)
            self.mps_tensor_extents.append(extents)
            tensor = cp.zeros(extents, dtype=self.dtype, order=self.order)
            self.mps_tensors.append(tensor)
            self.mps_tensor_ptrs.append(tensor.data.ptr)
            self.mps_tensor_strides.append([stride_in_bytes // tensor.itemsize for stride_in_bytes in tensor.strides])

    def _setup_mps_state(self, handle, state, workspace, stream):
        svd_algorithm_dtype = cutn.state_get_attribute_dtype(cutn.StateAttribute.MPS_SVD_CONFIG_ALGO)
        svd_algorithm = np.array(cutn.TensorSVDAlgo.GESVDJ, dtype=svd_algorithm_dtype)
        cutn.state_configure(handle, state, cutn.StateAttribute.MPS_SVD_CONFIG_ALGO, 
                                svd_algorithm.ctypes.data, svd_algorithm.dtype.itemsize)
        free_mem = cp.cuda.Device().mem_info[0]
        max_scratch_size = free_mem // 4 
        cutn.state_prepare(handle, state, max_scratch_size, workspace, stream.ptr)
        workspace_size_d = cutn.workspace_get_memory_size(handle, workspace, cutn.WorksizePref.RECOMMENDED, 
                                                            cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)
        scratch_space = cp.cuda.alloc(workspace_size_d)
        cutn.workspace_set_memory(handle, workspace, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, 
                                    scratch_space.ptr, workspace_size_d)
        cutn.state_compute(handle, state, workspace, self.mps_tensor_ptrs, stream.ptr)


class TensorNetworkFactory:

    # TODO(leofang): replace the utilities here by high-level private APIs

    # This factory CANNOT be reused; once a TN descriptor uses it, it must
    # be discarded.

    def __init__(self, einsum, shapes, dtype, *, order='C'):
        self.einsum = einsum
        inputs, output = einsum.split('->') if "->" in einsum else (einsum, None)
        i_shapes, o_shape = shapes[:-1], shapes[-1]
        inputs = tuple(tuple(_input) for _input in inputs.split(","))
        assert all([len(i) == len(s) for i, s in zip(inputs, i_shapes)])
        assert len(output) == len(o_shape)

        # xp strides in bytes, cutn strides in counts
        itemsize = cp.dtype(dtype).itemsize

        self.input_tensors = [
            testing.shaped_random(s, cp, dtype, seed=i, order=order)
            for i, s in enumerate(i_shapes)]
        self.input_n_modes = [len(i) for i in inputs]
        self.input_extents = i_shapes
        self.input_strides = [[stride // itemsize for stride in arr.strides]
                              for arr in self.input_tensors]
        self.input_modes = [tuple([ord(m) for m in i]) for i in inputs]

        self.output_tensor = cp.empty(o_shape, dtype=dtype, order=order)
        self.output_n_modes = len(o_shape)
        self.output_extent = o_shape
        self.output_stride = [stride // itemsize for stride in self.output_tensor.strides]
        self.output_mode = tuple([ord(m) for m in output])

        self.gradients = None

    def _get_data_type(self, category):
        if 'n_modes' in category:
            return np.int32
        elif 'extent' in category:
            return np.int64
        elif 'stride' in category:
            return np.int64
        elif 'mode' in category:
            return np.int32
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
            elif category in ('input_tensors', 'gradients'):
                # special case for device arrays, return int as void**
                data = np.asarray([d.data.ptr for d in data],
                    dtype=np.intp)
                setattr(self, f'{category}_ptrs', data)  # keep data alive
            # some data are not nested in nature, so we peek at the first
            # element to determine
            elif isinstance(data[0], abc.Sequence):
                # return int as void**
                data = [np.asarray(d, dtype=self._get_data_type(category))
                    for d in data]
                setattr(self, category, data)  # keep data alive
                data = np.asarray([d.ctypes.data for d in data],
                    dtype=np.intp)
                setattr(self, f'{category}_ptrs', data)  # keep data alive
            else:
                # return int as void*
                data = np.asarray(data, dtype=self._get_data_type(category))
                setattr(self, category, data)  # keep data alive
            return data.ctypes.data
        elif return_value == 'seq':
            if len(data) == 0:
                # empty, leave it as is
                pass
            elif category in ('input_tensors', 'gradients'):
                # special case for device arrays
                data = [d.data.ptr for d in data]
                setattr(self, f'{category}_ptrs', data)  # keep data alive
            # some data are not nested in nature, so we peek at the first
            # element to determine
            elif isinstance(data[0], abc.Sequence):
                data = [np.asarray(d, dtype=self._get_data_type(category))
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

    def get_gradient_tensors(self, **kwargs):
        if self.gradients is None:
            # as of 23.06, the gradient tensors' strides follow those of the
            # input tensors
            self.gradients = [cp.empty_like(arr) for arr in self.input_tensors]
        data = self._return_data('gradients', kwargs['data'])
        return data


@testing.parameterize(*testing.product({
    'tn': (
        ('ab,bc->ac', [(2, 3), (3, 2), (2, 2)]),
        ('ab,ba->', [(2, 3), (3, 2), ()]),
        ('abc,bca->', [(2, 3, 4), (3, 4, 2), ()]),
        ('ab,bc,cd->ad', [(2, 3), (3, 1), (1, 5), (2, 5)]),
    ),
    'dtype': (
        np.float32, np.float64, np.complex64, np.complex128
    ),
    # use the same format for both input/output tensors
    'input_form': (
        {'n_modes': 'int', 'extent': 'int', 'stride': 'int',
         'mode': 'int', 'data': 'int'},
        {'n_modes': 'int', 'extent': 'seq', 'stride': 'seq',
         'mode': 'seq', 'data': 'seq'},
        {'n_modes': 'seq', 'extent': 'nested_seq', 'stride': 'nested_seq',
         'mode': 'seq', 'data': 'seq'},
    ),
    'order': ('C', 'F'),
    # mainly for gradient tests
    'qual': (None, True),
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

        tensor_dscr = cutn.get_output_tensor_descriptor(handle, dscr)
        num_modes, modes, extents, strides = cutn.get_tensor_details(
            handle, tensor_dscr)

        assert num_modes == self.tn.output_n_modes
        assert (modes == np.asarray(self.tn.output_mode, dtype=np.int32)).all()
        assert (extents == np.asarray(self.tn.output_extent, dtype=np.int64)).all()
        assert (strides == np.asarray(self.tn.output_stride, dtype=np.int64)).all()

        cutn.destroy_tensor_descriptor(tensor_dscr)


class TestOptimizerInfo(TestTensorNetworkBase):

    def _get_path(self, handle, info):
        raise NotImplementedError

    def _set_path(self, handle, info, path):
        attr = cutn.ContractionOptimizerInfoAttribute.PATH
        dtype = cutn.contraction_optimizer_info_get_attribute_dtype(attr)
        if not isinstance(path, np.ndarray):
            path = np.ascontiguousarray(path, dtype=np.int32)
        path_obj = np.asarray((path.shape[0], path.ctypes.data), dtype=dtype)
        self._set_scalar_attr(handle, info, attr, path_obj)

    def _get_scalar_attr(self, handle, info, attr):
        dtype = cutn.contraction_optimizer_info_get_attribute_dtype(attr)
        data = np.empty((1,), dtype=dtype)
        cutn.contraction_optimizer_info_get_attribute(
            handle, info, attr,
            data.ctypes.data, data.dtype.itemsize)
        return data

    def _set_scalar_attr(self, handle, info, attr, data):
        dtype = cutn.contraction_optimizer_info_get_attribute_dtype(attr)
        if not isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data, dtype=dtype)
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
        path, _ = np.einsum_path(
            tn.einsum,
            *[arr for arr in map(lambda a: np.broadcast_to(0, a.shape),
                                 tn.input_tensors)])

        # set the path in info (a few other attributes would be computed too)
        # and then serialize it
        self._set_path(handle, info, path[1:])
        buf_size = cutn.contraction_optimizer_info_get_packed_size(
            handle, info)
        buf_data = np.empty((buf_size,), dtype=np.int8)
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
                    cutn.ContractionOptimizerConfigAttribute.COST_FUNCTION_OBJECTIVE,
                    cutn.ContractionOptimizerConfigAttribute.SMART_OPTION):
            factor = np.asarray([1], dtype=dtype)
        else:
            factor = np.asarray([30], dtype=dtype)
        cutn.contraction_optimizer_config_set_attribute(
            handle, config, attr,
            factor.ctypes.data, factor.dtype.itemsize)
        # do a round-trip test as a sanity check
        factor2 = np.zeros_like(factor)
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
        factor = np.asarray([2], dtype=dtype)
        cutn.contraction_autotune_preference_set_attribute(
            handle, pref, attr,
            factor.ctypes.data, factor.dtype.itemsize)
        # do a round-trip test as a sanity check
        factor2 = np.zeros_like(factor)
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
    'contract', ("slice_group", "gradient")
)
@pytest.mark.parametrize(
    'stream', (cp.cuda.Stream.null, get_stream_for_backend(cp))
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
    def test_contraction_gradient_workflow(
            self, mempool, workspace_pref, autotune, contract, stream):
        if (isinstance(mempool, str) and mempool.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        # unpack
        handle, dscr, info, config, pref = self.handle, self.dscr, self.info, self.config, self.autotune
        workspace = self.workspace
        tn, input_form = self.tn, self.input_form

        # make sure inputs are ready
        # TODO: use stream_wait_event to establish stream order is better
        cp.cuda.Device().synchronize()

        if mempool:
            mr = MemoryResourceFactory(mempool)
            handler = mr.get_dev_mem_handler()
            cutn.set_device_mem_handler(handle, handler)

        workspace_hint = 32*1024**2  # large enough for our test cases
        # we have to run this API in any case in order to create a path
        cutn.contraction_optimize(
            handle, dscr, config, workspace_hint, info)

        # for simplicity, compute grads for all tensors
        if contract == "gradient":
            if self.qual is None:
                # set up the grad flag via TN attributes instead of input qualifiers
                tensor_id_range = np.arange(len(tn.input_tensors), dtype=np.int32)
                net_attr_dtype = cutn.network_get_attribute_dtype(
                    cutn.NetworkAttribute.INPUT_TENSORS_REQUIRE_GRAD)
                tensor_ids = np.zeros(1, dtype=net_attr_dtype)
                tensor_ids['num_tensors'] = tensor_id_range.size
                tensor_ids['data'] = tensor_id_range.ctypes.data
                cutn.network_set_attribute(
                    handle, dscr, cutn.NetworkAttribute.INPUT_TENSORS_REQUIRE_GRAD,
                    tensor_ids.ctypes.data, tensor_ids.dtype.itemsize)
                # round-trip
                tensor_id_range_back = np.zeros_like(tensor_id_range)
                tensor_ids['num_tensors'] = tensor_id_range_back.size
                tensor_ids['data'] = tensor_id_range_back.ctypes.data
                cutn.network_get_attribute(
                    handle, dscr, cutn.NetworkAttribute.INPUT_TENSORS_REQUIRE_GRAD,
                    tensor_ids.ctypes.data, tensor_ids.dtype.itemsize)
                assert (tensor_id_range_back == tensor_id_range).all()

            output_grads = cp.ones_like(tn.output_tensor)

        # manage workspace
        placeholder = []
        if mempool is None:
            cutn.workspace_compute_contraction_sizes(handle, dscr, info, workspace)
            for kind in cutn.WorkspaceKind:  # for both scratch & cache
                required_size = cutn.workspace_get_memory_size(
                    handle, workspace,
                    getattr(cutn.WorksizePref, f"{workspace_pref.upper()}"),
                    cutn.Memspace.DEVICE,  # TODO: parametrize memspace?
                    kind)
                if contract != "gradient":
                    cutn.workspace_compute_contraction_sizes(handle, dscr, info, workspace)
                    required_size_deprecated = cutn.workspace_get_memory_size(
                        handle, workspace,
                        getattr(cutn.WorksizePref, f"{workspace_pref.upper()}"),
                        cutn.Memspace.DEVICE,  # TODO: parametrize memspace?
                        kind)
                    assert required_size == required_size_deprecated
                if workspace_pref == 'min':
                    # This only holds when workspace_pref set to min
                    assert required_size <= workspace_hint, \
                        f"wrong assumption on the workspace size " \
                        f"(given: {workspace_hint}, needed: {required_size})"
                if required_size > 0:
                    workspace_ptr = cp.cuda.alloc(required_size)
                    cutn.workspace_set_memory(
                        handle, workspace,
                        cutn.Memspace.DEVICE,
                        kind,
                        workspace_ptr.ptr, required_size)
                    placeholder.append(workspace_ptr)  # keep it alive
                    # round-trip check
                    assert ((workspace_ptr.ptr, required_size) ==
                        cutn.workspace_get_memory(handle, workspace,
                                                  cutn.Memspace.DEVICE, kind))
        else:
            for kind in cutn.WorkspaceKind:
                cutn.workspace_set_memory(
                    handle, workspace,
                    cutn.Memspace.DEVICE,
                    kind,
                    0, -1)  # TODO: check custom workspace size?

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
            if contract in ("slice_group", "gradient"):
                accumulate = 0
                cutn.contract_slices(
                    handle, plan,
                    tn.get_input_tensors(**input_form),
                    tn.get_output_tensor(),
                    accumulate,
                    workspace, self.slice_group, stream.ptr)
                if contract == "gradient":
                    cutn.compute_gradients_backward(
                        handle, plan,
                        tn.get_input_tensors(**input_form),
                        output_grads.data.ptr,
                        tn.get_gradient_tensors(**input_form),
                        accumulate, workspace, stream.ptr)
            stream.synchronize()
        finally:
            if plan is not None:
                cutn.destroy_contraction_plan(plan)

        if contract == "gradient" and torch:

            # TODO: The second condition should be removed once PyTorch has support for Blackwell kernels
            if not torch.cuda.is_available() or int(cp.cuda.Device().compute_capability) >= 100:
                # copy data back to CPU
                dev = "cpu"
                func = cp.asnumpy
            else:
                # zero-copy from CuPy to PyTorch!
                dev = "cuda"
                func = (lambda x: x)  # no op

            inputs = [torch.as_tensor(func(t), device=dev)
                      for t in tn.input_tensors]
            output_grads = torch.as_tensor(func(output_grads), device=dev)
            for t in inputs:
                t.requires_grad_(True)
                assert t.grad is None

            # repeat the same calculation with PyTorch so that it fills up the
            # gradients for us to do verification
            out = torch.einsum(tn.einsum, *inputs)
            out.backward(output_grads)

            # compare gradients
            for grad_cutn, in_torch in zip(tn.gradients, inputs):
                grad_torch = in_torch.grad
                # zero-copy if on GPU
                assert cp.allclose(grad_cutn, cp.asarray(grad_torch))


@testing.parameterize(*testing.product({
    'dtype': (np.float32, np.float64, np.complex64, np.complex128),
    'extents': ('int', 'seq'),
    'strides': ('int', 'seq'),
    'order': ('C', 'F'),
    'state_purity': (cutn.StatePurity.PURE,),
    'num_qubits': (4,),
    'mps': (False, 2),
    'stream': (cp.cuda.Stream.null, get_stream_for_backend(cp)),
}))
class TestStateBase:
    pass
        
        
class TestStateAPIs(TestStateBase):

    def _configure_prepare(self, task_type, handle, property_object, workspace, stream):
        attr_type = getattr(cutn, f"{task_type.capitalize()}Attribute").CONFIG_NUM_HYPER_SAMPLES
        attr_dtype = getattr(cutn, f"{task_type}_get_attribute_dtype")
        configure_func = getattr(cutn, f"{task_type}_configure")
        prepare_func = getattr(cutn, f"{task_type}_prepare")

        # Configure num_hyper_samples
        num_hyper_samples = np.array(8, dtype=attr_dtype(attr_type))
        configure_func(handle, property_object, attr_type,
                      num_hyper_samples.ctypes.data, num_hyper_samples.dtype.itemsize)

        # Prepare workspace
        free_mem = cp.cuda.Device().mem_info[0]
        max_scratch_size = free_mem // 4
        prepare_func(handle, property_object, max_scratch_size, workspace, stream.ptr)

        # Set up scratch space
        workspace_size_d = cutn.workspace_get_memory_size(handle, workspace, cutn.WorksizePref.RECOMMENDED,
                                                        cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)
        scratch_space = cp.cuda.alloc(workspace_size_d)
        cutn.workspace_set_memory(handle, workspace, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH,
                                scratch_space.ptr, workspace_size_d)
        return scratch_space

    @manage_resource('handle')
    @manage_resource('workspace')
    @manage_resource('state')
    @manage_resource('accessor')
    def test_accessor_bindings(self):
        fixed_values = ()
        state_norm = np.empty(1, dtype=self.dtype)
        scratch_space = self._configure_prepare('accessor', self.handle, self.accessor, self.workspace, self.stream)
        cutn.accessor_compute(self.handle, self.accessor, fixed_values, self.workspace,
                              self.amplitudes.data.ptr, state_norm.ctypes.data, self.stream.ptr)
        self.stream.synchronize()
        if self.mps == False: # for mps simulation and max_extent=2, the state norm is 1 if the state is sparsed
            assert np.isclose(state_norm[0], 1.0, atol=1e-6)
        
    @manage_resource('handle')
    @manage_resource('workspace')
    @manage_resource('state')
    @manage_resource('sampler')
    def test_sampling_bindings(self):
        num_samples = 2000
        samples = np.empty((self.num_qubits, num_samples), dtype=np.int64, order='F') # samples are stored in F order with shape (num_qubits, num_samples)
        scratch_space = self._configure_prepare('sampler', self.handle, self.sampler, self.workspace, self.stream)
        rng_dtype = cutn.sampler_get_attribute_dtype(cutn.SamplerAttribute.CONFIG_DETERMINISTIC)
        rng = np.asarray(13, dtype=rng_dtype)
        cutn.sampler_configure(self.handle, self.sampler, cutn.SamplerAttribute.CONFIG_DETERMINISTIC, 
                               rng.ctypes.data, rng.dtype.itemsize)
        cutn.sampler_sample(self.handle, self.sampler, num_samples, self.workspace, samples.ctypes.data, self.stream.ptr)
        self.stream.synchronize()
        
    @manage_resource('handle')
    @manage_resource('workspace')
    @manage_resource('state')
    @manage_resource('hamiltonian')
    @manage_resource('expectation')
    def test_expectation_bindings(self):
        scratch_space = self._configure_prepare('expectation', self.handle, self.expectation, self.workspace, self.stream)
        expectation_value = np.empty(1, dtype=self.dtype)
        state_norm = np.empty(1, dtype=self.dtype)
        cutn.expectation_compute(self.handle, self.expectation, self.workspace, 
                                 expectation_value.ctypes.data, state_norm.ctypes.data, self.stream.ptr)
        self.stream.synchronize()
        if self.mps == False:
            assert np.isclose(state_norm[0], 1.0, atol=1e-6)
        
    @manage_resource('handle')
    @manage_resource('workspace')
    @manage_resource('state')
    @manage_resource('marginal')
    def test_marginal_bindings(self): 
        scratch_space = self._configure_prepare('marginal', self.handle, self.marginal, self.workspace, self.stream)
        cutn.marginal_compute(self.handle, self.marginal, 0, self.workspace, self.rdm.data.ptr, self.stream.ptr)
        self.stream.synchronize()
        
  
@pytest.mark.parametrize(
    'source', ('int', 'seq', 'range')
)
class TestSliceGroup:

    @manage_resource('handle')
    def test_slice_group(self, source):
        # we don't do a simple round-trip test here because there are two
        # flavors of constructors
        if source == "int":
            ids = np.arange(10, dtype=np.int64)
            slice_group = cutn.create_slice_group_from_ids(
                self.handle, ids.ctypes.data, ids.size)
        elif source == "seq":
            ids = np.arange(10, dtype=np.int64)
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

    # QR/SVD Example: "ab->ax,xb"
    # Gate Example: "ijk,klm,jkpq->->ipk,kqm" 
    # This factory CANNOT be reused; once a tensor descriptor uses it, it must
    # be discarded.

    def __init__(self, subscript, shapes, dtype, max_extent=None, order='C'):
        self.subscript = subscript

        if len(shapes) not in [1, 3]:
            raise NotImplementedError
        
        modes_in, left_modes, right_modes, shared_mode = approxTN_utils.parse_split_expression(subscript)
        modes_in = modes_in.split(",")
        size_dict = dict()
        for modes, shape in zip(modes_in, shapes):
            for mode, extent in zip(modes, shape):
                if mode in size_dict:
                    assert size_dict[mode] == extent
                else:
                    size_dict[mode] = extent
        _, left_modes_out, right_modes_out, shared_mode_out, _, mid_extent = approxTN_utils.parse_modes_extents(size_dict, subscript)
        # Note: we need to parse options as this is where max_extent is specified
        self.shared_mode_idx_left = left_modes_out.find(shared_mode_out)
        self.shared_mode_idx_right = right_modes_out.find(shared_mode_out)
        if max_extent is None:
            # no truncation on extent
            self.mid_extent = mid_extent
        else:
            assert max_extent > 0
            self.mid_extent = min(mid_extent, max_extent)
        
        self.order = order
        self.tensor_names = [f"input_{i}" for i in range(len(shapes))] + ["left", "right"] # note s needs to be explictly managed in the tester function

        # xp strides in bytes, cutn strides in counts
        dtype = cp.dtype(dtype)
        itemsize = dtype.itemsize

        for i, (name, modes) in enumerate(zip(self.tensor_names, modes_in + [left_modes_out, right_modes_out])):
            if name.startswith('input'):
                shape = [size_dict[mode] for mode in modes]
                arr = testing.shaped_random(shape, cp, dtype, seed=i, order=self.order)
            else:
                shape = [self.mid_extent if mode == shared_mode_out else size_dict[mode] for mode in modes]
                arr = cp.empty(shape, dtype=dtype, order=self.order)
            setattr(self, f'{name}_tensor', arr)
            setattr(self, f'{name}_n_modes', len(arr.shape))
            setattr(self, f'{name}_extent', arr.shape)
            setattr(self, f'{name}_stride', [stride // itemsize for stride in arr.strides])
            setattr(self, f'{name}_mode', tuple([ord(m) for m in modes]))

    def _get_data_type(self, category):
        if 'n_modes' in category:
            return np.int32
        elif 'extent' in category:
            return np.int64
        elif 'stride' in category:
            return np.int64
        elif 'mode' in category:
            return np.int32
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
                data = np.asarray(data, dtype=self._get_data_type(category))
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
    
    def get_operands(self, include_inputs=True, include_outputs=True):
        operands = []
        for name in self.tensor_names:
            if include_inputs and name.startswith('input'):
                operands.append(getattr(self, f'{name}_tensor'))
            elif include_outputs and not name.startswith('input'):
                operands.append(getattr(self, f'{name}_tensor'))
        return operands


@testing.parameterize(*testing.product({
    'tn': tensor_decomp_expressions,
    'dtype': (
        np.float32, np.float64, np.complex64, np.complex128
    ),
    'tensor_form': (
        {'extent': 'int', 'stride': 'int', 'mode': 'int'},
        {'extent': 'seq', 'stride': 'seq', 'mode': 'seq'},
    ),
    'order': ('C', 'F'),
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
        dtype = cp.dtype(self.dtype)

        # prepare workspace
        cutn.workspace_compute_qr_sizes(
            handle, tensor_in, tensor_q, tensor_r, workspace)
        # for now host workspace is always 0, so just query device one
        # also, it doesn't matter which one (min/recommended/max) is queried
        required_size = cutn.workspace_get_memory_size(
            handle, workspace, cutn.WorksizePref.MIN,
            cutn.Memspace.DEVICE,  # TODO: parametrize memspace?
            cutn.WorkspaceKind.SCRATCH)
        if required_size > 0:
            workspace_ptr = cp.cuda.alloc(required_size)
            cutn.workspace_set_memory(
                handle, workspace, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH,
                workspace_ptr.ptr, required_size)
            # round-trip check
            assert (workspace_ptr.ptr, required_size) == cutn.workspace_get_memory(
                handle, workspace, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)

        # perform QR
        stream = cp.cuda.get_current_stream().ptr  # TODO
        cutn.tensor_qr(
            handle, tensor_in, tn.get_tensor_ptr('input_0'),
            tensor_q, tn.get_tensor_ptr('left'),
            tensor_r, tn.get_tensor_ptr('right'),
            workspace, stream)

        # for QR, no need to compute the reference for correctness check
        operands = tn.get_operands(include_inputs=True, include_outputs=True) # input, q, r
        assert approxTN_utils.verify_split_QR(tn.subscript, *operands, None, None)


@testing.parameterize(*testing.product({
    'tn': tensor_decomp_expressions,
    'dtype': (
        np.float32, np.float64, np.complex64, np.complex128
    ),
    'tensor_form': (
        {'extent': 'int', 'stride': 'int', 'mode': 'int'},
        {'extent': 'seq', 'stride': 'seq', 'mode': 'seq'},
    ),
    'options': (
        {}, # standard exact svd
        {'max_extent': 4, 'normalization':'L1', 'partition':'U', 'algorithm': 'gesvdr', 'gesvdr_niters': 100}, # fix extent truncation
        {'abs_cutoff': 0.1, 'discarded_weight_cutoff': 0.05, 'normalization': 'L2'}, # discarded weight truncation
        {'abs_cutoff': 0.1, 'rel_cutoff': 0.1, 'algorithm': 'gesvdj', 'gesvdj_tol':1e-14, 'gesvdj_max_sweeps': 80}, # value based truncation
        {'abs_cutoff': 0.1, 'normalization':'L2', 'partition':'V', 'algorithm': 'gesvdj'}, # absolute value based truncation
        {'normalization':'LInf', 'partition':'UV', 'algorithm': 'gesvdp'}, # exact gesvdp
        {'max_extent': 4, 'abs_cutoff': 0.1, 'rel_cutoff': 0.1, 'normalization':'L1', 'partition':'UV'}, # compound truncation
    ),
    'order': ('C', 'F'),
}))
class TestTensorSVD:

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
        dtype = cp.dtype(self.dtype)

        # switch to default gesvdj_tol for single precision operand
        algorithm = self.options.get('algorithm', None)
        if algorithm == 'gesvdj' and self.dtype in [np.float32, np.complex64]:
            self.options.pop('gesvdj_tol', None)
            
        # parse svdConfig
        svd_method = check_or_create_options(tensor.SVDMethod, self.options, "SVDMethod")
        parse_svd_config(handle, svd_config, svd_method, logger=None)

        # prepare workspace
        cutn.workspace_compute_svd_sizes(
            handle, tensor_in, tensor_u, tensor_v, svd_config, workspace)
        # for now host workspace is always 0, so just query device one
        # also, it doesn't matter which one (min/recommended/max) is queried
        workspaces = {}
        allocators = {cutn.Memspace.DEVICE: cp.cuda.alloc,
                      cutn.Memspace.HOST: lambda nbytes: np.empty(nbytes, dtype=np.int8)}
        for mem_space, allocator in allocators.items():
            required_size = cutn.workspace_get_memory_size(
                handle, workspace, cutn.WorksizePref.MIN,
                mem_space,  
                cutn.WorkspaceKind.SCRATCH)
            if required_size > 0:
                workspaces[mem_space] = workspace_ptr = allocator(required_size) # keep alive
                if mem_space == cutn.Memspace.DEVICE:
                    workspace_ptr_address = workspace_ptr.ptr
                else:
                    workspace_ptr_address = workspace_ptr.ctypes.data
                cutn.workspace_set_memory(
                    handle, workspace, mem_space, cutn.WorkspaceKind.SCRATCH,
                    workspace_ptr_address, required_size)
                # round-trip check
                assert (workspace_ptr_address, required_size) == cutn.workspace_get_memory(
                    handle, workspace, mem_space, cutn.WorkspaceKind.SCRATCH)
        
        partition = self.options.get("partition", None)
        if partition is None:
            s = cp.empty(tn.mid_extent, dtype=dtype.char.lower())
            s_ptr = s.data.ptr
        else:
            s = None
            s_ptr = 0
        
        # perform SVD
        stream = cp.cuda.get_current_stream().ptr  # TODO
        cutn.tensor_svd(
            handle, tensor_in, tn.get_tensor_ptr('input_0'),
            tensor_u, tn.get_tensor_ptr('left'),
            s_ptr,
            tensor_v, tn.get_tensor_ptr('right'),
            svd_config, svd_info, workspace, stream)
        
        # get runtime truncation details
        info = get_svd_info_dict(handle, svd_info)
        
        T, u, v = tn.get_operands(include_inputs=True, include_outputs=True)

        # update the container if reduced extent if found to be different from specified mid extent
        extent_U_out, strides_U_out = cutn.get_tensor_details(handle, tensor_u)[2:]
        extent_V_out, strides_V_out = cutn.get_tensor_details(handle, tensor_v)[2:]
        reduced_extent = info['reduced_extent']
        assert extent_U_out[tn.shared_mode_idx_left] == reduced_extent
        assert extent_V_out[tn.shared_mode_idx_right] == reduced_extent
        if tuple(extent_U_out) != u.shape:
            strides_U_out = [i * u.itemsize for i in strides_U_out]
            strides_V_out = [i * v.itemsize for i in strides_V_out]
            tn.left_tensor = u = cp.ndarray(extent_U_out, dtype=u.dtype, memptr=u.data, strides=strides_U_out)
            if s is not None:
                s = cp.ndarray(reduced_extent, dtype=s.dtype, memptr=s.data, order='F')
            tn.right_tensor = v = cp.ndarray(extent_V_out, dtype=v.dtype, memptr=v.data, strides=strides_V_out)
        
        try:
            u_ref, s_ref, v_ref, info_ref = approxTN_utils.tensor_decompose(
                tn.subscript, T, 
                method='svd', return_info=True, 
                **self.options)
        except approxTN_utils.SingularValueDegeneracyError:
            pytest.skip("Test skipped due to singular value degeneracy issue")            

        assert approxTN_utils.verify_split_SVD(
            tn.subscript, T, 
            tn.left_tensor, s, tn.right_tensor,
            u_ref, s_ref, v_ref,
            info=info, info_ref=info_ref,
            **self.options) 


@testing.parameterize(*testing.product({
    'tn': gate_decomp_expressions,
    'dtype': (
        np.float32, np.float64, np.complex64, np.complex128
    ),
    'tensor_form': (
        {'extent': 'int', 'stride': 'int', 'mode': 'int'},
        {'extent': 'seq', 'stride': 'seq', 'mode': 'seq'},
    ),
    'algo': (
        "direct", "reduced"
    ),
    'options': (
        {}, # standard exact svd
        {'max_extent': 4, 'normalization':'L1', 'partition':'U', 'algorithm': 'gesvdr', 'gesvdr_niters': 100}, # fix extent truncation
        {'abs_cutoff': 0.1, 'discarded_weight_cutoff': 0.05, 'normalization': 'L2'}, # discarded weight truncation
        {'abs_cutoff': 0.1, 'rel_cutoff': 0.1, 'algorithm': 'gesvdj', 'gesvdj_tol':1e-14, 'gesvdj_max_sweeps': 80}, # value based truncation
        {'abs_cutoff': 0.1, 'normalization':'L2', 'partition':'V', 'algorithm': 'gesvdj'}, # absolute value based truncation
        {'normalization':'LInf', 'partition':'UV', 'algorithm': 'gesvdp'}, # exact gesvdp
        {'max_extent': 4, 'abs_cutoff': 0.1, 'rel_cutoff': 0.1, 'normalization':'L1', 'partition':'UV'}, # compound truncation
    ),
    'order': ('C', 'F'),
}))
class TestTensorGate:
    
    GATE_ALGO_MAP = {"direct": cutn.GateSplitAlgo.DIRECT,
                     "reduced": cutn.GateSplitAlgo.REDUCED}
    
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
        algo = self.algo
        gate_algorithm = self.GATE_ALGO_MAP[algo]
        svd_config, svd_info = self.svd_config, self.svd_info

        # switch to default gesvdj_tol for single precision operand
        algorithm = self.options.get('algorithm', None)
        if algorithm == 'gesvdj' and self.dtype in [np.float32, np.complex64]:
            self.options.pop('gesvdj_tol', None)
        
        # parse svdConfig
        svd_method = check_or_create_options(tensor.SVDMethod, self.options, "SVDMethod")
        parse_svd_config(handle, svd_config, svd_method, logger=None)

        dtype = cp.dtype(self.dtype)
        compute_type = dtype_to_compute_type[self.dtype]
        # prepare workspace
        cutn.workspace_compute_gate_split_sizes(handle, 
            tensor_in_a, tensor_in_b, tensor_in_g, tensor_u, tensor_v, 
            gate_algorithm, svd_config, compute_type, workspace)
        workspaces = {}
        allocators = {cutn.Memspace.DEVICE: cp.cuda.alloc,
                      cutn.Memspace.HOST: lambda nbytes: np.empty(nbytes, dtype=np.int8)}
        for mem_space, allocator in allocators.items():
            required_size = cutn.workspace_get_memory_size(
                handle, workspace, cutn.WorksizePref.MIN,
                mem_space,  
                cutn.WorkspaceKind.SCRATCH)
            if required_size > 0:
                workspaces[mem_space] = workspace_ptr = allocator(required_size) # keep alive
                if mem_space == cutn.Memspace.DEVICE:
                    workspace_ptr_address = workspace_ptr.ptr
                else:
                    workspace_ptr_address = workspace_ptr.ctypes.data
                cutn.workspace_set_memory(
                    handle, workspace, mem_space, cutn.WorkspaceKind.SCRATCH,
                    workspace_ptr_address, required_size)
                # round-trip check
                assert (workspace_ptr_address, required_size) == cutn.workspace_get_memory(
                    handle, workspace, mem_space, cutn.WorkspaceKind.SCRATCH)

        partition = self.options.get("partition", None)
        if partition is None:
            s = cp.empty(tn.mid_extent, dtype=dtype.char.lower())
            s_ptr = s.data.ptr
        else:
            s = None
            s_ptr = 0
        
        # perform gate split
        stream = cp.cuda.get_current_stream().ptr  # TODO
        cutn.gate_split(handle, tensor_in_a, tn.get_tensor_ptr('input_0'),
            tensor_in_b, tn.get_tensor_ptr('input_1'),
            tensor_in_g, tn.get_tensor_ptr('input_2'),
            tensor_u, tn.get_tensor_ptr('left'), s_ptr, 
            tensor_v, tn.get_tensor_ptr('right'),
            gate_algorithm, svd_config, compute_type, 
            svd_info, workspace, stream)
        
        # get runtime truncation information 
        info = get_svd_info_dict(handle, svd_info)

        arr_a, arr_b, arr_gate, u, v = tn.get_operands(include_inputs=True, include_outputs=True)

        # update the container if reduced extent if found to be different from specified mid extent
        extent_U_out, strides_U_out = cutn.get_tensor_details(handle, tensor_u)[2:]
        extent_V_out, strides_V_out = cutn.get_tensor_details(handle, tensor_v)[2:]
        reduced_extent = info['reduced_extent']
        assert extent_U_out[tn.shared_mode_idx_left] == reduced_extent
        assert extent_V_out[tn.shared_mode_idx_right] == reduced_extent
        if tuple(extent_U_out) != u.shape:
            strides_U_out = [i * u.itemsize for i in strides_U_out]
            strides_V_out = [i * v.itemsize for i in strides_V_out]
            tn.left_tensor = u = cp.ndarray(extent_U_out, dtype=u.dtype, memptr=u.data, strides=strides_U_out)
            if s is not None:
                s = cp.ndarray(reduced_extent, dtype=s.dtype, memptr=s.data, order='F')
            tn.right_tensor = v = cp.ndarray(extent_V_out, dtype=v.dtype, memptr=v.data, strides=strides_V_out)
        
        try:
            u_ref, s_ref, v_ref, info_ref = approxTN_utils.gate_decompose(
                tn.subscript, 
                arr_a, 
                arr_b, 
                arr_gate, 
                gate_algo=algo, 
                return_info=True, 
                **self.options)
        except approxTN_utils.SingularValueDegeneracyError:
            pytest.skip("Test skipped due to singular value degeneracy issue")
        
        assert approxTN_utils.verify_split_SVD(
            tn.subscript, None, 
            u, s, v, 
            u_ref, s_ref, v_ref,
            info=info, info_ref=info_ref, 
            **self.options)


class TestTensorSVDConfig:

    @manage_resource('handle')
    @manage_resource('svd_config')
    def test_tensor_svd_config_create_destroy(self):
        # simple round-trip test
        pass

    @pytest.mark.parametrize(
        'attr', [val for val in cutn.TensorSVDConfigAttribute if val != cutn.TensorSVDConfigAttribute.ALGO_PARAMS]
    )
    @manage_resource('handle')
    @manage_resource('svd_config')
    def test_tensor_svd_config_get_set_attribute(self, attr):
        handle, svd_config = self.handle, self.svd_config
        dtype = cutn.tensor_svd_config_get_attribute_dtype(attr)
        # Hack: assume this is a valid value for all attrs
        factor = np.asarray([0.8], dtype=dtype)
        cutn.tensor_svd_config_set_attribute(
            handle, svd_config, attr,
            factor.ctypes.data, factor.dtype.itemsize)
        # do a round-trip test as a sanity check
        factor2 = np.zeros_like(factor)
        cutn.tensor_svd_config_get_attribute(
            handle, svd_config, attr,
            factor2.ctypes.data, factor2.dtype.itemsize)
        assert factor == factor2
    
    @pytest.mark.parametrize(
        'svd_algorithm', (cutn.TensorSVDAlgo.GESVDJ, cutn.TensorSVDAlgo.GESVDR)
    )
    @manage_resource('handle')
    @manage_resource('svd_config')
    def test_tensor_svd_config_get_set_params_attribute(self, svd_algorithm):
        handle, svd_config = self.handle, self.svd_config
        # set ALGO first
        algo_dtype = cutn.tensor_svd_config_get_attribute_dtype(cutn.TensorSVDConfigAttribute.ALGO)
        algo = np.asarray(svd_algorithm, dtype=algo_dtype)
        cutn.tensor_svd_config_set_attribute(
            handle, svd_config, cutn.TensorSVDConfigAttribute.ALGO,
            algo.ctypes.data, algo.dtype.itemsize)
        
        algo_params_dtype = cutn.tensor_svd_algo_params_get_dtype(svd_algorithm)
        # Hack: assume this is a valid value for all SVD parameters
        factor = np.asarray([1.8], dtype=algo_params_dtype) # 0 may trigger default behavior, eg, gesvdr_niters set to 0 means default (10)
        cutn.tensor_svd_config_set_attribute(
            handle, svd_config, cutn.TensorSVDConfigAttribute.ALGO_PARAMS,
            factor.ctypes.data, factor.dtype.itemsize)
        # do a round-trip test as a sanity check
        factor2 = np.zeros_like(factor)
        cutn.tensor_svd_config_get_attribute(
            handle, svd_config, cutn.TensorSVDConfigAttribute.ALGO_PARAMS,
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
            handle, *get_mpi_comm_pointer(comm))
        assert comm.Get_size() == cutn.distributed_get_num_ranks(handle)
        assert comm.Get_rank() == cutn.distributed_get_proc_rank(handle)
        cutn.distributed_synchronize(handle)
        cutn.distributed_reset_configuration(handle, 0, 0)  # reset
        # no need to free the comm, for world/self mpi4py does it for us...


class TestLogger(LoggerTestBase):

    mod = cutn
    prefix = "cutensornet"


class TestMisc:

    @pytest.mark.parametrize("cutn_compute_type", cutn.ComputeType)
    def test_compute_type(self, cutn_compute_type):
        # check if all compute types under cutn.ComputeType are included in cuquantum.ComputeType
        cuqnt_compute_type = ComputeType(cutn_compute_type)


class TestBindingDeprecation(BindingsDeprecationTestBase):

    lib_name = "cutensornet"
