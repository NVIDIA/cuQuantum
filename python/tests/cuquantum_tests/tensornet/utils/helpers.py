# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import hashlib
import importlib
from itertools import count, cycle


import pytest
try:
    # cuda.core >= 0.5.0
    from cuda.core import system
    NUM_DEVICES = system.get_num_devices()
except ImportError:
    # cuda.core < 0.5.0
    from cuda.core.experimental import system
    NUM_DEVICES = system.num_devices

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
except ImportError:
    torch = None

from cuquantum.bindings import cutensornet as cutn
from cuquantum.tensornet import OptimizerOptions, contract
from cuquantum.tensornet import tensor
from cuquantum.tensornet.experimental import NetworkOperator
from cuquantum.tensornet.experimental._internal.network_state_utils import _get_asarray_function
from cuquantum.tensornet._internal.circuit_converter_utils import EINSUM_SYMBOLS_BASE
from cuquantum.tensornet._internal.decomposition_utils import DECOMPOSITION_DTYPE_NAMES
from cuquantum.tensornet._internal.einsum_parser import infer_output_mode_labels
from cuquantum.tensornet._internal.helpers import _get_backend_asarray_func
from nvmath.internal import tensor_wrapper, memory

from nvmath.internal.utils import infer_object_package, get_or_create_stream
from nvmath.internal.tensor_wrapper import wrap_operand, TensorHolder

from .data import dtype_names, ARRAY_BACKENDS, BACKEND_MEMSPACE


machine_epsilon_values = [np.finfo(dtype).eps for dtype in dtype_names]

rtol_mapper = dict(zip(
    dtype_names,
    [np.sqrt(m_eps) for m_eps in machine_epsilon_values]
))

atol_mapper = dict(zip(
    dtype_names,
    [10 * m_eps for m_eps in machine_epsilon_values]
))

def get_contraction_tolerance(dtype):
    tolerance = {'atol': atol_mapper[dtype],
                 'rtol': rtol_mapper[dtype]}
    return tolerance

def set_path_to_optimizer_options(optimizer_opts, path):
    if optimizer_opts is None:
        optimizer_opts = {"path": path}
    elif isinstance(optimizer_opts, dict):
        optimizer_opts["path"] = path
    else:
        assert isinstance(optimizer_opts, OptimizerOptions)
        optimizer_opts.path = path
    return optimizer_opts


def compute_and_normalize_numpy_path(data, num_operands):
    try:
        # this can fail if the TN is too large (ex: containing unicode)
        path, _ = np.einsum_path(*data, optimize=True)
    except:
        raise NotImplementedError
    path = path[1:]

    # now we need to normalize the NumPy path, because NumPy supports
    # contracting a group of tensors at once whereas we only support
    # pairwise contraction
    num_operands -= 1
    norm_path = []
    for indices in path:
        assert all(idx >= 0 for idx in indices)
        if len(indices) >= 2:
            indices = sorted(indices, reverse=True)
            norm_path.append((indices[0], indices[1]))
            num_operands -= 1
            for idx in indices[2:]:
                # keep contracting with the latest intermediate
                norm_path.append((num_operands, idx))
                num_operands -= 1
        else:
            # single TN reduction is supported by NumPy, but we can't handle
            # that, just raise to avoid testing against NumPy path
            assert len(indices) > 0
            raise NotImplementedError

    return norm_path


def convert_linear_to_ssa(path):
    n_inputs = len(path)+1
    remaining = [*range(n_inputs)]
    ssa_path = []
    counter = n_inputs

    for first, second in path:
        idx1 = remaining[first]
        idx2 = remaining[second]
        ssa_path.append((idx1, idx2))
        remaining.remove(idx1)
        remaining.remove(idx2)
        remaining.append(counter)
        counter += 1

    return ssa_path


def check_ellipsis(modes):
   # find ellipsis, record the position, remove it, and modify the modes
   if isinstance(modes, str):
       ellipsis = modes.find("...")
       if ellipsis >= 0:
           modes = modes.replace("...", "")
   else:
       try:
           ellipsis = modes.index(Ellipsis)
       except ValueError:
           ellipsis = -1
       if ellipsis >= 0:
           modes = modes[:ellipsis] + modes[ellipsis+1:]
   return ellipsis, modes


def check_intermediate_modes(
        intermediate_modes, input_modes, output_modes, path):

    # remove ellipsis, if any, since it's singleton
    input_modes = list(map(
        lambda modes: (lambda modes: check_ellipsis(modes))(modes)[1],
        input_modes
    ))
    _, output_modes = check_ellipsis(output_modes)
    # peek at the very first element
    if (isinstance(intermediate_modes[0], tuple)
            and isinstance(intermediate_modes[0][0], str)):
        # this is our internal mode label for ellipsis
        custom_label = re.compile(r'\b__\d+__\b')
        intermediate_modes = list(map(
            lambda modes: list(filter(lambda mode: not custom_label.match(mode), modes)),
            intermediate_modes
        ))

    ssa_path = convert_linear_to_ssa(path)
    contraction_list = input_modes
    contraction_list += intermediate_modes

    for k, (i, j) in enumerate(ssa_path):
        modesA = set(contraction_list[i])
        modesB = set(contraction_list[j])
        modesOut = set(intermediate_modes[k])
        assert modesOut.issubset(modesA.union(modesB))
    assert set(output_modes) == set(intermediate_modes[-1])


class ExpressionFactory:
    """Take a valid einsum expression and compute shapes, modes, etc for testing."""

    size_dict = dict(zip(EINSUM_SYMBOLS_BASE, (2, 3, 4)*18))

    def __init__(self, expression, rng):
        self.expr = expression
        if isinstance(expression, str):
            self.expr_format = "subscript"
        elif isinstance(expression, tuple):
            self.expr_format = "interleaved"
        else:
            assert False
        self._modes = None
        self._num_inputs = 0
        self._num_outputs = 0
        self.rng = rng

    def _gen_shape(self, modes):
        shape = []

        # find ellipsis, record the position, and remove it
        ellipsis, modes = check_ellipsis(modes)

        # generate extents for remaining modes
        for mode in modes:
            if mode in self.size_dict:
                extent = self.size_dict[mode]
            else:
                # exotic mode label, let's assign an extent to it
                if isinstance(mode, str):
                    extent = ord(mode) % 3 + 2
                else:
                    extent = abs(hash(mode)) % 3 + 2
                self.size_dict[mode] = extent
            shape.append(extent)

        # put back ellipsis, assuming it has single axis of extent 5
        if ellipsis >= 0:
            shape.insert(ellipsis, 5)

        return shape
    
    @property
    def num_inputs(self):
        return self._num_inputs
    
    @property
    def num_outputs(self):
        return self._num_outputs
    
    @property
    def input_shapes(self):
        out = []

        for modes in self.input_modes:
            shape = self._gen_shape(modes)
            out.append(shape)

        return out

    @property
    def output_shape(self):
        raise NotImplementedError  # TODO

    @property
    def modes(self):
        raise NotImplementedError

    @property
    def input_modes(self):
        return self.modes[:self.num_inputs]

    @property
    def output_modes(self):
        return self.modes[self.num_inputs:]

    def generate_operands(self, shapes, xp, dtype, order):
        # we always generate data from shaped_random as CuPy fixes
        # the RNG seed for us
        operands = [
            TensorBackend(backend="numpy").random(shape, dtype, self.rng).astype(dtype, order=order)
            for shape in shapes
        ]

        if xp == "torch-cpu":
            operands = [torch.as_tensor(op, device="cpu") for op in operands]
        elif xp == "torch-gpu":
            operands = [torch.as_tensor(op, device="cuda") for op in operands]
        elif xp == "cupy":
            operands = [cp.asarray(op) for op in operands]
        return operands


class EinsumFactory(ExpressionFactory):
    """Take a valid einsum expression and compute shapes, modes, etc for testing."""

    @property
    def modes(self):
        if self._modes is None:
            if self.expr_format == "subscript":
                if "->" in self.expr:
                    inputs, output = self.expr.split("->")
                    inputs = inputs.split(",")
                else:
                    inputs = self.expr.split(",")
                    output = infer_output_mode_labels(inputs)
            else:
                # output could be a placeholder
                inputs = self.expr[:-1]
                if self.expr[-1] is None:
                    output = infer_output_mode_labels(inputs)
                else:
                    output = self.expr[-1]
            self._num_inputs = len(inputs)
            self._num_outputs = 1
            self._modes = tuple(inputs) + tuple([output])
        return self._modes

    def convert_by_format(self, operands, *, dummy=False):
        if dummy:
            # create dummy NumPy arrays to bypass the __array_function__
            # dispatcher, see numpy/numpy#21379 for discussion
            operands = [np.broadcast_to(0, arr.shape) for arr in operands]

        if self.expr_format == "subscript":
            data = [self.expr, *operands]
        elif self.expr_format == "interleaved":
            modes = [tuple(modes) for modes in self.input_modes]
            data = [i for pair in zip(operands, modes) for i in pair]
            data.append(tuple(self.output_modes[0]))

        return data

    def generate_qualifiers(self, xp, gradient):
        if not gradient:
            qualifiers = None
            picks = None
        elif gradient == "random":
            # picks could be all false, and torch would not be happy during
            # backprop
            while True:
                picks = self.rng.choice(2, size=self.num_inputs)
                if any(picks):
                    break
            if "torch" in xp:
                # for torch, test auto-detect, will set up torch operands later
                qualifiers = None
            else:
                qualifiers = np.zeros(
                    self.num_inputs, dtype=cutn.tensor_qualifiers_dtype)
                qualifiers[:]["requires_gradient"] = picks
        elif gradient == "all":
            # for torch, test overwrite
            qualifiers = np.zeros(
                self.num_inputs, dtype=cutn.tensor_qualifiers_dtype)
            qualifiers[:]["requires_gradient"] = True
            picks = tuple(True for i in range(self.num_inputs))

        return qualifiers, picks

    def setup_torch_grads(self, xp, picks, operands):
        if not "torch" in xp or picks is None:
            return

        for op, pick in zip(operands, picks):
            if pick:
                op.requires_grad_(True)
            else:
                op.requires_grad_(False)
            op.grad = None  # reset


class DecomposeFactory(ExpressionFactory):

    def __init__(self, expression, rng, *, shapes=None):
        super().__init__(expression, rng)

        if shapes is not None:
            # overwrite the base class's dict
            inputs, _ = self.expr.split("->")
            inputs = inputs.split(",")
            self.size_dict = dict((m, e) for k, v in zip(inputs, shapes) for m, e in zip(k, v))

    @property
    def modes(self):
        if self._modes is None:
            if self.expr_format == "subscript":
                if "->" in self.expr:
                    inputs, outputs = self.expr.split("->")
                    inputs = inputs.split(",")
                    outputs = outputs.split(",")
                    self._num_inputs = len(inputs)
                    self._num_outputs = len(outputs)
                    self._modes = tuple(inputs) + tuple(outputs)
                else:
                    raise ValueError("output tensor must be explicitly specified for decomposition")
            else:
                raise ValueError("decomposition does not support interleave format")
            
        return self._modes


def gen_rand_svd_method(rng, dtype, fixed=None, exclude=None):
    assert dtype in DECOMPOSITION_DTYPE_NAMES, f"dtype {dtype} not supported"
    method = {"max_extent": rng.choice(range(1, 7)), 
              "abs_cutoff": rng.random() / 2.0,  # [0, 0.5)
              "rel_cutoff": 0.1 + rng.random() / 2.5,  # [0.1, 0.5)
              "normalization": rng.choice([None, "L1", "L2", "LInf"]),
              "partition": rng.choice([None, "U", "V", "UV"]),
              "algorithm": rng.choice(['gesvd', 'gesvdj', 'gesvdp', 'gesvdr'])}
    algorithm = method["algorithm"]
    if fixed is not None and "algorithm" in fixed:
        algorithm = fixed["algorithm"]
    skip_svd_params = exclude is not None and "algorithm" in exclude
    if not skip_svd_params:
        if algorithm != 'gesvdr':
            # gesvdr + max_extent can't be used with discarded weight truncation
            method["discarded_weight_cutoff"] = rng.random() / 10.0  # [0, 0.1)
        if algorithm == 'gesvdj':
            if dtype in ("float32", "complex64"):
                # for single precision, lowered down gesvdj_tol for convergence
                method["gesvdj_tol"] = rng.choice([0, 1e-7])
            else:
                method["gesvdj_tol"] = rng.choice([0, 1e-14])
            method["gesvdj_max_sweeps"] = rng.choice([0, 100])
        elif algorithm == 'gesvdr':
            method["gesvdr_niters"] = rng.choice([0, 40])
        # we can't set oversampling as it depends on matrix size here
    # updating method again in case svd_params are already
    if fixed is not None:
        method.update(fixed)
    if exclude is not None:
        for key in exclude:
            method.pop(key, None)
    return method

def get_svd_methods_for_test(num, dtype, rng):
    # single dw cutoff to verify dw < dw_cutoff
    methods = [tensor.SVDMethod(), tensor.SVDMethod(discarded_weight_cutoff=0.05)]
    for _ in range(num):
        svd_method = gen_rand_svd_method(rng, dtype)
        methods.append(tensor.SVDMethod(**svd_method))
    return methods

# We want to avoid fragmenting the stream-ordered mempools
_predefined_streams = {np: None}
if cp is not None:
    _predefined_streams[cp] = cp.cuda.Stream()
else:
    _predefined_streams[np] = None
if torch is not None:
    _predefined_streams[torch] = torch.cuda.Stream()

def get_stream_for_backend(backend):
    return _predefined_streams[backend]

def get_rng_iterator():
    return (np.random.default_rng(i) for i in count())

def get_array_framework_iterator():
    yield from cycle(ARRAY_BACKENDS)

class _BaseTester:

    def _get_rng(self, *args):
        for arg in args:
            if not hasattr(arg, '__str__'):
                raise ValueError(f"Argument {arg} is not stringable")
        string = ''.join([str(arg) for arg in args])
        # python hash is not deterministic across different runs, so we use md5 to make it deterministic
        seed = int(hashlib.md5(string.encode()).hexdigest(), 16)
        return np.random.default_rng(seed)

    def _get_array_framework(self, *args):
        rng = self._get_rng(*args)
        return rng.choice(ARRAY_BACKENDS).item()

    def _get_xp(self, *args):
        rng = self._get_rng(*args)
        return rng.choice(BACKEND_MEMSPACE).item()
# We use the pytest marker hook to deselect/ignore collected tests
# that we do not want to run. This is better than showing a ton of
# tests as "skipped" at the end, since technically they never get
# tested.
#
# Note the arguments here must be named and ordered in exactly the
# same way as the tests being marked by @pytest.mark.uncollect_if().

def skip_torch_cpu_float16_tests(xp, dtype, *args, **kwargs):
    # float16 only implemented for gpu
    return xp == 'torch-cpu' and dtype == 'float16'

def skip_einsum_expr_pack_tests(einsum_expr_pack, dtype, *args, **kwargs):
    if isinstance(einsum_expr_pack, list):
        _, _, _, overwrite_dtype = einsum_expr_pack
        if dtype != overwrite_dtype:
            return True
    return False

class Deselector:
    # used by test_contract.py
    @staticmethod
    def deselect_gradient_tests(xp, gradient, dtype, *args, **kwargs):
        if gradient and 'torch' not in xp:
            return True
        return skip_torch_cpu_float16_tests(xp, dtype, *args, **kwargs)

    # used by test_contract.py, test_network.py
    @staticmethod
    def deselect_contract_tests(einsum_expr_pack, xp, dtype, gradient, *args, **kwargs):
        return (skip_einsum_expr_pack_tests(einsum_expr_pack, dtype, *args, **kwargs)
                or Deselector.deselect_gradient_tests(xp, gradient, dtype, *args, **kwargs))

    # used by test_contract.py
    @staticmethod
    def deselect_einsum_tests(einsum_expr_pack, xp, dtype, *args, **kwargs):
        return (skip_einsum_expr_pack_tests(einsum_expr_pack, dtype, *args, **kwargs)
                or skip_torch_cpu_float16_tests(xp, dtype, *args, **kwargs))
    
    # used by test_contract_decompose.py
    @staticmethod
    def deselect_decompose_tests(
            decompose_expr, xp, dtype, *args, **kwargs):
        if xp.startswith('torch') and torch is None:
            return True
        return False

    # used by test_contract_decompose.py
    @staticmethod
    def deselect_contract_decompose_algorithm_tests(qr_method, svd_method, *args, **kwargs):
        if qr_method is False and svd_method is False: # not a valid algorithm
            return True
        return False

def is_device_id_valid(device_id):
    if device_id is not None:
        return device_id < NUM_DEVICES
    return True

def deselect_network_operator_from_pauli_string_tests(*args, **kwargs):
    backend = kwargs.get('backend')
    dtype = kwargs.get('dtype')
    if backend == 'torch-cpu' or dtype.startswith('float'): # NetworkOperator.from_pauli_strings not support torch-cpu
        return True
    return deselect_invalid_network_operator_tests(*args, **kwargs)

def deselect_invalid_device_id_tests(*args, **kwargs):
    device_id = kwargs.get('device_id', None)
    return not is_device_id_valid(device_id)

def deselect_invalid_network_operator_tests(*args, **kwargs):
    backend = kwargs.get('backend')
    return deselect_invalid_device_id_tests(*args, **kwargs) or (backend.startswith('torch') and torch is None)

def get_state_internal_backend_device(backend, device_id):
    expected_backend = {
            'numpy': 'cuda',
            'cupy': 'cupy',
            'torch': 'torch', # same as torch-gpu
            'torch-cpu': 'torch', 
            'torch-gpu': 'torch', 
        }[backend]
    expected_device = 0 if device_id is None else device_id
    return expected_backend, expected_device

def get_dtype_name(dtype):
    if not isinstance(dtype, str):
        dtype = getattr(dtype, '__name__', str(dtype).split('.')[-1])
    return dtype

def get_or_create_tensor_backend(backend, **kwargs):
    if not isinstance(backend, TensorBackend):
        return TensorBackend(backend=backend, **kwargs)
    return backend

class TensorBackend:
    def __init__(self, backend='cupy', device_id=None):
        assert backend in {'cupy', 'numpy', 'torch-cpu', 'torch-gpu', 'torch'}
        if backend in {'numpy', 'torch-cpu'}:
            self.device = 'cpu'
        else:
            self.device = device_id if device_id is not None else 0
        self.full_name = backend
        self.name = backend = backend.split('-')[0]
        self.module = importlib.import_module(backend)
        if backend in {'torch', 'cupy'}:
            tensor_wrapper.maybe_register_package(backend)
        self._asarray = _get_asarray_function(backend, self.device, None)
    
    @property
    def __name__(self):
        return self.name
    
    def random(self, shape, dtype, rng):
        if dtype.startswith('complex'):
            real_dtype = {'complex128': 'float64', 'complex64': 'float32'}[dtype]
        else:
            real_dtype = dtype if dtype != 'float16' else 'float32' # special case for float16
        array = rng.random(shape, dtype=real_dtype)
        if dtype == 'float16':
            array = array.astype(dtype)
        if dtype.startswith('complex'):
            array = array + 1.j * rng.random(shape, dtype=real_dtype)
        return self._asarray(array)
    
    @staticmethod
    def to_numpy(array):
        package = infer_object_package(array)
        if package == "cupy":
            return array.get()
        elif package == "torch":
            if array.device.type == "cpu":
                return array.numpy()
            else:
                return array.cpu().numpy()
        else:
            return array
    
    @classmethod
    def from_array(cls, array):
        package = infer_object_package(array)
        if package == "cupy":
            return cls(backend="cupy", device_id=array.device.id)
        elif package == "torch":
            return cls(backend="torch-gpu", device_id=array.device.index)
        elif package == "numpy":
            return cls(backend="numpy")
        else:
            raise ValueError(f"Unsupported package: {package}")
    
    def asarray(self, *args, **kwargs):
        return self._asarray(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        array = np.zeros(*args, **kwargs)
        return self._asarray(array)
    
    def norm(self, *args, **kwargs):
        return self.module.linalg.norm(*args, **kwargs)
    
    def allclose(self, *args, **kwargs):
        if np.isscalar(args[0]) and np.isscalar(args[1]):
            return np.allclose(*args, **kwargs)
        return self.module.allclose(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return getattr(self.module, name)
        except AttributeError as e:
            raise e
    
    @staticmethod
    def verify_close(a, b, **kwargs):
        package = infer_object_package(a)
        if package == infer_object_package(b) and package in {'numpy', 'cupy', 'torch'}:
            module = importlib.import_module(package)
        else:
            # scalar also included in this branch
            a = TensorBackend.to_numpy(a)
            b = TensorBackend.to_numpy(b)
            module = np
        return module.allclose(a, b, **kwargs)


@pytest.fixture(scope="function", autouse=True)
def cleanup_between_tests(request):
    yield
    # Get parametrized arguments
    if hasattr(request.node, 'callspec'):
        for arg_name, arg_value in request.node.callspec.params.items():
            if arg_name == 'xp':
                if arg_value == "numpy":
                    memory.free_reserved_memory()
                elif arg_value == "cupy" and cp is not None:
                    cp.get_default_memory_pool().free_all_blocks()
                elif arg_value.startswith("torch") and torch is not None:
                    torch.cuda.empty_cache()


def _contract_mpo_to_operator(mpo_tensors_torch, mpo_modes, state_dims):
    """Contract MPO chain to full operator O with shape (K, K), K = prod(state_dims[m] for m in mpo_modes).
    O is in (ket, bra) layout so O[i,j] = ⟨i|O|j⟩.
    """
    n = len(state_dims)
    mode_frontier = n
    current_modes = list(range(n))
    tensors = []
    label_lists = []
    ket_labels_in_order = []
    bra_labels_in_order = []
    prev_mode = None
    for i, m in enumerate(mpo_modes):
        ket_mode = current_modes[m]
        current_modes[m] = bra_mode = mode_frontier
        ket_labels_in_order.append(ket_mode)
        bra_labels_in_order.append(bra_mode)
        mode_frontier += 1
        next_mode = mode_frontier
        mode_frontier += 1
        if i == 0:
            tensors.append(mpo_tensors_torch[i])
            label_lists.append([ket_mode, next_mode, bra_mode])
        elif i == len(mpo_modes) - 1:
            tensors.append(mpo_tensors_torch[i])
            label_lists.append([prev_mode, ket_mode, bra_mode])
        else:
            tensors.append(mpo_tensors_torch[i])
            label_lists.append([prev_mode, ket_mode, next_mode, bra_mode])
        prev_mode = next_mode
    all_labels = sorted(set().union(*[set(ll) for ll in label_lists]))
    label_to_char = {}
    for idx, lab in enumerate(all_labels):
        label_to_char[lab] = chr(ord("a") + idx) if idx < 26 else chr(ord("A") + idx - 26)
    out_labels = [label_to_char[lab] for lab in ket_labels_in_order] + [label_to_char[lab] for lab in bra_labels_in_order]
    subscripts = ",".join("".join(label_to_char[lab] for lab in ll) for ll in label_lists) + "->" + "".join(out_labels)
    out = torch.einsum(subscripts, *tensors)
    dims = [state_dims[m] for m in mpo_modes]
    K = int(np.prod(dims))
    out = out.reshape(K, K)
    return out


def _check_mpo_hermitian(mpo_tensors_torch, mpo_modes, state_dims, atol=1e-6, rtol=1e-5):
    """Contract MPO to full operator and check op_matrix == op_matrix†. Print result and return is_hermitian."""
    op_matrix = _contract_mpo_to_operator(mpo_tensors_torch, mpo_modes, state_dims)
    op_matrix_dag = op_matrix.conj().T
    diff = op_matrix - op_matrix_dag
    max_abs_diff = torch.max(torch.abs(diff)).item()
    nrm_operator = torch.linalg.norm(op_matrix).item()
    is_hermitian = torch.allclose(op_matrix, op_matrix_dag, atol=atol, rtol=rtol)
    print(f"[_check_mpo_hermitian] mpo_modes={mpo_modes} state_dims={tuple(state_dims)}")
    print(f"  op_matrix shape={tuple(op_matrix.shape)} norm={nrm_operator:.6g} max|O - O†|={max_abs_diff:.6g} is_hermitian={is_hermitian}")
    return is_hermitian


def _mpo_expectation_torch(state, mpo_tensors_torch, mpo_modes):
    """Compute ⟨ψ|O_mpo|ψ⟩ with torch state and MPO tensors (differentiable). """
    state_dims = tuple(state.shape)
    _check_mpo_hermitian(mpo_tensors_torch, mpo_modes, state_dims)
    n = state.ndim
    mode_frontier = n
    modes = list(range(n))
    current_modes = modes.copy()
    operands = [state, modes]
    prev_mode = None
    for i, m in enumerate(mpo_modes):
        ket_mode = current_modes[m]
        current_modes[m] = bra_mode = mode_frontier
        mode_frontier += 1
        next_mode = mode_frontier
        mode_frontier += 1
        if i == 0:
            operands += [mpo_tensors_torch[i], (ket_mode, next_mode, bra_mode)]
        elif i == len(mpo_modes) - 1:
            operands += [mpo_tensors_torch[i], (prev_mode, ket_mode, bra_mode)]
        else:
            operands += [mpo_tensors_torch[i], (prev_mode, ket_mode, next_mode, bra_mode)]
        prev_mode = next_mode
    operands += [state.conj(), current_modes]
    return contract(*operands)


class TorchRef:
    """PyTorch-based reference for expectation value and gradients (⟨ψ|H|ψ⟩ and ∂E/∂θ).
    Used to compare against NetworkState.compute_expectation_with_gradients.
    """

    def create_hamiltonian_terms(
        self,
        state_dims,
        hamiltonian,
        *,
        dtype="complex128",
        device=None,
        torch_dtype=None,
        np_dtype=None,
        torch_asarray=None,
    ):
        """
        Build the list of Hamiltonian terms (product and/or MPO) for use in the ref.
        Can be called outside compute_expectation_with_gradients.

        Parameters
        ----------
        state_dims : tuple of int
            Local dimensions per mode.
        hamiltonian : dict or NetworkOperator
            Pauli string -> coefficient dict, or NetworkOperator (tensor_products and mpos).
        dtype : str or torch.dtype
            Data type; default "complex128".
        device : torch.device, optional
            Device for product-term tensors; default torch.device("cpu").
        torch_dtype : torch.dtype, optional
            Inferred from dtype if not given.
        np_dtype : np.dtype, optional
            Inferred from dtype if not given.
        torch_asarray : callable, optional
            Inferred from _get_backend_asarray_func(torch) if not given.

        Returns
        -------
        terms : list
            Each element is either:
            - ("product", coeff, gate_list): gate_list is list of (tensor, (q,)) for each mode,
              tensors on device with torch_dtype (apply as psi_ket @ g per mode).
            - ("mpo", coeff, mpo_tensors_np, mpo_modes): mpo_tensors_np list of np.ndarray,
              mpo_modes list of int.
        """
        if torch is None:
            raise RuntimeError("PyTorch is required")
        _str_to_torch = {
            "complex128": torch.complex128, "complex64": torch.complex64,
            "float64": torch.float64, "float32": torch.float32,
        }
        _torch_to_np = {
            torch.complex128: np.complex128, torch.complex64: np.complex64,
            torch.float64: np.float64, torch.float32: np.float32,
        }
        if isinstance(dtype, str):
            _torch_dtype = _str_to_torch.get(dtype, torch.complex64)
        else:
            _torch_dtype = dtype
        _np_dtype = _torch_to_np.get(_torch_dtype, np.complex64)
        device = device if device is not None else torch.device("cpu")
        torch_dtype = torch_dtype if torch_dtype is not None else _torch_dtype
        np_dtype = np_dtype if np_dtype is not None else _np_dtype
        torch_asarray = torch_asarray if torch_asarray is not None else _get_backend_asarray_func(torch)

        terms = []

        if isinstance(hamiltonian, NetworkOperator):
            # Tensor product terms: (tensors_per_mode, modes, coeff). Operands are (in, out).
            for tensors, modes, coeff in hamiltonian.tensor_products:
                mode_list = [m[0] if isinstance(m, (list, tuple)) else m for m in modes]
                gate_list_torch = []
                for i, q in enumerate(mode_list):
                    raw = getattr(tensors[i], "tensor", tensors[i])
                    if getattr(raw, "is_contiguous", None) is not None and not raw.is_contiguous():
                        raw = raw.contiguous()
                    arr = TensorBackend.to_numpy(raw)
                    try:
                        arr_np = np.asarray(arr, dtype=np_dtype)
                    except TypeError:
                        stream_holder = get_or_create_stream(getattr(raw, "device_id", 0), None, "cuda")
                        arr_np = np.asarray(ndbuffer_to_numpy(arr, stream_holder), dtype=np_dtype)
                    gate_list_torch.append(
                        (torch_asarray(arr_np).to(device=device, dtype=torch_dtype).detach(), (q,))
                    )
                coeff = complex(coeff) if np.iscomplexobj(coeff) else float(coeff)
                terms.append(("product", coeff, gate_list_torch))
            # MPO terms
            for mpo_tensors, mpo_modes, coeff in hamiltonian.mpos:
                mpo_tensors_np = []
                for t in mpo_tensors:
                    raw = getattr(t, "tensor", t)
                    if getattr(raw, "is_contiguous", None) is not None and not raw.is_contiguous():
                        raw = raw.contiguous()
                    arr = TensorBackend.to_numpy(raw)
                    try:
                        mpo_tensors_np.append(np.asarray(arr, dtype=np_dtype))
                    except TypeError:
                        stream_holder = get_or_create_stream(getattr(raw, "device_id", 0), None, "cuda")
                        mpo_tensors_np.append(np.asarray(ndbuffer_to_numpy(arr, stream_holder), dtype=np_dtype))
                coeff = complex(coeff) if np.iscomplexobj(coeff) else float(coeff)
                terms.append(("mpo", coeff, mpo_tensors_np, list(mpo_modes)))
        elif isinstance(hamiltonian, dict):
            # Convert via from_pauli_strings to match exactly what NetworkState passes to C++
            # (including identity removal behavior).
            from cuquantum.tensornet.experimental import NetworkOperator as _NO
            op = _NO.from_pauli_strings(hamiltonian, dtype=dtype, backend="numpy")
            return self.create_hamiltonian_terms(
                state_dims, op, dtype=dtype, device=device,
                torch_dtype=torch_dtype, np_dtype=np_dtype, torch_asarray=torch_asarray,
            )
        else:
            raise TypeError(
                "hamiltonian must be a Pauli string dict or a NetworkOperator, "
                f"got {type(hamiltonian).__name__}"
            )
        return terms

    def compute_expectation_with_gradients(
        self,
        state_dims,
        gate_sequence,
        hamiltonian,
        *,
        dtype="complex128",
        expectation_value_adjoint=1.0,
    ):
        """
        Build state from |0…0⟩ by applying gate_sequence, compute E = ⟨ψ|H|ψ⟩,
        and gradients of E w.r.t. each gate with requires_grad=True.

        Parameters
        ----------
        state_dims : tuple of int
            Local dimensions per mode, e.g. (2, 2, 2, 2).
        gate_sequence : list of (modes, gate_tensor, requires_grad)
            modes: tuple of mode indices; gate_tensor: array (e.g. 2×2 for 1-qubit).
        hamiltonian : dict or NetworkOperator
            Pauli string -> coefficient, e.g. {"ZZII": 2.0, "IZII": 3.0}, or a NetworkOperator
            (e.g. for qudits). Internally dispatched to dict (Pauli) or operator-terms path.
        dtype : str or torch.dtype, optional
            Data type for state and gates, e.g. "complex128", "complex64"; default "complex128".
        expectation_value_adjoint : scalar, optional
            Adjoint scaling for the expectation value (gradients are scaled by this); default 1.0.

        Returns
        -------
        expectation_value : float
            Real part of ⟨ψ|H|ψ⟩.
        gradients_list : list of arrays
            Gradient of E w.r.t. each parameterized gate, in application order (scaled by expectation_value_adjoint).
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for TorchRef")
        _dtype_map = {
            "complex128": torch.complex128,
            "complex64": torch.complex64,
            "float64": torch.float64,
            "float32": torch.float32,
        }
        if isinstance(dtype, str):
            torch_dtype = _dtype_map.get(dtype, torch.complex64)
        else:
            torch_dtype = dtype
        n = len(state_dims)
        device = torch.device("cpu")
        torch.manual_seed(42)
        torch_asarray = _get_backend_asarray_func(torch)
        hamiltonian_terms = self.create_hamiltonian_terms(
            state_dims,
            hamiltonian,
            dtype=dtype,
            device=device,
            torch_dtype=torch_dtype,
            torch_asarray=torch_asarray,
        )

        # Shared gate tensors: autograd accumulates gradients across per-term backward calls.
        gate_tensors = []
        param_gates = []
        for modes, gate, requires_grad in gate_sequence:
            gate = torch.as_tensor(gate, dtype=torch_dtype, device=device).clone()
            if requires_grad:
                gate = gate.requires_grad_(True)
                param_gates.append((modes, gate))
            gate_tensors.append((modes, gate))

        def _get_operator_modes(term):
            """Return set of modes that have an operator tensor in this term.

            collapseIsometries checks network structure, not tensor values,
            so even an identity-valued operator tensor blocks U U† collapse.
            """
            tag = term[0]
            if tag == "product":
                _, _, gate_list = term
                modes_set = set()
                for (_gate_t, qs) in gate_list:
                    modes_set.update(qs)
                return modes_set
            elif tag == "mpo":
                return set(term[3])
            return set()

        def _prune_gates(operator_modes):
            """Per-term gate cancellation (lightcone simplification).

            Iterate gates in reverse: a gate cancels iff none of its wires
            have been touched by the operator or by any subsequent active gate.
            Returns only the active gates.
            """
            touched = set(operator_modes)
            active = [True] * len(gate_tensors)
            for i in range(len(gate_tensors) - 1, -1, -1):
                gate_modes = gate_sequence[i][0]
                if any(m in touched for m in gate_modes):
                    touched.update(gate_modes)
                else:
                    active[i] = False
            return [gate_tensors[i] for i in range(len(gate_tensors)) if active[i]]

        def _build_state(gates):
            s = torch.zeros(tuple(state_dims), dtype=torch_dtype, device=device)
            s[(0,) * n] = 1.0
            for modes, gate in gates:
                if len(modes) == 1:
                    (q,) = modes
                    s = s.moveaxis(q, -1)
                    s = s @ gate.T
                    s = s.moveaxis(-1, q)
                else:
                    other = [i for i in range(n) if i not in modes]
                    perm = other + list(modes)
                    s = s.permute(perm)
                    shape_in = int(np.prod([state_dims[m] for m in modes]))
                    s = s.reshape(-1, shape_in) @ gate.reshape(shape_in, shape_in).T
                    s = s.reshape(
                        tuple(state_dims[m] for m in other)
                        + tuple(state_dims[m] for m in modes),
                    )
                    inv_perm = [0] * len(perm)
                    for i, p in enumerate(perm):
                        inv_perm[p] = i
                    s = s.permute(inv_perm)
            return s

        def _single_term_expectation(term):
            tag = term[0]
            active_gates = _prune_gates(_get_operator_modes(term))
            state = _build_state(active_gates)
            if tag == "product":
                _, coeff, gate_list = term
                psi_ket = state.clone()
                for (gate, qs) in gate_list:
                    q = qs[0]
                    g = gate.to(device=device) if gate.device != device else gate
                    psi_ket = psi_ket.moveaxis(q, -1)
                    psi_ket = psi_ket @ g
                    psi_ket = psi_ket.moveaxis(-1, q)
                return coeff * torch.vdot(state.reshape(-1), psi_ket.reshape(-1))
            else:
                assert tag == "mpo", term
                _, coeff, mpo_tensors_np, mpo_modes = term
                mpo_tensors_torch = [
                    torch_asarray(t).to(device=device, dtype=torch_dtype).detach()
                    for t in mpo_tensors_np
                ]
                return coeff * _mpo_expectation_torch(state, mpo_tensors_torch, mpo_modes)

        # Accumulate expectation value and gradients across terms.
        total_expectation = 0.0
        for term in hamiltonian_terms:
            term_val = _single_term_expectation(term)
            total_expectation += term_val.real.item()
            term_val.real.backward()

        adjoint = torch.as_tensor(expectation_value_adjoint, dtype=torch_dtype, device=device)
        scale = getattr(hamiltonian, "_expectation_gradient_scale", 1.0) * adjoint
        gradients_list = []
        for modes, gate_tensor in param_gates:
            if gate_tensor.grad is None:
                g = torch.zeros_like(gate_tensor)
            else:
                g = gate_tensor.grad.detach() * scale
            if len(modes) == 1:
                g = g.T
            else:
                g = g.permute(*torch.arange(g.ndim - 1, -1, -1))
            gradients_list.append(g)

        return total_expectation, gradients_list


def ndbuffer_to_numpy(arr, stream_holder):
    """Convert nvmath NDBuffer to numpy (used when gradient .to('cpu').tensor is NDBuffer for cupy/torch)."""
    try:
        from nvmath.internal.tensor_ifc_ndbuffer import ndbuffer, NDBufferTensor
        from nvmath.internal.tensor_ifc_numpy import NumpyTensor
    except ImportError:
        return np.asarray(arr).copy()
    if not isinstance(arr, ndbuffer.NDBuffer):
        return np.asarray(arr).copy()
    arr = NDBufferTensor(arr)
    arr = wrap_operand(arr) if not isinstance(arr, TensorHolder) else arr
    return np.asarray(NumpyTensor.create_host_from(arr.to("cpu", stream_holder), stream_holder).tensor).copy()


def _needs_ndbuffer_convert(arr):
    """True if arr is a device array or NDBuffer and must be converted via ndbuffer_to_numpy."""
    return getattr(arr, "device_id", None) is not None or (
        type(arr).__name__ == "NDBuffer" and "ndbuffer" in type(arr).__module__
    )


def extract_gradient_array(grad):
    """Extract numpy array from cuQuantum gradient (wrapper or raw array). Preserve shape from holder if conversion loses it."""
    stream_holder = get_or_create_stream(getattr(grad, "device_id", 0), None, "cuda")
    raw = grad.tensor if hasattr(grad, "tensor") else grad
    if hasattr(grad, "to") and callable(getattr(grad, "to")):
        raw = grad.to("cpu", stream_holder)
        raw = raw.tensor if hasattr(raw, "tensor") else raw

    if _needs_ndbuffer_convert(raw):
        out = ndbuffer_to_numpy(raw, stream_holder)
    else:
        try:
            out = np.asarray(raw).copy()
        except (TypeError, ValueError):
            out = np.asarray(TensorBackend.to_numpy(raw))
        if isinstance(out, np.ndarray) and out.dtype == np.dtype("O") and out.size > 0:
            elem = out.flat[0] if out.size == 1 else out
            if _needs_ndbuffer_convert(elem):
                out = ndbuffer_to_numpy(elem, stream_holder)

    if hasattr(grad, "shape") and grad.shape and out.size == np.prod(grad.shape) and out.shape != grad.shape:
        out = out.reshape(grad.shape)
    return out


def expectation_as_real(exp, dtype):
    """Cast expectation value to real and to the config's real dtype for comparison."""
    real_dtype = np.float64 if ("128" in str(dtype) or dtype == "float64") else np.float32
    scalar = np.asarray(exp).reshape(-1)[0]
    return np.asarray(np.real(scalar), dtype=real_dtype)

def assert_gradients_match(gate_sequence, gradients_cutn, gradients_ref_list, tol, *, gradient_tensor_ids=None):
    """Assert that cuQuantum gradients match the reference list (same count and values).
    Order cutn gradients by gradient_tensor_ids (application order) when provided;
    otherwise by sorted(tensor_id).
    On mismatch, prints all gradient tensors (cutn and ref) for debugging.
    """
    num_grad_gates = sum(1 for g in gate_sequence if g[2])
    assert num_grad_gates > 0, "no gradient gates in gate_sequence"
    assert len(gradients_cutn) == num_grad_gates, (
        f"expected {num_grad_gates} gradients, got {len(gradients_cutn)}"
    )
    
    if gradient_tensor_ids is not None:
        assert len(gradient_tensor_ids) == num_grad_gates, (
            f"gradient_tensor_ids length {len(gradient_tensor_ids)} != num_grad_gates {num_grad_gates}"
        )
        grad_vals = [gradients_cutn[tid] for tid in gradient_tensor_ids]
    else:
        grad_vals = [gradients_cutn[k] for k in sorted(gradients_cutn.keys())]
    for i, g_cutn in enumerate(grad_vals):
        g_ref = TensorBackend.to_numpy(gradients_ref_list[i])
        g_cutn_np = extract_gradient_array(g_cutn)
        assert TensorBackend.verify_close(g_cutn_np, g_ref, **tol), (
            f"gradient {i} mismatch",
            g_cutn_np,
            g_ref,
        )