# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import sys
from types import MappingProxyType
import importlib

import cupy as cp
from cupy.testing import shaped_random
import numpy as np
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
except ImportError:
    torch = None

from cuquantum import cutensornet as cutn
from cuquantum import OptimizerOptions
from cuquantum import tensor
from cuquantum.cutensornet.experimental._internal.network_state_utils import _get_asarray_function
from cuquantum.cutensornet._internal.circuit_converter_utils import EINSUM_SYMBOLS_BASE
from cuquantum.cutensornet._internal.decomposition_utils import DECOMPOSITION_DTYPE_NAMES
from cuquantum.cutensornet._internal.einsum_parser import infer_output_mode_labels

from .data import dtype_names


machine_epsilon_values = [np.finfo(dtype).eps for dtype in dtype_names]

rtol_mapper = dict(zip(
    dtype_names,
    [np.sqrt(m_eps) for m_eps in machine_epsilon_values]
))

atol_mapper = dict(zip(
    dtype_names,
    [10 * m_eps for m_eps in machine_epsilon_values]
))

EMPTY_DICT = MappingProxyType(dict())
DEFAULT_RNG = np.random.default_rng(2023)

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

    def __init__(self, expression):
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

    def generate_operands(self, shapes, xp, dtype, order, seed=0):
        # we always generate data from shaped_random as CuPy fixes
        # the RNG seed for us
        if xp == "torch-cpu":
            _xp = np
        elif xp == "torch-gpu":
            _xp = cp
        else:
            _xp = sys.modules[xp]

        operands = [
            shaped_random(shape, xp=_xp, dtype=dtype, order=order, seed=seed)
            for shape in shapes
        ]

        if xp == "torch-cpu":
            operands = [torch.as_tensor(op, device="cpu") for op in operands]
        elif xp == "torch-gpu":
            operands = [torch.as_tensor(op, device="cuda") for op in operands]

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
                picks = np.random.choice(2, size=self.num_inputs)
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

    def __init__(self, expression, *, shapes=None):
        super().__init__(expression)

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

def get_svd_methods_for_test(num, dtype):
    # single dw cutoff to verify dw < dw_cutoff
    methods = [tensor.SVDMethod(), tensor.SVDMethod(discarded_weight_cutoff=0.05)]
    rng = np.random.default_rng(2021)
    for _ in range(num):
        svd_method = gen_rand_svd_method(rng, dtype)
        methods.append(tensor.SVDMethod(**svd_method))
    return methods

# We want to avoid fragmenting the stream-ordered mempools
_predefined_streams = {
    np: cp.cuda.Stream(),  # implementation detail
    cp: cp.cuda.Stream(),
}
if torch is not None:
    _predefined_streams[torch] = torch.cuda.Stream()

def get_stream_for_backend(backend):
    return _predefined_streams[backend]


# We use the pytest marker hook to deselect/ignore collected tests
# that we do not want to run. This is better than showing a ton of
# tests as "skipped" at the end, since technically they never get
# tested.
#
# Note the arguments here must be named and ordered in exactly the
# same way as the tests being marked by @pytest.mark.uncollect_if().
def deselect_contract_tests(
        einsum_expr_pack, xp, dtype, *args, **kwargs):
    if xp.startswith('torch') and torch is None:
        return True
    if xp == 'torch-cpu' and dtype == 'float16':
        # float16 only implemented for gpu
        return True
    if isinstance(einsum_expr_pack, list):
        _, _, _, overwrite_dtype = einsum_expr_pack
        if dtype != overwrite_dtype:
            return True
    return False

def deselect_gradient_tests(
        einsum_expr_pack, xp, dtype, order, stream,
        use_numpy_path, gradient, *args, **kwargs):
    if xp.startswith('torch') and torch is None:
        return True
    assert gradient in (False, "random", "all", "skip")
    if gradient == "skip":
        return True
    if gradient and "torch" not in xp:
        return True
    return False

def deselect_decompose_tests(
        decompose_expr, xp, dtype, *args, **kwargs):
    if xp.startswith('torch') and torch is None:
        return True
    return False

def deselect_contract_decompose_algorithm_tests(qr_method, svd_method, *args, **kwargs):
    if qr_method is False and svd_method is False: # not a valid algorithm
        return True
    return False

def is_device_id_valid(device_id):
    if device_id is not None:
        num_devices = cp.cuda.runtime.getDeviceCount()
        return device_id < num_devices
    return True

def deselect_network_operator_from_pauli_string_tests(backend, n_qubits, num_pauli_strings, dtype, device_id, *args, **kwargs):
    if backend == 'torch-cpu' or dtype.startswith('float'): # NetworkOperator.from_pauli_strings not support torch-cpu
        return True
    return not is_device_id_valid(device_id)

def deselect_invalid_device_id_tests(*args, **kwargs):
    device_id = kwargs.get('device_id', None)
    return not is_device_id_valid(device_id)

def get_state_internal_backend_device(backend, device_id):
    expected_backend = {
            'numpy': 'cupy',
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
        self._asarray = _get_asarray_function(backend, self.device, None)
    
    @property
    def __name__(self):
        return self.name
    
    def random(self, shape, dtype, rng=None):
        if dtype.startswith('complex'):
            real_dtype = {'complex128': 'float64', 'complex64': 'float32'}[dtype]
        else:
            real_dtype = dtype
        if rng is None:
            rng = np.random.default_rng(7)
        array = rng.random(shape, dtype=real_dtype)
        if real_dtype != dtype:
            array = array + 1.j * rng.random(shape, dtype=real_dtype)
        return self._asarray(array)
    
    def asarray(self, *args, **kwargs):
        return self._asarray(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        array = np.zeros(*args, **kwargs)
        return self._asarray(array)
    
    def norm(self, *args, **kwargs):
        return self.module.linalg.norm(*args, **kwargs)
    
    def einsum(self, *args, **kwargs):
        return self.module.einsum(*args, **kwargs)
    
    def allclose(self, *args, **kwargs):
        return self.module.allclose(*args, **kwargs)
    
    def vstack(self, *args, **kwargs):
        return self.module.vstack(*args, **kwargs)
