# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
import functools
import importlib

import numpy as np

from nvmath.internal import tensor_wrapper, utils
from cuquantum.bindings import cutensornet as cutn
from ..._internal.helpers import transpose_tensor, swap_bra_ket_tensor
from ...._internal.tensor_wrapper import check_valid_package

# constant parameters for MPS and tensor network simulation
STATE_DEFAULT_DTYPE = 'complex128'

STATE_SUPPORTED_DTYPE_NAMES = {'float32', 'float64', 'complex64', 'complex128'}

EXACT_MPS_EXTENT_LIMIT = 2**10 # bond extent limit for MPS simulation without an explicit MPSConfig.max_extent


def get_mps_output_extents(state_mode_extents, max_extent, itemsize):
    """Extents of the output MPS tensors: ``(mode, right)``, ``(left, mode, right)``, ..., ``(left, mode)``.

    The natural extent of a bond is the smaller product of the mode extents on either side, capped by
    ``max_extent`` if given. Without ``max_extent`` the bond extents may not exceed ``EXACT_MPS_EXTENT_LIMIT``. Every
    tensor's element count and size in bytes must fit in int64, the range of the C API. Running products saturate
    one past the int64 maximum, so no intermediate value grows without bound.
    """
    int64_max = np.iinfo(np.int64).max
    bound = int64_max + 1 if max_extent is None else min(int(max_extent), int64_max + 1)

    def bounded_cumprod(extents):
        product = 1
        result = []
        for extent in extents:
            product = min(product * int(extent), bound)
            result.append(product)
        return result

    extents_left = bounded_cumprod(state_mode_extents[:-1])
    extents_right = bounded_cumprod(reversed(state_mode_extents[1:]))
    bond_extents = [min(left, right) for left, right in zip(extents_left, reversed(extents_right))]
    if max_extent is None and max(bond_extents) > EXACT_MPS_EXTENT_LIMIT:
        needed = max(bond_extents)
        needed = str(needed) if needed <= int64_max else f"more than {int64_max}"
        raise ValueError(
            f"Without MPSConfig.max_extent, MPS bond extents are limited to {EXACT_MPS_EXTENT_LIMIT}, but the exact "
            f"bond extents of this state reach {needed}. Set MPSConfig.max_extent explicitly; a value at or above "
            f"the largest exact bond extent truncates no bond extent.")

    n = len(state_mode_extents)
    extents = []
    for i, mode_extent in enumerate(state_mode_extents):
        site = ((bond_extents[i-1],) if i > 0 else ()) + (int(mode_extent),) + ((bond_extents[i],) if i < n - 1 else ())
        elements = 1
        for extent in site:
            elements *= extent
        if elements * itemsize > int64_max:
            raise ValueError(
                f"The MPS tensor at site {i} would have extents {site} ({elements} elements of {itemsize} bytes), "
                f"beyond the int64 range of the C API; set a smaller MPSConfig.max_extent.")
        extents.append(site)
    return extents

MPS_STATE_ATTRIBUTE_MAP = {
    'canonical_center' : cutn.StateAttribute.CONFIG_MPS_CANONICAL_CENTER,
    'abs_cutoff' : cutn.StateAttribute.CONFIG_MPS_SVD_ABS_CUTOFF,
    'rel_cutoff' : cutn.StateAttribute.CONFIG_MPS_SVD_REL_CUTOFF,
    'normalization' : cutn.StateAttribute.CONFIG_MPS_SVD_S_NORMALIZATION,
    'discarded_weight_cutoff' : cutn.StateAttribute.CONFIG_MPS_SVD_DISCARDED_WEIGHT_CUTOFF,
    'algorithm' : cutn.StateAttribute.CONFIG_MPS_SVD_ALGO,
    'mpo_application': cutn.StateAttribute.CONFIG_MPS_MPO_APPLICATION, 
    'gauge_option': cutn.StateAttribute.CONFIG_MPS_GAUGE_OPTION,
    #'algorithm_params' : cutn.StateAttribute.CONFIG_MPS_SVD_ALGO_PARAMS, # NOTE: special treatment required
}

MPO_OPTION_MAP = {
    'approximate': cutn.StateMPOApplication.INEXACT,
    'exact': cutn.StateMPOApplication.EXACT
}

GAUGE_OPTION_MAP = {
    'free': cutn.StateMPSGaugeOption.STATE_MPS_GAUGE_FREE,
    'simple': cutn.StateMPSGaugeOption.STATE_MPS_GAUGE_SIMPLE
}

def check_dtype_supported(dtype_name):
    assert dtype_name in STATE_SUPPORTED_DTYPE_NAMES, f"{dtype_name} supported, must be real/complex data with single or double precision"


def check_expectation_with_gradients_norm_args(return_norm, state_norm_adjoint):
    """``return_norm`` and ``state_norm_adjoint`` must be requested together or omitted together."""
    want_norm = bool(return_norm)
    want_adj = state_norm_adjoint is not None
    if want_norm != want_adj:
        raise ValueError(
            "compute_expectation_with_gradients requires return_norm and state_norm_adjoint to be "
            "consistent: use return_norm=False with state_norm_adjoint=None to skip the squared "
            "state 2-norm and its adjoint, or return_norm=True with a non-None state_norm_adjoint; got "
            f"return_norm={return_norm!r}, state_norm_adjoint={state_norm_adjoint!r}."
        )


def state_labels_wrapper(*, marker_index=None, key=None, marker_type='seq'):
    assert marker_type in {'seq', 'dict'}, f"marker_type {marker_type} not supported"
    assert (marker_index is None and key is not None) or (marker_index is not None and key is None), "Internal Error"

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            obj = args[0]
            if obj.state_labels is not None:
                if marker_index is not None:
                    old_indices = args[marker_index]
                else:
                    old_indices = kwargs.get(key, None)
                
                if old_indices is None or all(isinstance(item, int) for item in old_indices):
                    # no need to parse
                    new_indices = old_indices
                else:
                    if marker_type == 'seq':
                        new_indices = [obj.state_labels.index(i) for i in old_indices]
                    elif marker_type == 'dict':
                        new_indices = dict()
                        for k, val in old_indices.items():
                            new_indices[obj.state_labels.index(k)] = val
                if marker_index is not None:
                    args = *args[:marker_index], new_indices, *args[marker_index+1:]
                else:
                    kwargs[key] = new_indices
                return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def resolve_unitary_kwarg(operand_arg_index, resolver_name):
    """Resolve ``unitary=None`` into a concrete bool by calling ``resolver_name``
    on the decorated object with the raw native operand, before any operand
    wrapping or device transfer (the native operand is guaranteed to carry
    array arithmetic there; the post-transfer buffer for numpy-backend states
    is not). Explicit flags pass through untouched."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs.get('unitary', None) is None:
                operand = args[operand_arg_index]
                try:
                    kwargs['unitary'] = getattr(args[0], resolver_name)(operand, args, kwargs)
                except (AttributeError, TypeError):
                    # An operand that is not ndarray-like fails classification
                    # with a raw attribute/type error; surface the canonical
                    # unsupported-operand error instead. If the package check
                    # passes, the failure is genuinely unexpected: re-raise it.
                    check_valid_package([operand])
                    raise
            return func(*args, **kwargs)
        return wrapper
    return decorator

def state_operands_wrapper(operands_arg_index=1, is_single_operand=True, transpose=False, swap_bra_ket=False):
    assert operands_arg_index >= 1
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            obj = args[0]
            device_id = obj.device_id
            stream = kwargs.get('stream', None)
            stream_holder = None
            operands = args[operands_arg_index]
            if is_single_operand:
                operands = (operands, )
            operands = tensor_wrapper.wrap_operands(operands)
            if transpose:
                operands = [transpose_tensor(o) for o in operands]
            elif swap_bra_ket:
                operands = [swap_bra_ket_tensor(o) for o in operands]
            if not obj.backend_setup:
                obj._setup_backend(operands[0])

            new_operands = []
            for o in operands:
                if o.dtype != obj.dtype:
                    raise RuntimeError(f"input operand type ({o.dtype}) different than the underlying object ({obj.dtype})")
                if o.name != obj.backend:
                    raise RuntimeError(f"input operand belongs to a different package ({o.name}) than previously specified ({obj.backend})")
                if o.device == 'cpu':
                    if stream_holder is None:
                        stream_holder = utils.get_or_create_stream(device_id, stream, obj.internal_package)
                    o = o.to(device_id, stream_holder=stream_holder)
                elif o.device_id != obj.device_id:
                    raise RuntimeError(
                        f"input operand resides on a different device ({o.device_id}) than specified in options ({obj.device_id}). "
                        f"Construct the object with options={{'device_id': {o.device_id}}} to use that device, "
                        f"or move the operand to device {obj.device_id} before applying it.")
                new_operands.append(o)
            if is_single_operand:
                new_operands = new_operands[0]
            return func(*args[:operands_arg_index], new_operands, *args[operands_arg_index+1:], **kwargs)
        return wrapper    
    return decorator

def host_scalar_to_holder(output_class, device_id, dtype, value, stream_holder):
    """
    Materialize a CUTN host 0-D ``numpy.ndarray`` buffer as an ``output_class`` TensorHolder.

    For NumPy output there is no extra allocation: the existing host buffer is wrapped.
    For CuPy/Torch a 0-D device tensor is allocated and filled via ``copy_``.
    """
    if output_class.name == "numpy":
        return tensor_wrapper.wrap_operand(value)
    out = output_class.empty((), device_id=device_id, dtype=dtype, stream_holder=stream_holder)
    if output_class.name == "torch":
        import torch
        src = tensor_wrapper.wrap_operand(
            torch.as_tensor(value, device="cpu", dtype=out.tensor.dtype)
        )
    else:
        src = tensor_wrapper.wrap_operand(value)
    out.copy_(src, stream_holder=stream_holder)
    return out


def unwrap_output_tensor(obj, holder, stream_holder):
    """Unwrap a TensorHolder to the native backend tensor for public API return."""
    if obj.output_location == "cpu" and holder.device != "cpu":
        return holder.to("cpu", stream_holder=stream_holder).tensor
    return holder.tensor


def state_result_wrapper():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            norm = None
            if result is not None:
                obj = args[0]
                if obj.backend == "numpy":
                    stream = kwargs.get('stream')
                    stream_holder = utils.get_or_create_stream(obj.device_id, stream, 'cuda' if obj.backend == 'numpy' else obj.backend)
                else:
                    stream_holder = None
                if isinstance(result, tuple):
                    result, norm = result
                result = unwrap_output_tensor(obj, result, stream_holder)
                if norm is not None:
                    norm = unwrap_output_tensor(obj, norm, stream_holder)
            if norm is None:
                return result
            return result, norm
        return wrapper
    return decorator

def _get_asarray_function(backend, device_id, stream):
    if backend not in {'numpy', 'cupy', 'torch'}:
        raise ValueError(f"only support numpy, cupy and torch")
    tensor_wrapper.maybe_register_package(backend)
    package = importlib.import_module(backend)
    if backend == 'numpy':
        return package.asarray
    if device_id == 'cpu':
        stream_holder = None
    else:
        if device_id is None:
            device_id = 0
        stream_holder = utils.get_or_create_stream(device_id, stream, backend)
    if backend == 'cupy':
        def asarray(*args, **kwargs):
            with stream_holder.ctx, package.cuda.Device(device_id):
                out = package.asarray(*args, **kwargs)
            return out
        return asarray
    else:
        def asarray(*args, **kwargs):
            dtype = kwargs.get('dtype', None)
            if isinstance(dtype, str):
                dtype = getattr(package, dtype)
                kwargs['dtype'] = dtype
            if device_id == 'cpu':
                out = package.as_tensor(*args, device=device_id, **kwargs)
            else:
                with stream_holder.ctx:
                    device = 'cuda' if device_id is None else f'cuda:{device_id}'
                    out = package.as_tensor(*args, device=device, **kwargs)
            return out
        return asarray


def get_pauli_map(backend, dtype, device_id=None, stream=None):
    asarray = _get_asarray_function(backend, device_id, stream)
    if backend == 'torch':
        module = importlib.import_module(backend)
        dtype_name = dtype
        dtype = getattr(module, dtype)
    else:
        dtype_name = dtype
    
    pauli_map = {'I': asarray([[1,0], [0,1]], dtype=dtype),
                 'X': asarray([[0,1], [1,0]], dtype=dtype),
                 'Z': asarray([[1,0], [0,-1]], dtype=dtype)}
    if not dtype_name.startswith('float'):
        pauli_map['Y'] = asarray([[0,-1j], [1j,0]], dtype=dtype)
    return pauli_map

def create_pauli_operands(pauli_strings, backend, dtype, remove_identity, device_id=None, stream=None):
    assert isinstance(remove_identity, bool), "INTERNAL ERROR: remove_identity must be a boolean"
    pauli_map = get_pauli_map(backend, dtype, device_id=device_id, stream=stream)
    operands_data = []
    n_qubits = None
    for pauli_string, coefficient in pauli_strings.items():
        if n_qubits is None:
            n_qubits = len(pauli_string)
        else:
            assert n_qubits == len(pauli_string), f"All Pauli string must be equal in length"
        tensors = []
        modes = []
        for q, pauli_char in enumerate(pauli_string):
            if pauli_char == 'I' and remove_identity: continue
            tensors.append(pauli_map[pauli_char])
            modes.append((q, ))
        if len(tensors) == 0:
            # IIIIIII
            tensors = [pauli_map['I'],] * n_qubits
            modes = [(q, ) for q in range(n_qubits)]
        operands_data.append([tensors, modes, coefficient])
    return operands_data

def get_operand_key(o):
    """Return a key that marks the underlying operand"""
    return o.shape, o.strides, o.data_ptr

def get_mps_key(mps_operands):
    """Return a key that marks the underlying MPS state"""
    return [get_operand_key(o) for o in mps_operands]