# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
import functools
import importlib

from cuquantum.bindings import cutensornet as cutn
from ...._internal import tensor_wrapper, utils

# constant parameters for MPS and tensor network simulation
STATE_DEFAULT_DTYPE = 'complex128'

STATE_SUPPORTED_DTYPE_NAMES = {'float32', 'float64', 'complex64', 'complex128'}

EXACT_MPS_EXTENT_LIMIT = 2**10 # limit the number of qubits for exact MPS to avoid extent overflowing

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

def state_operands_wrapper(operands_arg_index=1, is_single_operand=True):
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
                    o = tensor_wrapper.wrap_operand(o.to(device_id, stream_holder=stream_holder))
                elif o.device_id != obj.device_id:
                    raise RuntimeError(f"input operand resides on a different device ({o.device_id}) than specified in options ({obj.device_id})")
                new_operands.append(o)
            if is_single_operand:
                new_operands = new_operands[0]
            return func(*args[:operands_arg_index], new_operands, *args[operands_arg_index+1:], **kwargs)
        return wrapper    
    return decorator

def state_result_wrapper(is_scalar=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            norm = None
            if result is not None:
                if isinstance(result, tuple):
                    result, norm = result
                if is_scalar:
                    if norm is None:
                        return result.tensor.item()
                    else: 
                        return result.tensor.item(), norm
                obj = args[0]
                if obj.output_location == 'cpu':
                    stream = kwargs.get('stream')
                    stream_holder = utils.get_or_create_stream(obj.device_id, stream, 'cupy' if obj.backend == 'numpy' else obj.backend)
                    result = result.to('cpu', stream_holder=stream_holder)
                else:
                    result = result.tensor
            if norm is None:
                return result
            else:
                return result, norm
        return wrapper    
    return decorator

def _get_asarray_function(backend, device_id, stream):
    if backend not in {'numpy', 'cupy', 'torch'}:
        raise ValueError(f"only support numpy, cupy and torch")
    package = importlib.import_module(backend)
    if backend == 'numpy':
        return package.asarray
    if device_id == 'cpu':
        stream_holder = None
    else:
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


def get_pauli_map(dtype, backend='cupy', device_id=None, stream=None):
    asarray = _get_asarray_function(backend, device_id, stream)
    if backend == 'torch':
        module = importlib.import_module(backend)
        dtype = getattr(module, dtype)
    pauli_i = asarray([[1,0], [0,1]], dtype=dtype)
    pauli_x = asarray([[0,1], [1,0]], dtype=dtype)
    pauli_y = asarray([[0,-1j], [1j,0]], dtype=dtype)
    pauli_z = asarray([[1,0], [0,-1]], dtype=dtype)
    
    pauli_map = {'I': pauli_i,
                 'X': pauli_x,
                 'Y': pauli_y,
                 'Z': pauli_z}
    return pauli_map

def create_pauli_operands(pauli_strings, dtype, backend='cupy', device_id=None, stream=None):
    pauli_map = get_pauli_map(dtype, backend=backend, device_id=device_id, stream=stream)
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
            if pauli_char == 'I': continue
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