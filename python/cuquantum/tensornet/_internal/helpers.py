# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import numpy as np
from typing import Mapping

from nvmath.internal import formatters
from nvmath.internal.utils import Value, create_empty_tensor

torch_asarray = None
AUTO_BACKEND = None

def _get_backend_asarray_func(backend):
    if backend.__name__ == 'torch':
        global torch_asarray
        if torch_asarray is None:
            torch_asarray = functools.partial(backend.tensor, device='cuda')
        return torch_asarray
    else:
        return backend.asarray


def get_operands_data(operands):
    """
    Get the raw data pointer of the input operands for cuTensorNet.
    """
    op_data = tuple(o.data_ptr if o is not None else 0 for o in operands)
    return op_data

def get_operands_strides(operands):
    """
    Get the raw data strides in elements of the input operands for cuTensorNet.
    """
    op_strides = tuple(o.strides if o is not None else 0 for o in operands)
    return op_strides


def check_tensor_qualifiers(qualifiers, dtype, num_inputs):
    """
    Check if the tensor qualifiers array is valid.
    """

    if qualifiers is None:
        return 0

    prolog = f"The tensor qualifiers must be specified as an one-dimensional NumPy ndarray of 'tensor_qualifiers_dtype' objects."
    if not isinstance(qualifiers, np.ndarray):
        raise ValueError(prolog)
    elif qualifiers.dtype != dtype:
        message = prolog + f" The dtype of the ndarray is '{qualifiers.dtype}'."
        raise ValueError(message)
    elif qualifiers.ndim != 1:
        message = prolog + f" The shape of the ndarray is {qualifiers.shape}."
        raise ValueError(message)
    elif len(qualifiers) != num_inputs:
        message = prolog + f" The length of the ndarray is {len(qualifiers)}, while the expected length is {num_inputs}."
        raise ValueError(message)

    return qualifiers


def check_autotune_params(iterations):
    """
    Check if the autotune parameters are of the correct type and within range.
    """

    if not isinstance(iterations, int):
        raise ValueError("Integer expected.")
    if iterations < 0:
        raise ValueError("Integer >= 0 expected.")

    message = f"Autotuning parameters: iterations = {iterations}."

    return message


def check_attributes_match(orig_attributes, new_attributes, description):
    """
    Check if the corresponding attributes match between the corresponding new and old operands, and raise an exception if it 
    doesn't.
    """
    checks = [o == n for o, n in zip(orig_attributes, new_attributes)]

    if not all(checks): 
        mismatch = [f"{location}: {orig_attributes[location]} => {new_attributes[location]}"
                        for location, predicate in enumerate(checks) if predicate is False]
        mismatch = formatters.array2string(mismatch)
        message = f"""The {description} of each new operand must match the {description} of the corresponding original operand.
The mismatch in {description} as a sequence of "position: original {description} => new {description}" is: \n{mismatch}"""
        raise ValueError(message)


def check_and_set_options(required: Mapping[str, Value], provided: Mapping[str, object]):
    """
    Update each option specified in 'required' by getting the value from 'provided' if it exists or using a default.
    """
    for option, value in required.items():
        try:
            value.data = provided.pop(option)
        except KeyError:
            pass
        required[option] = value.data

    assert not provided, "Unrecognized options."

def create_output_tensor(cls, output, size_dict, device_id, stream_holder, data_type):
    """
    Create output tensor and associated data (modes, extents, strides). This operation is
    ordered through events and is safe to use with asynchronous memory pools.
    """
    modes = tuple(m for m in output)
    extents = tuple(size_dict[m] for m in output)

    output = create_empty_tensor(cls, extents, data_type, device_id, stream_holder, False)
    output_event = stream_holder.obj.record() if stream_holder is not None else None

    strides = output.strides
    return output, output_event, modes, extents, strides

def get_dtype_name(dtype):
    if not isinstance(dtype, str):
        dtype = getattr(dtype, '__name__', str(dtype).split('.')[-1])
    return dtype

def transpose_tensor(tensor_holder):
    if tensor_holder.name == 'torch':
        import torch
        tensor_t = tensor_holder.tensor.permute(*torch.arange(tensor_holder.tensor.ndim - 1, -1, -1))
    else:
        tensor_t = tensor_holder.tensor.T
    return tensor_holder.__class__(tensor_t)

def get_auto_backend():
    global AUTO_BACKEND
    if AUTO_BACKEND is None:
        try:
            import cupy as cp
            AUTO_BACKEND = cp
        except ImportError:
            AUTO_BACKEND = np
    return AUTO_BACKEND

def get_auto_backend_name():
    return get_auto_backend().__name__
