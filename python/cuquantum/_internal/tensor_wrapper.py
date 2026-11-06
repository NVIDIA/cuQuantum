# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Entry point to using tensors from different libraries seamlessly.
"""

__all__ = [ 'infer_tensor_package', 'wrap_operand', 'wrap_operands', 'to', 'copy_']

import functools
import importlib.util
import sys

import numpy as np

from nvmath.internal import formatters
from .tensor_ifc_cupy import CupyTensor
from .tensor_ifc_numpy import NumpyTensor


_TENSOR_TYPES = {
    'cupy': CupyTensor,
    'numpy': NumpyTensor
}

_SUPPORTED_PACKAGES = tuple(_TENSOR_TYPES.keys())

# torch checked but not imported; tensor type is registered on first use.
if "torch" in sys.modules or importlib.util.find_spec("torch") is not None:
    _SUPPORTED_PACKAGES += ("torch",)

def infer_tensor_package(tensor):
    """
    Infer the package that defines this tensor.
    """
    if issubclass(tensor.__class__, np.ndarray):
        return 'numpy'
    module = tensor.__class__.__module__
    package = module.split(".")[0]
    if package == "torch" and package not in _TENSOR_TYPES:
        from .tensor_ifc_torch import TorchTensor

        _TENSOR_TYPES["torch"] = TorchTensor
    return package

def wrap_operand(native_operand):
    """
    Wrap one "native" operand so that package-agnostic API can be used.
    """
    wrapped_operand = _TENSOR_TYPES[infer_tensor_package(native_operand)](native_operand)
    return wrapped_operand


def check_valid_package(native_operands):
    """
    Check if the operands belong to one of the supported packages.
    """
    operands_pkg = [infer_tensor_package(o) for o in native_operands]
    checks = [p in _SUPPORTED_PACKAGES for p in operands_pkg]
    if not all(checks):
        unknown = [f"{location}: {operands_pkg[location]}" for location, predicate in enumerate(checks) if predicate is False]
        unknown = formatters.array2string(unknown)
        message = f"""The operands should be ndarray-like objects from one of {_SUPPORTED_PACKAGES} packages.
The unsupported operands as a sequence of "position: package" is: \n{unknown}"""
        raise ValueError(message)

    return operands_pkg

def check_valid_operand_type(wrapped_operands):
    """
    Check if the wrapped operands are ndarray-like.
    """
    istensor = [o.istensor() for o in wrapped_operands]
    if not all(istensor):
        unknown = [f"{location}: {type(wrapped_operands[location].tensor)}"
                    for location, predicate in enumerate(istensor) if predicate is False]
        unknown = formatters.array2string(unknown)
        message = f"""The operands should be ndarray-like objects from one of {_SUPPORTED_PACKAGES} packages.
The unsupported operands as a sequence of "position: type" is: \n{unknown}"""
        raise ValueError(message)


def wrap_operands(native_operands):
    """
    Wrap the "native" operands so that package-agnostic API can be used.
    """

    operands_pkg = check_valid_package(native_operands)

    wrapped_operands = tuple(_TENSOR_TYPES[operands_pkg[i]](o) for i, o in enumerate(native_operands))

    check_valid_operand_type(wrapped_operands)

    return wrapped_operands


def to(operands, device, stream_holder):
    """
    Copy the wrapped operands to the specified device ('cpu' or int) and return the
    wrapped operands on the device.
    """
    operands = tuple(o.to(device, stream_holder) for o in operands)

    return wrap_operands(operands)


def copy_(src, dest, stream_holder):
    """
    Copy the wrapped operands in dest to the corresponding wrapped operands in src.
    """
    for s, d in zip(src, dest):
        if s.device_id is None:
            s = wrap_operand(s.to(d.device_id, stream_holder=stream_holder))
        d.copy_(s.tensor, stream_holder=stream_holder)
