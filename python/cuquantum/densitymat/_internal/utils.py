# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import collections
import weakref
from typing import Optional, Union, Tuple
from contextlib import nullcontext

import numpy as np
import cupy as cp

from cuquantum._internal.tensor_ifc import Tensor
from cuquantum._internal.tensor_ifc_numpy import NumpyTensor
from cuquantum._internal.tensor_wrapper import wrap_operand, wrap_operands
from cuquantum._internal.utils import device_ctx, StreamHolder
from cuquantum._internal import utils as cutn_utils


NDArrayType = Union[np.ndarray, cp.ndarray]


class InvalidObjectState(Exception):
    pass


def cuda_call_ctx(ctx, blocking: Optional[bool] = None):
    blocking = ctx.blocking if blocking is None else blocking
    return cutn_utils.cuda_call_ctx(ctx._stream_holder, blocking, ctx._do_timing)


def device_ctx_from_array(arr):
    device_id = wrap_operand(arr).device_id
    if device_id is not None:
        return device_ctx(device_id)
    else:
        return nullcontext()

def generic_finalizer(
    logger,
    upstream_finalizers: collections.OrderedDict,
    *destructor_ptr_pairs,
    msg="In generic finalizer call.",
):
    logger.debug(msg + " - Outer Loop")
    for upstream_finalizer in reversed(upstream_finalizers):
        upstream_finalizer()
    logger.debug(msg + " - Inner Loop")
    for destructor, ptr in destructor_ptr_pairs:
        if ptr is not None:
            destructor(ptr)
            logger.debug(f"Released resource: {ptr}.")
        else:
            logger.debug("Resource already released.")
    logger.debug(msg + " - End of Inner Loop")
    logger.debug(msg + " - End of Outer Loop")


def register_with(user, downstream_dependency, logger):
    try:
        assert downstream_dependency._valid_state
    except AssertionError as e:
        raise RuntimeError(
            f"Failing to register {user} as dependent on {downstream_dependency} because the latter's finalizer has already been called."
        ) from e
    if downstream_dependency == user:
        raise RuntimeError(f"Cannot register {user} as dependent on itself.")
    downstream_dependency._upstream_finalizers[user._finalizer] = weakref.ref(
        user
    )  # We may not want to store weakref as value here, but let's see
    logger.debug(f"{downstream_dependency} registered user {user} for finalizer execution.")


def unregister_with(user, downstream_dependency, logger):
    if downstream_dependency is not None:
        if downstream_dependency == user:
            raise RuntimeError(f"Cannot register {user} as dependent on itself.")
        if downstream_dependency._upstream_finalizers is not None:
            del downstream_dependency._upstream_finalizers[user._finalizer]
            logger.debug(
                f"{downstream_dependency} unregistered user {user} for finalizer execution."
            )

def transpose_bipartite_tensor(tensor):
    shape = tensor.shape
    return matricize_bipartite_tensor(tensor).transpose(1, 0, 2).reshape(shape)

def matricize_bipartite_tensor(tensor):
    """
    Matricization of batched input tensor of shape (*dims, *dims, batch_dimension) to a batched matrix of shape (prod(dims), prod(dims), batch_dimension).
    """
    if len(tensor.shape) % 2 != 0:
        batchsize = tensor.shape[-1]
        dims = tensor.shape[:-1]
    else:
        raise ValueError("Only tensors with odd number of modes, the last being the batch dimension are supported")
    ndims = len(dims)
    if not dims[: ndims // 2] == dims[ndims // 2 :]:
        raise ValueError("Only tensors in which the first half of modes is identical to the second half (excluding batch dimension) are supported.")
    matricized_dim = np.prod(dims[: ndims // 2])
    return tensor.reshape(matricized_dim, matricized_dim, batchsize)


def dense_batched_matmul(a, b):
    """
    Performs batched matrix multiply. If batched bipartite tensors are input, they are first converted to batched matrices
    and reshaped to batched tensors after performing the matrix multiply.
    Both operands need to be on same device.
    """
    am = matricize_bipartite_tensor(a)
    bm = matricize_bipartite_tensor(b)
    batchdim = max(am.shape[-1], bm.shape[-1])
    assert am.shape[-1] == 1 or am.shape[-1] == batchdim
    assert bm.shape[-1] == 1 or bm.shape[-1] == batchdim
    assert am.shape[:-1] == bm.shape[:-1]
    target_shape = (*(a.shape[:-1]), batchdim)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.einsum("IJk,JLk->ILk", am, bm).reshape(target_shape).copy(order="F")
    elif isinstance(a, cp.ndarray) and isinstance(b, cp.ndarray):
        return cp.einsum("IJk,JLk->ILk", am, bm).reshape(target_shape).copy(order="F")
    else:
        raise ValueError(f"Unsupported operands for matrix multiply, {type(a)} and {type(b)}.")

def optimize_strides(tensor: NDArrayType) -> NDArrayType:
    """
    Return `tensor` as a contiguous array in F-order.

    Args:
        Input tensor.

    Returns:
        Input tensor in F-order. If input was not F-ordered, a copy on the same device/host is returned.
    """
    if tensor.flags["F_CONTIGUOUS"]:
        return tensor
    else:
        wrapped_tensor = wrap_operand(tensor)
        device_id = wrapped_tensor.device_id
        if device_id is None:
            return tensor.copy(order="F")
        else:
            with device_ctx(device_id):
                return tensor.copy(order="F")


def check_and_get_batchsize(a: int, b: int, silent: bool=False) -> int:
    if not (a == b or a == 1 or b == 1):
        if not silent:
            raise ValueError(f"Incompatible batchsizes {a} and {b}.")
    return max(a, b)


def maybe_move_array(
    array_to_be_moved, array_on_target_device, stream_holder=StreamHolder()
) -> Tuple[NDArrayType, bool]:
    _array_to_be_moved = wrap_operand(array_to_be_moved)
    _array_on_target_device = wrap_operand(array_on_target_device)

    target_device_or_device_id = (
        _array_on_target_device.device_id
        if _array_on_target_device.device_id is not None
        else "cpu"
    )
    maybe_moved_array = wrap_operand(
        _array_to_be_moved.to(target_device_or_device_id, stream_holder=stream_holder)
    )

    return maybe_moved_array.tensor, _array_to_be_moved.device_id == maybe_moved_array.device_id


def maybe_move_arrays(a, b):
    _a = wrap_operand(a)
    _b = wrap_operand(b)
    print(_a, _b, _a.device_id, _b.device_id)
    if _b.device_id is None:
        b, _ = maybe_move_array(b, a)
        return a, b
    elif _a.device_id is None:
        a, _ = maybe_move_array(a, b)
        return a, b
    else:
        assert _a.device_id == _b.device_id
        return a, b


def check_binary_tensor_shape(a, b) -> Tuple[int]:
    """
    Checks whether two batched tensors have identical shapes and compatible batch dimensions, and returns the
    shape of a tensor that would result from an element-wise broadcast operation between the two tensors.
    """
    if len(a.shape) % 2 == 0 or len(b.shape) % 2 == 0:
        raise ValueError(f"Tensors are expected to have an odd number of modes.")

    batch_size_compatible = a.shape[-1] == b.shape[-1] or a.shape[-1] == 1 or b.shape[-1] == 1
    if not batch_size_compatible:
        raise ValueError(f"Tensor batch sizes {a.shape[-1]} and {b.shape[-1]} are not compatible.")

    shape_compatible = a.shape[:-1] == b.shape[:-1]
    if not shape_compatible:
        raise ValueError(f"Tensor shapes, {a.shape[:-1]} and {b.shape[:-1]}, are not compatible")

    return max(a.shape, b.shape)
