# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import collections
import weakref
from typing import Optional, Union

import numpy as np
import cupy as cp

from cuquantum.cutensornet._internal import tensor_wrapper
from cuquantum.cutensornet._internal.tensor_ifc import Tensor
from cuquantum.cutensornet._internal.tensor_ifc_numpy import NumpyTensor
from cuquantum.cutensornet._internal.utils import device_ctx, StreamHolder
from cuquantum.cutensornet._internal import utils as cutn_utils


NDArrayType = Union[np.ndarray, cp.ndarray]


class InvalidObjectState(Exception):
    pass


def cuda_call_ctx(ctx, blocking: Optional[bool] = None):
    blocking = ctx.blocking if blocking is None else blocking
    return cutn_utils.cuda_call_ctx(ctx._stream_holder, blocking, ctx._do_timing)


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


def wrap_callback(func):
    """
    Returns callback that writes into scalar ndarray `storage` (t,args,storage) -> None, given `func` with signature (t,args)->Union[Number,ndarray].

    Parameters:
    -----------
    func: Callable
        Function with signature (t: float, args: Tuple[float]) returning a scalar.
    """

    if func is not None:

        def inplace_func(t: np.float64, args: tuple, buf: np.ndarray):
            buf[:] = func(t, args)

        return inplace_func
    else:
        return None


def single_tensor_copy(maybe_wrapped_operand, ctx):
    """
    Blocking copy on device of wrapped_operand.
    """
    wrapped_operand = (
        single_tensor_wrap(maybe_wrapped_operand)
        if not isinstance(maybe_wrapped_operand, Tensor)
        else maybe_wrapped_operand
    )
    if isinstance(wrapped_operand, NumpyTensor):
        return wrapped_operand.tensor.copy()
    else:
        if ctx is None:
            stream_holder = StreamHolder(obj=cp.cuda.Stream())
            with cutn_utils.device_ctx(wrapped_operand.device), cutn_utils.cuda_call_ctx(stream_holder, timing=False):
                tensor_copy = wrapped_operand.tensor.copy()
        else:
            assert wrapped_operand.device == ctx.device_id
            with cutn_utils.device_ctx(wrapped_operand.device), cuda_call_ctx(ctx, blocking=True):
                tensor_copy = wrapped_operand.tensor.copy()
        return tensor_copy


def single_tensor_to(operand: Tensor, device, stream_holder: StreamHolder):
    """
    Moves a single tensor to device and wraps the copied tensor in tensor wrapper.
    The equivalent of cutensornet._internal.tensor.to for a single tensor and extended to treating multidiagonal Tensors.

    Parameters:
    -----------
    operand: Tensor
        Wrapped input tensor (subclass of Tensor)
    device: int
        Destination GPU.
    stream_holder: StreamHolder
        Stream onto which the data transfer is submitted, wrapped in StreamHolder class.
    """
    device_operand = operand.to(device, stream_holder)
    return single_tensor_wrap(device_operand)


def single_tensor_wrap(operand) -> Tensor:
    """
    Wraps a single tensor in the corresponding Tensor wrapper.
    The equivalent of cutensornet._internal.tensor.wrap_operands for a single tensor and extended to treating multidiagonal Tensors.

    Parameters:
    -----------
    operand:
        Input tensor. Either a subclass of NDArray for dense tensors or a cudensitymat.MultiDiagonalTensor.

    Returns:
    --------
    Tensor
        Input tensor wrapped in Tensor subclass.
    """
    # TODO: Should use wrap_operand instead of wrap_operands
    return tensor_wrapper.wrap_operands((operand,))[0]


def transpose_bipartite_tensor(tensor):
    # if isinstance(t, MultiDiagonalTensor):
    #     return t.T
    dims = tensor.shape
    return matricize_bipartite_tensor(tensor).transpose().reshape(dims)


def matricize_bipartite_tensor(tensor):
    dims = tensor.shape
    ndims = len(dims)
    assert ndims % 2 == 0
    assert dims[: ndims // 2] == dims[ndims // 2 :]
    matricized_dim = np.prod(dims[: ndims // 2])
    return tensor.reshape(matricized_dim, matricized_dim)


def multidiagonal_to_dense(sparse_data, offsets, package):
    shape = (sparse_data.shape[0], sparse_data.shape[0])
    dense_matrix = package.zeros(shape, sparse_data.dtype, order="F")
    row, col = package.indices(shape)
    for i, offset in enumerate(offsets):
        dense_matrix[row == col - offset] = (
            sparse_data[: -abs(offset), i] if offset != 0 else sparse_data[:, i]
        )
    return dense_matrix


# TODO: Possibly remove this function in a future release
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
        wrapped_tensor = tensor_wrapper.wrap_operand(tensor)
        device_id = wrapped_tensor.device_id
        if device_id is None:
            return tensor.copy(order="F")
        else:
            with device_ctx(device_id):
                return tensor.copy(order="F")
