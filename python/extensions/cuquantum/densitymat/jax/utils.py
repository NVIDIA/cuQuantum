# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility functions for cuQuantum Python JAX.
"""

import ctypes
from dataclasses import dataclass, field

import cupy as cp
import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm


@dataclass
class BufferMetadata:
    """
    Metadata for a buffer.
    """
    indices: list[int] = field(default_factory=list)
    types: list[int] = field(default_factory=list)
    ptrs: list[int] = field(default_factory=list)
    shape_dtypes: list[jax.ShapeDtypeStruct] = field(default_factory=list)

    def __add__(self, other: "BufferMetadata") -> "BufferMetadata":
        """
        Add two BufferMetadata objects.
        """
        return BufferMetadata(
            indices=self.indices + other.indices,
            types=self.types + other.types,
            ptrs=self.ptrs + other.ptrs,    
            shape_dtypes=self.shape_dtypes + other.shape_dtypes,
        )
    
    def __len__(self) -> int:
        """
        Return the number of buffers.
        """
        return len(self.indices)


def is_vmap_traced(buf):
    """
    Check if a buffer is traced by jax.vmap.
    """
    return isinstance(buf, jax.core.Tracer) and type(buf).__name__ == "BatchTracer"


def get_vmap_depth(obj) -> int:
    """
    Return the number of nested vmap transformations active on a traced object.
    """
    depth = 0
    trace = jax.core.find_top_trace(obj)
    while hasattr(trace, "parent_trace"):
        if type(trace).__name__ == "BatchTrace":
            depth += 1
        trace = trace.parent_trace
    return depth


def is_grad_inside_vmap(obj) -> bool:
    """
    Check if an AD transformation is active inside (inner to) a vmap transformation.
    """
    found_ad = False
    trace = jax.core.find_top_trace(obj)
    while hasattr(trace, "parent_trace"):
        if type(trace).__name__ in ["LinearizeTrace", "JVPTrace"]:
            found_ad = True
        elif type(trace).__name__ == "BatchTrace" and found_ad:
            return True
        trace = trace.parent_trace
    return False


def maybe_expand_dim(bufs, ndim):
    """
    Expand to a leading batch dimension when jax.vmap is applied or when the state is non-batched.
    """
    # TODO: This function expands dimension assuming the operand_layout in lowering has the batch
    # dimension as the most major axis. In multi-vmap situation this is not true in general.
    if not is_vmap_traced(bufs[0]) and len(bufs[0].shape) in [ndim, 2 * ndim]:
        bufs = tuple([buf.reshape((1, *buf.shape)) for buf in bufs])
    return bufs


def maybe_squeeze_dim(bufs, ndim):
    """
    Squeeze the leading batch dimension when jax.vmap is applied or when the state is non-batched.
    """
    # TODO: This function squeezes dimension assuming the operand_layout in lowering has the batch
    # dimension as the most major axis. In multi-vmap situation this is not true in general.
    if not is_vmap_traced(bufs[0]) and (
        len(bufs[0].shape) in [ndim + 1, 2 * ndim + 1] and bufs[0].shape[0] == 1
    ):
        bufs = tuple([buf.reshape(*buf.shape[1:]) for buf in bufs])
    return bufs


def get_state_batch_size_and_purity(state_in_bufs, ndim):
    """
    Get the batch size and purity of the state.
    """
    # First check all state buffer shapes.
    shape = state_in_bufs[0].shape
    for buf in state_in_bufs[1:]:
        if buf.shape != shape:
            raise ValueError("All input state buffers must have the same shape.")
    if not is_vmap_traced(state_in_bufs[0]) and len(shape) not in [ndim, 2 * ndim, ndim + 1, 2 * ndim + 1]:
        raise ValueError("The dimensions of the input state do not match the dimensions of the operator.")

    # Extract batch size based on the four values ndim, 2 * ndim, ndim + 1, 2 * ndim + 1.
    if is_vmap_traced(state_in_bufs[0]):
        # Detect batch size by traversing the trace stack.
        trace = jax.core.find_top_trace(state_in_bufs[0])
        batch_size = 1
        while hasattr(trace, "parent_trace"):
            if type(trace).__name__ == "BatchTrace":
                batch_size *= trace.axis_data.size
            trace = trace.parent_trace
    else:
        if len(shape) % ndim == 0:
            batch_size = 1
        else:  # ndim + 1 or 2 * ndim + 1
            batch_size = shape[0]

    # Extract purity based on the four values ndim, 2 * ndim, ndim + 1, 2 * ndim + 1.
    if len(shape) // ndim == 1:
        purity = cudm.StatePurity.PURE
    else:  # 2 * ndim or 2 * ndim + 1
        purity = cudm.StatePurity.MIXED

    return batch_size, purity


def check_and_return_final_batch_size(state_in_bufs, state_batch_size, op_batch_size):
    """
    Check and return the final batch size.
    """
    if len({1, state_batch_size, op_batch_size}) > 2:  # the set should be either {1} or {1, N}
        raise ValueError("The batch size of the input state does not match the batch size of the operator.")
    return max(state_batch_size, op_batch_size)


def check_and_return_op_device(op: "Operator") -> jax.Device | None:
    """
    Check if all coefficients and base operators are on the same device and return the device.
    """
    device = None

    def _check_device(obj):
        nonlocal device  # tell Python to use the outer scope's device variable
        if not isinstance(obj, jax.core.Tracer):  # only check object device if it is not traced
            # Check if the object device is a GPU.
            if obj.device.platform != 'gpu':
                raise ValueError("cuQuantum Python JAX only supports GPU devices.")

            # If device is already set, check it against the current object's device.
            # Otherwise, set it to the current object's device.
            if device is not None:
                if obj.device != device:
                    raise ValueError("All objects must be on the same device.")
            else:
                device = obj.device

    for op_term, op_term_coeff in zip(op.op_terms, op.coeffs):
        _check_device(op_term_coeff)

        for op_prod, op_prod_coeff in zip(op_term.op_prods, op_term.coeffs):
            _check_device(op_prod_coeff)

            for base_op in op_prod:
                _check_device(base_op.data)

    return device


def check_and_return_state_device(state_in_bufs: tuple[jax.Array, ...]) -> jax.Device | None:
    """
    Check if all state buffers are on the same device and return the device.
    """
    device = None

    def _check_device(obj):
        nonlocal device  # tell Python to use the outer scope's device variable
        if not isinstance(obj, jax.core.Tracer):  # only check object device if it is not traced
            # Check if the object device is a GPU.
            if obj.device.platform != 'gpu':
                raise ValueError("cuQuantum Python JAX only supports GPU devices.")

            # If device is already set, check it against the current object's device.
            # Otherwise, set it to the current object's device.
            if device is not None:
                if obj.device != device:
                    raise ValueError("All objects must be on the same device.")
            else:
                device = obj.device

    for buf in state_in_bufs:
        _check_device(buf)

    return device


def detect_ad_traced_object(obj) -> bool:
    """
    Detect if an object has been AD-traced.
    """
    if isinstance(obj, jax.core.Tracer):
        # trace = jax.core.find_top_trace(obj)  # innermost trace
        trace = obj._trace  # XXX
        while hasattr(trace, "parent_trace"):
            if type(trace).__name__ in ["LinearizeTrace", "JVPTrace"]:
                return True
            trace = trace.parent_trace
    return False


def get_empty_scalar_callback():
    """
    Return an empty scalar callback for gradient attachment.
    """
    def f(*args, **kwargs):
        return
    return cudm.WrappedScalarCallback(f, cudm.CallbackDevice.GPU)


def get_scalar_assignment_callback(coeff):
    """
    Return a scalar assignment callback.
    """
    def f(t, args, storage):
        storage[:] = f.coeff[0]
    f.coeff = cp.zeros((1,), dtype=coeff.dtype)
    return cudm.WrappedScalarCallback(f, cudm.CallbackDevice.GPU)


def get_empty_tensor_callback():
    """
    Return an empty tensor callback for gradient attachment.
    """
    def f(*args, **kwargs):
        return
    return cudm.WrappedTensorCallback(f, cudm.CallbackDevice.GPU)


def get_scalar_gradient_attachment_callback(data):
    """
    Return a scalar gradient attachment callback.
    """
    def f(t, args, scalar_grad, params_grad):
        f.scalar_grad += scalar_grad

    # f.scalar_grad is created as a CuPy array to escape JAX tracing.
    if is_vmap_traced(data):
        f.scalar_grad = cp.zeros(data.val.shape, dtype=data.val.dtype)
    else:
        f.scalar_grad = cp.zeros(data.shape, dtype=data.dtype)

    grad_callback = cudm.WrappedScalarGradientCallback(f, cudm.CallbackDevice.GPU)
    return grad_callback


def get_tensor_gradient_attachment_callback(data):
    """
    Return a tensor gradient attachment callback.
    """
    def f(t, args, tensor_grad, params_grad):
        # Transpose tensor_grad so that the batch dimension is the first dimension.
        transpose_inds = (tensor_grad.ndim - 1, *range(tensor_grad.ndim - 1))
        f.tensor_grad += tensor_grad.transpose(transpose_inds)

    # f.tensor_grad is created as a CuPy array to escape JAX tracing.
    if is_vmap_traced(data):
        f.tensor_grad = cp.zeros(data.val.shape, dtype=data.val.dtype)
    else:
        f.tensor_grad = cp.zeros(data.shape, dtype=data.dtype)

    grad_callback = cudm.WrappedTensorGradientCallback(f, cudm.CallbackDevice.GPU)
    return grad_callback


def get_random_odd_pointer_and_object():
    """
    Return a random odd pointer and its ctypes object.

    Returns:
        tuple: (pointer, obj) where obj must be kept alive by the caller
               to ensure the pointer remains valid.
    """
    obj = ctypes.c_short()
    ptr = ctypes.addressof(obj) + 1
    assert ptr % 2 == 1, "Temporary pointer must be an odd number."
    return ptr, obj


def fuse_batched_inputs(
    input_batch_axes: tuple[int | None, ...],
    batched_inputs: tuple[jax.Array, ...],
    vmap_axes: tuple[int | None, ...],
    num_state_components: int,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...], tuple[int, ...]]:
    """
    Fuse vmap and batch dimensions in input arguments to a JAX primitive.

    Args:
        input_batch_axes: The batch axes of the input tensors.
        batched_inputs: The input tensors.
        vmap_axes: The vmap axes of the input tensors.
        num_state_components: The number of state components.

    Returns:
        prepared_inputs: The prepared input tensors.
        batch_sizes: The sizes of the batch dimensions.
        vmap_sizes: The sizes of the vmap dimensions.
    """
    assert len(vmap_axes) == len(batched_inputs)
    assert len(input_batch_axes) == len(batched_inputs)
    
    # NOTE: Assuming 0 is the dimension to be accumulated into.
    assert all(batch_axis == 0 for batch_axis in input_batch_axes)

    prepared_inputs = []

    # Prepare inputs: handle batch/vmap axes
    for arr, batch_axis, vmap_axis in zip(batched_inputs, input_batch_axes, vmap_axes):
        # NOTE: Expand to a batch dimension to be accumulated into. This only works for single vmap.
        arr = jnp.expand_dims(arr, axis=batch_axis)
        if vmap_axis is not None:
            # Since batch_axis is hardcoded to 0, the original vmap_axis moves to one position later.
            # The new vmap axis (vmap_axis + 1) is then moved to the position after batch_axis.
            arr = jnp.moveaxis(arr, vmap_axis + 1, batch_axis + 1)
        prepared_inputs.append(arr)

    # Fuse dimensions for primitive call
    fused_inputs = []
    batch_sizes = [None] * num_state_components
    vmap_sizes = [None] * num_state_components
    for i, (arr, batch_axis) in enumerate(zip(prepared_inputs, input_batch_axes)):
        if batch_axis is not None:
            shape = list(arr.shape)
            batch_size = shape[batch_axis]  # 1 or N; primitive handles the difference
            vmap_size = shape[batch_axis + 1]
            if i < num_state_components:
                batch_sizes[i] = batch_size
                vmap_sizes[i] = vmap_size

            shape[batch_axis] = vmap_size * batch_size
            shape.pop(batch_axis + 1)
            arr = arr.reshape(shape)  # (..., batch_size * vmap_size, ...)
        fused_inputs.append(arr)

    return tuple(fused_inputs), tuple(batch_sizes), tuple(vmap_sizes)


def unfuse_batched_outputs(fused_outputs: tuple[jax.Array, ...],
                           output_batch_axes: tuple[int, ...],
                           batch_sizes: tuple[int, ...],
                           vmap_sizes: tuple[int, ...],
                           ) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    """
    Unfuse vmap and batch dimensions in output arguments from a JAX primitive.

    Args:
        fused_outputs: The fused output tensors.
        output_batch_axes: The batch axes of the output tensors.
        batch_sizes: The sizes of the batch dimensions.
        vmap_sizes: The sizes of the vmap dimensions.

    Returns:
        unfused_outputs: The unfused output tensors.
        output_batch_axes: The batch axes of the output tensors.
    """
    assert len(fused_outputs) == len(output_batch_axes)

    # Unfuse output dimensions
    outputs = []
    for i, (out, batch_axis) in enumerate(zip(fused_outputs, output_batch_axes)):
        # Unfuse and move vmap dimension to position 0
        shape = list(out.shape)
        shape[batch_axis : batch_axis + 1] = [batch_sizes[i], vmap_sizes[i]]
        out = out.reshape(shape)
        out = jnp.moveaxis(out, batch_axis, 0)  # move batch dimension to position 0

        # NOTE: Squeeze the original batch dimension. This only works for single vmap.
        out = jnp.squeeze(out, axis=0)
        
        outputs.append(out)

    return outputs
