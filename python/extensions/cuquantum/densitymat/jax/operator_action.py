# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Sequence, List, Tuple

import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm

from .pysrc.context import CudensitymatContext
from .pysrc.operators import Operator
from .pysrc.operator_action import (
    operator_action_prim,
    operator_action_backward_diff_prim,
    OperatorActionPrimitive,
    OperatorActionBackwardDiffPrimitive
)

logger = logging.getLogger("cudensitymat-jax.operator_action")


def operator_action(op: Operator,
                    t: float,
                    state_in_bufs: jax.Array | Sequence[jax.Array],
                    params: jax.Array | None = None,
                    device: jax.Device | None = None,
                    batch_size: int = 1,
                    ) -> jax.Array | List[jax.Array]:
    """
    Compute the action of an operator on a state.

    Args:
        op: Operator to compute the action of.
        t: The time to compute the operator action at.
        state_in_bufs: Buffers of the input state components.
        params: Callback parameters used to construct the operator.
        device: Device to use for the operator action.
        batch_size: Batch size of the operator action.

    Returns:
        Buffers of the output state components.
    """
    logger.info(f"Calling operator_action")

    # Check if GPU is available and set to GPU 0.
    if device is None:
        gpu_devices = jax.devices('gpu')
        if gpu_devices == []:
            raise RuntimeError("No GPU devices found.")
        device = gpu_devices[0]
        logger.info("Running a single-GPU version of the library and setting device to GPU 0.")

    # Check batch size and state purity.
    if batch_size != 1:
        raise NotImplementedError("Batch size not equal to 1 is not supported yet.")
    
    # Process input arguments.
    if isinstance(state_in_bufs, jax.Array):
        state_in_bufs = (state_in_bufs,)
    else:
        state_in_bufs = tuple(state_in_bufs)

    # Check all states are of the same shape.
    state_in_shape = state_in_bufs[0].shape
    for buf in state_in_bufs[1:]:
        if buf.shape != state_in_shape:
            raise ValueError("All input state buffers must have the same shape.")

    # Determine the state purity.
    if state_in_shape == op.dims:  # state vector
        purity = cudm.StatePurity.PURE
    elif state_in_shape == (*op.dims, *op.dims):  # density matrix
        purity = cudm.StatePurity.MIXED
    else:
        raise ValueError("The dimensions of the input state do not match the dimensions of the operator.")

    # Prepare library context for forward operator action.
    OperatorActionPrimitive.operator = op
    CudensitymatContext.maybe_create_context(op, device, batch_size, purity)

    op_act_ctx = CudensitymatContext.get_context(op)
    leaves = jax.tree.leaves(op)
    op_act_ctx.base_op_ptrs = leaves[1::3]
    op_act_ctx.is_elem_op = leaves[2::3]

    if params is None or params.shape == (0,):
        # Empty params causes a problem when reconstructing params from pointer 
        # in the bindings. We're providing a value here to guard against that.
        params = jnp.array([0.0])
    else:
        if params.dtype != jnp.float64:
            raise ValueError("params must be a float64 array")

    # Invoke operator action.
    state_out_bufs = _operator_action(op, t, state_in_bufs, params)

    # Process output argument.
    if len(state_out_bufs) == 1:
        state_out_bufs = state_out_bufs[0]

    return state_out_bufs


@jax.custom_vjp
def _operator_action(op: Operator,
                     t: float,
                     state_in_bufs: Tuple[jax.Array, ...],
                     params: jax.Array
                     ) -> List[jax.Array]:
    """
    Custom VJP rule for operator_action.
    """
    logger.info(f"Calling _operator_action")
    state_out_bufs, _ = _operator_action_fwd(op, t, state_in_bufs, params)
    return state_out_bufs


def _operator_action_fwd(op: Operator,
                         t: float,
                         state_in_bufs: Tuple[jax.Array, ...],
                         params: jax.Array
                         ) -> Tuple[List[jax.Array], tuple]:
    """
    Forward rule for operator_action.
    """
    logger.info(f"Calling _operator_action_fwd")
    state_out_bufs = operator_action_prim(op, t, state_in_bufs, params)
    return state_out_bufs, (op, t, state_in_bufs, params)


def _operator_action_bwd(res: tuple, state_out_adj_bufs: jax.Array | Sequence[jax.Array]) -> tuple:
    """
    Backward rule for operator_action.

    Args:
        state_out_adj_bufs: Data buffers of the output state adjoint.
    """
    logger.info(f"Calling _operator_action_bwd")

    op, t, state_in_bufs, params = res

    # Prepare library context for backward operator action.
    OperatorActionBackwardDiffPrimitive.operator = op
    op_act_ctx = CudensitymatContext.get_context(op)
    op_act_ctx.create_adjoint_buffers()

    # Process input argument.
    if isinstance(state_out_adj_bufs, jax.Array):
        state_out_adj_bufs = (state_out_adj_bufs,)
    else:
        state_out_adj_bufs = tuple(state_out_adj_bufs)

    if len(state_in_bufs) != len(state_out_adj_bufs):
        raise ValueError("state_in_bufs and state_out_adj_bufs must have the same number of components.")

    params_grad, *state_in_adj_bufs = operator_action_backward_diff_prim(op, t, state_in_bufs, state_out_adj_bufs, params)

    # Process output argument.
    if len(state_in_adj_bufs) == 1:
        state_in_adj_bufs = state_in_adj_bufs[0]

    # TODO: Make "scalar_grad" and "tensor_grad" attributes configurable.
    def get_grad(grad_callback, attr_name, zeros_shape):
        if grad_callback is not None and hasattr(grad_callback.callback, attr_name):
            return getattr(grad_callback.callback, attr_name)
        else:
            return jnp.zeros(zeros_shape)

    for i, (op_term, coeff_grad_callback) in enumerate(zip(op.op_terms, op.coeff_grad_callbacks)):
        op.coeffs[i] = get_grad(coeff_grad_callback, "scalar_grad", jnp.array(op.coeffs[i]).shape)
        for j, (op_prod, coeff_grad_callback) in enumerate(zip(op_term.op_prods, op_term.coeff_grad_callbacks)):
            op_term.coeffs[j] = get_grad(coeff_grad_callback, "scalar_grad", jnp.array(op_term.coeffs[j]).shape)
            for elem_op in op_prod:
                elem_op.data = get_grad(elem_op.grad_callback, "tensor_grad", elem_op.data.shape)

    return op, t, state_in_adj_bufs, params_grad


_operator_action.defvjp(_operator_action_fwd, _operator_action_bwd)
