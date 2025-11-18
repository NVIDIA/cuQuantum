# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
cuQuantum JAX operator action primitive.
"""

import logging
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.interpreters import mlir

from cuquantum.bindings import cudensitymat as cudm

from .base import BasePrimitive, register_primitive
from .context import CudensitymatContext
from .operators import Operator

logger = logging.getLogger("cudensitymat-jax.operator_action")


class OperatorActionPrimitive(BasePrimitive):
    """
    JAX primitive for operator action.
    """

    name = "operator_action"
    inner_multiple_results = True
    outer_multiple_results = True
    inner_primitive = None
    outer_primitive = None
    operator = None

    logger = logging.getLogger(f"cudensitymat-jax.OperatorActionPrimitive")

    @staticmethod
    def abstract(t_aval, params_aval, *other_buf_avals):
        """
        Abstract evaluation of the inner primitive of operator action.
        """
        OperatorActionPrimitive.logger.info(f"Calling abstract evaluation of the inner primitive")

        op_act_ctx = CudensitymatContext.get_context(OperatorActionPrimitive.operator)

        # Create abstract arrays for the output state buffers.
        num_base_ops = len(op_act_ctx.base_op_ptrs)
        state_out_buf_avals = [
            jax.core.ShapedArray(other_buf_avals[i].shape, other_buf_avals[i].dtype)
            for i in range(num_base_ops, len(other_buf_avals))
        ]

        # Obtain workspace limit and stream from the device.
        workspace_limit = (
            op_act_ctx.device.memory_stats()['bytes_limit'] -
            op_act_ctx.device.memory_stats()['bytes_in_use']
        )
        stream = op_act_ctx.device.get_stream_for_external_ready_events()

        # Prepare operator action.
        cudm.operator_prepare_action(
            CudensitymatContext._handle,
            op_act_ctx._operator,
            op_act_ctx._state_in,
            op_act_ctx._state_out,
            op_act_ctx._compute_type,
            workspace_limit,
            CudensitymatContext._workspace_desc,
            stream)

        # Query the required buffer size for the workspace.
        required_buffer_size = cudm.workspace_get_memory_size(
            CudensitymatContext._handle,
            CudensitymatContext._workspace_desc,
            cudm.Memspace.DEVICE,
            cudm.WorkspaceKind.WORKSPACE_SCRATCH)

        if required_buffer_size > op_act_ctx._required_buffer_size:
            op_act_ctx._required_buffer_size = required_buffer_size

        # Create abstract workspace array.
        # NOTE: Memory buffers from cudaMalloc is automatically 256-aligned, which is not 
        # the case for JAX. 255 is added to the buffer size to ensure workspace is 256-aligned.
        workspace_aval = jax.core.ShapedArray((op_act_ctx._required_buffer_size + 255,), jnp.uint8)
        return workspace_aval, *state_out_buf_avals

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        Abstract evaluation of the outer primitive of operator action.
        """
        OperatorActionPrimitive.logger.info(f"Calling abstract evaluation of the outer primitive")
        _, *state_out_buf_avals = OperatorActionPrimitive.abstract(*args, **kwargs)
        return state_out_buf_avals

    @staticmethod
    def lowering(ctx, t, params, *other_bufs):
        """
        Lowering rule of the operator action primitive.
        """
        OperatorActionPrimitive.logger.info(f"Calling lowering rule")

        op_act_ctx = CudensitymatContext.get_context(OperatorActionPrimitive.operator)

        # Revert indices in input and output states. Note the layout is specified as
        # minor-to-major axis order.
        operand_layouts = [
            None,  # t
            None,  # params
        ] + [
            tuple(range(ctx.avals_in[i].ndim)) for i in range(2, len(ctx.avals_in))  # state_in_bufs
        ]
        result_layouts = [
            None,  # workspace
        ] + [
            tuple(range(ctx.avals_out[i].ndim)) for i in range(1, len(ctx.avals_out))  # state_out_bufs
        ]
        outputs = jax.ffi.ffi_lowering(
            OperatorActionPrimitive.name,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts
        )(
            ctx,
            t,
            params,
            *other_bufs,
            base_op_ptrs=mlir.dense_int_elements(op_act_ctx.base_op_ptrs),
            is_elem_op=mlir.dense_int_elements(op_act_ctx.is_elem_op),
            batch_size=op_act_ctx.batch_size,
            handle=CudensitymatContext._handle,
            operator=op_act_ctx._operator,
            state_in=op_act_ctx._state_in,
            state_out=op_act_ctx._state_out,
            workspace_desc=CudensitymatContext._workspace_desc
        )
        return outputs

    @staticmethod
    def impl(t, params, *other_bufs):
        """
        Primal evaluation of the operator action primitive.
        """
        OperatorActionPrimitive.logger.info(f"Calling primal evaluation")

        assert OperatorActionPrimitive.inner_primitive is not None
        _, *state_out_bufs = OperatorActionPrimitive.inner_primitive.bind(t, params, *other_bufs)
        return state_out_bufs


register_primitive(OperatorActionPrimitive)


def operator_action_prim(op: Operator,
                         t: float,
                         state_in_bufs: Tuple[jax.Array, ...],
                         params: jax.Array
                         ) -> List[jax.Array]:
    """
    Function wrapper around OperatorActionPrimitive.
    """
    logger.info(f"Calling operator_action_prim")

    # Extract buffers and pointers from operator leaves.
    base_op_bufs = jax.tree.leaves(op)[::3]

    return OperatorActionPrimitive.outer_primitive.bind(t, params, *base_op_bufs, *state_in_bufs)


class OperatorActionBackwardDiffPrimitive(BasePrimitive):
    """
    JAX primitive for operator action backward differentiation.
    """

    name = "operator_action_backward_diff"
    inner_multiple_results = True
    outer_multiple_results = True
    inner_primitive = None
    outer_primitive = None
    operator = None

    logger = logging.getLogger("cudensitymat-jax.OperatorActionBackwardDiffPrimitive")

    @staticmethod
    def abstract(t_aval, params_aval, *other_buf_avals):
        """
        Abstract evaluation of the inner primitive of operator action backward differentiation.
        """
        OperatorActionBackwardDiffPrimitive.logger.info(f"Calling abstract evaluation of the inner primitive")

        op_act_ctx = CudensitymatContext.get_context(OperatorActionBackwardDiffPrimitive.operator)

        # Obtain number of buffer pointers and number of state components.
        num_base_ops = len(op_act_ctx.base_op_ptrs)
        num_state_components = (len(other_buf_avals) - num_base_ops) // 2

        # Create abstract arrays for the output quantities. In other_avals, 0, ..., num_ptrs - 1
        # are the buffer pointers, and num_ptrs, ..., num_ptrs + num_state_components - 1 are
        # input state components, num_ptrs + num_state_components till the end are output state
        # adjoint components.
        params_grad_aval = jax.core.ShapedArray(params_aval.shape, params_aval.dtype)
        state_in_adj_buf_avals = [
            jax.core.ShapedArray(other_buf_avals[i].shape, other_buf_avals[i].dtype)
            for i in range(num_base_ops, num_base_ops + num_state_components)
        ]

        # Obtain workspace limit and stream from the device.
        workspace_limit = (
            op_act_ctx.device.memory_stats()['bytes_limit'] -
            op_act_ctx.device.memory_stats()['bytes_in_use']
        )
        stream = op_act_ctx.device.get_stream_for_external_ready_events()

        # Prepare operator action backward differentiation.
        cudm.operator_prepare_action_backward_diff(
            CudensitymatContext._handle,
            op_act_ctx._operator,
            op_act_ctx._state_in,
            op_act_ctx._state_out_adj,
            op_act_ctx._compute_type,
            workspace_limit,
            CudensitymatContext._workspace_desc,
            stream)

        # Query the required buffer size for the workspace.
        required_buffer_size = cudm.workspace_get_memory_size(
            CudensitymatContext._handle,
            CudensitymatContext._workspace_desc,
            cudm.Memspace.DEVICE,
            cudm.WorkspaceKind.WORKSPACE_SCRATCH)
        if required_buffer_size > op_act_ctx._required_buffer_size:
            op_act_ctx._required_buffer_size = required_buffer_size

        # Create abstract workspace array.
        # NOTE: Memory buffers from cudaMalloc is automatically 256-aligned, which is not 
        # the case for JAX. 255 is added to the buffer size to ensure workspace is 256-aligned.
        workspace_aval = jax.core.ShapedArray((op_act_ctx._required_buffer_size + 255,), jnp.uint8)
        return workspace_aval, params_grad_aval, *state_in_adj_buf_avals

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        Abstract evaluation of the outer primitive of operator action backward differentiation.
        """
        OperatorActionBackwardDiffPrimitive.logger.info(f"Calling abstract evaluation of the outer primitive")
        _, params_grad_aval, *state_in_adj_buf_avals = OperatorActionBackwardDiffPrimitive.abstract(*args, **kwargs)
        return params_grad_aval, *state_in_adj_buf_avals

    @staticmethod
    def lowering(ctx, t, params, *other_bufs):
        """
        Lowering rule of the operator action backward differentiation primitive.
        """
        OperatorActionBackwardDiffPrimitive.logger.info(f"Calling lowering rule")

        op_act_ctx = CudensitymatContext.get_context(OperatorActionBackwardDiffPrimitive.operator)

        # Revert indices in input and output states. Note the layout is specified as
        # minor-to-major axis order.
        operand_layouts = [
            None,  # t
            None,  # params
        ] + [
            tuple(range(ctx.avals_in[i].ndim)) for i in range(2, len(ctx.avals_in))  # state_out_adj_bufs
        ]
        result_layouts = [
            None,  # workspace
            None,  # params_grad
        ] + [
            tuple(range(ctx.avals_out[i].ndim)) for i in range(2, len(ctx.avals_out))  # state_in_adj_bufs
        ]
        outputs = jax.ffi.ffi_lowering(
            OperatorActionBackwardDiffPrimitive.name,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts
        )(
            ctx,
            t,
            params,
            *other_bufs,
            base_op_ptrs=mlir.dense_int_elements(op_act_ctx.base_op_ptrs),
            is_elem_op=mlir.dense_int_elements(op_act_ctx.is_elem_op),
            batch_size=op_act_ctx.batch_size,
            handle=CudensitymatContext._handle,
            operator=op_act_ctx._operator,
            state_in=op_act_ctx._state_in,
            state_out_adj=op_act_ctx._state_out_adj,
            state_in_adj=op_act_ctx._state_in_adj,
            workspace_desc=CudensitymatContext._workspace_desc
        )
        return outputs

    @staticmethod
    def impl(t, params, *other_bufs):
        """
        Primal evaluation of the operator action backward differentiation primitive.
        """
        OperatorActionBackwardDiffPrimitive.logger.info(f"Calling primal evaluation")
        assert OperatorActionBackwardDiffPrimitive.inner_primitive is not None
        _, params_grad, *state_in_adj_bufs = OperatorActionBackwardDiffPrimitive.inner_primitive.bind(
            t, params, *other_bufs)
        return params_grad, *state_in_adj_bufs


register_primitive(OperatorActionBackwardDiffPrimitive)


def operator_action_backward_diff_prim(op: Operator,
                                       t: float,
                                       state_in_bufs: Tuple[jax.Array, ...],
                                       state_out_adj_bufs: Tuple[jax.Array, ...],
                                       params: jax.Array
                                       ) -> Tuple[jax.Array, ...]:
    """
    Wrapper around the outer primitive of OperatorActionBackwardDiffPrimitive.
    """
    logger.info(f"Calling operator_action_backward_diff_prim")

    # Extract buffers and pointers from operator leaves.
    base_op_bufs = jax.tree.leaves(op)[::3]

    return OperatorActionBackwardDiffPrimitive.outer_primitive.bind(
        t, params, *base_op_bufs, *state_in_bufs, *state_out_adj_bufs)
