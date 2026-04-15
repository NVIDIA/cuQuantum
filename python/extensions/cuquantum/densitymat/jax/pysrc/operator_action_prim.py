# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator action primitive.
"""

import logging

import jax
import jax.numpy as jnp
from jax.interpreters import mlir

from cuquantum.bindings import cudensitymat as cudm

from ..utils import fuse_batched_inputs, unfuse_batched_outputs
from .base import BasePrimitive, register_primitive
from .context import CudensitymatContext
from .operator import Operator


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

    logger = logging.getLogger("cudensitymat-jax.OperatorActionPrimitive")

    @staticmethod
    def abstract(*in_buf_avals: list[jax.core.ShapedArray],
                 device: jax.Device,
                 batch_size: int,
                 num_state_components: int,
                 other_in_types: tuple[int, ...],
                 other_in_ptrs: tuple[int, ...],
                 other_out_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...],
                 other_out_types: tuple[int, ...],
                 other_out_ptrs: tuple[int, ...],
                 op_ptr: int,
                 state_shape: tuple[int, ...],
                 purity: cudm.StatePurity,
                 ) -> tuple[jax.core.ShapedArray, ...]:
        """
        Abstract evaluation of the inner primitive of operator action.
        """
        OperatorActionPrimitive.logger.info("Calling abstract evaluation of the inner primitive")

        dtype = in_buf_avals[0].dtype
        op_ctx = CudensitymatContext.get_operator_context(op_ptr)
        state_ctx = CudensitymatContext.get_state_context(purity, state_shape, batch_size, dtype)

        # Create abstract arrays for the output state buffers.
        state_out_buf_avals = [
            jax.core.ShapedArray(in_buf_avals[i].shape, in_buf_avals[i].dtype)
            for i in range(num_state_components)
        ]

        other_out_buf_avals = [
            jax.core.ShapedArray(other_out_shape_dtypes[i].shape, other_out_shape_dtypes[i].dtype)
            for i in range(len(other_out_shape_dtypes))
        ]

        # Obtain workspace limit and stream from the device.
        workspace_limit = device.memory_stats()['bytes_limit'] - device.memory_stats()['bytes_in_use']
        stream = device.get_stream_for_external_ready_events()

        # Prepare operator action.
        cudm.operator_prepare_action(
            CudensitymatContext._handle,
            op_ctx._operator,
            state_ctx._state_in,
            state_ctx._state_out,
            op_ctx._compute_type,
            workspace_limit,
            CudensitymatContext._workspace_desc,
            stream)

        # Query the required buffer size for the workspace.
        required_buffer_size = cudm.workspace_get_memory_size(
            CudensitymatContext._handle,
            CudensitymatContext._workspace_desc,
            cudm.Memspace.DEVICE,
            cudm.WorkspaceKind.WORKSPACE_SCRATCH)

        # Create abstract workspace array.
        # NOTE: Memory buffers from cudaMalloc is automatically 256-aligned, which is not 
        # the case for JAX. 255 is added to the buffer size to ensure workspace is 256-aligned.
        workspace_aval = jax.core.ShapedArray((required_buffer_size + 255,), jnp.uint8)
        return workspace_aval, *state_out_buf_avals, *other_out_buf_avals

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        Abstract evaluation of the outer primitive of operator action.
        """
        OperatorActionPrimitive.logger.info("Calling abstract evaluation of the outer primitive")
        _, *out = OperatorActionPrimitive.abstract(*args, **kwargs)
        return out

    @staticmethod
    def lowering(ctx: mlir.LoweringRuleContext,
                 *in_bufs: tuple[mlir.ir.OpResult, ...],
                 device: jax.Device,
                 batch_size: int,
                 num_state_components: int,
                 other_in_types: tuple[int, ...],
                 other_in_ptrs: tuple[int, ...],
                 other_out_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...],
                 other_out_types: tuple[int, ...],
                 other_out_ptrs: tuple[int, ...],
                 op_ptr: int,
                 state_shape: tuple[int, ...],
                 purity: cudm.StatePurity,
                 ) -> tuple[mlir.ir.OpResult, ...]:
        """
        Lowering rule of the operator action primitive.
        """
        OperatorActionPrimitive.logger.info("Calling lowering rule")

        dtype = ctx.avals_in[0].dtype
        state_ctx = CudensitymatContext.get_state_context(purity, state_shape, batch_size, dtype)

        # Revert indices in input and output states. Note the layout is specified as
        # minor-to-major axis order. For 0-d tensors use () (StableHLO requires empty layout).
        def _layout_for_ndim(ndim):
            return () if ndim == 0 else tuple(range(1, ndim)) + (0,)

        operand_layouts = [None] * len(ctx.avals_in)
        for i in range(len(ctx.avals_in)):
            # 0 (the batch dimension) is the most major axis in input buffers and also when passed
            # to the cuQuantum library. Other dimensions (Hilbert space modes) need to be reversed.
            operand_layouts[i] = _layout_for_ndim(ctx.avals_in[i].ndim)

        result_layouts = [None] * len(ctx.avals_out)
        for i in range(len(ctx.avals_out)):
            result_layouts[i] = _layout_for_ndim(ctx.avals_out[i].ndim)

        # Lower to the XLA FFI handler.
        outputs = jax.ffi.ffi_lowering(
            OperatorActionPrimitive.name,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts
        )(
            ctx,
            *in_bufs,
            other_in_types=mlir.dense_int_elements(other_in_types),
            other_in_ptrs=mlir.dense_int_elements(other_in_ptrs),
            batch_size=batch_size,
            num_state_components=num_state_components,
            other_out_types=mlir.dense_int_elements(other_out_types),
            other_out_ptrs=mlir.dense_int_elements(other_out_ptrs),
            handle=CudensitymatContext._handle,
            operator=op_ptr,
            state_in=state_ctx._state_in,
            state_out=state_ctx._state_out
        )
        return outputs

    @staticmethod
    def impl(*args, **kwargs):
        """
        Primal evaluation of the operator action primitive.
        """
        OperatorActionPrimitive.logger.info("Calling primal evaluation")

        assert OperatorActionPrimitive.inner_primitive is not None
        _, *out = OperatorActionPrimitive.inner_primitive.bind(*args, **kwargs)
        return out

    @staticmethod
    def batcher(batched_args, batch_dims, **kwargs):
        """
        Batching rule of the operator action primitive.
        """
        OperatorActionPrimitive.logger.info("Calling batcher")

        num_state_components = kwargs['num_state_components']

        # Fuse pivot batch axis with vmap axis.
        fused_inputs, batch_sizes, vmap_sizes = fuse_batched_inputs(
            (0,) * len(batched_args),
            batched_args,
            batch_dims,
            num_state_components,
        )

        # State context key uses fused state shape and batch size.
        kwargs = dict(kwargs)
        kwargs['state_shape'] = tuple(fused_inputs[0].shape)
        kwargs['batch_size'] = int(fused_inputs[0].shape[0])

        # Invoke outer primitive.
        outputs = OperatorActionPrimitive.outer_primitive.bind(*fused_inputs, **kwargs)

        # Unfuse pivot batch axis and vmap axis.
        outputs[:num_state_components] = unfuse_batched_outputs(
            outputs[:num_state_components],
            (0,) * num_state_components,
            batch_sizes,
            vmap_sizes,
        )
        return outputs, (0,) * len(outputs)

register_primitive(OperatorActionPrimitive)


def operator_action_prim(op: Operator,
                         state_in_bufs: tuple[jax.Array, ...],
                         device: jax.Device,
                         batch_size: int,
                         num_state_components: int,
                         purity: cudm.StatePurity,
                         other_in_types: tuple[int, ...],
                         other_in_ptrs: tuple[int, ...],
                         other_out_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...],
                         other_out_types: tuple[int, ...],
                         other_out_ptrs: tuple[int, ...],
                         op_term_coeffs_indices: tuple[int, ...],
                         op_prod_coeffs_indices: tuple[tuple[int, int], ...],
                         base_op_indices: tuple[tuple[int, int, int], ...],
                         ) -> tuple[jax.Array, ...]:
    """
    Function wrapper around OperatorActionPrimitive.
    """
    logger.info("Calling operator_action_prim")

    other_in_bufs = []

    # Extract buffers using the same index structure as operator_action.
    for i in op_term_coeffs_indices:
        other_in_bufs.append(op.coeffs[i])

    for i, j in op_prod_coeffs_indices:
        other_in_bufs.append(op[i].coeffs[j])

    for i, j, k in base_op_indices:
        other_in_bufs.append(op[i][j][k].data)

    state_shape = tuple(state_in_bufs[0].shape)
    out = OperatorActionPrimitive.outer_primitive.bind(
        *state_in_bufs,
        *other_in_bufs,
        device=device,
        batch_size=batch_size,
        num_state_components=num_state_components,
        other_in_types=tuple(other_in_types),
        other_in_ptrs=tuple(other_in_ptrs),
        other_out_shape_dtypes=other_out_shape_dtypes,
        other_out_types=other_out_types,
        other_out_ptrs=other_out_ptrs,
        op_ptr=op._ptr,
        state_shape=state_shape,
        purity=purity,
    )

    state_out_bufs = out[:num_state_components]
    return state_out_bufs


class OperatorActionBackwardDiffPrimitive(BasePrimitive):
    """
    JAX primitive for operator action backward differentiation.
    """

    name = "operator_action_backward_diff"
    inner_multiple_results = True
    outer_multiple_results = True
    inner_primitive = None
    outer_primitive = None

    logger = logging.getLogger("cudensitymat-jax.OperatorActionBackwardDiffPrimitive")

    @staticmethod
    def abstract(*in_buf_avals: list[jax.core.ShapedArray],
                 device: jax.Device,
                 batch_size: int,
                 num_state_components: int,
                 other_in_types: tuple[int, ...],
                 other_in_ptrs: tuple[int, ...],
                 other_out_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...],
                 other_out_types: tuple[int, ...],
                 other_out_ptrs: tuple[int, ...],
                 op_ptr: int,
                 state_shape: tuple[int, ...],
                 purity: cudm.StatePurity,
                 ) -> tuple[jax.core.ShapedArray, ...]:
        """
        Abstract evaluation of the inner primitive of operator action backward differentiation.
        """
        OperatorActionBackwardDiffPrimitive.logger.info("Calling abstract evaluation of the inner primitive")

        dtype = in_buf_avals[0].dtype
        op_ctx = CudensitymatContext.get_operator_context(op_ptr)
        state_ctx = CudensitymatContext.get_state_context(purity, state_shape, batch_size, dtype)

        # Extract state input adjoint buffer shapes from state input buffers.
        state_in_adj_buf_avals = [
            jax.core.ShapedArray(in_buf_avals[i].shape, in_buf_avals[i].dtype)
            for i in range(num_state_components)
        ]

        other_out_buf_avals = [
            jax.core.ShapedArray(other_out_shape_dtypes[i].shape, other_out_shape_dtypes[i].dtype)
            for i in range(len(other_out_shape_dtypes))
        ]

        # Obtain workspace limit and stream from the device.
        workspace_limit = device.memory_stats()['bytes_limit'] - device.memory_stats()['bytes_in_use']
        stream = device.get_stream_for_external_ready_events()

        # Prepare operator action backward differentiation.
        cudm.operator_prepare_action_backward_diff(
            CudensitymatContext._handle,
            op_ctx._operator,
            state_ctx._state_in,
            state_ctx._state_out_adj,
            op_ctx._compute_type,
            workspace_limit,
            CudensitymatContext._workspace_desc,
            stream)

        # Query the required buffer size for the workspace.
        required_buffer_size = cudm.workspace_get_memory_size(
            CudensitymatContext._handle,
            CudensitymatContext._workspace_desc,
            cudm.Memspace.DEVICE,
            cudm.WorkspaceKind.WORKSPACE_SCRATCH)

        # Create abstract workspace array.
        # NOTE: Memory buffers from cudaMalloc is automatically 256-aligned, which is not 
        # the case for JAX. 255 is added to the buffer size to ensure workspace is 256-aligned.
        workspace_aval = jax.core.ShapedArray((required_buffer_size + 255,), jnp.uint8)
        return workspace_aval, *state_in_adj_buf_avals, *other_out_buf_avals

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        Abstract evaluation of the outer primitive of operator action backward differentiation.
        """
        OperatorActionBackwardDiffPrimitive.logger.info("Calling abstract evaluation of the outer primitive")
        _, *out = OperatorActionBackwardDiffPrimitive.abstract(*args, **kwargs)
        return out

    @staticmethod
    def lowering(ctx: mlir.LoweringRuleContext,
                 *in_bufs: tuple[mlir.ir.OpResult, ...],
                 device: jax.Device,
                 batch_size: int,
                 num_state_components: int,
                 other_in_types: tuple[int, ...],
                 other_in_ptrs: tuple[int, ...],
                 other_out_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...],
                 other_out_types: tuple[int, ...],
                 other_out_ptrs: tuple[int, ...],
                 op_ptr: int,
                 state_shape: tuple[int, ...],
                 purity: cudm.StatePurity,
                 ) -> tuple[mlir.ir.OpResult, ...]:
        """
        Lowering rule of the operator action backward differentiation primitive.
        """
        OperatorActionBackwardDiffPrimitive.logger.info("Calling lowering rule")

        dtype = ctx.avals_in[0].dtype
        state_ctx = CudensitymatContext.get_state_context(purity, state_shape, batch_size, dtype)

        # Revert indices in input and output states. Note the layout is specified as
        # minor-to-major axis order. For 0-d tensors use () (StableHLO requires empty layout).
        def _layout_for_ndim(ndim):
            return () if ndim == 0 else tuple(range(1, ndim)) + (0,)

        operand_layouts = [None] * len(ctx.avals_in)
        for i in range(len(ctx.avals_in)):
            # 0 (the batch dimension) is the most major axis in input buffers and also when passed
            # to the cuQuantum library. Other dimensions (Hilbert space modes) need to be reversed.
            operand_layouts[i] = _layout_for_ndim(ctx.avals_in[i].ndim)

        result_layouts = [None] * len(ctx.avals_out)
        # for i in range(len(ctx.avals_out)):
        for i in range(num_state_components):  # XXX
            result_layouts[i] = _layout_for_ndim(ctx.avals_out[i].ndim)

        # Lower to the XLA FFI handler.
        outputs = jax.ffi.ffi_lowering(
            OperatorActionBackwardDiffPrimitive.name,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts
        )(
            ctx,
            *in_bufs,
            other_in_types=mlir.dense_int_elements(other_in_types),
            other_in_ptrs=mlir.dense_int_elements(other_in_ptrs),
            batch_size=batch_size,
            num_state_components=num_state_components,
            other_out_types=mlir.dense_int_elements(other_out_types),
            other_out_ptrs=mlir.dense_int_elements(other_out_ptrs),
            handle=CudensitymatContext._handle,
            operator=op_ptr,
            state_in=state_ctx._state_in,
            state_out_adj=state_ctx._state_out_adj,
            state_in_adj=state_ctx._state_in_adj
        )
        return outputs

    @staticmethod
    def impl(*args, **kwargs):
        """
        Primal evaluation of the operator action backward differentiation primitive.
        """
        OperatorActionBackwardDiffPrimitive.logger.info("Calling primal evaluation")

        assert OperatorActionBackwardDiffPrimitive.inner_primitive is not None
        _, *out = OperatorActionBackwardDiffPrimitive.inner_primitive.bind(*args, **kwargs)
        return out

    @staticmethod
    def batcher(batched_args, batch_dims, **kwargs):
        """
        Batching rule of the operator action backward differentiation primitive.
        """
        OperatorActionBackwardDiffPrimitive.logger.info("Calling batcher")

        num_state_components = kwargs['num_state_components']

        # Fuse pivot batch axis with vmap axis.
        fused_inputs, batch_sizes, vmap_sizes = fuse_batched_inputs(
            (0,) * len(batched_args),
            batched_args,
            batch_dims,
            num_state_components,
        )

        kwargs = dict(kwargs)
        kwargs['state_shape'] = tuple(fused_inputs[0].shape)
        kwargs['batch_size'] = int(fused_inputs[0].shape[0])

        # Invoke outer primitive.
        outputs = OperatorActionBackwardDiffPrimitive.outer_primitive.bind(
            *fused_inputs, **kwargs)

        # Unfuse pivot batch axis and vmap axis for state adjoint outputs only.
        outputs[:num_state_components] = unfuse_batched_outputs(
            outputs[:num_state_components],
            (0,) * num_state_components,
            batch_sizes,
            vmap_sizes,
        )
        return outputs, (0,) * len(outputs)

register_primitive(OperatorActionBackwardDiffPrimitive)


def operator_action_backward_diff_prim(op: Operator,
                                       state_in_bufs: tuple[jax.Array, ...],
                                       state_out_adj_bufs: tuple[jax.Array, ...],
                                       device: jax.Device,
                                       batch_size: int,
                                       num_state_components: int,
                                       state_shape: tuple[int, ...],
                                       purity: cudm.StatePurity,
                                       other_in_types: tuple[int, ...],
                                       other_in_ptrs: tuple[int, ...],
                                       other_out_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...],
                                       other_out_types: tuple[int, ...],
                                       other_out_ptrs: tuple[int, ...],
                                       op_term_coeffs_indices: tuple[int, ...],
                                       op_prod_coeffs_indices: tuple[tuple[int, int], ...],
                                       base_op_indices: tuple[tuple[int, int, int], ...],
                                       op_term_coeff_grad_indices: tuple[int, ...],
                                       op_prod_coeff_grad_indices: tuple[tuple[int, int], ...],
                                       base_op_grad_indices: tuple[tuple[int, int, int], ...],
                                       ) -> tuple[jax.Array, ...]:
    """
    Wrapper around the outer primitive of OperatorActionBackwardDiffPrimitive.
    """
    logger.info("Calling operator_action_backward_diff_prim")

    other_in_bufs = []

    # Extract buffers using the same index structure as operator_action.
    for i in op_term_coeffs_indices:
        other_in_bufs.append(op.coeffs[i])

    for i, j in op_prod_coeffs_indices:
        other_in_bufs.append(op[i].coeffs[j])

    for i, j, k in base_op_indices:
        other_in_bufs.append(op[i][j][k].data)

    out = OperatorActionBackwardDiffPrimitive.outer_primitive.bind(
        *state_in_bufs,
        *state_out_adj_bufs,
        *other_in_bufs,
        device=device,
        batch_size=batch_size,
        num_state_components=num_state_components,
        state_shape=tuple(state_shape),
        purity=purity,
        other_in_types=other_in_types,
        other_in_ptrs=other_in_ptrs,
        other_out_shape_dtypes=other_out_shape_dtypes,
        other_out_types=other_out_types,
        other_out_ptrs=other_out_ptrs,
        op_ptr=op._ptr,
    )

    state_in_adj_bufs = out[:num_state_components]
    grad_bufs = out[num_state_components:]

    # In _operator_action_bwd, after getting grad_bufs:
    op_grad = op._copy()
    grad_idx = 0

    # Zero out all base operators that require gradient. The backend accumulates the
    # total gradient into a single buffer per unique base operator (_grad_ptr). We only
    # create one output buffer per unique _grad_ptr (deduplicated in operator_action.py),
    # so the first occurrence gets the correct total gradient and all other (duplicate)
    # occurrences must be zero so JAX doesn't overcount when it sums pytree leaves.
    for i, op_term in enumerate(op_grad.op_terms):
        op_grad.coeffs[i] = jnp.zeros_like(op_grad.coeffs[i])
        for j, op_prod in enumerate(op_term.op_prods):
            op_grad[i].coeffs[j] = jnp.zeros_like(op_grad[i].coeffs[j])
            for k, base_op in enumerate(op_prod):
                if base_op.requires_grad:
                    op_grad[i][j][k].data = jnp.zeros_like(op_grad[i][j][k].data)

    for i in op_term_coeff_grad_indices:
        op_grad.coeffs[i] = grad_bufs[grad_idx]
        grad_idx += 1

    for i, j in op_prod_coeff_grad_indices:
        op_grad[i].coeffs[j] = grad_bufs[grad_idx]
        grad_idx += 1

    for i, j, k in base_op_grad_indices:
        op_grad[i][j][k].data = grad_bufs[grad_idx]
        grad_idx += 1

    return op_grad, state_in_adj_bufs
