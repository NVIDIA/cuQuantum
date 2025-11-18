# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test the cuDensityMat context manager.
"""

import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import ElementaryOperator, OperatorTerm, Operator
from cuquantum.densitymat.jax.pysrc.context import CudensitymatContext, OperatorActionContext

def generate_operator(dims, dtype):
    """
    Create a random operator for the tests.
    """
    data = jax.random.normal(jax.random.key(0), (dims[0], dims[0]), dtype=dtype)
    elem_op = ElementaryOperator(data)

    op_term = OperatorTerm(dims)
    op_term.append([elem_op], modes=(0,))

    op = Operator(dims)
    op.append(op_term)
    return op


class TestCudensitymatContext:
    """
    Test the CudensitymatContext class.
    """

    def test_maybe_create_context(self):
        """
        Test the create_context method of CudensitymatContext.
        """
        assert CudensitymatContext._handle is None
        assert CudensitymatContext._workspace_desc is None
        assert CudensitymatContext._operators == set()
        assert CudensitymatContext._contexts == {}

        op = generate_operator((3, 4, 5), jnp.float32)
        CudensitymatContext.maybe_create_context(op, jax.devices('gpu')[0], 1, cudm.StatePurity.MIXED)

        # After creating the context, the attributes should be set.
        assert CudensitymatContext._handle is not None
        assert CudensitymatContext._workspace_desc is not None
        assert CudensitymatContext._operators == {op}
        assert CudensitymatContext._contexts[op._ptr] is not None
        ctx = CudensitymatContext._contexts[op._ptr]

        # Creating the context on the same operator again has no effect.
        CudensitymatContext.maybe_create_context(op, jax.devices('gpu')[0], 1, cudm.StatePurity.MIXED)
        assert CudensitymatContext._operators == {op}
        assert CudensitymatContext._contexts[op._ptr] is ctx

        # Creating the context on a different operator creates a new context.
        op_ = generate_operator((3, 4, 5), jnp.float64)
        CudensitymatContext.maybe_create_context(op_, jax.devices('gpu')[0], 1, cudm.StatePurity.MIXED)
        assert CudensitymatContext._operators == {op, op_}
        assert CudensitymatContext._contexts[op_._ptr] is not None

    def test_get_context(self):
        """
        Test the get_context method of CudensitymatContext.
        """
        op = generate_operator((3, 4, 5), jnp.float32)
        op_ = generate_operator((3, 4, 5), jnp.float32)
        CudensitymatContext.maybe_create_context(op, jax.devices('gpu')[0], 1, cudm.StatePurity.MIXED)

        CudensitymatContext.get_context(op)

        with pytest.raises(KeyError):
            CudensitymatContext.get_context(op_)

    def test_free(self):
        """
        Test the free method of CudensitymatContext.
        """
        op = generate_operator((3, 4, 5), jnp.float32)
        CudensitymatContext.maybe_create_context(op, jax.devices('gpu')[0], 1, cudm.StatePurity.MIXED)

        for ctx in CudensitymatContext._contexts.values():
            assert ctx._operator is not None
            assert ctx._state_in is not None
            assert ctx._state_out is not None
            assert ctx._state_in_adj is None
            assert ctx._state_out_adj is None

        CudensitymatContext.free()

        assert CudensitymatContext._handle is None
        assert CudensitymatContext._workspace_desc is None
        for ctx in CudensitymatContext._contexts.values():
            assert ctx._operator is None
            assert ctx._state_in is None
            assert ctx._state_out is None
            assert ctx._state_in_adj is None
            assert ctx._state_out_adj is None


class TestOperatorActionContext:
    """
    Test the OperatorActionContext class.
    """

    def test_init_regular(self):
        """
        Test OperatorActionContext.__init__ in regular workflow.
        """
        op = generate_operator((3, 4, 5), jnp.float32)
        CudensitymatContext._maybe_create_handle_and_workspace()

        ctx = OperatorActionContext(op, jax.devices('gpu')[0], 1, cudm.StatePurity.MIXED)
        assert ctx.operator == op
        assert ctx.device == jax.devices('gpu')[0] # TODO: Test with multiple GPUs.
        assert ctx.batch_size == 1
        assert ctx.state_purity == cudm.StatePurity.MIXED

        assert isinstance(ctx._operator, int)
        assert isinstance(ctx._state_in, int)
        assert isinstance(ctx._state_out, int)
        assert ctx._state_in_adj is None
        assert ctx._state_out_adj is None

    def test_create_adjoint_buffers(self):
        """
        Test OperatorActionContext.create_adjoint_buffers.
        """
        op = generate_operator((3, 4, 5), jnp.float32)
        CudensitymatContext._maybe_create_handle_and_workspace()

        op_act_ctx = OperatorActionContext(op, jax.devices('gpu')[0], 1, cudm.StatePurity.MIXED)

        assert op_act_ctx._state_in_adj is None
        assert op_act_ctx._state_out_adj is None

        op_act_ctx.create_adjoint_buffers()

        assert isinstance(op_act_ctx._state_in_adj, int)
        assert isinstance(op_act_ctx._state_out_adj, int)

    def test_free(self):
        """
        Test OperatorActionContext.free.
        """
        op = generate_operator((3, 4, 5), jnp.float32)
        CudensitymatContext._maybe_create_handle_and_workspace()

        op_act_ctx = OperatorActionContext(op, jax.devices('gpu')[0], 1, cudm.StatePurity.MIXED)
        op_act_ctx.create_adjoint_buffers()

        op_act_ctx.free()

        assert op_act_ctx._state_in is None
        assert op_act_ctx._state_out is None
        assert op_act_ctx._state_in_adj is None
        assert op_act_ctx._state_out_adj is None
