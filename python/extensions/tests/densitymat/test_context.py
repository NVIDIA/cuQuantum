# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for context manager.
"""

import subprocess
import sys
import textwrap

import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import ElementaryOperator, OperatorTerm, Operator
from cuquantum.densitymat.jax.pysrc.context import CudensitymatContext, OperatorContext, StateContext


def test_lazy_loading():
    """
    Test that cuDensityMat-JAX defers loading its native FFI extension until first use.

    Runs in a subprocess so that `sys.modules` starts clean, regardless of what other
    tests in the same pytest session may have already imported.
    """
    script = textwrap.dedent("""
        import sys
        import cuquantum.densitymat.jax
        assert "cuquantum.lib.cudensitymat_jax" not in sys.modules

        from cuquantum.densitymat.jax.operator_action import _register_ffi_targets
        _register_ffi_targets()
        assert "cuquantum.lib.cudensitymat_jax" in sys.modules
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


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

    def test_maybe_create_operator_context(self):
        """
        Test the maybe_create_operator_context method of CudensitymatContext.
        """
        assert CudensitymatContext._handle is None
        assert CudensitymatContext._workspace_desc is None
        assert CudensitymatContext._operator_contexts == {}

        op = generate_operator((3, 4, 5), jnp.float32)
        CudensitymatContext.maybe_create_operator_context(op)

        # After creating the context, the attributes should be set.
        assert CudensitymatContext._handle is not None
        assert CudensitymatContext._workspace_desc is not None
        assert (op._ptr, 1) in CudensitymatContext._operator_contexts
        op_ctx = CudensitymatContext._operator_contexts[(op._ptr, 1)]

        # Creating the context on the same operator again has no effect.
        CudensitymatContext.maybe_create_operator_context(op)
        assert CudensitymatContext._operator_contexts[(op._ptr, 1)] is op_ctx

        # Creating the context on a different operator creates a new context.
        op_ = generate_operator((3, 4, 5), jnp.float64)
        CudensitymatContext.maybe_create_operator_context(op_)
        assert (op_._ptr, 1) in CudensitymatContext._operator_contexts
        assert CudensitymatContext._operator_contexts[(op_._ptr, 1)] is not op_ctx

    def test_maybe_create_state_context(self):
        """
        Test the maybe_create_state_context method of CudensitymatContext.
        """
        assert CudensitymatContext._state_contexts == {}

        state_shape = (1, 3, 4, 5, 3, 4, 5)
        batch_size = 1
        dtype = jnp.float32
        purity = cudm.StatePurity.MIXED

        state_ctx = CudensitymatContext.maybe_create_state_context(purity, state_shape, batch_size, dtype)

        state_key = (purity, state_shape, batch_size, jnp.dtype(dtype).name)
        assert state_key in CudensitymatContext._state_contexts
        assert CudensitymatContext._state_contexts[state_key] is state_ctx

        # Creating with the same key again returns the same context.
        state_ctx2 = CudensitymatContext.maybe_create_state_context(purity, state_shape, batch_size, dtype)
        assert state_ctx2 is state_ctx

        # Creating with a different key creates a new context.
        state_ctx3 = CudensitymatContext.maybe_create_state_context(purity, (2, 3, 4, 5, 3, 4, 5), 2, dtype)
        assert state_ctx3 is not state_ctx

    def test_get_operator_context(self):
        """
        Test the get_operator_context method of CudensitymatContext.
        """
        op = generate_operator((3, 4, 5), jnp.float32)
        CudensitymatContext.maybe_create_operator_context(op)

        CudensitymatContext.get_operator_context(op._ptr, 1)

        with pytest.raises(RuntimeError):
            CudensitymatContext.get_operator_context(-1, 1)

    def test_get_state_context(self):
        """
        Test the get_state_context method of CudensitymatContext.
        """
        state_shape = (1, 3, 4, 5, 3, 4, 5)
        batch_size = 1
        dtype = jnp.float32
        purity = cudm.StatePurity.MIXED

        CudensitymatContext.maybe_create_state_context(purity, state_shape, batch_size, dtype)
        CudensitymatContext.get_state_context(purity, state_shape, batch_size, dtype)

        with pytest.raises(RuntimeError):
            CudensitymatContext.get_state_context(cudm.StatePurity.PURE, state_shape, batch_size, dtype)

    def test_free(self):
        """
        Test the free method of CudensitymatContext.
        """
        op = generate_operator((3, 4, 5), jnp.float32)
        state_shape = (1, 3, 4, 5, 3, 4, 5)
        batch_size = 1
        dtype = jnp.float32
        purity = cudm.StatePurity.MIXED

        CudensitymatContext.maybe_create_operator_context(op)
        state_ctx = CudensitymatContext.maybe_create_state_context(purity, state_shape, batch_size, dtype)

        assert state_ctx._state_in is not None
        assert state_ctx._state_out is not None
        assert state_ctx._state_in_adj is None
        assert state_ctx._state_out_adj is None

        CudensitymatContext.free()

        assert CudensitymatContext._handle is None
        assert CudensitymatContext._workspace_desc is None
        assert CudensitymatContext._operator_contexts == {}
        assert CudensitymatContext._state_contexts == {}
        assert state_ctx._state_in is None
        assert state_ctx._state_out is None
        assert state_ctx._state_in_adj is None
        assert state_ctx._state_out_adj is None


class TestOperatorContext:
    """
    Test the OperatorContext class.
    """

    def test_init(self):
        """
        Test OperatorContext.__init__.
        """
        op = generate_operator((3, 4, 5), jnp.float32)
        CudensitymatContext._maybe_create_handle_and_workspace()

        op_ctx = OperatorContext(op)
        assert isinstance(op_ctx._operator, int)
        assert op_ctx._space_mode_extents == op.dims
        assert op_ctx._data_type is not None
        assert op_ctx._compute_type is not None


class TestStateContext:
    """
    Test the StateContext class.
    """

    def test_init(self):
        """
        Test StateContext.__init__.
        """
        CudensitymatContext._maybe_create_handle_and_workspace()

        state_ctx = StateContext(cudm.StatePurity.MIXED, (3, 4, 5), 1, jnp.float32)
        assert state_ctx.state_purity == cudm.StatePurity.MIXED
        assert state_ctx.batch_size == 1
        assert isinstance(state_ctx._state_in, int)
        assert isinstance(state_ctx._state_out, int)
        assert state_ctx._state_in_adj is None
        assert state_ctx._state_out_adj is None

    def test_create_adjoint_buffers(self):
        """
        Test StateContext.create_adjoint_buffers.
        """
        CudensitymatContext._maybe_create_handle_and_workspace()

        state_ctx = StateContext(cudm.StatePurity.MIXED, (3, 4, 5), 1, jnp.float32)

        assert state_ctx._state_in_adj is None
        assert state_ctx._state_out_adj is None

        state_ctx.create_adjoint_buffers()

        assert isinstance(state_ctx._state_in_adj, int)
        assert isinstance(state_ctx._state_out_adj, int)

        # A second backward pass must reuse the same handles: traced backward primitives embed
        # their addresses, so destroying them would leave those primitives pointing at freed handles.
        in_ptr, out_ptr = state_ctx._state_in_adj, state_ctx._state_out_adj
        state_ctx.create_adjoint_buffers()

        assert state_ctx._state_in_adj == in_ptr
        assert state_ctx._state_out_adj == out_ptr

    def test_free(self):
        """
        Test StateContext.free.
        """
        CudensitymatContext._maybe_create_handle_and_workspace()

        state_ctx = StateContext(cudm.StatePurity.MIXED, (3, 4, 5), 1, jnp.float32)
        state_ctx.create_adjoint_buffers()

        state_ctx.free()

        assert state_ctx._state_in is None
        assert state_ctx._state_out is None
        assert state_ctx._state_in_adj is None
        assert state_ctx._state_out_adj is None
