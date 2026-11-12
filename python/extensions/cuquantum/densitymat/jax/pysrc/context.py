# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
cuDensityMat context classes.
"""

import atexit
import logging

import jax.numpy as jnp
from numpy.typing import DTypeLike

from cuquantum.bindings import cudensitymat as cudm
from nvmath.internal import typemaps

from .operator import Operator


class CudensitymatContext:
    """
    cuDensityMat library context.

    This class holds the library handle and the workspace descriptor handle, which are used for all
    operator actions. A specific operator context and state context are created for each operator,
    stored in _operator_contexts and _state_contexts respectively, and retrieved when the specific
    operator action is invoked.
    """

    _handle = None
    _workspace_desc = None
    _operator_contexts = {}  # key: (operator opaque pointer, batch_size)
    _state_contexts = {}  # key: (purity, tuple(state_shape), batch_size, dtype_name)

    logger = logging.getLogger("cudensitymat-jax.CudensitymatContext")

    @classmethod
    def _maybe_create_handle_and_workspace(cls):
        """
        Create the handle and workspace for the context if they are not already created.
        """
        if cls._handle is None:
            cls.logger.info("Initializing CudensitymatContext")
            cls._handle = cudm.create()
            cls.logger.info(f"Created handle at {hex(cls._handle)}")

            if cls._workspace_desc is None:
                cls._workspace_desc = cudm.create_workspace(cls._handle)
                cls.logger.info(f"Created workspace descriptor at {hex(cls._workspace_desc)}")
            else:  # handle is None but workspace is not None
                raise RuntimeError("Workspace descriptor and handle should be created at the same time")
        else:
            if cls._workspace_desc is None:  # handle is not None but workspace is None
                raise RuntimeError("Workspace descriptor and handle should be created at the same time")

    @classmethod
    def maybe_create_operator_context(cls, op: Operator, batch_size: int = 1) -> None:
        """
        Create the OperatorContext for the operator if it does not already exist.

        Args:
            op: The operator.
            batch_size: The batch size of the operator action (state batch size).
        """
        cls._maybe_create_handle_and_workspace()

        if op._ptr is None or (op._ptr, batch_size) not in cls._operator_contexts:
            if op._ptr is not None:
                # Operators are frozen once used, matching append()'s "Cannot modify
                # operator after it has been used" guard: reusing at a different batch
                # size is a form of post-use mutation and is disallowed the same way.
                raise RuntimeError(
                    "Cannot reuse an operator at a different batch size after it has "
                    "been used in an operator action."
                )
            op_ctx = OperatorContext(op, batch_size)
            # op._ptr is now set (op._create was called inside OperatorContext.__init__)
            cls._operator_contexts[(op._ptr, batch_size)] = op_ctx
            cls.logger.info(f"Created OperatorContext for operator {hex(id(op))} batch_size={batch_size}")

    @classmethod
    def get_operator_context(cls, op_ptr: int, batch_size: int) -> "OperatorContext":
        """
        Get the OperatorContext for a given operator pointer and batch size.
        """
        op_key = (op_ptr, batch_size)
        if op_key not in cls._operator_contexts:
            raise RuntimeError(
                f"No OperatorContext found for operator pointer {hex(op_ptr)} batch_size={batch_size}. "
                "Ensure maybe_create_operator_context() was called before get_operator_context()."
            )
        return cls._operator_contexts[op_key]

    @classmethod
    def maybe_create_state_context(
        cls,
        purity: cudm.StatePurity,
        state_shape: tuple[int, ...],
        batch_size: int,
        dtype: DTypeLike,
    ) -> "StateContext":
        """
        Get or create the StateContext for the given key (purity, state_shape, batch_size, dtype).
        """
        dtype_name = jnp.dtype(dtype).name
        state_key = (purity, tuple(state_shape), batch_size, dtype_name)
        # state_shape always has a leading batch dimension (added by maybe_expand_dim when needed).
        if purity == cudm.StatePurity.MIXED:
            num_modes = (len(state_shape) - 1) // 2
        else:
            num_modes = len(state_shape) - 1
        space_mode_extents = state_shape[-num_modes:]
        if state_key not in cls._state_contexts:
            state_ctx = StateContext(purity, space_mode_extents, batch_size, dtype)
            cls._state_contexts[state_key] = state_ctx
            cls.logger.info(f"Created StateContext for purity={purity} state_shape={state_shape} batch_size={batch_size} dtype={dtype}")
        return cls._state_contexts[state_key]

    @classmethod
    def get_state_context(
        cls,
        purity: cudm.StatePurity,
        state_shape: tuple[int, ...],
        batch_size: int,
        dtype: DTypeLike,
    ) -> "StateContext":
        """
        Get the StateContext for the given purity, state_shape, batch_size and dtype.
        """
        dtype_name = jnp.dtype(dtype).name
        state_key = (purity, tuple(state_shape), batch_size, dtype_name)
        if state_key not in cls._state_contexts:
            raise RuntimeError(
                f"No StateContext found for purity={purity}, state_shape={state_shape}, "
                f"batch_size={batch_size}, dtype={dtype}. "
                "Ensure maybe_create_state_context() was called before get_state_context()."
            )
        return cls._state_contexts[state_key]

    @classmethod
    def free(cls):
        """
        Free opaque handles to the library.

        Note: We don't store operator objects in contexts to avoid leaking JAX tracers.
        Operator objects and their GPU handles will be cleaned up by Python's garbage
        collector when they're no longer referenced. Here we only clean up the state
        handles and workspace that are managed by contexts.
        """
        # During interpreter shutdown Python sets module globals to None before (or
        # concurrently with) running atexit handlers. Capture a local ref once; if it is
        # already None the C extension is being torn down and calling into it would segfault.
        _cudm = cudm
        if _cudm is None:
            return

        cls.logger.info("Freeing CudensitymatContext")

        # Free all state handles from state contexts
        for state_ctx in cls._state_contexts.values():
            state_ctx.free()

        # Release gradient callback function references from operator contexts.
        # Multiple OperatorContext entries can share the same underlying C operator handle
        # (e.g., original and JAX-traced copies of the same op used with different vmap
        # batch sizes both carry the same _ptr integer). Track destroyed handles to avoid
        # calling cudm.destroy_operator on the same pointer twice.
        destroyed_op_ptrs = set()
        for op_ctx in cls._operator_contexts.values():
            if op_ctx._operator not in destroyed_op_ptrs:
                destroyed_op_ptrs.add(op_ctx._operator)
                op_ctx.free()
            else:
                op_ctx._op = None
                op_ctx._callback_fns.clear()

        # Free workspace and library handle
        if cls._workspace_desc is not None:
            _cudm.destroy_workspace(cls._workspace_desc)
            cls._workspace_desc = None

        if cls._handle is not None:
            _cudm.destroy(cls._handle)
            cls._handle = None

        # Clear tracking dictionaries
        CudensitymatContext._operator_contexts.clear()
        CudensitymatContext._state_contexts.clear()


atexit.register(CudensitymatContext.free)


class OperatorContext:
    """
    Operator context.

    Holds the operator-specific C-side handle and metadata needed for operator actions.
    """

    logger = logging.getLogger("cudensitymat-jax.OperatorContext")

    def __init__(self, op: Operator, batch_size: int = 1) -> None:
        """
        Initialize OperatorContext.

        Args:
            op: The operator object for operator action.
            batch_size: The batch size of the operator action (state batch size).
        """
        self.logger.info("Initializing OperatorContext")

        # Derived attributes from operator.
        self._space_mode_extents = op.dims
        # op.dtype is None when no operator term contributed one, which happens for an operator
        # built only from identity/scalar terms: those carry no data and are skipped when
        # inferring the dtype, so that they cannot constrain an operator built from real
        # float32/float64/complex64 operators. Such an operator still needs a concrete compute
        # type, so fall back to the widest one; safe here because there is no operator data whose
        # precision could be demoted.
        dtype = op.dtype if op.dtype is not None else jnp.dtype(jnp.complex128)
        self._data_type = typemaps.NAME_TO_DATA_TYPE[dtype.name]
        self._compute_type = typemaps.NAME_TO_COMPUTE_TYPE[dtype.name]

        # Create opaque handle to the operator.
        op._create(CudensitymatContext._handle, batch_size)
        self._operator = op._ptr
        self._op = op

        # Hold all gradient callback functions (f) to prevent GC while C handles are alive.
        # The traced operator passed to _create may be GC'd after JIT tracing due to async
        # GPU dispatch, so we collect the f objects here where they outlive the traced op.
        # f is accessible via callback.callback on the existing WrappedScalar/TensorGradientCallback objects.
        self._callback_fns = []
        for callback in op._coeff_grad_callbacks:
            if callback is not None:
                self._callback_fns.append(callback.callback)
        for op_term in op.op_terms:
            for callback in op_term._coeff_grad_callbacks:
                if callback is not None:
                    self._callback_fns.append(callback.callback)
            for op_prod in op_term.op_prods:
                for base_op in op_prod:
                    if base_op._grad_callback is not None:
                        self._callback_fns.append(base_op._grad_callback.callback)

    def free(self):
        """
        Destroy the operator C handle and release gradient callback function references.
        """
        self._op._destroy()
        self._op = None
        self._callback_fns.clear()


class StateContext:
    """
    State context.

    Holds the C-side handles for the input/output states (and their adjoints) used in
    operator actions.
    """

    logger = logging.getLogger("cudensitymat-jax.StateContext")

    def __init__(self,
                 purity: cudm.StatePurity,
                 space_mode_extents: tuple,
                 batch_size: int,
                 dtype: DTypeLike,
                 ) -> None:
        self.logger.info("Initializing StateContext")

        self.state_purity = purity
        self.batch_size = batch_size

        # Store for use in create_adjoint_buffers.
        self._space_mode_extents = space_mode_extents
        self._data_type = typemaps.NAME_TO_DATA_TYPE[jnp.dtype(dtype).name]

        # Create opaque handles to the input and output states.
        self._state_in = cudm.create_state(
            CudensitymatContext._handle,
            self.state_purity,
            len(self._space_mode_extents),
            self._space_mode_extents,
            self.batch_size,
            self._data_type
        )
        self.logger.debug(f"Created input state at {hex(self._state_in)}")

        self._state_out = cudm.create_state(
            CudensitymatContext._handle,
            self.state_purity,
            len(self._space_mode_extents),
            self._space_mode_extents,
            self.batch_size,
            self._data_type
        )
        self.logger.debug(f"Created output state at {hex(self._state_out)}")

        # The state adjoints are created by the first backward rule, in create_adjoint_buffers.
        self._state_in_adj = None
        self._state_out_adj = None

    def create_adjoint_buffers(self):
        """
        Create the input and output state adjoints once per config; later backward passes reuse them.

        Like the input and output handles, the adjoint handles are config-only (buffers are attached
        on every call), and traced backward primitives embed their addresses as static parameters.
        They must therefore live as long as this context: destroying them here would leave earlier
        traced or compiled backward actions pointing at freed handles. free() releases them.
        """
        if self._state_in_adj is None:
            self._state_in_adj = cudm.create_state(
                CudensitymatContext._handle,
                self.state_purity,
                len(self._space_mode_extents),
                self._space_mode_extents,
                self.batch_size,
                self._data_type
            )
            self.logger.debug(f"Created input state adjoint at {hex(self._state_in_adj)}")

        if self._state_out_adj is None:
            self._state_out_adj = cudm.create_state(
                CudensitymatContext._handle,
                self.state_purity,
                len(self._space_mode_extents),
                self._space_mode_extents,
                self.batch_size,
                self._data_type
            )
            self.logger.debug(f"Created output state adjoint at {hex(self._state_out_adj)}")

    def free(self):
        """
        Free opaque handles to the library.
        """
        self.logger.info("Freeing StateContext")

        if self._state_out_adj is not None:
            cudm.destroy_state(self._state_out_adj)
            self.logger.debug(f"Destroyed output state adjoint at {hex(self._state_out_adj)}")
            self._state_out_adj = None

        if self._state_in_adj is not None:
            cudm.destroy_state(self._state_in_adj)
            self.logger.debug(f"Destroyed input state adjoint at {hex(self._state_in_adj)}")
            self._state_in_adj = None

        if self._state_out is not None:
            cudm.destroy_state(self._state_out)
            self.logger.debug(f"Destroyed output state at {hex(self._state_out)}")
            self._state_out = None

        if self._state_in is not None:
            cudm.destroy_state(self._state_in)
            self.logger.debug(f"Destroyed input state at {hex(self._state_in)}")
            self._state_in = None
