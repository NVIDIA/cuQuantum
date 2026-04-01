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
    _operator_contexts = {}  # key: operator opaque pointer
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
    def maybe_create_operator_context(cls, op: Operator) -> None:
        """
        Create the OperatorContext for the operator if it does not already exist.

        Args:
            op: The operator.
        """
        cls._maybe_create_handle_and_workspace()

        if op._ptr is None or op._ptr not in cls._operator_contexts:
            op_ctx = OperatorContext(op)
            cls._operator_contexts[op._ptr] = op_ctx
            cls.logger.info(f"Created OperatorContext for operator {hex(id(op))}")

    @classmethod
    def get_operator_context(cls, op_ptr: int) -> "OperatorContext":
        """
        Get the OperatorContext for a given operator pointer.
        """
        if op_ptr not in cls._operator_contexts:
            raise RuntimeError(
                f"No OperatorContext found for operator pointer {hex(op_ptr)}. "
                "Ensure maybe_create_operator_context() was called before get_operator_context()."
            )
        return cls._operator_contexts[op_ptr]

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
        # State shape is with the batch dimension, so need - 1.
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
        cls.logger.info("Freeing CudensitymatContext")

        # Free all state handles from state contexts
        for state_ctx in cls._state_contexts.values():
            state_ctx.free()

        # Release gradient callback function references from operator contexts
        for op_ctx in cls._operator_contexts.values():
            op_ctx.free()

        # Free workspace and library handle
        if cls._workspace_desc is not None:
            cudm.destroy_workspace(cls._workspace_desc)
            cls._workspace_desc = None

        if cls._handle is not None:
            cudm.destroy(cls._handle)
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

    def __init__(self, op: Operator) -> None:
        """
        Initialize OperatorContext.

        Args:
            op: The operator object for operator action.
        """
        self.logger.info("Initializing OperatorContext")

        # Derived attributes from operator.
        self._space_mode_extents = op.dims
        self._data_type = typemaps.NAME_TO_DATA_TYPE[op.dtype.name]
        self._compute_type = typemaps.NAME_TO_COMPUTE_TYPE[op.dtype.name]

        # Create opaque handle to the operator.
        op._create(CudensitymatContext._handle)
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

        # The state adjoints are to be set in create_adjoint_buffers when backward rule is called.
        self._state_in_adj = None
        self._state_out_adj = None

    def create_adjoint_buffers(self):
        """
        Create adjoint buffers for the input and output states.

        Frees any previously allocated adjoint handles before allocating new
        ones so that repeated backward passes do not leak C-side GPU memory.
        """
        if self._state_in_adj is not None:
            cudm.destroy_state(self._state_in_adj)
            self.logger.debug(f"Destroyed stale input state adjoint at {hex(self._state_in_adj)}")
            self._state_in_adj = None

        if self._state_out_adj is not None:
            cudm.destroy_state(self._state_out_adj)
            self.logger.debug(f"Destroyed stale output state adjoint at {hex(self._state_out_adj)}")
            self._state_out_adj = None

        self._state_in_adj = cudm.create_state(
            CudensitymatContext._handle,
            self.state_purity,
            len(self._space_mode_extents),
            self._space_mode_extents,
            self.batch_size,
            self._data_type
        )
        self.logger.debug(f"Created input state adjoint at {hex(self._state_in_adj)}")

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
