# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
cuDensityMat context classes.
"""

import atexit
import logging

import jax

from cuquantum.bindings import cudensitymat as cudm
from nvmath.internal import typemaps

from .operators import ElementaryOperator, MatrixOperator, OperatorTerm, Operator


class CudensitymatContext:
    """
    cuDensityMat library context.

    This class holds the library handle and the workspace descriptor handle, which are used for all
    operator actions. A specific operator action context is created for each operator,
    appended to _contexts, and retrieved when the specific operator action is invoked.
    """

    _handle = None
    _workspace_desc = None
    _operators = set()
    _contexts = {}

    logger = logging.getLogger("cudensitymat-jax.CudensitymatContext")

    @classmethod
    def _maybe_create_handle_and_workspace(cls):
        """
        Create the handle and workspace for the context if they are not already created.
        """
        if cls._handle is None:
            cls.logger.info(f"Initializing CudensitymatContext")
            cls._handle = cudm.create()

            if cls._workspace_desc is None:
                cls._workspace_desc = cudm.create_workspace(cls._handle)
            else:  # handle is None but workspace is not None
                raise RuntimeError("Workspace descriptor and handle should be created at the same time")
        else:
            if cls._workspace_desc is None:  # handle is not None but workspace is None
                raise RuntimeError("Workspace descriptor and handle should be created at the same time")

    @classmethod
    def maybe_create_context(cls, op: Operator, device: jax.Device, batch_size: int, purity: cudm.StatePurity) -> None:
        """
        Create the OperatorActionContext for a new operator.
        """
        cls._maybe_create_handle_and_workspace()

        # Check if there is already a context for the operator and if not, create it.
        # NOTE: Registered operators are recorded using the Python object. _contexts is indexed by
        # opaque pointers instead since the operator objects themselves change when they are
        # flattened and unflattened during PyTree manipulation.
        if op not in cls._operators:
            cls._operators.add(op)

            ctx = OperatorActionContext(op, device, batch_size, purity)
            cls._contexts[op._ptr] = ctx

            cls.logger.info(f"Created OperatorActionContext for operator {hex(id(op))}")

    @classmethod
    def get_context(cls, op: Operator) -> "OperatorActionContext":
        """
        Get the OperatorActionContext for a given operator.
        """
        # TODO: Raise an error with error message when op._ptr is not in cls._contexts.
        return cls._contexts[op._ptr]

    @classmethod
    def free(cls):
        """
        Free opaque handles to the library.
        """
        op_terms = set()
        base_ops = set()

        for op in cls._operators:
            op._destroy()
            op_terms.update(jax.tree.leaves(op, is_leaf=lambda x: isinstance(x, OperatorTerm)))
            base_ops.update(jax.tree.leaves(op, is_leaf=lambda x: isinstance(x, (ElementaryOperator, MatrixOperator))))

        for op_term in op_terms:
            op_term._destroy()

        for base_op in base_ops:
            base_op._destroy()

        for ctx in cls._contexts.values():
            ctx.free()

        cls.logger.info(f"Freeing CudensitymatContext")

        if cls._workspace_desc is not None:
            cudm.destroy_workspace(cls._workspace_desc)
            cls._workspace_desc = None

        if cls._handle is not None:
            cudm.destroy(cls._handle)
            cls._handle = None


atexit.register(CudensitymatContext.free)


class OperatorActionContext:
    """
    Operator action context.
    """

    logger = logging.getLogger("cudensitymat-jax.OperatorActionContext")

    def __init__(self,
                 op: Operator,
                 device: jax.Device,
                 batch_size: int,
                 purity: cudm.StatePurity
                 ) -> None:
        """
        Initialize OperatorActionContext.

        Args:
            op: The operator object for operator action.
            device: The device to use in operator action. If None, the first GPU device will be used.
            batch_size: Batch size of the operator action.
            purity: Purity of the state.
        """
        self.logger.info(f"Initializing OperatorActionContext")

        # Instance attributes.
        self.operator = op
        self.device = device
        self.batch_size = batch_size
        self.state_purity = purity

        # Derived attributes from operator.
        self._space_mode_extents = op.dims
        self._data_type = typemaps.NAME_TO_DATA_TYPE[op.dtype.name]
        self._compute_type = typemaps.NAME_TO_COMPUTE_TYPE[op.dtype.name]

        # A unified buffer size for all primitives.
        self._required_buffer_size = 0

        # Create opaque handle to the operator.
        self.operator._create(CudensitymatContext._handle)
        self._operator = self.operator._ptr

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
        """
        # Create opaque handles to the input and output state adjoints.
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
        self.logger.info(f"Freeing OperatorActionContext")

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

        if self._operator is not None:
            self.operator._destroy()
            self._operator = None
