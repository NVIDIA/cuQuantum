# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator class in cuDensityMat.
"""

import ctypes
import logging
from collections.abc import Sequence

import cupy as cp
import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm

from .operator_term import OperatorTerm
from ..utils import (
    get_scalar_assignment_callback,
    get_empty_scalar_callback,
    get_scalar_gradient_attachment_callback,
    get_random_odd_pointer_and_object,
    detect_ad_traced_object,
    is_vmap_traced,
)


@jax.tree_util.register_pytree_node_class
class Operator:
    """
    PyTree class for cuDensityMat's operator.
    """

    logger = logging.getLogger("cudensitymat-jax.Operator")

    def __init__(self, dims: Sequence[int]) -> None:
        """
        Initialize an Operator object.

        Args:
            dims: Hilbert space dimensions.
        """
        # Attribute set from constructor.
        self.dims: tuple[int, ...] = tuple(dims)

        # Attributes for arguments in append.
        self.op_terms: list[OperatorTerm] = []
        self.duals: list[bool] = []
        self.coeffs: list[jax.Array] = []

        # Attributes inferred from multiple append calls.
        self.batch_sizes: list[int] = []  # keep track of batch sizes of all operator terms
        self.batch_size: int = 1
        self.dtype: jnp.dtype | None = None

        # Internal attributes from interfacing to cuDensityMat.
        self._ptr: int | None = None

        # Attributes for handling JIT tracing.
        self._coeff_ptrs: list[int] = []
        self._coeff_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive

        self._coeff_grad_ptrs: list[int] = []
        self._coeff_grad_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive

        # Attributes for handling gradients.
        self._coeff_requires_grads: list[bool] = []
        self._coeff_callbacks = []
        self._coeff_grad_callbacks = []
        self._op_term_ids: list[int] = []
        self._total_coeffs_ptrs: list[int] = []
        self._total_coeffs_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive

    def tree_flatten(self):
        """
        Flatten the operator PyTree.
        """
        children = (self.op_terms, self.coeffs)
        aux_data = (
            self.dims,
            self.duals,
            self.batch_size,
            self.batch_sizes,
            self.dtype,
            self._ptr,
            self._coeff_ptrs,
            self._coeff_ptr_objs,
            self._coeff_grad_ptrs,
            self._coeff_grad_ptr_objs,
            self._coeff_requires_grads,
            self._coeff_callbacks,
            self._coeff_grad_callbacks,
            self._op_term_ids,
            self._total_coeffs_ptrs,
            self._total_coeffs_ptr_objs,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the operator PyTree.
        """
        inst = cls.__new__(cls)
        inst.op_terms, inst.coeffs = children
        (
            inst.dims,
            inst.duals,
            inst.batch_size,
            inst.batch_sizes,
            inst.dtype,
            inst._ptr,
            inst._coeff_ptrs,
            inst._coeff_ptr_objs,
            inst._coeff_grad_ptrs,
            inst._coeff_grad_ptr_objs,
            inst._coeff_requires_grads,
            inst._coeff_callbacks,
            inst._coeff_grad_callbacks,
            inst._op_term_ids,
            inst._total_coeffs_ptrs,
            inst._total_coeffs_ptr_objs,
        ) = aux_data
        return inst

    @property
    def in_axes(self) -> "Operator":
        """
        Return the in_axes PyTree spec for vmapping over the batch dimension.
        """
        in_axes_op_terms = [op_term.in_axes for op_term in self.op_terms]
        in_axes_coeffs = []
        for coeff in self.coeffs:
            if is_vmap_traced(coeff) or len(coeff) > 1:
                in_axes_coeffs.append(0)
            else:
                in_axes_coeffs.append(None)

        _, aux_data = self.tree_flatten()
        return type(self).tree_unflatten(aux_data, (in_axes_op_terms, in_axes_coeffs))
    
    def _copy(self) -> "Operator":
        """
        Internal method to copy the operator for VJP backward pass.
        """
        op = type(self).__new__(type(self))

        op.op_terms = [op_term._copy() for op_term in self.op_terms]
        op.coeffs = [jnp.copy(c) for c in self.coeffs]

        op.dims = self.dims
        op.duals = self.duals.copy()
        op.batch_size = self.batch_size
        op.batch_sizes = self.batch_sizes.copy()
        op.dtype = self.dtype
        op._ptr = self._ptr
        op._coeff_ptrs = self._coeff_ptrs.copy()
        op._coeff_ptr_objs = self._coeff_ptr_objs.copy()
        op._coeff_grad_ptrs = self._coeff_grad_ptrs.copy()
        op._coeff_grad_ptr_objs = self._coeff_grad_ptr_objs.copy()
        op._coeff_requires_grads = self._coeff_requires_grads.copy()
        op._coeff_callbacks = self._coeff_callbacks.copy()
        op._coeff_grad_callbacks = self._coeff_grad_callbacks.copy()
        op._op_term_ids = self._op_term_ids.copy()
        op._total_coeffs_ptrs = self._total_coeffs_ptrs.copy()
        op._total_coeffs_ptr_objs = self._total_coeffs_ptr_objs.copy()
        return op

    def _check_and_set_dtype(self, op_term: OperatorTerm) -> None:
        """
        Check if the operator term has the same data type as the operator.
        """
        if op_term.dtype is not None:  # for empty operator term, skip the check.
            if self.dtype is None:
                # If the data type is not set, set it to the data type of the first operator term.
                self.dtype = op_term.dtype
            else:
                # If the data type is set, check if the operator term has the same data type as the operator.
                if op_term.dtype != self.dtype:
                    raise ValueError("All operator terms must have the same data type.")

    def _check_and_set_batch_size(self, op_term: OperatorTerm, coeff: jax.Array) -> None:
        """
        Check if the operator term and coefficient batch sizes are consistent.
        """
        # Possibly update the batch size of this operator and check consistency.
        batch_size = max(op_term.batch_size, len(coeff))
        if self.batch_size == 1:
            self.batch_size = batch_size
        else:
            if batch_size not in (1, self.batch_size):
                raise ValueError("Batch size in this operator term does not match batch size of this operator.")

    def append(self,
               op_term: OperatorTerm,
               dual: bool = False,
               coeff: float | complex | jax.Array = 1.0,
               ) -> None:
        """
        Append an operator term to an operator.

        Args:
            op_term: Operator term to be appended.
            dual: Duality of the operator term.
            coeff: Non-batched coefficient or batched coefficients of the operator term.
            coeff_requires_grad: Whether the coefficients require gradient.
        """
        # TODO: Instead of an explicit coeff_requires_grad argument, this should be detected
        # automatically from the trace stack.

        if self._ptr is not None:
            raise RuntimeError("Cannot modify operator after it has been used in an operator action.")

        # coeff is converted to a length-1 array if it is a Python scalar.
        if (
            isinstance(coeff, (float, complex)) or
            (isinstance(coeff, jax.Array) and coeff.ndim == 0)  # scalar but traced
        ):
            coeff = jnp.array([coeff], dtype=jnp.complex128)
        elif isinstance(coeff, jax.Array) and coeff.ndim > 0:
            if coeff.dtype != jnp.complex128:
                raise ValueError("Coefficient must be of type complex128.")
        else:
            raise ValueError("Coefficient must be a float, complex, or jax.Array.")

        # Attributes from function arguments.
        self.op_terms.append(op_term)
        self.duals.append(dual)
        self.coeffs.append(coeff)

        # Set batch size and dtype.
        self.batch_sizes.append(len(coeff))
        self._check_and_set_batch_size(op_term, coeff)  # setting batch size of the operator
        self._check_and_set_dtype(op_term)

        # Internal attributes.
        self._coeff_callbacks.append(None)
        self._coeff_ptrs.append(0)
        self._coeff_ptr_objs.append(None)

        self._coeff_requires_grads.append(None)
        self._coeff_grad_callbacks.append(None)
        self._coeff_grad_ptrs.append(0)
        self._coeff_grad_ptr_objs.append(None)
        self._total_coeffs_ptrs.append(0)
        self._total_coeffs_ptr_objs.append(None)
        self._op_term_ids.append(id(op_term))

    def __getitem__(self, index: int) -> OperatorTerm:
        """
        Get an operator term from the operator.
        """
        return self.op_terms[index]

    def _create(self, handle):
        """
        Create opaque handle to the operator.
        """
        # Create a dictionary to map from the original op_term_id to the first index of
        # the op_term in the operator. The original op_term_id need to be used since id(op_term)
        # changes when JAX flattens and unflattens the PyTrees.
        id_to_first_index = {}
        for i, op_term_id in enumerate(self._op_term_ids):
            if op_term_id not in id_to_first_index:
                id_to_first_index[op_term_id] = i

        # Only create the opaque handles to the unique operator terms.
        for i in id_to_first_index.values():
            self.op_terms[i]._create(handle)

        # Create the current operator.
        if self._ptr is None:
            self._ptr = cudm.create_operator(handle, len(self.dims), self.dims)
            self.logger.debug(f"Created operator at {hex(self._ptr)}")
            # Keep batched coefficient buffers alive so C API pointers remain valid.
            self._batch_coeff_arrs = []

            for i in range(len(self.op_terms)):
                # Detect if the coefficient requires gradient and assign callback, gradient callback,
                # temporary coefficient pointer and object.
                self._coeff_requires_grads[i] = detect_ad_traced_object(self.coeffs[i])
                if not is_vmap_traced(self.coeffs[i]) and len(self.coeffs[i]) == 1:
                    # Traced scalars need to be passed through an intermediate memory slot.
                    self._coeff_callbacks[i] = get_scalar_assignment_callback(self.coeffs[i])
                    self._coeff_ptrs[i] = self._coeff_callbacks[i].callback.coeff.data.ptr

                    # If gradient is computed on the coefficient, assign gradient callback and pointer.
                    if self._coeff_requires_grads[i]:
                        self._coeff_grad_callbacks[i] = get_scalar_gradient_attachment_callback(self.coeffs[i])
                        self._coeff_grad_ptrs[i] = self._coeff_grad_callbacks[i].callback.scalar_grad.data.ptr

                else:
                    if is_vmap_traced(self.coeffs[i]):
                        coeff_shape = self.coeffs[i].val.shape
                        coeff_dtype = self.coeffs[i].val.dtype
                    else:
                        coeff_shape = self.coeffs[i].shape
                        coeff_dtype = self.coeffs[i].dtype
                    static_coeff_buf = cp.ones(coeff_shape, dtype=coeff_dtype)
                    self._coeff_ptrs[i] = static_coeff_buf.data.ptr
                    self._coeff_ptr_objs[i] = static_coeff_buf

                    # For batched coefficients, callback is required by the API but is a no-op.
                    self._coeff_callbacks[i] = get_empty_scalar_callback()
                    self._total_coeffs_ptrs[i], self._total_coeffs_ptr_objs[i] = get_random_odd_pointer_and_object()

                    if self._coeff_requires_grads[i]:
                        self._coeff_grad_callbacks[i] = get_scalar_gradient_attachment_callback(self.coeffs[i])
                        self._coeff_grad_ptrs[i] = self._coeff_grad_callbacks[i].callback.scalar_grad.data.ptr

                if self.batch_sizes[i] == 1:
                    cudm.operator_append_term(
                        handle,
                        self._ptr,
                        self.op_terms[id_to_first_index[self._op_term_ids[i]]]._ptr,
                        self.duals[i],
                        1.0,  # coefficient, to be updated to the real coefficient by callback
                        self._coeff_callbacks[i],
                        self._coeff_grad_callbacks[i],
                    )
                else:
                    # This is only needed when is_vmap_traced(self.coeffs[i]) or len(self.coeffs[i]) > 1,
                    # and when _self._coeff_requires_grads[i] is True.
                    cudm.operator_append_term_batch(
                        handle,
                        self._ptr,
                        self.op_terms[id_to_first_index[self._op_term_ids[i]]]._ptr,
                        self.duals[i],
                        self.batch_sizes[i],
                        self._coeff_ptrs[i],
                        self._total_coeffs_ptrs[i],
                        self._coeff_callbacks[i],
                        self._coeff_grad_callbacks[i],
                    )
            self.logger.debug(f"Appended operator terms to operator at {hex(self._ptr)}")
    
    def _destroy(self):
        """
        Destroy opaque handle to the operator.
        """
        if self._ptr is not None:
            # Destroy the current operator.
            cudm.destroy_operator(self._ptr)
            self.logger.debug(f"Destroyed operator at {hex(self._ptr)}")
            self._ptr = None
