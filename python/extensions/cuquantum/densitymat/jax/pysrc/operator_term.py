# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator term class in cuDensityMat.
"""

import ctypes
import logging
from collections.abc import Sequence

import cupy as cp
import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm

from .elementary_operator import ElementaryOperator
from .matrix_operator import MatrixOperator
from ..utils import (
    get_scalar_assignment_callback,
    get_empty_scalar_callback,
    get_scalar_gradient_attachment_callback,
    get_random_odd_pointer_and_object,
    detect_ad_traced_object,
    is_vmap_traced,
)


@jax.tree_util.register_pytree_node_class
class OperatorTerm:
    """
    PyTree class for cuDensityMat's operator term.
    """

    logger = logging.getLogger("cudensitymat-jax.OperatorTerm")

    def __init__(self, dims: Sequence[int]) -> None:
        """
        Initialize an OperatorTerm object.

        Args:
            dims: Hilbert space dimensions.
        """
        # Input argument.
        self.dims: tuple[int, ...] = tuple(dims)

        # Attributes for handling arguments in append.
        self.op_prods: list[tuple[ElementaryOperator | MatrixOperator, ...]] = []
        self.modes: list[tuple[int, ...]] = []
        self.conjs: list[tuple[bool, ...]] = []
        self.duals: list[tuple[bool, ...]] = []

        # When coeff is stored inside, it is always a jax.Array.
        self.coeffs: list[jax.Array] = []

        # Attributes inferred from multiple append calls.
        self.batch_sizes: list[int] = []  # keep track of batch sizes of all operator products
        self.batch_size: int = 1
        self.dtype: jnp.dtype | None = None

        # Internal attributes from interfacing to cuDensityMat.
        self._op_prod_types: list[type[ElementaryOperator] | type[MatrixOperator]] = []
        self._ptr: int | None = None

        # Attributes for handling gradients.
        self._coeff_requires_grads = []
        self._coeff_callbacks = []
        self._coeff_grad_callbacks = []
        self._coeff_ptrs: list[int] = []
        self._coeff_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive
        self._coeff_grad_ptrs: list[int] = []
        self._coeff_grad_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive
        self._total_coeffs_ptrs: list[int] = []
        self._total_coeffs_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive

    def tree_flatten(self):
        """
        Flatten the operator term into a PyTree.
        """
        children = (self.op_prods, self.coeffs)
        aux_data = (
            self.dims,
            self.modes,
            self.conjs,
            self.duals,
            self.batch_size,
            self.batch_sizes,
            self.dtype,
            self._op_prod_types,
            self._ptr,
            self._coeff_ptrs,
            self._coeff_ptr_objs,
            self._coeff_grad_ptrs,
            self._coeff_requires_grads,
            self._coeff_callbacks,
            self._coeff_grad_callbacks,
            self._coeff_grad_ptr_objs,
            self._total_coeffs_ptrs,
            self._total_coeffs_ptr_objs,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the operator term from a PyTree.
        """
        inst = cls.__new__(cls)
        inst.op_prods, inst.coeffs = children
        (
            inst.dims,
            inst.modes,
            inst.conjs,
            inst.duals,
            inst.batch_size,
            inst.batch_sizes,
            inst.dtype,
            inst._op_prod_types,
            inst._ptr,
            inst._coeff_ptrs,
            inst._coeff_ptr_objs,
            inst._coeff_grad_ptrs,
            inst._coeff_requires_grads,
            inst._coeff_callbacks,
            inst._coeff_grad_callbacks,
            inst._coeff_grad_ptr_objs,
            inst._total_coeffs_ptrs,
            inst._total_coeffs_ptr_objs,
        ) = aux_data
        return inst

    @property
    def in_axes(self) -> "OperatorTerm":
        """
        Return the in_axes PyTree spec for vmapping over the batch dimension.
        """
        in_axes_op_prods = [tuple(op.in_axes for op in op_prod) for op_prod in self.op_prods]
        in_axes_coeffs = []
        for coeff in self.coeffs:
            if is_vmap_traced(coeff) or len(coeff) > 1:
                in_axes_coeffs.append(0)
            else:
                in_axes_coeffs.append(None)

        _, aux_data = self.tree_flatten()
        return type(self).tree_unflatten(aux_data, (in_axes_op_prods, in_axes_coeffs))

    def _copy(self) -> "OperatorTerm":
        """
        Internal method to copy the operator term for VJP backward pass.
        """
        op_term = type(self).__new__(type(self))

        op_term.op_prods = [tuple(base_op._copy() for base_op in op_prod) for op_prod in self.op_prods]
        op_term.coeffs = [jnp.copy(c) for c in self.coeffs]

        op_term.dims = self.dims
        op_term.modes = self.modes.copy()
        op_term.conjs = self.conjs.copy()
        op_term.duals = self.duals.copy()
        op_term.batch_size = self.batch_size
        op_term.batch_sizes = self.batch_sizes.copy()
        op_term.dtype = self.dtype
        op_term._op_prod_types = self._op_prod_types.copy()
        op_term._ptr = self._ptr
        op_term._coeff_ptrs = self._coeff_ptrs.copy()
        op_term._coeff_ptr_objs = self._coeff_ptr_objs.copy()
        op_term._coeff_grad_ptrs = self._coeff_grad_ptrs.copy()
        op_term._coeff_grad_ptr_objs = self._coeff_grad_ptr_objs.copy()
        op_term._coeff_requires_grads = self._coeff_requires_grads.copy()
        op_term._coeff_callbacks = self._coeff_callbacks.copy()
        op_term._coeff_grad_callbacks = self._coeff_grad_callbacks.copy()
        op_term._total_coeffs_ptrs = self._total_coeffs_ptrs.copy()
        op_term._total_coeffs_ptr_objs = self._total_coeffs_ptr_objs.copy()

        return op_term

    def _check_and_set_dtype(self, op_prod) -> None:
        """
        Check if the operator product has the same data type as the operator term.
        """
        if self.dtype is None:
            # If the data type is not set, set it to the data type of the first operator.
            self.dtype = op_prod[0].dtype
            for op in op_prod[1:]:
                if op.dtype != self.dtype:
                    raise ValueError("All elementary or matrix operators must have the same data type.")
        else:
            # If the data type is set, check if all elementary or matrix operators
            # have the same data type.
            for op in op_prod:
                if op.dtype != self.dtype:
                    raise ValueError("All elementary or matrix operators must have the same data type.")

    def _check_and_append_op_prod_type(self, op_prod):
        """
        Check if all terms in an operator product are of the same type.
        """
        op_prod_type = type(op_prod[0])
        for op in op_prod[1:]:
            if not isinstance(op, op_prod_type):
                raise ValueError("All terms in an operator product must be of the same type.")
        self._op_prod_types.append(op_prod_type)

    def _check_and_set_batch_size(self,
                                  op_prod: Sequence[ElementaryOperator | MatrixOperator],
                                  coeff: jax.Array,
                                  ) -> None:
        """
        Check if the operator product and coefficient batch sizes are consistent.
        """
        # Extract the batch size for the operator product.
        op_prod_batch_size = max([base_op.batch_size for base_op in op_prod])
        for base_op in op_prod:
            if base_op.batch_size not in (1, op_prod_batch_size):
                raise ValueError("All basic operators in an operator product must have batch size 1 or N.")

        # Possibly update the batch size of this operator term and check consistency.
        batch_size = max(op_prod_batch_size, len(coeff))
        if self.batch_size == 1:
            self.batch_size = batch_size
        else:
            if batch_size not in (1, self.batch_size):
                raise ValueError("Batch size in this operator product does not match batch size of this operator term.")

    def append(self,
               op_prod: Sequence[ElementaryOperator | MatrixOperator],
               modes: Sequence[int] | None = None,
               conjs: Sequence[bool] | None = None,
               duals: Sequence[bool] | None = None,
               coeff: float | complex | jax.Array = 1.0,
               ) -> None:
        """
        Append an elementary or matrix product to an operator term.

        Args:
            op_prod: Product of elementary or matrix operators to be appended.
            modes: Modes acted on by the operator product.
            conjs: Conjugations in the operator product. Only applies to MatrixOperators.
            duals: Dualities of the operator product.
            coeff: Non-batched coefficient or batched coefficients of the operator product.
        """
        if self._ptr is not None:
            raise RuntimeError("Cannot modify operator term after it has been used in an operator action.")

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

        # Check if all elementary or matrix operators have the same data type.
        self._check_and_set_dtype(op_prod)
        self._check_and_append_op_prod_type(op_prod)
        self._check_and_set_batch_size(op_prod, coeff)

        # Check consistency and append modes, conjs and duals.
        if self._op_prod_types[-1] is ElementaryOperator:
            # Modes have to specified for elementary operators.
            if modes is None:
                raise ValueError("Modes acted on must be specified for elementary operators.")

            # Check all modes are in Hilbert space.
            if not set(modes) <= set(range(len(self.dims))):
                raise ValueError("Modes acted on must be in the Hilbert space, i.e. between 0 and len(self.dims) - 1")

            # Check length of modes acted on are the same as combined number of modes in the operator product.
            if len(modes) != sum([elem_op.num_modes for elem_op in op_prod]):
                raise ValueError(f"Number of modes acted on {len(modes)} does not match combined number of modes in the operator product.")

            # Check mode extents of each elementary operator match corresponding qubit dimensions.
            modes_index = 0
            for elem_op in op_prod:
                if elem_op.mode_extents != tuple(
                    [self.dims[modes[i]] for i in range(modes_index, modes_index + elem_op.num_modes)]
                ):
                    raise ValueError("Mode extents of each elementary operator must match corresponding qubit dimensions.")
                modes_index += elem_op.num_modes

            # Check that matrix conjugations cannot be specified for elementary operators.
            if conjs is not None:
                raise ValueError("Matrix conjugations cannot be specified for elementary operators.")

            # Check that number of duals matches number of modes.
            if duals is None:
                duals = (False,) * len(modes)
            else:
                if len(duals) != len(modes):
                    raise ValueError("Number of duals must match number of modes acted on for elementary operator product.")

            # For elementary operator product, we only need modes and duals.
            self.modes.append(tuple(modes))
            self.conjs.append(())  # empty tuple is appended here to preserve length
            self.duals.append(tuple(duals))

        else:  # matrix operator product
            # Check that mode extents match Hilbert space dimensions.
            for matrix_op in op_prod:
                if matrix_op.mode_extents != self.dims:
                    raise ValueError("Mode extents must match Hilbert space dimensions for matrix operators.")

            # Check that modes acted on cannot be specified for matrix operators.
            if modes is not None:
                raise ValueError("Modes acted on cannot be specified for matrix operators.")

            # Check consistency of conjs.
            if conjs is None:
                conjs = (False,) * len(op_prod)
            else:
                if len(conjs) != len(op_prod):
                    raise ValueError("Number of matrix conjugations must match number of operator products.")

            # Check that number of duals matches number of matrix operators.
            if duals is None:
                duals = (False,) * len(op_prod)
            else:
                if len(duals) != len(op_prod):
                    raise ValueError("Number of duals must match number of matrix operators.")

            # For matrix operator product, we only need conjs and duals.
            self.modes.append(tuple(range(len(self.dims))))  # used in reference implementation during testing
            self.conjs.append(tuple(conjs))
            self.duals.append(tuple(duals))

        # Attributes from function arguments.
        self.op_prods.append(tuple(op_prod))
        self.coeffs.append(coeff)

        # Set batch size.
        self.batch_sizes.append(len(coeff))

        # Attributes for handling gradients. None is appended here to preserve length, which is then
        # updated in the _create method.
        self._coeff_requires_grads.append(None)
        self._coeff_ptrs.append(0)
        self._coeff_ptr_objs.append(None)
        self._coeff_grad_ptrs.append(0)
        self._coeff_callbacks.append(None)
        self._coeff_grad_callbacks.append(None)
        self._coeff_grad_ptr_objs.append(None)
        self._total_coeffs_ptrs.append(0)
        self._total_coeffs_ptr_objs.append(None)

    def __getitem__(self, index: int) -> tuple[ElementaryOperator | MatrixOperator, ...]:
        """
        Get an operator product from the operator term.
        """
        return self.op_prods[index]

    def _create(self, handle):
        """
        Create opaque handle to the operator term.
        """
        # Create opaque handle to dependent elementary or matrix operators.
        # XXX: We are creating redundant elementary operators here.
        # When JAX flattens and unflattens the PyTrees, it creates new Python objects.
        for op_prod in self.op_prods:
            for elem_op in op_prod:
                elem_op._create(handle)

        # Create the current operator term.
        if self._ptr is None:
            self._ptr = cudm.create_operator_term(handle, len(self.dims), self.dims)
            self.logger.debug(f"Created operator term at {hex(self._ptr)}")

            for i in range(len(self.op_prods)):

                # Detect if the coefficient requires gradient and assign callback, gradient callback,
                # temporary coefficient pointer and object.
                self._coeff_requires_grads[i] = detect_ad_traced_object(self.coeffs[i])
                if not is_vmap_traced(self.coeffs[i]) and len(self.coeffs[i]) == 1:  # non-batched coefficient

                    # Traced scalars need to be passed through an intermediate memory slot.
                    self._coeff_callbacks[i] = get_scalar_assignment_callback(self.coeffs[i])
                    self._coeff_ptrs[i] = self._coeff_callbacks[i].callback.coeff.data.ptr

                    # If gradient is computed on the coefficient, assign gradient callback and pointer.
                    if self._coeff_requires_grads[i]:
                        self._coeff_grad_callbacks[i] = get_scalar_gradient_attachment_callback(self.coeffs[i])
                        self._coeff_grad_ptrs[i] = self._coeff_grad_callbacks[i].callback.scalar_grad.data.ptr

                else:  # batched coefficients
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

                if self._op_prod_types[i] is ElementaryOperator:
                    if self.batch_sizes[i] == 1:
                        cudm.operator_term_append_elementary_product(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [elem_op._ptr for elem_op in self.op_prods[i]],
                            self.modes[i],
                            self.duals[i],
                            1.0,  # coefficient, to be updated to the real coefficient by callback
                            self._coeff_callbacks[i],
                            self._coeff_grad_callbacks[i],
                        )
                    else:
                        cudm.operator_term_append_elementary_product_batch(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [elem_op._ptr for elem_op in self.op_prods[i]],
                            self.modes[i],
                            self.duals[i],
                            self.batch_sizes[i],
                            self._coeff_ptrs[i],
                            self._total_coeffs_ptrs[i],
                            self._coeff_callbacks[i],
                            self._coeff_grad_callbacks[i],
                        )
                else:  # MatrixOperator
                    if self.batch_sizes[i] == 1:
                        cudm.operator_term_append_matrix_product(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [mat_op._ptr for mat_op in self.op_prods[i]],
                            self.conjs[i],
                            self.duals[i],
                            1.0,  # coefficient, to be updated to the real coefficient by callback
                            self._coeff_callbacks[i],
                            self._coeff_grad_callbacks[i],
                        )
                    else:
                        cudm.operator_term_append_matrix_product_batch(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [mat_op._ptr for mat_op in self.op_prods[i]],
                            self.conjs[i],
                            self.duals[i],
                            self.batch_sizes[i],
                            self._coeff_ptrs[i],
                            self._total_coeffs_ptrs[i],
                            self._coeff_callbacks[i],
                            self._coeff_grad_callbacks[i],
                        )

            self.logger.debug(f"Appended operator products to operator term at {hex(self._ptr)}")

    def _destroy(self):
        """
        Destroy opaque handle to the operator term.
        """
        if self._ptr is not None:
            cudm.destroy_operator_term(self._ptr)
            self.logger.debug(f"Destroyed operator term at {hex(self._ptr)}")
            self._ptr = None
