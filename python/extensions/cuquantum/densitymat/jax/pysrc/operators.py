# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
cuDensityMat operator API classes for cuQuantum-JAX.
"""

import logging
from typing import List, Tuple, Sequence, Type

import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm
from nvmath.internal import typemaps


@jax.tree_util.register_pytree_node_class
class ElementaryOperator:
    """
    PyTree class for cuDensityMat's elementary operator.
    """

    logger = logging.getLogger("cudensitymat-jax.ElementaryOperator")

    def __init__(self,
                 data: jax.Array,
                 callback: cudm.WrappedTensorCallback | None = None,
                 grad_callback: cudm.WrappedTensorGradientCallback | None = None,
                 offsets: Tuple[int, ...] = ()
                 ) -> None:
        """
        Initialize an ElementaryOperator object.

        Args:
            data: Data buffer of the elementary operator.
            callback: Forward callback for the elementary operator.
            grad_callback: Gradient callback for the elementary operator.
            offsets: Diagonal offsets of the elementary operator.
        """
        # Check consistency of tensor data.
        if isinstance(data, jax.Array):
            if len(offsets) == 0:  # dense elementary operator
                if len(data.shape) % 2 != 0:
                    raise ValueError("Data must have an even number of dimensions.")
                if data.shape[:data.ndim // 2] != data.shape[data.ndim // 2:]:
                    raise ValueError("Data must have the same shape on the bra and ket modes.")
                self.sparsity = cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_NONE
            else:  # multidiagonal elementary operator
                if len(offsets) != len(set(offsets)):
                    raise ValueError("Offsets cannot contain duplicate elements.")
                if data.shape[1] != len(offsets):
                    raise ValueError("Number of columns in data does not match length of offsets.")
                if data.ndim != 2:
                    raise ValueError("Only single-mode multidiagonal elementary operator is supported.")
                self.sparsity = cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL

            # Attributes derived from data.
            self.data = data
            self.num_modes: int = data.ndim // 2
            self.mode_extents: Tuple[int, ...] = data.shape[:self.num_modes]
            self.dtype: jnp.dtype = self.data.dtype

        else:
            # data becomes object() during AD tracing.
            assert type(data) is object

            # Dummy variables for derived attributes.
            self.data = data
            self.num_modes: int = 0
            self.mode_extents: Tuple[int, ...] = ()
            self.dtype: jnp.dtype = jnp.dtype(float)

        # Callbacks and diagonal offsets.
        self.callback: cudm.WrappedTensorCallback | None = callback
        self.grad_callback: cudm.WrappedTensorGradientCallback | None = grad_callback
        self.offsets: Tuple[int, ...] = offsets

        self._ptr: int | None = None
        self._is_elementary: int | None = None

    def tree_flatten(self):
        """
        Flatten the elementary operator PyTree.
        """
        children = (self.data, self._ptr, self._is_elementary)
        aux_data = (self.callback, self.grad_callback, self.offsets)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the elementary operator PyTree.
        """
        data, ptr, is_elementary = children
        callback, grad_callback, offsets = aux_data
        inst = cls(data, callback, grad_callback, offsets)
        inst._ptr = ptr
        inst._is_elementary = is_elementary
        return inst

    def _create(self, handle):
        """
        Create opaque handle to the elementary operator.
        """
        # Create opaque handle to the elementary operator.
        if self._ptr is None:
            self._ptr = cudm.create_elementary_operator(
                handle,
                self.num_modes,
                self.mode_extents,
                self.sparsity,
                len(self.offsets),
                self.offsets,
                typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                0,  # buffer pointer to be attached in the XLA layer
                self.callback,
                self.grad_callback
            )
            self.logger.debug(f"Created elementary operator at {hex(self._ptr)}")

        # Set in _create to prevent tracing.
        if self._is_elementary is None:
            self._is_elementary = 1
    
    def _destroy(self):
        """
        Destroy opaque handle to the elementary operator.
        """
        if self._ptr is not None:
            cudm.destroy_elementary_operator(self._ptr)
            self.logger.debug(f"Destroyed elementary operator at {hex(self._ptr)}")
            self._ptr = None


@jax.tree_util.register_pytree_node_class
class MatrixOperator:
    """
    PyTree class for cuDensityMat's matrix operator.
    """

    logger = logging.getLogger("cudensitymat-jax.MatrixOperator")

    def __init__(self,
                 data: jax.Array,
                 callback: cudm.WrappedTensorCallback | None = None,
                 grad_callback: cudm.WrappedTensorGradientCallback | None = None
                 ) -> None:
        """
        Initialize a MatrixOperator object.

        Args:
            data: Data buffer of the matrix operator.
            callback: Forward callback for the matrix operator.
            grad_callback: Gradient callback for the matrix operator.
        """
        if isinstance(data, jax.Array): 
            if len(data.shape) % 2 != 0:
                raise ValueError("Data must have an even number of dimensions.")
            if data.shape[:data.ndim // 2] != data.shape[data.ndim // 2:]:
                raise ValueError("Data must have the same shape on the bra and ket modes.")
            self.data = data
            self.num_modes: int = data.ndim // 2
            self.mode_extents: Tuple[int, ...] = data.shape[:self.num_modes]
            self.dtype: jnp.dtype = data.dtype
        else:
            # data becomes object() during AD tracing.
            assert type(data) is object

            # Dummy variables for derived attributes.
            self.data = data
            self.num_modes: int = 0
            self.mode_extents: Tuple[int, ...] = ()
            self.dtype: jnp.dtype = jnp.dtype(float)

        # Callbacks.
        self.callback: cudm.WrappedTensorCallback | None = callback
        self.grad_callback: cudm.WrappedTensorGradientCallback | None = grad_callback

        self._ptr: int | None = None
        self._is_elementary: int | None = None

    def tree_flatten(self):
        """
        Flatten the matrix operator PyTree.
        """
        children = (self.data, self._ptr, self._is_elementary)
        aux_data = (self.callback, self.grad_callback)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the matrix operator PyTree.
        """
        data, ptr, is_elementary = children
        callback, grad_callback = aux_data
        inst = cls(data, callback, grad_callback)
        inst._ptr = ptr
        inst._is_elementary = is_elementary
        return inst

    def _create(self, handle):
        """
        Create opaque handle to the matrix operator.
        """
        # Create opaque handle to the matrix operator.
        if self._ptr is None:
            self._ptr = cudm.create_matrix_operator_dense_local(
                handle,
                self.num_modes,
                self.mode_extents,
                typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                0,  # buffer pointer to be attached in the XLA layer
                self.callback,
                self.grad_callback)
            self.logger.debug(f"Created matrix operator at {hex(self._ptr)}")

        # Set in _create to prevent tracing.
        if self._is_elementary is None:
            self._is_elementary = 0

    def _destroy(self):
        """
        Destroy opaque handle to the matrix operator.
        """
        if self._ptr is not None:
            cudm.destroy_matrix_operator(self._ptr)
            self.logger.debug(f"Destroyed matrix operator at {hex(self._ptr)}")
            self._ptr = None


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
        self.dims: Tuple[int, ...] = tuple(dims)

        # Attributes for handling arguments in append.
        self.op_prods: List[Tuple[ElementaryOperator | MatrixOperator, ...]] = []
        self.modes: List[Tuple[int, ...]] = []
        self.conjs: List[Tuple[bool, ...]] = []
        self.duals: List[Tuple[bool, ...]] = []
        self.coeffs: List[float] = []
        self.coeff_callbacks: List[cudm.WrappedScalarCallback | None] = []
        self.coeff_grad_callbacks: List[cudm.WrappedScalarGradientCallback | None] = []

        self.dtype: jnp.dtype | None = None
        self._op_prod_types: List[Type[ElementaryOperator] | Type[MatrixOperator]] = []
        self._ptr: int | None = None

    def tree_flatten(self):
        """
        Flatten the operator term into a PyTree.
        """
        children = (self.op_prods,)
        aux_data = (
            self.dims,
            self.modes,
            self.conjs,
            self.duals,
            self.coeffs,
            self.coeff_callbacks,
            self.coeff_grad_callbacks,
            self.dtype,
            self._op_prod_types,
            self._ptr
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the operator term from a PyTree.
        """
        op_prods = children[0]
        (
            dims,
            modes,
            conjs,
            duals,
            coeffs,
            coeff_callbacks,
            coeff_grad_callbacks,
            dtype,
            op_prod_types,
            ptr
        ) = aux_data

        inst = cls(dims)
        inst.op_prods = op_prods
        inst.modes = modes
        inst.conjs = conjs
        inst.duals = duals
        inst.coeffs = coeffs
        inst.coeff_callbacks = coeff_callbacks
        inst.coeff_grad_callbacks = coeff_grad_callbacks
        inst.dtype = dtype
        inst._op_prod_types = op_prod_types
        inst._ptr = ptr
        return inst

    def _check_dtype(self, op_prod):
        """
        Check if all elementary or matrix operators have the same data type.
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

    def append(self,
               op_prod: Sequence[ElementaryOperator | MatrixOperator],
               *,
               modes: Sequence[int] | None = None,
               conjs: Sequence[bool] | None = None,
               duals: Sequence[bool] | None = None,
               coeff: float = 1.0,
               coeff_callback: cudm.WrappedScalarCallback | None = None,
               coeff_grad_callback: cudm.WrappedScalarGradientCallback | None = None
               ) -> None:
        """
        Append an elementary or matrix product to an operator term.

        Args:
            op_prod: Product of elementary or matrix operators to be appended.
            modes: Modes acted on by the operator product.
            duals: Dualities of the operator product.
            coeff: Coefficient of the operator product.
            coeff_callback: Forward callback for the coeffient.
            coeff_grad_callback: Gradient callback for the coefficient.
        """
        # Check if all elementary or matrix operators have the same data type.
        self._check_dtype(op_prod)
        self._check_and_append_op_prod_type(op_prod)

        # TODO: Check that modes have to be in dims.

        if self._op_prod_types[-1] is ElementaryOperator:
            # Check consistency of modes.
            if modes is None:
                raise ValueError("Modes acted on must be specified for elementary operators.")
            
            # FIXME: Check dims of each elementary operator
            if len(modes) != len(op_prod):
                pass
                # raise ValueError(f"Number of modes acted on {len(modes)} does not match number of operator products {len(op_prod)}.")
            
            # Check that matrix conjugations cannot be specified for elementary operators.
            if conjs is not None:
                raise ValueError("Matrix conjugations cannot be specified for elementary operators.")

        else:  # matrix operator product
            # Check consistency of conjs.
            if conjs is None:
                conjs = (False,) * len(op_prod)
            else:
                # FIXME: Check dims of each elementary operator
                if len(conjs) != len(op_prod):
                    pass
                    # raise ValueError("Number of matrix conjugations must match number of operator products.")

            # Check that modes acted on cannot be specified for matrix operators.
            if modes is not None:
                raise ValueError("Modes acted on cannot be specified for matrix operators.")

        if duals is None:
            duals = (False,) * len(op_prod)
        else:
            # FIXME: Check dims of each elementary operator
            if len(duals) != len(op_prod):
                pass
                # raise ValueError("Number of duals must match number of operator products.")

        # Populate instance attributes.
        self.op_prods.append(tuple(op_prod))
        if modes is not None:
            self.modes.append(tuple(modes))
        else:
            # Append modes here for testing purposes.
            self.modes.append(tuple(range(len(self.dims))))
        if conjs is not None:
            self.conjs.append(tuple(conjs))
        self.duals.append(tuple(duals))
        self.coeffs.append(coeff)
        self.coeff_callbacks.append(coeff_callback)
        self.coeff_grad_callbacks.append(coeff_grad_callback)

    def __getitem__(self, index: int) -> Tuple[ElementaryOperator | MatrixOperator, ...]:
        """
        Get an operator product from the operator term.
        """
        return self.op_prods[index]

    def _create(self, handle):
        """
        Create opaque handle to the operator term.
        """
        # Create opaque handle to dependent elementary or matrix operators.
        for op_prod in self.op_prods:
            for elem_op in op_prod:
                elem_op._create(handle)

        # Create opaque handle to the operator term.    
        if self._ptr is None:
            self._ptr = cudm.create_operator_term(
                handle,
                len(self.dims),
                self.dims
            )
            self.logger.debug(f"Created operator term at {hex(self._ptr)}")

            for i in range(len(self.op_prods)):
                if self._op_prod_types[i] is ElementaryOperator:
                    cudm.operator_term_append_elementary_product(
                        handle,
                        self._ptr,
                        len(self.op_prods[i]),
                        [elem_op._ptr for elem_op in self.op_prods[i]],
                        self.modes[i],
                        self.duals[i],
                        self.coeffs[i],
                        self.coeff_callbacks[i],
                        self.coeff_grad_callbacks[i]
                    )
                else:
                    cudm.operator_term_append_matrix_product(
                        handle,
                        self._ptr,
                        len(self.op_prods[i]),
                        [mat_op._ptr for mat_op in self.op_prods[i]],
                        self.conjs[i],
                        self.duals[i],
                        self.coeffs[i],
                        self.coeff_callbacks[i],
                        self.coeff_grad_callbacks[i]
                    )
            self.logger.debug(f"Appended operator products to operator term at {hex(self._ptr)}")

    def _destroy(self):
        """
        Destroy opaque handle to the operator term.
        """
        # Destroy opaque handle to the operator term.
        if self._ptr is not None:
            cudm.destroy_operator_term(self._ptr)
            self.logger.debug(f"Destroyed operator term at {hex(self._ptr)}")
            self._ptr = None


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
        self.dims: Tuple[int, ...] = tuple(dims)

        self.op_terms: List[OperatorTerm] = []
        self.duals: List[bool] = []
        self.coeffs: List[float] = []
        self.coeff_callbacks: List[cudm.WrappedScalarCallback | None] = []
        self.coeff_grad_callbacks: List[cudm.WrappedScalarGradientCallback | None] = []

        self.dtype: jnp.dtype | None = None
        self._ptr: int | None = None

    def tree_flatten(self):
        """
        Flatten the operator PyTree.
        """
        children = (self.op_terms,)
        aux_data = (
            self.dims,
            self.duals,
            self.coeffs,
            self.coeff_callbacks,
            self.coeff_grad_callbacks,
            self.dtype,
            self._ptr
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the operator PyTree.
        """
        op_terms = children[0]
        (
            dims,
            duals,
            coeffs,
            coeff_callbacks,
            coeff_grad_callbacks,
            dtype,
            ptr
        ) = aux_data

        inst = cls(dims)
        inst.op_terms = op_terms
        inst.duals = duals
        inst.coeffs = coeffs
        inst.coeff_callbacks = coeff_callbacks
        inst.coeff_grad_callbacks = coeff_grad_callbacks
        inst.dtype = dtype
        inst._ptr = ptr
        return inst

    def _check_dtype(self, op_term: OperatorTerm):
        """
        Check if all operator terms have the same data type.
        """
        if op_term.dtype is not None:  # for empty operator term, skip the check.
            if self.dtype is None:
                # If the data type is not set, set it to the data type of the first operator term.
                self.dtype = op_term.dtype
            else:
                # If the data type is set, check if the operator term has the same data type as the operator.
                if op_term.dtype != self.dtype:
                    raise ValueError("All operator terms must have the same data type.")

    def append(self,
               op_term: OperatorTerm,
               *,
               dual: bool = False,
               coeff: float = 1.0,
               coeff_callback: cudm.WrappedScalarCallback | None = None,
               coeff_grad_callback: cudm.WrappedScalarGradientCallback | None = None
               ) -> None:
        """
        Append an operator term to an operator.

        Args:
            op_term: Operator term to be appended.
            dual: Duality of the operator term.
            coeff: Coefficient of the operator term.
            coeff_callback: Forward callback for the coefficient.
            coeff_grad_callback: Gradient callback for the coefficient.
        """
        # Check if the operator term has the same data type as the operator.
        self._check_dtype(op_term)

        # Populate inst attributes.
        self.op_terms.append(op_term)
        self.duals.append(dual)
        self.coeffs.append(coeff)
        self.coeff_callbacks.append(coeff_callback)
        self.coeff_grad_callbacks.append(coeff_grad_callback)

    def __getitem__(self, index: int) -> OperatorTerm:
        """
        Get an operator term from the operator.
        """
        return self.op_terms[index]

    def _create(self, handle):
        """
        Create opaque handle to the operator.
        """
        # Create dependent operator terms.
        for op_term in self.op_terms:
            op_term._create(handle)

        # Create the current operator.
        if self._ptr is None:
            self._ptr = cudm.create_operator(
                handle,
                len(self.dims),
                self.dims
            )
            self.logger.debug(f"Created operator at {hex(self._ptr)}")

            for i in range(len(self.op_terms)):
                cudm.operator_append_term(
                    handle,
                    self._ptr,
                    self.op_terms[i]._ptr,
                    self.duals[i],
                    self.coeffs[i],
                    self.coeff_callbacks[i],
                    self.coeff_grad_callbacks[i]
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
