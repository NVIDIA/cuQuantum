# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Elementary operator class in cuDensityMat.
"""

import logging
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm
from nvmath.internal import typemaps

from ..utils import (
    get_empty_tensor_callback,
    get_tensor_gradient_attachment_callback,
    detect_ad_traced_object,
    is_vmap_traced,
)


@jax.tree_util.register_pytree_node_class
class ElementaryOperator:
    """
    PyTree class for cuDensityMat's elementary operator.
    """

    logger = logging.getLogger("cudensitymat-jax.ElementaryOperator")

    def __init__(self, data: jax.Array, diag_offsets: Sequence[int] = ()) -> None:
        """
        Initialize an ElementaryOperator object.

        Args:
            data: Data buffer of the elementary operator.
            diag_offsets: Diagonal offsets of the elementary operator.
        """
        # Check consistency of tensor data.
        if isinstance(data, jax.Array):

            if len(diag_offsets) == 0:  # dense elementary operator

                # Set batch size and data.
                if data.ndim % 2 == 0:
                    # Expanding to a leading dimension 1 is necessary since we are taking 0 as the batch
                    # dimension when passing to ffi_lowering.
                    self.data = jnp.expand_dims(data, 0)
                    self.batch_size = 1
                else:  # batched elementary operator
                    # TODO: Batch dimension is assumed to be dimension 0. This constraint could be
                    # relaxed in the future.
                    self.data = data
                    self.batch_size = data.shape[0]

                self.sparsity = cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_NONE

            else:  # multidiagonal elementary operator

                # Set batch size and data.
                if data.ndim == 2:
                    # Expanding to a leading dimension 1 is necessary since we are taking 0 as the batch
                    # dimension when passing to ffi_lowering.
                    self.data = jnp.expand_dims(data, 0)
                    self.batch_size = 1
                elif data.ndim == 3:
                    # TODO: Batch dimension is assumed to be dimension 0. This constraint could be
                    # relaxed in the future.
                    self.data = data
                    self.batch_size = data.shape[0]
                else:
                    raise ValueError("Only single-mode multidiagonal elementary operator is supported.")

                # Check diagonal offsets.
                if len(diag_offsets) != len(set(diag_offsets)):
                    raise ValueError("Diagonal offsets cannot contain duplicate elements.")
                if data.shape[-1] != len(diag_offsets):
                    raise ValueError("Number of columns in data does not match length of diagonal offsets.")

                self.sparsity = cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL

            # The following attributes are set in the same way for dense and multidiagonal elementary 
            # operators. Also check that bra and ket modes have the same shape.
            self.num_modes = data.ndim // 2
            if len(diag_offsets) == 0:  # only check on dense elementary operators
                bra_modes = self.data.shape[-2 * self.num_modes:-self.num_modes]
                ket_modes = self.data.shape[-self.num_modes:]
                if bra_modes != ket_modes:
                    raise ValueError("Dense elementary operator data must have the same shape on the bra and ket modes.")
            self.mode_extents = self.data.shape[1:self.num_modes + 1]  # skip the batch dimension.
            self.dtype: jnp.dtype = self.data.dtype

        elif type(data) is object:  # data is object() during AD tracing.
            # Dummy variables for derived attributes.
            self.batch_size: int = 1
            self.data: object = data
            self.num_modes: int = 0
            self.mode_extents: tuple[int, ...] = ()
            self.dtype: jnp.dtype = jnp.dtype(float)

        # Callbacks and diagonal offsets.
        self.diag_offsets: tuple[int, ...] = tuple(diag_offsets)
        
        self.requires_grad = None
        self._callback = None
        self._grad_callback = None
        self._grad_ptr: int = 0

        self._ptr: int | None = None
        self._is_elementary: bool = True

    def tree_flatten(self):
        """
        Flatten the elementary operator PyTree.
        """
        children = (self.data,)
        aux_data = (
            self.diag_offsets,
            self.batch_size,
            self.sparsity,
            self.num_modes,
            self.mode_extents,
            self.dtype,
            self.requires_grad,
            self._callback,
            self._grad_callback,
            self._grad_ptr,
            self._is_elementary,
            self._ptr,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the elementary operator PyTree.
        """
        inst = cls.__new__(cls)
        inst.data = children[0]
        (
            inst.diag_offsets,
            inst.batch_size,
            inst.sparsity,
            inst.num_modes,
            inst.mode_extents,
            inst.dtype,
            inst.requires_grad,
            inst._callback,
            inst._grad_callback,
            inst._grad_ptr,
            inst._is_elementary,
            inst._ptr,
        ) = aux_data
        return inst

    @property
    def in_axes(self) -> "ElementaryOperator":
        """
        Return the in_axes PyTree spec for vmapping over the batch dimension (axis 0 of data).
        """
        _, aux_data = self.tree_flatten()
        if is_vmap_traced(self.data) or self.batch_size > 1:
            in_axes_data = 0
        else:
            in_axes_data = None
        return type(self).tree_unflatten(aux_data, (in_axes_data,))

    def _copy(self) -> "ElementaryOperator":
        """
        Copy the elementary operator.
        """
        elem_op = type(self).__new__(type(self))

        elem_op.data = jnp.copy(self.data)

        elem_op.diag_offsets = self.diag_offsets
        elem_op.batch_size = self.batch_size
        elem_op.sparsity = self.sparsity
        elem_op.num_modes = self.num_modes
        elem_op.mode_extents = self.mode_extents
        elem_op.dtype = self.dtype
        elem_op.requires_grad = self.requires_grad
        elem_op._callback = self._callback
        elem_op._grad_callback = self._grad_callback
        elem_op._grad_ptr = self._grad_ptr
        elem_op._is_elementary = self._is_elementary
        elem_op._ptr = self._ptr

        return elem_op

    def _create(self, handle):
        """
        Create opaque handle to the elementary operator.
        """
        # Create opaque handle to the elementary operator
        if self._ptr is None:

            # Detect if data requires gradient and assign callback, gradient callback.
            self.requires_grad = detect_ad_traced_object(self.data)
            if self.requires_grad:
                self._callback = get_empty_tensor_callback()
                self._grad_callback = get_tensor_gradient_attachment_callback(self.data)
                self._grad_ptr = self._grad_callback.callback.tensor_grad.data.ptr

            if self.batch_size == 1:
                self._ptr = cudm.create_elementary_operator(
                    handle,
                    self.num_modes,
                    self.mode_extents,
                    self.sparsity,
                    len(self.diag_offsets),
                    self.diag_offsets,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    self._callback,
                    self._grad_callback
                )
            else:
                self._ptr = cudm.create_elementary_operator_batch(
                    handle,
                    self.num_modes,
                    self.mode_extents,
                    self.batch_size,
                    self.sparsity,
                    len(self.diag_offsets),
                    self.diag_offsets,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    self._callback,
                    self._grad_callback,
                )

            self.logger.debug(f"Created elementary operator at {hex(self._ptr)}")

    def _destroy(self):
        """
        Destroy opaque handle to the elementary operator.
        """
        if self._ptr is not None:
            cudm.destroy_elementary_operator(self._ptr)
            self.logger.debug(f"Destroyed elementary operator at {hex(self._ptr)}")
            self._ptr = None
