# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Matrix operator class in cuDensityMat.
"""

import logging

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
class MatrixOperator:
    """
    PyTree class for cuDensityMat's matrix operator.
    """

    logger = logging.getLogger("cudensitymat-jax.MatrixOperator")

    def __init__(self, data: jax.Array) -> None:
        """
        Initialize a MatrixOperator object.

        Args:
            data: Data buffer of the matrix operator.
        """
        if isinstance(data, jax.Array):

            # Set data and batch size.
            if data.ndim % 2 == 0:
                # Expanding to a leading dimension 1 is necessary since we are taking 0 as the batch
                # dimension when passing to ffi_lowering.
                self.data = jnp.expand_dims(data, 0)
                self.batch_size = 1
            else:  # batched
                self.data = data
                self.batch_size = data.shape[0]

            # Set other attributes derived from data.
            self.num_modes: int = len(self.data.shape) // 2
            if self.data.shape[-2 * self.num_modes:-self.num_modes] != self.data.shape[-self.num_modes:]:
                raise ValueError("Data must have the same shape on the bra and ket modes.")
            self.mode_extents: tuple[int, ...] = self.data.shape[-self.num_modes:]
            self.dtype: jnp.dtype = self.data.dtype

        elif type(data) is object:  # data is object() during AD tracing.
            # Dummy variables for derived attributes.
            self.data: object = data
            self.batch_size: int = 1
            self.num_modes: int = 0
            self.mode_extents: tuple[int, ...] = ()
            self.dtype: jnp.dtype = jnp.dtype(float)

        # Callbacks and requires_grad are set in _create() when AD tracing is detected.
        self.requires_grad = None
        self._callback = None
        self._grad_callback = None
        self._grad_ptr: int = 0

        self._ptr: int | None = None
        self._is_elementary: bool = False

    def tree_flatten(self):
        """
        Flatten the matrix operator PyTree.
        """
        children = (self.data,)
        aux_data = (
            self.batch_size,
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
        Unflatten the matrix operator PyTree.
        """
        inst = cls.__new__(cls)
        inst.data = children[0]
        (
            inst.batch_size,
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
    def in_axes(self) -> "MatrixOperator":
        """
        Return the in_axes PyTree spec for vmapping over the batch dimension (axis 0 of data).
        """
        _, aux_data = self.tree_flatten()
        if is_vmap_traced(self.data) or self.batch_size > 1:
            in_axes_data = 0
        else:
            in_axes_data = None
        return type(self).tree_unflatten(aux_data, (in_axes_data,))

    def _copy(self) -> "MatrixOperator":
        """
        Copy the matrix operator.
        """
        mat_op = type(self).__new__(type(self))

        mat_op.data = jnp.copy(self.data)

        mat_op.batch_size = self.batch_size
        mat_op.num_modes = self.num_modes
        mat_op.mode_extents = self.mode_extents
        mat_op.dtype = self.dtype
        mat_op.requires_grad = self.requires_grad
        mat_op._callback = self._callback
        mat_op._grad_callback = self._grad_callback
        mat_op._grad_ptr = self._grad_ptr
        mat_op._is_elementary = self._is_elementary
        mat_op._ptr = self._ptr

        return mat_op

    def _create(self, handle):
        """
        Create opaque handle to the matrix operator.
        """
        # Create opaque handle to the matrix operator.
        if self._ptr is None:

            # Detect if data requires gradient and assign callback, gradient callback.
            self.requires_grad = detect_ad_traced_object(self.data)
            if self.requires_grad:
                self._callback = get_empty_tensor_callback()
                self._grad_callback = get_tensor_gradient_attachment_callback(self.data)
                self._grad_ptr = self._grad_callback.callback.tensor_grad.data.ptr

            if self.batch_size == 1:
                self._ptr = cudm.create_matrix_operator_dense_local(
                    handle,
                    self.num_modes,
                    self.mode_extents,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    self._callback,
                    self._grad_callback,
                )
            else:
                self._ptr = cudm.create_matrix_operator_dense_local_batch(
                    handle,
                    self.num_modes,
                    self.mode_extents,
                    self.batch_size,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    self._callback,
                    self._grad_callback,
                )

            self.logger.debug(f"Created matrix operator at {hex(self._ptr)}")

    def _destroy(self):
        """
        Destroy opaque handle to the matrix operator.
        """
        if self._ptr is not None:
            cudm.destroy_matrix_operator(self._ptr)
            self.logger.debug(f"Destroyed matrix operator at {hex(self._ptr)}")
            self._ptr = None
