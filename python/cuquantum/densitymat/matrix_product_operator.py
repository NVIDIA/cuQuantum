# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matrix product operator (MPO)."""

from typing import Sequence, List, Tuple
import weakref
import collections

import numpy as np
import cupy as cp

from nvmath.internal.utils import precondition
from nvmath.internal import typemaps
from cuquantum.bindings import cudensitymat as cudm
from ._internal.utils import (
    generic_finalizer,
    register_with,
    InvalidObjectState,
)
from .work_stream import WorkStream
from .callbacks import Callback, GPUCallback


__all__ = ["MatrixProductOperator"]


class MatrixProductOperator:
    """
    MatrixProductOperator(tensor_data, hilbert_space_dims, bond_dims, callbacks=None, gradient_callbacks=None)

    Matrix product operator (MPO) defined by a chain of tensors acting on a tensor product Hilbert space.

    Each site tensor has up to 4 modes with the following ordering (Fortran contiguous):

    - Leftmost site (i=0): ``(phys_ket, bond_right, phys_bra)``
    - Interior sites: ``(bond_left, phys_ket, bond_right, phys_bra)``
    - Rightmost site (i=N-1): ``(bond_left, phys_ket, phys_bra)``

    Here ``phys_ket`` and ``phys_bra`` have the extent of the local Hilbert space dimension at that site,
    while ``bond_left`` and ``bond_right`` have the extents of the corresponding bond dimensions.

    Args:
        tensor_data: A list of GPU data buffers (``cp.ndarray``), one per site tensor of the MPO.
            Each tensor must be Fortran contiguous.
        hilbert_space_dims: A tuple of the local Hilbert space dimensions.
        bond_dims: A tuple of the bond dimensions between adjacent sites. For open boundary conditions,
            the length must be equal to ``len(hilbert_space_dims) - 1``.
        callbacks: An optional list of :class:`GPUCallback` instances (or ``None`` entries), one per site tensor.
        gradient_callbacks: An optional list of :class:`GPUCallback` instances (or ``None`` entries), one per site tensor.

    Examples:
        >>> import cupy as cp
        >>> from cuquantum.densitymat import WorkStream, MatrixProductOperator

        Construct an MPO for a 3-site system with local dimension 2 and bond dimension 4

        >>> hilbert_space_dims = (2, 2, 2)
        >>> bond_dims = (4, 4)
        >>> tensors = [
        ...     cp.zeros((2, 4, 2), dtype="complex128", order="F"),    # left boundary:  (d_ket, bond_R, d_bra)
        ...     cp.zeros((4, 2, 4, 2), dtype="complex128", order="F"), # bulk:           (bond_L, d_ket, bond_R, d_bra)
        ...     cp.zeros((4, 2, 2), dtype="complex128", order="F"),    # right boundary: (bond_L, d_ket, d_bra)
        ... ]
        >>> mpo = MatrixProductOperator(tensors, hilbert_space_dims, bond_dims)
    """

    def __init__(
        self,
        tensor_data: List[cp.ndarray],
        hilbert_space_dims: Sequence[int],
        bond_dims: Sequence[int],
        callbacks: List[GPUCallback | None] | None = None,
        gradient_callbacks: List[GPUCallback | None] | None = None,
    ) -> None:
        num_sites = len(hilbert_space_dims)
        if len(tensor_data) != num_sites:
            raise ValueError(
                f"Expected {num_sites} tensor data buffers (one per site), got {len(tensor_data)}."
            )
        if len(bond_dims) != num_sites - 1:
            raise ValueError(
                f"Expected {num_sites - 1} bond dimensions for open boundary condition, got {len(bond_dims)}."
            )

        dtype = tensor_data[0].dtype.name
        device_id = tensor_data[0].device
        for i, t in enumerate(tensor_data):
            if not isinstance(t, cp.ndarray):
                raise TypeError(f"Tensor {i} must be a cp.ndarray, got {type(t)}.")
            if not t.flags["F_CONTIGUOUS"]:
                raise ValueError(f"Tensor {i} must be Fortran contiguous.")
            if t.dtype.name != dtype:
                raise ValueError(
                    f"Tensor {i} dtype {t.dtype.name} does not match tensor 0 dtype {dtype}."
                )
            if t.device != device_id:
                raise ValueError(
                    f"Tensor {i} is on device {t.device}, expected device {device_id}."
                )

        if callbacks is not None:
            if len(callbacks) != num_sites:
                raise ValueError(
                    f"Expected {num_sites} callbacks (one per site), got {len(callbacks)}."
                )
            for i, cb in enumerate(callbacks):
                if cb is not None and not isinstance(cb, GPUCallback):
                    raise TypeError(f"Callback {i} must be a GPUCallback or None, got {type(cb)}.")

        if gradient_callbacks is not None:
            if len(gradient_callbacks) != num_sites:
                raise ValueError(
                    f"Expected {num_sites} gradient callbacks (one per site), got {len(gradient_callbacks)}."
                )
            for i, cb in enumerate(gradient_callbacks):
                if cb is not None and not isinstance(cb, GPUCallback):
                    raise TypeError(f"Gradient callback {i} must be a GPUCallback or None, got {type(cb)}.")

        self.hilbert_space_dims: Tuple[int] = tuple(hilbert_space_dims)
        self.bond_dims: Tuple[int] = tuple(bond_dims)
        self.dtype: str = dtype
        self._dtype = typemaps.NAME_TO_DATA_TYPE[self.dtype]
        self._tensor_data: List[cp.ndarray] = list(tensor_data)
        self.callbacks: List[GPUCallback | None] | None = callbacks
        self.gradient_callbacks: List[GPUCallback | None] | None = gradient_callbacks

        self.batch_size: int = 1

        self._ctx: WorkStream | None = None
        self._ptr = None
        self._last_compute_event: cp.cuda.Event | None = None
        self._upstream_finalizers = collections.OrderedDict()
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()

    @property
    def num_sites(self) -> int:
        """Number of sites in the MPO."""
        return len(self.hilbert_space_dims)

    @property
    def data(self) -> List[cp.ndarray]:
        """
        The list of MPO tensor data buffers, one per site.
        """
        return self._tensor_data

    @property
    def device_id(self) -> int:
        """Device ID of the tensor data."""
        return self._tensor_data[0].device.id

    @property
    def has_gradient(self) -> bool:
        """Whether any site tensor has a gradient callback."""
        if self.gradient_callbacks is None:
            return False
        return any(cb is not None for cb in self.gradient_callbacks)

    @property
    def _valid_state(self):
        return self._finalizer.alive

    def _check_valid_state(self, *args, **kwargs):
        if not self._valid_state:
            raise InvalidObjectState("The matrix product operator cannot be used after resources are freed.")

    @property
    @precondition(_check_valid_state)
    def _validated_ptr(self):
        return self._ptr

    def _sync(self) -> None:
        if self._last_compute_event:
            self._last_compute_event.synchronize()
            self._last_compute_event = None

    def _instantiate(self, ctx: WorkStream) -> None:
        if ctx.device_id != self.device_id:
            raise ValueError(
                "MatrixProductOperator tensor data resides on a different device than the WorkStream."
            )
        self._ctx = ctx

        tensor_ptrs = [t.data.ptr for t in self._tensor_data]

        wrapped_callbacks = None
        if self.callbacks is not None:
            wrapped_callbacks = [
                cb._get_internal_wrapper(which="tensor") if cb is not None else None
                for cb in self.callbacks
            ]

        wrapped_gradient_callbacks = None
        if self.gradient_callbacks is not None:
            wrapped_gradient_callbacks = [
                cb._get_internal_gradient_wrapper(which="tensor") if cb is not None else None
                for cb in self.gradient_callbacks
            ]

        self._ptr = cudm.create_matrix_product_operator(
            self._ctx._handle._validated_ptr,
            len(self.hilbert_space_dims),
            self.hilbert_space_dims,
            0,  # CUDENSITYMAT_BOUNDARY_CONDITION_OPEN
            self.bond_dims,
            self._dtype,
            tensor_ptrs,
            wrapped_callbacks,
            wrapped_gradient_callbacks,
        )
        self._finalizer = weakref.finalize(
            self,
            generic_finalizer,
            self._ctx.logger,
            self._upstream_finalizers,
            (cudm.destroy_matrix_product_operator, self._ptr),
            msg=f"Destroying MatrixProductOperator instance {self}, ptr: {self._ptr}.",
        )
        register_with(self, self._ctx, self._ctx.logger)

    def _maybe_instantiate(self, ctx: WorkStream) -> None:
        """
        Instantiate the MPO if it hasn't been instantiated yet.
        """
        if self._ctx is not None and self._ctx != ctx:
            raise ValueError(
                "Using a MatrixProductOperator with a different WorkStream from its original WorkStream is not supported."
            )
        if not self._valid_state:
            self._instantiate(ctx)
