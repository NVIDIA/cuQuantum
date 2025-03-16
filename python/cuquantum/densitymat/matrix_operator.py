# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matrix operator."""

from typing import Sequence, Tuple
from abc import ABC, abstractmethod
import weakref
import collections

import numpy as np
import cupy as cp

from cuquantum.bindings import cudensitymat as cudm
from ._internal.utils import (
    generic_finalizer,
    register_with,
    device_ctx,
    NDArrayType,
    InvalidObjectState,
)
from .work_stream import WorkStream
from .callbacks import Callback, GPUCallback, CPUCallback
from .._internal import typemaps
from .._internal.tensor_wrapper import wrap_operand
from .._internal.utils import precondition, StreamHolder, cuda_call_ctx


__all__ = ["LocalDenseMatrixOperator"]


class MatrixOperator(ABC):

    def __init__(self, hilbert_space_dims: Sequence[int], batch_size, dtype) -> None:
        self.batch_size: int = batch_size
        self.hilbert_space_dims: Tuple[int] = tuple(hilbert_space_dims)
        self.dtype: str = dtype
        self._dtype = typemaps.NAME_TO_DATA_TYPE[self.dtype]
        self._last_compute_event: cp.cuda.Event | None = None
        self._upstream_finalizers = collections.OrderedDict()
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()
        self.callback: Callback | None = None
        self._ctx: WorkStream | None = None

    @property
    def _valid_state(self):
        return self._finalizer.alive

    def _check_valid_state(self, *args, **kwargs):
        if not self._valid_state:
            raise InvalidObjectState("The matrix operator cannot be used after resources are freed.")

    def _check_invalid_state(self, *args, **kwargs):
        if self._valid_state:
            raise InvalidObjectState("The matrix operator cannot be modified after it has been registered with the library.")

    @precondition(_check_invalid_state)
    def _set_finalizer(self):
        self._finalizer = weakref.finalize(
            self,
            generic_finalizer,
            self._ctx.logger,
            self._upstream_finalizers,
            (cudm.destroy_matrix_operator, self._ptr),
            msg=f"Destroying DenseMatrixOperator instance {self}, ptr: {self._ptr}.",
        )

    @abstractmethod
    def _instantiate(self):
        pass

    @precondition(_check_valid_state)
    @abstractmethod
    def _maybe_instantiate(self, ctx: WorkStream) -> None:
        pass

    @property
    @precondition(_check_valid_state)
    def _validated_ptr(self):
        return self._ptr

    def _sync(self) -> None:
        if self._last_compute_event:
            self._last_compute_event.synchronize()
            self._last_compute_event = None


class LocalDenseMatrixOperator(MatrixOperator):
    """
    LocalDenseMatrixOperator(data, callback=None)

    Dense matrix operator stored fully on a single device.

    Args:
        data: Data buffer for operator elements on GPU.
            Required to be fortran-contiguousand of shape ``(*hilbert_space_dims, *hilbert_space_dims, batch_size)`` with
            ``hilbert_space_dims`` a tuple of local Hilbert space dimensions defining the full Hilbert space, and batch_size an integer (for ``batch=size=1`` the last dimension may be ommitted.)
        callback: An optional inplace callback function that modifies GPU buffer.
    """

    def __init__(self, data: cp.ndarray, callback: GPUCallback | None = None) -> None:
        """
        Initialize a matrix operator.
        """
        dtype = data.dtype.name
        original_shape = data.shape
        if len(data.shape) % 2 == 0:
            # add batch dimension
            data = data.reshape(*data.shape, 1)
        batch_size = data.shape[-1]
        hilbert_space_dims: Tuple[int] = data.shape[: len(data.shape) // 2]

        if not isinstance(data, cp.ndarray):
            raise TypeError(f"LocalDenseMatrixOperator requires `data` argument to be cp.ndarray. Received {type(data)}.")
        if not data.flags["F_CONTIGUOUS"]:
            raise ValueError(f"LocalDenseMatrixOperator requires `data` argument to be Fortran-contiguous cp.ndarray.")
        if data.shape[len(hilbert_space_dims) : 2 * len(hilbert_space_dims)] != hilbert_space_dims:
            raise ValueError(
                f"LocalDenseMatrixOperator requires `data` input as tensor, with two consecutive sets of identical modes, followed optionally by a batch mode, but received a tensor of shape: {original_shape}."
            )

        if callback is not None:
            if not isinstance(callback, GPUCallback):
                raise TypeError(f"LocalDenseMatrixOperator needs to be specified as GPUCallback. Received {type(callback)}.")
            if not callback.is_inplace:
                raise ValueError("Only in-place GPU callbacks are supported.")

        super().__init__(hilbert_space_dims, batch_size, dtype)
        self.callback = callback
        self._data = wrap_operand(data)

    @property
    def device_id(self) -> int:
        """
        Return device ID if stored on GPU and ``None`` if stored on CPU.
        """
        return self._data.device_id

    @property
    def data(self) -> NDArrayType:
        """
        Data buffer of the matrix operator.
        """
        return self._data.tensor

    # @precondition(_check_invalid_state)
    def _instantiate(self, ctx):
        if ctx.device_id is not self.device_id:
            raise ValueError(
                "Using a LocalDenseMatrixOperator residing on a different device than the employed WorkStream is not supported."
            )
        self._ctx = ctx
        if self.batch_size > 1:
            self._ptr = cudm.create_matrix_operator_dense_local_batch(
                self._ctx._handle._validated_ptr,
                len(self.hilbert_space_dims),
                self.hilbert_space_dims,
                self.batch_size,
                self._dtype,
                self._data.data_ptr,
                (self.callback._get_internal_wrapper(which="tensor") if self.callback is not None else None),
            )
        elif self.batch_size == 1:
            self._ptr = cudm.create_matrix_operator_dense_local(
                self._ctx._handle._validated_ptr,
                len(self.hilbert_space_dims),
                self.hilbert_space_dims,
                self._dtype,
                self._data.data_ptr,
                (self.callback._get_internal_wrapper(which="tensor") if self.callback is not None else None),
            )
        self._set_finalizer()
        register_with(self, self._ctx, self._ctx.logger)

    def _maybe_instantiate(self, ctx: WorkStream) -> None:
        """
        Since (Dense)MatrixOperator are instantiated at creation,
        this method's main purpose is a uniform interface between Elementary and Matrix Operators.
        """
        if self._ctx is not None and self._ctx != ctx:
            raise ValueError(
                "Using an ElementaryOperator with a different WorkStream from its original WorkStream is not supported."
            )
        if not self._valid_state:
            self._instantiate(ctx)

    def to_array(self, t: float | None = None, args: cp.ndarray | None = None, device: str | int | None = None) -> cp.ndarray:
        r"""
        Return the array form of the local dense matrix operator.

        Args:
            t: Time variable in callback, only required if callback is not ``None``.
            args: Additional arguments in callback, only required if callback is not ``None``.
            device: Device on which to return the array. Defaults to ``"cpu"``.
        """
        callback_args = None if (t is None or args is None) else (t, args)

        if self.callback is None:
            if device is None:
                return self.data
            elif device == "cpu" or isinstance(device, int):
                return self._data.to(device)
        else:
            # check input args types
            if callback_args == ():
                raise ValueError(
                    "For a LocalDenseMatrixOperator with callback, callback arguments must be passed in "
                    "when converted to an array."
                )
            if self.callback.is_gpu_callback:
                params_arr = callback_args[1]
                arr = cp.empty(self.data.shape, dtype=self.dtype, order="F")
                self.callback(*callback_args, arr)
            else:
                arr = np.empty(self.data.shape, dtype=self.dtype, order="F")
                self.callback(*callback_args, arr)
            # move to target device
            if device is not None:
                return wrap_operand(arr).to(device)
            else:
                return arr
