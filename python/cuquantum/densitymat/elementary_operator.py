# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Elementary operator."""

from typing import Callable, Sequence, Union, Tuple
from abc import ABC, abstractmethod
import weakref
import collections
from numbers import Number
from operator import add, sub

import numpy as np
import cupy as cp

from cuquantum.bindings import cudensitymat as cudm
from ._internal.utils import (
    generic_finalizer,
    register_with,
    optimize_strides,
    maybe_move_array,
    device_ctx,
    device_ctx_from_array,
    transpose_bipartite_tensor,
    matricize_bipartite_tensor,
    dense_batched_matmul,
    NDArrayType,
    InvalidObjectState,
    check_binary_tensor_shape,
)
from .work_stream import WorkStream
from .callbacks import Callback, GPUCallback, CPUCallback
from .._internal import typemaps
from .._internal.tensor_wrapper import wrap_operand
from .._internal.utils import precondition, StreamHolder, cuda_call_ctx


__all__ = ["DenseOperator", "MultidiagonalOperator"]

ElementaryOperatorType = Union["DenseOperator", "MultidiagonalOperator"]


class ElementaryOperator(ABC):
    """
    Elementary operator abstract base class.
    """

    __array_ufunc__ = None

    def __init__(self, data: NDArrayType, callback: Callback | None = None, copy: bool = True) -> None:
        """
        Initialize an elementary operator from data buffer and callback.
        
        A copy of `data` will be made unless ``copy=False``.
        If ``copy=False``, an error will be raised if `data` is not a Fortran-contiguous `cp.ndarray`.
        """
        # Input attributes
        if not isinstance(callback, (Callback, type(None))):
            raise TypeError("Callback need to be passed as an instance of CPUCallback or GPUCallback.")
        self.callback: Callback | None = callback
        self.dtype: str = data.dtype.name

        # Internal attributes
        if not isinstance(data, NDArrayType):
            raise TypeError("`ElementaryOperator` instance requires a cp.ndarray or np.ndarray for its `data` argument.")
        if not copy:
            if not isinstance(data, cp.ndarray):
                raise TypeError("`ElementaryOperator` instance requires a cp.ndarray for its `data` argument when ``copy=False``.")
            if not data.flags["F_CONTIGUOUS"]:
                raise ValueError("`ElementaryOperator` instance requires a Fortran-ordered cp.ndarray for its `data` argument when ``copy=False``.")
        else:
            data = data.copy(order="F")
        self._data = wrap_operand(data)
        self._dtype = typemaps.NAME_TO_DATA_TYPE[self.dtype]

        self._ctx: WorkStream = None
        self._ptr = None

        self._last_compute_event = None
        self._upstream_finalizers = collections.OrderedDict()
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()

    @property
    @abstractmethod
    def _sparsity_and_diagonal_offsets(self):
        pass

    @property
    def data(self) -> NDArrayType:
        """
        Data buffer of the elementary operator.
        """
        return self._data.tensor

    @property
    def device_id(self) -> None | int:
        """
        Return device ID if stored on GPU and ``None`` if stored on CPU.
        """
        return self._data.device_id

    @abstractmethod
    def to_array(self, t: float | None, args: Sequence | None):
        pass


    @abstractmethod
    def __mul__(self, scalar: Number):
        pass

    @abstractmethod
    def __rmul__(self, scalar: Number):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __matmul__(self, other):
        pass

    @abstractmethod
    def dag(self):
        pass

    def _check_scalar_operation_compability(self, scalar, what: str = ""):
        if not isinstance(scalar, (Number, NDArrayType)):
            raise TypeError(f"Cannot multiply {type(self).__name__} with {type(scalar).__name__}.")

    def _check_binary_operation_compability(self, other: "ElementaryOperator", what: str = ""):
        if not isinstance(other, ElementaryOperator):
            raise TypeError(f"Cannot perform {what} between {type(self).__name__} and {type(other).__name__}.")

        if self.shape != other.shape:
            raise ValueError(
                f"Cannot perform {what} between {type(self).__name__} and {type(other).__name__} "
                f"due to incompatible underlying tensor shapes: {self.shape} and {other.shape}."
            )

        if not (self.batch_size == 1 or other.batch_size == 1) and self.batch_size != other.batch_size:
            raise ValueError(
                f"Cannot perform {what} between {type(self).__name__} and {type(other).__name__} "
                f"with mismatching batch dimensions: {self.batch_size} and {other.batch_size}."
            )

        if not (self.device_id is None or other.device_id is None) and self.device_id != other.device_id:
            raise ValueError(
                f"Cannot perform {what} between {type(self).__name__} and {type(other).__name__} "
                f"based on arrays from different devices: device {self.device_id} and device {other.device_id}."
            )

        if self.callback or other.callback:
            is_out_of_place = True if (self.callback is None or not (self.callback.is_inplace)) else False
            is_out_of_place *= True if (other.callback is None or not (other.callback.is_inplace)) else False
            if not is_out_of_place:
                raise ValueError(f"Cannot perform {what} between ElementaryOperators with inplace callbacks.")
            if isinstance(self.callback, GPUCallback) or isinstance(other.callback, GPUCallback):
                raise ValueError(
                    f"Performing a binary operation ({what}) between ElementaryOperators is not supported "
                    "if either of them has a GPUCallback."
                )

    def _check_valid_state(self, *args, **kwargs):
        if not self._valid_state:
            raise InvalidObjectState("The tensor operator cannot be used after resources are freed.")

    def _sync(self) -> None:
        if self._last_compute_event:
            self._last_compute_event.synchronize()
            self._last_compute_event = None

    @property
    def _valid_state(self):
        return self._finalizer.alive

    @property
    @precondition(_check_valid_state)
    def _validated_ptr(self):
        return self._ptr

    def _maybe_instantiate(self, ctx: WorkStream) -> None:
        """
        Instantiate this instance if it hasn't been instantiated yet.
        """
        if self._ctx is not None:
            if self._ctx != ctx:
                raise ValueError(
                    "Using an ElementaryOperator with a different WorkStream from its original WorkStream is not supported."
                )
        if not self._ctx:
            self._ctx = ctx
            if self._data.device == "cpu":
                self._data = wrap_operand(self._data.to(self._ctx.device_id, stream_holder=self._ctx._stream_holder))
            else:
                try:
                    assert self.device_id == self._ctx.device_id
                except AssertionError as e:
                    raise RuntimeError("Device id of input array does not match device id of library context.") from e
            self._instantiate()
            register_with(self, self._ctx, self._ctx.logger)

    def _instantiate(self) -> None:
        """
        Instantiate an ElementaryOperator.
        """
        # synchronize both current stream and context stream to avoid inplace mutation of static components due to unfinished operations
        # this implies that ElementaryOperators operate on the current stream, but semantics are not super clear so we also cover the case that the user switched current stream in the meantime but was expecting stream semantics to be operational w.r.t. context
        self._ctx._stream_holder.obj.synchronize()
        cp.cuda.get_current_stream().synchronize()

        sparsity, num_diagonals, diagonal_offsets = self._sparsity_and_diagonal_offsets
        if self.batch_size > 1:
            self._ptr = cudm.create_elementary_operator_batch(
                self._ctx._handle._validated_ptr,
                self.num_modes,
                self.mode_dims,
                self.batch_size,
                sparsity,
                num_diagonals,
                diagonal_offsets,
                self._dtype,
                self._data.data_ptr,
                (self.callback._get_internal_wrapper("tensor") if self.callback is not None else None),
            )
        elif self.batch_size == 1:
            self._ptr = cudm.create_elementary_operator(
                self._ctx._handle._validated_ptr,
                self.num_modes,
                self.mode_dims,
                sparsity,
                num_diagonals,
                diagonal_offsets,
                self._dtype,
                self._data.data_ptr,
                (self.callback._get_internal_wrapper("tensor") if self.callback is not None else None),
            )
        else:
            raise RuntimeError(f"Unsupported batch size {self.batch_size} of ElementaryOperator.")
        self._finalizer = weakref.finalize(
            self,
            generic_finalizer,
            self._ctx.logger,
            self._upstream_finalizers,
            (cudm.destroy_elementary_operator, self._ptr),
            msg=f"Destroying ElementaryOperator instance {self}, ptr: {self._ptr}.",
        )

    def _prepare_scalar_for_unary_op(self, scalar):
        if isinstance(scalar, NDArrayType):
            if scalar.shape[0] > 1 and self.batch_size > 1 and scalar.shape[0] != self.batch_size:
                raise ValueError(
                    f"Batch size of scalar multiplication {scalar.shape} is incompatible with "
                    f"ElementaryOperator's batch size, {self.batch_size}."
                )
            scalar, data_was_moved = maybe_move_array(scalar, self.data)
            # stream sync only necessary if we are moving from GPU to CPU, since numpy is not async
            if data_was_moved and self.device_id is None:
                cp.cuda.get_current_stream().synchronize()
        return scalar


class DenseOperator(ElementaryOperator):
    """
    DenseOperator(data, callback=None, copy=True)

    Dense elementary operator from data buffer and optional callback.

    Args:
        data: Data buffer for operator elements.
        callback: An inplace or out-of-place callback function that modifies CPU or GPU buffer.

    .. note::
        - If number of dimensions in ``data`` is odd, the last dimension is assumed to be the batch dimension.
        - If ``copy=True``, a copy will be created on the data buffer and can be accessed through the :attr:`data` attribute. Note that if a np.ndarray is passed, it will be copied to GPU at a later stage.
        - If ``copy=False``, the provided data buffer is required to be a cp.ndarray and Fortran-contiguous.
        - The current underlying data buffer is accessible via the :attr:`data` attribute. Modification of the underlying data buffer by the user will lead to undefined behaviour.
        - If an out-of-place callback is provided, it needs to return an array needs that is consistent with the provided data buffer in terms of shape and data type.
        - If an inplace callback is provided, it needs perform an inplace modification on an array that is provided as its third positional argument.
        - The data buffer will be updated when this instance is involved in a ``compute`` method of an :class:`Operator` or :class:`OperatorAction` if a callback is passed.

    Examples:

        >>> import numpy as np
        >>> from cuquantum.densitymat import DenseOperator

        Suppose we want to construct a creation operator on a Hilbert space of dimension 3 as a ``DenseOperator``. It can be constructed from the data buffer directly as

        >>> data = np.array([
        >>>     [0, 0, 0],
        >>>     [1.0, 0, 0],
        >>>     [0, np.sqrt(2), 0],
        >>> ])
        >>> dense_op = DenseOperator(data)
    """

    def __init__(self, data: NDArrayType, callback: Callback | None = None, copy: bool = True) -> None:
        """
        Initialize a dense elementary operator from data buffer and optional callback.
        """
        if len(data.shape) % 2 == 0:
            # add batch dimension
            data = data.reshape(*data.shape, 1)
        super().__init__(data, callback, copy)
        self.shape, self.batch_size = (data.shape[:-1], data.shape[-1])
        self.num_modes = len(self.shape) // 2
        self.mode_dims = self.shape[: self.num_modes]

    @property
    def _sparsity_and_diagonal_offsets(self):
        return cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_NONE, 0, 0

    def to_array(
        self,
        t: float | None = None,
        args: NDArrayType | None = None,
        device: str | int | None = None,
    ) -> NDArrayType:
        r"""
        Return the array form of the dense elementary operator.

        Args:
            t: Time variable in callback, only required if callback is not ``None``.
            args: Additional arguments in callback, only required if callback is not ``None``.
            device: Device on which to return the array. Defaults to ``"cpu"``.

        .. note::
            - If the device is not specified, an `ElementaryOperator` without callback will return a reference to its current underlying data. Otherwise, the return location will match that of the `Callback` instance.
            - If this instance has a callback, the callback arguments `t` and `args` are required.
            - This call is blocking if it involves device-to-host or device-to-device transfer. Otherwise, it is stream-ordered on the current stream.
        """
        callback_args = None if (t is None or args is None) else (t, args)

        if self.callback is None:
            if device is None:
                return self.data
            elif device == "cpu" or isinstance(device, int):
                return self._data.to(device)
        else:
            # check input args types
            if callback_args is None:
                raise ValueError(
                    "For a DenseOperator with callback, callback arguments must be passed in " "when converted to an array."
                )
            if self.callback.is_inplace:
                if self.callback.is_gpu_callback:
                    params_arr = callback_args[1]
                    _params_arr = wrap_operand(params_arr)
                    params_arr = _params_arr.to(cp.cuda.Device().id)
                    arr = cp.empty(self.data.shape, dtype=self.dtype, order="F")
                    self.callback(callback_args[0], params_arr, arr)
                else:
                    arr = np.empty(self.data.shape, dtype=self.dtype, order="F")
                    _params_arr = wrap_operand(params_arr)
                    params_arr = _params_arr.to("cpu")
                    self.callback(callback_args[0], params_arr, arr)
            else:
                params_arr = callback_args[1]
                _params_arr = wrap_operand(params_arr)
                if self.callback.is_gpu_callback:
                    params_arr = _params_arr.to(cp.cuda.Device().id)
                else:
                    params_arr = _params_arr.to("cpu")
                arr = self.callback(callback_args[0], params_arr).reshape(self.data.shape)

            # move to target device
            if device is not None:
                return wrap_operand(arr).to(device)
            else:
                return arr

    @staticmethod
    def _unary_operation(operation):
        def _operation(dense_op):
            # safe to operate on tensor regardless of whether entries they are initialized since currently only used for conj and mul
            with device_ctx_from_array(dense_op.data):
                _data = operation(dense_op.data)
                if dense_op.callback is None:
                    data = _data
                    callback = None
                else:
                    if dense_op.callback.is_inplace:
                        raise ValueError("Unary operations are not supported for ElementaryOperators with inplace callbacks.")
                    data = dense_op._data.module.empty_like(
                        _data, dtype=dense_op.dtype
                    )  # FIXME: dtype should be inferable from _data? by empty_like
                    callback = type(dense_op.callback)(
                        lambda t, args: operation(dense_op.callback(t, args).reshape(dense_op.data.shape))
                    )
            return DenseOperator(data, callback)

        return _operation

    @precondition(ElementaryOperator._check_scalar_operation_compability)
    def __mul__(self, scalar: Union[Number, NDArrayType]) -> "DenseOperator":
        """
        Multiply this instance with a scalar on the left.
        """
        scalar = self._prepare_scalar_for_unary_op(scalar)
        if self.device_id is None:
            return DenseOperator._unary_operation(lambda x: x * scalar)(self)
        else:
            with cp.cuda.Device(self.device_id):
                return DenseOperator._unary_operation(lambda x: x * scalar)(self)

    @precondition(ElementaryOperator._check_scalar_operation_compability)
    def __rmul__(self, scalar: Union[Number, np.ndarray]) -> "DenseOperator":
        """
        Multiply this instance with a scalar on the right.
        """
        return self.__mul__(scalar)

    @_unary_operation
    def _dag(data):
        return transpose_bipartite_tensor(data).conj()

    def dag(self) -> "DenseOperator":
        """
        Return the conjugate complex transpose of this instance.
        """
        return self._dag()

    @staticmethod
    def _binary_operation(operation):
        def _operation(dense_op1, dense_op2):
            move_op1 = dense_op1.device_id is None and dense_op2.device_id is not None
            move_op2 = dense_op2.device_id is None and dense_op1.device_id is not None
            template_op = dense_op1
            if move_op1:
                template_op = dense_op2
            with device_ctx_from_array(template_op.data):
                if dense_op1.callback is None and dense_op2.callback is None:
                    data = operation(
                        maybe_move_array(dense_op1.data, template_op.data)[0],
                        maybe_move_array(dense_op2.data, template_op.data)[0],
                    )
                    callback = None
                else:  # result is dynamic DenseOperator
                    data = template_op._data.module.empty(
                        shape=check_binary_tensor_shape(dense_op1.data, dense_op2.data),
                        dtype=np.promote_types(dense_op1.dtype, dense_op2.dtype),
                        order="F",
                    )
                    callback_type = type(dense_op1.callback) if dense_op1.callback else type(dense_op2.callback)
                    if dense_op2.callback is not None and dense_op1.callback is None:
                        data_consumable_by_callback = dense_op1._data.to(device="cpu")
                        callback = lambda t, args: operation(
                            data_consumable_by_callback,
                            dense_op2.callback(t, args).reshape(dense_op2.data.shape),
                        )
                    elif dense_op1.callback is not None and dense_op2.callback is None:
                        data_consumable_by_callback = dense_op2._data.to(device="cpu")
                        callback = lambda t, args: operation(
                            dense_op1.callback(t, args).reshape(dense_op1.data.shape),
                            data_consumable_by_callback,
                        )
                    else:
                        # both inputs are dynamic, constrained to be CPUCallbacks by prior checks
                        # reshape to shape including batch dimension in order to broadcast over batch dimension also if unbatched operator callback return array lacks batch dimension
                        callback = lambda t, args: operation(
                            dense_op1.callback(t, args).reshape(dense_op1.data.shape),
                            dense_op2.callback(t, args).reshape(dense_op2.data.shape),
                        )
                    callback = callback_type(callback, is_inplace=False)

            return DenseOperator(data, callback)

        return _operation

    @precondition(ElementaryOperator._check_binary_operation_compability, what="addition")
    def __add__(self, other: ElementaryOperatorType) -> "DenseOperator":
        """
        Add an elementary operator to this instance and return a new :class:`DenseOperator`.
        """
        if isinstance(other, DenseOperator):
            if self.device_id is None:
                return DenseOperator._binary_operation(add)(self, other)
            else:
                with cp.cuda.Device(self.device_id):
                    return DenseOperator._binary_operation(add)(self, other)
        else:
            return self + other.to_dense()

    @precondition(ElementaryOperator._check_binary_operation_compability, what="subtraction")
    def __sub__(self, other: ElementaryOperatorType) -> "DenseOperator":
        """
        Subtract an elementary operator from this instance and return a new :class:`DenseOperator`.
        """
        if isinstance(other, DenseOperator):
            if self.device_id is None:
                return DenseOperator._binary_operation(sub)(self, other)
            else:
                with cp.cuda.Device(self.device_id):
                    return DenseOperator._binary_operation(sub)(self, other)
        else:
            return self - other.to_dense()

    @_binary_operation
    def _matmul(a, b):
        return dense_batched_matmul(a, b)

    @precondition(ElementaryOperator._check_binary_operation_compability, what="matrix multiplication")
    def __matmul__(self, other: ElementaryOperatorType) -> "DenseOperator":
        """
        Perform matrix multiplication between this instance and an elementary operator and return a new :class:`DenseOperator`.
        """
        if isinstance(other, DenseOperator):
            if self.device_id is None:
                return self._matmul(other)
            else:
                with cp.cuda.Device(self.device_id):
                    return self._matmul(other)
        else:
            return self @ other.to_dense()


class MultidiagonalOperator(ElementaryOperator):
    """
    MultidiagonalOperator(data, offsets, callback=None, copy=True)

    Multidiagonal single-mode operator from data buffer, offsets and optional callback.

    Args:
        data: Data buffer for diagonal elements, of shape ``(mode_dimension, num_diagonals)`` and an optional batch dimension at the end.
        offsets: The diagonal offsets of length ``num_diagonals``.
        callback: An inplace or out-of-place callback function that modifies CPU or GPU buffer.

    .. note::
        - ``data`` should be of shape ``(mode_dimension, num_diagonals)`` or ``(mode_dimension, num_diagonals, batch_size)``.
        - The data layout is different from :class:`scipy.sparse.dia_matrix` and :class:`cupyx.scipy.sparse.dia_matrix`.
          In this class, the elements of the ``offsets[i]``-th diagonal corresponds to the ``i``-th column of the input data buffer read from the top of the column.
        - If ``copy=True``, a copy will be created on the data buffer and can be accessed through the :attr:`data` attribute. Note that if a np.ndarray is passed, it will be copied to GPU at a later stage.
        - If ``copy=False``, the provided data buffer is required to be a cp.ndarray and Fortran-contiguous.
        - The current underlying data buffer is accessible via the :attr:`data` attribute. Modification of the underlying data buffer by the user will lead to undefined behaviour.
        - If an out-of-place callback is provided, it needs to return an array needs that is consistent with the provided data buffer in terms of shape and data type.
        - If an inplace callback is provided, it needs perform an inplace modification on an array that is provided as its third positional argument.
        - The data buffer will be updated when this instance is involved in a ``compute`` method of an :class:`Operator` or :class:`OperatorAction` if a callback is passed.

    Examples:

        >>> import numpy as np
        >>> from cuquantum.densitymat import MultidiagonalOperator

        Suppose we want to construct a creation operator on a Hilbert space of dimension 3 as a ``MultidiagonalOperator``. It can be constructed from the data buffer and diagonal offsets as

        >>> data = np.array([[1], [np.sqrt(2)], [0]]) # the last element doesn't matter
        >>> offsets = [-1]
        >>> dia_op = MultidiagonalOperator(data, offsets)

        If we already have the elementary operator in :class:`scipy.sparse.dia_matrix` format, e.g,

        >>> dia_matrix = scipy.sparse.dia_matrix(...) # put your data here

        We can create a ``MultidiagonalOperator`` with the following:

        >>> offsets = list(dia_matrix.offsets)
        >>> data = np.zeros((dia_matrix.shape[0], len(offsets)), dtype=dia_matrix.dtype)
        >>> for i, offset in enumerate(offsets):
        >>>    end = None if offset == 0 else -abs(offset)
        >>>    data[:end, i] = dia_matrix.diagonal(offset)
        >>> dia_op = MultidiagonalOperator(data, offsets)
    """

    def __init__(self, data: NDArrayType, offsets: Sequence[int], callback: Callback | None = None, copy: bool = True) -> None:
        """
        Initialize a multidiagonal single-mode operator from data buffer, offsets and optional callback.
        """
        if len(offsets) != len(set(offsets)):
            raise ValueError("Offsets cannot contain duplicate elements.")
        if data.shape[1] != len(offsets):
            raise ValueError("Number of columns in data does not match length of offsets.")
        if len(data.shape) == 2:
            # add batch dimension
            data = data.reshape(*data.shape, 1)

        mode_dim, self.num_diagonals, self.batch_size = data.shape
        super().__init__(data, callback, copy)
        self.offsets = list(offsets)

        self.shape = (mode_dim, mode_dim)
        self.num_modes = 1
        self.mode_dims = (mode_dim,)

    @property
    def _sparsity_and_diagonal_offsets(self):
        return (
            cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL,
            self.num_diagonals,
            self.offsets,
        )

    @staticmethod
    def _unary_operation(return_type: str = "multidiagonal"):
        assert return_type in ["multidiagonal", "dense"]

        def _decorator(operation):
            def _operation(dia_op, offsets=None):
                if offsets is None:
                    offsets = dia_op.offsets
                package = dia_op._data.module
                with device_ctx_from_array(dia_op.data):
                    _data = operation(dia_op.data, offsets)
                    if dia_op.callback is None:
                        data = _data
                        callback = None
                    else:
                        if dia_op.callback.is_inplace:
                            raise ValueError(
                                "Unary operations are not supported for ElementaryOperators with inplace callbacks."
                            )
                        if return_type == "multidiagonal":
                            data = package.empty_like(_data, dtype=dia_op.dtype)
                        else:
                            data = package.empty((*dia_op.shape, _data.shape[-1]), dtype=dia_op.dtype)

                        callback = type(dia_op.callback)(
                            lambda t, args: operation(dia_op.callback(t, args).reshape(dia_op.data.shape), offsets)
                        )

                    if return_type == "multidiagonal":
                        return MultidiagonalOperator(data, offsets, callback)
                    else:
                        return DenseOperator(data, callback)

            return _operation

        return _decorator

    @staticmethod
    def _multidiagonal_to_dense(sparse_data, offsets):
        _sparse_data = wrap_operand(sparse_data)
        package = _sparse_data.module
        assert len(sparse_data.shape) == 3
        shape = (sparse_data.shape[0], sparse_data.shape[0], sparse_data.shape[2])
        row, col = package.indices(shape[:-1])
        dense_matrix_data = package.zeros(shape, sparse_data.dtype, order="F")
        for i, offset in enumerate(offsets):
            dense_matrix_data[row == col - offset] = sparse_data[: -abs(offset), i] if offset != 0 else sparse_data[:, i]
        return dense_matrix_data

    @_unary_operation("dense")
    def _to_dense(sparse_data, offsets):
        return MultidiagonalOperator._multidiagonal_to_dense(sparse_data, offsets)

    def to_dense(self) -> DenseOperator:
        """
        Return the `DenseOperator` form of the multidiagonal elementary operator.
        """
        return self._to_dense()

    def to_array(
        self,
        t: float | None = None,
        args: NDArrayType | None = None,
        device: str | int | None = None,
    ) -> NDArrayType:
        """
        Return the array form of the multidiagonal elementary operator on the specified device.

        If the device is not specified, an `ElementaryOperator` without callback will return a reference to its current underlying data,
        else the return location will be the return location of the `Callback` instance passed.
        This call is blocking if it involves device-to-host or device-to-device
        transfer, otherwise it is stream-ordered on the current stream.

        .. note::
            This function returns the dense array form of the multidiagonal elementary operator.
            If the original data buffer containing the diagonal elements is needed, use the :attr:`data` attribute if no callback was passed or invoke :attr:`callback` with arguments `t` and `args`.
        """
        return self.to_dense().to_array(t, args, device)

    @precondition(ElementaryOperator._check_scalar_operation_compability)
    def __mul__(self, scalar: Union[Number, NDArrayType]) -> "MultidiagonalOperator":
        """
        Multiply this instance with a scalar on the left.
        """
        scalar = self._prepare_scalar_for_unary_op(scalar)
        if self.device_id is None:
            return MultidiagonalOperator._unary_operation()(lambda x, _: x * scalar)(self)
        else:
            with cp.cuda.Device(self.device_id):
                return MultidiagonalOperator._unary_operation()(lambda x, _: x * scalar)(self)

    @precondition(ElementaryOperator._check_scalar_operation_compability)
    def __rmul__(self, scalar: Number) -> "MultidiagonalOperator":
        """
        Multiply this instance with a scalar on the right.
        """
        return self.__mul__(scalar)

    @_unary_operation()
    def _dag(data, offsets):
        return data.conj()

    def dag(self) -> "MultidiagonalOperator":
        """
        Return the conjugate complex transpose of this instance.
        """
        offsets = [-offset for offset in self.offsets]
        return self._dag(offsets)

    @staticmethod
    def _binary_operation(operation):
        def _operation(
            dia_op1: "MultidiagonalOperator",
            dia_op2: "MultidiagonalOperator",
            offsets: Sequence[int],
        ) -> "MultidiagonalOperator":
            move_op1 = dia_op1.device_id is None and dia_op2.device_id is not None
            move_op2 = dia_op2.device_id is None and dia_op1.device_id is not None
            template_op = dia_op1
            if move_op1:
                template_op = dia_op2
            package = template_op._data.module
            with device_ctx_from_array(template_op.data):

                _data = operation(
                    maybe_move_array(dia_op1.data, template_op.data)[0],
                    dia_op1.offsets,
                    maybe_move_array(dia_op2.data, template_op.data)[0],
                    dia_op2.offsets,
                    offsets,
                )
                if dia_op1.callback is None and dia_op2.callback is None:
                    data = _data
                    callback = None
                else:  # result is dynamic MultidiagonalOperator
                    data = package.empty(
                        _data.shape,
                        dtype=np.promote_types(dia_op1.dtype, dia_op2.dtype),
                        order="F",
                    )
                    callback_type = type(dia_op1.callback) if dia_op1.callback else type(dia_op2.callback)
                    if dia_op1.callback is None and dia_op2.callback is not None:
                        data_consumable_by_callback = dia_op1._data.to(device="cpu")

                        callback = lambda t, args: operation(
                            data_consumable_by_callback,
                            dia_op1.offsets,
                            dia_op2.callback(t, args).reshape(dia_op2.data.shape),
                            dia_op2.offsets,
                            offsets,
                        )
                    elif dia_op1.callback is not None and dia_op2.callback is None:
                        data_consumable_by_callback = dia_op2._data.to(device="cpu")
                        callback = lambda t, args: operation(
                            dia_op1.callback(t, args).reshape(dia_op1.data.shape),
                            dia_op1.offsets,
                            data_consumable_by_callback,
                            dia_op2.offsets,
                            offsets,
                        )
                    else:  # both inputs are dynamic
                        callback = lambda t, args: operation(
                            dia_op1.callback(t, args).reshape(dia_op1.data.shape),
                            dia_op1.offsets,
                            dia_op2.callback(t, args).reshape(dia_op2.data.shape),
                            dia_op2.offsets,
                            offsets,
                        )
                    callback = callback_type(callback, is_inplace=False)
            return MultidiagonalOperator(data, offsets, callback)

        return _operation

    @_binary_operation
    def _add(a, offsets_a, b, offsets_b, offsets):
        batchsize = max(a.shape[-1], b.shape[-1])
        package = wrap_operand(a).module
        data = package.zeros(
            (a.shape[0], len(offsets), batchsize),
            dtype=package.promote_types(a.dtype, b.dtype),
            order="F",
        )
        for i, offset in enumerate(offsets):
            if offset in offsets_a:
                index = offsets_a.index(offset)
                data[:, i] += a[:, index]
            if offset in offsets_b:
                index = offsets_b.index(offset)
                data[:, i] += b[:, index]
        return data

    @precondition(ElementaryOperator._check_binary_operation_compability, what="addition")
    def __add__(self, other: ElementaryOperatorType) -> ElementaryOperatorType:
        """
        Add an elementary operator to this instance and return a new elementary operator of the same type as ``other``.
        """
        if isinstance(other, MultidiagonalOperator):
            offsets = sorted(list(set(self.offsets) | set(other.offsets)))
            if self.device_id is None:
                return self._add(other, offsets)
            else:
                with cp.cuda.Device(self.device_id):
                    return self._add(other, offsets)
        else:
            return self.to_dense() + other

    @_binary_operation
    def _sub(a, offsets_a, b, offsets_b, offsets):
        batchsize = max(a.shape[-1], b.shape[-1])
        package = wrap_operand(a).module
        data = package.zeros(
            (a.shape[0], len(offsets), batchsize),
            dtype=package.promote_types(a.dtype, b.dtype),
            order="F",
        )
        for i, offset in enumerate(offsets):
            if offset in offsets_a:
                index = offsets_a.index(offset)
                data[:, i] += a[:, index]
            if offset in offsets_b:
                index = offsets_b.index(offset)
                data[:, i] -= b[:, index]
        return data

    @precondition(ElementaryOperator._check_binary_operation_compability, what="subtraction")
    def __sub__(self, other: ElementaryOperatorType) -> ElementaryOperatorType:
        """
        Subtract an elementary operator from this instance and return a new elementary operator of the same type as ``other``.
        """
        if isinstance(other, MultidiagonalOperator):
            offsets = sorted(list(set(self.offsets) | set(other.offsets)))
            if self.device_id is None:
                return self._sub(other, offsets)
            else:
                with cp.cuda.Device(self.device_id):
                    return self._sub(other, offsets)
        else:
            return self.to_dense() - other

    @_binary_operation
    def _matmul(a, offsets_a, b, offsets_b, offsets):
        matrix_dim = a.shape[0]
        batchsize = max(a.shape[-1], b.shape[-1])
        a_dense = MultidiagonalOperator._multidiagonal_to_dense(a, offsets_a)
        b_dense = MultidiagonalOperator._multidiagonal_to_dense(b, offsets_b)
        package = wrap_operand(a_dense).module
        result_array = dense_batched_matmul(a_dense, b_dense)
        a_dense, b_dense = None, None
        data = package.zeros(
            (matrix_dim, len(offsets), batchsize),
            dtype=package.promote_types(a.dtype, b.dtype),
            order="F",
        )
        for i, offset in enumerate(offsets):
            end = matrix_dim - abs(offset)
            # TODO: optimization would be nice here to avoid nested loop
            for k in range(batchsize):
                data[:end, i, k] = package.diag(result_array[..., k], offset)
        return data

    @precondition(ElementaryOperator._check_binary_operation_compability, what="matrix multiplication")
    def __matmul__(self, other: ElementaryOperatorType) -> ElementaryOperatorType:
        """
        Perform matrix multiplication between this instance and another elementary operator and
        return a new elementary operator of the same type as ``other``.
        """
        if isinstance(other, MultidiagonalOperator):
            offsets = np.kron(self.offsets, np.ones(len(other.offsets), dtype=int)) + np.kron(
                np.ones(len(self.offsets), dtype=int), other.offsets
            )
            offsets = np.unique(offsets)
            matrix_dim = self.shape[0]
            offsets = list(offsets[np.logical_and(-matrix_dim < offsets, offsets < matrix_dim)])

            if self.device_id is None:
                return self._matmul(other, offsets)
            else:
                with cp.cuda.Device(self.device_id):
                    return self._matmul(other, offsets)
        else:
            return self.to_dense() @ other
