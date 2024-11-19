# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Elementary operator."""

from typing import Callable, Sequence, Union
from abc import ABC, abstractmethod
import weakref
import collections
from numbers import Number
import importlib
from operator import add, sub

import numpy as np
import cupy as cp
import cupyx

try:
    import scipy as sp
except ImportError:
    sp = None

from cuquantum.bindings import cudensitymat as cudm
from ._internal.utils import (
    generic_finalizer,
    register_with,
    wrap_callback,
    single_tensor_to,
    single_tensor_copy,
    device_ctx,
    transpose_bipartite_tensor,
    matricize_bipartite_tensor,
    NDArrayType,
    InvalidObjectState,
)
from .work_stream import WorkStream
from cuquantum.cutensornet._internal import tensor_wrapper, typemaps
from cuquantum.cutensornet._internal.utils import precondition, StreamHolder, cuda_call_ctx


__all__ = ["DenseOperator", "MultidiagonalOperator"]


DiaMatrixType = Union["sp.sparse.dia_matrix", "cupyx.scipy.sparse.dia_matrix"]
CallbackType = Callable[[float, Sequence], np.ndarray]
ElementaryOperatorType = Union["DenseOperator", "MultidiagonalOperator"]


class ElementaryOperator(ABC):
    """
    Elementary operator abstract base class.
    """

    def __init__(self, data: NDArrayType, callback: CallbackType | None = None) -> None:
        """
        Initialize an elementary operator from data buffer and callback.
        """
        # Input attributes
        self.callback = callback
        self.dtype: str = data.dtype.name

        # Internal attributes
        self._data = tensor_wrapper.wrap_operand(data)
        if self._data.device_id is None:
            self._data.tensor = self._data.tensor.copy(order="F")
        else:
            with device_ctx(self._data.device_id):
                self._data.tensor = self._data.tensor.copy(order="F")

        self._callback = wrap_callback(callback)
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

    @abstractmethod
    def to_array(self, t: float | None, args: Sequence | None):
        pass

    @abstractmethod
    def copy(self):
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
        if not isinstance(scalar, Number):
            raise TypeError(f"Cannot multiply {type(self).__name__} with {type(scalar).__name__}.")

    def _check_binary_operation_compability(self, other: "ElementaryOperator", what: str = ""):
        if not isinstance(other, ElementaryOperator):
            raise TypeError(
                f"Cannot perform {what} between {type(self).__name__} and {type(other).__name__}."
            )

        if self.shape != other.shape:
            raise ValueError(
                f"Cannot perform {what} between {type(self).__name__}s with mismatching shapes: {self.shape} and {other.shape}."
            )

        if self._data.module != other._data.module:
            raise ValueError(
                f"Cannot perform {what} between {type(self).__name__}s based on arrays from "
                f"different packages: {self._data.module} and {other._data.module}."
            )

        if self._data.device_id != other._data.device_id:
            raise ValueError(
                f"Cannot perform {what} between {type(self).__name__}s based on arrays from "
                f"different devices: device {self._data.device_id} and device {other._data.device_id}"
            )

    def _check_valid_state(self, *args, **kwargs):
        if not self._valid_state:
            raise InvalidObjectState("The tensor operator cannot be used after resources are freed")

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
                # NOTE: Check if this preserve stridedness
                self._data = single_tensor_to(
                    self._data, self._ctx.device_id, self._ctx._stream_holder
                )
            else:
                try:
                    assert self._data.device_id == self._ctx.device_id
                except AssertionError as e:
                    raise RuntimeError(
                        "Device id of input array does not match device id of library context."
                    ) from e
            self._instantiate()
            register_with(self, self._ctx, self._ctx.logger)

    def _instantiate(self) -> None:
        """
        Instantiate an ElementaryOperator.
        """
        sparsity, num_diagonals, diagonal_offsets = self._sparsity_and_diagonal_offsets
        self._ptr = cudm.create_elementary_operator(
            self._ctx._handle._validated_ptr,
            self.num_modes,
            self.mode_dims,
            sparsity,
            num_diagonals,
            diagonal_offsets,
            self._dtype,
            self._data.data_ptr,
            self._callback,
        )

        self._finalizer = weakref.finalize(
            self,
            generic_finalizer,
            self._ctx.logger,
            self._upstream_finalizers,
            (cudm.destroy_elementary_operator, self._ptr),
            msg=f"Destroying ElementaryOperator instance {self}, ptr: {self._ptr}.",
        )


class DenseOperator(ElementaryOperator):
    """
    DenseOperator(data, callback=None)

    Dense elementary operator from data buffer and optional callback.

    Args:
        data: Data buffer for operator elements.
        callback: A CPU callback function with signature ``(t, args) -> np.ndarray``.

    .. note::
        - A copy will be created on the data buffer and can be accessed through the :attr:`data` attribute.
        - The returned array needs to be consistent with the provided data buffer in terms of shape and data type. 
          The data buffer will be updated when this instance is involved in a ``compute`` method of an :class:`Operator` or :class:`OperatorAction`.
    
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

    def __init__(self, data: NDArrayType, callback: CallbackType | None = None) -> None:
        """
        Initialize a dense elementary operator from data buffer and optional callback.
        """
        super().__init__(data, callback)
        self.shape = data.shape

        self.num_modes = len(self.shape) // 2
        self.mode_dims = self.data.shape[: self.num_modes]

    @property
    def _sparsity_and_diagonal_offsets(self):
        return cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_NONE, 0, 0

    def to_array(
        self, t: float | None = None, args: Sequence | None = None, device: str = "cpu"
    ) -> NDArrayType:
        r"""
        Return the array form of the dense elementary operator.

        Args:
            t: Time variable in callback, only required if callback is not ``None``.
            args: Additional arguments in callback, only required if callback is not ``None``.
            device: Device on which to return the array. Defaults to ``"cpu"``.

        Returns:
            Array form of the dense elementary operator on the specified device.
        """
        if self.callback is None:
            return self._data.to(device, StreamHolder(obj=cp.cuda.Stream()))
        else:
            if t is None or args is None:
                raise ValueError(
                    "For a DenseOperator with callback, callback arguments must be passed in "
                    "when converted to an array."
                )
            return self.callback(t, args)

    def copy(self) -> "DenseOperator":
        """
        Return a copy of the dense elementary operator.
        """
        return DenseOperator(single_tensor_copy(self._data, self._ctx), self.callback)

    @staticmethod
    def _unary_operation(operation):
        def _operation(dense_op):
            if dense_op.callback is None:
                data = operation(dense_op.data)
                callback = None
            else:
                data = dense_op._data.module.empty_like(dense_op.data, dtype=dense_op.dtype)
                callback = lambda t, args: operation(dense_op.callback(t, args))
            return DenseOperator(data, callback)

        return _operation

    @precondition(ElementaryOperator._check_scalar_operation_compability)
    def __mul__(self, scalar: Number) -> "DenseOperator":
        """
        Multiply this instance with a scalar on the left.
        """
        if self._data.device_id is None:
            return DenseOperator._unary_operation(lambda x: x * scalar)(self)
        else:
            with cp.cuda.Device(self._data.device_id):
                return DenseOperator._unary_operation(lambda x: x * scalar)(self)

    @precondition(ElementaryOperator._check_scalar_operation_compability)
    def __rmul__(self, scalar: Number) -> "DenseOperator":
        """
        Multiply this instance with a scalar on the right.
        """
        return self * scalar

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
            if dense_op1.callback is None and dense_op2.callback is None:
                data = operation(dense_op1.data, dense_op2.data)
                callback = None
            else:  # result is dynamic DenseOperator
                data = dense_op1._data.module.empty_like(
                    dense_op1.data,
                    dtype=dense_op1._data.module.promote_types(dense_op1.dtype, dense_op2.dtype),
                )
                if dense_op1.callback is None and dense_op2.callback is not None:
                    stream_holder = StreamHolder(obj=cp.cuda.Stream())
                    with cuda_call_ctx(stream_holder, timing=False):
                        data1 = dense_op1._data.to(stream_holder=stream_holder)
                    callback = lambda t, args: operation(data1, dense_op2.callback(t, args))
                elif dense_op1.callback is not None and dense_op2.callback is None:
                    stream_holder = StreamHolder(obj=cp.cuda.Stream())
                    with cuda_call_ctx(stream_holder, timing=False):
                        data2 = dense_op2._data.to(stream_holder=stream_holder)
                    callback = lambda t, args: operation(dense_op1.callback(t, args), data2)
                else:  # both inputs are dynamic
                    callback = lambda t, args: operation(
                        dense_op1.callback(t, args), dense_op2.callback(t, args)
                    )
            return DenseOperator(data, callback)

        return _operation

    @precondition(ElementaryOperator._check_binary_operation_compability, what="addition")
    def __add__(self, other: ElementaryOperatorType) -> "DenseOperator":
        """
        Add an elementary operator to this instance and return a new :class:`DenseOperator`.
        """
        if isinstance(other, DenseOperator):
            if self._data.device_id is None:
                return DenseOperator._binary_operation(add)(self, other)
            else:
                with cp.cuda.Device(self._data.device_id):
                    return DenseOperator._binary_operation(add)(self, other)
        else:
            package = self._data.module if self.callback is None and other.callback is None else np
            return self + other._to_dense(package=package)

    @precondition(ElementaryOperator._check_binary_operation_compability, what="subtraction")
    def __sub__(self, other: ElementaryOperatorType) -> "DenseOperator":
        """
        Subtract an elementary operator from this instance and return a new :class:`DenseOperator`.
        """
        if isinstance(other, DenseOperator):
            if self._data.device_id is None:
                return DenseOperator._binary_operation(sub)(self, other)
            else:
                with cp.cuda.Device(self._data.device_id):
                    return DenseOperator._binary_operation(sub)(self, other)
        else:
            package = self._data.module if self.callback is None and other.callback is None else np
            return self - other._to_dense(package=package)

    @_binary_operation
    def _matmul(a, b):
        return (
            (matricize_bipartite_tensor(a) @ matricize_bipartite_tensor(b))
            .reshape(a.shape)
            .copy(order="F")
        )

    @precondition(
        ElementaryOperator._check_binary_operation_compability, what="matrix multiplication"
    )
    def __matmul__(self, other: ElementaryOperatorType) -> "DenseOperator":
        """
        Perform matrix multiplication between this instance and an elementary operator and return a new :class:`DenseOperator`.
        """
        if isinstance(other, DenseOperator):
            if self._data.device_id is None:
                return self._matmul(other)
            else:
                with cp.cuda.Device(self._data.device_id):
                    return self._matmul(other)
        else:
            package = self._data.module if self.callback is None and other.callback is None else np
            return self @ other._to_dense(package=package)


class MultidiagonalOperator(ElementaryOperator):
    """
    MultidiagonalOperator(data, offsets, callback=None)

    Multidiagonal single-mode operator from data buffer, offsets and optional callback.

    Args:
        data: Data buffer for diagonal elements, of shape ``(mode_dimension, num_diagonals)``.
        offsets: The diagonal offsets of length ``num_diagonals``.
        callback: A CPU callback function with signature ``(t, args) -> np.ndarray``.

    .. note::
        - The data layout is different from :class:`scipy.sparse.dia_matrix` and :class:`cupyx.scipy.sparse.dia_matrix`. 
          In this class, the elements of the ``offsets[i]``-th diagonal corresponds to the ``i``-th column of the input data buffer read from the top of the column.
        - A copy will be created on the data buffer and can be accessed through the :attr:`data` attribute.
        - The returned array needs to be consistent with the provided data buffer in terms of shape and data type. 
          The data buffer will be updated when this instance is involved in a ``compute`` method of an :class:`Operator` or :class:`OperatorAction`.

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

    def __init__(
        self, data: NDArrayType, offsets: Sequence[int], callback: CallbackType | None = None
    ) -> None:
        """
        Initialize a multidiagonal single-mode operator from data buffer, offsets and optional callback.
        """
        if len(offsets) != len(set(offsets)):
            raise ValueError("Offsets cannot contain duplicate elements.")
        if data.shape[1] != len(offsets):
            raise ValueError("Number of columns in data does not match length of offsets.")

        super().__init__(data, callback)
        self.offsets = list(offsets)

        mode_dim, self.num_diagonals = data.shape
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
            def _operation(dia_op, offsets=None, package=None):
                if offsets is None:
                    offsets = dia_op.offsets
                if package is None:
                    package = dia_op._data.module

                if dia_op.callback is None:
                    data = operation(dia_op.data, offsets, dia_op._data.module)
                    callback = None
                else:
                    if return_type == "multidiagonal":
                        data = dia_op._data.module.empty_like(dia_op.data, dtype=dia_op.dtype)
                    else:
                        data = dia_op._data.module.empty(dia_op.shape, dtype=dia_op.dtype)

                    callback = lambda t, args: operation(dia_op.callback(t, args), offsets, package)

                if return_type == "multidiagonal":
                    return MultidiagonalOperator(data, offsets, callback)
                else:
                    return DenseOperator(data, callback)

            return _operation

        return _decorator

    @staticmethod
    def _multidiagonal_to_dense(sparse_data, offsets, package):
        shape = (sparse_data.shape[0], sparse_data.shape[0])
        dense_matrix = package.zeros(shape, sparse_data.dtype, order="F")
        row, col = package.indices(shape)
        for i, offset in enumerate(offsets):
            dense_matrix[row == col - offset] = (
                sparse_data[: -abs(offset), i] if offset != 0 else sparse_data[:, i]
            )
        return dense_matrix

    @_unary_operation("dense")
    def _to_dense(sparse_data, offsets, package):
        return MultidiagonalOperator._multidiagonal_to_dense(sparse_data, offsets, package)

    def to_dense(self) -> DenseOperator:
        """
        Return the `DenseOperator` form of the multidiagonal elementary operator.
        """
        return self._to_dense()

    def to_array(
        self, t: float | None = None, args: Sequence | None = None, device: str = "cpu"
    ) -> NDArrayType:
        """
        Return the array form of the multidiagonal elementary operator.

        Args:
            t: Time variable in callback, only required is callback is not ``None``.
            args: Additional arguments in callback, only required if callback is not ``None``.
            device: Device on which to return the array. Defaults to ``"cpu"``.

        Returns:
            Array form of the multidiagonal elementary operator on the specified device.

        .. note::
            This function returns the dense array form of the multidiagonal elementary operator. If the original data buffer containing the diagonal elements is needed, use the :attr:`data` attribute.
        """
        package = np if device == "cpu" else cp
        return self._to_dense(package=package).to_array(t, args, device)

    def copy(self) -> "MultidiagonalOperator":
        """
        Return a copy of the multidiagonal elementary operator.
        """
        return MultidiagonalOperator(
            single_tensor_copy(self._data, self._ctx), self.offsets, self.callback
        )

    @precondition(ElementaryOperator._check_scalar_operation_compability)
    def __mul__(self, scalar: Number) -> "MultidiagonalOperator":
        """
        Multiply this instance with a scalar on the left.
        """

        @MultidiagonalOperator._unary_operation()
        def _mul(data, offsets, package):
            return data * scalar

        if self._data.device_id is None:
            return _mul(self)
        else:
            with cp.cuda.Device(self._data.device_id):
                return _mul(self)

    @precondition(ElementaryOperator._check_scalar_operation_compability)
    def __rmul__(self, scalar: Number) -> "MultidiagonalOperator":
        """
        Multiply this instance with a scalar on the right.
        """
        return self * scalar

    @_unary_operation()
    def _dag(data, offsets, package):
        return data.conj()

    def dag(self) -> "MultidiagonalOperator":
        """
        Return the conjugate complex transpose of this instance.
        """
        offsets = [-offset for offset in self.offsets]
        return self._dag(offsets)

    @staticmethod
    def _binary_operation(operation):
        def _operation(dia_op1, dia_op2, offsets):
            if dia_op1.callback is None and dia_op2.callback is None:
                data = operation(
                    dia_op1.data,
                    dia_op1.offsets,
                    dia_op2.data,
                    dia_op2.offsets,
                    offsets,
                    package=dia_op1._data.module,
                )
                callback = None
            else:  # result is dynamic MultidiagonalOperator
                data = dia_op1._data.module.empty(
                    (dia_op1.shape[0], len(offsets)),
                    dtype=np.promote_types(dia_op1.dtype, dia_op2.dtype),
                )
                if dia_op1.callback is None and dia_op2.callback is not None:
                    stream_holder = StreamHolder(obj=cp.cuda.Stream())
                    with cuda_call_ctx(stream_holder, timing=False):
                        data1 = dia_op1._data.to(stream_holder=stream_holder)
                    callback = lambda t, args: operation(
                        data1,
                        dia_op1.offsets,
                        dia_op2.callback(t, args),
                        dia_op2.offsets,
                        offsets,
                        package=np,
                    )
                elif dia_op1.callback is not None and dia_op2.callback is None:
                    stream_holder = StreamHolder(obj=cp.cuda.Stream())
                    with cuda_call_ctx(stream_holder, timing=False):
                        data2 = dia_op2._data.to(stream_holder=stream_holder)
                    callback = lambda t, args: operation(
                        dia_op1.callback(t, args),
                        dia_op1.offsets,
                        data2,
                        dia_op2.offsets,
                        offsets,
                        package=np,
                    )
                else:  # both inputs are dynamic
                    callback = lambda t, args: operation(
                        dia_op1.callback(t, args),
                        dia_op1.offsets,
                        dia_op2.callback(t, args),
                        dia_op2.offsets,
                        offsets,
                        package=np,
                    )
            return MultidiagonalOperator(data, offsets, callback)

        return _operation

    @_binary_operation
    def _add(a, offsets_a, b, offsets_b, offsets, package):
        data = package.zeros(
            (a.shape[0], len(offsets)),
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
            if self._data.device_id is None:
                return self._add(other, offsets)
            else:
                with cp.cuda.Device(self._data.device_id):
                    return self._add(other, offsets)
        else:
            package = self._data.module if self.callback is None and other.callback is None else np
            return self._to_dense(package=package) + other

    @_binary_operation
    def _sub(a, offsets_a, b, offsets_b, offsets, package):
        data = package.zeros(
            (a.shape[0], len(offsets)),
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
            if self._data.device_id is None:
                return self._sub(other, offsets)
            else:
                with cp.cuda.Device(self._data.device_id):
                    return self._sub(other, offsets)
        else:
            package = self._data.module if self.callback is None and other.callback is None else np
            return self._to_dense(package=package) - other

    @_binary_operation
    def _matmul(a, offsets_a, b, offsets_b, offsets, package):
        matrix_dim = a.shape[0]
        result_array = MultidiagonalOperator._multidiagonal_to_dense(
            a, offsets_a, package
        ) @ MultidiagonalOperator._multidiagonal_to_dense(b, offsets_b, package)

        data = package.zeros(
            (matrix_dim, len(offsets)),
            dtype=package.promote_types(a.dtype, b.dtype),
            order="F",
        )
        for i, offset in enumerate(offsets):
            end = matrix_dim - abs(offset)
            data[:, i][:end] = package.diag(result_array, offset)
        return data

    @precondition(
        ElementaryOperator._check_binary_operation_compability, what="matrix multiplication"
    )
    def __matmul__(self, other: ElementaryOperatorType) -> ElementaryOperatorType:
        """
        Perform matrix multiplication between this instance and another elementary operator and return a new elementary operator of the same type as ``other``.
        """
        if isinstance(other, MultidiagonalOperator):
            offsets = np.kron(self.offsets, np.ones(len(other.offsets), dtype=int)) + np.kron(
                np.ones(len(self.offsets), dtype=int), other.offsets
            )
            offsets = np.unique(offsets)
            matrix_dim = self.shape[0]
            offsets = list(offsets[np.logical_and(-matrix_dim < offsets, offsets < matrix_dim)])

            if self._data.device_id is None:
                return self._matmul(other, offsets)
            else:
                with cp.cuda.Device(self._data.device_id):
                    return self._matmul(other, offsets)
        else:
            package = self._data.module if self.callback is None and other.callback is None else np
            return self._to_dense(package=package) @ other
