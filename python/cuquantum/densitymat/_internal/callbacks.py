# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from numbers import Number
from typing import Optional, Callable, Union, Tuple, Iterable
from operator import mul, add, sub
from abc import ABC, abstractmethod
from cuquantum._internal.tensor_ifc import Tensor
from cuquantum._internal.tensor_wrapper import wrap_operand
import numpy as np
import cupy as cp
from .utils import check_and_get_batchsize, maybe_move_arrays, NDArrayType, device_ctx_from_array
from ..callbacks import Callback, GPUCallback, CPUCallback
from cuquantum.bindings.cudensitymat import WrappedScalarCallback
from cuquantum._internal.utils import precondition

CoefficientType = Union[
    Callback,
    Union[Number, NDArrayType],
    Tuple[Number, Callback],
    Tuple[NDArrayType, Callback]
]


def _check_for_static_and_dynamic_type(coeff: Tuple):
    if not isinstance(coeff, tuple):
        return False
    if not len(coeff) == 2:
        return False
    if len(coeff) != 2:
        return False
    if isinstance(coeff[0], (Number, NDArrayType)) and isinstance(coeff[1], Callback):
        return True
    else:
        return False


class CallbackCoefficient(ABC):
    """
    Wrapper class for treating static and dynamics coefficients on the same footing.
    """

    def __init__(self, static_coeff, dynamic_coeff, batch_size):
        self._static_coeff = static_coeff
        self._callback = dynamic_coeff
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        """
        The batch size of this :class:`CallbackCoefficient`.
        """
        return self._batch_size
    
    @property
    def callback(self) -> Callback:
        """
        Callable wrapped as ::class`Callback` instance.
        """
        return self._callback

    @staticmethod
    def create(coeff: CoefficientType = complex(1), batch_size=1) -> "CallbackCoefficient":
        """
        Constructor of concrete subclasses of :class:`CallbackCoefficient`, will return either a
        :class:`ScalarCallbackCoefficient` or :class:`BatchedCallbackCoefficent` depending on the input.
        """
        if not isinstance(coeff, (Callback, Number, NDArrayType, Tuple)):
            raise TypeError(
                f"CallbackCoefficient constructor does not support type {type(coeff)} as `coeff` argument in its constructor."
            )
        if not isinstance(coeff, (Callback, Number, NDArrayType)):
            if not _check_for_static_and_dynamic_type(coeff):
                raise TypeError(
                    f"CallbackCoefficient constructor only supports the following types for its `coeff` argument:  `Callback`, `Number`, `NDArrayType` and tuple with two member, the first of which is an instance of an `NDArrayType` and the second an instance of `Callback`."
                )
        static_coeff, dynamic_coeff = None, None
        if isinstance(coeff, (Number, NDArrayType)):
            static_coeff = coeff
        elif isinstance(coeff, Callback) or _check_for_static_and_dynamic_type(coeff):
            dynamic_coeff = coeff if isinstance(coeff, Callback) else coeff[1]
            static_coeff = None if isinstance(coeff, Callback) else coeff[0]
        else:
            # TODO: fix error message
            raise ValueError("Incompatible input coefficient type?")
        static_batch_size = static_coeff.size if isinstance(static_coeff, NDArrayType) else 1
        batch_size = check_and_get_batchsize(batch_size, static_batch_size)
        if batch_size == 1:
            if static_coeff is None:
                static_coeff = 1.0 + 0.0j
            elif isinstance(static_coeff, Number):
                static_coeff = complex(static_coeff)
            elif isinstance(static_coeff, NDArrayType):
                static_coeff = complex(
                    static_coeff[()] if static_coeff.shape == () else static_coeff[0]
                )
            return ScalarCallbackCoefficient(static_coeff, dynamic_coeff)
        else:
            if static_coeff is None or isinstance(static_coeff, Number):
                static_coeff = np.ones(batch_size, dtype=np.complex128)
            elif static_batch_size == 1 and isinstance(static_coeff, NDArrayType):
                raise ValueError("Inconsistent batch size and coefficient array dimension.")
            return BatchedCallbackCoefficient(
                static_coeff.astype("complex128"), dynamic_coeff, batch_size
            )

    @property
    @abstractmethod
    def static_coeff(self) -> NDArrayType | Number:
        pass

    @property
    def is_callable(self) -> bool:
        return True if self.callback else False

    @property
    def dtype(self) -> bool:
        return "complex128"

    def _check_binary_operation_compatibility(self, other: "CallbackCoefficient", what: str = ""):
        if not isinstance(other, CallbackCoefficient):
            raise TypeError(
                f"Cannot perform {what} between {type(self).__name__} and {type(other).__name__}."
            )

        if self.batch_size != other.batch_size and not (
            self.batch_size == 1 or other.batch_size == 1
        ):
            raise ValueError(
                f"Cannot perform {what} between {type(self).__name__}s with mismatching batch dimensions: {self.batch_size} and {other.batch_size}."
            )

        # not checking device compat here, needs to be performed by binary operation
        
        # Allow only multiplication of out-of-place callbacks
        if self.callback or other.callback:
            is_out_of_place = (
                True if (self.callback is None or not (self.callback.is_inplace)) else False
            )
            is_out_of_place *= (
                True if (other.callback is None or not (other.callback.is_inplace)) else False
            )
            if not is_out_of_place:
                raise ValueError(
                    "Cannot perform {what} between CallbackCoefficients with inplace callbacks."
                )

        # Allow only multiplication of same kind of Callback
        if self.callback and other.callback:
            both_gpu = isinstance(self.callback, GPUCallback) and isinstance(
                other.callback, GPUCallback
            )
            both_cpu = isinstance(self.callback, CPUCallback) and isinstance(
                other.callback, CPUCallback
            )
            if not (both_gpu or both_cpu):
                raise ValueError(
                    f"Cannot perform {what} between CallbackCoefficients where one has a GPU callback and the other has a CPU callbacks."
                )

    @staticmethod
    def _binary_operation(operation):
        def _operation(coeff1: "CallbackCoefficient", coeff2: "CallbackCoefficient"):
            if isinstance(coeff1, BatchedCallbackCoefficient) and isinstance(
                coeff2, BatchedCallbackCoefficient
            ):
                # if one coefficient on CPU and one on GPU, move to GPU
                # if coefficients on separate GPUs an error will be raised
                move_coeff1 = coeff1.device_id is None and coeff2.device_id is not None
                move_coeff2 = coeff2.device_id is None and coeff1.device_id is not None
                template_coeff = coeff1
                if move_coeff1:
                    template_coeff = coeff2
                with device_ctx_from_array(template_coeff.static_coeff):
                    static_coeff = operation(
                        *maybe_move_arrays(coeff1.static_coeff, coeff2.static_coeff)
                    )
            else:
                # scalars always on CPU, scalar vector multiply works for both CPU and GPU vectors
                static_coeff = operation(coeff1.static_coeff, coeff2.static_coeff)
            if coeff1.is_callable and coeff2.is_callable:
                callback = type(coeff1.callback)(
                    lambda t, args: operation(coeff1.callback(t, args), coeff2.callback(t, args))
                )
            elif coeff1.is_callable:
                callback = coeff1.callback
            elif coeff2.is_callable:
                callback = coeff2.callback
            else:
                callback = None
            return CallbackCoefficient.create(
                static_coeff if callback is None else (static_coeff, callback)
            )

        return _operation

    @precondition(_check_binary_operation_compatibility, what="multiplication")
    def __mul__(self, factor: "CallbackCoefficient") -> "CallbackCoefficient":
        """
        Multiplication.
        """
        # ToDo: Add test that cover all branches?
        if not isinstance(factor, CallbackCoefficient):
            factor = CallbackCoefficient(factor)
        return CallbackCoefficient._binary_operation(mul)(self, factor)

    def __neg__(self) -> "CallbackCoefficient":
        return self.__mul__(CallbackCoefficient.create(-1))

    def __rmul__(self, factor: "CallbackCoefficient") -> "CallbackCoefficient":
        # right/left logic is handled in __mul__ and is symmetric
        return self.__mul__(factor)

    def __lmul__(self, factor: "CallbackCoefficient") -> "CallbackCoefficient":
        # right/left logic is handled in __mul__ and is symmetric
        return self.__mul__(factor)

    def conjugate(self) -> "CallbackCoefficient":
        conj_static_coeff = self.static_coeff.conjugate()
        if self.is_callable:
            if self.callback.is_inplace:
                raise RuntimeError(
                    "Inplace callbacks do not support composition via arithmetic operations or conjugation."
                )
            conj_callback = type(self.callback)(lambda t, args: self.callback(t, args).conjugate())
            return CallbackCoefficient.create((conj_static_coeff, conj_callback))
        else:
            return CallbackCoefficient.create(conj_static_coeff)

    def unpack(self):
        return (self.static_coeff, self.callback) if self.is_callable else self.static_coeff

    @property
    def wrapper(self) -> WrappedScalarCallback | None:
        if self.is_callable:
            return self.callback._get_internal_wrapper("scalar")
        else:
            return None


# TODO: maybe disable in-place callback for ScalarCallbackCoefficient
# TODO: add test for inplace callback for ScalarCallbackCoefficient, if not supported disable
class ScalarCallbackCoefficient(CallbackCoefficient):
    def __init__(self, static_coeff: Number, dynamic_coeff: Callback | None):
        super().__init__(static_coeff, dynamic_coeff, 1)
        
    @property
    def static_coeff(self) -> Number:
        """
        The static component of this :class:`ScalarCallbackCoefficient`.
        """
        return self._static_coeff

    def to_array(
        self, t: float = 0.0, args: NDArrayType | None = None, device: str | int | None = None
    ) -> Number:
        r"""
        Return the total (static * dynamic) coefficient.
        Returns:
            Scalar
        """
        callback_args = None if (t is None or args is None) else (t, args)
        if self.callback is None:
            if device == "cpu":
                return np.array(self.static_coeff)
            # FIXME inconsistent None behaviour in comparison with Batched
            elif isinstance(device, int) or None:
                with cp.cuda.Device(device):
                    return cp.array(self._static_coeff)
            else:
                raise ValueError(
                    f'Received invalid input for `device` argument of to_array method, {device}. Accepted inputs are `None`, integers larger equal zero and "cpu".'
                )
        else:
            # check input args types
            if callback_args == None:
                raise ValueError(
                    "For a CallbackCoefficient with callback, callback arguments must be passed in "
                    "when converted to an array."
                )
            if self.callback.is_inplace:
                if self.callback.is_gpu_callback:
                    params_arr = callback_args[1]
                    arr = cp.empty(self.static_coeff.shape, dtype=self.dtype, order="F")
                    self.callback(*callback_args, arr)
                else:
                    arr = np.empty(self.data.shape, dtype=self.dtype, order="F")
                    self.callback(*callback_args, arr)
            else:
                arr = self.callback(*callback_args)
            # multiply with static coeff
            arr = arr * self.static_coeff
            if isinstance(arr, NDArrayType):
                if arr.ndim > 0:
                    arr = arr[0]
                else:
                    arr = arr[()]
            return complex(arr)


class BatchedCallbackCoefficient(CallbackCoefficient):
    def __init__(self, static_coeff: NDArrayType, dynamic_coeff: Callback | None, batch_size):
        assert batch_size == static_coeff.size
        _static_coeff = wrap_operand(static_coeff)
        if _static_coeff.dtype != "complex128":
            _static_coeff = wrap_operand(_static_coeff.tensor * complex(1))
        super().__init__(_static_coeff, dynamic_coeff, batch_size)
        self._dynamic_coeff = None

    @property
    def static_coeff(self) -> NDArrayType:
        """
        The static component of this :class:`BatchedCallbackCoefficient`.
        """
        return self._static_coeff.tensor

    @property
    def dynamic_coeff_ptr(self) -> int:
        if self._dynamic_coeff is None and self.callback is not None:
            self._dynamic_coeff = wrap_operand(
                cp.zeros(shape=self.static_coeff.shape, dtype=self.static_coeff.dtype)
            )
        return self._dynamic_coeff.data_ptr if self._dynamic_coeff is not None else 0

    @property
    def static_coeff_ptr(self) -> int:
        if self._static_coeff.device_id is None:
            self._static_coeff = wrap_operand(self._static_coeff.to(cp.cuda.Device().id))
        assert self._static_coeff.dtype == "complex128"
        return self._static_coeff.data_ptr

    @property
    def device_id(self) -> int | None:
        return self._static_coeff.device_id

    def to_array(
        self,
        t: float | None = None,
        args: NDArrayType | None = None,
        device: str | int | None = None,
    ) -> NDArrayType:
        """
        Return the total (static * dynamic) coefficients.
        """
        callback_args = None if (t is None or args is None) else (t, args)
        if self.callback is None:
            if device is None:
                return self.static_coeff
            elif device == "cpu" or isinstance(device, int):
                return self._static_coeff.to(device)
        else:
            # check input args types
            if callback_args == None:
                raise ValueError(
                    "For a CallbackCoefficient with callback, callback arguments must be passed in "
                    "when converted to an array."
                )
            if self.callback.is_inplace:
                if self.callback.is_gpu_callback:
                    params_arr = callback_args[1]
                    arr = cp.empty(self.static_coeff.shape, dtype=self.dtype, order="F")
                    self.callback(*callback_args, arr)
                else:
                    arr = np.empty(self.data.shape, dtype=self.dtype, order="F")
                    self.callback(*callback_args, arr)
            else:
                arr = self.callback(*callback_args)
            # multiply with static coeff
            arr, static_coeff = maybe_move_arrays(arr, self.static_coeff)
            arr = arr * static_coeff

            # move to target device
            if device is not None:
                return wrap_operand(arr).to(device)
            else:
                return arr
