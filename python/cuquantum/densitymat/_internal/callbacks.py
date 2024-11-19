# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# TODO[FUTURE]: Implement gradients for the callback.
from dataclasses import dataclass
from numbers import Number
from typing import Optional, Callable, Union
from cuquantum.cutensornet._internal.tensor_ifc import Tensor
import numpy as np
import cupy as cp


def _wrap_callback(func):
    """
    Returns callback that writes into scalar ndarray `storage` (t,args,storage) -> None, given `func` with signature (t,args)->Union[Number,ndarray].

    Parameters:
    -----------
    func: Callable
        Function with signature (t: float, args: Tuple[float]) returning a scalar.
    """

    def inplace_func(t: np.float64, args: tuple, buf: np.ndarray):
        buf[:] = func(t, args)

    return inplace_func


@dataclass
class CallbackCoefficient:
    """
    Wrapper class for treating static and dynamics coefficients on the same footing.

    Attributes
    callback: Optional[Callable]
        Callable with signature (t,args) -> Number, returning the dynamic coefficient.
    scalar: Number
        The static coefficient. If both callback and scalar are specified, the effective coefficient is the product of static and dynamic coefficient.
    """

    callback: Optional[Callable] = None
    scalar: Optional[Number] = 1.0 + 0.0j

    def __post_init__(self):
        if not isinstance(self.scalar, Number):
            raise TypeError(
                f"CallbackCoefficient received a scalar argument of type {type(self.scalar)}. CallbackCoefficient only accepts scalar arguments that are instances of Number."
            )
        self._wrapped_callback = 0
        if self.callback is not None:
            if not isinstance(self.callback, Callable):
                raise TypeError(
                    f"CallbackCoefficient received a callback argument of type {type(self.callback)}. CallbackCoefficient only accepts callback arguments that are instances of Callable."
                )
            self._wrapped_callback = _wrap_callback(self.callback)

    @property
    def is_callable(self) -> bool:
        return True if self.callback else False

    def __neg__(self) -> "CallbackCoefficient":
        return self.__mul__(self, -1)

    def __mul__(self, factor: Union[Number, Callable, "CallbackCoefficient"]):
        """
        Multiplication
        """
        # ToDo: Add test that cover all branches
        if isinstance(factor, CallbackCoefficient):
            if self.is_callable and factor.is_callable:
                callback = lambda t, args: self.callback(t, args) * factor.callback(t, args)
            elif self.is_callable:
                callback = self.callback
            elif factor.is_callable:
                callback = factor.callback
            else:
                callback = None
            return CallbackCoefficient(callback, self.scalar * factor.scalar)
        # Probably Not required unless this is exposed to user
        elif isinstance(factor, Number):
            return CallbackCoefficient(self.callback, self.scalar * factor)
        elif isinstance(factor, Callable):
            return CallbackCoefficient(
                lambda t, args: (self.callback(t, args) * factor(t, args)), self.scalar
            )

    def __rmul__(
        self, factor: Union[Number, Callable, "CallbackCoefficient"]
    ) -> "CallbackCoefficient":
        # right/left logic is handled in __mul__ and is symmetric
        return self.__mul__(factor)

    def __lmul__(
        self, factor: Union[Number, Callable, "CallbackCoefficient"]
    ) -> "CallbackCoefficient":
        # right/left logic is handled in __mul__ and is symmetric
        return self.__mul__(factor)

    def __add__(self, summand: Union["CallbackCoefficient", Number, Callable]):
        """
        Addition
        """
        if isinstance(summand, CallbackCoefficient):
            if self.callback == summand.callback:
                return CallbackCoefficient(self.callback, self.scalar + summand.scalar)
            elif np.isclose(self.scalar, summand.scalar):
                if self.is_callable and summand.is_callable:
                    callback = lambda t, args: self.callback(t, args) + summand.callback(t, args)
                elif self.is_callable:
                    callback = lambda t, args: self.callback(t, args) + 1
                elif summand.is_callable:
                    callback = lambda t, args: summand.callback(t, args) + 1
                return CallbackCoefficient(callback, self.scalar)
            else:
                if self.is_callable and summand.is_callable:
                    callback = (
                        lambda t, args: self.callback(t, args) * self.scalar
                        + summand.callback(t, args) * summand.scalar
                    )
                elif self.is_callable:
                    callback = lambda t, args: self.callback(t, args) * self.scalar + summand.scalar
                elif summand.is_callable:
                    callback = (
                        lambda t, args: self.scalar + summand.callback(t, args) * summand.scalar
                    )
                return CallbackCoefficient(callback)
        elif isinstance(summand, Number):
            return self + CallbackCoefficient(None, summand)
        elif isinstance(summand, Callable):
            return self + CallbackCoefficient(summand)
        else:
            raise TypeError(
                f"{type(summand)} cannot be added to CallbackCoefficient. CallbackCoefficient only supports addition of CallbackCoefficient, Number or Callable."
            )

    def __sub__(self, subtrahend: Union["CallbackCoefficient", Number, Callable]):
        """
        Substraction
        """
        return self + (-1) * subtrahend

    def conjugate(self):
        conj_callback = (
            (lambda t, args: self.callback(t, args).conjugate()) if self.is_callable else None
        )
        conj_scalar = self.scalar.conjugate()
        return CallbackCoefficient(conj_callback, conj_scalar)
