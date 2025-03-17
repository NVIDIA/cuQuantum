# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Callback wrappers."""

from typing import Callable, Tuple, Union
from numbers import Number
from abc import ABC, abstractmethod

import numpy as np
import cupy as cp

from cuquantum.bindings.cudensitymat import WrappedScalarCallback, WrappedTensorCallback
from .._internal.tensor_wrapper import wrap_operand
from ._internal.utils import device_ctx_from_array, NDArrayType

CallbackType = Union[
    Callable[[float, NDArrayType], NDArrayType],
    Callable[[float, NDArrayType, NDArrayType], None],
]

__all__ = ["GPUCallback", "CPUCallback"]


class Callback(ABC):
    """
    Callback abstract base class.
    """

    def __init__(
        self,
        callback: CallbackType,
        is_inplace: bool,
    ) -> None:
        self._callback: Callable = callback
        self._is_inplace: bool = is_inplace
        if self._is_inplace:
            self._inplace_callback = self._callback
        else:

            def inplace_func(t, args, arr: NDArrayType):

                _arr = self.callback(t, args)
                if isinstance(_arr, Number):
                    returned_shape = ()
                else:
                    returned_shape = _arr.shape
                if not _is_shape_compatible(returned_shape, arr.shape):
                    raise ValueError(
                        f"The callback function returns an array of shape {_arr.shape} while an array of shape {arr.shape} is expected."
                    )
                if isinstance(_arr, Number):
                    arr[:] = _arr
                else:
                    arr[:] = _arr.reshape(arr.shape)

            self._inplace_callback = inplace_func
        self.batchsize: int | None = None
        self._wrapper: None | WrappedScalarCallback | WrappedTensorCallback = None

    @property
    def callback(self) -> CallbackType:
        """
        Return the callback function wrapped by this :class:`Callback`.
        """
        return self._callback

    @property
    def inplace_callback(self) -> Callable[[float, NDArrayType, NDArrayType], None]:
        """
        Return the inplace version of the callback function wrapped by this :class:`Callback`.
        """
        return self._inplace_callback

    @property
    @abstractmethod
    def is_gpu_callback(self) -> bool:
        """
        Return whether the callback function operates on GPU arrays.
        """
        pass

    @property
    def is_inplace(self):
        """
        Returns `True` if the callback function wrapped by this :class:`Callback` is acting in-place instead of returning its result.
        """
        return self._is_inplace

    def _get_internal_wrapper(self, which: str):
        """ """
        if which == "scalar":
            if self._wrapper is None:
                self._wrapper = WrappedScalarCallback(self._inplace_callback, self.is_gpu_callback)
            else:
                assert isinstance(self._wrapper, WrappedScalarCallback)
        elif which == "tensor":
            if self._wrapper is None:
                self._wrapper = WrappedTensorCallback(self._inplace_callback, self.is_gpu_callback)
            else:
                assert isinstance(self._wrapper, WrappedTensorCallback)
        else:
            raise ValueError(
                'Only tensor callbacks, `which="scalar"` and tensor callbacks, `which="scalar"` are accepted arguments.'
            )
        # keeps a wrapper reference to be independent of cython binding implementation's own reference counting
        return self._wrapper

    def __call__(self, *args):
        return self.callback(*args)


class GPUCallback(Callback):
    """
    GPUCallback(callback, is_inplace=False)

    Wrapper for GPU callback functions.
    
    .. note::
        - Out-of-place callback functions have a signature of `(t: float, args: cp.ndarray) -> cp.ndarray`.
        - Inplace callback functions have a signature of `(t: float, args: cp.ndarray, arr: cp.ndarray) -> None`.
        - Callback functions are invoked in a stream and device context and are assumed to return (for out-of-place callbacks) a ``cp.ndarray`` on the current device.
        - Callback functions receive `args` argument as cp.ndarray of shape (num_params, batch_size).

    Args:
        callback: The callback function to be wrapped, which can be either an inplace callback modifying 
            its third argument, or an out-of-place callback returning a ``cp.ndarray``.
        is_inplace: Specifies whether the callback is inplace.
    """

    def __init__(
        self,
        callback: Callable[[float, cp.ndarray], np.ndarray] | Callable[[float, cp.ndarray, cp.ndarray], None],
        is_inplace: bool = False,
    ):
        super().__init__(callback, is_inplace)

    @property
    def is_gpu_callback(self) -> bool:
        """
        Return whether the callback function operates on GPU arrays.
        """
        return True


class CPUCallback(Callback):
    """
    CPUCallback(callback, is_inplace=False)

    Wrapper for CPU callback functions.

    .. note::
        - Out-of-place callback functions have a signature of `(t: float, args: np.ndarray) -> np.ndarray`.
        - Inplace callback functions have a signature of `(t: float, args: np.ndarray, arr: np.ndarray) -> None`.
        - Callback functions are invoked in a stream and device context and are assumed to return (for out-of-place callbacks) an ``np.ndarray`` on the current device.
        - Callback functions receive `args` argument as np.ndarray of shape (num_params, batch_size).

    Args:
        callback: The callback function to be wrapped, which can be either an inplace callback modifying 
            its third argument, or an out-of-place callback returning a ``np.ndarray``.
        is_inplace: Specifies whether the callback is inplace.
    """

    def __init__(
        self,
        callback: Callable[[float, np.ndarray], np.ndarray] | Callable[[float, np.ndarray, np.ndarray], None],
        is_inplace: bool = False,
    ):
        super().__init__(callback, is_inplace)

    @property
    def is_gpu_callback(self) -> bool:
        """
        Return whether the callback function operates on GPU arrays.
        """
        return False


def _is_shape_compatible(dest_shape: Tuple[int], src_shape: Tuple[int]) -> bool:
    is_compatible = False
    if np.prod(dest_shape) == np.prod(src_shape):
        if dest_shape == src_shape:
            is_compatible = True
        elif len(dest_shape) - len(src_shape) == 1 and dest_shape[-1] == 1:
            is_compatible = True
        elif len(src_shape) - len(dest_shape) == 1 and src_shape[-1] == 1:
            is_compatible = True
    return is_compatible
