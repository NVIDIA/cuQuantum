# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Callback wrappers.
"""

from typing import Callable, Tuple
from numbers import Number
from abc import ABC, abstractmethod

import numpy as np
import cupy as cp

import cuquantum.bindings.cudensitymat as cudm
from ._internal.utils import NDArrayType

InplaceRegularCallbackType = Callable[[float, NDArrayType, NDArrayType], None]
OutofplaceRegularCallbackType = Callable[[float, NDArrayType], NDArrayType]
RegularCallbackType = InplaceRegularCallbackType | OutofplaceRegularCallbackType

CPUInplaceRegularCallbackType = Callable[[float, np.ndarray, np.ndarray], None]
CPUOutofplaceRegularCallbackType = Callable[[float, np.ndarray], np.ndarray]
CPURegularCallbackType = CPUInplaceRegularCallbackType | CPUOutofplaceRegularCallbackType
GPUInplaceRegularCallbackType = Callable[[float, cp.ndarray, cp.ndarray], None]
GPUOutofplaceRegularCallbackType = Callable[[float, cp.ndarray], cp.ndarray]
GPURegularCallbackType = GPUInplaceRegularCallbackType | GPUOutofplaceRegularCallbackType

InplaceGradientCallbackType = Callable[[float, NDArrayType, NDArrayType, NDArrayType], None]
OutofplaceGradientCallbackType = Callable[[float, NDArrayType, NDArrayType], NDArrayType]
GradientCallbackType = InplaceGradientCallbackType | OutofplaceGradientCallbackType

CPUInplaceGradientCallbackType = Callable[[float, np.ndarray, np.ndarray, np.ndarray], None]
CPUOutofplaceGradientCallbackType = Callable[[float, np.ndarray, np.ndarray], np.ndarray]
CPUGradientCallbackType = CPUInplaceGradientCallbackType | CPUOutofplaceGradientCallbackType
GPUInplaceGradientCallbackType = Callable[[float, cp.ndarray, cp.ndarray, cp.ndarray], None]
GPUOutofplaceGradientCallbackType = Callable[[float, cp.ndarray, cp.ndarray], cp.ndarray]
GPUGradientCallbackType = GPUInplaceGradientCallbackType | GPUOutofplaceGradientCallbackType

__all__ = ["GPUCallback", "CPUCallback"]


class Callback(ABC):
    """
    Callback abstract base class.
    """

    def __init__(
        self,
        callback: RegularCallbackType,
        is_inplace: bool,
        gradient_callback: GradientCallbackType | None = None,
        gradient_dir: str = "backward"
    ) -> None:
        """
        Initialize a Callback object.

        Args:
            callback: The callback function to be wrapped.
            is_inplace: Specifies whether the callback is inplace.
            gradient_callback: The gradient callback function to be wrapped.
            gradient_dir: The gradient direction.
        """
        self._callback: RegularCallbackType = callback
        self._is_inplace: bool = is_inplace
        self._gradient_callback: GradientCallbackType | None = gradient_callback
        self._gradient_dir_str: str = gradient_dir

        self._gradient_dir: cudm.DifferentiationDir | None = None
        self._inplace_callback: InplaceRegularCallbackType | None = None
        self._inplace_gradient_callback: InplaceGradientCallbackType | None = None

        if gradient_callback is not None:
            gradient_dir = "backward" if gradient_dir is None else gradient_dir
            if gradient_dir == "backward":
                self._gradient_dir = cudm.DifferentiationDir.BACKWARD
            elif gradient_dir == "forward":
                raise NotImplementedError(f"Gradient direction {gradient_dir} not currently supported.")
            else:
                raise ValueError(f"Invalid gradient direction {gradient_dir}.")

        if self._is_inplace:
            self._inplace_callback = self._callback
            self._inplace_gradient_callback = self._gradient_callback

        else:
            def inplace_func(t, args, arr):
                _arr = self.callback(t, args)

                # Check if shape of the array returned from user-defined out-of-place callback is compatible 
                # with the shape of the array in the implemented inplace callback.
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
            
            if self._gradient_callback is not None:
                def inplace_gradient_func_backward(t, args, adj, args_grad):
                    _args_grad = self.gradient_callback(t, args, adj)

                    # Check if shape of the array returned from user-defined out-of-place callback is compatible 
                    # with the shape of the array in the implemented inplace callback.
                    if isinstance(_args_grad, Number):
                        returned_shape = ()
                    else:
                        returned_shape = _args_grad.shape
                    if not _is_shape_compatible(returned_shape, args_grad.shape):
                        raise ValueError(
                            f"The callback function returns an array of shape {_args_grad.shape} while an array of shape {args_grad.shape} is expected."
                        )
                    
                    # NOTE: Gradient callbacks are accumulative.
                    if isinstance(_args_grad, Number):
                        args_grad[:] += _args_grad
                    else:
                        args_grad[:] += _args_grad.reshape(args_grad.shape)

                self._inplace_gradient_callback = inplace_gradient_func_backward
            
        self.batchsize: int | None = None
        self._wrapper: None | cudm.WrappedScalarCallback | cudm.WrappedTensorCallback = None
        self._gradient_wrapper: None | cudm.WrappedScalarGradientCallback | cudm.WrappedTensorGradientCallback = None


    @property
    def callback(self) -> RegularCallbackType:
        """
        Return the callback function wrapped by this :class:`Callback`.
        """
        return self._callback
    
    @property
    def gradient_callback(self) -> GradientCallbackType | None:
        """
        Return the gradient callback function wrapped by this :class:`Callback`.
        """
        return self._gradient_callback

    @property
    def gradient_dir(self) -> str | None:
        """
        Return the gradient direction of the gradient callback function wrapped by this :class:`Callback`.
        """
        return self._gradient_dir_str

    @property
    def inplace_callback(self) -> InplaceRegularCallbackType:
        """
        Return the inplace version of the callback function wrapped by this :class:`Callback`.
        """
        return self._inplace_callback
 
    @property
    def inplace_gradient_callback(self) -> InplaceGradientCallbackType | None:
        """
        Return the inplace version of the gradient callback function wrapped by this :class:`Callback`.
        """
        return self._inplace_gradient_callback

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
    
    @property
    def has_gradient(self) -> bool:
        """
        Returns whether a gradient callback function is available for this :class:`Callback` instance.
        """
        return self._gradient_callback is not None

    def _get_internal_wrapper(self, which: str) -> cudm.WrappedScalarCallback | cudm.WrappedTensorCallback:
        """
        Wrap the callback function to WrappedScalarCallback or WrappedTensorCallback.
        """
        if which == "scalar":
            if self._wrapper is None:
                self._wrapper = cudm.WrappedScalarCallback(self._inplace_callback, self.is_gpu_callback)
            else:
                assert isinstance(self._wrapper, cudm.WrappedScalarCallback)
        elif which == "tensor":
            if self._wrapper is None:
                self._wrapper = cudm.WrappedTensorCallback(self._inplace_callback, self.is_gpu_callback)
            else:
                assert isinstance(self._wrapper, cudm.WrappedTensorCallback)
        else:
            raise ValueError(
                'Only tensor callbacks, `which="scalar"` and tensor callbacks, `which="scalar"` are accepted arguments.'
            )
        # keeps a wrapper reference to be independent of cython binding implementation's own reference counting
        return self._wrapper
    
    def _get_internal_gradient_wrapper(self, which: str) -> cudm.WrappedScalarGradientCallback | cudm.WrappedTensorGradientCallback | None:
        """
        Wrap the gradient callback function to WrappedScalarGradientCallback or WrappedTensorGradientCallback.
        """
        if which == "scalar":
            if self._gradient_wrapper is None:
                if self.has_gradient:
                    self._gradient_wrapper = cudm.WrappedScalarGradientCallback(self._inplace_gradient_callback, self.is_gpu_callback, self._gradient_dir)
            else:
                assert isinstance(self._gradient_wrapper, cudm.WrappedScalarGradientCallback)
        elif which == "tensor":
            if self._gradient_wrapper is None:
                if self.has_gradient:
                    self._gradient_wrapper = cudm.WrappedTensorGradientCallback(self._inplace_gradient_callback, self.is_gpu_callback, self._gradient_dir)
            else:
                assert isinstance(self._gradient_wrapper, cudm.WrappedTensorGradientCallback)
        else:
            raise ValueError(
                'Only tensor callbacks, `which="scalar"` and tensor callbacks, `which="scalar"` are accepted arguments.'
            )
        # keeps a wrapper reference to be independent of cython binding implementation's own reference counting
        return self._gradient_wrapper
    
    def __call__(self, *args):
        return self.callback(*args)


class GPUCallback(Callback):
    """
    GPUCallback(callback, is_inplace=False, gradient_callback=None, gradient_dir=None)

    Wrapper for GPU callback functions.
    
    .. note::
        - Out-of-place callback functions have a signature of ``(t: float, args: cp.ndarray) -> cp.ndarray``.
        - Inplace callback functions have a signature of ``(t: float, args: cp.ndarray, arr: cp.ndarray) -> None``.
        - Callback functions are invoked in a stream and device context and are assumed to return (for out-of-place callbacks) a ``cp.ndarray`` on the current device.
        - Callback functions receive ``args`` argument as cp.ndarray of shape ``(num_params, batch_size)``.
        - In-place gradient callbacks are required to accumulate the gradient contributions of the callback instead of overwriting the output array. 
        - Backward gradient callback functions have a signature of ``(t: float, args: cp.ndarray, adj: cp.ndarray) -> cp.ndarray``.
          ``adj`` is the adjoint computed through backward pass and of the same shape as the output of the regular callback.
        - In-place gradient callbacks have a signature of ``(t: float, args: cp.ndarray, adj: cp.ndarray, args_grad: cp.ndarray) -> None``.
          ``args_grad`` is the array to accumulate the gradient contributions into, and of the same shape as ``args``.

    Args:
        callback: The callback function to be wrapped, which can be either an inplace callback modifying 
            its third argument, or an out-of-place callback returning a ``cp.ndarray``.
        is_inplace: Specifies whether the callback is inplace.
        gradient_callback: The gradient callback function to be wrapped. If not specified, the callback is not considered differentiable
            and the contribution of this callback to the operator action gradient is not taken into account.
        gradient_dir: Whether or not the gradient is evaluated in the forward or backward direction.
            All differentiable callbacks in a :class:`Operator` must have the same gradient direction.
    """

    def __init__(
        self,
        callback: GPURegularCallbackType,
        is_inplace: bool = False,
        gradient_callback: GPUGradientCallbackType | None = None,
        gradient_dir: str | None = None
    ):
        super().__init__(callback, is_inplace, gradient_callback, gradient_dir)

    @property
    def is_gpu_callback(self) -> bool:
        """
        Return whether the callback function operates on GPU arrays.
        """
        return True

class CPUCallback(Callback):
    """
    CPUCallback(callback, is_inplace=False, gradient_callback=None, gradient_dir=None)

    Wrapper for CPU callback functions.

    .. note::
        - Out-of-place callback functions have a signature of ``(t: float, args: np.ndarray) -> np.ndarray``.
        - Inplace callback functions have a signature of ``(t: float, args: np.ndarray, arr: np.ndarray) -> None``.
        - Callback functions are invoked in a stream and device context and are assumed to return (for out-of-place callbacks) an ``np.ndarray`` on the current device.
        - Callback functions receive ``args`` argument as ``np.ndarray`` of shape ``(num_params, batch_size)``.
        - In-place gradient callbacks are required to accumulate the gradient contributions of the callback instead of overwriting the output array. 
        - Gradient callback functions have a signature of ``(t: float, args: np.ndarray, adj: np.ndarray) -> np.ndarray``.
          ``adj`` is the adjoint computed through backward pass and of the same shape as the output of the regular callback.
        - In-place gradient callbacks have a signature of ``(t: float, args: np.ndarray, adj: np.ndarray, args_grad: np.ndarray) -> None``.
          ``args_grad`` is the array to accumulate the gradient contributions into, and of the same shape as ``args``.

    Args:
        callback: The callback function to be wrapped, which can be either an inplace callback modifying 
            its third argument, or an out-of-place callback returning a ``np.ndarray``.
        is_inplace: Specifies whether the callback is inplace.
        gradient_callback: The gradient callback function to be wrapped. If not specified, the callback is not considered differentiable
            and the contribution of this callback to the operator action gradient is not taken into account.
        gradient_dir: Whether or not the gradient is evaluated in the forward or backward direction.
            All differentiable callbacks in a :class:`Operator` must have the same gradient direction.
    """

    def __init__(
        self,
        callback: CPURegularCallbackType,
        is_inplace: bool = False,
        gradient_callback: CPUGradientCallbackType | None = None,
        gradient_dir: str | None = None
    ):
        super().__init__(callback, is_inplace, gradient_callback, gradient_dir)

    @property
    def is_gpu_callback(self) -> bool:
        """
        Return whether the callback function operates on GPU arrays.
        """
        return False


def _is_shape_compatible(dest_shape: Tuple[int], src_shape: Tuple[int]) -> bool:
    """
    Check if the array shapes are compatible up to batch size 1.
    """
    is_compatible = False
    if np.prod(dest_shape) == np.prod(src_shape):
        if dest_shape == src_shape:
            is_compatible = True
        elif len(dest_shape) - len(src_shape) == 1 and dest_shape[-1] == 1:
            is_compatible = True
        elif len(src_shape) - len(dest_shape) == 1 and src_shape[-1] == 1:
            is_compatible = True
    return is_compatible
