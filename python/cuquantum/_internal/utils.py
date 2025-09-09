# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
A collection of (internal use) helper functions.
"""

import contextlib
import inspect
import functools
from typing import Callable
import warnings

import cupy as cp

from nvmath.internal.utils import infer_object_package, device_ctx, create_empty_tensor
from nvmath.internal import package_wrapper
from nvmath.internal.tensor_wrapper import maybe_register_package

from . import tensor_wrapper
from .package_ifc import StreamHolder


def _create_stream_ctx_ptr_cupy_stream(package_ifc, stream):
    """
    Utility function to create a stream context as a "package-native" object, get stream pointer as well as
    create a cupy stream object.
    """
    stream_ctx = package_ifc.to_stream_context(stream)
    stream_ptr = package_ifc.to_stream_pointer(stream)
    stream = cp.cuda.ExternalStream(stream_ptr)

    return stream, stream_ctx, stream_ptr


def is_hashable(obj):
    try:
        hash(obj)
    except TypeError:
        return False
    return True

@functools.lru_cache(maxsize=128)
def cached_get_or_create_stream(device_id, stream, op_package):
    maybe_register_package(op_package)
    op_package_ifc = package_wrapper.PACKAGE[op_package]
    if stream is None:
        stream = op_package_ifc.get_current_stream(device_id)
        obj, ctx, ptr = _create_stream_ctx_ptr_cupy_stream(op_package_ifc, stream)
        return StreamHolder(
            **{'ctx': ctx, 'obj': obj, 'ptr': ptr, 'device_id': device_id, 'package': op_package})

    if isinstance(stream, int):
        ptr = stream
        if op_package == 'torch':
            message = "A stream object must be provided for PyTorch operands, not stream pointer."
            raise TypeError(message)
        obj = cp.cuda.ExternalStream(ptr)
        ctx = op_package_ifc.to_stream_context(obj)
        return StreamHolder(
            **{'ctx': ctx, 'obj': obj, 'ptr': ptr, 'device_id': device_id, 'package': op_package})

    stream_package = infer_object_package(stream)
    if stream_package != op_package:
        message = "The stream object must belong to the same package as the tensor network operands."
        raise TypeError(message)

    obj, ctx, ptr = _create_stream_ctx_ptr_cupy_stream(op_package_ifc, stream)
    return StreamHolder(
        **{'ctx': ctx, 'obj': obj, 'ptr': ptr, 'device_id': device_id, 'package': op_package})


def get_or_create_stream(device_id, stream, op_package):
    """
    Create a stream object from a stream pointer or extract the stream pointer from a stream object, or
    use the current stream.

    Args:
        device_id: The device ID.
        stream: A stream object, stream pointer, or None.
        op_package: The package the tensor network operands belong to.

    Returns:
        StreamHolder: Hold a CuPy stream object, package stream context, stream pointer, ...
    """
    if stream is not None and is_hashable(stream): # cupy.cuda.Stream from cupy-10.4 is unhashable (if one installs cupy from conda with cuda11.8)
        return cached_get_or_create_stream(device_id, stream, op_package)
    else:
        return cached_get_or_create_stream.__wrapped__(device_id, stream, op_package)


class Value:
    """
    A simple value wrapper holding a default value.
    """
    def __init__(self, default, *, validator: Callable[[object], bool]):
        """
        Args:
            default: The default value to use.
            validator: A callable that validates the provided value.
        """
        self.validator = validator
        self._data = default

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = self._validate(value)

    def _validate(self, value):
        if self.validator(value):
            return value
        raise ValueError(f"Internal Error: value '{value}' is not valid.")


@contextlib.contextmanager
def cuda_call_ctx(stream_holder, blocking=True, timing=True):
    """
    A simple context manager that provides (non-)blocking behavior depending on the `blocking` parameter for CUDA calls.
      The call is timed only for blocking behavior when timing is requested.

    An `end` event is recorded after the CUDA call for use in establishing stream ordering for non-blocking calls. This
    event is returned together with a `Value` object that stores the elapsed time if the call is blocking and timing is
    requested, or None otherwise.
    """
    stream = stream_holder.obj

    if blocking:
       start = cp.cuda.Event(disable_timing = False if timing else True)
       stream.record(start)

    end = cp.cuda.Event(disable_timing = False if timing and blocking else True)

    time = Value(None, validator=lambda v: True)
    yield end, time

    stream.record(end)

    if not blocking:
        return

    end.synchronize()

    if timing:
        time.data = cp.cuda.get_elapsed_time(start, end)


def get_mpi_comm_pointer(comm):
    """Simple helper to get the address to and size of a ``MPI_Comm`` handle.

    Args:
        comm (mpi4py.MPI.Comm): An MPI communicator.

    Returns:
        tuple: A pair of int values representing the address and the size.
    """
    try:
        from mpi4py import MPI  # init!
    except ImportError as e:
        raise RuntimeError("please install mpi4py") from e

    if not isinstance(comm, MPI.Comm):
        raise ValueError("invalid MPI communicator")
    comm_ptr = MPI._addressof(comm)  # = MPI_Comm*
    mpi_comm_size = MPI._sizeof(MPI.Comm)
    return comm_ptr, mpi_comm_size

def deprecate_function(my_func, message, deprecation_class):
    def add_deprecation_warning(message):
        def decorator(func):
            @functools.wraps(func) 
            def wrapper(*args, **kwargs):
                warnings.warn(message, deprecation_class, stacklevel=2)
                return func(*args, **kwargs)
            return wrapper
        return decorator
    return add_deprecation_warning(message)(my_func)

def deprecate_class(cls, message, deprecation_class):
    class DeprecatedClass(cls):
        def __new__(cls, *args, **kwargs):
            warnings.warn(message, deprecation_class, stacklevel=2)
            return super(DeprecatedClass, cls).__new__(cls)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    DeprecatedClass.__name__ = cls.__name__
    DeprecatedClass.__doc__ = cls.__doc__
    return DeprecatedClass

def deprecate(api, message, deprecation_class):
    if inspect.isfunction(api):
        return deprecate_function(api, message, deprecation_class)
    elif inspect.isclass(api):
        return deprecate_class(api, message, deprecation_class)
    else:
        raise ValueError(f'API type {type(api)} not supported')