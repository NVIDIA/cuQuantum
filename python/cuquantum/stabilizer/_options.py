# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
from logging import Logger
from typing import Optional, Union, Any, Literal, Sequence
from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from cuquantum.bindings import custabilizer as custab
import cuda.bindings.runtime as cudart
from nvmath import memory
from nvmath.internal import utils as nvmath_utils
from nvmath.internal.tensor_wrapper import maybe_register_package
from nvmath.memory import BaseCUDAMemoryManager, BaseCUDAMemoryManagerAsync

from nvmath.internal import tensor_wrapper

from .utils import Stream


@dataclass
class Options:
    """A data class for providing options to cuStabilizer objects.

    Attributes:
        device_id : int
            CUDA device ordinal (default: 0). Device 0 will be used if not specified.
        handle : Optional[Any]
            cuStabilizer library handle. A handle will be created if one is not provided.
        logger : Optional[Logger]
            Python Logger object. The root logger will be used if not provided.
        allocator : Optional[BaseCUDAMemoryManager]
            An object that supports the BaseCUDAMemoryManager protocol,
            used to draw device memory. If not provided, cupy.cuda.alloc will be used.
    """

    device_id: int = 0
    handle: Optional[Any] = None
    logger: Optional[Logger] = None
    allocator: Optional[BaseCUDAMemoryManager] = None


class _ManagedOptions:
    """A class for managing options that objects own."""

    options: Options

    handle: Any
    logger: Logger

    allocator: Union[BaseCUDAMemoryManagerAsync, BaseCUDAMemoryManager]
    operands_package: str
    operands_device_id: Union[int, Literal["cpu"]]
    device_id: int
    package: str

    _buffers: set[Union[memory.MemoryPointer, memory._UnmanagedMemoryPointer]]

    _own_handle: bool = False

    def __init__(
        self,
        options: Options,
        package: str,
    ):
        self.options = options
        self._buffers = set()
        self.logger = (
            options.logger if options.logger is not None else logging.getLogger()
        )
        self.device_id = options.device_id
        self.set_package(package)

        if options.handle is not None:
            self._own_handle = False
            self.handle = options.handle
        else:
            self._own_handle = True
            self.handle = custab.create()

    def set_package(self, package: str):
        self.package = package if package != "numpy" else "cuda"
        maybe_register_package(self.package)
        self.allocator = (
            self.options.allocator
            if self.options.allocator is not None
            else memory._MEMORY_MANAGER[self.package](self.device_id, self.logger)
        )

    def on_new_operands(
        self, operands_package: str, operands_device_id: Union[int, str]
    ):
        self.operands_package = operands_package
        self.operands_device_id = operands_device_id
        maybe_register_package(operands_package)
        if isinstance(operands_device_id, str):
            if operands_device_id != "cpu":
                raise ValueError(f"Invalid operands_device_id: {operands_device_id}")
        else:
            if operands_device_id != self.device_id:
                raise ValueError(
                    f"Operands device ID({operands_device_id}) does"
                    f" not match options.device_id({self.options.device_id})"
                )
        self.set_package(operands_package)

    def allocate_memory(
        self, num_bytes: int, stream: Stream = None, reset=False
    ) -> memory.MemoryPointer:
        stream_holder = nvmath_utils.get_or_create_stream(
            self.device_id, stream, self.package
        )
        self.logger.debug(f"Allocating {num_bytes} bytes on device {self.device_id}")
        with nvmath_utils.device_ctx(self.device_id), stream_holder.ctx:
            if isinstance(self.allocator, memory.BaseCUDAMemoryManagerAsync):
                ptr = self.allocator.memalloc_async(num_bytes, stream_holder.obj)
            else:
                ptr = self.allocator.memalloc(num_bytes)  # type: ignore[union-attr]
            if reset:
                s = stream_holder.obj
                stream_ptr = int(s.handle) if s is not None and hasattr(s, 'handle') else 0
                cudart.cudaMemsetAsync(ptr.device_ptr, 0, num_bytes, stream_ptr)
            self._buffers.add(ptr)
            return ptr

    def allocate_tensor(
        self, shape: Sequence[int], stream: Stream = None, reset=False
    ) -> tensor_wrapper.TensorHolder:
        stream_holder = nvmath_utils.get_or_create_stream(
            self.device_id, stream, self.package
        )
        holderType = tensor_wrapper._TENSOR_TYPES[self.package]
        tensor = holderType.empty(
            shape, device_id=self.device_id, stream_holder=stream_holder
        )
        if reset:
            if self.package == "numpy":
                raise ValueError("Cannot reset numpy tensors")
            cudart.cudaMemset(tensor.data_ptr, 0, tensor.size)
        return tensor

    def get_or_create_stream(self, stream: Stream = None) -> nvmath_utils.StreamHolder:
        return nvmath_utils.get_or_create_stream(self.device_id, stream, self.package)

    def __del__(self):
        if not hasattr(self, "handle"):
            return
        self.logger.debug("Options destructor called")
        if self._own_handle:
            custab.destroy(self.handle)
        for ptr in self._buffers:
            if isinstance(ptr, memory._UnmanagedMemoryPointer):
                self.logger.debug(
                    f"Freeing unmanaged memory pointer {ptr.device_ptr:x}"
                )
                ptr.free()
        self._buffers.clear()
