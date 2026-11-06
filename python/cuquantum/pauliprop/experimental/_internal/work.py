# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Workspace management for the pauliprop experimental module."""

from typing import Optional, Union, Any
from contextlib import contextmanager

import numpy as np
import nvmath.internal.utils as nvmath_utils
from nvmath.internal.package_ifc import StreamHolder
import nvmath.memory as nvmath_memory

import cuquantum.bindings.cupauliprop as cupp
from . import typemaps
from .utils import register_finalizer


class Workspace:
    """
    Lightweight workspace helper to attach device/host scratch buffers to cuPP descriptors.
    """

    def __init__(
        self,
        library_handle,
        allocator: nvmath_memory.BaseCUDAMemoryManager,
        memory_limit: Optional[Union[int, str]] = r"80%",
    ):
        self._device_id = library_handle.device_id
        self._ptr = cupp.create_workspace_descriptor(int(library_handle))

        try:
            self._library_handle = library_handle
            self._logger = library_handle._logger
            self._logger.debug(f"C API cupaulipropCreateWorkspaceDescriptor returned ptr={self._ptr}")
            self._memory_limit = nvmath_utils.get_memory_limit_from_device_id(memory_limit, self.device_id)
            self._allocator = allocator
            # Register cleanup finalizer for safe resource release
            self._finalizer = register_finalizer(self, cupp.destroy_workspace_descriptor, self._ptr, self._logger, "Workspace")
        except Exception:
            cupp.destroy_workspace_descriptor(self._ptr)
            self._ptr = None
            raise

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def memory_limit(self) -> int:
        return self._memory_limit

    @property
    def allocator(self) -> nvmath_memory.BaseCUDAMemoryManagerAsync:
        return self._allocator

    def __int__(self) -> int:
        if self._ptr is None:
            return 0
        else:
            return self._ptr
        
    def get_required_sizes(self) -> tuple[int, int]:
        """
        Returns the scratch workspace sizes (device_size, host_size) in bytes required after a prepare call.
        """
        
        dev = self.get_required_size(memspace="DEVICE", kind="SCRATCH")
        host = self.get_required_size(memspace="HOST", kind="SCRATCH")
        return dev, host

    def get_required_size(self, memspace="DEVICE", kind="SCRATCH") -> int:
        """
        Returns the workspace size for a given memory space and kind in bytes following a prepare call.
        """
        if kind.lower() != "scratch":
            raise NotImplementedError(
                'Currently only scratch workspaces are supported.'
            )
        if memspace.lower() == "device":
            return cupp.workspace_get_memory_size(
                int(self._library_handle),
                int(self),
                typemaps.MEM_SPACE_MAP["DEVICE"],
                typemaps.WORK_SPACE_KIND_MAP["SCRATCH"],
            )
        elif memspace.lower() == "host":
            return cupp.workspace_get_memory_size(
                int(self._library_handle),
                int(self),
                typemaps.MEM_SPACE_MAP["HOST"],
                typemaps.WORK_SPACE_KIND_MAP["SCRATCH"],
            )
        else:
            raise ValueError(f"Invalid memory space: {memspace}")

    def _allocate(self, device_size: int, host_size: int, stream_holder: StreamHolder) -> tuple[nvmath_memory.MemoryPointer | None, np.ndarray | None]:
        """
        Attach scratch buffers for device and host. Returns the buffers so callers can
        manage lifetimes/synchronization if needed.
        """
        buf_device = None
        buf_host = None

        # Device buffer
        if device_size > 0:
            with nvmath_utils.device_ctx(self.device_id), stream_holder.ctx:
                try:
                    buf_device = self._allocator.memalloc_async(device_size, stream_holder.obj)
                except TypeError as e:
                    msg = (
                        "The method 'memalloc_async' in the allocator object must conform to the interface in the "
                        "'BaseCUDAMemoryManagerAsync' protocol."
                    )
                    raise TypeError(msg) from e
            cupp.workspace_set_memory(
                int(self._library_handle),
                int(self),
                typemaps.MEM_SPACE_MAP["DEVICE"],
                typemaps.WORK_SPACE_KIND_MAP["SCRATCH"],
                buf_device.device_ptr,
                buf_device.size,
            )
        else:
            cupp.workspace_set_memory(
                int(self._library_handle),
                int(self),
                typemaps.MEM_SPACE_MAP["DEVICE"],
                typemaps.WORK_SPACE_KIND_MAP["SCRATCH"],
                0,
                0,
            )

        # Host buffer (allocate via numpy)
        if host_size > 0:
            buf_host = np.empty(host_size, dtype=np.uint8)
            cupp.workspace_set_memory(
                int(self._library_handle),
                int(self),
                typemaps.MEM_SPACE_MAP["HOST"],
                typemaps.WORK_SPACE_KIND_MAP["SCRATCH"],
                buf_host.ctypes.data,
                host_size,
            )
        else:
            cupp.workspace_set_memory(
                int(self._library_handle),
                int(self),
                typemaps.MEM_SPACE_MAP["HOST"],
                typemaps.WORK_SPACE_KIND_MAP["SCRATCH"],
                0,
                0,
            )

        return buf_device, buf_host

    @contextmanager
    def scratch_context(self, device_size: int = 0, host_size: int = 0, stream: Any = None):
        """
        Context manager to allocate/attach scratch buffers using a fresh descriptor,
        restoring the previous descriptor and attached buffers on exit.

        Args:
            device_size: Bytes of device scratch to allocate/attach.
            host_size: Bytes of host scratch to allocate/attach.
            stream: Stream object or pointer; None uses current stream.
        """
        # Preserve current descriptor.
        old_ptr = self._ptr

        # Use a fresh descriptor for this context.
        new_ptr = cupp.create_workspace_descriptor(int(self._library_handle))
        self._ptr = new_ptr

        buf_host = None
        try:
            if isinstance(stream, StreamHolder):
                stream_holder = stream
            else:
                stream_holder = nvmath_utils.get_or_create_stream(self.device_id, stream, self._library_handle._package_str)
            buf_device, buf_host = self._allocate(device_size, host_size, stream_holder)
            yield self, buf_device, buf_host
        finally:
            # Release and destroy the temporary workspace descriptor.
            cupp.destroy_workspace_descriptor(new_ptr)

            # Restore previous descriptor and buffers
            self._ptr = old_ptr

            # Device scratch is freed in stream order, but host scratch is not, so wait for
            # stream work that may still use it. This keeps non-blocking calls safe. Done last
            # so a failed sync cannot skip the cleanup; buf_host stays alive until then.
            if buf_host is not None:
                stream_holder.obj.sync()

