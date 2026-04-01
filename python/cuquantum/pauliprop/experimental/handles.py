# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause


from logging import Logger, getLogger

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device
import nvmath.internal.utils as nvmath_utils

import cuquantum.bindings.cupauliprop as cupp
from ._internal.utils import register_finalizer

__all__ = ["LibraryHandle"]


class LibraryHandle:
    def __init__(self,
                device_id : int | None = None,
                logger : Logger | None = None):
        self._device_id = device_id if device_id is not None else Device().device_id
        self._logger = getLogger() if logger is None else logger
        self._ptr: None | int = None
        with nvmath_utils.device_ctx(self._device_id):
            self._ptr = cupp.create()
        self._has_been_freed: bool = False
        self._logger.debug(f"C API cupaulipropCreate returned handle ptr={self._ptr}")
        self._logger.info(f"cuPauliProp library handle created on device {self._device_id}.")
        # Register cleanup finalizer for safe resource release
        self._finalizer = register_finalizer(self, cupp.destroy, self._ptr, self._logger, "LibraryHandle")

    def _check_valid_state(self, *args, **kwargs):
        if self._has_been_freed:
            raise RuntimeError("Trying to use library handle after it has been freed is not supported.")

    @property
    @nvmath_utils.precondition(_check_valid_state)
    def _validated_ptr(self):
        return self._ptr
    
    @property
    def device_id(self):
        return self._device_id

    @property
    def logger(self) -> Logger:
        """The logger instance associated with this library handle."""
        return self._logger

    def __int__(self):
        return self._validated_ptr
