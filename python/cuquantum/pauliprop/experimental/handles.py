# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause


from logging import Logger, getLogger
from typing import TYPE_CHECKING

import numpy as np

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device
import nvmath.internal.utils as nvmath_utils

from cuquantum._internal.utils import get_mpi_comm_pointer
import cuquantum.bindings.cupauliprop as cupp
from ._internal.distributed import (
    ProviderName,
    is_nccl_comm,
    is_nvmath_distributed_context,
    is_ptr_and_size,
    resolve_provider,
)
from ._internal.utils import register_finalizer

if TYPE_CHECKING:
    import mpi4py.MPI
    from nccl.core import Communicator as NcclCommunicator

__all__ = ["LibraryHandle"]


class LibraryHandle:
    """A cuPauliProp library context.

    A new handle uses single-process execution. To enable multi-process
    execution, call :meth:`set_communicator` before creating any
    :class:`~cuquantum.pauliprop.experimental.PauliExpansion` or other object
    owned by the handle.

    Args:
        device_id: GPU device ordinal. If omitted, use the current device.
        logger: Logger used by the handle and its child objects.
    """

    def __init__(self,
                device_id : int | None = None,
                logger : Logger | None = None):
        self._device_id = device_id if device_id is not None else Device().device_id
        self._logger = getLogger() if logger is None else logger
        self._ptr: None | int = None
        with nvmath_utils.device_ctx(self._device_id):
            self._ptr = cupp.create()
        self._comm = None
        self._nccl_comm_holder = None
        self._logger.debug(f"C API cupaulipropCreate returned handle ptr={self._ptr}")
        self._logger.info(f"cuPauliProp library handle created on device {self._device_id}.")
        # Register cleanup finalizer for safe resource release
        self._finalizer = register_finalizer(self, cupp.destroy, self._ptr, self._logger, "LibraryHandle")

    def _check_valid_state(self, *args, **kwargs):
        if not self._finalizer.alive:
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

    def set_communicator(
        self,
        comm: "mpi4py.MPI.Comm | NcclCommunicator | DistributedContext | tuple[int, int] | int | None" = None,
        provider: ProviderName | cupp.DistributedProvider | None = None,
    ) -> None:
        """Register a communicator with this library handle.

        This is a collective operation over all processes in the current or
        replacement communicator and may block for synchronization. Every
        expansion subsequently created from this handle is distributed across
        that communicator.

        ``provider`` is inferred from ``mpi4py.MPI.Comm``,
        ``nccl.core.Communicator``, and ``nvmath.distributed.DistributedContext``
        (which supplies its NCCL communicator and therefore requires
        initialization with the ``"nccl"`` backend). Raw ``int`` /
        ``(ptr, size)`` inputs require an explicit provider.

        Before replacing or clearing a communicator, destroy all expansions,
        views, workspace descriptors, and operator objects created from this
        handle; existing objects are not converted to the new execution mode.
        Keep the communicator alive until the handle and all objects that use
        it have been destroyed.

        Args:
            comm: Communicator object, ``nvmath.distributed`` context, raw
                pointer, ``(ptr, size)`` pair, or ``None``.
            provider: ``"MPI"`` or ``"NCCL"`` (uppercase), or a
                :class:`~cuquantum.bindings.cupauliprop.DistributedProvider`.
                Optional when inferable. Pass no arguments, or pass ``comm=None``,
                to restore single-process execution.
        """
        resolved = resolve_provider(comm, provider)

        if resolved == cupp.DistributedProvider.NCCL:
            self._set_nccl_communicator(comm)
        elif resolved == cupp.DistributedProvider.MPI:
            self._set_mpi_communicator(comm)
        else:
            if comm is not None:
                raise ValueError("Single-process configuration requires comm=None.")
            cupp.reset_distributed_configuration(
                self._validated_ptr, cupp.DistributedProvider.NONE, 0, 0
            )
            self._comm = None
            self._nccl_comm_holder = None

    def _set_mpi_communicator(self, comm) -> None:
        if comm is None:
            raise ValueError(
                "MPI provider requires an explicit communicator. "
                "Pass an mpi4py.MPI.Comm, an integer pointer, or a (pointer, size) tuple."
            )
        if is_ptr_and_size(comm):
            comm_ptr, comm_size = comm
        elif isinstance(comm, int):
            comm_ptr = comm
            comm_size = np.dtype(np.intp).itemsize
        else:
            comm_ptr, comm_size = get_mpi_comm_pointer(comm)
        cupp.reset_distributed_configuration(
            self._validated_ptr, cupp.DistributedProvider.MPI, comm_ptr, comm_size
        )
        self._comm = comm
        self._nccl_comm_holder = None

    def _set_nccl_communicator(self, comm) -> None:
        if comm is None:
            raise ValueError(
                "NCCL provider requires an explicit communicator. "
                "Pass an nccl.core.Communicator, an integer pointer, or a (pointer, size) tuple."
            )
        if is_nvmath_distributed_context(comm):
            nccl_comm = comm.nccl_comm
            if nccl_comm is None:
                raise ValueError(
                    "The nvmath.distributed context has no NCCL communicator. "
                    "Initialize nvmath.distributed with the 'nccl' backend."
                )
            nccl_comm_ptr = nccl_comm if isinstance(nccl_comm, int) else int(nccl_comm.ptr)
        elif is_nccl_comm(comm):
            nccl_comm_ptr = int(comm.ptr)
        elif is_ptr_and_size(comm):
            nccl_comm_ptr = comm[0]
        elif isinstance(comm, int):
            nccl_comm_ptr = comm
        else:
            raise ValueError(
                "NCCL provider requires an nccl.core.Communicator, an "
                "nvmath.distributed context, an ncclComm_t pointer, or a "
                "(pointer, size) tuple."
            )

        self._logger.info(
            f"Using ncclComm_t pointer (comm_ptr={nccl_comm_ptr}) on device {self.device_id}."
        )
        nccl_comm_holder = np.array([nccl_comm_ptr], dtype=np.intp)
        cupp.reset_distributed_configuration(
            self._validated_ptr,
            cupp.DistributedProvider.NCCL,
            nccl_comm_holder.ctypes.data,
            nccl_comm_holder.itemsize,
        )
        self._comm = comm
        self._nccl_comm_holder = nccl_comm_holder

    def get_communicator(self):
        """Return the process-local communicator object passed to :meth:`set_communicator`.

        Returns ``None`` when the handle uses single-process execution.
        """
        return self._comm

    @nvmath_utils.precondition(_check_valid_state)
    def get_num_ranks(self) -> int:
        """Return the number of processes in the handle's current configuration."""
        return cupp.get_num_ranks(self._validated_ptr)

    @nvmath_utils.precondition(_check_valid_state)
    def get_proc_rank(self) -> int:
        """Return this process's rank in the handle's current configuration."""
        return cupp.get_proc_rank(self._validated_ptr)
