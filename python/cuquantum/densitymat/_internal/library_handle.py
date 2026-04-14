# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
from logging import Logger
import weakref
import collections
from typing import Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mpi4py.MPI

from nvmath.internal import utils as nvmath_utils
from cuquantum._internal import utils as cutn_utils

from cuquantum.bindings import cudensitymat as cudm
from . import utils

# TODO[OPTIONAL] move elsewhere and refactor

_comm_provider_map = {}
_comm_provider_map["None"] = cudm.DistributedProvider.NONE
_comm_provider_map["MPI"] = cudm.DistributedProvider.MPI
_comm_provider_map["NCCL"] = cudm.DistributedProvider.NCCL
# _comm_provider_map["NVSHMEM"] = cudm.DistributedProvider.NVSHMEM


class LibraryHandle:
    """
    A wrapper around the library handle for cudensitymat.
    """

    def __init__(self, device_id: int, logger: Logger):
        """
        Create a library handle on the specified device.

        Parameters:
        -----------
        device_id: int
            If not provided, device 0 will be used by default.
        logger: Logger
            If a logger is passed, creation and destruction of library handle are logged there.
        """
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()

        self._device_id = device_id
        with nvmath_utils.device_ctx(self._device_id):
            self._ptr = cudm.create()
        self.logger = logger
        self._comm = None
        self._comm_set = False
        self._nccl_comm_holder = None  # Numpy array holding ncclComm_t pointer for stable address
        self.logger.info(f"cuDensityMat library handle created on device {self.device_id}.")
        self.logger.debug(
            f"{self} instance holds cuDensityMat library handle with pointer {self._ptr} on device {self.device_id}."
        )
        self._upstream_finalizers = collections.OrderedDict()
        self._finalizer = weakref.finalize(
            self,
            utils.generic_finalizer,
            self.logger,
            self._upstream_finalizers,
            (cudm.destroy, self._ptr),
            msg=f"Destroying Handle instance {self}",
        )  # may also use trivial finalizer here
        self.logger.debug(f"{self} instance's finalizer registered.")

    def _check_valid_state(self, *args, **kwargs):
        if not self._valid_state:
            raise utils.InvalidObjectState("The handle cannot be used after resources are freed.")

    @property
    def _valid_state(self):
        return self._finalizer.alive

    @property
    @nvmath_utils.precondition(_check_valid_state)
    def _validated_ptr(self):
        return self._ptr

    @property
    def device_id(self):
        return self._device_id

    def set_communicator(
        self,
        comm: "mpi4py.MPI.Comm" | Tuple[int, int] | int | None = None,
        provider: str = "None",
    ) -> None:
        """
        Sets the communicator attached to the current context's library handle.

        Parameters:
        -----------
        comm:
            The communicator instance. Accepted types depend on the provider:
            
            - For "MPI": an ``mpi4py.MPI.Comm`` instance, an integer pointer, or a tuple of (pointer, size).
            - For "NCCL":
                - An integer pointer or a tuple of (pointer, size) is required and is assumed to be an ``ncclComm_t`` pointer.
            - For "None": comm should be ``None``.
            
        provider: str
            The communication backend: "None", "MPI", or "NCCL".
        """
        if self._comm_set:
            raise RuntimeError(
                "Communicator has already been set on library handle. "
                "Resetting the communicator is not supported."
            )

        if provider not in _comm_provider_map:
            raise ValueError(
                f"Unknown provider: {provider}. Supported: {list(_comm_provider_map.keys())}"
            )

        if provider == "NCCL":
            self._set_nccl_communicator(comm)
        elif provider == "MPI":
            self._set_mpi_communicator(comm)
        else:  # provider == "None"
            cudm.reset_distributed_configuration(
                self._validated_ptr, _comm_provider_map["None"], 0, 0
            )

        self._comm_set = True

    def _set_mpi_communicator(
        self, comm: Union["mpi4py.MPI.Comm", Tuple[int, int], int]
    ) -> None:
        """Set MPI as the distributed provider."""
        if comm is None:
            raise ValueError(
                "MPI provider requires an explicit communicator. "
                "Pass an mpi4py.MPI.Comm or a (pointer, size) tuple."
            )

        self._comm = comm
        if isinstance(comm, Sequence):
            _comm_ptr, _size = comm
        elif isinstance(comm, int):
            _comm_ptr = comm
            _size = np.dtype(np.intp).itemsize
        else:
            _comm_ptr, _size = cutn_utils.get_mpi_comm_pointer(comm)
        cudm.reset_distributed_configuration(
            self._validated_ptr, _comm_provider_map["MPI"], _comm_ptr, _size
        )

    def _set_nccl_communicator(self, comm: Tuple[int, int] | int) -> None:
        """
        Set NCCL as the distributed provider.

        The NCCL communicator must be provided as an integer pointer or a tuple of (pointer, size).
        """
        if comm is None:
            raise ValueError(
                "NCCL provider requires an explicit communicator. "
                "Pass an integer pointer or a (pointer, size) tuple for an existing ncclComm_t."
            )

        if isinstance(comm, Sequence) and len(comm) == 2:
            # Explicit (pointer, size) tuple provided - assume ncclComm_t pointer (user-managed)
            nccl_comm_ptr = comm[0]
        elif isinstance(comm, int):
            nccl_comm_ptr = comm
        else:
            raise ValueError(
                "NCCL provider requires an ncclComm_t pointer or (pointer, size) tuple."
            )

        self.logger.info(
            f"Using explicit ncclComm_t pointer (comm_ptr={nccl_comm_ptr}) on device {self.device_id}."
        )

        # Pass the ncclComm_t pointer directly (like MPI passes MPI_Comm*)
        # Store in numpy array to get stable address; np.intp matches pointer size
        nccl_comm_holder = np.array([nccl_comm_ptr], dtype=np.intp)
        cudm.reset_distributed_configuration(
            self._validated_ptr,
            _comm_provider_map["NCCL"],
            nccl_comm_holder.ctypes.data,
            nccl_comm_holder.itemsize,
        )

        # All configuration succeeded - now update instance state
        self._comm = nccl_comm_ptr
        self._nccl_comm_holder = nccl_comm_holder

    def get_communicator(self):
        """
        Returns the communicator associated with the given library context.
        """
        return self._comm

    @nvmath_utils.precondition(_check_valid_state)
    def get_num_ranks(self) -> int:
        """
        Returns the total number of distributed processes associated with the given library context.
        """
        return cudm.get_num_ranks(self._validated_ptr)

    @nvmath_utils.precondition(_check_valid_state)
    def get_proc_rank(self) -> int:
        """
        Returns the rank of the current process in the distributed configuration associated with the given library context.
        """
        return cudm.get_proc_rank(self._validated_ptr)

    @nvmath_utils.precondition(_check_valid_state)
    def set_random_seed(self, seed: int) -> None:
        """
        Sets the random seed used by the random number generator inside the library context.
        """
        cudm.reset_random_seed(self._validated_ptr, seed)

    @nvmath_utils.precondition(_check_valid_state)
    def _register_user(self, user):
        assert self != user
        self._upstream_finalizers[user._finalizer] = weakref.ref(
            user
        )  # We may not want to store weakref as value here, but let's see
        self.logger.debug(f"{self} registered user {user} for finalizer execution.")

    def _unregister_user(self, user):
        assert self != user
        if self._upstream_finalizers is not None:
            del self._upstream_finalizers[user._finalizer]
            self.logger.debug(f"{self} unregistered user {user} for finalizer execution.")
