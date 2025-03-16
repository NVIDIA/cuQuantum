# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
from logging import Logger
import weakref
import collections
from typing import Sequence, Tuple

from cuquantum._internal import utils as cutn_utils

from cuquantum.bindings import cudensitymat as cudm
from . import utils

# TODO[OPTIONAL] move elsewhere and refactor

_comm_provider_map = {}
_comm_provider_map["None"] = cudm.DistributedProvider.NONE
_comm_provider_map["MPI"] = cudm.DistributedProvider.MPI
# _comm_provider_map["NCCL"] = cudm.DistributedProvider.NCCL
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
        with cutn_utils.device_ctx(self._device_id):
            self._ptr = cudm.create()
        self.logger = logger
        self._comm = None
        self._comm_set = False
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
    @cutn_utils.precondition(_check_valid_state)
    def _validated_ptr(self):
        return self._ptr

    @property
    def device_id(self):
        return self._device_id

    def set_communicator(
        self, comm: "mpi4py.MPI.Comm" | Tuple[int, int], provider: str = "None"
    ) -> None:
        """
        Sets the communicator attached to the current context's library handle.

        Parameters:
        -----------
        comm:
            The communicator instance with which to set the library context's communicator.
            Can either be specified as an instance of `mpi4py.Comm` or
            as a tuple of two integers (the pointer to the communicator and its size).
        provider: str
            The package/backend providing the communicator.
        """
        if self._comm_set:
            raise RuntimeError(
                "Communicator has already been set on library handle.\
            Resetting the communicator is not supported."
            )
        assert provider in ["None", "MPI"]
        self._comm = comm
        if isinstance(comm, Sequence):
            _comm_ptr, _size = comm
        else:
            _comm_ptr, _size = cutn_utils.get_mpi_comm_pointer(comm)
        cudm.reset_distributed_configuration(
            self._validated_ptr, _comm_provider_map[provider], _comm_ptr, _size
        )
        self._comm_set = True

    def get_communicator(self):
        """
        Returns the communicator associated with the given library context.
        """
        return self._comm

    @cutn_utils.precondition(_check_valid_state)
    def get_num_ranks(self) -> int:
        """
        Returns the total number of distributed processes associated with the given library context.
        """
        return cudm.get_num_ranks(self._validated_ptr)

    @cutn_utils.precondition(_check_valid_state)
    def get_proc_rank(self) -> int:
        """
        Returns the rank of the current process in the distributed configuration associated with the given library context.
        """
        return cudm.get_proc_rank(self._validated_ptr)

    @cutn_utils.precondition(_check_valid_state)
    def set_random_seed(self, seed: int) -> None:
        """
        Sets the random seed used by the random number generator inside the library context.
        """
        cudm.reset_random_seed(self._validated_ptr, seed)

    @cutn_utils.precondition(_check_valid_state)
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
