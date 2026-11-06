# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = ["WorkStream"]

from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Optional, Union, Tuple, TYPE_CHECKING
import weakref
import collections

import cupy as cp
import numpy as np

if TYPE_CHECKING:
    import mpi4py.MPI

from nvmath.internal import utils as nvmath_utils
from nvmath.internal.mem_limit import check_memory_str

from . import memory
from .memory import BaseCUDAMemoryManager
from cuquantum._internal import utils as cutn_utils

from cuquantum.bindings import cudensitymat as cudm
from ._internal.library_handle import LibraryHandle
from ._internal import utils
from ._internal.utils import InvalidObjectState


# TODO[OPTIONAL]: move this map elsewhere
WORK_SPACE_KIND_MAP = {}
WORK_SPACE_KIND_MAP["SCRATCH"] = cudm.WorkspaceKind.WORKSPACE_SCRATCH
# WORK_SPACE_KIND_MAP["CACHE"] = cudm.WorkspaceKind.WORKSPACE_CACHE #Not yet implemented

# TODO[OPTIONAL]: move this map elsewhere
MEM_SPACE_MAP = {}
MEM_SPACE_MAP["DEVICE"] = cudm.Memspace.DEVICE
MEM_SPACE_MAP["HOST"] = cudm.Memspace.HOST


@dataclass
class WorkStream:
    """
    A data class containing the library handle, stream, workspace and configuration parameters.

    This object handles allocation and synchronization automatically. Additionally, a method to release the workspace is provided. The size of the workspace buffer is determined by either the :attr:`memory_limit` attribute or the maximum required workspace size among all objects using this ``WorkStream``.

    Attributes:
        device_id: CUDA device ordinal (used if the tensor network resides on the CPU). Device 0 will be used if not specified.
        stream: CUDA stream. The current stream will be used if not specified.
        memory_limit: Maximum memory available. It can be specified as a value (with optional suffix like
            K[iB], M[iB], G[iB]) or as a percentage. The default is 80% of the device memory.
        allocator: An object that supports the :class:`BaseCUDAMemoryManager` protocol, used to draw device memory. If an allocator is not provided, a memory allocator from the library package will be used (:func:`torch.cuda.caching_allocator_alloc` for PyTorch operands, :func:`cupy.cuda.alloc` otherwise).
        compute_type (cuquantum.ComputeType): CUDA compute type. A suitable compute type will be selected if not specified.
        logger (logging.Logger): Python Logger object. The root logger will be used if a logger object is not provided.
        workspace_info: A property attribute that stores a 2-tuple of ints representing currently allocated and anticipated device workspace size in bytes.
        host_workspace_info: A property attribute that stores a 2-tuple of ints representing currently allocated and anticipated host workspace size in bytes. Host workspace is allocated (as pageable host memory) and attached automatically whenever a ``prepare`` call reports a host requirement, e.g. for the SVD inside 2-site MPS updates.

    Methods:

        set_communicator(comm: "mpi4py.MPI.Comm" | Tuple[int,int] | int | None, provider="None") -> None
            Register a communicator with the library.
            For ``"MPI"`` provider: pass an ``mpi4py.MPI.Comm`` object, an integer pointer, or a tuple of (pointer, size).
            For ``"NCCL"`` provider: pass an integer pointer or a tuple of (pointer, size) pointing to an existing ``ncclComm_t``.
            For ``"None"`` provider: pass ``comm=None`` to disable distributed execution.

        get_proc_rank() -> int
            Return the process rank if a communicator was set previously via :meth:`WorkStream.set_communicator`.

        get_num_ranks() -> int
            Return the number of processes in the communicator that was set previously via :meth:`WorkStream.set_communicator`.

        get_communicator()
            Return the communicator object if set previously via :meth:`WorkStream.set_communicator`.

        release_workspace(kind="SCRATCH") -> None
            Release the workspace.

    .. note::
        - Releasing the workspace releases both its workspace buffer and resets the maximum required size among the objects that uses this ``WorkStream`` instance.
        - Objects which have previously been exposed to this ``WorkStream`` instance do not require explicit calls to their ``prepare`` methods after the workspace has been released.
        - Releasing the workspace buffer may be useful when intermediate computations do not involve the cuDensityMat API, or when the following computations require less workspace than the preceding ones.
        - Objects can only interact with each other if they use the same ``WorkStream`` and cannot change the ``WorkStream`` they use.
        - Some objects require a ``WorkStream`` at creation (``State``, :class:`OperatorAction`), while other objects require it only when their ``prepare`` method is called (:class:`Operator`).
        - Some objects acquire the ``WorkStream`` possibly indirectly (:class:`Operator`), while other objects acquire it always indirectly (:class:`OperatorTerm`, :class:`DenseOperator`, :class:`MultidiagonalOperator`).

    .. attention::
        The ``compute_type`` argument is currently not used and will default to the data type.

    Examples:

        >>> import cupy as cp
        >>> from cuquantum.densitymat import WorkStream

        To create a ``WorkStream`` on a new CUDA stream, we can do

        >>> ctx = WorkStream(stream=cp.cuda.Stream())
    """

    device_id: Optional[int] = None
    stream: Optional[cp.cuda.Stream] = None
    memory_limit: Optional[Union[int, str]] = r"80%"
    allocator: Optional[BaseCUDAMemoryManager] = memory._MEMORY_MANAGER["cupy"]
    compute_type: Optional[str] = None
    logger: Optional[Logger] = None

    def __post_init__(self):
        """
        Cast to cuquantum types, infer values dependent on multiple attributes, create handle if not passed and perform checks.
        """
        # register dummy finalizer, for safe cleanup if error occurs before proper finalizer is set
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()

        self.blocking = True  # TODO: Support non-blocking WorkStream
        self.logger = getLogger() if self.logger is None else self.logger
        self.device_id = self.device_id if self.device_id is not None else 0
        self._handle = LibraryHandle(self.device_id, self.logger)
        self._do_timing = bool(self.logger and self.logger.handlers)
        # TODO: remove restrictions to cupy.cuda.Stream
        self._stream_holder = cutn_utils.get_or_create_stream(self.device_id, self.stream, "cupy")
        self.stream = self._stream_holder.obj
        check_memory_str(self.memory_limit, "memory limit")
        self._memory_limit = nvmath_utils.get_memory_limit_from_device_id(
            self.memory_limit, self.device_id
        )
        if issubclass(self.allocator, BaseCUDAMemoryManager):
            self.allocator = self.allocator(self.device_id, self.logger)
        if not isinstance(self.allocator, BaseCUDAMemoryManager):
            raise TypeError(
                "The allocator must be an object of type (or subclass of) that fulfils the BaseCUDAMemoryManager protocol."
            )

        # internal resource creation and release
        self._ptr = cudm.create_workspace(
            self._handle._validated_ptr
        )  # lifetime tied to instance lifetime
        self.logger.debug(
            f"WorkStream instance {self} created workspace descriptor {self._ptr} on device {self.device_id} with stream {self.stream}."
        )
        self._upstream_finalizers = collections.OrderedDict()
        self._finalizer = weakref.finalize(
            self,
            utils.generic_finalizer,
            self.logger,
            self._upstream_finalizers,
            (cudm.destroy_workspace, self._ptr),
            msg=f"Destroying Workspace instance {self}",
        )
        utils.register_with(self, self._handle, self.logger)

        # initialize other private attributes
        self._buf_scratch = None
        self._size_scratch = 0
        self._buf_host_scratch = None
        self._size_host_scratch = 0
        self._last_compute_event = None
        self._required_size_upper_bound = 0
        self._required_host_size_upper_bound = 0
        self.logger.info(
            f"Created WorkStream on device {self.device_id} with stream {self.stream}."
        )

    def _check_valid_state(self, *args, **kwargs):
        """ """
        if not self._valid_state:
            raise InvalidObjectState("The workspace cannot be used after resources are free'd")

    @property
    def _valid_state(self):
        return self._finalizer.alive

    @property
    @nvmath_utils.precondition(_check_valid_state)
    def _validated_ptr(self) -> int:
        """
        The workspace descriptor.
        """
        return self._ptr

    @property
    def workspace_info(self) -> Tuple[int, int]:
        """
        Information on current and anticipated workspace size in bytes.

        Returns:
            int
                the size of the currently allocated workspace buffer.
            int
                the size of the workspace buffer to be allocated in the future based on previous ``prepare`` calls of other API objects.
        """
        return self._size_scratch, self._required_size_upper_bound

    @property
    def host_workspace_info(self) -> Tuple[int, int]:
        """
        Information on current and anticipated host workspace size in bytes.

        Returns:
            int
                the size of the currently allocated host workspace buffer.
            int
                the size of the host workspace buffer to be allocated in the future based on previous ``prepare`` calls of other API objects.
        """
        return self._size_host_scratch, self._required_host_size_upper_bound

    def set_communicator(
        self,
        comm: "mpi4py.MPI.Comm | Tuple[int, int] | int | None" = None,
        provider: str = "None",
    ) -> None:
        """
        Register a communicator with the library.

        Args:
            comm: The communicator to use. Accepted types depend on the provider:

                - For ``"MPI"``: an ``mpi4py.MPI.Comm`` instance, an integer pointer, or a tuple of (pointer, size).
                - For ``"NCCL"``: an integer pointer or a tuple of (pointer, size) for an existing ``ncclComm_t``.
                - For ``"None"``: comm should be ``None``.

            provider: The communication backend - ``"None"``, ``"MPI"``, or ``"NCCL"``.

        Examples:
            Using MPI with explicit communicator::

                from mpi4py import MPI
                ctx = WorkStream()
                ctx.set_communicator(MPI.COMM_WORLD, provider="MPI")

            Using NCCL with an existing communicator::

                nccl_comm_ptr = ...
                ctx = WorkStream(device_id=local_rank)
                ctx.set_communicator(nccl_comm_ptr, provider="NCCL")
        """
        self._handle.set_communicator(comm, provider)

    def get_proc_rank(self) -> int:
        """
        Returns the process rank if a communicator was set previously via ``WorkStream.set_communicator``.
        """
        return self._handle.get_proc_rank()

    def get_num_ranks(self) -> int:
        """
        Returns the number of processes in the communicator that was set previously via ``WorkStream.set_communicator``.
        """
        return self._handle.get_num_ranks()

    def get_communicator(self):
        """
        Returns the communicator object if set previously via ``WorkStream.set_communicator``.
        """
        return self._handle._comm

    @nvmath_utils.precondition(_check_valid_state)
    def release_workspace(self, kind="SCRATCH") -> None:
        """
        Releases the workspace.

        This method has no direct user-facing side effects on other API objects.
        Releasing the workspace releases both its workspace buffer and resets the maximum required size among its users. Objects which have previously been exposed to this instance of WorkStream do not require explicit calls to their prepare methods after the workspace has been released.
        Releasing the workspace buffer may be useful when performing intermediate computation not involving the cudensitymat API.
        Furthermore, releasing the workspace buffer may be useful if the following computations require less workspace than the preceding ones.
        """
        self._sync()  # may be redundand currently due to the way the memory buffer works
        if kind.lower() != "scratch":
            raise NotImplementedError(
                'WorkStream object does not support workspaces other than "scratch" at the moment.'
            )
        new_ptr = cudm.create_workspace(self._handle._validated_ptr)
        old_ptr = self._ptr
        # this is required for checks for whether prepare has been called on this workspace for a given instance, if this fails we need to implement a wrapper around the pointer
        assert new_ptr != old_ptr
        self._ptr = cudm.destroy_workspace(self._validated_ptr)
        setattr(self, f"_buf_{kind.lower()}", None)
        setattr(self, f"_size_{kind.lower()}", 0)
        setattr(self, f"_buf_host_{kind.lower()}", None)
        setattr(self, f"_size_host_{kind.lower()}", 0)
        self._ptr = new_ptr
        self._finalizer.detach()
        self._finalizer = weakref.finalize(
            self,
            utils.generic_finalizer,
            self.logger,
            self._upstream_finalizers,
            (cudm.destroy_workspace, self._ptr),
            msg=f"Destroying Workspace instance {self}",
        )
        self._required_size_upper_bound = 0
        self._required_host_size_upper_bound = 0

    def _workspace_set_memory(
        self,
        memory_ptr: int,
        size: int,
        memspace: str = "DEVICE",
        kind: str = "SCRATCH",
    ):
        """
        Attach memory buffer to a workspace descriptor.

        Args:
            memory_ptr: int
                Pointer to memory.
            size: int
                Size of allocated buffer in bytes.
            memspace: str
                "DEVICE" (default) or "HOST".
            kind: str
                "SCRATCH" (default) or "CACHE". Currently only "SCRATCH" is supported.
        """

        self.logger.info(
            f"Attaching {memspace.lower()} memory buffer of size {size} for device {self.device_id} with stream {self.stream}."
        )
        if memspace not in MEM_SPACE_MAP:
            raise ValueError(f'memspace must be "DEVICE" or "HOST", got {memspace!r}.')
        if kind != "SCRATCH":
            raise NotImplementedError(
                'Currently only kind = "SCRATCH" is supported in cudensitymat.workspace_set_memory.'
            )

        cudm.workspace_set_memory(
            self._handle._validated_ptr,
            self._validated_ptr,
            MEM_SPACE_MAP[memspace],
            WORK_SPACE_KIND_MAP[kind],
            memory_ptr,
            size,
        )

    def _update_required_size_upper_bound(self, memspace="DEVICE", kind="SCRATCH") -> tuple[int]:
        """
        Updates the upper bound to workspace sizes required among all previous prepare calls.

        The host requirement reported by the same prepare call is tracked alongside the device
        requirement, so that callers only need to track the device size.

        Returns:
            int:
                Workspace size required by most recent prepare call in ``memspace``.
            int:
                Upper bound to workspace sizes required in ``memspace``.
        """
        if memspace == "DEVICE":
            self._update_required_size_upper_bound("HOST", kind)
        _, size = cudm.workspace_get_memory(
            self._handle._validated_ptr,
            self._validated_ptr,
            MEM_SPACE_MAP[memspace],
            WORK_SPACE_KIND_MAP[kind],
        )
        bound_attr = "_required_size_upper_bound" if memspace == "DEVICE" else "_required_host_size_upper_bound"
        setattr(self, bound_attr, max(getattr(self, bound_attr), size))
        return size, getattr(self, bound_attr)

    @nvmath_utils.precondition(_check_valid_state)
    def _sync(self) -> None:
        if self._last_compute_event:
            self._last_compute_event.synchronize()
            self._last_compute_event = None

    @nvmath_utils.precondition(_check_valid_state)
    def _maybe_allocate(self, memspace="DEVICE", kind="SCRATCH") -> None:
        """
        Allocates workspace buffer and attaches it to workspace descriptor, if necessary.

        Allocating the device workspace also allocates and attaches the host workspace when a
        previous ``prepare`` call reported a host requirement, so callers only need to request
        the device workspace.

        Args:
            memspace: str
                "DEVICE" (default) or "HOST".
            kind: str
                "SCRATCH" (default) or "CACHE". Currently only "SCRATCH" is supported.
        """
        if kind != "SCRATCH":
            raise NotImplementedError('Currently only kind = "SCRATCH" is supported.')
        if memspace == "HOST":
            self._maybe_allocate_host(kind)
            return
        if memspace != "DEVICE":
            raise ValueError(f'memspace must be "DEVICE" or "HOST", got {memspace!r}.')
        self._maybe_allocate_host(kind)
        _ptr, _size = cudm.workspace_get_memory(
            self._handle._validated_ptr,
            self._validated_ptr,
            MEM_SPACE_MAP[memspace],
            WORK_SPACE_KIND_MAP[kind],
        )
        if _ptr == 0:
            # normal state after prepare call
            _buf_size = getattr(self, f"_size_{kind.lower()}")
            if _buf_size is None or _buf_size < self._required_size_upper_bound:
                with nvmath_utils.device_ctx(self.device_id), self._stream_holder.ctx:
                    try:
                        self.logger.info(
                            f"Allocating memory buffer of size {self._required_size_upper_bound} on device {self.device_id} with stream {self.stream}."
                        )
                        _buf = self.allocator.memalloc(self._required_size_upper_bound)
                    except TypeError as e:
                        message = (
                            "The method 'memalloc' in the allocator object must conform to the interface in the "
                            "'BaseCUDAMemoryManager' protocol."
                        )
                        raise TypeError(message) from e
                setattr(self, f"_size_{kind.lower()}", self._required_size_upper_bound)
                setattr(self, f"_buf_{kind.lower()}", _buf)
                ptr = _buf.ptr if _buf is not None else 0
            else:
                _buf = getattr(self, f"_buf_{kind.lower()}")
                ptr = _buf.ptr if _buf is not None else 0

            self._workspace_set_memory(ptr, getattr(self, f"_size_{kind.lower()}"), memspace, kind)

        else:
            # buffer currently attached
            # nothing to do here
            assert (
                getattr(self, f"_buf_{kind.lower()}").ptr,
                getattr(self, f"_size_{kind.lower()}"),
            ) == (_ptr, _size)

    def _maybe_allocate_host(self, kind="SCRATCH") -> None:
        """
        Allocates the host workspace buffer and attaches it to the workspace descriptor, if a
        previous ``prepare`` call reported a host requirement and no sufficient buffer is attached.

        The buffer is pageable host memory owned by this instance; it is kept alive until the
        workspace is released or a larger buffer replaces it.
        """
        required = self._required_host_size_upper_bound
        if required == 0:
            return
        _ptr, _size = cudm.workspace_get_memory(
            self._handle._validated_ptr,
            self._validated_ptr,
            MEM_SPACE_MAP["HOST"],
            WORK_SPACE_KIND_MAP[kind],
        )
        buf_attr, size_attr = f"_buf_host_{kind.lower()}", f"_size_host_{kind.lower()}"
        if _ptr != 0 and _size >= required:
            # buffer currently attached and large enough
            assert getattr(self, buf_attr) is not None and getattr(self, buf_attr).ctypes.data == _ptr
            return
        if getattr(self, buf_attr) is None or getattr(self, size_attr) < required:
            self.logger.info(f"Allocating host memory buffer of size {required} for device {self.device_id}.")
            setattr(self, buf_attr, np.empty(required, dtype=np.uint8))
            setattr(self, size_attr, required)
        buf = getattr(self, buf_attr)
        self._workspace_set_memory(buf.ctypes.data, getattr(self, size_attr), "HOST", kind)
