# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from numbers import Number
from typing import Sequence, Any, Tuple, Union, List
import weakref
import collections

import cupy as cp
import numpy as np
from cuquantum._internal import utils as cutn_utils, tensor_wrapper, typemaps
from cuquantum._internal.tensor_ifc import Tensor

from cuquantum.bindings import cudensitymat as cudm
from .work_stream import WorkStream
from ._internal import utils
from ._internal.utils import InvalidObjectState


__all__ = ["DensePureState", "DenseMixedState"]


class State(ABC):
    """
    An base class on which all concrete state representations are based.
    This class mirrors the C-API more closely than its subclasses.

    Args:
        ctx: WorkStream
            Library context and other configuration information.
        hilbert_space_dims: Tuple[int]
            A tuple of the local Hilbert space dimensions.
        purity: str
            The states purity, either "PURE" or "MIXED".
        batch_size: int
            The batch dimension of the state.
        dtype: str
            The numeric datatype for the state's coefficients.
    """

    def __init__(
        self, ctx: WorkStream, hilbert_space_dims: Tuple[int], batch_size: int, dtype: str
    ) -> None:

        self.batch_size = batch_size
        self.hilbert_space_dims = tuple(hilbert_space_dims)
        self.dtype = dtype

        self._bufs = None
        self._last_compute_event = None
        self._ctx = ctx

        # register dummy finalizer, for safe cleanup if error occurs before proper finalizer is set
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()
        self._upstream_finalizers = collections.OrderedDict()  # not really needed here

    def _instantiate(self, ctx: WorkStream):
        assert ctx is not None
        if self._valid_state:
            assert self._ctx == ctx
        else:
            self._ctx = ctx
            # create state handle
            self._ptr = cudm.create_state(
                self._ctx._handle._validated_ptr,
                self._purity,
                len(self.hilbert_space_dims),
                self.hilbert_space_dims,
                self.batch_size,
                typemaps.NAME_TO_DATA_TYPE[self.dtype],
            )
            self._finalizer = weakref.finalize(
                self,
                utils.generic_finalizer,
                self._ctx.logger,
                self._upstream_finalizers,
                (cudm.destroy_state, self._ptr),
                msg=f"Destroying State instance {self}, ptr: {self._ptr}.",
            )
            utils.register_with(self, self._ctx, self._ctx.logger)

    def _check_valid_state(self, *args, **kwargs):
        if not self._valid_state:
            raise InvalidObjectState("The state cannot be used after resources are freed")

    @property
    def _valid_state(self):
        return self._finalizer.alive

    @property
    @cutn_utils.precondition(_check_valid_state)
    def _validated_ptr(self):
        """
        Pointer to C-API counterpart.
        """
        return self._ptr

    @property
    @abstractmethod
    def _purity(self):
        pass

    @property
    @abstractmethod
    def storage(self) -> Any:
        pass

    @abstractmethod
    def attach_storage(self, storage: Any):
        pass

    @property
    @abstractmethod
    def local_info(self) -> Any:
        pass

    def _sync(self) -> None:
        """ """
        if self._last_compute_event:
            self._last_compute_event.synchronize()
            self._last_compute_event = None

    # override in concrete subclasses if other criteria for compatibility exist
    def _check_state_compatibility(self, other):
        try:
            assert type(self) == type(other)
            assert self.hilbert_space_dims == other.hilbert_space_dims
            assert self.batch_size == other.batch_size
            assert self.dtype == other.dtype
            assert self._ctx == other._ctx
            assert self._purity == other._purity
        except AssertionError as e:
            raise ValueError(
                "`other` argument in State.inner(other) is incompatible with instance."
            ) from e

    def _check_and_return_factors(self, factors):
        # Check input shape
        if isinstance(factors, Number):
            factors = np.full((self.batch_size,), factors)
        elif isinstance(factors, Sequence):
            if not len(factors) == self.batch_size:
                raise ValueError("factors must be of same length as State's batch_size.")
            factors = np.array(factors)
        elif isinstance(factors, (np.ndarray, cp.ndarray)):
            if not factors.shape == (self.batch_size,):
                raise ValueError(
                    "factors passed as NDArray must be one-dimensional and of length batch_size."
                )
        else:
            raise TypeError("factors must be of type Number, Sequence, np.ndarray or cp.ndarray.")

        # Put factors onto GPU
        with cutn_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(
            self._ctx, blocking=True
        ):
            if not cp.can_cast(factors, self.dtype, casting="same_kind"):
                raise TypeError(
                    f"The provided scaling factors with data type {type(factors.dtype)} "
                    f"cannot be safely cast to State's data type {self.dtype}."
                )
            factors_arr = cp.asarray(factors, dtype=self.dtype)

        return factors_arr

    @cutn_utils.precondition(_check_valid_state)
    def inplace_scale(self, factors: Union[Number, Sequence, np.ndarray, cp.ndarray]) -> None:
        """
        Scale the state by scalar factor(s).

        Args:
            factors: Scalar factor(s) used in scaling the state. If a single number is provided,
                scale all batched states by the same factor.
        """
        factors_arr = self._check_and_return_factors(factors)

        with cutn_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(self._ctx) as (
            self._last_compute_event,
            elapsed,
        ):
            cudm.state_compute_scaling(
                self._ctx._handle._validated_ptr,
                self._ptr,
                factors_arr.data.ptr,
                self._ctx._stream_holder.ptr,
            )

    @cutn_utils.precondition(_check_valid_state)
    def norm(self) -> cp.ndarray:
        """
        Compute the squared Frobenius norm(s) of the state.

        Returns:
            An array of squared Frobenius norm(s) of length ``batch_size``.
        """
        # Translate complex datatypes to real datatypes
        if self.dtype.startswith("complex"):
            dtype = self.storage.real.dtype.name
        else:
            dtype = self.dtype

        with cutn_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(self._ctx) as (
            self._last_compute_event,
            elapsed,
        ):
            res = cp.empty(self.batch_size, dtype=dtype, order="F")
            cudm.state_compute_norm(
                self._ctx._handle._validated_ptr,
                self._ptr,
                res.data.ptr,
                self._ctx._stream_holder.ptr,
            )
        return res

    @cutn_utils.precondition(_check_valid_state)
    def trace(self) -> cp.ndarray:
        """
        Compute the trace(s) of the state.

        Returns:
            An array of trace(s) of length ``batch_size``.
        """
        with cutn_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(self._ctx) as (
            self._last_compute_event,
            elapsed,
        ):
            res = cp.empty(self.batch_size, dtype=self.dtype, order="F")
            cudm.state_compute_trace(
                self._ctx._handle._validated_ptr,
                self._ptr,
                res.data.ptr,
                self._ctx._stream_holder.ptr,
            )
        return res

    @cutn_utils.precondition(_check_valid_state)
    def inplace_accumulate(
        self, other, factors: Union[Number, Sequence, np.ndarray, cp.ndarray] = 1
    ) -> None:
        """
        Inplace accumulate another state scaled by factor(s) into this state.

        Args:
            other: The other state to be scaled and accumulated into this state.
            factors: Scalar factor(s) used in scaling `other`. If a single number is provided,
                scale all batched states in `other` by the same factor. Defaults to 1.
        """
        self._check_state_compatibility(other)
        factors_arr = self._check_and_return_factors(factors)

        with cutn_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(self._ctx) as (
            self._last_compute_event,
            elapsed,
        ):
            # update last event in other
            other._last_compute_event = self._last_compute_event
            cudm.state_compute_accumulation(
                self._ctx._handle._validated_ptr,
                other._validated_ptr,
                self._ptr,
                factors_arr.data.ptr,
                self._ctx._stream_holder.ptr,
            )

    @cutn_utils.precondition(_check_valid_state)
    def inner_product(self, other) -> cp.ndarray:
        """
        Compute the inner product(s) between two states.

        Args:
            other: The other state to compute inner product with.

        Returns:
            An array of inner product(s) of length ``batch_size``.
        """
        self._check_state_compatibility(other)
        with cutn_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(self._ctx) as (
            self._last_compute_event,
            elapsed,
        ):
            # update last event in other
            other._last_compute_event = self._last_compute_event
            # update last event in participating elementary/general operators to ensure proper stream synchronization and shutdown order
            res = cp.empty(self.batch_size, dtype=self.dtype, order="F")
            cudm.state_compute_inner_product(
                self._ctx._handle._validated_ptr,
                self._ptr,
                other._validated_ptr,
                res.data.ptr,
                self._ctx._stream_holder.ptr,
            )
        return res

    @cutn_utils.precondition(_check_valid_state)
    def _attach_component_storage(self, data: Sequence) -> None:
        """
        Attaches GPU buffers to this instance. This instance doesn't own the buffers.
        All elements of data need to be on this instances device_id and Fortran ordered. No copy is created and the buffer is a reference to data argument.

        Args:
            data: Sequence
                Sequence of NDarray like objects containing the statevector coefficients.
                The length of the sequence should be identical to self._num_components.
                The elements of data need to be on device(options.device_id) and need to be F-ordered, otherwise exceptions are raised in the initialization.
        """
        bufs = [tensor_wrapper.wrap_operands((d,))[0] for d in data]
        try:
            num_components = len(data)
            if num_components != self._num_components:
                raise ValueError(
                    "Trying to attach component storage of incorrect length to this instance of state.",
                )
            for buf in bufs:
                if buf.dtype != self.dtype:
                    raise ValueError("Supplied buffer's dtype doesn't match instances dtype.")
                if buf.device != "cuda" or buf.device_id != self._ctx.device_id:
                    raise ValueError(
                        "State component storages needs to be provided as GPU residing ndarray-like objects located on same GPU as State instance.",
                    )
                if not buf.tensor.flags["F_CONTIGUOUS"]:
                    raise ValueError(
                        "State component storages need to be contiguous and F-ordered."
                    )
            expected_sizes = self._component_storage_size
            received_sizes = tuple(buf.tensor.dtype.itemsize * buf.tensor.size for buf in bufs)
            if not received_sizes == expected_sizes:
                raise ValueError(
                    f"The supplied storage sizes, {received_sizes}, do not match the expected storage sizes, {expected_sizes}.\
                Both storage sizes are reported in bytes."
                )
        except ValueError as e:
            raise e

        cudm.state_attach_component_storage(
            self._ctx._handle._validated_ptr,
            self._ptr,
            num_components,
            tuple(buf.data_ptr for buf in bufs),
            received_sizes,
        )
        self._bufs = bufs

    @property
    @cutn_utils.precondition(_check_valid_state)
    def _num_components(self):
        """
        Number of components in the state storage.
        """
        return cudm.state_get_num_components(self._ctx._handle._validated_ptr, self._ptr)

    @property
    @cutn_utils.precondition(_check_valid_state)
    def _component_storage_size(self):
        """
        Size of each of the components in the state storage in bytes.
        """
        sizes = cudm.state_get_component_storage_size(
            self._ctx._handle._validated_ptr, self._ptr, self._num_components
        )
        return sizes

    @property
    @cutn_utils.precondition(_check_valid_state)
    def _component_storage(self) -> Sequence[Tensor]:
        """
        Non-blocking return of reference to buffers.
        """
        return self._bufs

    @property
    @cutn_utils.precondition(_check_valid_state)
    def _local_info(self) -> List[Tuple[Tuple[int], Tuple[int]]]:
        infos = []
        for local_component_index in range(self._num_components):
            _, component_num_modes, _ = self._get_component_num_modes(local_component_index)
            component_mode_extents = np.zeros((component_num_modes,), dtype="int64")
            component_mode_offsets = np.zeros((component_num_modes,), dtype="int64")
            _global_component_index = np.zeros((1,), dtype="int32")
            _component_num_modes = np.zeros((1,), dtype="int32")

            cudm.state_get_component_info(
                self._ctx._handle._validated_ptr,
                self._validated_ptr,
                local_component_index,
                _global_component_index.ctypes.data,
                _component_num_modes.ctypes.data,
                component_mode_extents.ctypes.data,
                component_mode_offsets.ctypes.data,
            )
            component_mode_extents = tuple(component_mode_extents)
            component_mode_offsets = tuple(component_mode_offsets)
            if self.batch_size == 1:
                component_mode_extents = component_mode_extents + (1,)
                component_mode_offsets = component_mode_offsets + (0,)
            infos.append((component_mode_extents, component_mode_offsets))
        return infos

    def _get_component_num_modes(self, local_component_index: int):
        batch_mode_location = np.zeros((1,), dtype=np.int32)
        component_num_modes = np.zeros((1,), dtype=np.int32)
        global_component_index = np.zeros((1,), dtype=np.int32)
        cudm.state_get_component_num_modes(
            self._ctx._handle._validated_ptr,
            self._validated_ptr,
            local_component_index,
            global_component_index.ctypes.data,
            component_num_modes.ctypes.data,
            batch_mode_location.ctypes.data,
        )
        return global_component_index[0], component_num_modes[0], batch_mode_location[0]

    @abstractmethod
    def clone(self, bufs) -> "State":
        pass


class DenseState(State):
    """
    A state in dense representation.
    """

    @property
    def storage(self) -> cp.ndarray:
        """
        The state's local storage buffer.

        Returns:
            cp.ndarray:
                The state's local storage buffer.
        """
        data = self._component_storage
        if data is not None:
            return data[0].tensor

    @property
    def storage_size(self) -> int:
        """
        Storage buffer size in number of elements of data type `dtype`.

        Returns:
            int: Storage buffer size in number of elements of data type `dtype`.
        """
        return self._component_storage_size[0] // np.dtype(self.dtype).itemsize

    def view(self) -> cp.ndarray:
        """
        Return a multidimensional view on the local slice of the storage buffer.

        .. note::
            When ``batch_size`` is 1, the last mode of the view will be the batch mode of dimension 1.
        """
        shape, _ = self.local_info
        if self.storage.size == np.prod(shape) and len(self.storage.shape) > 1:
            view = self.storage.reshape(shape, order="F")

        else:
            view = self.storage[: np.prod(shape)].reshape(shape, order="F")
        assert view.base is self.storage
        return view

    @property
    def local_info(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Local storage buffer dimensions as well as local mode offsets.

        Returns:
            Tuple[int]
                Local storage buffer dimensions, with the last dimension being the batch dimension.
            Tuple[int]
                Local mode offsets.
        """
        dims, offsets = self._local_info[0]
        return dims, offsets

    def attach_storage(self, data: cp.ndarray) -> None:
        """
        Attach a data buffer to the state.

        Args:
            data: The data buffer to be attached to the state.

        .. note::
            The data buffer needs to match the hilbert space dimensions, batch size and data type
            passed to the ``__init__`` function. In addition, the data buffer needs to be Fortran
            contiguous and located on the same device as the :class:`WorkStream` passed to the ``__init__`` function.
        """
        self._attach_component_storage((data,))

    def allocate_storage(self) -> None:
        """
        Allocate an appropriately sized data buffer and attach it to the state.
        """
        with cp.cuda.Device(self._ctx.device_id):
            state_storage_buf = cp.zeros((self.storage_size,), dtype=self.dtype)
            self.attach_storage(state_storage_buf)

    def clone(self, buf: cp.ndarray) -> "DenseState":
        """Clone the state with a new data buffer.

        Args:
            buf: The data buffer to be attached to the new state.

        Returns:
            A state with same metadata as the original state and a new data buffer.
        """
        if buf.dtype != self.dtype:
            raise ValueError(
                f"The supplied data buffer's data type {buf.dtype} does not match the original "
                f"instances data type {self.dtype}."
            )
        new_instance = type(self)(self._ctx, self.hilbert_space_dims, self.batch_size, self.dtype)
        size = new_instance.storage_size
        if not buf.flags.f_contiguous:
            raise ValueError("The supplied data buffer is not Fortran ordered and contiguous.")
        if np.prod(buf.shape) != size:
            raise ValueError(
                f"The supplied data buffer size, {buf.size} does not match the expected size: {size}."
            )
        if len(buf.shape) > 1:
            # only applicable to multi-GPU usage, may break for MGMN with correctly sized buffers
            new_instance_shape, _ = new_instance.local_info
            squeezed_shape = new_instance_shape[:-1] if self.batch_size == 1 else new_instance_shape
            if not (buf.shape == new_instance_shape or buf.shape == squeezed_shape):
                raise ValueError(
                    f"The supplied data buffer shape is not compatible with the required local state slice size."
                    " Note that non-1D data buffers are only supported in single-GPU usage."
                )
        new_instance.attach_storage(buf)
        return new_instance


class DensePureState(DenseState):
    """
    DensePureState(ctx, hilbert_space_dims, batch_size, dtype)

    Pure state in dense (state-vector) representation.

    A storage buffer needs to be attached via the :meth:`attach_storage` method or allocated via the :meth:`allocate_storage` method. The appropriate size for the storage buffer as well as information on the storage layout is available in the :attr:`local_info` attribute.

    Args:
        ctx: The execution context, which contains information on device ID, logging and blocking/non-blocking execution.
        hilbert_space_dims: A tuple of the local Hilbert space dimensions.
        batch_size: Batch dimension of the state.
        dtype: Numeric data type of the state's coefficients.

    Examples:
        >>> import cupy as cp
        >>> from cuquantum.densitymat import WorkStream, DensePureState

        To create a ``DensePureState`` of batch size 1 and double-precision complex data type, we need to first initialize it and then attach the storage buffer through the :meth:`attach_storage` method as follows

        >>> ctx = WorkStream(stream=cp.cuda.Stream())
        >>> hilbert_space_dims = (2, 2, 2)
        >>> rho = DensePureState(ctx, hilbert_space_dims, 1, "complex128")
        >>> rho.attach_storage(cp.zeros(rho.storage_size, dtype=rho.dtype))
    """

    def __init__(
        self, ctx: WorkStream, hilbert_space_dims: Sequence[int], batch_size: int, dtype: str
    ) -> None:
        """
        Initialize a pure state in dense (state-vector) representation.
        """
        super().__init__(ctx, hilbert_space_dims, batch_size, dtype)
        self._instantiate(ctx)

    @property
    def _purity(self):
        return cudm.StatePurity.PURE


class DenseMixedState(DenseState):
    """
    DenseMixedState(ctx, hilbert_space_dims, batch_size, dtype)

    Mixed state in dense (density-matrix) representation.

    A storage buffer needs to be attached via the :meth:`attach_storage` method or allocated via the :meth:`allocate_storage` method. The appropriate size for the storage buffer as well as information on the storage layout is available in the :attr:`local_info` attribute.

    Args:
        ctx: The execution context, which contains information on device ID, logging and blocking/non-blocking execution.
        hilbert_space_dims: A tuple of the local Hilbert space dimensions.
        batch_size: Batch dimension of the state.
        dtype: Numeric data type of the state's coefficients.

    Examples:
        >>> import cupy as cp
        >>> from cuquantum.densitymat import WorkStream, DenseMixedState

        To create a ``DenseMixedState`` of batch size 1 and double-precision complex data type, we need to first initialize it and then attach the storage buffer through the :meth:`attach_storage` method as follows

        >>> ctx = WorkStream(stream=cp.cuda.Stream())
        >>> hilbert_space_dims = (2, 2, 2)
        >>> rho = DenseMixedState(ctx, hilbert_space_dims, 1, "complex128")
        >>> rho.attach_storage(cp.zeros(rho.storage_size, dtype=rho.dtype))
    """

    def __init__(
        self, ctx: WorkStream, hilbert_space_dims: Sequence[int], batch_size: int, dtype: str
    ) -> None:
        """
        Initialize a mixed state in dense (density-matrix) representation.
        """
        super().__init__(ctx, hilbert_space_dims, batch_size, dtype)
        self._instantiate(ctx)

    @property
    def _purity(self):
        return cudm.StatePurity.MIXED
