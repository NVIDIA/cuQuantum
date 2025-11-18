# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pythonic API for cuStabilizer FrameSimulator."""

import logging
from logging import Logger
import numpy as np
from typing import Optional, Tuple, Union, Any, Literal, Sequence
from dataclasses import dataclass

try:
    import cupy as cp

    xp = cp
except ImportError:
    xp = np

from cuquantum.bindings import custabilizer as custab
import cuda.bindings.runtime as cudart
from nvmath import memory
from nvmath.internal import utils as nvmath_utils
from nvmath.internal.tensor_wrapper import maybe_register_package
from nvmath.memory import BaseCUDAMemoryManager, BaseCUDAMemoryManagerAsync

from nvmath.internal import tensor_wrapper

from .utils import (
    Array,
    Stream,
    _unpack_arrays,
    _pack_arrays,
    _get_memptr,
    _ptr_as_cupy,
)
from .pauli_table import PauliTable



@dataclass
class Options:
    """A data class for providing options to the Frame Simulator.

    This class follows the design pattern used in cuTensorNet's NetworkOptions
    for consistent user experience across the cuQuantum package.

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
    # Inputs and outputs
    operands_package: str
    operands_device_id: Union[int, Literal["cpu"]]
    # Internal state
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
        # Initialize the cuda context for first call.
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
                cudart.cudaMemset(ptr.device_ptr, 0, num_bytes)
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
        self.logger.debug("Options destructor called")
        if self._own_handle:
            custab.destroy(self.handle)
        for ptr in self._buffers:
            if isinstance(ptr, memory._UnmanagedMemoryPointer):
                self.logger.debug(
                    f"Freeing unmanaged memory pointer {ptr.device_ptr:x}"
                )
                ptr.free()
            else:
                continue


class Circuit:
    """Represents a quantum circuit for the frame simulator.

    This class wraps a circuit defined in Stim-compatible string format.
    The circuit owns the device buffer where the circuit data is stored.

    Args:
        circuit_string: Stim-compatible circuit string representation.
        stream: Optional CUDA stream.
        options: Optional Options object for configuration.

    Example:
        >>> circuit = Circuit("H 0\\nCNOT 0 1\\nM 0 1")
        >>> # Or with options
        >>> options = Options(device_id=0)
        >>> circuit = Circuit("H 0\\nCNOT 0 1", options=options)
    """

    _options: _ManagedOptions

    def __init__(
        self,
        circuit: Union[str, "stim.Circuit"], #noqa: F821
        stream: Stream = None,
        options: Optional[Options] = None,
    ):
        """Initialize a Circuit object.

        Args:
            circuit_string: Stim-compatible circuit string.
            stream: Optional CUDA stream identifier.
            options: Optional Options configuration.
        """
        if options is None:
            options = Options()
        self.circuit_string = str(circuit)

        # No input or output operands, internal package is "cuda"
        self._options = _ManagedOptions(options, "cuda")

        # Get required buffer size from the C API
        buffer_size = custab.circuit_size_from_string(self.handle, self.circuit_string)
        self._device_ptr = self._options.allocate_memory(buffer_size, stream)

        # Create circuit on device
        self._circuit = custab.create_circuit_from_string(
            self.handle,
            self.circuit_string,
            nvmath_utils.get_ptr_from_memory_pointer(self._device_ptr),
            buffer_size,
        )

        if self._circuit is None or self._circuit == 0:
            raise RuntimeError("Failed to create circuit from string")

        self._logger.debug(f"Created circuit with {buffer_size} bytes buffer")

    @property
    def _logger(self):
        return self._options.logger

    @property
    def circuit(self):
        """Return the underlying C circuit object."""
        return self._circuit

    @property
    def handle(self):
        """Return the underlying C handle object."""
        return self._options.handle

    def __del__(self):
        """Clean up circuit and device buffer."""
        self._options.logger.debug("Circuit destructor called")
        if hasattr(self, "_circuit") and self._circuit is not None:
            custab.destroy_circuit(self._circuit)


class FrameSimulator:
    """Simulates quantum circuits using the stabilizer frame formalism.

    This class simulates quantum circuits by tracking Pauli frame errors.
    It manages the X and Z bit tables, measurement table, and applies circuits.

    Args:
        num_qubits: Number of qubits in the simulation.
        num_paulis: Number of Pauli frame samples (shots).
        num_measurements: Number of measurements in the circuit (default: 0).
        randomize_measurements: Whether to randomize frame after measurements (default: True).
        x_table: Optional initial X bit table. If provided, memory is not owned by simulator.
        z_table: Optional initial Z bit table. If provided, memory is not owned by simulator.
        measurement_table: Optional initial measurement table.
        bit_packed: Whether the input tables are in bit-packed format.
        package: Package to use for the tables. "numpy" or "cupy". This has
                    lower priority than the package of the input tables, if any.
        random_seed: Random seed for the simulation (default: None).
        stream: Optional CUDA stream.
        options: Optional Options configuration.

    Attributes:
        num_qubits: Number of qubits.
        num_paulis: Number of Pauli samples.
        num_measurements: Number of measurements.

    Example:
        >>> circ = Circuit("H 0\\nCNOT 0 1\\nM 1")
        >>> sim = FrameSimulator(2, 1024, num_measurements=1)
        >>> sim.apply(circ)
        >>> measurements = sim.get_measurement_bits()
    """

    randomize_measurements: bool = True

    _options: _ManagedOptions
    _x_table_ptr: Union[memory.MemoryPointer, "cp.ndarray"]
    _z_table_ptr: Union[memory.MemoryPointer, "cp.ndarray"]
    _measurement_table_ptr: Union[memory.MemoryPointer, "cp.ndarray"]
    _rng: np.random.Generator

    def __init__(
        self,
        num_qubits: int,
        num_paulis: int,
        num_measurements: int = 0,
        num_detectors: int = 0,
        randomize_measurements: bool = True,
        x_table: Optional[Array] = None,
        z_table: Optional[Array] = None,
        measurement_table: Optional[Array] = None,
        bit_packed: bool = False,
        package: Literal["numpy", "cupy"] = "numpy",
        seed: Optional[int] = None,
        stream: Stream = None,
        options: Optional[Options] = None,
    ):
        """Initialize a FrameSimulator.

        If input bit tables are not provided, the simulator will allocate internal memory and own it.
        The simulator converts inputs and owns the converted tables if the input is

        1. a :py:class:`numpy.ndarray` and `bit_packed=False`
        2. a :py:class:`cupy.ndarray` and `bit_packed=False`

        If input is a :py:class:`cupy.ndarray` and `bit_packed=True`, the simulator will not allocate internal memory
        and will use the provided tables.

        Args:
            num_qubits: Number of qubits to simulate.
            num_paulis: Number of Pauli frame samples.
            num_measurements: Number of measurements to track.
            num_detectors: Number of detector instructions.
            randomize_measurements: Randomize frame after measurement gates.
            x_table: Pre-allocated X bit table.
            z_table: Pre-allocated Z bit table.
            measurement_table: Pre-allocated measurement table.
            bit_packed: Whether the input tables are in bit-packed format.
            package: Package to use for the tables, either `"numpy"` or `"cupy"`. This has
                    lower priority than the package of the input tables, if any.
            seed: Seed for a generator that will produce default seed for every
                  call of :py:meth:`apply`.
            stream: Optional CUDA stream.
            options: Optional Options configuration.
        """
        if options is None:
            options = Options()

        if x_table is not None and z_table is None:
            raise ValueError("If x_table is provided, z_table must also be provided.")
        if z_table is not None and x_table is None:
            raise ValueError("If z_table is provided, x_table must also be provided.")

        self.num_qubits = num_qubits
        self.num_paulis = num_paulis
        self.num_measurements = num_measurements
        self.randomize_measurements = randomize_measurements
        self._rng = np.random.default_rng(seed)

        # Calculate stride: must be multiple of 4 bytes (32 bits)
        self._table_stride_major = ((num_paulis + 31) // 32) * 4
        # else:
        self._options = _ManagedOptions(options, package)
        if x_table is None and z_table is None:
            # No inputs
            self._options.on_new_operands(package, "cpu")
            # allocate internal tables
            bit_bytes = (num_qubits * self._table_stride_major)
            self._options.logger.debug(
                f"Allocating X and Z tables with {num_qubits} qubits and {num_paulis} samples: {bit_bytes} bytes"
            )
            self._x_table_ptr = self._options.allocate_memory(
                bit_bytes, stream, reset=True
            )
            self._z_table_ptr = self._options.allocate_memory(
                bit_bytes, stream, reset=True
            )
            self._options.logger.debug(
                f"Allocated X({self._x_table_ptr.device_ptr:x}) and Z({self._z_table_ptr.device_ptr:x}) tables"
            )
        else:
            # Inputs provided
            self.set_input_tables(x=x_table, z=z_table, bit_packed=bit_packed)
        if measurement_table is None:
            mem_bytes = (num_measurements + num_detectors) * self._table_stride_major
            self._options.logger.debug(
                f"Allocating measurement table with {num_measurements} measurements and {num_paulis} samples: {mem_bytes} bytes"
            )
            self._measurement_table_ptr = self._options.allocate_memory(
                mem_bytes, stream
            )

        # Create frame simulator handle
        self._frame_simulator = custab.create_frame_simulator(
            self.handle,
            num_qubits,
            num_paulis,
            num_measurements,
            self._table_stride_major,
        )

        self._logger.debug(
            f"Created FrameSimulator: {num_qubits} qubits, "
            f"{num_paulis} samples, stride={self._table_stride_major}"
        )

    @property
    def _logger(self):
        return self._options.logger
    
    @property
    def operands_package(self):
        """
        The package last used inputs and of returned outputs.
        """
        return self._options.package
    
    @property
    def device_id(self):
        """
        The device for inputs and outputs.
        """
        return self._options.device_id

    @property
    def handle(self):
        """Return the underlying C handle object."""
        return self._options.handle

    def apply(
        self,
        circuit: Circuit,
        seed: Optional[int] = None,
        stream: Stream = None,
    ) -> None:
        """Apply a circuit to the Pauli frames.

        Args:
            circuit: Circuit object to apply.
            seed: Optional random seed for measurement randomization.
                        If provided, overrides the seed set during initialization.
            randomize_measurements: Optional boolean to randomize measurements.
                                   If provided, overrides the setting from initialization.

        Raises:
            ValueError: If circuit has more qubits than simulator.
        """
        # The C API allows circuits with fewer qubits
        # Circuit must have <= num_qubits and <= num_measurements
        if not isinstance(circuit, Circuit):
            raise ValueError(
                f"circuit argument must be custabilizer.Circuit, got {circuit.__class__}"
            )

        seed_ = seed if seed is not None else self._rng.integers(0, 2**31)
        stream_holder = self._options.get_or_create_stream(stream)

        blocking = True
        timing = bool(self._options.logger and self._options.logger.handlers)
        with nvmath_utils.cuda_call_ctx(stream_holder, blocking, timing) as (
            self.last_compute_event,
            elapsed,
        ):
            custab.frame_simulator_apply_circuit(
                self.handle,
                self._frame_simulator,
                circuit.circuit,
                self.randomize_measurements,
                seed_,
                _get_memptr(self._x_table_ptr),
                _get_memptr(self._z_table_ptr),
                _get_memptr(self._measurement_table_ptr),
                stream_holder.ptr,
            )
        if elapsed.data is not None:
            self._options.logger.info(f"The simulation took {elapsed.data:.3f} ms")

        self._options.logger.debug("Applied circuit to frame simulator")

    def get_pauli_table(self, bit_packed: bool = True) -> PauliTable:
        """Retrieve the X and Z Pauli tables.

        Args:
            bit_packed: If True, return as bit-packed arrays (default).
                        If False, unpack bits and return as (num_qubits, num_paulis) arrays.

        Returns:
            PauliTable object.


        The 
            If bit_packed=False, and `self.operands_package` is cupy, the
            returned PauliTable has a view into the simulator state. 
            That is, the contents of PauliTable can be indirectly changed by the
            simulator.
            In other cases, the returned PauliTable references a copy of the simulator state.

        Example:
            >>> sim = FrameSimulator(2, 1024)
            >>> pauli_table = sim.get_pauli_table(bit_packed=False)
            >>> pauli_table.num_qubits
            2
            >>> pauli_table.num_paulis
            1024
        """

        x, z = self.get_pauli_xz_bits(bit_packed=bit_packed)
        return PauliTable(
            x,
            z,
            num_qubits=self.num_qubits,
            num_paulis=self.num_paulis,
            bit_packed=bit_packed,
        )

    def get_pauli_xz_bits(self, bit_packed: bool = True) -> Tuple[Array, Array]:
        """
        Get the X and Z bits as raw arrays.
        Args:
            bit_packed: If True, return as bit-packed arrays (default).
                       If False, unpack bits and return as (num_qubits, num_paulis) arrays.

        Returns:
            Tuple of (x_bits, z_bits) as arrays.

        The package of arrays is determined by `self.operands_package`, which is
        set by the last call to set_input_tables or the package parameter to
        constructor.

        If bit_packed=False, and package is cupy, the returned arrays are views
        into the simulator state.
        In other cases, the returned arrays are copies of the simulator state.

        Example:
            >>> sim = FrameSimulator(2, 1024)
            >>> x_bits, z_bits = sim.get_pauli_xz_bits(bit_packed=False)
            >>> x_bits.shape
            (2, 1024)
            >>> z_bits.shape
            (2, 1024)
        """
        bit_shape = (self.num_qubits, self._table_stride_major)
        bit_size = bit_shape[0] * bit_shape[1] * 4
        x = _ptr_as_cupy(self._x_table_ptr, bit_size, shape=bit_shape, dtype="uint8")
        z = _ptr_as_cupy(self._z_table_ptr, bit_size, shape=bit_shape, dtype="uint8")
        self._logger.debug(
            f"Returning X({x.data.ptr if hasattr(x, 'data') else 'N/A'}) and Z({z.data.ptr if hasattr(z, 'data') else 'N/A'}) tables"
        )
        if not bit_packed:
            x, z = _unpack_arrays(self.num_paulis, x, z)
            x = x.reshape((self.num_qubits, self.num_paulis))
            z = z.reshape((self.num_qubits, self.num_paulis))
        if self._options.operands_package == "numpy":
            x = x.get()
            z = z.get()
        return x, z

    def get_measurement_bits(self, bit_packed: bool = True) -> Array:
        """Retrieve the measurement table.

        Args:
            bit_packed: If `True`, return as bit-packed array (default).
                       If `False`, unpack bits and return as (num_measurements, num_paulis) array.

        Returns:
            Measurement results array.

        If `bit_packed=False`, and :py:attr:`FrameSimulator.operands_package` tt
        :py:attr:`operands_package` is `"cupy"`, the returned array is a view
        into the simulator state.
        In other cases, the returned array is a copy of the simulator state.

        Example:
            >>> sim = FrameSimulator(2, 1024, num_measurements=1)
            >>> measurements = sim.get_measurement_bits(bit_packed=False)
            >>> measurements.shape
            (1, 1024)
        """
        num_bytes = self.num_measurements * self._table_stride_major
        shape = (self.num_measurements, self._table_stride_major)

        m = _ptr_as_cupy(
            self._measurement_table_ptr, num_bytes, shape=shape, dtype="uint8"
        )
        if self._options.operands_package == "numpy":
            m = m.get()
        if not bit_packed:
            (m,) = _unpack_arrays(self.num_paulis, m)
            m = m.reshape((self.num_measurements, self.num_paulis))
        return m

    def set_input_tables(
        self,
        x: Union[Array, None] = None,
        z: Union[Array, None] = None,
        m: Union[Array, None] = None,
        bit_packed: bool = True,
        stream: Stream = None,
    ) -> None:
        """Set the X and Z Pauli tables.

        Args:
            x_table: New X bit table.
            z_table: New Z bit table. Must be the same package and shape as x_table.
            m_table: New measurement table.
            bit_packed: If `True`, input is expected to be bit-packed (default).
                        If `False`, will pack bits automatically
            stream: Optional CUDA stream for the operation.

        The tables can be located either on CPU or device as specified by :py:attr:`device_id`.
        This call will update :py:attr:`operands_package`.
        The returned data from subsequent calls to :py:meth:`get_pauli_table`, :py:meth:`get_pauli_xz_bits`,
        and :py:meth:`get_measurement_bits` will be of the same type and device as the input tables.

        Example:
            >>> sim = FrameSimulator(2, 1024)
            >>> x_table = np.zeros(2*1024//8, dtype=np.uint8)
            >>> z_table = np.zeros(2*1024//8, dtype=np.uint8)
            >>> sim.set_pauli_table(x_table, z_table)
        """

        xz_len = self.num_paulis
        m_len = self.num_paulis

        if x is not None and z is not None:
            if x.__class__ != z.__class__:
                raise ValueError(f"X and Z tables must be same type, got {x.__class__} and {z.__class__}")
            if x.shape != z.shape:
                raise ValueError(f"X and Z tables must be same shape, got {x.shape} and {z.shape}")

        tables = (x, z, m)
        # outputs = [self._x_table_ptr, self._z_table_ptr, self._measurement_table_ptr]
        outputs = [
            getattr(self, f"_{table_name}_table_ptr", None)
            for table_name in ["x", "z", "m"]
        ]
        num_bits = (xz_len, xz_len, m_len)
        new_operands = tuple(t for t in tables if t is not None)
        if len(new_operands) == 0:
            operands_package = "numpy"
            operands_device_id = "cpu"
        else:
            tables_wrap = tensor_wrapper.wrap_operands(new_operands)
            operands_package = nvmath_utils.get_operands_package(tables_wrap)
            operands_device_id = nvmath_utils.get_operands_device_id(tables_wrap)
        self._options.on_new_operands(operands_package, operands_device_id)

        # Otherwise, process tables normally (copy/convert)
        for i, table, t_len in zip(range(len(tables)), tables, num_bits):
            if table is None:
                continue
            if isinstance(table, np.ndarray):
                stream_holder = self._options.get_or_create_stream(stream)
                self._options.logger.debug(
                    f"Converting input of size {table.nbytes} to device {self._options.operands_device_id} with package {self._options.package}"
                )
                with (
                    nvmath_utils.device_ctx(self._options.device_id),
                    stream_holder.ctx,
                ):
                    table = cp.asarray(table)
            if not bit_packed:
                (table,) = _pack_arrays(t_len, table)
            outputs[i] = table

        def update_table(table_name, ptr):
            def fmt_ptr(x):
                return _get_memptr(x) if x is not None else None
            if getattr(self, f"_{table_name}_table_ptr", None) is None:
                self._logger.debug(f"Setting {table_name} table to {fmt_ptr(ptr)}")
                setattr(self, f"_{table_name}_table_ptr", ptr)
            else:
                if ptr is not None:
                    self._logger.debug(
                        f"Updating {table_name} table from {fmt_ptr(getattr(self, f'_{table_name}_table_ptr'))} to {fmt_ptr(ptr)}"
                    )
                    setattr(self, f"_{table_name}_table_ptr", ptr)

        update_table("x", outputs[0])
        update_table("z", outputs[1])
        update_table("measurement", outputs[2])

    def __del__(self):
        """Clean up frame simulator."""
        if hasattr(self, "_frame_simulator") and self._frame_simulator is not None:
            custab.destroy_frame_simulator(self._frame_simulator)
