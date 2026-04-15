# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Sequence, TYPE_CHECKING

import numpy as np

import cuquantum.bindings.cupauliprop as cupp
from nvmath.internal.tensor_wrapper import wrap_operand
from ._internal import typemaps

if TYPE_CHECKING:
    from .handles import LibraryHandle

__all__ = ["QuantumOperator", "PauliNoiseChannel", "PauliRotationGate", "CliffordGate", "AmplitudeDampingChannel"]


class _QuantumOperator(ABC):
    """Abstract base class for quantum operators.

    Concrete subclasses are stateless dataclasses that describe the operator's
    parameters.  The C-API operator object is created and destroyed ephemerally
    via the :meth:`_as_c_operator` and :meth:`_as_c_operator_with_grad` context
    managers, which are used internally by :class:`PauliExpansionView` methods.
    """

    @abstractmethod
    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        """Return the C API create function and its arguments (excluding library handle).

        Returns:
            A tuple of (create_function, args) where create_function is called as
            ``create_function(library_handle, *args)``.
        """
        ...

    @property
    @abstractmethod
    def num_differentiable_params(self) -> int:
        """Number of differentiable parameters in this operator.

        For example, a :class:`PauliRotationGate` has 1 (the rotation angle),
        a :class:`CliffordGate` has 0.
        """
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Return a human-readable string representation of the operator."""
        ...

    # ------------------------------------------------------------------
    # Context managers for ephemeral C-API operator lifecycle
    # ------------------------------------------------------------------

    @contextmanager
    def _as_c_operator(self, library_handle: "LibraryHandle"):
        """Context manager that creates an ephemeral C-API operator and destroys it on exit.

        Yields:
            int: The C-API operator pointer.
        """
        create_func, args = self._get_create_args()
        ptr = create_func(int(library_handle), *args)
        try:
            yield ptr
        finally:
            cupp.destroy_operator(ptr)

    @contextmanager
    def _as_c_operator_with_grad(self, library_handle: "LibraryHandle", param_grads_out, dtype):
        """Context manager that creates an ephemeral C-API operator with a gradient buffer.

        If the operator has no differentiable parameters, yields ``(ptr, None)``.
        Otherwise allocates (or uses the provided) gradient buffer, attaches it as
        the cotangent buffer, and yields ``(ptr, grad_buf)``.

        Args:
            library_handle: The library handle.
            param_grads_out: A user-provided buffer for parameter gradients, or ``None``
                to auto-allocate a zeroed numpy array.
            dtype: Numpy dtype for the auto-allocated gradient buffer.

        Yields:
            tuple[int, numpy.ndarray | None]: The C-API operator pointer and the
            gradient buffer (or ``None`` for non-differentiable operators).
        """
        with self._as_c_operator(library_handle) as ptr:
            n = self.num_differentiable_params
            if n == 0:
                yield ptr, None
                return
            grad_buf = param_grads_out if param_grads_out is not None else np.zeros(n, dtype=dtype)
            wrapped = wrap_operand(grad_buf)
            location = "DEVICE" if hasattr(grad_buf, '__cuda_array_interface__') else "HOST"
            cupp.quantum_operator_attach_cotangent_buffer(
                int(library_handle), ptr, wrapped.data_ptr,
                wrapped.size * wrapped.itemsize,
                typemaps.NAME_TO_DATA_TYPE[wrapped.dtype],
                typemaps.MEM_SPACE_MAP[location])
            yield ptr, grad_buf


# ---------------------------------------------------------------------------
# Helper for PauliNoiseChannel reference ordering
# ---------------------------------------------------------------------------

def _build_noise_paulis(num_qubits: int) -> tuple[str, ...]:
    """Build the reference Pauli ordering from typemaps for consistency with bindings."""
    paulis = []
    for i in range(4 ** num_qubits):
        if num_qubits == 1:
            paulis.append(typemaps.PAULI_MAP_INV[i])
        else:  # num_qubits == 2
            paulis.append(f"{typemaps.PAULI_MAP_INV[i % 4]}{typemaps.PAULI_MAP_INV[i // 4]}")
    return tuple(paulis)


# ---------------------------------------------------------------------------
# Concrete operator dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PauliNoiseChannel(_QuantumOperator):
    """A Pauli noise channel acting on 1 or 2 qubits.

    Attributes:
        qubit_indices (Sequence[int]): The qubit indices the channel acts on (1 or 2 qubits).
        noise_probabilities (Mapping[str, float]): A dictionary mapping Pauli strings to their probabilities.
            Pauli strings not present in the dictionary are assumed to have zero probability.
            For single-qubit channels, valid keys are ``"I"``, ``"X"``, ``"Y"``, ``"Z"``.
            For two-qubit channels, valid keys are ``"II"``, ``"XI"``, ``"YI"``, ``"ZI"``, ``"IX"``, etc.
    """

    qubit_indices: Sequence[int]
    noise_probabilities: Mapping[str, float]

    # Reference Pauli orderings (class-level constants, not dataclass fields)
    _SINGLE_QUBIT_PAULIS: ClassVar[tuple[str, ...]] = _build_noise_paulis(1)
    _TWO_QUBIT_PAULIS: ClassVar[tuple[str, ...]] = _build_noise_paulis(2)

    def __post_init__(self):
        self._num_qubits: int = len(self.qubit_indices)
        if self._num_qubits not in (1, 2):
            raise ValueError(f"Number of qubits must be 1 or 2, got {self._num_qubits}")
        # Convert input dict to tuple in reference Pauli order
        ref_paulis = self._SINGLE_QUBIT_PAULIS if self._num_qubits == 1 else self._TWO_QUBIT_PAULIS
        self.noise_probabilities = tuple(
            self.noise_probabilities.get(pauli, 0.0) for pauli in ref_paulis
        )

    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        return cupp.create_pauli_noise_channel_operator, (self._num_qubits, self.qubit_indices, self.noise_probabilities)

    @property
    def noise_paulis(self) -> tuple[str, ...]:
        """The Pauli strings in reference order corresponding to each probability."""
        return self._SINGLE_QUBIT_PAULIS if self._num_qubits == 1 else self._TWO_QUBIT_PAULIS

    @property
    def num_differentiable_params(self) -> int:
        return 4 ** self._num_qubits

    def __str__(self) -> str:
        nonzero = {p: prob for p, prob in zip(self.noise_paulis, self.noise_probabilities) if prob != 0.0}
        return f"PauliNoiseChannel(qubit indices={list(self.qubit_indices)}, noise probabilities={nonzero})"


@dataclass
class PauliRotationGate(_QuantumOperator):
    """A Pauli rotation gate ``exp(-i * angle/2 * P)`` where P is a Pauli string.

    Attributes:
        angle (float): The rotation angle.
        pauli_string (str | Sequence[str]): The Pauli string defining the rotation axis, either as a
            single string (e.g. ``"XYZ"``) or a sequence of single-character
            Pauli labels (e.g. ``["X", "Y", "Z"]``).
        qubit_indices (Sequence[int] | None): The qubit indices this gate acts on.  If ``None``,
            defaults to ``[0, 1, ..., len(pauli_string)-1]``.
    """

    angle: float
    pauli_string: str | Sequence[str]
    qubit_indices: Sequence[int] | None = None

    def __post_init__(self):
        if isinstance(self.pauli_string, str):
            self.pauli_string = list(self.pauli_string)
        self._pauli_string_enums: list[int] = [typemaps.PAULI_MAP[p] for p in self.pauli_string]

    @property
    def num_qubits(self) -> int:
        """The number of qubits this gate acts on."""
        return len(self.pauli_string)

    @property
    def num_differentiable_params(self) -> int:
        return 1

    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        return cupp.create_pauli_rotation_gate_operator, (
            self.angle,
            self.num_qubits,
            self.qubit_indices if self.qubit_indices else 0,
            self._pauli_string_enums,
        )

    def __str__(self) -> str:
        qi = self.qubit_indices if self.qubit_indices is not None else list(range(self.num_qubits))
        return f"PauliRotationGate(angle={self.angle}, pauli string={list(self.pauli_string)}, qubit indices={qi})"


@dataclass
class CliffordGate(_QuantumOperator):
    """A Clifford gate (I, X, Y, Z, H, S, CX, CY, CZ, SWAP, iSWAP, SqrtX, SqrtY, SqrtZ).

    Attributes:
        name (str): The name of the Clifford gate (case-insensitive, must match one of
            :attr:`SUPPORTED_GATES`).
        qubit_indices (Sequence[int]): The qubit indices this gate acts on.
    """

    name: str
    qubit_indices: Sequence[int]

    SUPPORTED_GATES: ClassVar[frozenset[str]] = frozenset(typemaps.CLIFFORD_MAP.keys())

    def __post_init__(self):
        if self.name.upper() not in self.SUPPORTED_GATES:
            raise ValueError(
                f"Unsupported Clifford gate '{self.name}'. "
                f"Supported gates: {sorted(self.SUPPORTED_GATES)}"
            )

    @property
    def num_differentiable_params(self) -> int:
        return 0

    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        return cupp.create_clifford_gate_operator, (typemaps.CLIFFORD_MAP[self.name], self.qubit_indices)

    def __str__(self) -> str:
        return f"CliffordGate(which_clifford='{self.name}', qubit_indices={list(self.qubit_indices)})"


@dataclass
class AmplitudeDampingChannel(_QuantumOperator):
    """An amplitude damping channel with damping and excitation probabilities.

    Attributes:
        damping_probability (float): The damping probability.
        excitation_probability (float): The excitation probability.
        qubit_index (int): The qubit index this channel acts on.
    """

    damping_probability: float
    excitation_probability: float
    qubit_index: int

    @property
    def num_differentiable_params(self) -> int:
        return 2

    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        return cupp.create_amplitude_damping_channel_operator, (
            self.qubit_index, self.damping_probability, self.excitation_probability,
        )

    def __str__(self) -> str:
        return (
            f"AmplitudeDampingChannel(damping probability={self.damping_probability}, "
            f"excitation probability={self.excitation_probability}, qubit index={self.qubit_index})"
        )


QuantumOperator = PauliNoiseChannel | PauliRotationGate | CliffordGate | AmplitudeDampingChannel
