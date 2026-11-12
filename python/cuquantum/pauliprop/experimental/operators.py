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
from ._internal.utils import is_c_contiguous

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

    def _validate_against_num_qubits(self, num_qubits: int) -> None:
        """Validate this operator's qubit indices against the target system size.

        The C-API operator constructors are not given ``num_qubits`` and so cannot
        reject out-of-range qubit indices.  Subclasses that act on specific qubit
        indices override this to perform the check before the C-API call.  The
        default implementation is a no-op.
        """
        pass

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
        the cotangent buffer, and yields ``(ptr, grad_buf)``. A non-contiguous NumPy
        buffer is attached via a contiguous copy, whose accumulated gradients are
        copied back into it when the context exits without an exception.

        Operator cotangent buffers and backward differentiation do not presently
        support multi-process execution.

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
            # C accumulates the n gradients into consecutive elements from the attached pointer.
            attached = grad_buf
            if param_grads_out is not None:
                # Validate a user-provided gradient buffer against the required contract
                # before attaching it to the C API.
                if np.dtype(wrapped.dtype) != np.dtype(dtype):
                    raise ValueError(
                        f"param_grads_out has dtype {np.dtype(wrapped.dtype)}, expected "
                        f"{np.dtype(dtype)} (the expansion's coefficient dtype)."
                    )
                if wrapped.size != n:
                    raise ValueError(
                        f"param_grads_out must have {n} element(s) (num_differentiable_params), "
                        f"got {wrapped.size}."
                    )
                if isinstance(grad_buf, np.ndarray) and not grad_buf.flags.writeable:
                    raise ValueError("param_grads_out must be writeable.")
                # A strided NumPy buffer accumulates into a contiguous copy instead. A strided
                # device buffer would need stream-ordered copies, so it is rejected.
                if not is_c_contiguous(wrapped.shape, wrapped.strides):
                    if not isinstance(grad_buf, np.ndarray):
                        raise ValueError(
                            "param_grads_out must be contiguous in row-major (C) order unless it is a NumPy array."
                        )
                    attached = np.ascontiguousarray(grad_buf)
                    wrapped = wrap_operand(attached)
            location = "DEVICE" if hasattr(attached, '__cuda_array_interface__') else "HOST"
            cupp.quantum_operator_attach_cotangent_buffer(
                int(library_handle), ptr, wrapped.data_ptr,
                wrapped.size * wrapped.itemsize,
                typemaps.NAME_TO_DATA_TYPE[wrapped.dtype],
                typemaps.MEM_SPACE_MAP[location])
            yield ptr, grad_buf
            if attached is not grad_buf:
                # C finishes accumulating into a host buffer before it returns, so the copy is final.
                grad_buf[...] = attached


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


def _normalize_qubit_indices(qubit_indices: Sequence[int]) -> list[int]:
    """Normalize an array-like of qubit indices to a list of plain ints.

    Rejects non-integer values (e.g. ``0.5``) rather than silently truncating them
    with ``int()`` -- a fractional index would otherwise be floored to the wrong
    qubit with no error.  ``bool`` is rejected as almost certainly a mistake.  numpy
    integer types are accepted and converted to plain ints.  Normalizing to a list
    also avoids ambiguous truth-value tests on array-like inputs downstream.
    """
    normalized: list[int] = []
    for i in qubit_indices:
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
            raise TypeError(f"qubit_indices must contain integers, got {i!r}.")
        normalized.append(int(i))
    return normalized


def _pauli_kinds(pauli_string: Sequence[str]) -> list[int]:
    """The C Pauli kinds of the (case-insensitive) letters of a Pauli string."""
    unknown = [p for p in pauli_string if p not in typemaps.PAULI_MAP]
    if unknown:
        raise ValueError(f"Unknown Pauli letter(s) {unknown} in pauli_string; valid letters are I, X, Y and Z.")
    return [typemaps.PAULI_MAP[p] for p in pauli_string]


# ---------------------------------------------------------------------------
# Concrete operator dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PauliNoiseChannel(_QuantumOperator):
    """A Pauli noise channel acting on 1 or 2 qubits.

    In multi-process execution, corresponding channels used by each process
    must have equivalent qubit indices and probabilities.

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
        # Normalize and validate the qubit indices up front (type / non-negativity /
        # uniqueness).
        self.qubit_indices = _normalize_qubit_indices(self.qubit_indices)
        if any(i < 0 for i in self.qubit_indices):
            raise ValueError(f"qubit_indices must be non-negative, got {list(self.qubit_indices)}.")
        if len(set(self.qubit_indices)) != len(self.qubit_indices):
            raise ValueError(f"qubit_indices must be unique, got {list(self.qubit_indices)}.")
        ref_paulis = self._SINGLE_QUBIT_PAULIS if self._num_qubits == 1 else self._TWO_QUBIT_PAULIS
        # Validate up front: the user mapping is consumed only via ``.get()`` below.
        #
        # Negative probabilities are intentionally allowed: PauliNoiseChannel
        # supports quasi-probabilities, so only non-finite (NaN / inf) values are
        # rejected here.
        valid_keys = set(ref_paulis)
        unknown = [k for k in self.noise_probabilities if k not in valid_keys]
        if unknown:
            raise ValueError(
                f"Unknown Pauli key(s) {sorted(unknown)} for a {self._num_qubits}-qubit "
                f"PauliNoiseChannel; valid keys are {sorted(valid_keys)}."
            )
        for pauli, prob in self.noise_probabilities.items():
            if isinstance(prob, bool) or not isinstance(prob, (int, float, np.integer, np.floating)):
                raise TypeError(f"noise_probabilities[{pauli!r}] must be a real number, got {prob!r}.")
            if not np.isfinite(prob):
                raise ValueError(
                    f"noise_probabilities[{pauli!r}]={prob} must be finite (NaN and inf are not allowed)."
                )
        # Convert input dict to tuple in reference Pauli order.
        self.noise_probabilities = tuple(
            self.noise_probabilities.get(pauli, 0.0) for pauli in ref_paulis
        )

    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        # The C call reads as many qubit indices and probabilities as the qubit count fixed at
        # construction implies, so reassigned fields must still match it.
        if len(self.qubit_indices) != self._num_qubits:
            raise ValueError(
                f"This PauliNoiseChannel acts on {self._num_qubits} qubit(s) but qubit_indices now has "
                f"{len(self.qubit_indices)}; create a new PauliNoiseChannel to change the number of qubits."
            )
        num_probabilities = 4 ** self._num_qubits
        if isinstance(self.noise_probabilities, Mapping) or len(self.noise_probabilities) != num_probabilities:
            raise ValueError(
                f"noise_probabilities must hold {num_probabilities} probabilities in noise_paulis order; "
                "a mapping is converted only at construction, so create a new PauliNoiseChannel to use one."
            )
        return cupp.create_pauli_noise_channel_operator, (self._num_qubits, self.qubit_indices, self.noise_probabilities)

    @property
    def noise_paulis(self) -> tuple[str, ...]:
        """The Pauli strings in reference order corresponding to each probability."""
        return self._SINGLE_QUBIT_PAULIS if self._num_qubits == 1 else self._TWO_QUBIT_PAULIS

    @property
    def num_differentiable_params(self) -> int:
        return 4 ** self._num_qubits

    def _validate_against_num_qubits(self, num_qubits: int) -> None:
        for i in self.qubit_indices:
            if i >= num_qubits:
                raise ValueError(
                    f"PauliNoiseChannel qubit index {i} is out of range for a "
                    f"{num_qubits}-qubit system (valid range [0, {num_qubits}))."
                )

    def __str__(self) -> str:
        nonzero = {p: prob for p, prob in zip(self.noise_paulis, self.noise_probabilities) if prob != 0.0}
        return f"PauliNoiseChannel(qubit indices={list(self.qubit_indices)}, noise probabilities={nonzero})"


@dataclass
class PauliRotationGate(_QuantumOperator):
    """A Pauli rotation gate ``exp(-i * angle/2 * P)`` where P is a Pauli string.

    In multi-process execution, corresponding gates used by each process must
    have equivalent angles, qubit indices, and Pauli strings.

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
        self.pauli_string = list(self.pauli_string)
        if len(self.pauli_string) == 0:
            raise ValueError("pauli_string must be non-empty")
        _pauli_kinds(self.pauli_string)
        # qubit_indices is None -> use the C-API default [0, 1, ..., num_qubits-1].
        # Otherwise normalize to a list of plain ints (avoids ambiguous truth-tests
        # on array-like inputs) and validate length/range/uniqueness up front.
        if self.qubit_indices is not None:
            self.qubit_indices = _normalize_qubit_indices(self.qubit_indices)
            if len(self.qubit_indices) != self.num_qubits:
                raise ValueError(
                    f"len(qubit_indices)={len(self.qubit_indices)} != num_qubits={self.num_qubits} "
                    f"(num_qubits is derived from len(pauli_string))."
                )
            if any(i < 0 for i in self.qubit_indices):
                raise ValueError(f"qubit_indices must be non-negative, got {self.qubit_indices}.")
            if len(set(self.qubit_indices)) != len(self.qubit_indices):
                raise ValueError(f"qubit_indices must be unique, got {self.qubit_indices}.")

    @property
    def num_qubits(self) -> int:
        """The number of qubits this gate acts on."""
        return len(self.pauli_string)

    @property
    def num_differentiable_params(self) -> int:
        return 1

    def _validate_against_num_qubits(self, num_qubits: int) -> None:
        # ``None`` is the C default ``[0, ..., len(pauli_string) - 1]``. Check that
        # range here; the C packer is not given the system size.
        indices = self.qubit_indices if self.qubit_indices is not None else range(self.num_qubits)
        for i in indices:
            if i >= num_qubits:
                raise ValueError(
                    f"PauliRotationGate qubit index {i} is out of range for a "
                    f"{num_qubits}-qubit system (valid range [0, {num_qubits}))."
                )

    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        # Built from the fields on every application, so reassigning them takes effect. The C call
        # reads num_qubits entries of qubit_indices, so a reassigned string must keep their count.
        if self.qubit_indices is not None and len(self.qubit_indices) != self.num_qubits:
            raise ValueError(
                f"len(qubit_indices)={len(self.qubit_indices)} != num_qubits={self.num_qubits} "
                f"(num_qubits is derived from len(pauli_string))."
            )
        return cupp.create_pauli_rotation_gate_operator, (
            self.angle,
            self.num_qubits,
            self.qubit_indices if self.qubit_indices is not None else 0,
            _pauli_kinds(self.pauli_string),
        )

    def __str__(self) -> str:
        qi = self.qubit_indices if self.qubit_indices is not None else list(range(self.num_qubits))
        return f"PauliRotationGate(angle={self.angle}, pauli string={list(self.pauli_string)}, qubit indices={qi})"


@dataclass
class CliffordGate(_QuantumOperator):
    """A Clifford gate (I, X, Y, Z, H, S, CX, CY, CZ, SWAP, iSWAP, SqrtX, SqrtY, SqrtZ).

    In multi-process execution, corresponding gates used by each process must
    have equivalent names and qubit indices.

    Attributes:
        name (str): The name of the Clifford gate (case-insensitive, must match one of
            :attr:`SUPPORTED_GATES`).
        qubit_indices (Sequence[int]): The qubit indices this gate acts on. For two-qubit
            Clifford gates, the qubit indices are specified with increasing significance, and
            so the control qubit (in gates such as ``CX``) is specified *after* the target
            qubit, i.e. ``qubit_indices = [target, control]``.
    """

    name: str
    qubit_indices: Sequence[int]

    SUPPORTED_GATES: ClassVar[frozenset[str]] = frozenset(typemaps.CLIFFORD_MAP.keys())
    # Clifford gates that act on two qubits (upper-case names); all others act on one.
    _TWO_QUBIT_GATES: ClassVar[frozenset[str]] = frozenset({"CX", "CY", "CZ", "SWAP", "ISWAP"})

    def __post_init__(self):
        self._arity()  # validates the name
        # Normalize to a list of plain ints (also avoids ambiguous truth-tests on
        # array-like inputs) and validate arity/range/uniqueness up front.
        self.qubit_indices = _normalize_qubit_indices(self.qubit_indices)
        self._check_arity()
        if any(i < 0 for i in self.qubit_indices):
            raise ValueError(f"qubit_indices must be non-negative, got {list(self.qubit_indices)}.")
        if len(set(self.qubit_indices)) != len(self.qubit_indices):
            raise ValueError(f"qubit_indices must be unique, got {list(self.qubit_indices)}.")

    def _arity(self) -> int:
        """The number of qubits the named gate acts on. Raises for an unsupported name."""
        name_upper = self.name.upper()
        if name_upper not in self.SUPPORTED_GATES:
            raise ValueError(
                f"Unsupported Clifford gate '{self.name}'. "
                f"Supported gates: {sorted(self.SUPPORTED_GATES)}"
            )
        return 2 if name_upper in self._TWO_QUBIT_GATES else 1

    def _check_arity(self) -> None:
        arity = self._arity()
        if len(self.qubit_indices) != arity:
            raise ValueError(
                f"Clifford gate '{self.name}' acts on {arity} qubit(s), but got "
                f"{len(self.qubit_indices)} qubit index/indices: {list(self.qubit_indices)}."
            )

    @property
    def num_differentiable_params(self) -> int:
        return 0

    def _validate_against_num_qubits(self, num_qubits: int) -> None:
        for i in self.qubit_indices:
            if i >= num_qubits:
                raise ValueError(
                    f"Clifford gate '{self.name}' qubit index {i} is out of range for a "
                    f"{num_qubits}-qubit system (valid range [0, {num_qubits}))."
                )

    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        # The C call reads one qubit index per qubit the named gate acts on, so a reassigned
        # name or qubit_indices must still agree.
        self._check_arity()
        return cupp.create_clifford_gate_operator, (typemaps.CLIFFORD_MAP[self.name], self.qubit_indices)

    def __str__(self) -> str:
        return f"CliffordGate(which_clifford='{self.name}', qubit_indices={list(self.qubit_indices)})"


@dataclass
class AmplitudeDampingChannel(_QuantumOperator):
    """An amplitude damping channel with damping and excitation probabilities.

    In multi-process execution, corresponding channels used by each process
    must have equivalent qubit indices and probabilities.

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
