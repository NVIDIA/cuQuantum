# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import collections.abc
import importlib
import importlib.metadata
import sys
import types
from dataclasses import dataclass

import numpy as np

from nvmath.internal.tensor_wrapper import infer_tensor_package
from .helpers import _get_backend_asarray_func, get_dtype_name
from ...bindings._utils import WHITESPACE_UNICODE


@dataclass
class GateEntry:
    """Describes a single gate or channel parsed from a quantum circuit.

    Attributes:
        kind: One of ``'gate'``, ``'general_channel'``, or ``'unitary_channel'``.
        operand: A single tensor for gates, or a list of tensors for channels.
        qubits: The qubits this entry acts on.
        is_diagonal: Whether the gate operand is diagonal (only for ``kind='gate'``).
        probabilities: Per-operand probabilities (only for ``kind='unitary_channel'``).
    """
    kind: str
    operand: object
    qubits: tuple
    is_diagonal: bool = False
    probabilities: tuple = None


EINSUM_SYMBOLS_BASE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
WHITESPACE_SYMBOLS_ID = None

CIRQ_MIN_VERSION = '0.6.0'
QISKIT_MIN_VERSION = '1.4.2'  # qiskit metapackage version

EMPTY_DICT = types.MappingProxyType({})


def check_version(package_name, version, minimum_version):
    """
    Check if the current version of a package is above the required minimum.
    """
    version_numbers = [int(i) for i in version.split('.')]
    minimum_version_numbers = [int(i) for i in minimum_version.split('.')]
    if version_numbers < minimum_version_numbers:
        raise NotImplementedError(f'CircuitToEinsum currently supports {package_name} above {minimum_version},'
                                  f'current version: {version}')
    return None


def _get_symbol(i):
    """
    Return a unicode as label for index. Whitespace unicode characters are skipped.

    This function can offer 1113955 (= sys.maxunicode - 140 - 16) unique symbols.
    """
    if i < 52:
        return EINSUM_SYMBOLS_BASE[i]

    global WHITESPACE_SYMBOLS_ID
    if WHITESPACE_SYMBOLS_ID is None:
        whitespace = WHITESPACE_UNICODE
        WHITESPACE_SYMBOLS_ID = np.asarray([ord(c) for c in whitespace], dtype=np.int32)
        WHITESPACE_SYMBOLS_ID = WHITESPACE_SYMBOLS_ID[WHITESPACE_SYMBOLS_ID >= 192]

    # leave "holes" in the integer -> unicode mapping to avoid using whitespaces as symbols
    i += 140
    offset = 0
    for hole in WHITESPACE_SYMBOLS_ID:  # loop size = 16
        if i + offset < hole:
            break
        offset += 1

    try:
        return chr(i + offset)
    except ValueError as e:
        raise ValueError(f"{i=} would exceed unicode limit") from e


def infer_parser(circuit):
    """
    Infer the package that defines the circuit object.
    """

    def import_package(name):
        try:
            return importlib.import_module(name)
        except ImportError:
            return None

    def find_package(get_package):
        qiskit = get_package("qiskit")
        if qiskit is not None and isinstance(circuit, qiskit.QuantumCircuit):
            return qiskit
        cirq = get_package("cirq")
        if cirq is not None and isinstance(circuit, cirq.Circuit):
            return cirq
        return None

    # Fast path: only check packages that are already imported. Covers common case and
    # means cirq users won't have to import qiskit. sys.modules could contain a partially
    # loaded package (no thread lock), so protect with AttributeError.
    try:
        package = find_package(sys.modules.get)
    except AttributeError:
        package = None
    # Fall back: import the packages, e.g. for re-exported or virtual-subclass types.
    # Catches partially imported cases due to import lock.
    if package is None:
        package = find_package(import_package)

    if package is None:
        base = circuit.__module__.split(".")[0]
        raise NotImplementedError(f"circuit from {base} not supported")

    if package.__name__ == "qiskit":
        qiskit_version = importlib.metadata.version("qiskit")  # qiskit metapackage version
        check_version("qiskit", qiskit_version, QISKIT_MIN_VERSION)
        from . import circuit_parser_utils_qiskit

        return circuit_parser_utils_qiskit
    if package.__name__ == "cirq":
        check_version("cirq", package.__version__, CIRQ_MIN_VERSION)
        from . import circuit_parser_utils_cirq

        return circuit_parser_utils_cirq
    raise AssertionError(f"no parser for circuit package {package.__name__}")

def parse_inputs(qubits, gate_entries, dtype, backend):
    """
    Given a sequence of qubits and gate entries, generate the mode labels,
    tensor operands and qubits_frontier map for the initial states and gate operations.
    """
    n_qubits = len(qubits)
    operands = get_bitstring_tensors('0'*n_qubits, backend, dtype)
    mode_labels, qubits_frontier, next_frontier = _init_mode_labels_from_qubits(qubits)
    gate_mode_labels, gate_operands = parse_gates_to_mode_labels_operands(
        gate_entries, qubits_frontier, next_frontier)
    mode_labels += gate_mode_labels
    operands += gate_operands
    return mode_labels, operands, qubits_frontier

def parse_bitstring(bitstring, n_qubits=None):
    """
    Parse the bitstring into standard form.
    """
    if n_qubits is not None:
        if len(bitstring) != n_qubits:
            raise ValueError(f'bitstring must be of the same length as number of qubits {n_qubits}')
    if not isinstance(bitstring, str):
        bitstring = ''.join(map(str, bitstring))
    if not set(bitstring).issubset(set('01')):
        raise ValueError('bitstring must be a sequence of 0/1')
    return bitstring

def parse_fixed_qubits(fixed):
    """
    Given a set of qubits with fixed states, return the output bitstring and corresponding qubits order.

    Fixed state values may be given as ints (``0``/``1``) or single-character strings
    (``'0'``/``'1'``); they are normalized to ``'0'``/``'1'`` so callers can index the
    basis maps consistently (matching :func:`parse_bitstring`).
    """
    if fixed:
        fixed_qubits, raw_values = zip(*fixed.items())
        fixed_bitstring = tuple(str(v) for v in raw_values)
        if not set(fixed_bitstring).issubset(set('01')):
            raise ValueError("fixed state values must be 0/1 (as int or str)")
    else:
        fixed_qubits, fixed_bitstring = (), ()
    return fixed_qubits, fixed_bitstring

def split_mixed_fixed(fixed):
    """Split a mixed-state ``fixed`` dict into independent ``(fixed_ket, fixed_bra)`` dicts.

    Each value of ``fixed`` selects the ket and bra index for the corresponding key and may be:

    * a scalar (``0``/``1`` as :class:`int` or ``'0'``/``'1'`` :class:`str`) -- shorthand for
      fixing the ket and bra to the same value (a diagonal/symmetric projection); or
    * a 2-tuple ``(ket, bra)`` where each entry is a scalar or ``None``. ``None`` leaves that side
      open (the key is omitted from the corresponding output dict), enabling independent ket/bra
      projections (e.g. ``(0, None)`` fixes only the ket).

    Keys absent from ``fixed`` are left open on both sides. Values are passed through unchanged so
    that callers can apply their own normalization/coercion.
    """
    fixed_ket, fixed_bra = {}, {}
    for key, value in fixed.items():
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(
                    f"a per-key (ket, bra) value must have length 2, got {value!r}")
            ket_value, bra_value = value
            if ket_value is not None:
                fixed_ket[key] = ket_value
            if bra_value is not None:
                fixed_bra[key] = bra_value
        else:
            fixed_ket[key] = fixed_bra[key] = value
    return fixed_ket, fixed_bra

def split_mixed_bitstring(bitstring, n_qubits):
    """Split a mixed-state amplitude ``bitstring`` argument into ``(ket, bra)`` specs.

    Accepts either:

    * a single length-``n_qubits`` bitstring -- interpreted symmetrically as ``ket == bra`` to
      extract a diagonal density-matrix element (a probability); or
    * a 2-tuple ``(ket_bitstring, bra_bitstring)`` of length-``n_qubits`` bitstrings to extract an
      off-diagonal element.

    The two forms are disambiguated by checking whether ``bitstring`` is a length-2 tuple whose
    *elements* are themselves length-``n_qubits`` sequences; otherwise the whole argument is treated
    as a single (symmetric) bitstring. The returned specs are passed through unchanged for the
    caller to normalize via :func:`parse_bitstring`.
    """
    if (isinstance(bitstring, tuple) and len(bitstring) == 2
            and all(isinstance(side, collections.abc.Sequence) and len(side) == n_qubits
                    for side in bitstring)):
        return bitstring[0], bitstring[1]
    return bitstring, bitstring

def _init_mode_labels_from_qubits(qubits):
    """
    Given a set of qubits, initialize the mode labels, tensor operands and index mapping for the input state.

    Returns mode labels, qubit-frontier map, and the next frontier.
    """
    from itertools import count
    n = len(qubits)
    return [[i] for i in range(n)], dict(zip(qubits, count())), n

def get_bitstring_tensors(bitstring, backend, dtype):
    """
    Create the tensors operands for a given bitstring state.

    Args:
        bitstring: A sequence of 0/1 specifing the product state.
        dtype: Data type for the tensor operands.
        backend: The package the tensor operands belong to.

    Returns:
        A list of tensor operands stored as `backend` array
    """
    package = importlib.import_module(backend)
    asarray = _get_backend_asarray_func(package)
    state_0 = asarray([1, 0], dtype=dtype)
    state_1 = asarray([0, 1], dtype=dtype)

    basis_map = {'0': state_0,
                 '1': state_1}

    operands = [basis_map[ibit] for ibit in bitstring]
    return operands

def convert_mode_labels_to_expression(input_mode_labels, output_mode_labels):
    """
    Create an Einsum expression from input and output index labels.

    Args:
        input_mode_labels: A sequence of mode labels for each input tensor.
        output_mode_labels: The desired mode labels for the output tensor.

    Returns:
        An Einsum expression in explicit form.
    """
    input_symbols = [''.join(map(_get_symbol, idx)) for idx in input_mode_labels]
    expression = ','.join(input_symbols) + '->' + ''.join(map(_get_symbol, output_mode_labels))
    return expression

def get_pauli_gates(pauli_map, backend, dtype):
    """
    Populate the gates for all pauli operators.

    Args:
        pauli_map: A dictionary mapping qubits to pauli operators.
        dtype: Data type for the tensor operands.
        backend: The package the tensor operands belong to.

    Returns:
        A sequence of pauli gates.
    """
    package = importlib.import_module(backend)
    asarray = _get_backend_asarray_func(package)
    pauli_i = asarray([[1,0], [0,1]], dtype=dtype)
    pauli_x = asarray([[0,1], [1,0]], dtype=dtype)
    pauli_z = asarray([[1,0], [0,-1]], dtype=dtype)

    operand_map = {'I': pauli_i,
                   'X': pauli_x,
                   'Z': pauli_z}
    if not get_dtype_name(dtype).startswith("float"):
        operand_map['Y'] = asarray([[0,-1j], [1j,0]], dtype=dtype)

    entries = []
    for qubit, pauli_char in pauli_map.items():
        operand = operand_map.get(pauli_char)
        if operand is None:
            raise ValueError('pauli string character must be one of I/X/Y/Z')
        entries.append(GateEntry(
            kind='gate', operand=operand, qubits=(qubit,),
            is_diagonal=(pauli_char in {'I', 'Z'})))
    return entries

def parse_gates_to_mode_labels_operands(gate_entries, qubits_frontier, next_frontier):
    """
    Populate the indices for all gate tensors (pure-state path only).

    Args:
        gate_entries: A list of :class:`GateEntry` objects.
        qubits_frontier: The map of the qubits to its current frontier index.
        next_frontier: The next index to use.

    Returns:
        Gate mode labels and gate operands.
    """
    mode_labels = []
    operands = []

    if not gate_entries:
        return mode_labels, operands

    first_operand = gate_entries[0].operand
    if isinstance(first_operand, list):
        first_operand = first_operand[0]
    package = infer_tensor_package(first_operand)
    module = importlib.import_module(package)

    def _get_diag(t):
        if t.ndim == 2:
            return t.diagonal()
        modes = [i for i in range(t.ndim // 2)]
        return module.einsum(t, modes*2, modes)

    for entry in gate_entries:
        if entry.kind != 'gate':
            raise RuntimeError(
                "Channel entries cannot be used in pure-state tensor network contraction. "
                "Use CircuitToEinsum.density_matrix()/reduced_density_matrix() for density-matrix "
                "quantities, or NetworkState.from_circuit/from_converter for trajectory simulation.")
        if entry.is_diagonal:
            modes = [qubits_frontier[q] for q in entry.qubits]
            operands.append(_get_diag(entry.operand))
            mode_labels.append(modes)
        else:
            operands.append(entry.operand)
            input_mode_labels = []
            output_mode_labels = []
            for q in entry.qubits:
                input_mode_labels.append(qubits_frontier[q])
                output_mode_labels.append(next_frontier)
                qubits_frontier[q] = next_frontier
                next_frontier += 1
            mode_labels.append(output_mode_labels+input_mode_labels)
    return mode_labels, operands


def stack_kraus_operators(entry, package, asarray, dtype):
    """
    Stack a channel entry's Kraus operators into a single tensor whose leading mode
    indexes the ``m`` Kraus operators (shape ``(m, AB...ab...)``, i.e. rank ``2k+1`` for a
    ``k``-qubit channel). Unitary channels are returned in Kraus form ``sqrt(p_k) * U_k`` so
    the stacked tensor fully describes the channel action ``sum_k K_k rho K_k^dagger``.

    Args:
        entry: A ``GateEntry`` with ``kind`` in ``('general_channel', 'unitary_channel')``.
        package: The already-imported backend module (e.g. ``numpy``).
        asarray: The backend ``asarray``-like function from ``_get_backend_asarray_func``.
        dtype: Data type for the stacked tensor.
    """
    if entry.kind == 'unitary_channel':
        kraus_ops = [np.sqrt(p) * op for p, op in zip(entry.probabilities, entry.operand, strict=True)]
    else:
        kraus_ops = entry.operand
    backend_name = package.__name__
    assert backend_name in ('numpy', 'cupy', 'torch'), \
        f"Unsupported backend '{backend_name}'; expected one of numpy, cupy, or torch."
    # The stacking axis is the second positional arg for all three backends (numpy/cupy `axis`,
    # torch `dim`); pass it positionally and stack natively so device tensors stay on device.
    k_stacked = package.stack(kraus_ops, 0)
    return asarray(k_stacked, dtype=dtype)


def _build_mixed_core(qubits, gate_entries, dtype, backend):
    """
    Build the core doubled tensor network (initial states + gates) for mixed-state
    computation.  Returns raw building blocks so callers can handle output modes,
    fixed projections, traced qubits, and Pauli insertions as needed.

    Returns:
        (mode_labels, operands, ket_frontier, bra_frontier, next_frontier)
    """
    package = importlib.import_module(backend)
    asarray = _get_backend_asarray_func(package)

    n_qubits = len(qubits)

    ket_initial = get_bitstring_tensors('0' * n_qubits, backend, dtype)
    bra_initial = get_bitstring_tensors('0' * n_qubits, backend, dtype)

    ket_frontier = {}
    bra_frontier = {}
    mode_labels = []
    operands = []

    for i, q in enumerate(qubits):
        ket_frontier[q] = i
        mode_labels.append([i])
        operands.append(ket_initial[i])
    for i, q in enumerate(qubits):
        bra_frontier[q] = n_qubits + i
        mode_labels.append([n_qubits + i])
        operands.append(bra_initial[i])

    next_frontier = 2 * n_qubits

    def _get_diag(t):
        if t.ndim == 2:
            return t.diagonal()
        modes = [i for i in range(t.ndim // 2)]
        return package.einsum(t, modes * 2, modes)

    for entry in gate_entries:
        if entry.kind in ('general_channel', 'unitary_channel'):
            k_stacked = stack_kraus_operators(entry, package, asarray, dtype)
            k_stacked_conj = k_stacked.conj()

            batch_mode = next_frontier
            next_frontier += 1

            ket_in_modes = [ket_frontier[q] for q in entry.qubits]
            ket_out_modes = []
            for q in entry.qubits:
                ket_out_modes.append(next_frontier)
                ket_frontier[q] = next_frontier
                next_frontier += 1
            mode_labels.append([batch_mode] + ket_out_modes + ket_in_modes)
            operands.append(k_stacked)

            bra_in_modes = [bra_frontier[q] for q in entry.qubits]
            bra_out_modes = []
            for q in entry.qubits:
                bra_out_modes.append(next_frontier)
                bra_frontier[q] = next_frontier
                next_frontier += 1
            mode_labels.append([batch_mode] + bra_out_modes + bra_in_modes)
            operands.append(k_stacked_conj)

        elif entry.is_diagonal:
            diag_ket = _get_diag(entry.operand)
            diag_bra = diag_ket.conj()
            ket_modes = [ket_frontier[q] for q in entry.qubits]
            bra_modes = [bra_frontier[q] for q in entry.qubits]
            mode_labels.append(ket_modes)
            operands.append(diag_ket)
            mode_labels.append(bra_modes)
            operands.append(diag_bra)

        else:
            ket_in = [ket_frontier[q] for q in entry.qubits]
            ket_out = []
            for q in entry.qubits:
                ket_out.append(next_frontier)
                ket_frontier[q] = next_frontier
                next_frontier += 1
            mode_labels.append(ket_out + ket_in)
            operands.append(entry.operand)

            bra_in = [bra_frontier[q] for q in entry.qubits]
            bra_out = []
            for q in entry.qubits:
                bra_out.append(next_frontier)
                bra_frontier[q] = next_frontier
                next_frontier += 1
            mode_labels.append(bra_out + bra_in)
            operands.append(entry.operand.conj())

    return mode_labels, operands, ket_frontier, bra_frontier, next_frontier


def build_mixed_tn(qubits, gate_entries, dtype, backend, where, fixed):
    """
    Build the doubled tensor network for mixed-state reduced density matrix
    using parallel ket+bra construction.

    Returns:
        (expression, operands): Einsum expression and list of tensor operands.
    """
    where_set = set(where)
    fixed_qubits, fixed_bitstring = parse_fixed_qubits(fixed)
    fixed_set = set(fixed_qubits)

    mode_labels, operands, ket_frontier, bra_frontier, next_frontier = \
        _build_mixed_core(qubits, gate_entries, dtype, backend)

    fixed_operands = get_bitstring_tensors(fixed_bitstring, backend, dtype)
    for i, q in enumerate(fixed_qubits):
        mode_labels.append([ket_frontier[q]])
        operands.append(fixed_operands[i])

        mode_labels.append([bra_frontier[q]])
        operands.append(fixed_operands[i])

    # Traced qubits: create hyperedges by remapping bra mode to ket mode
    traced_remap = {}
    for q in qubits:
        if q not in where_set and q not in fixed_set:
            traced_remap[bra_frontier[q]] = ket_frontier[q]
    if traced_remap:
        mode_labels = [
            [traced_remap.get(m, m) for m in ml]
            for ml in mode_labels
        ]

    output_mode_labels = []
    for q in where:
        output_mode_labels.append(ket_frontier[q])
    for q in where:
        output_mode_labels.append(traced_remap.get(bra_frontier[q], bra_frontier[q]))

    expression = convert_mode_labels_to_expression(mode_labels, output_mode_labels)
    return expression, operands


def build_mixed_dm_slice_tn(qubits, gate_entries, dtype, backend, fixed_ket, fixed_bra):
    """
    Build the doubled tensor network for a slice of the density matrix with independent
    ket and bra projections. No qubits are traced; modes that are not projected on either
    side appear as open output modes.

    Output mode order is ``(open_ket_modes_asc, open_bra_modes_asc)`` to match the
    underlying mixed-state Accessor C contract.

    Args:
        fixed_ket: dict mapping qubits to 0/1 to project on the ket side.
        fixed_bra: dict mapping qubits to 0/1 to project on the bra side.

    Returns:
        (expression, operands): Einsum expression and list of tensor operands.
    """
    fixed_ket_qubits, fixed_ket_bs = parse_fixed_qubits(fixed_ket)
    fixed_bra_qubits, fixed_bra_bs = parse_fixed_qubits(fixed_bra)

    mode_labels, operands, ket_frontier, bra_frontier, _ = \
        _build_mixed_core(qubits, gate_entries, dtype, backend)

    fixed_ket_operands = get_bitstring_tensors(fixed_ket_bs, backend, dtype)
    for i, q in enumerate(fixed_ket_qubits):
        mode_labels.append([ket_frontier[q]])
        operands.append(fixed_ket_operands[i])

    fixed_bra_operands = get_bitstring_tensors(fixed_bra_bs, backend, dtype)
    for i, q in enumerate(fixed_bra_qubits):
        mode_labels.append([bra_frontier[q]])
        operands.append(fixed_bra_operands[i])

    fixed_ket_set = set(fixed_ket_qubits)
    fixed_bra_set = set(fixed_bra_qubits)
    open_ket_modes = [ket_frontier[q] for q in qubits if q not in fixed_ket_set]
    open_bra_modes = [bra_frontier[q] for q in qubits if q not in fixed_bra_set]
    output_mode_labels = open_ket_modes + open_bra_modes

    expression = convert_mode_labels_to_expression(mode_labels, output_mode_labels)
    return expression, operands


def build_mixed_expectation_tn(qubits, gate_entries, dtype, backend, pauli_map):
    """
    Build the doubled tensor network for mixed-state expectation value of a Pauli string.

    For each Pauli qubit, inserts the Pauli matrix between ket and bra output modes.
    Diagonal Paulis (I, Z) are optimized: the diagonal is placed on the ket mode
    and the bra mode is unified (traced implicitly).
    Non-Pauli qubits are traced via hyperedges (shared ket/bra mode labels).

    Returns:
        (expression, operands): Einsum expression and list of tensor operands.
    """
    pauli_set = set(pauli_map.keys())

    mode_labels, operands, ket_frontier, bra_frontier, next_frontier = \
        _build_mixed_core(qubits, gate_entries, dtype, backend)

    pauli_entries = get_pauli_gates(pauli_map, backend, dtype)

    hyperedge_remap = {}

    for entry in pauli_entries:
        q = entry.qubits[0]
        if entry.is_diagonal:
            diag_vals = entry.operand.diagonal() if entry.operand.ndim == 2 else entry.operand
            mode_labels.append([ket_frontier[q]])
            operands.append(diag_vals)
            hyperedge_remap[bra_frontier[q]] = ket_frontier[q]
        else:
            # Tr(rho P) = sum_{i,j} rho_{ij} P_{ji}, where i is the open ket leg and j the
            # open bra leg. Labelling the operand axes (bra, ket) makes its element P[j, i]
            # = P_{ji}, which is the transpose required by the trace. Using (ket, bra) would
            # instead contract P_{ij} (i.e. compute Tr(rho P^T)) and flip the sign of <Y>.
            mode_labels.append([bra_frontier[q], ket_frontier[q]])
            operands.append(entry.operand)

    for q in qubits:
        if q not in pauli_set:
            hyperedge_remap[bra_frontier[q]] = ket_frontier[q]

    if hyperedge_remap:
        mode_labels = [
            [hyperedge_remap.get(m, m) for m in ml]
            for ml in mode_labels
        ]

    output_mode_labels = []
    expression = convert_mode_labels_to_expression(mode_labels, output_mode_labels)
    return expression, operands
