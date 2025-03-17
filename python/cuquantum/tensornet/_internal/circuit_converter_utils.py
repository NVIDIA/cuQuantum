# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import types

try:
    import cirq
    from . import circuit_parser_utils_cirq
except ImportError:
    cirq = circuit_parser_utils_cirq = None
import cupy as cp
import numpy as np
try:
    import qiskit
    from . import circuit_parser_utils_qiskit
except ImportError:
    qiskit = circuit_parser_utils_qiskit = None

from ..._internal.tensor_wrapper import _get_backend_asarray_func
from ...bindings._utils import WHITESPACE_UNICODE


EINSUM_SYMBOLS_BASE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
WHITESPACE_SYMBOLS_ID = None

CIRQ_MIN_VERSION = '0.6.0'
QISKIT_MIN_VERSION = '0.24.0'  # qiskit metapackage version

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
    if qiskit and isinstance(circuit, qiskit.QuantumCircuit):
        import importlib.metadata
        qiskit_version = importlib.metadata.version('qiskit') # qiskit metapackage version
        check_version('qiskit', qiskit_version, QISKIT_MIN_VERSION)
        return circuit_parser_utils_qiskit
    elif cirq and isinstance(circuit, cirq.Circuit):
        cirq_version  = cirq.__version__
        check_version('cirq', cirq_version, CIRQ_MIN_VERSION)
        return circuit_parser_utils_cirq
    else:
        base = circuit.__module__.split('.')[0]
        raise NotImplementedError(f'circuit from {base} not supported')

def parse_inputs(qubits, gates, dtype, backend):
    """
    Given a sequence of qubits and gates, generate the mode labels, 
    tensor operands and qubits_frontier map for the initial states and gate operations.
    """
    n_qubits = len(qubits)
    operands = get_bitstring_tensors('0'*n_qubits, dtype, backend=backend)
    mode_labels, qubits_frontier, next_frontier = _init_mode_labels_from_qubits(qubits)
    gate_mode_labels, gate_operands = parse_gates_to_mode_labels_operands(gates, 
                                                                          qubits_frontier, 
                                                                          next_frontier)
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
    """
    if fixed:
        fixed_qubits, fixed_bitstring = zip(*fixed.items())
    else:
        fixed_qubits, fixed_bitstring = (), ()
    return fixed_qubits, fixed_bitstring

def _init_mode_labels_from_qubits(qubits):
    """
    Given a set of qubits, initialize the mode labels, tensor operands and index mapping for the input state.

    Returns mode labels, qubit-frontier map, and the next frontier.
    """
    from itertools import count
    n = len(qubits)
    return [[i] for i in range(n)], dict(zip(qubits, count())), n

def get_bitstring_tensors(bitstring, dtype='complex128', backend=cp):
    """
    Create the tensors operands for a given bitstring state.

    Args:
        bitstring: A sequence of 0/1 specifing the product state.
        dtype: Data type for the tensor operands.
        backend: The package the tensor operands belong to.

    Returns:
        A list of tensor operands stored as `backend` array
    """
    asarray = _get_backend_asarray_func(backend)
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

def get_pauli_gates(pauli_map, dtype='complex128', backend=cp):
    """
    Populate the gates for all pauli operators.

    Args:
        pauli_map: A dictionary mapping qubits to pauli operators. 
        dtype: Data type for the tensor operands.
        backend: The package the tensor operands belong to.

    Returns:
        A sequence of pauli gates.
    """
    asarray = _get_backend_asarray_func(backend)
    pauli_i = asarray([[1,0], [0,1]], dtype=dtype)
    pauli_x = asarray([[0,1], [1,0]], dtype=dtype)
    pauli_y = asarray([[0,-1j], [1j,0]], dtype=dtype)
    pauli_z = asarray([[1,0], [0,-1]], dtype=dtype)
    
    operand_map = {'I': pauli_i,
                   'X': pauli_x,
                   'Y': pauli_y,
                   'Z': pauli_z}
    gates = []
    for qubit, pauli_char in pauli_map.items():
        operand = operand_map.get(pauli_char)
        if operand is None:
            raise ValueError('pauli string character must be one of I/X/Y/Z')
        gates.append((operand, (qubit,)))
    return gates

def parse_gates_to_mode_labels_operands(
    gates, 
    qubits_frontier, 
    next_frontier
):
    """
    Populate the indices for all gate tensors

    Args:
        gates: An list of gate tensors and the corresponding qubits.
        qubits_frontier: The map of the qubits to its current frontier index.
        next_frontier: The next index to use. 

    Returns:
        Gate mode labels and gate operands.
    """
    mode_labels = []
    operands = []

    for tensor, gate_qubits in gates:
        operands.append(tensor)
        input_mode_labels = []
        output_mode_labels = []
        for q in gate_qubits:
            input_mode_labels.append(qubits_frontier[q])
            output_mode_labels.append(next_frontier)
            qubits_frontier[q] = next_frontier
            next_frontier += 1
        mode_labels.append(output_mode_labels+input_mode_labels)
    return mode_labels, operands
