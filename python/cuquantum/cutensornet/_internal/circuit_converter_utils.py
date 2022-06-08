# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

try:
    import cirq
    from . import cirq_parser_utils
except ImportError:
    cirq = cirq_parser_utils = None
import cupy as cp
try:
    import qiskit
    from . import qiskit_parser_utils
except ImportError:
    qiskit = qiskit_parser_utils = None

from .tensor_wrapper import _get_backend_asarray_func

EINSUM_SYMBOLS_BASE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

CIRQ_MIN_VERSION = '0.6.0'
QISKIT_MIN_VERSION = '0.24.0' # qiskit metapackage version

import types
EMPTY_DICT = types.MappingProxyType({})

def check_version(package_name, version, minimum_version):
    """
    Check if the current version of a package is above the required minimum
    """
    version_numbers = [int(i) for i in version.split('.')]
    minimum_version_numbers = [int(i) for i in minimum_version.split('.')]
    if version_numbers < minimum_version_numbers:
        raise NotImplementedError(f'CircuitToEinsum currently supports {package_name} above {minimum_version},'
                                  f'current version: {version}')
    return None

def _get_symbol(i):
    """
    Return a Unicode as label for index.

    .. note:: This function is adopted from `opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/_modules/opt_einsum/parser.html#get_symbol>`_
    """
    if i < 52:
        return EINSUM_SYMBOLS_BASE[i]
    return chr(i + 140)

def infer_parser(circuit):
    """
    Infer the package that defines the circuit object.
    """
    if qiskit and isinstance(circuit, qiskit.QuantumCircuit):
        qiskit_version  = qiskit.__qiskit_version__['qiskit'] # qiskit metapackage version
        check_version('qiskit', qiskit_version, QISKIT_MIN_VERSION)
        return qiskit_parser_utils
    elif cirq and isinstance(circuit, cirq.Circuit):
        cirq_version  = cirq.__version__
        check_version('cirq', cirq_version, CIRQ_MIN_VERSION)
        return cirq_parser_utils
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
    Given a set of qubits with fixed states, return the output bitstring and corresponding qubits order
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
    Create an Einsum expression from input and output index labels

    Args:
        input_mode_labels: A sequence of mode labels for each input tensor.
        output_mode_labels: The desired mode labels for the output tensor.

    Returns:
        An Einsum expression in explicit form
    """    
    input_symbols = [''.join(map(_get_symbol, idx)) for idx in input_mode_labels]
    expression = ','.join(input_symbols) + '->' + ''.join(map(_get_symbol, output_mode_labels))
    return expression

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
