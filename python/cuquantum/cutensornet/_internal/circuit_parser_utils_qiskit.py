# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, ControlledGate, Delay, Gate, Measure
from qiskit.extensions import UnitaryGate

from .tensor_wrapper import _get_backend_asarray_func

def remove_measurements(circuit):
    """
    Return a circuit with final measurement operations removed.
    """
    circuit = circuit.copy()
    circuit.remove_final_measurements()
    for operation, _, _ in circuit:
        if isinstance(operation, Measure):
            raise ValueError('mid-circuit measurement not supported in tensor network simulation')
    return circuit

def get_inverse_circuit(circuit):
    """
    Return a circuit with all gate operations inversed.
    """
    return circuit.inverse()

def get_decomposed_gates(circuit, qubit_map=None, gates=None, gate_process_func=None):
    """
    Return the gate sequence for the given circuit. Compound gates/instructions will be decomposed 
    to either standard gates or customized unitary gates.
    """
    if gates is None:
        gates = []
    for operation, gate_qubits, _ in circuit:
        if qubit_map:
            gate_qubits = [qubit_map[q] for q in gate_qubits]
        if isinstance(operation, Gate):
            if 'standard_gate' in str(type(operation)) or isinstance(operation, UnitaryGate):
                if callable(gate_process_func):
                    gates.append(gate_process_func(operation, gate_qubits))
                else:
                    gates.append((operation, gate_qubits))
                continue
        else:
            if isinstance(operation, (Barrier, Delay)):
                # no physical meaning in tensor network simulation
                continue
            elif not isinstance(operation.definition, QuantumCircuit):
                # Instruction as composite gate
                raise ValueError(f'operation type {type(operation)} not supported')
        # for composite gate, must provide a map from the sub circuit to the original circuit
        next_qubit_map = dict(zip(operation.definition.qubits, gate_qubits))
        gates = get_decomposed_gates(operation.definition, qubit_map=next_qubit_map, gates=gates, gate_process_func=gate_process_func)
    return gates

def unfold_circuit(circuit, dtype='complex128', backend=cp):
    """
    Unfold the circuit to obtain the qubits and all gate tensors. All :class:`qiskit.circuit.Gate` and 
    :class:`qiskit.circuit.Instruction` in the circuit will be decomposed into either standard gates or customized unitary gates.
    Barrier and delay operations will be discarded.

    Args:
        circuit: A :class:`qiskit.QuantumCircuit` object. All parameters in the circuit must be binded.
        dtype: Data type for the tensor operands.
        backend: The package the tensor operands belong to.

    Returns:
        All qubits and gate operations from the input circuit
    """
    asarray = _get_backend_asarray_func(backend)
    qubits = circuit.qubits
    
    def gate_process_func(operation, gate_qubits):
        tensor = operation.to_matrix().reshape((2,2)*len(gate_qubits))
        tensor = asarray(tensor, dtype=dtype)
        if isinstance(operation, ControlledGate):
            # in qiskit notation, qubit at high index is the target qubit
            gate_qubits = gate_qubits[::-1]
        return tensor, gate_qubits
    
    gates = get_decomposed_gates(circuit, gate_process_func=gate_process_func)

    return qubits, gates

def get_lightcone_circuit(circuit, coned_qubits):
    """
    Use unitary reversed lightcone cancellation technique to reduce the effective circuit size based on the qubits to be coned. 

    Args:
        circuit: A :class:`qiskit.QuantumCircuit` object. 
        coned_qubits: An iterable of qubits to be coned.

    Returns:
        A :class:`qiskit.QuantumCircuit` object that potentially contains less number of gates
    """
    coned_qubits = set(coned_qubits)
    gates = get_decomposed_gates(circuit)
    newqc = QuantumCircuit(circuit.qubits)
    ix = len(gates)
    tail_operations = []
    while len(coned_qubits) != circuit.num_qubits and ix>0:
        ix -= 1
        operation, gate_qubits = gates[ix]
        qubit_set = set(gate_qubits)
        if qubit_set & coned_qubits:
            tail_operations.append([operation, gate_qubits])
            coned_qubits |= qubit_set
    for operation, gate_qubits in gates[:ix] + tail_operations[::-1]:
        newqc.append(operation, gate_qubits)
    return newqc
