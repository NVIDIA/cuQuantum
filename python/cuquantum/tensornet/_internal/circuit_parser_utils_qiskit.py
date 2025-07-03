# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, ControlledGate, Delay, Gate, Measure
from qiskit.circuit.library import (
    UnitaryGate, ZGate, SGate, SdgGate, TGate, 
    TdgGate, PhaseGate, RZGate, CZGate, CCZGate, RZZGate,
    DiagonalGate, IGate, U1Gate
)

from qiskit.quantum_info import (
    Operator, 
    Choi,
    SuperOp,
    Kraus,
    Stinespring,
    Chi,
    PTM,
)

DIAGONAL_GATE_CLASSES = (
    ZGate, SGate, SdgGate, TGate, TdgGate,
    PhaseGate, RZGate, CZGate, CCZGate, RZZGate,
    DiagonalGate, IGate, U1Gate,
)

from ..._internal.tensor_wrapper import _get_backend_asarray_func

# https://docs.quantum.ibm.com/api/qiskit/quantum_info#channels
NOISY_CHANNEL_TYPES = (
    Operator, 
    Choi,
    SuperOp,
    Kraus,
    Stinespring,
    Chi,
    PTM,
)

def remove_measurements(circuit):
    """
    Return a circuit with final measurement operations removed.
    """
    circuit = circuit.copy()
    circuit.remove_final_measurements()
    for instruction in circuit.data:
        if isinstance(instruction.operation, Measure):
            raise ValueError('mid-circuit measurement not supported in tensor network simulation')
    return circuit

def get_inverse_circuit(circuit):
    """
    Return a circuit with all gate operations inversed.
    """
    return circuit.inverse()

def should_parse_operation_operand(operation, qubits, decompose_gates):
    """
    Return whether the input operation should be parsed as full operand or further decomposed.
    If the operation is a Gate instance and also satisifies at least one of the following conditions, directly parse the operand:
        1. is a customized unitary gate
        2. number of qubits no larger than two
        3. number of qubits larger than two but decompose_gate is set to False
    """
    return isinstance(operation, Gate) and (isinstance(operation, UnitaryGate) or len(qubits) <= 2 or not decompose_gates)

def is_diagonal_gate(inst):
    if isinstance(inst, ControlledGate):
        return is_diagonal_gate(inst.base_gate)
    return isinstance(inst, DIAGONAL_GATE_CLASSES)

def parse_gate_sequence(
    circuit, 
    *, 
    asarray=None, 
    dtype='complex128', 
    qubit_map=None, 
    gates=None, 
    global_phase=0, 
    decompose_gates=True, 
    check_diagonal=True, 
    gates_are_diagonal=None
):
    """
    Return the gate sequence for the given circuit.
    Args:
        circuit: A :class:`qiskit.QuantumCircuit` object. 
        asarray: An asarray function to convert the operand to the ndarray for a specific package.
            If is None, the original operation will be returned
        dtype: The dtype to convert the operation operand to
        qubit_map: A dictionary mapping the local operation qubits to the global qubits.
        gates: The current gate sequences
        global_phase: An additional global phase to add to.
        decompose_gate: Whether the operation should be decomposed when operand is being parsed into ndarrays.
    """
    if gates is None:
        gates = []
    if gates_are_diagonal is None:
        gates_are_diagonal = []
    global_phase += circuit.global_phase
    for instruction in circuit.data:
        operation = instruction.operation
        gate_qubits = instruction.qubits
        if qubit_map:
            gate_qubits = [qubit_map[q] for q in gate_qubits]
        if isinstance(operation, (Barrier, Delay)):
            # no physical meaning in tensor network simulation
            continue
        if isinstance(operation, NOISY_CHANNEL_TYPES):
            raise RuntimeError("CircuitToEinsum currently doesn't support qiskit Circuits with QuantumChannels")
        if asarray is None:
            gates.append((operation, gate_qubits))
            continue
        if should_parse_operation_operand(operation, gate_qubits, decompose_gates):
            tensor = Operator(operation).data.reshape((2,2)*len(gate_qubits))
            tensor = asarray(tensor, dtype=dtype)
            # in qiskit notation, qubits are labelled in the inverse order
            gates.append((tensor, gate_qubits[::-1]))
            gates_are_diagonal.append(check_diagonal and is_diagonal_gate(operation))
            continue
        # Instruction as composite gate
        if not isinstance(operation.definition, QuantumCircuit):            
            raise ValueError(f'operation type {type(operation)} not supported')
        # for composite gate, must provide a map from the sub circuit to the original circuit
        next_qubit_map = dict(zip(operation.definition.qubits, gate_qubits))
        gates, global_phase, gates_are_diagonal = parse_gate_sequence(
            operation.definition, 
            asarray=asarray, 
            dtype=dtype, 
            qubit_map=next_qubit_map, 
            gates=gates, 
            global_phase=global_phase,
            decompose_gates=decompose_gates,
            check_diagonal=check_diagonal,
            gates_are_diagonal=gates_are_diagonal,
        )
    return gates, global_phase, gates_are_diagonal

def unfold_circuit(circuit, *, dtype='complex128', backend=cp, decompose_gates=True, check_diagonal=True):
    """
    Unfold the circuit to obtain the qubits and all gate tensors. 

    Args:
        circuit: A :class:`qiskit.QuantumCircuit` object. All parameters in the circuit must be binded.
        dtype: Data type for the tensor operands.
        backend: The package the tensor operands belong to.
        decompose_gates: Whether to decompose composite gates down to at most two qubits.

    Returns:
        All qubits and gate operations from the input circuit
    """
    if circuit.parameters:
        raise ValueError(f"Input circuit contains following parameters: {circuit.parameters}. Must be fully parameterized")
    asarray = _get_backend_asarray_func(backend)
    qubits = circuit.qubits

    gates, global_phase, gates_are_diagonal = parse_gate_sequence(
        circuit, 
        asarray=asarray,
        dtype=dtype,
        global_phase=0, 
        decompose_gates=decompose_gates,
        check_diagonal=check_diagonal,
    )
    if global_phase != 0:
        phase = np.exp(1j*global_phase)
        phase_gate = asarray([[phase, 0], [0, phase]], dtype=dtype)
        gates = [(phase_gate, qubits[:1]), ] + gates
        gates_are_diagonal = [True] + gates_are_diagonal

    return qubits, gates, gates_are_diagonal

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
    # No need to explicitly decompose gates here
    gates, global_phase, _ = parse_gate_sequence(circuit, asarray=None, decompose_gates=False, check_diagonal=False)
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
    newqc.global_phase = global_phase
    return newqc
