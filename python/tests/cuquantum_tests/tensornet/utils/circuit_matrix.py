# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = ['CircuitMatrix', 'CircuitMatrixABC', 'QiskitCircuitMatrix', 'CirqCircuitMatrix', 'get_qiskit_unitary_gate']

import numpy as np
try:
    import cirq
except ImportError:
    cirq = None
try:
    import qiskit
except ImportError:
    qiskit = None

from abc import ABC, abstractmethod


################################################
# functions to generate cirq.Circuit for testing
################################################

def get_cirq_qft_circuit(n_qubits):
    qubits = cirq.LineQubit.range(n_qubits)
    qreg = list(qubits)[::-1]
    operations = []
    while len(qreg) > 0:
        q_head = qreg.pop(0)
        operations.append(cirq.H(q_head))
        for i, qubit in enumerate(qreg):
            operations.append((cirq.CZ ** (1 / 2 ** (i + 1)))(qubit, q_head))
    circuit = cirq.Circuit(operations)
    return circuit


def get_cirq_random_circuit(n_qubits, n_moments, op_density=0.9, seed=3):
    return cirq.testing.random_circuit(n_qubits, n_moments, op_density, random_state=seed)

def get_real_cirq_circuit():
    qr = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(qr[0]))  # Pauli-X gate
    circuit.append(cirq.Z(qr[1]))  # Pauli-Z gate
    circuit.append(cirq.H(qr[0]))  # Hadamard gate
    circuit.append(cirq.CX(qr[0], qr[1]))  # CNOT gate
    return circuit
    


#########################################################
# functions to generate qiskit.QuantumCircuit for testing
#########################################################

def get_qiskit_unitary_gate(rng, control=None):
    # random unitary two qubit gate
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    random_unitary = random_unitary(4, seed=rng)
    gate = UnitaryGate(random_unitary)
    if control is None:
        return gate
    else:
        return gate.control(control)


def get_qiskit_qft_circuit(n_qubits):
    return qiskit.circuit.library.QFTGate(n_qubits).definition


def get_qiskit_random_circuit(n_qubits, depth, **kwargs):
    from qiskit.circuit.random import random_circuit
    circuit = random_circuit(n_qubits, depth, max_operands=3, **kwargs)
    return circuit


def get_qiskit_composite_circuit():
    sub_q = qiskit.QuantumRegister(2)
    sub_circ = qiskit.QuantumCircuit(sub_q, name='sub_circ')
    sub_circ.h(sub_q[0])
    sub_circ.crz(1, sub_q[0], sub_q[1])
    sub_circ.barrier()
    sub_circ.id(sub_q[1])
    sub_circ.u(1, 2, -2, sub_q[0])

    # Convert to a gate and stick it into an arbitrary place in the bigger circuit
    sub_inst = sub_circ.to_instruction()

    qr = qiskit.QuantumRegister(3, 'q')
    circ = qiskit.QuantumCircuit(qr)
    circ.h(qr[0])
    circ.cx(qr[0], qr[1])
    circ.cx(qr[1], qr[2])
    circ.append(sub_inst, [qr[1], qr[2]])
    circ.append(sub_inst, [qr[0], qr[2]])
    circ.append(sub_inst, [qr[0], qr[1]])
    return circ


def get_qiskit_nested_circuit():
    qr = qiskit.QuantumRegister(6, 'q')
    circ = qiskit.QuantumCircuit(qr)
    sub_ins = get_qiskit_composite_circuit().to_instruction()
    circ.append(sub_ins, [qr[0], qr[2], qr[4]])
    circ.append(sub_ins, [qr[1], qr[3], qr[5]])
    circ.cx(qr[0], qr[3])
    circ.cx(qr[1], qr[4])
    circ.cx(qr[2], qr[5])
    return circ


def get_qiskit_multi_control_circuit():
    qubits = qiskit.QuantumRegister(5)
    circuit = qiskit.QuantumCircuit(qubits)
    for q in qubits:
        circuit.h(q)
    qs = list(qubits)
    # 3 layers of multi-controlled qubits
    rng = np.random.default_rng(1234)
    for i in range(2):
        rng.shuffle(qs)
        ccu_gate = get_qiskit_unitary_gate(rng, control=2)
        circuit.append(ccu_gate, qs[:4])
        for q in qubits:
            if i % 2 == 1:
                circuit.h(q)
            else:
                circuit.x(q)
    circuit.global_phase = 0.5
    circuit.p(0.1, qubits[0])
    return circuit

def get_real_qiskit_circuit():
    qr = qiskit.QuantumRegister(2)
    circuit = qiskit.QuantumCircuit(qr)
    # Add gates with real matrices
    circuit.x(qr[0])        # Pauli-X gate: [[0, 1], [1, 0]]
    circuit.z(qr[1])        # Pauli-Z gate: [[1, 0], [0, -1]]
    circuit.h(qr[0])        # Hadamard gate: (1/√2)[[1, 1], [1, -1]]
    circuit.cx(qr[0], qr[1]) # CNOT gate: controlled-X
    return circuit

class CircuitMatrixABC(ABC):

    @staticmethod
    @abstractmethod
    def L0(): # For basic functionality tests
        """Return basic circuits for testing."""
        pass

    @staticmethod
    @abstractmethod
    def L1(): 
        """Return intermediate complexity circuits for testing."""
        pass
    
    @staticmethod
    @abstractmethod
    def L2(): # L2 is a larger set of circuits including extended problem sizes
        """Return complex circuits for extended testing."""
        pass

    @staticmethod
    @abstractmethod
    def realL0():
        """Return real circuits for testing."""
        pass

    @staticmethod
    @abstractmethod
    def complexL0():
        """Return complex circuits for testing."""
        pass

qiskit_L0_circuits = []
qiskit_L1_circuits = []
qiskit_L2_circuits = []

if qiskit:
    qiskit_L0_circuits.append(get_qiskit_qft_circuit(3))

    qiskit_L1_circuits.append(get_qiskit_composite_circuit())
    qiskit_L1_circuits.append(get_qiskit_qft_circuit(4))
    qiskit_L1_circuits.append(get_qiskit_random_circuit(5, 4, seed=2))

    circuit = get_qiskit_composite_circuit()
    circuit.global_phase = 0.5
    qiskit_L2_circuits.append(circuit)
    qiskit_L2_circuits.append(get_qiskit_nested_circuit())
    qiskit_L2_circuits.append(get_qiskit_multi_control_circuit())
    qiskit_L2_circuits.append(get_qiskit_qft_circuit(7))
    qiskit_L2_circuits.append(get_qiskit_random_circuit(6, 5, seed=23))
    qiskit_L2_circuits.append(get_qiskit_random_circuit(6, 4, seed=0))
    

class QiskitCircuitMatrix(CircuitMatrixABC):

    @staticmethod
    def L0():
        return qiskit_L0_circuits

    @staticmethod
    def L1():
        return qiskit_L1_circuits

    @staticmethod
    def L2():
        return qiskit_L2_circuits
    
    @staticmethod
    def realL0():
        if qiskit is None:
            return []
        return [get_real_qiskit_circuit()]

    @staticmethod
    def complexL0():
        if qiskit is None:
            return []
        return [get_qiskit_qft_circuit(3)]

cirq_L0_circuits = []
cirq_L1_circuits = []
cirq_L2_circuits = []

if cirq:
    cirq_L0_circuits.append(get_cirq_qft_circuit(3))
    cirq_L1_circuits.append(get_cirq_qft_circuit(4))
    cirq_L1_circuits.append(get_cirq_random_circuit(5, 4, seed=2))

    cirq_L2_circuits.append(get_cirq_qft_circuit(7))
    cirq_L2_circuits.append(get_cirq_random_circuit(6, 5, seed=23))
    cirq_L2_circuits.append(get_cirq_random_circuit(6, 4, seed=0))

class CirqCircuitMatrix(CircuitMatrixABC):

    @staticmethod
    def L0():
        return cirq_L0_circuits

    @staticmethod
    def L1():
        return cirq_L1_circuits
    
    @staticmethod
    def L2():
        return cirq_L2_circuits
    
    @staticmethod
    def realL0():
        if cirq is None:
            return []
        return [get_real_cirq_circuit()]

    @staticmethod
    def complexL0():
        if cirq is None:
            return []
        return [get_cirq_qft_circuit(3)]

class CircuitMatrix(CircuitMatrixABC):

    @staticmethod
    def L0():
        return CirqCircuitMatrix.L0() + QiskitCircuitMatrix.L0()

    @staticmethod
    def L1():
        return CirqCircuitMatrix.L1() + QiskitCircuitMatrix.L1()

    @staticmethod
    def L2():
        return CirqCircuitMatrix.L2() + QiskitCircuitMatrix.L2()
    
    @staticmethod
    def realL0():
        return CirqCircuitMatrix.realL0() + QiskitCircuitMatrix.realL0()

    @staticmethod
    def complexL0():
        return CirqCircuitMatrix.complexL0() + QiskitCircuitMatrix.complexL0()