# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import pytest

import numpy as np
try:
    import cirq
except ImportError:
    cirq = None
try:
    import qiskit
except ImportError:
    qiskit = None

from .test_utils import DEFAULT_RNG

cirq_circuits = []
qiskit_circuits = []


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


N_QUBITS_RANGE = range(5, 7)
N_MOMENTS_RANGE = DEPTH_RANGE = range(4, 6)

if cirq:
    for n_qubits in N_QUBITS_RANGE:
        cirq_circuits.append(get_cirq_qft_circuit(n_qubits))
        for n_moments in N_MOMENTS_RANGE:
            cirq_circuits.append(get_cirq_random_circuit(n_qubits, n_moments))

try:
    from cuquantum_benchmarks.frontends.frontend_cirq import Cirq as cuqnt_cirq
    from cuquantum_benchmarks.benchmarks import qpe, quantum_volume, qaoa
    cirq_generators = [qpe.QPE, quantum_volume.QuantumVolume, qaoa.QAOA]
    config = {'measure': True, 'unfold': True, 'p': 4}
    for generator in cirq_generators:
        for n_qubits in (5, 6):
            seq = generator.generateGatesSequence(n_qubits, config)
            circuit = cuqnt_cirq(n_qubits, config).generateCircuit(seq)
            cirq_circuits.append(circuit)
except:
    pass


#########################################################
# functions to generate qiskit.QuantumCircuit for testing
#########################################################

def get_qiskit_unitary_gate(rng=DEFAULT_RNG, control=None):
    # random unitary two qubit gate
    try:
        # qiskit 1.0
        from qiskit.circuit.library import UnitaryGate
    except ModuleNotFoundError:
        # qiskit < 1.0
        from qiskit.extensions import UnitaryGate
    from qiskit.quantum_info import random_unitary
    random_unitary = random_unitary(4, seed=rng)
    gate = UnitaryGate(random_unitary)
    if control is None:
        return gate
    else:
        return gate.control(control)


def get_qiskit_qft_circuit(n_qubits):
    return qiskit.circuit.library.QFT(n_qubits, do_swaps=False).decompose()


def get_qiskit_random_circuit(n_qubits, depth):
    from qiskit.circuit.random import random_circuit
    circuit = random_circuit(n_qubits, depth, max_operands=3)
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
    np.random.seed(0)
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


if qiskit:
    circuit = get_qiskit_composite_circuit()
    qiskit_circuits.append(circuit.copy())
    circuit.global_phase = 0.5
    qiskit_circuits.append(circuit)
    qiskit_circuits.append(get_qiskit_nested_circuit())
    qiskit_circuits.append(get_qiskit_multi_control_circuit())
    from qiskit.circuit.random import random_circuit
    qiskit_circuits.append(random_circuit(6, 4, seed=0)) # contains c3sx gate, not materialized in qiskit 0.44.0
    for n_qubits in N_QUBITS_RANGE:
        qiskit_circuits.append(get_qiskit_qft_circuit(n_qubits))
        for depth in DEPTH_RANGE:
            qiskit_circuits.append(get_qiskit_random_circuit(n_qubits, depth))

try:
    from cuquantum_benchmarks.frontends.frontend_qiskit import Qiskit as cuqnt_qiskit
    from cuquantum_benchmarks.benchmarks import qpe, quantum_volume, qaoa
    qiskit_generators = [qpe.QPE, quantum_volume.QuantumVolume, qaoa.QAOA]
    config = {'measure': True, 'unfold': True, 'p': 4}
    for generator in qiskit_generators:
        for n_qubits in (5, 6):
            seq = generator.generateGatesSequence(n_qubits, config)
            circuit = cuqnt_qiskit(n_qubits, config).generateCircuit(seq)
            qiskit_circuits.append(circuit)
except:
    pass

testing_circuits = cirq_circuits + qiskit_circuits

@pytest.fixture(scope="session")
def backend_cycle():
    return itertools.cycle(('numpy', 'cupy', 'torch'))

@pytest.fixture(scope="function")
def backend(backend_cycle):
    return next(backend_cycle)