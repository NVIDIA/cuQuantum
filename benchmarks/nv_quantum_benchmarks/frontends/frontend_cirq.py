# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

try:
    import cirq
except ImportError:
    cirq = None

from .frontend import Frontend


class Cirq(Frontend):

    def __init__(self, nqubits, config):
        if cirq is None:
            raise RuntimeError('cirq is not installed')

        self.nqubits = nqubits
        self.config = config

    def generateCircuit(self, gateSeq):
        qubits = cirq.LineQubit.range(self.nqubits)
        circuit = cirq.Circuit()

        for g in gateSeq:
            if g.id == 'h':
                circuit.append(cirq.H(qubits[g.targets]))

            elif g.id == 'x':
                circuit.append(cirq.X(qubits[g.targets]))

            elif g.id == 'cnot':
                circuit.append(cirq.CNOT(qubits[g.controls], qubits[g.targets]))

            elif g.id == 'cz':
                circuit.append(cirq.CZ(qubits[g.controls], qubits[g.targets]))

            elif g.id == 'rz':
                circuit.append(cirq.rz(g.params).on(qubits[g.targets]))

            elif g.id == 'rx':
                circuit.append(cirq.rx(g.params).on(qubits[g.targets]))

            elif g.id == 'ry':
                circuit.append(cirq.ry(g.params).on(qubits[g.targets]))

            elif g.id == 'czpowgate':
                circuit.append(cirq.CZPowGate(exponent=g.params).on(qubits[g.controls], qubits[g.targets]))

            elif g.id == 'swap':
                assert len(g.targets) == 2
                circuit.append(cirq.SWAP(qubits[g.targets[0]], qubits[g.targets[1]]))

            elif g.id == 'cu':
                U_gate = cirq.MatrixGate(g.matrix, name=g.name)
                circuit.append(U_gate.on(*[qubits[i] for i in g.targets]).controlled_by(qubits[g.controls]))

            elif g.id == 'u':
                U_gate = cirq.MatrixGate(g.matrix, name=g.name)
                circuit.append(U_gate.on(*[qubits[i] for i in g.targets]))

            elif g.id == 'measure':
                circuit.append(cirq.measure(*[qubits[i] for i in g.targets], key='result'))

            else:
                raise NotImplementedError(f"The gate type {g.id} is not defined")
        
        return circuit
