# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import sys

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

        for g, l in gateSeq:
            if g=='h': # l = [target qubit]
                circuit.append(cirq.H(qubits[l[0]]))

            elif g=='x': # l = [target qubit]
                circuit.append(cirq.X(qubits[l[0]]))

            elif g=='cnot': # l = [control qubit, target qubit]
                circuit.append(cirq.CNOT(qubits[l[0]], qubits[l[1]]))

            elif g=='cz': # l = [control qubit, target qubit]
                circuit.append(cirq.CZ(qubits[l[0]], qubits[l[1]]))

            elif g=='rz': # l = [angle, target qubit]
                circuit.append(cirq.rz(l[0]).on(qubits[l[1]]))

            elif g=='rx': # l = [angle, target qubit]
                circuit.append(cirq.rx(l[0]).on(qubits[l[1]]))

            elif g=='ry': # l = [angle, target qubit]
                circuit.append(cirq.ry(l[0]).on(qubits[l[1]]))

            elif g=='czpowgate': # l = [exponent, control qubit, target qubit]
                circuit.append(cirq.CZPowGate(exponent=l[0]).on(qubits[l[1]], qubits[l[2]]))

            elif g=='swap': # l = [first qubit, second qubit]
                circuit.append(cirq.SWAP(qubits[l[0]], qubits[l[1]]))

            elif g=='cu': # l = [U matrix, U name, control qubit, target qubit]
                U_gate = cirq.MatrixGate(l[0], name=l[1])
                circuit.append(U_gate.on(*[qubits[i] for i in l[3]]).controlled_by(qubits[l[2]]))

            elif g == 'u':  # l = [U matrix, target qubits]
                # TODO: give the gate a name?
                U_gate = cirq.MatrixGate(l[0])
                circuit.append(U_gate.on(*[qubits[i] for i in l[1]]))

            elif g=='measure': # l = [qubits ids]
                circuit.append(cirq.measure(*[qubits[i] for i in l[0]], key='result'))

            else:
                sys.exit("The gate type [" + g +"] is not defined")
        
        return circuit
