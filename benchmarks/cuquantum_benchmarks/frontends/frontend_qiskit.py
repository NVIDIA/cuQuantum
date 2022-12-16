# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from math import pi
import sys

try:
    import qiskit
    from qiskit.extensions import UnitaryGate
except ImportError:
    qiskit = UnitaryGate = None

from .frontend import Frontend


class Qiskit(Frontend):

    def __init__(self, nqubits, config):
        if qiskit is None:
            raise RuntimeError("qiskit is not installed")

        self.nqubits = nqubits
        self.config = config

    def generateCircuit(self, gateSeq):
        last_g, last_l = gateSeq[-1]
        assert last_g == "measure"  # TODO: relax this?
        circuit = qiskit.QuantumCircuit(self.nqubits, len(last_l[0]))

        for g, l in gateSeq:
            if g=='h': # l = [target qubit]
                circuit.h(l[0])

            elif g=='x': # l = [target qubit]
                circuit.x(l[0])

            elif g=='cnot': # l = [control qubit, target qubit]
                circuit.cnot(l[0], l[1])

            elif g=='cz': # l = [control qubit, target qubit]
                circuit.cz(l[0], l[1])

            elif g=='rz': # l = [angle, target qubit]
                circuit.rz(l[0], l[1])

            elif g=='rx': # l = [angle, target qubit]
                circuit.rx(l[0], l[1])

            elif g=='ry': # l = [angle, target qubit]
                circuit.ry(l[0], l[1])

            elif g=='czpowgate': # l = [exponent, control qubit, target qubit]
                circuit.cp(pi*l[0], l[1], l[2])

            elif g=='swap': # l = [first qubit, second qubit]
                circuit.swap(l[0], l[1])

            elif g=='cu': # l = [U matrix, U name, control qubit, target qubit]
                U_gate = UnitaryGate(l[0], l[1]).control(1)
                circuit.append(U_gate, [l[2]]+l[3])

            elif g == 'u':  # l = [U matrix, target qubits]
                # TODO: give the gate a name?
                U_gate = UnitaryGate(l[0])
                circuit.append(U_gate, l[1])

            elif g=='measure': # l = [qubits ids]
                circuit.measure(l[0], l[0])

            else:
                sys.exit("The gate type [" + g +"] is not defined")
        
        return circuit
