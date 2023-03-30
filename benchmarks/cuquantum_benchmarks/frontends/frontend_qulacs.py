# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from cmath import pi, exp

try:
    import qulacs
except ImportError:
    qulacs = None

from .frontend import Frontend


class Qulacs(Frontend):

    def __init__(self, nqubits, config):
        if qulacs is None:
            raise RuntimeError('qulacs is not installed')

        self.nqubits = nqubits
        self.config = config

    def generateCircuit(self, gateSeq):
        circuit = qulacs.QuantumCircuit(self.nqubits)

        for g in gateSeq:
            if g.id == 'h':
                circuit.add_H_gate(g.targets)

            elif g.id == 'x':
                circuit.add_X_gate(g.targets)

            elif g.id == 'cnot':
                circuit.add_CNOT_gate(g.controls, g.targets)

            elif g.id == 'cz':
                circuit.add_CZ_gate(g.controls, g.targets)

            elif g.id == 'rz':
                circuit.add_RZ_gate(g.targets, g.params)

            elif g.id == 'rx':
                circuit.add_RX_gate(g.targets, g.params)

            elif g.id == 'ry':
                circuit.add_RY_gate(g.targets, g.params)

            elif g.id == 'czpowgate':
                CZPow_matrix = [[1, 0], [0, exp(1j*pi*g.params)]]
                CZPowgate = qulacs.gate.DenseMatrix(g.targets, CZPow_matrix)
                CZPowgate.add_control_qubit(g.controls, 1)
                circuit.add_gate(CZPowgate)

            elif g.id == 'swap':
                assert len(g.targets) == 2
                circuit.add_SWAP_gate(g.targets[0], g.targets[1])

            elif g.id == 'cu':
                gate = qulacs.gate.DenseMatrix(g.targets, g.matrix)
                gate.add_control_qubit(g.controls, 1)
                circuit.add_gate(gate)

            elif g.id == 'u':
                gate = qulacs.gate.DenseMatrix(g.targets, g.matrix)
                circuit.add_gate(gate)

            elif g.id == 'measure':
                for i in g.targets:
                    circuit.add_gate(qulacs.gate.Measurement(i, i))

            else:
                raise NotImplementedError(f"The gate type {g.id} is not defined")
        
        return circuit
