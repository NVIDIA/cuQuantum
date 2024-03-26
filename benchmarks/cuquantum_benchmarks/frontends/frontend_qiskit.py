# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from math import pi

try:
    import qiskit
    if hasattr(qiskit, "__version__") and qiskit.__version__ >= "1.0.0":
        from qiskit.circuit.library import UnitaryGate
    else:
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
        last_g = gateSeq[-1]
        assert last_g.id == "measure"  # TODO: relax this?
        circuit = qiskit.QuantumCircuit(self.nqubits, len(last_g.targets))

        for g in gateSeq:
            if g.id == 'h':
                circuit.h(g.targets)

            elif g.id == 'x':
                circuit.x(g.targets)

            elif g.id == 'cnot':
                circuit.cx(g.controls, g.targets)

            elif g.id == 'cz':
                circuit.cz(g.controls, g.targets)

            elif g.id == 'rz':
                circuit.rz(g.params, g.targets)

            elif g.id == 'rx':
                circuit.rx(g.params, g.targets)

            elif g.id == 'ry':
                circuit.ry(g.params, g.targets)

            elif g.id == 'czpowgate':
                circuit.cp(pi*g.params, g.controls, g.targets)

            elif g.id == 'swap':
                circuit.swap(*g.targets)

            elif g.id == 'cu':
                U_gate = UnitaryGate(g.matrix, g.name).control(1)
                circuit.append(U_gate, [g.controls]+g.targets[::-1])

            elif g.id == 'u':
                # TODO: give the gate a name?
                U_gate = UnitaryGate(g.matrix)
                circuit.append(U_gate, g.targets[::-1])

            elif g.id == 'measure':
                circuit.measure(g.targets, g.targets)

            else:
                raise NotImplementedError(f"The gate type {g.id} is not defined")
        
        return circuit
