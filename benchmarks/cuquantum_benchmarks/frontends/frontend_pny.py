# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
from cmath import pi, exp

try:
    import pennylane
except ImportError:
    pennylane = None

from .frontend import Frontend


class Pennylane(Frontend):

    def __init__(self, nqubits, config):
        if pennylane is None:
            raise RuntimeError('pennylane is not installed')

        self.nqubits = nqubits
        self.config = config

    def generateCircuit(self, gateSeq):
        last_g = gateSeq[-1]
        assert last_g.id == "measure"  # TODO: relax this?
        
        def circuit():
            measured_qs = None

            for g in gateSeq:
                if g.id =='h': 
                    pennylane.Hadamard(wires=g.targets)

                elif g.id =='x': 
                    pennylane.PauliX(wires=g.targets)

                elif g.id =='cnot': 
                    pennylane.CNOT(wires=[g.controls, g.targets])

                elif g.id =='cz': 
                    pennylane.CZ(wires=[g.controls, g.targets])

                elif g.id =='rz': 
                    pennylane.RZ(g.params, g.targets)

                elif g.id =='rx': 
                    pennylane.RX(g.params, g.targets)

                elif g.id =='ry': 
                    pennylane.RY(g.params, g.targets)

                elif g.id =='czpowgate': 
                    CZPow_matrix = [[1,0],[0,exp(1j*pi*g.params)]]
                    pennylane.ControlledQubitUnitary(CZPow_matrix,control_wires=g.controls, wires=g.targets)

                elif g.id =='swap': 
                    pennylane.SWAP(wires=[g.targets[0], g.targets[1]])

                elif g.id =='cu': 
                    pennylane.ControlledQubitUnitary(g.matrix, control_wires=g.controls, wires=g.targets)

                elif g.id == 'u':  
                    pennylane.QubitUnitary(g.matrix, wires=g.targets)

                elif g.id == "measure":
                    measured_qs = g.targets

                else:
                    raise NotImplementedError(f"The gate type {g.id} is not defined")
            
            return pennylane.sample(wires=measured_qs) 
    
        return circuit