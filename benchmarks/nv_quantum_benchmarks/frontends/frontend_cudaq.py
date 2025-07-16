# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from math import pi
import logging
import time
import numpy as np
try:
    import cudaq
except ImportError:
    cudaq = None

from .frontend import Frontend
from ..constants import LOGGER_NAME


# set up a logger
logger = logging.getLogger(LOGGER_NAME)


class CUDAQ(Frontend):

    def __init__(self, nqubits, config):
        if cudaq is None:
            raise RuntimeError("CUDA Quantum (cudaq) is not installed")
        self.nqubits = nqubits
        precision = config['precision']
        self.dtype = np.complex64 if precision == "single" else np.complex128  # TODO

    def generateCircuit(self, gateSeq):
        t0 = time.perf_counter()
        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(self.nqubits)
        
        # to generate unique names for u and cu gates
        u_idx = 0
        for g in gateSeq:
            if g.id == 'h':
                kernel.h(qubits[g.targets])

            elif g.id == 'x':
                kernel.x(qubits[g.targets])

            elif g.id == 'cnot':
                kernel.cx(control=qubits[g.controls], target=qubits[g.targets])

            elif g.id == 'cz':
                kernel.cz(control=qubits[g.controls], target=qubits[g.targets])

            elif g.id == 'rz':
                kernel.rz(parameter=g.params, target=qubits[g.targets])

            elif g.id == 'rx':
                kernel.rx(parameter=g.params, target=qubits[g.targets])

            elif g.id == 'ry':
                kernel.ry(parameter=g.params, target=qubits[g.targets])

            elif g.id == 'czpowgate':
                kernel.cr1(pi*g.params, qubits[g.controls], qubits[g.targets])

            elif g.id == 'swap':
                assert len(g.targets) == 2
                kernel.swap(qubits[g.targets[0]], qubits[g.targets[1]])

            elif g.id == 'u': 
                cudaq.register_operation(f'u_{u_idx}', g.matrix)
                kernel.__getattr__(f'u_{u_idx}')(qubits[int(g.targets[0])], qubits[int(g.targets[1])])
                u_idx += 1

            elif g.id == 'cu':
                cudaq.register_operation(f'u_{u_idx}', g.matrix)
                kernel.__getattr__(f'u_{u_idx}')(qubits[g.controls], qubits[int(g.targets[0])])
                u_idx += 1
                
            elif g.id == 'measure':
                pass

            else:
                raise NotImplementedError(f"the gate type {g.id} is not defined")

        t1 = time.perf_counter()
        logger.debug(f"cudaq kernel creation took {t1-t0} s")
        print(f'\nKernel creation time: {t1-t0} s\n')
        return kernel
