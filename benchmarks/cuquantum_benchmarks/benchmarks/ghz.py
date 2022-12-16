# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .benchmark import Benchmark


class GHZ(Benchmark):

    @staticmethod
    def generateGatesSequence(nqubits, config):
        circuit = [('h', [0])]
        circuit += [('cnot', [idx, idx+1]) for idx in range(nqubits-1)]
        measure = config['measure']
        if measure:
            circuit.append(('measure', [list(range(nqubits))]))
        return circuit
