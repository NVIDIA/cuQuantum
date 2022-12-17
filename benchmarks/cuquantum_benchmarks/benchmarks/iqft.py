# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .benchmark import Benchmark


class IQFT(Benchmark):

    @staticmethod
    def generateGatesSequence(nqubits, config):
        circuit = IQFT._iqft_component(nqubits)
        measure = config['measure']
        if measure:
            circuit.append(('measure', [list(range(nqubits))]))
        return circuit

    def _iqft_component(nqubits):
        iqft = []
        for q in range(nqubits//2):
            iqft.append( ('swap', [q, nqubits-q-1]) )
            pass
        for q in range(nqubits-1, -1, -1):
            for p in range(nqubits-1, q, -1):
                iqft.append( ('czpowgate', [-1 / (2 ** (p - q)), q, p]) )
            iqft.append(('h', [q]))
        return iqft
