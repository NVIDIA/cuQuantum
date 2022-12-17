# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .benchmark import Benchmark


class QFT(Benchmark):

    @staticmethod
    def generateGatesSequence(nqubits, config):
        circuit = QFT._qft_component(nqubits)
        measure = config['measure']
        if measure:
            circuit.append(('measure', [list(range(nqubits))]))
        return circuit

    def _qft_component(nqubits):
        qft = []
        for q in range(nqubits):
            qft.append(('h', [q]))
            for p in range(q+1, nqubits):
                qft.append( ('czpowgate', [1 / (2 ** (p - q)), q, p]) )
        for q in range(nqubits//2):
            qft.append( ('swap', [q, nqubits-q-1]) )
        return qft
