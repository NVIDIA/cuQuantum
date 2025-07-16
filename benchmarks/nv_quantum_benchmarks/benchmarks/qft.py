# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .benchmark import Benchmark
from .._utils import Gate


class QFT(Benchmark):

    @staticmethod
    def generateGatesSequence(nqubits, config):
        circuit = QFT._qft_component(nqubits)
        measure = config['measure']
        if measure:
            circuit.append(Gate(id='measure', targets=list(range(nqubits))))
        return circuit

    def _qft_component(nqubits):
        qft = []
        for q in range(nqubits):
            qft.append(Gate(id='h', targets=q))
            for p in range(q+1, nqubits):
                qft.append(Gate(id='czpowgate', params=1/(2**(p-q)), controls=q, targets=p))
        for q in range(nqubits//2):
            qft.append(Gate(id='swap', targets=(q, nqubits-q-1)))
        return qft
