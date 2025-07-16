# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .benchmark import Benchmark
from .._utils import Gate


class IQFT(Benchmark):

    @staticmethod
    def generateGatesSequence(nqubits, config):
        circuit = IQFT._iqft_component(nqubits)
        measure = config['measure']
        if measure:
            circuit.append(Gate(id='measure', targets=list(range(nqubits))))
        return circuit

    def _iqft_component(nqubits):
        iqft = []
        for q in range(nqubits//2):
            iqft.append(Gate(id='swap', targets=(q, nqubits-q-1)))
            pass
        for q in range(nqubits-1, -1, -1):
            for p in range(nqubits-1, q, -1):
                iqft.append(Gate(id='czpowgate', params=-1/(2**(p-q)), controls=q, targets=p))
            iqft.append(Gate(id='h', targets=q))
        return iqft
