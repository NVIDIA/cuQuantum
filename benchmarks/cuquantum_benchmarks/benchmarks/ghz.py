# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .benchmark import Benchmark
from .._utils import Gate


class GHZ(Benchmark):

    @staticmethod
    def generateGatesSequence(nqubits, config):
        circuit = [Gate(id='h', targets=0)]
        circuit += [Gate(id='cnot', controls=idx, targets=idx+1) for idx in range(nqubits-1)]
        measure = config['measure']
        if measure:
            circuit.append(Gate(id='measure', targets=list(range(nqubits))))
        return circuit
