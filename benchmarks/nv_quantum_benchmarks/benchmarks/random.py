# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import random

from .benchmark import Benchmark
from .._utils import Gate


class Random(Benchmark):

    # TODO: this should be frontend's property
    gate_types = ('h', 'x', 'rz', 'rx', 'ry', 'cnot', 'cz', 'swap')

    @staticmethod
    def generateGatesSequence(nqubits, config):
        try:
            depth = config['depth']
        except KeyError:
            depth = nqubits

        gate_types = Random.gate_types
        seed = np.iinfo(np.int32).max
        rng = np.random.default_rng(seed)
        circuit = []

        # apply arbitrary random operations at every depth
        for _ in range(depth):
            # choose either 1, 2, or 3 qubits for the operation
            qubits_shuffle = list(range(nqubits))
            rng.shuffle(qubits_shuffle)
            while qubits_shuffle:
                max_possible_operands = min(len(qubits_shuffle), 2)
                num_operands = rng.choice(range(max_possible_operands)) + 1
                operands = [qubits_shuffle.pop() for _ in range(num_operands)]
                # TODO: gate_type_num depends on the gate order in gate_types
                if num_operands == 1:
                    gate_type_num = random.randint(0, 4)
                    if gate_type_num < 2:
                        circuit.append(Gate(id=gate_types[gate_type_num], targets=operands[0]))
                    else:
                        angle = rng.uniform(0, 2 * np.pi)
                        circuit.append(Gate(id=gate_types[gate_type_num], params=angle, targets=operands[0]))
                elif num_operands == 2:
                    gate_type_num = random.randint(5, 7)
                    if gate_type_num in (5, 6):
                        circuit.append(Gate(id=gate_types[gate_type_num], controls=operands[0], targets=operands[1]))
                    else:
                        circuit.append(Gate(id=gate_types[gate_type_num], targets=[operands[0], operands[1]]))

        measure = config['measure']
        if measure:
            circuit.append(Gate(id='measure', targets=list(range(nqubits))))

        return circuit
