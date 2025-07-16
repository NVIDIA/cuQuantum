# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .benchmark import Benchmark
from .._utils import Gate, random_unitary


class QuantumVolume(Benchmark):

    @staticmethod
    def generateGatesSequence(nqubits, config):
        # This routine is roughly equivalent to Cirq's quantum_volume.generate_model_circuit()
        # and Qiskit's QuantumVolume(..., classical_permutation=True).

        # copied from cusvaer
        n_variations = 10  # unused
        depth = 30
        seed = 1000
        width = nqubits // 2
        rng = np.random.default_rng(seed)
        measure = config['measure']

        circuit = []
        for _ in range(depth):
            perm = rng.permutation(nqubits)

            # apply an su4 gate on each pair in the layer
            for w in range(width):
                su4 = random_unitary(4, rng)
                assert su4.shape == (4, 4)
                idx = [perm[2*w], perm[2*w+1]]
                circuit.append(Gate(id='u', matrix=su4, targets=idx))

        if measure:
            circuit.append(Gate(id='measure', targets=list(range(nqubits))))

        return circuit
