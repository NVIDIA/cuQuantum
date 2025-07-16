# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

class Frontend:

    # The current assumptions for measurement:
    #   1. we only do measure once throughout a gate sequence, and it’s done at the end of the sequence
    #   2. measure is applied to first x (most likely x=all) qubits in the circuit
    # When we introduce benchmarks that do mid-circuit measurement, we must revisit the assumption!

    def __init__(self, nqubits, config):
        raise NotImplementedError

    def generateCircuit(self, gateSeq):
        raise NotImplementedError
