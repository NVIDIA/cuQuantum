# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

class Frontend:

    def __init__(self, nqubits, config):
        raise NotImplementedError

    def generateCircuit(self, gateSeq):
        raise NotImplementedError
