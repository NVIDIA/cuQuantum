# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

class Benchmark:

    @staticmethod
    def generateGatesSequence(nqubits, config):
        raise NotImplementedError
        
    @staticmethod
    def postProcess(nqubits, results):
        return False
