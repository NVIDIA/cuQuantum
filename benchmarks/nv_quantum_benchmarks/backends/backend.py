# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

class Backend:

    def __init__(self, ngpus, ncpu_threads, precision, *args, **kwargs):
        raise NotImplementedError

    def preprocess_circuit(self, circuit, *args, **kwargs):
        return {}

    def pre_run(self, circuit, *args, **kwargs):
        pass

    def run(self, circuit, nshots):
        raise NotImplementedError
