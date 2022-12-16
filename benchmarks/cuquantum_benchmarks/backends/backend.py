# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

class Backend:

    def __init__(self, ngpus, ncpu_threads, precision, logger, *args, **kwargs):
        raise NotImplementedError

    def preprocess_circuit(self, circuit, *args, **kwargs):
        return {}

    def run(self, circuit, nshots):
        raise NotImplementedError
