# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

try:
    import cirq
except ImportError:
    cirq = None

from .backend import Backend


class Cirq(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, **kwargs):
        if cirq is None:
            raise RuntimeError("cirq is not installed")
        if ngpus > 0:
            raise ValueError("the cirq backend only runs on CPU")
        if ncpu_threads > 1:
            warnings.warn("cannot set the number of CPU threads for the cirq backend")
        if precision != 'single':
            raise ValueError("the cirq backend only supports single precision")

        self.backend = cirq.Simulator()

    def run(self, circuit, nshots=1024):
        run_data = {}
        if nshots > 0:
            results = self.backend.run(circuit, repetitions=nshots)
        else:
            results = self.backend.simulate(circuit)
        post_res = results.measurements['result']
        return {'results': results, 'post_results': post_res, 'run_data': run_data}
