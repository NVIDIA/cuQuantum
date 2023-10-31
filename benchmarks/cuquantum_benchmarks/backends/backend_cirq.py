# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import warnings

try:
    import cirq
except ImportError:
    cirq = None
    
try:
    from .. import _internal_utils
except ImportError:
    _internal_utils = None
from .backend import Backend


class _Cirq(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, identifier=None, **kwargs):
        if cirq is None:
            raise RuntimeError("cirq is not installed")
        if ngpus > 0:
            raise ValueError("the cirq backend only runs on CPU")
        if ncpu_threads > 1:
            warnings.warn("cannot set the number of CPU threads for the cirq backend")
        if precision != 'single':
            raise ValueError("the cirq backend only supports single precision")

        self.backend = cirq.Simulator()
        self.identifier = identifier
        self.version = cirq.__version__

    def preprocess_circuit(self, circuit, *args, **kwargs):
        if _internal_utils is not None:
            _internal_utils.preprocess_circuit(self.identifier, circuit, *args, **kwargs)
        
        return {}
    
    def run(self, circuit, nshots=1024):
        run_data = {}
        if nshots > 0:
            results = self.backend.run(circuit, repetitions=nshots)
        else:
            results = self.backend.simulate(circuit)
        post_res = results.measurements['result']
        return {'results': results, 'post_results': post_res, 'run_data': run_data}


Cirq = functools.partial(_Cirq, identifier='cirq')
