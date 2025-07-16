# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import warnings
import logging
try:
    import cirq
except ImportError:
    cirq = None
    
try:
    from .. import _internal_utils
except ImportError:
    _internal_utils = None
from .backend import Backend
from ..constants import LOGGER_NAME


# set up a logger
logger = logging.getLogger(LOGGER_NAME)


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
        self.meta = {}
        self.meta['ncputhreads'] = ncpu_threads

    def preprocess_circuit(self, circuit, *args, **kwargs):
        if _internal_utils is not None:
            _internal_utils.preprocess_circuit(self.identifier, circuit, *args, **kwargs)
        
        self.compute_mode = kwargs.pop('compute_mode')
        valid_choices = ['statevector', 'sampling']
        if self.compute_mode not in valid_choices:
            raise ValueError(f"The '{self.compute_mode}' computation mode is not supported for this backend. Supported modes are: {valid_choices}")
        
        self.updated_circuit = circuit
        if self.compute_mode == 'statevector':
            self.updated_circuit = cirq.drop_terminal_measurements(circuit)
            
        self.meta['compute-mode'] = f'{self.compute_mode}()'
        logger.info(f'data: {self.meta}')

        pre_data = self.meta
        return pre_data
    
    def run(self, circuit, nshots=1024):
        if self.compute_mode == 'sampling':
            results = self.backend.run(self.updated_circuit, repetitions=nshots)
            samples = results.histogram(key='result')
            post_res = results.measurements['result']
        elif self.compute_mode == 'statevector':
            results = self.backend.simulate(self.updated_circuit)
            sv = results.final_state_vector
            post_res = None

        return {'results': None, 'post_results': post_res, 'run_data': {}}


Cirq = functools.partial(_Cirq, identifier='cirq')
