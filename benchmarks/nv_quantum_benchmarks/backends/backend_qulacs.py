# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import os
import logging
import warnings
try:
    import qulacs 
    from qulacs.gate import Measurement
except ImportError:
    qulacs = None

from .backend import Backend
from ..constants import LOGGER_NAME


# set up a logger
logger = logging.getLogger(LOGGER_NAME)


class Qulacs(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, identifier=None, **kwargs):
        if qulacs is None:
            raise RuntimeError("qulacs is not installed")
        if precision != 'double':
            raise ValueError("qulacs backends only support double precision")
        self.identifier = identifier
        self.ngpus = ngpus
        self.ncpu_threads = ncpu_threads
        self.nqubits = kwargs.pop('nqubits')
        self.state = self.create_qulacs_state()
        self.version = qulacs.__version__
        self.meta = {}
        self.meta['ncputhreads'] = ncpu_threads

    def preprocess_circuit(self, circuit, *args, **kwargs):
        self.compute_mode = kwargs.pop('compute_mode')
        valid_choices = ['statevector', 'sampling']
        if self.compute_mode not in valid_choices:
            raise ValueError(f"The '{self.compute_mode}' computation mode is not supported for this backend. Supported modes are: {valid_choices}")
        
        self.updated_circuit = circuit
        if self.compute_mode == 'statevector':
            # remove all measurement gates
            gate_count = circuit.get_gate_count()
            for i in range(gate_count - 1, -1, -1):  
                gate = circuit.get_gate(i)
                if gate.get_name() == 'CPTP':  
                    print("measure")
                    self.updated_circuit.remove_gate(i)

        self.meta['compute-mode'] = f'{self.compute_mode}()'
        logger.info(f'data: {self.meta}')

        pre_data = self.meta
        return pre_data
    
    def create_qulacs_state(self):
        if self.identifier == 'qulacs-gpu':
            if self.ngpus > 1:
                raise ValueError(f"cannot specify --ngpus > 1 for the backend {self.identifier}")
            try:
                state = qulacs.QuantumStateGpu(self.nqubits)
            except AttributeError as e:
                raise RuntimeError("please clone Qulacs and build it from source via \"USE_GPU=Yes "
                                   "pip install .\", or follow Qulacs instruction for customized "
                                   "builds") from e
        elif self.identifier == 'qulacs-cpu':
            if self.ngpus != 0:
                raise ValueError(f"cannot specify --ngpus for the backend {self.identifier}")
            if self.ncpu_threads > 1 and self.ncpu_threads != (
                    int(os.environ.get("OMP_NUM_THREADS", "-1")) or int(os.environ.get("QULACS_NUM_THREADS", "-1"))):
                warnings.warn(f"--ncputhreads is ignored, for {self.identifier} please set the env var OMP_NUM_THREADS or QULACS_NUM_THREADS instead",
                              stacklevel=2)
            state = qulacs.QuantumState(self.nqubits)
        else:
            raise ValueError(f"the backend {self.identifier} is not recognized")
        return state

    def run(self, circuit, nshots=1024):
        # init/reset sv
        self.state.set_zero_state()

        if self.compute_mode == 'sampling':
            self.updated_circuit.update_quantum_state(self.state)
            samples = self.state.sampling(nshots) 
        elif self.compute_mode == 'statevector':
            self.updated_circuit.update_quantum_state(self.state)
            sv = self.state.get_vector() 
        
        return {'results': None, 'post_results': None, 'run_data': {}}


QulacsGpu = functools.partial(Qulacs, identifier='qulacs-gpu')
QulacsCpu = functools.partial(Qulacs, identifier='qulacs-cpu')
