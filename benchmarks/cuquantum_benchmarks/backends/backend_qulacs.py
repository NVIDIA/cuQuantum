# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import os
import time
import warnings

try:
    import qulacs 
except ImportError:
    qulacs = None

from .backend import Backend


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

        # actual circuit simulation
        circuit.update_quantum_state(self.state)

        run_data = {}
        if nshots > 0:
            results = self.state.sampling(nshots)
        else:
            results = self.state.get_vector()  # TODO: too heavyweight?

        return {'results': results, 'post_results': None, 'run_data': run_data}


QulacsGpu = functools.partial(Qulacs, identifier='qulacs-gpu')
QulacsCpu = functools.partial(Qulacs, identifier='qulacs-cpu')
