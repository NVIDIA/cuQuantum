# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import cupy as cp
try:
    import qsimcirq
except ImportError:
    qsimcirq = None
try:
    import cirq
except ImportError:
    cirq = None

from .backend import Backend
from ..constants import LOGGER_NAME


# set up a logger
logger = logging.getLogger(LOGGER_NAME)


class QsimCirq(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, identifier=None, **kwargs):
        if qsimcirq is None:
            raise RuntimeError("qsimcirq is not installed")
        if precision != 'single':
            raise ValueError("all qsim backends only support single precision")
        self.identifier = identifier
        qsim_options = self.create_qsim_options(identifier, ngpus, ncpu_threads, **kwargs)
        self.version = qsimcirq.__version__
        self.backend = qsimcirq.QSimSimulator(qsim_options=qsim_options)
        self.meta = {}
        self.meta['ncputhreads'] = ncpu_threads

    def preprocess_circuit(self, circuit, *args, **kwargs):
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
    
    @staticmethod
    def create_qsim_options(identifier, ngpus, ncpu_threads, **kwargs):
        nfused = kwargs.pop('nfused')
        if identifier == "qsim-mgpu":
            if ngpus >= 1:
                # use cuQuantum Appliance interface
                ops = qsimcirq.QSimOptions(gpu_mode=tuple(range(ngpus)), max_fused_gate_size=nfused)
            else:
                raise ValueError(f"need to specify --ngpus for the backend {identifier}")
        elif identifier == "qsim-cuda":
            if ngpus == 1:
                try:
                    # use public interface
                    ops = qsimcirq.QSimOptions(gpu_mode=0, use_gpu=True, max_fused_gate_size=nfused)
                except TypeError:
                    # use cuQuantum Appliance interface
                    ops = qsimcirq.QSimOptions(gpu_mode=0, disable_gpu=False, use_sampler=False, max_fused_gate_size=nfused)
            else:
                raise ValueError(f"need to specify --ngpus 1 for the backend {identifier}")
        elif identifier == "qsim-cusv":
            if ngpus == 1:
                try:
                    # use public interface
                    ops = qsimcirq.QSimOptions(gpu_mode=1, use_gpu=True, max_fused_gate_size=nfused)
                except TypeError:
                    # use cuQuantum Appliance interface
                    ops = qsimcirq.QSimOptions(gpu_mode=1, disable_gpu=False, use_sampler=False, max_fused_gate_size=nfused)
            else:
                raise ValueError(f"need to specify --ngpus 1 for the backend {identifier}")
        elif identifier == "qsim":
            if ngpus != 0:
                raise ValueError(f"cannot specify --ngpus for the backend {identifier}")
            try:
                # use public interface
                ops = qsimcirq.QSimOptions(use_gpu=False, cpu_threads=ncpu_threads, max_fused_gate_size=nfused)
            except TypeError:
                # use cuQuantum Appliance interface
                ops = qsimcirq.QSimOptions(disable_gpu=True, use_sampler=False, cpu_threads=ncpu_threads, max_fused_gate_size=nfused,
                                           gpu_mode=0)
        else:
            raise ValueError(f"the backend {identifier} is not recognized")

        return ops

    def run(self, circuit, nshots=1024):
        if self.identifier == "qsim-mgpu":
            dev = cp.cuda.Device()

        if self.compute_mode == 'sampling':
            results = self.backend.run(self.updated_circuit, repetitions=nshots)
            samples = results.data
        elif self.compute_mode == 'statevector':
            results = self.backend.simulate(self.updated_circuit)
            sv = results.state_vector()
            
        if self.identifier == "qsim-mgpu":
            # work around a bug
            if dev != cp.cuda.Device():
                dev.use()

        return {'results': None, 'post_results': None, 'run_data': {}}


QsimMgpu = functools.partial(QsimCirq, identifier='qsim-mgpu')
QsimCuda = functools.partial(QsimCirq, identifier='qsim-cuda')
QsimCusv = functools.partial(QsimCirq, identifier='qsim-cusv')
Qsim = functools.partial(QsimCirq, identifier='qsim')
