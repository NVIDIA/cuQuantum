# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools

import cupy as cp
try:
    import qsimcirq
except ImportError:
    qsimcirq = None

from .backend import Backend


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

    def run(self, circuit, nshots=1024):
        run_data = {}

        if self.identifier == "qsim-mgpu":
            dev = cp.cuda.Device()
        if nshots > 0:
            results = self.backend.run(circuit, repetitions=nshots)
        else:
            results = self.backend.simulate(circuit)
        if self.identifier == "qsim-mgpu":
            # work around a bug
            if dev != cp.cuda.Device():
                dev.use()

        post_res = None # TODO
        return {'results': results, 'post_results': post_res, 'run_data': run_data}

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


QsimMgpu = functools.partial(QsimCirq, identifier='qsim-mgpu')
QsimCuda = functools.partial(QsimCirq, identifier='qsim-cuda')
QsimCusv = functools.partial(QsimCirq, identifier='qsim-cusv')
Qsim = functools.partial(QsimCirq, identifier='qsim')
