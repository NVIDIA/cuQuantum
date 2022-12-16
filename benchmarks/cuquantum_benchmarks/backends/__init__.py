# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .backend_cirq import Cirq
from .backend_cutn import cuTensorNet
from .backend_qsim import Qsim, QsimCuda, QsimCusv, QsimMgpu
from .backend_qiskit import Aer, AerCuda, AerCusv, CusvAer


backends = {
    'aer': Aer,
    'aer-cuda': AerCuda,
    'aer-cusv': AerCusv,
    'cusvaer': CusvAer,
    'cirq': Cirq,
    'cutn': cuTensorNet,
    'qsim': Qsim,
    'qsim-cuda': QsimCuda,
    'qsim-cusv': QsimCusv,
    'qsim-mgpu': QsimMgpu,
}

def createBackend(backend_name, ngpus, ncpu_threads, precision, logger, *args, **kwargs):
    return backends[backend_name](ngpus, ncpu_threads, precision, logger, *args, **kwargs)
