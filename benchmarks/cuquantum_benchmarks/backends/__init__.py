# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .backend_cirq import Cirq
from .backend_cutn import cuTensorNet
from .backend_pny import Pny, PnyLightningGpu, PnyLightningCpu, PnyLightningKokkos
from .backend_qsim import Qsim, QsimCuda, QsimCusv, QsimMgpu
from .backend_qiskit import Aer, AerCuda, AerCusv, CusvAer
from .backend_qulacs import QulacsGpu, QulacsCpu
try:
    from .backend_naive import Naive
except ImportError:
    Naive = None


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
    'pennylane': Pny,
    'pennylane-lightning-gpu': PnyLightningGpu,
    'pennylane-lightning-qubit': PnyLightningCpu,
    'pennylane-lightning-kokkos': PnyLightningKokkos,
    'qulacs-cpu': QulacsCpu,
    'qulacs-gpu': QulacsGpu,
}
if Naive:
    backends['naive'] = Naive


def createBackend(backend_name, ngpus, ncpu_threads, precision, *args, **kwargs):
    return backends[backend_name](ngpus, ncpu_threads, precision, *args, **kwargs)
