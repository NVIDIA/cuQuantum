# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .frontend_cirq import Cirq
from .frontend_qiskit import Qiskit


frontends = {
    'cirq': Cirq,
    'qiskit': Qiskit,
}

def createFrontend(frontend_name, nqubits, config):
    return frontends[frontend_name](nqubits, config)
