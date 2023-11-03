# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .frontend_cirq import Cirq
from .frontend_qiskit import Qiskit
from .frontend_pny import Pennylane
from .frontend_qulacs import Qulacs


frontends = {
    'cirq': Cirq,
    'qiskit': Qiskit,
    'pennylane': Pennylane,
    'qulacs': Qulacs
}


def createFrontend(frontend_name, nqubits, config):
    return frontends[frontend_name](nqubits, config)
