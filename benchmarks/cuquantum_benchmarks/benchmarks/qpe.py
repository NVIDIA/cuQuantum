# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .benchmark import Benchmark
from .iqft import IQFT
from .._utils import Gate


class QPE(Benchmark):

    @staticmethod
    def generateGatesSequence(nqubits, config):
        if nqubits < 2:
            raise ValueError("for qpe the number of qubits should be >=2")

        # Example instantiation of QPE circuit paramterized by nqubits
        phase = 1/3
        U = np.mat([[1, 0], [0, np.exp(np.pi * 1j * phase)]])
        in_nqubits = 1
        unfold = config['unfold']
        measure = config['measure']
        circuit = QPE._make_qpe_circuit(in_nqubits, nqubits-in_nqubits, U, unfold, 'P(1/3)')

        if measure:
            # Measure Counting Qubits
            circuit.append(Gate(id='measure', targets=list(range(nqubits-in_nqubits))))

        return circuit

    def _make_qpe_component(in_qubits, t_qubits, U, unfold, U_name):
        component = []
        in_nqubits = len(in_qubits)
        t_nqubits = len(t_qubits)

        # 1. Setup Eigenstate
        component.append(Gate(id='x', targets=in_qubits[0]))

        # 2. Superposition Counting Qubits
        all_h = [Gate(id='h', targets=idx) for idx in t_qubits]
        component += all_h

        # 3. Controlled-U
        prev_U = np.identity(2 ** in_nqubits)
        for t in range(t_nqubits):
            if unfold:
                for i in range(2 ** t):
                    component.append(Gate(id='cu', matrix=U, name=f'{U_name}', controls=t_qubits[t], targets=in_qubits))

            else:
                new_U = prev_U @ U
                component.append(Gate(id='cu', matrix=new_U, name=f'{U_name}^(2^{t})', controls=t_qubits[t_nqubits-t-1], targets=in_qubits))
                prev_U = new_U

        # 4. Inverse QFT Counting Qubits
        iqft_component = IQFT._iqft_component(t_nqubits)
        component += iqft_component

        return component

    # Input:
    #   :- U is a numpy unitary array 2*2
    #   :- unfold implements the powers of U using U, rather than create new power gates
    def _make_qpe_circuit(in_nqubits, t_nqubits, U, unfold, U_name):
        assert 2 ** in_nqubits == U.shape[0] == U.shape[1], "Mismatched number of qubits between U and in_nqubits"

        t_qubits = [idx for idx in range(t_nqubits)]
        in_qubits = [idx for idx in range(t_nqubits, t_nqubits + in_nqubits)]
        circuit = QPE._make_qpe_component(in_qubits, t_qubits, U, unfold, U_name)
        return circuit
