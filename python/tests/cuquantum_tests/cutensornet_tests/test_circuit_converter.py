# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from .circuit_utils import *

class TestCircuits:
    @pytest.mark.skipif(cirq is None, reason="Cirq is not installed")
    @pytest.mark.parametrize("circuit", cirq_circuits)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("backend", backends)
    def test_cirq(self, circuit, dtype, backend, nsample=3, nsite_max=3, nfix_max=3):
        if isinstance(backend, str):
            pytest.skip(f'backend {backend} not found')
        else:
            cirq_tests = CirqTester(circuit, dtype, backend, nsample, nsite_max, nfix_max)
            cirq_tests.run_tests()
    
    @pytest.mark.skipif(qiskit is None, reason="Qiskit is not installed")
    @pytest.mark.parametrize("circuit", qiskit_circuits)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("backend", backends)
    def test_qiskit(self, circuit, dtype, backend, nsample=3, nsite_max=3, nfix_max=3):
        if isinstance(backend, str):
            pytest.skip(f'backend {backend} not found')
        else:
            qiskit_tests = QiskitTester(circuit, dtype, backend, nsample, nsite_max, nfix_max)
            qiskit_tests.run_tests()
