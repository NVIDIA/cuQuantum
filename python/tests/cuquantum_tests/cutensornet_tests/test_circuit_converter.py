# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from .circuit_utils import backends
from .circuit_utils import cirq_circuits, CirqTester
from .circuit_utils import qiskit_circuits, QiskitTester


class TestCircuitToEinsum:
    # If PyTorch/Qiskit/Cirq is not installed, the corresponding tests are silently
    # skipped.

    @pytest.mark.parametrize("circuit", cirq_circuits)
    @pytest.mark.parametrize("dtype", ('complex64', 'complex128',))
    @pytest.mark.parametrize("backend", backends)
    def test_cirq(self, circuit, dtype, backend, nsample=3, nsite_max=3, nfix_max=3):
        cirq_tests = CirqTester(circuit, dtype, backend, nsample, nsite_max, nfix_max)
        cirq_tests.run_tests()
    
    @pytest.mark.parametrize("circuit", qiskit_circuits)
    @pytest.mark.parametrize("dtype", ('complex64', 'complex128',))
    @pytest.mark.parametrize("backend", backends)
    def test_qiskit(self, circuit, dtype, backend, nsample=3, nsite_max=3, nfix_max=3):
        qiskit_tests = QiskitTester(circuit, dtype, backend, nsample, nsite_max, nfix_max)
        qiskit_tests.run_tests()
