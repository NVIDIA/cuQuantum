# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from cuquantum import cutensornet as cutn

from .circuit_data import backend, backend_cycle, testing_circuits
from .circuit_tester import CircuitToEinsumTester

CIRCUIT_TEST_SETTING = {'num_tests_per_task': 3,
                        'num_rdm_sites_max': 3,
                        'num_fix_sites_max': 3}

@pytest.fixture(scope='class')
def handle():
    _handle = cutn.create()
    yield _handle
    cutn.destroy(_handle)

class TestCircuitToEinsum:
    @pytest.mark.parametrize("circuit", testing_circuits)
    @pytest.mark.parametrize("dtype", ('complex64', 'complex128',))
    def test_circuit_converter(self, circuit, dtype, backend, handle):
        print(f"{backend=}")
        # Results from Cirq/Qiskit are compared with CircuitToEinsum
        circuit_tests = CircuitToEinsumTester.from_circuit(circuit, dtype, backend, handle=handle, **CIRCUIT_TEST_SETTING)
        circuit_tests.run_tests()
