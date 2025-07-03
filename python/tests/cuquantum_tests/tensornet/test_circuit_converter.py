# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np

from cuquantum.tensornet import CircuitToEinsum, CirqParserOptions, QiskitParserOptions
from cuquantum._internal import tensor_wrapper, utils

from .utils.circuit_data import backend, backend_cycle, qiskit_circuits, input_testing_circuits, VALID_BACKENDS, testing_circuits
from .utils.circuit_tester import CircuitToEinsumTester
from .utils.test_utils import TensorBackend

CIRCUIT_TEST_SETTING = {'num_tests_per_task': 3,
                        'num_rdm_sites_max': 3,
                        'num_fix_sites_max': 3}

class TestParserOptions:

    @pytest.mark.parametrize("check_diagonal", (True, False))
    def test_cirq_parser_options(self, check_diagonal):
        options = CirqParserOptions(check_diagonal=check_diagonal)
        assert options.check_diagonal == check_diagonal
        with pytest.raises(TypeError):
            options = CirqParserOptions(decompose_gates=True)
    
    @pytest.mark.parametrize("decompose_gates", (True, False))
    @pytest.mark.parametrize("check_diagonal", (True, False))
    def test_qiskit_parser_options(self, decompose_gates, check_diagonal):
        options = QiskitParserOptions(decompose_gates=decompose_gates, check_diagonal=check_diagonal)
        assert options.decompose_gates == decompose_gates
        assert options.check_diagonal == check_diagonal


class TestCircuitToEinsum:

    @pytest.mark.parametrize("circuit", input_testing_circuits) # minimal circuits
    @pytest.mark.parametrize("backend", VALID_BACKENDS)
    @pytest.mark.parametrize("dtype", ('complex64', 'complex128'))
    @pytest.mark.parametrize("options", (
        None,
        QiskitParserOptions(decompose_gates=True, check_diagonal=False),
        CirqParserOptions(check_diagonal=True),
        {'check_diagonal': False},
        {'decompose_gates': False, 'check_diagonal': True},
    ))
    def test_input_args(self, circuit, backend, dtype, options):
        """Test all variants of the input arguments to the CircuitToEinsum constructor."""
        circuit_type = utils.infer_object_package(circuit)
        if circuit_type == 'qiskit':
            expect_type_error = isinstance(options, CirqParserOptions)
        else:
            expect_type_error = (isinstance(options, QiskitParserOptions) or
                (isinstance(options, dict) and 'decompose_gates' in options))
        if expect_type_error:
            with pytest.raises(TypeError):
                converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend, options=options)
        else:
            converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend, options=options)
            expr, operands = converter.amplitude('0'*len(converter.qubits))
            assert isinstance(expr, str)
            for o in operands:
                assert utils.infer_object_package(o) == backend
                o = tensor_wrapper.wrap_operand(o)
                assert o.dtype == dtype
    
    def _verify_dtype_correctness(self, circuit, options, backend):
        expr = {}
        operands = {}
        for dtype in ('complex64', 'complex128'):
            converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend, options=options)
            expr[dtype], operands[dtype] = converter.amplitude('0'*len(converter.qubits))
        assert expr['complex64'] == expr['complex128']
        tensor_backend = TensorBackend(backend)
        for o1, o2 in zip(operands['complex64'], operands['complex128']):
            if backend == 'torch':
                # torch.allclose requires same dtype for both operands
                o1 = tensor_backend.asarray(o1, dtype=o2.dtype)
            assert tensor_backend.allclose(o1, o2)
        
    @pytest.mark.parametrize("circuit", testing_circuits)
    @pytest.mark.parametrize("check_diagonal", (True, False))
    @pytest.mark.parametrize("decompose_gates", (True, False))
    def test_dtype_correctness(self, circuit, check_diagonal, decompose_gates, backend):
        circuit_type = utils.infer_object_package(circuit)
        if circuit_type == 'cirq':
            if decompose_gates:
                pytest.skip("decompose_gates option not applicable for Cirq circuits")
            options = {'check_diagonal': check_diagonal}
        else:
            options = {'decompose_gates': decompose_gates, 'check_diagonal': check_diagonal}
        self._verify_dtype_correctness(circuit, options, backend)
    
    @pytest.mark.parametrize("circuit", qiskit_circuits) # only qiskit circuits support decompose_gates option
    def test_decompose_gates_option(self, circuit):
        converter_1 = CircuitToEinsum(circuit, dtype='complex64', backend='numpy', options={'decompose_gates': True})
        converter_2 = CircuitToEinsum(circuit, dtype='complex64', backend='numpy', options={'decompose_gates': False})
        assert len(converter_1.gates) >= len(converter_2.gates)

    @pytest.mark.parametrize("circuit", testing_circuits)
    def test_check_diagonal_option(self, circuit):
        converter_1 = CircuitToEinsum(circuit, dtype='complex64', backend='numpy', options={'check_diagonal': True})
        converter_2 = CircuitToEinsum(circuit, dtype='complex64', backend='numpy', options={'check_diagonal': False})
        assert len(converter_1.gates) == len(converter_2.gates)

        expr_1, operands_1 = converter_1.amplitude('0'*len(converter_1.qubits))
        expr_2, operands_2 = converter_2.amplitude('0'*len(converter_2.qubits))
        assert len(operands_1) == len(operands_2)
        for o1, o2 in zip(operands_1, operands_2):
            if o1.shape == o2.shape:
                assert np.allclose(o1, o2)
            else:
                assert o1.shape == o2.shape[:o2.ndim//2]
                input_modes = [i for i in range(o1.ndim)]
                o2_diag = np.einsum(o2, input_modes*2, input_modes)
                assert np.allclose(o1, o2_diag)
    
    @pytest.mark.parametrize("circuit", testing_circuits)
    @pytest.mark.parametrize("check_diagonal", (True, False))
    @pytest.mark.parametrize("decompose_gates", (True, False))
    def test_expr_correctness(self, circuit, check_diagonal, decompose_gates, backend):
        circuit_type = utils.infer_object_package(circuit)
        if circuit_type == 'qiskit':
            options = QiskitParserOptions(decompose_gates=decompose_gates, check_diagonal=check_diagonal)
        else:
            if decompose_gates:
                pytest.skip("decompose_gates option not applicable for Cirq circuits")
            options = CirqParserOptions(check_diagonal=check_diagonal)
        # no need to test with complex128 as it's validated by test_dtype_correctness
        dtype = 'complex64'
        circuit_tests = CircuitToEinsumTester.from_circuit(circuit, dtype, backend, options=options, **CIRCUIT_TEST_SETTING)
        circuit_tests.run_tests()
