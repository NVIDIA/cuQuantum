# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import pytest
import numpy as np

from nvmath.internal import utils, tensor_wrapper
from cuquantum.tensornet import contract, CircuitToEinsum, CirqParserOptions, QiskitParserOptions

from .utils.circuit_matrix import CircuitMatrix, QiskitCircuitMatrix, CirqCircuitMatrix
from .utils.circuit_ifc import CircuitHelper, PropertyComputeHelper
from .utils.circuit_tester import BaseCircuitToEinsumTester
from .utils.data import ARRAY_BACKENDS
from .utils.helpers import TensorBackend


NUM_TESTS_PER_TASK = 3

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

@pytest.fixture(params=CircuitMatrix.realL0(), scope="class")
def real_circuit(request):
    return request.param

@pytest.fixture(scope="class")
def real_circuit_sv(real_circuit):
    return CircuitHelper.compute_state_vector(real_circuit)


class TestCircuitToEinsumFunctionality:

    @pytest.mark.parametrize("options", (
        None,
        CirqParserOptions(check_diagonal=False),
        {'check_diagonal': True},
        QiskitParserOptions(decompose_gates=True, check_diagonal=False),
        {'decompose_gates': False, 'check_diagonal': True},
    ))
    @pytest.mark.parametrize("circuit", CircuitMatrix.L2())
    def test_circuit_options(self, circuit, options):
        circuit_type = utils.infer_object_package(circuit)
        if circuit_type == 'qiskit':
            expect_type_error = isinstance(options, CirqParserOptions)
        else:
            expect_type_error = (isinstance(options, QiskitParserOptions) or
                (isinstance(options, dict) and 'decompose_gates' in options))
        if expect_type_error:
            context = pytest.raises(TypeError)
        else:
            context = contextlib.nullcontext()
        with context:
            converter = CircuitToEinsum(circuit, dtype='complex64', backend="numpy", options=options)
            assert converter.gates is not None
            assert converter.qubits is not None
    
    @pytest.mark.parametrize("circuit", QiskitCircuitMatrix.L1()) # only qiskit circuits support decompose_gates option
    def test_decompose_gates_option(self, circuit):
        converter_1 = CircuitToEinsum(circuit, dtype='complex64', backend="numpy", options={'decompose_gates': True})
        converter_2 = CircuitToEinsum(circuit, dtype='complex64', backend="numpy", options={'decompose_gates': False})
        assert len(converter_1.gates) >= len(converter_2.gates)

    @pytest.mark.parametrize("circuit", CircuitMatrix.L1())
    def test_check_diagonal_option(self, circuit):
        converter_1 = CircuitToEinsum(circuit, dtype='complex64', backend="numpy", options={'check_diagonal': True})
        converter_2 = CircuitToEinsum(circuit, dtype='complex64', backend="numpy", options={'check_diagonal': False})
        assert len(converter_1.gates) == len(converter_2.gates)
        operands_1 = converter_1.amplitude('0'*len(converter_1.qubits))[1]
        operands_2 = converter_2.amplitude('0'*len(converter_2.qubits))[1]
        assert len(operands_1) == len(operands_2)
        for o1, o2 in zip(operands_1, operands_2):
            if o1.shape == o2.shape:
                assert np.allclose(o1, o2)
            else:
                assert o1.shape == o2.shape[:o2.ndim//2]
                input_modes = [i for i in range(o1.ndim)]
                o2_diag = np.einsum(o2, input_modes*2, input_modes)
                assert np.allclose(o1, o2_diag)

    @pytest.mark.parametrize("circuit", CircuitMatrix.L1())
    def test_batched_amplitudes_marginal_cases(self, circuit):
        converter = CircuitToEinsum(circuit, dtype='complex64', backend="numpy")
        expr, operands = converter.state_vector()
        expr1, operands1 = converter.batched_amplitudes({})
        assert expr == expr1
        for o1, o2 in zip(operands, operands1):
            assert np.allclose(o1, o2)

        qubits = converter.qubits
        bitstring = '0' * len(qubits)
        fixed = dict(zip(qubits, bitstring))
        expr, operands = converter.amplitude(bitstring)
        expr1, operands1 = converter.batched_amplitudes(fixed)
        assert expr == expr1
        for o1, o2 in zip(operands, operands1):
            assert np.allclose(o1, o2)
    
    @pytest.mark.parametrize("backend", ARRAY_BACKENDS)
    @pytest.mark.parametrize("dtype", ("float32", "float64", "complex64", "complex128"))
    def test_real_circuit(self, real_circuit, real_circuit_sv, backend, dtype):
        converter = CircuitToEinsum(real_circuit, dtype=dtype, backend=backend)
        expr, operands = converter.state_vector()
        sv = contract(expr, *operands)
        assert TensorBackend.verify_close(sv, real_circuit_sv)
        wrapped_operands = tensor_wrapper.wrap_operands(operands)
        assert utils.get_operands_dtype(wrapped_operands) == dtype
        assert utils.get_operands_package(wrapped_operands) == backend

        n_qubits = sv.ndim
        for pauli_char in 'IXZ': # real Pauli strings, e.g, "II", "XX", "ZZ"
            pauli_string = pauli_char * n_qubits
            expr, operands = converter.expectation(pauli_string)
            exp = contract(expr, *operands)
            exp_ref = PropertyComputeHelper.expectation_from_sv(real_circuit_sv, pauli_string)
            assert TensorBackend.verify_close(exp, exp_ref)
        
        if dtype.startswith("float"):
            with pytest.raises(ValueError) as e:
                expr, operands = converter.expectation('Y' * n_qubits)
            assert "Pauli Y operator" in str(e.value)
    
    @pytest.mark.parametrize("circuit", CircuitMatrix.complexL0())
    @pytest.mark.parametrize("dtype", ("float32", "float64"))
    def test_negative_complex_circuit(self, circuit, dtype):
        with pytest.raises(RuntimeError) as e:
            CircuitToEinsum(circuit, dtype=dtype)
        assert "imaginary part" in str(e.value)


@pytest.fixture(params=QiskitCircuitMatrix.L2(), scope="class")
def qiskit_circuit(request):
    return request.param

@pytest.fixture(scope="class")
def qiskit_sv(qiskit_circuit):
    return CircuitHelper.compute_state_vector(qiskit_circuit)

@pytest.mark.parametrize("option", (
    None, # default (True, True)
    QiskitParserOptions(decompose_gates=True, check_diagonal=False),
    QiskitParserOptions(decompose_gates=False, check_diagonal=True),
    {'decompose_gates': False, 'check_diagonal': True},
))
class TestQiskitCorrectness(BaseCircuitToEinsumTester):

    num_tests_per_task = 3

    def test_state_vector(self, qiskit_circuit, option, qiskit_sv):
        self._test_state_vector(qiskit_circuit, option, qiskit_sv)
    
    def test_amplitude(self, qiskit_circuit, option, qiskit_sv):
        self._test_amplitude(qiskit_circuit, option, qiskit_sv)
    
    def test_batched_amplitudes(self, qiskit_circuit, option, qiskit_sv):
        self._test_batched_amplitudes(qiskit_circuit, option, qiskit_sv)
    
    @pytest.mark.parametrize("lightcone", (True, False))
    def test_reduced_density_matrix(self, qiskit_circuit, option, qiskit_sv, lightcone):
        self._test_reduced_density_matrix(qiskit_circuit, option, qiskit_sv, lightcone)
    
    @pytest.mark.parametrize("lightcone", (True, False))
    def test_expectation(self, qiskit_circuit, option, qiskit_sv, lightcone):
        self._test_expectation(qiskit_circuit, option, qiskit_sv, lightcone)
    
    def test_backend_dtype_consistency(self, qiskit_circuit, option):
        self._test_backend_dtype_consistency(qiskit_circuit, option)
    
    def test_auto_backend(self, qiskit_circuit, option):
        self._test_auto_backend(qiskit_circuit, "complex64", option)


@pytest.fixture(params=CirqCircuitMatrix.L2(), scope="class")
def cirq_circuit(request):
    return request.param

@pytest.fixture(scope="class")
def cirq_sv(cirq_circuit):
    return CircuitHelper.compute_state_vector(cirq_circuit)

@pytest.mark.parametrize("option", (
    None, # default (True,)
    CirqParserOptions(check_diagonal=False),
))
class TestCirqCorrectness(BaseCircuitToEinsumTester):

    num_tests_per_task = 3

    def test_state_vector(self, cirq_circuit, option, cirq_sv):
        self._test_state_vector(cirq_circuit, option, cirq_sv)
    
    def test_amplitude(self, cirq_circuit, option, cirq_sv):
        self._test_amplitude(cirq_circuit, option, cirq_sv)
    
    def test_batched_amplitudes(self, cirq_circuit, option, cirq_sv):
        self._test_batched_amplitudes(cirq_circuit, option, cirq_sv)
    
    @pytest.mark.parametrize("lightcone", (True, False))
    def test_reduced_density_matrix(self, cirq_circuit, option, cirq_sv, lightcone):
        self._test_reduced_density_matrix(cirq_circuit, option, cirq_sv, lightcone)
    
    @pytest.mark.parametrize("lightcone", (True, False))    
    def test_expectation(self, cirq_circuit, option, cirq_sv, lightcone):
        self._test_expectation(cirq_circuit, option, cirq_sv, lightcone)
    
    def test_backend_dtype_consistency(self, cirq_circuit, option):
        self._test_backend_dtype_consistency(cirq_circuit, option)
