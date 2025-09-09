# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from nvmath.internal import tensor_wrapper, utils

from cuquantum.tensornet import CircuitToEinsum, contract

from .data import ARRAY_BACKENDS
from .circuit_ifc import QuantumStateTestHelper, CircuitHelper
from .helpers import _BaseTester, get_contraction_tolerance


class BaseCircuitToEinsumTester(_BaseTester):

    num_tests_per_task = 3

    def get_config(self, *args, **kwargs):
        rng = self._get_rng(*args, **kwargs)
        backend = rng.choice(ARRAY_BACKENDS).item()
        dtype = rng.choice(('complex64', 'complex128')).item()
        return rng, backend, dtype
    
    def _compute_property(self, converter, method_name, *args, **kwargs):
        expr, operands = getattr(converter, method_name)(*args, **kwargs)
        return contract(expr, *operands)

    def _get_tolerance(self, circuit, dtype):
        framework = utils.infer_object_package(circuit)
        if framework == "qiskit" and dtype not in {"float32", "complex64"}:
            # qiskit reference SV seems computed with a difference precision, therefore we relax the tolerance for double precision comparison
            return {}
        return get_contraction_tolerance(dtype)

    def _test_state_vector(self, circuit, option, state_vector):
        _, backend, dtype = self.get_config(circuit, option, 'state_vector')
        converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend, options=option)
        result = self._compute_property(converter, 'state_vector')
        QuantumStateTestHelper.verify_state_vector(state_vector, result, **self._get_tolerance(circuit, dtype))


    def _test_amplitude(self, circuit, option, state_vector):
        rng, backend, dtype = self.get_config(circuit, option, 'amplitude')
        converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend, options=option)
        num_qubits = state_vector.ndim
        for bitstring in CircuitHelper.bitstring_iterator(num_qubits, self.num_tests_per_task, rng=rng):
            result = self._compute_property(converter, 'amplitude', bitstring)
            QuantumStateTestHelper.verify_amplitude(state_vector, bitstring, result, **self._get_tolerance(circuit, dtype))
    
    def _test_batched_amplitudes(self, circuit, option, state_vector):
        rng, backend, dtype = self.get_config(circuit, option, 'batched_amplitude')
        converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend, options=option)
        qubits = converter.qubits
        for fixed in CircuitHelper.fixed_iterator(qubits, self.num_tests_per_task, rng):
            result = self._compute_property(converter, 'batched_amplitudes', fixed)
            fixed = {qubits.index(q): i for q, i in fixed.items()}
            QuantumStateTestHelper.verify_batched_amplitudes(state_vector, fixed, result, **self._get_tolerance(circuit, dtype))


    def _test_reduced_density_matrix(self, circuit, option, state_vector, lightcone):
        rng, backend, dtype = self.get_config(circuit, option, lightcone, 'reduced_density_matrix')
        converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend, options=option)
        qubits = converter.qubits
        for where, fixed in CircuitHelper.where_fixed_iterator(qubits, self.num_tests_per_task, rng):
            result = self._compute_property(converter, 'reduced_density_matrix', where, fixed=fixed, lightcone=lightcone)
            where = [qubits.index(q) for q in where]
            fixed = {qubits.index(q): i for q, i in fixed.items()}
            QuantumStateTestHelper.verify_reduced_density_matrix(state_vector, where, result, fixed=fixed, **self._get_tolerance(circuit, dtype))
    
    def _test_expectation(self, circuit, option, state_vector, lightcone):
        rng, backend, dtype = self.get_config(circuit, option, lightcone, 'expectation')
        converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend, options=option)
        num_qubits = len(converter.qubits)
        pauli_strings = ['I' * num_qubits] + list(CircuitHelper.get_random_pauli_strings(num_qubits, self.num_tests_per_task, rng=rng).keys())
        for pauli_string in pauli_strings:
            result = self._compute_property(converter, 'expectation', pauli_string, lightcone=lightcone)
            QuantumStateTestHelper.verify_expectation(state_vector, pauli_string, result, **self._get_tolerance(circuit, dtype))
    
    def _test_backend_dtype_consistency(self, circuit, option):
        converters = {}
        for backend in ARRAY_BACKENDS:
            for dtype in ('complex64', 'complex128'):
                converters[f"{backend}-{dtype}"] = CircuitToEinsum(circuit, dtype=dtype, backend=backend, options=option)
        base_converter = converters.pop('numpy-complex128')
        bitstring = '0'*len(base_converter.qubits)
        expr_np, operands_np = base_converter.amplitude(bitstring)

        for test_params, converter in converters.items():
            backend, dtype = test_params.split('-')
            expr, operands = converter.amplitude(bitstring)
            assert expr == expr_np
            wrapped_operands = tensor_wrapper.wrap_operands(operands)
            assert utils.get_operands_package(wrapped_operands) == backend
            assert utils.get_operands_dtype(wrapped_operands) == dtype

            if backend.startswith('torch'):
                converted_np_operands = [o.cpu().numpy() for o in operands]
            elif backend.startswith('cupy'):
                converted_np_operands = [o.get() for o in operands]
            elif backend.startswith('numpy'):
                converted_np_operands = operands
            else:
                raise ValueError(f"Unsupported backend: {backend}")

            for o1, o2 in zip(converted_np_operands, operands_np):
                if not np.allclose(o1, o2):
                    max_diff = np.max(np.abs(o1 - o2))
                    raise ValueError(f"operands for numpy complex128 not matching {backend} {dtype} for amplitude computation, {max_diff=}")

    def _test_auto_backend(self, circuit, dtype, option):
        expected_package = "cupy" if "cupy" in ARRAY_BACKENDS else "numpy"
        for kwargs in ({}, {'backend': 'auto'}):
            converter = CircuitToEinsum(circuit, dtype=dtype, options=option, **kwargs)
            operands = converter.state_vector()[1]
            wrapped_operands = tensor_wrapper.wrap_operands(operands)
            package = utils.get_operands_package(wrapped_operands)
            assert package == expected_package