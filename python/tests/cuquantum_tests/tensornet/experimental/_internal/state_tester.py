# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ...utils.helpers import _BaseTester
from ...utils.circuit_ifc import PropertyComputeHelper, CircuitHelper, QuantumStateTestHelper
from .state_factory import get_random_network_operator
from .state_utils import verify_state_sampling
from cuquantum.tensornet.experimental import NetworkState

#TODO: extend norm tests
#TODO: random backend
class BaseStateTester(_BaseTester):
    
    def _test_state_vector(self, state, sv_ref):
        sv = state.compute_state_vector()
        QuantumStateTestHelper.verify_state_vector(sv_ref, sv)
    
    def _test_amplitude(self, state, bitstring, sv_ref):
        amp = state.compute_amplitude(bitstring)
        QuantumStateTestHelper.verify_amplitude(sv_ref, bitstring, amp)

    def _test_batched_amplitudes(self, state, fixed, sv_ref):
        batched_amps = state.compute_batched_amplitudes(fixed)
        if fixed:
            fixed_qubits = list(fixed.keys())
            if not isinstance(fixed_qubits[0], int):
                fixed = {state.state_labels.index(q): bit for q, bit in fixed.items()}
        QuantumStateTestHelper.verify_batched_amplitudes(sv_ref, fixed, batched_amps)

    def _test_expectation(self, state, pauli_strings, sv_ref, **kwargs):
        exp = state.compute_expectation(pauli_strings, **kwargs)
        QuantumStateTestHelper.verify_expectation(sv_ref, pauli_strings, exp)
    
    def _test_reduced_density_matrix(self, state, where, sv_ref, fixed=None, **kwargs):
        rdm = state.compute_reduced_density_matrix(where, fixed=fixed, **kwargs)
        if not isinstance(where[0], int):
            where = [state.state_labels.index(q) for q in where]
        if fixed is None:
            fixed = {}
        else:
            if fixed:
                fixed_qubits = list(fixed.keys())
                if not isinstance(fixed_qubits[0], int):
                    fixed = {state.state_labels.index(q): bit for q, bit in fixed.items()}
        QuantumStateTestHelper.verify_reduced_density_matrix(sv_ref, where, rdm, fixed=fixed)
    
    def _test_sampling(self, state, modes, nshots, sv_ref, max_trial, **kwargs):
        if modes is None or isinstance(modes[0], int):
            normalized_modes = modes
        else:
            normalized_modes = [state.state_labels.index(q) for q in modes]
        verify_state_sampling(state, normalized_modes, nshots, sv_ref, max_trial, **kwargs)


class BaseCircuitStateTester(BaseStateTester):
    def test_state_vector(self, circuit, config, sv):
        with NetworkState.from_circuit(circuit, config=config, backend='numpy') as state:
            self._test_state_vector(state, sv)

    def test_amplitude(self, circuit, config, sv, num_cases):
        rng = self._get_rng(circuit, config, "amplitude")
        n_qubits = len(CircuitHelper.get_qubits(circuit))
        bitstring_iter = CircuitHelper.bitstring_iterator(n_qubits, num_cases, rng)
        with NetworkState.from_circuit(circuit, config=config, backend='numpy') as state:
            for bitstring in bitstring_iter:
                self._test_amplitude(state, bitstring, sv)

    def test_batched_amplitudes(self, circuit, config, sv, num_cases):
        rng = self._get_rng(circuit, config, "batched_amplitudes")
        qubits = CircuitHelper.get_qubits(circuit)
        fixed_iter = CircuitHelper.fixed_iterator(qubits, num_cases, rng)
        with NetworkState.from_circuit(circuit, config=config, backend='numpy') as state:
            for fixed in fixed_iter:
                self._test_batched_amplitudes(state, fixed, sv)

    def test_expectation(self, circuit, config, sv, num_cases):
        rng = self._get_rng(circuit, config, "expectation")
        n_qubits = len(CircuitHelper.get_qubits(circuit))
        pauli_strings_to_test = [
            CircuitHelper.get_random_pauli_strings(n_qubits, 1, rng),
            CircuitHelper.get_random_pauli_strings(n_qubits, 2, rng),
            CircuitHelper.get_random_pauli_strings(n_qubits, 5, rng),
        ]
        if num_cases > 3:
            for _ in range(num_cases - 3):
                nterms = rng.randint(2, 10)
                pauli_strings_to_test.append(CircuitHelper.get_random_pauli_strings(n_qubits, nterms, rng))
        with NetworkState.from_circuit(circuit, config=config, backend='numpy') as state:
            for pauli_strings in pauli_strings_to_test:
                self._test_expectation(state, pauli_strings, sv)

    def test_reduced_density_matrix(self, circuit, config, sv, num_cases):
        rng = self._get_rng(circuit, config, "reduced_density_matrix")
        qubits = CircuitHelper.get_qubits(circuit)
        where_fixed_iter = CircuitHelper.where_fixed_iterator(qubits, num_cases, rng)
        with NetworkState.from_circuit(circuit, config=config, backend='numpy') as state:
            for where, fixed in where_fixed_iter:
                self._test_reduced_density_matrix(state, where, sv, fixed)

    def test_sampling(self, circuit, config, sv, num_cases):
        rng = self._get_rng(circuit, config, "sampling")
        qubits = CircuitHelper.get_qubits(circuit)
        with NetworkState.from_circuit(circuit, config=config, backend='numpy') as state:
            self._test_sampling(state, qubits, 1000, sv, num_cases) # test all qubits sampling
            where_fixed_iter = CircuitHelper.where_fixed_iterator(qubits, num_cases, rng)
            for where, _ in where_fixed_iter:
                self._test_sampling(state, where, 1000, sv, num_cases)


class BaseGenericStateTester(BaseStateTester):

    def test_state_vector(self, factory, config, sv):
        with factory.to_network_state(config=config) as state:
            self._test_state_vector(state, sv)
    
    def test_amplitude(self, factory, config, sv, num_cases):
        rng = self._get_rng(factory, config, "amplitude")
        bitstring_iter = CircuitHelper.bitstring_iterator(factory.state_dims, num_cases, rng)
        with factory.to_network_state(config=config) as state:
            for bitstring in bitstring_iter:
                self._test_amplitude(state, bitstring, sv)
    
    def test_batched_amplitudes(self, factory, config, sv, num_cases):
        rng = self._get_rng(factory, config, "batched_amplitudes")
        state_dims = factory.state_dims
        qudits = list(range(len(state_dims)))
        fixed_iter = CircuitHelper.fixed_iterator(qudits, num_cases, rng, state_dims=state_dims)
        with factory.to_network_state(config=config) as state:
            for fixed in fixed_iter:
                self._test_batched_amplitudes(state, fixed, sv)
    
    def test_expectation(self, factory, config, sv, num_cases):
        rng = self._get_rng(factory, config, "expectation")
        state_dims_set = set(factory.state_dims)
        n_qudits = len(factory.state_dims)
        if len(state_dims_set) == 1 and state_dims_set.pop() == 2 and "complex" in factory.dtype:
            pauli_strings_to_test = [
                CircuitHelper.get_random_pauli_strings(n_qudits, 1, rng),
                CircuitHelper.get_random_pauli_strings(n_qudits, 2, rng),
                CircuitHelper.get_random_pauli_strings(n_qudits, 5, rng),
            ]
            if num_cases > 3:
                for _ in range(num_cases - 3):
                    pauli_strings_to_test.append(
                        CircuitHelper.get_random_pauli_strings(n_qudits, 1, rng)
                    )
        else:
            pauli_strings_to_test = []
        
        with factory.to_network_state(config=config) as state:
            for pauli_strings in pauli_strings_to_test:
                self._test_expectation(state, pauli_strings, sv)
            
            network_operator = get_random_network_operator(
                factory.state_dims, rng, factory.backend.name, dtype=factory.dtype, options=state.options)
            expec = state.compute_expectation(network_operator)
            expec_ref = PropertyComputeHelper.expectation_from_sv(sv, network_operator)
            assert np.allclose(expec, expec_ref)
    
    def test_reduced_density_matrix(self, factory, config, sv, num_cases):
        rng = self._get_rng(factory, config, "reduced_density_matrix")
        qudits = list(range(len(factory.state_dims)))
        where_fixed_iter = CircuitHelper.where_fixed_iterator(qudits, num_cases, rng, state_dims=factory.state_dims)
        with factory.to_network_state(config=config) as state:
            for where, fixed in where_fixed_iter:
                self._test_reduced_density_matrix(state, where, sv, fixed)
    
    def test_sampling(self, factory, config, sv, num_cases):
        rng = self._get_rng(factory, config, "sampling")
        qudits = list(range(len(factory.state_dims)))
        with factory.to_network_state(config=config) as state:
            self._test_sampling(state, qudits, 1000, sv, num_cases) # test all qubits sampling
            where_fixed_iter = CircuitHelper.where_fixed_iterator(qudits, num_cases, rng, state_dims=factory.state_dims)
            for where, _ in where_fixed_iter:
                self._test_sampling(state, where, 1000, sv, num_cases)


class BaseNoisyStateTester(_BaseTester):
    
    def _test_state_vector(self, svs, state, num_trajectories):
        for _ in range(num_trajectories):
            sv = state.compute_state_vector()
            QuantumStateTestHelper.verify_state_vector(svs, sv)
    
    def _test_amplitude(self, svs, state, num_cases, rng, num_trajectories):
        bitstring_iterator = CircuitHelper.bitstring_iterator(state.state_mode_extents, num_cases, rng)
        for bitstring in bitstring_iterator:
            for _ in range(num_trajectories):
                amp = state.compute_amplitude(bitstring)
                QuantumStateTestHelper.verify_amplitude(svs, bitstring, amp)

    def _test_batched_amplitudes(self, svs, state, num_cases, rng, num_trajectories):
        qubits = list(range(len(state.state_mode_extents)))
        fixed_iter = CircuitHelper.fixed_iterator(qubits, num_cases, rng)

        for fixed in fixed_iter:
            for _ in range(num_trajectories):
                batched_amps = state.compute_batched_amplitudes(fixed)
                QuantumStateTestHelper.verify_batched_amplitudes(svs, fixed, batched_amps)
    
    
    def _test_expectation(self, svs, state, rng, num_trajectories):
        operator = get_random_network_operator(
            state.state_mode_extents, rng, state.backend, dtype=state.dtype, num_repeats=1, options=state.options)
        for _ in range(num_trajectories):
            exp = state.compute_expectation(operator)
            QuantumStateTestHelper.verify_expectation(svs, operator, exp)
        

    def _test_reduced_density_matrix(self, svs, state, num_cases, rng, num_trajectories):
        qudits = list(range(len(state.state_mode_extents)))
        where_fixed_iter = CircuitHelper.where_fixed_iterator(qudits, num_cases, rng, state_dims=state.state_mode_extents)
        for where, fixed in where_fixed_iter:
            for _ in range(num_trajectories):
                rdm = state.compute_reduced_density_matrix(where, fixed=fixed)
                QuantumStateTestHelper.verify_reduced_density_matrix(svs, where, rdm, fixed=fixed)
    
    def _test_sampling(self, svs, state, num_trajectories):
        qudits = list(range(len(state.state_mode_extents)))
        for _ in range(num_trajectories):
            verify_state_sampling(state, qudits, 5000, svs, 3)