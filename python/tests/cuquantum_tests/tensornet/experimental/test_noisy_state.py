# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np

from ..utils.circuit_ifc import QuantumStateTestHelper, PropertyComputeHelper
from ._internal.state_factory import get_random_network_operator, StateFactory
from ._internal.state_matrix import MPSConfigMatrix, NoisyStateMatrix, SimulationConfigMatrix
from ._internal.state_tester import BaseNoisyStateTester
from ._internal.state_utils import compute_noisy_sv, verify_state_sampling, analyze_trajectory_deviation
from ..utils.helpers import _BaseTester

from cuquantum.tensornet.experimental import MPSConfig, TNConfig

NUM_TESTS_PER_CONFIG = 3
NUM_TRAJECTORIES_PER_CONFIG = 10

def skip_unsupported_noisy_state_tests(noisy_factory, config):
    has_general_channel = 'g' in noisy_factory.layers or 'G' in noisy_factory.layers
    if has_general_channel:
        if isinstance(config, TNConfig) or config == {}:
            pytest.skip("TNConfig is not supported for general channel simulation")
        if isinstance(config, MPSConfig):
            is_simple_update = config.gauge_option == 'simple'
        elif isinstance(config, dict):
            is_simple_update = config.get('gauge_option', 'free') == 'simple'
        else:
            raise ValueError(f"Unknown config type: {type(config)}")
        if is_simple_update:
            pytest.skip("Simple update is not supported for general channel simulation")

@pytest.fixture(params=NoisyStateMatrix.L0(), scope="class")
def noisy_factory_L0(request):
    return request.param

@pytest.fixture(params=MPSConfigMatrix.approxConfigsL0(), scope="class")
def approx_mps_config_L0(request):
    return request.param

@pytest.fixture(scope="class")
def noisy_state_results_L0(noisy_factory_L0, approx_mps_config_L0):
    skip_unsupported_noisy_state_tests(noisy_factory_L0, approx_mps_config_L0)
    # NOTE: torch seems to be holding on to the result tensors, so here we force numpy as reference
    force_numpy = noisy_factory_L0.backend.name == 'torch'
    return compute_noisy_sv(noisy_factory_L0, approx_mps_config_L0, return_probabilities=True, force_numpy=force_numpy)

class TestNoisyStateFunctionality(_BaseTester):

    def test_release_operators(self, noisy_factory_L0, approx_mps_config_L0):
        skip_unsupported_noisy_state_tests(noisy_factory_L0, approx_mps_config_L0)
        with noisy_factory_L0.to_network_state(config=approx_mps_config_L0) as state:
            state.compute_output_state(release_operators=True)
            sv = state.compute_state_vector()
            # when release_operator is set to True, 
            # subsequent property calculations should be all corresponding to fixed final state
            n_qubits = sv.ndim
            bitstring = '0' * n_qubits
            amp = state.compute_amplitude(bitstring)
            QuantumStateTestHelper.verify_amplitude(sv, bitstring, amp)

            fixed = {0: '1', 1: '0'}
            batched_amps = state.compute_batched_amplitudes(fixed)
            QuantumStateTestHelper.verify_batched_amplitudes(sv, fixed, batched_amps)

            where = (0, 1)
            rdm = state.compute_reduced_density_matrix(where)
            QuantumStateTestHelper.verify_reduced_density_matrix(sv, where, rdm)
            if set(state.state_mode_extents) == {2} and "complex" in state.dtype:
                operators = {
                    'X' * n_qubits: 0.2,
                    'Y' * n_qubits: 0.3,
                    'Z' * n_qubits: 0.4,
                    'I' * n_qubits: 0.1,
                } 
                exp = state.compute_expectation(operators)
                QuantumStateTestHelper.verify_expectation(sv, operators, exp)
            modes = list(range(n_qubits))
            verify_state_sampling(state, modes, 1000, [sv, ], 3)
    

    NUM_MAX_TRAJECTORIES = 50000
    NUM_TRAJECTORIES_PER_CHECK = 500
    @pytest.mark.parametrize(
        "factory", (StateFactory(4, "float64", "SDuDS", np.random.default_rng(0)),
                    StateFactory(4, "float64", "SDgDS", np.random.default_rng(1)),
                    StateFactory(5, "float64", "SDugS", np.random.default_rng(2)))
    )
    @pytest.mark.parametrize(
        "config", (TNConfig(), 
                   MPSConfig(max_extent=2, gauge_option='simple'), 
                   MPSConfig(max_extent=2, gauge_option='free'))
    )
    def test_distribution(self, factory, config):
        skip_unsupported_noisy_state_tests(factory, config)
        rng = self._get_rng(factory, config, "distribution")
        operator = get_random_network_operator(
            factory.state_dims, 
            rng, 
            factory.backend.name,
            num_repeats=1,
            dtype=factory.dtype, 
        )
        results = compute_noisy_sv(factory, config, return_probabilities=True)
        
        expecs = []
        probs = []
        for p, sv in results:
            expecs.append(PropertyComputeHelper.expectation_from_sv(sv, operator))
            probs.append(p)
        
        expecs = np.asarray(expecs).reshape(1, -1)

        with factory.to_network_state(config=config) as state:
            expec_traj_results = []
            prob_deviation = 1
            for i in range(1, self.NUM_MAX_TRAJECTORIES + 1):
                expec_traj_results.append(state.compute_expectation(operator))
                if i % self.NUM_TRAJECTORIES_PER_CHECK == 0:
                    min_deviation, prob_deviation = analyze_trajectory_deviation(expec_traj_results, expecs, probs)
                    assert np.isclose(min_deviation, 0), f"trajectory results not found in the configuration space, min_deviation: {min_deviation}"
                    if prob_deviation < 0.05:
                        break
                    print(f"num_trajectories: {i}, prob_deviation: {prob_deviation}, adding another {self.NUM_TRAJECTORIES_PER_CHECK} trajectories")
            if prob_deviation < 0.05:
                print(f"Passed after {i} trajectories, {prob_deviation=}")
            else:
                assert False, f"Failed after {i} trajectories, {prob_deviation=}"


@pytest.fixture(params=NoisyStateMatrix.L1(), scope="class")
def noisy_factory_L1(request):
    return request.param

@pytest.fixture(params=MPSConfigMatrix.approxConfigsL1()+SimulationConfigMatrix.exactConfigs(), scope="class")
def noisy_config(request):
    return request.param

@pytest.fixture(scope="class")
def noisy_factory_svs_L1(noisy_factory_L1, noisy_config):
    skip_unsupported_noisy_state_tests(noisy_factory_L1, noisy_config)
    # NOTE: torch seems to be holding on to the result tensors, so here we force numpy as reference
    force_numpy = noisy_factory_L1.backend.name == 'torch'
    return compute_noisy_sv(noisy_factory_L1, noisy_config, return_probabilities=False, force_numpy=force_numpy)

class TestNoisyStateCorrectness(BaseNoisyStateTester):

    def test_state_vector(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_state_vector(noisy_factory_svs_L1, state, NUM_TRAJECTORIES_PER_CONFIG)
    
    def test_amplitude(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        rng = self._get_rng(noisy_factory_L1, "amplitude")
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_amplitude(noisy_factory_svs_L1, state, NUM_TESTS_PER_CONFIG, rng, NUM_TRAJECTORIES_PER_CONFIG)
    
    def test_batched_amplitudes(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        rng = self._get_rng(noisy_factory_L1, "batched_amplitudes")
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_batched_amplitudes(noisy_factory_svs_L1, state, NUM_TESTS_PER_CONFIG, rng, NUM_TRAJECTORIES_PER_CONFIG)

    def test_expectation(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        rng = self._get_rng(noisy_factory_L1, "expectation")
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_expectation(noisy_factory_svs_L1, state, rng, NUM_TRAJECTORIES_PER_CONFIG)

    def test_reduced_density_matrix(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        rng = self._get_rng(noisy_factory_L1, "reduced_density_matrix")
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_reduced_density_matrix(noisy_factory_svs_L1, state, NUM_TESTS_PER_CONFIG, rng, NUM_TRAJECTORIES_PER_CONFIG)

    def test_sampling(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_sampling(noisy_factory_svs_L1, state, NUM_TRAJECTORIES_PER_CONFIG)


