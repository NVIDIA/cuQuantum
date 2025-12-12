# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np

from nvmath.internal.utils import infer_object_package
from nvmath.internal.tensor_wrapper import wrap_operand

from cuquantum.tensornet import CircuitToEinsum
from cuquantum.tensornet.experimental import NetworkState, MPSConfig, TNConfig

from ..utils.data import ARRAY_BACKENDS
from ..utils.helpers import TensorBackend, _BaseTester, get_contraction_tolerance
from ..utils.circuit_ifc import CircuitHelper, QuantumStateTestHelper, PropertyComputeHelper
from ..utils.circuit_matrix import CircuitMatrix

from ._internal.mps_utils import MPS, trim_mps_config, verify_mps_canonicalization, get_mps_tolerance
from ._internal.state_matrix import CircuitStateMatrix, GenericStateMatrix, SimulationConfigMatrix, MPSConfigMatrix
from ._internal.state_tester import BaseCircuitStateTester, BaseGenericStateTester
from ._internal.state_factory import apply_factory_sequence, create_vqc_states, get_random_network_operator, StateFactory

NUM_TESTS_PER_CONFIG = 3

@pytest.fixture(params=CircuitStateMatrix.L0(), scope="class")
def circuit_L0(request):
    return request.param

@pytest.fixture(scope="class")
def circuit_exact_sv_L0(circuit_L0):
    return CircuitHelper.compute_state_vector(circuit_L0)

@pytest.fixture(params=SimulationConfigMatrix.exactConfigs(), scope="class")
def exact_config(request):
    return request.param

@pytest.fixture(params=CircuitMatrix.realL0(), scope="class")
def real_circuit_L0(request):
    return request.param

@pytest.fixture(scope="class")
def real_circuit_exact_sv_L0(real_circuit_L0):
    return CircuitHelper.compute_state_vector(real_circuit_L0)

@pytest.fixture(params=CircuitMatrix.complexL0(), scope="class")
def complex_circuit_L0(request):
    return request.param

class TestNetworkStateBasicFunctionality(_BaseTester):

    def test_from_circuit(self, circuit_L0, circuit_exact_sv_L0):
        backend = self._get_array_framework(circuit_L0, "from_circuit")
        with NetworkState.from_circuit(circuit_L0, backend=backend) as state:
            sv = state.compute_state_vector()
            tol = get_contraction_tolerance(state.dtype)
            QuantumStateTestHelper.verify_state_vector(sv, circuit_exact_sv_L0, **tol)
    
    @pytest.mark.parametrize(
        "kwargs", ({}, {'backend': 'auto'}),
    )
    def test_auto_backend(self, circuit_L0, kwargs):
        with NetworkState.from_circuit(circuit_L0, **kwargs) as state:
            rdm = state.compute_reduced_density_matrix((0,))
            package = infer_object_package(rdm)
            if "cupy" in ARRAY_BACKENDS:
                assert package == "cupy"
            else:
                assert package == "numpy"
    
    @pytest.mark.parametrize(
        "dtype", ('complex64', 'complex128')
    )
    def test_dtype(self, circuit_L0, circuit_exact_sv_L0, dtype):
        backend = self._get_array_framework(circuit_L0, dtype)
        with NetworkState.from_circuit(circuit_L0, backend=backend, dtype=dtype) as state:
            sv = state.compute_state_vector()
            tol = get_contraction_tolerance(state.dtype)
            QuantumStateTestHelper.verify_state_vector(sv, circuit_exact_sv_L0, **tol)
            assert wrap_operand(sv).dtype == dtype
    
    def test_from_converter(self, circuit_L0, circuit_exact_sv_L0):
        backend = self._get_array_framework(circuit_L0, "from_converter")
        converter = CircuitToEinsum(circuit_L0, backend=backend)
        with NetworkState.from_converter(converter) as state:
            sv = state.compute_state_vector()
            tol = get_contraction_tolerance(state.dtype)
            QuantumStateTestHelper.verify_state_vector(sv, circuit_exact_sv_L0, **tol)
    
    def test_config(self, circuit_L0, exact_config, circuit_exact_sv_L0):
        backend = self._get_array_framework(circuit_L0, exact_config)
        with NetworkState.from_circuit(circuit_L0, config=exact_config, backend=backend) as state:
            bitstring = '0'* state.n
            amp = state.compute_amplitude(bitstring)
            tol = get_contraction_tolerance(state.dtype)
            QuantumStateTestHelper.verify_amplitude(circuit_exact_sv_L0, bitstring, amp, **tol)

    @pytest.mark.parametrize("backend", ARRAY_BACKENDS)
    def test_backend(self, circuit_L0, backend, circuit_exact_sv_L0):
        with NetworkState.from_circuit(circuit_L0, backend=backend) as state:
            rdm = state.compute_reduced_density_matrix((0,1))
            assert infer_object_package(rdm) == backend
            tol = get_contraction_tolerance(state.dtype)
            QuantumStateTestHelper.verify_reduced_density_matrix(circuit_exact_sv_L0, (0,1), rdm, **tol)
    
    def test_qudits(self):
        state_dims = (2, 3)
        op = np.random.randn(*state_dims, *state_dims)
        sv_ref = op[:, :, 0,0]
        with NetworkState(state_dims, dtype="float64") as state:
            state.apply_tensor_operator((0, 1), op)
            sv = state.compute_state_vector()
            tol = get_contraction_tolerance(state.dtype)
            QuantumStateTestHelper.verify_state_vector(sv, sv_ref, **tol)
    
    def test_batched_amplitudes_usage(self, exact_config):
        state = NetworkState((2, 2), dtype='float64', config=exact_config)
        rng = np.random.default_rng(2019)
        op = rng.random((2, 2, 2, 2))
        state.apply_tensor_operator((0, 1), op)

        tol = get_contraction_tolerance(state.dtype)

        with state:
            sv = state.compute_state_vector()
            sv1 = state.compute_batched_amplitudes({})
            QuantumStateTestHelper.verify_state_vector(sv, sv1, **tol)

            amp = state.compute_amplitude('01')
            amp1 = state.compute_batched_amplitudes({0:0, 1:1})
            assert np.allclose(amp, amp1, **tol)
            assert np.allclose(sv[0,1], amp, **tol)
            assert np.allclose(sv1[0,1], amp1, **tol)
        
    def test_large_circuit_sampling(self):
        backend = self._get_array_framework("test_large_circuit_sampling")

        try:
            import qiskit
        except ImportError:
            pytest.skip("qiskit not installed")
        
        # a special case with large number of qubits and 16 non-zero bitstring output in the final state
        qubit_count = 20
        depth = qubit_count // 5

        circuit = qiskit.QuantumCircuit(qubit_count)
        qubits = circuit.qubits
        for i in range(0, depth):
            circuit.h(qubits[i * 5])
            for j in range(4):
                circuit.cx(qubits[i * 5+j], qubits[i * 5+j+1])

        nshots = 10000
        with NetworkState.from_circuit(circuit, backend=backend) as state:
            samples_1 = state.compute_sampling(nshots)
            samples_2 = state.compute_sampling(nshots, seed=123)
            assert len(samples_1) == len(samples_2) == 16
    
    @pytest.mark.parametrize("factory", GenericStateMatrix.L1())
    def test_mps_release_operators(self, factory):
        mps_config = MPSConfig(max_extent=4, rel_cutoff=1e-1, gauge_option='free')
        num_operands = len(factory.sequence)
        #############################################################
        # Case I. NetworkState with release_operators in the middle #
        #############################################################

        state = NetworkState(factory.state_dims, dtype=factory.dtype, config=mps_config)
        if factory.initial_mps_dim is not None:
            state.set_initial_mps(factory.get_initial_state())
        # apply the first half operators
        tensor_ids_first_half = set(apply_factory_sequence(state, factory.sequence[:num_operands//2]))
        tensors_0 = state.compute_output_state(release_operators=True)
        # create a copy as initial guess for another NetworkState object
        try:
            tensors_0 = [o.copy() for o in tensors_0] 
        except AttributeError:
            tensors_0 = [o.clone() for o in tensors_0] # torch
        
        # Apply the second half
        tensor_ids_second_half = set(apply_factory_sequence(state, factory.sequence[num_operands//2:]))
        # make sure that there is no overlap in the output tensor ids
        assert not tensor_ids_first_half.intersection(tensor_ids_second_half)
        with state:
            sv0 = state.compute_state_vector()
        
        #######################################################
        # Reference I. NetworkState without release_operators #
        #######################################################
        with factory.to_network_state(config=mps_config) as reference_state:
            sv1 = reference_state.compute_state_vector()
        
        ####################################################
        # Reference II. NetworkState with initial state    #
        ####################################################
        with NetworkState(factory.state_dims, dtype=factory.dtype, config=mps_config) as new_state:
            new_state.set_initial_mps(tensors_0)
            # Apply the second half operators
            apply_factory_sequence(new_state, factory.sequence[num_operands//2:])
            sv2 = new_state.compute_state_vector()
        
        tol = get_mps_tolerance(factory.dtype)
        QuantumStateTestHelper.verify_state_vector(sv0, sv1, **tol)
        QuantumStateTestHelper.verify_state_vector(sv0, sv2, **tol)

    @pytest.mark.parametrize(
        "config", ({}, {'max_extent': 2}, {'rel_cutoff': 0.12, 'gauge_option': 'simple'})
    )
    @pytest.mark.parametrize("with_control", (True, False))
    def test_update_reuse_correctness(self, config, with_control):
        (state_a, op_two_body_x), (state_b, op_two_body_y), operator, two_body_op_ids = create_vqc_states(config, "numpy", with_control=with_control)
        if isinstance(state_a.config, TNConfig):
            tolerance = get_contraction_tolerance("complex128")
        else:
            tolerance = get_mps_tolerance("complex128")
            if not state_a.config._is_fixed_extent_truncation():
                mps0 = state_a.compute_output_state()
                mps1 = state_b.compute_output_state()
                # for MPS with value based truncation, make sure that the test case is designed such that the output MPS have different shapes
                assert any(o0.shape != o1.shape for o0, o1 in zip(mps0, mps1))

        original_expec = []
        for state in [state_a, state_b]:
            e, norm = state.compute_expectation(operator, return_norm=True)
            original_expec.append(e/norm)
        
        for tensor_id in two_body_op_ids:
            state_a.update_tensor_operator(tensor_id, op_two_body_y, unitary=False)
            state_b.update_tensor_operator(tensor_id, op_two_body_x, unitary=False)
        
        updated_expec = []
        for state in [state_b, state_a]:
            e, norm = state.compute_expectation(operator, return_norm=True)
            updated_expec.append(e/norm)

        for e1, e2 in zip(original_expec, updated_expec):
            assert np.allclose(e1, e2, **tolerance)
        
        # we here first perform expectation check and then state vector check as caching in 24.08 is only activated for one compute object at one time.
        original_sv = []
        for state in [state_b, state_a]:
            original_sv.append(state.compute_state_vector())
        
        for tensor_id in two_body_op_ids:
            state_a.update_tensor_operator(tensor_id, op_two_body_x, unitary=False)
            state_b.update_tensor_operator(tensor_id, op_two_body_y, unitary=False)
        
        updated_sv = []
        for state in [state_a, state_b]:
            updated_sv.append(state.compute_state_vector())
            state.free()

        for sv1, sv2 in zip(original_sv, updated_sv):
            assert np.allclose(sv1, sv2, **tolerance)
    
    @pytest.mark.parametrize("factory", GenericStateMatrix.L0())
    @pytest.mark.parametrize("config", ({}, {'max_extent': 2}, {'rel_cutoff': 0.12, 'gauge_option': 'simple'}))
    def test_return_norm(self, factory, config):
        rng = self._get_rng(factory, config, "return_norm")
        with factory.to_network_state(config=config) as state:
            sv, norm = state.compute_state_vector(return_norm=True)
            ndim = sv.ndim
            backend = TensorBackend.from_array(sv)
            norm0 = TensorBackend.to_numpy(backend.norm(sv) ** 2)
            assert np.isclose(norm, norm0)
            norm = state.compute_amplitude('0'*ndim, return_norm=True)[1]
            assert np.isclose(norm, norm0)
            norm = state.compute_batched_amplitudes({0:'0'}, return_norm=True)[1]
            assert np.isclose(norm, norm0)
            operator = get_random_network_operator(factory.state_dims, rng, backend.name, num_repeats=1, dtype=state.dtype, options=state.options)
            norm = state.compute_expectation(operator, return_norm=True)[1]
            assert np.isclose(norm, norm0)


    @pytest.mark.parametrize("factory", GenericStateMatrix.L0())
    @pytest.mark.parametrize("config", ({'max_extent': 2, 'canonical_center': 0}, {'rel_cutoff': 0.12, 'gauge_option': 'simple', 'canonical_center': 2}))
    def test_canonical_center(self, factory, config):
        with factory.to_network_state(config=config) as state:
            mps_tensors = state.compute_output_state()
            canonical_center = config.get('canonical_center')
            assert verify_mps_canonicalization(mps_tensors, canonical_center)

    @pytest.mark.parametrize("dtype", ('complex64', 'complex128'))
    def test_double_init(self, dtype):
        num_qubits, bond_dim = 4, 2
        rng = self._get_rng(dtype, "double_init")
        factory = StateFactory(
            num_qubits,
            dtype,
            layers="",
            rng=rng,
            initial_mps_dim=bond_dim
        )
        tolerance = get_mps_tolerance(dtype)

        with factory.to_network_state() as state:
            sv0 = factory.compute_state_vector()
            sv1 = state.compute_state_vector()
            QuantumStateTestHelper.verify_state_vector(sv0, sv1, **tolerance)

            # reset factory to create a new initial MPS
            factory.psi = None
            factory._sequence =[]
            initial_mps_tensors = factory.get_initial_state()
            state.set_initial_mps(initial_mps_tensors)
            sv2_ref = factory.compute_state_vector()
            sv2 = state.compute_state_vector()
            QuantumStateTestHelper.verify_state_vector(sv2_ref, sv2, **tolerance)
    
    @pytest.mark.parametrize("backend", ARRAY_BACKENDS)
    @pytest.mark.parametrize("dtype", ("float32", "float64", "complex64", "complex128"))
    def test_real_circuit(self, real_circuit_L0, real_circuit_exact_sv_L0, backend, dtype):
        with NetworkState.from_circuit(real_circuit_L0, backend=backend, dtype=dtype) as state:
            sv = state.compute_state_vector()
            wrapped_sv = wrap_operand(sv)
            assert wrapped_sv.dtype == dtype
            assert wrapped_sv.name == backend
            tol = get_contraction_tolerance(state.dtype)
            QuantumStateTestHelper.verify_state_vector(sv, real_circuit_exact_sv_L0, **tol)
            nqubits = sv.ndim
            pauli_strings = {
                'I' * nqubits: 0.2,
                'X' * nqubits: 0.3,
                'Z' * nqubits: 0.5,
            }
            exp = state.compute_expectation(pauli_strings)
            exp_ref = PropertyComputeHelper.expectation_from_sv(real_circuit_exact_sv_L0, pauli_strings)
            assert TensorBackend.verify_close(exp, exp_ref)

            if dtype.startswith('float'):
                with pytest.raises(ValueError) as e:
                    state.compute_expectation('Y'*nqubits)
                assert "Pauli Y operator" in str(e.value)
    
    @pytest.mark.parametrize("circuit", CircuitMatrix.complexL0())
    @pytest.mark.parametrize("backend", ARRAY_BACKENDS)
    @pytest.mark.parametrize("dtype", ("float32", "float64"))
    def test_negative_complex_circuit(self, circuit, backend, dtype):
        with pytest.raises(RuntimeError) as e:
            with NetworkState.from_circuit(circuit, backend=backend, dtype=dtype) as state:
                pass
            assert "imaginary part" in str(e.value)
    
    def test_mps_with_fixed_bond_truncation(self):
        num_qubits = 6
        num_double_layers = 2

        config = MPSConfig(max_extent=3)
        state = NetworkState((2, ) * num_qubits, dtype='float64', config=config)

        for i in range(num_qubits):
            state.apply_tensor_operator((i,), np.random.random([2, 2]))

        for _ in range(num_double_layers):
            for i in range(2):
                for j in range(i, num_qubits-1, 2):
                    state.apply_tensor_operator((j, j+1), np.random.random([2, 2, 2, 2]))

        state.apply_tensor_operator((1, 3), np.random.random([2, 2, 2, 2]))

        with state:
            mps = state.compute_output_state()
            assert mps is not None


@pytest.fixture(params=CircuitStateMatrix.L1(), scope="class")
def circuit_L1(request):
    return request.param

@pytest.fixture(scope="class")
def circuit_exact_sv_L1(circuit_L1):
    return CircuitHelper.compute_state_vector(circuit_L1)


class TestExactCircuitSimulation(BaseCircuitStateTester):

    def test_state_vector(self, circuit_L1, exact_config, circuit_exact_sv_L1):
        super().test_state_vector(circuit_L1, exact_config, circuit_exact_sv_L1)

    def test_amplitude(self, circuit_L1, exact_config, circuit_exact_sv_L1):
        super().test_amplitude(circuit_L1, exact_config, circuit_exact_sv_L1, NUM_TESTS_PER_CONFIG)

    def test_batched_amplitudes(self, circuit_L1, exact_config, circuit_exact_sv_L1):
        super().test_batched_amplitudes(circuit_L1, exact_config, circuit_exact_sv_L1, NUM_TESTS_PER_CONFIG)

    def test_expectation(self, circuit_L1, exact_config, circuit_exact_sv_L1):
        super().test_expectation(circuit_L1, exact_config, circuit_exact_sv_L1, NUM_TESTS_PER_CONFIG)

    def test_reduced_density_matrix(self, circuit_L1, exact_config, circuit_exact_sv_L1):
        super().test_reduced_density_matrix(circuit_L1, exact_config, circuit_exact_sv_L1, NUM_TESTS_PER_CONFIG)

    def test_sampling(self, circuit_L1, exact_config, circuit_exact_sv_L1):
        super().test_sampling(circuit_L1, exact_config, circuit_exact_sv_L1, NUM_TESTS_PER_CONFIG)



@pytest.fixture(params=GenericStateMatrix.L1(), scope="class")
def factory_L1(request):
    return request.param

@pytest.fixture(scope="class")
def sv_factory_L1(factory_L1):
    result = factory_L1.compute_state_vector()
    if factory_L1.backend.name == 'torch':
        result = TensorBackend.to_numpy(result)
    return result


class TestExactGenericState(BaseGenericStateTester):

    def test_state_vector(self, factory_L1, exact_config, sv_factory_L1):
        super().test_state_vector(factory_L1, exact_config, sv_factory_L1)
    
    def test_amplitude(self, factory_L1, exact_config, sv_factory_L1):
        super().test_amplitude(factory_L1, exact_config, sv_factory_L1, NUM_TESTS_PER_CONFIG)
    
    def test_batched_amplitudes(self, factory_L1, exact_config, sv_factory_L1):
        super().test_batched_amplitudes(factory_L1, exact_config, sv_factory_L1, NUM_TESTS_PER_CONFIG)
    
    def test_expectation(self, factory_L1, exact_config, sv_factory_L1):
        super().test_expectation(factory_L1, exact_config, sv_factory_L1, NUM_TESTS_PER_CONFIG)
    
    def test_reduced_density_matrix(self, factory_L1, exact_config, sv_factory_L1):
        super().test_reduced_density_matrix(factory_L1, exact_config, sv_factory_L1, NUM_TESTS_PER_CONFIG)
    
    def test_sampling(self, factory_L1, exact_config, sv_factory_L1):
        super().test_sampling(factory_L1, exact_config, sv_factory_L1, NUM_TESTS_PER_CONFIG)
    

@pytest.fixture(params=CircuitStateMatrix.L2(), scope="class")
def circuit_L2(request):
    return request.param

@pytest.fixture(params=MPSConfigMatrix.approxConfigsL2(), scope="class")
def approx_mps_config(request):
    return request.param

@pytest.fixture(scope="class")
def circuit_approx_sv_L2(circuit_L2, approx_mps_config):
    reduced_config = trim_mps_config(approx_mps_config)
    # This is to provide a reference for the MPS based state vector computation, 
    # which can be fixed to numpy backend
    my_mps = MPS.from_circuit(circuit_L2, "numpy", **reduced_config)
    return my_mps.compute_state_vector()
    
class TestApproxCircuitSimulation(BaseCircuitStateTester):
    def test_state_vector(self, circuit_L2, approx_mps_config, circuit_approx_sv_L2):
        super().test_state_vector(circuit_L2, approx_mps_config, circuit_approx_sv_L2)

    def test_amplitude(self, circuit_L2, approx_mps_config, circuit_approx_sv_L2):
        super().test_amplitude(circuit_L2, approx_mps_config, circuit_approx_sv_L2, NUM_TESTS_PER_CONFIG)
    
    def test_batched_amplitudes(self, circuit_L2, approx_mps_config, circuit_approx_sv_L2):
        super().test_batched_amplitudes(circuit_L2, approx_mps_config, circuit_approx_sv_L2, NUM_TESTS_PER_CONFIG)

    def test_expectation(self, circuit_L2, approx_mps_config, circuit_approx_sv_L2):
        super().test_expectation(circuit_L2, approx_mps_config, circuit_approx_sv_L2, NUM_TESTS_PER_CONFIG)

    def test_reduced_density_matrix(self, circuit_L2, approx_mps_config, circuit_approx_sv_L2):
        super().test_reduced_density_matrix(circuit_L2, approx_mps_config, circuit_approx_sv_L2, NUM_TESTS_PER_CONFIG)
    
    def test_sampling(self, circuit_L2, approx_mps_config, circuit_approx_sv_L2):
        super().test_sampling(circuit_L2, approx_mps_config, circuit_approx_sv_L2, NUM_TESTS_PER_CONFIG)


@pytest.fixture(params=GenericStateMatrix.L2(), scope="class")
def factory_L2(request):
    return request.param

@pytest.fixture(scope="class")
def factory_approx_sv_L2(factory_L2, approx_mps_config):
    reduced_config = trim_mps_config(approx_mps_config)
    my_mps = MPS.from_factory(factory_L2, **reduced_config)
    result = my_mps.compute_state_vector()
    if factory_L2.backend.name == 'torch':
        result = TensorBackend.to_numpy(result)
    return result

class TestApproxGenericState(BaseGenericStateTester):
    def test_state_vector(self, factory_L2, approx_mps_config, factory_approx_sv_L2):
        super().test_state_vector(factory_L2, approx_mps_config, factory_approx_sv_L2)
    
    def test_amplitude(self, factory_L2, approx_mps_config, factory_approx_sv_L2):
        super().test_amplitude(factory_L2, approx_mps_config, factory_approx_sv_L2, NUM_TESTS_PER_CONFIG)
    
    def test_batched_amplitudes(self, factory_L2, approx_mps_config, factory_approx_sv_L2):
        super().test_batched_amplitudes(factory_L2, approx_mps_config, factory_approx_sv_L2, NUM_TESTS_PER_CONFIG)
    
    def test_expectation(self, factory_L2, approx_mps_config, factory_approx_sv_L2):
        super().test_expectation(factory_L2, approx_mps_config, factory_approx_sv_L2, NUM_TESTS_PER_CONFIG)
    
    def test_reduced_density_matrix(self, factory_L2, approx_mps_config, factory_approx_sv_L2):
        super().test_reduced_density_matrix(factory_L2, approx_mps_config, factory_approx_sv_L2, NUM_TESTS_PER_CONFIG)
    
    def test_sampling(self, factory_L2, approx_mps_config, factory_approx_sv_L2):
        super().test_sampling(factory_L2, approx_mps_config, factory_approx_sv_L2, NUM_TESTS_PER_CONFIG)
