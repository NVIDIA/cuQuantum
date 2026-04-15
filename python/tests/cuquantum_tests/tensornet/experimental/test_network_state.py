# Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import numpy as np
try:
    import torch
except ImportError:
    torch = None

from nvmath.internal.utils import infer_object_package
from nvmath.internal.tensor_wrapper import wrap_operand

from cuquantum.tensornet import CircuitToEinsum
from cuquantum.tensornet.experimental import NetworkState, MPSConfig, TNConfig, NetworkOperator

from ..utils.data import ARRAY_BACKENDS
from ..utils.helpers import (
    TensorBackend,
    TorchRef,
    _BaseTester,
    assert_gradients_match,
    expectation_as_real,
    get_contraction_tolerance,
)
from ..utils.circuit_ifc import CircuitHelper, QuantumStateTestHelper, PropertyComputeHelper
from ..utils.circuit_matrix import CircuitMatrix

from ._internal.mps_utils import MPS, trim_mps_config, verify_mps_canonicalization, get_mps_tolerance
from ._internal.state_matrix import (
    CircuitStateMatrix,
    ExpectationGradientConfig,
    GenericStateMatrix,
    MPSConfigMatrix,
    NetworkOperatorFactory,
    SimulationConfigMatrix,
)
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
        
        np.random.seed(2)
        state_dims = (2, 5, 2, 2)
        op01 = np.random.randn(2, 5, 2, 5)
        op12 = np.random.randn(5, 2, 5, 2)
        op02 = np.random.randn(2, 2, 2, 2)

        # exact TN simulation
        with NetworkState(state_dims, dtype="float64", config=TNConfig()) as state:
            state.apply_tensor_operator((0, 1), op01)
            state.apply_tensor_operator((1, 2), op12)
            state.apply_tensor_operator((0, 2), op02)
            sv_tn = state.compute_state_vector()
        
        # exact MPS simulation
        with NetworkState(state_dims, dtype="float64", config=MPSConfig()) as state:
            state.apply_tensor_operator((0, 1), op01)
            state.apply_tensor_operator((1, 2), op12)
            state.apply_tensor_operator((0, 2), op02)
            sv_mps = state.compute_state_vector()
        
        tol = get_contraction_tolerance(state.dtype)
        np.testing.assert_allclose(sv_tn, sv_mps, **tol)

    def test_control_values_default(self):
        """Test that control_values=None defaults to 1 for all control modes.
        
        This test verifies the fix for the bug where control_values=None 
        caused a TypeError instead of defaulting to 1 as documented.
        """
        # Create a 3-qubit state
        state_dims = (2, 2, 2)
        
        # X gate (Pauli-X)
        x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        
        # Test 1: control_values=None should work and default to 1
        with NetworkState(state_dims, dtype="complex128") as state:
            # Initialize with identity
            identity = np.eye(2, dtype=np.complex128)
            state.apply_tensor_operator((0,), identity, unitary=True)
            state.apply_tensor_operator((1,), identity, unitary=True)
            state.apply_tensor_operator((2,), identity, unitary=True)
            
            # Apply controlled-X with control_values=None (should default to 1)
            state.apply_tensor_operator(
                (0,), x_gate, 
                control_modes=(1,), 
                control_values=None,  # This was causing TypeError before fix
                unitary=True, 
                immutable=True
            )
            sv_none = state.compute_state_vector()
        
        # Test 2: control_values=(1,) explicit
        with NetworkState(state_dims, dtype="complex128") as state:
            identity = np.eye(2, dtype=np.complex128)
            state.apply_tensor_operator((0,), identity, unitary=True)
            state.apply_tensor_operator((1,), identity, unitary=True)
            state.apply_tensor_operator((2,), identity, unitary=True)
            
            # Apply controlled-X with explicit control_values=(1,)
            state.apply_tensor_operator(
                (0,), x_gate, 
                control_modes=(1,), 
                control_values=(1,),  # Explicit value
                unitary=True, 
                immutable=True
            )
            sv_explicit = state.compute_state_vector()
        
        # Both should produce identical results
        tol = get_contraction_tolerance("complex128")
        QuantumStateTestHelper.verify_state_vector(sv_none, sv_explicit, **tol)
    
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

        qiskit = pytest.importorskip("qiskit")
        
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

    def test_ghz_sampling_large(self):
        """Test sampling from a GHZ circuit with large number of qubits
        
        GHZ state: (|00...0⟩ + |11...1⟩) / √2
        - Only 2 possible outcomes: all-zeros or all-ones
        - Each has 50% probability
        
        Statistical test: With N=10000 samples, 1% tolerance
        """
        backend = self._get_array_framework("test_ghz_sampling_large")

        qiskit = pytest.importorskip("qiskit")

        n_qubits = 26
        circuit = qiskit.QuantumCircuit(n_qubits)
        
        # Create GHZ state: H on qubit 0, then CNOT chain
        circuit.h(0)
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)

        nshots = 10000
        with NetworkState.from_circuit(circuit, backend=backend) as state:
            samples = state.compute_sampling(nshots, seed=42)

        # GHZ state should only produce all-zeros or all-ones
        all_zeros = '0' * n_qubits
        all_ones = '1' * n_qubits
        assert set(samples.keys()).issubset({all_zeros, all_ones}), \
            f"Unexpected bitstrings in GHZ sampling: {set(samples.keys()) - {all_zeros, all_ones}}"

        # Both outcomes should appear (with overwhelming probability for 10000 samples)
        assert all_zeros in samples and all_ones in samples, \
            f"Expected both |0...0⟩ and |1...1⟩ in samples, got: {samples}"

        # Check that distribution is close to 50/50
        total = sum(samples.values())
        p_zeros = samples.get(all_zeros, 0) / total
        p_ones = samples.get(all_ones, 0) / total
        
        assert abs(p_zeros - 0.5) < 0.01, \
            f"GHZ |0...0⟩ probability {p_zeros:.3f} deviates >1% from expected 0.5"
        assert abs(p_ones - 0.5) < 0.01, \
            f"GHZ |1...1⟩ probability {p_ones:.3f} deviates >1% from expected 0.5"
    
    @pytest.mark.parametrize(
        "gauge_option", ('free', 'simple')
    )
    @pytest.mark.parametrize(
        "max_extent", (None, 2)
    )
    def test_mps_output_state_layout_contract(self, gauge_option, max_extent):
        num_qubits = 4
        dtype = 'complex128'
        config = MPSConfig(gauge_option=gauge_option, max_extent=max_extent)

        with NetworkState((2,) * num_qubits, dtype=dtype, config=config) as state:
            rng = np.random.default_rng(42)
            for i in range(num_qubits):
                u = np.linalg.qr(rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))[0]
                state.apply_tensor_operator((i,), u.astype(np.complex128))
            for i in range(num_qubits - 1):
                u = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
                state.apply_tensor_operator((i, i + 1), u.reshape(2, 2, 2, 2).astype(np.complex128))

            mps_tensors = state.compute_output_state()

            for i, t in enumerate(mps_tensors):
                if i == 0:
                    assert t.ndim == 2, f"Site {i}: expected 2 modes, got {t.ndim}"
                elif i == num_qubits - 1:
                    assert t.ndim == 2, f"Site {i}: expected 2 modes, got {t.ndim}"
                else:
                    assert t.ndim == 3, f"Site {i}: expected 3 modes, got {t.ndim}"

                if max_extent is not None:
                    for d in range(t.ndim):
                        assert t.shape[d] <= max(2, max_extent), \
                            f"Site {i} mode {d}: extent {t.shape[d]} exceeds max_extent={max_extent}"

            sv = state.compute_state_vector()
            assert sv.shape == (2,) * num_qubits

    @pytest.mark.parametrize("gauge_option", ('free', 'simple'))
    def test_mps_simple_gauge_correctness(self, gauge_option):
        """Verify MPS output tensors are numerically correct by contracting them
        and comparing against an exact TN state vector.

        Uses a nearest-neighbor circuit on 4 qubits with no truncation, so the
        MPS is exact and we can use tight tolerances.
        """
        num_qubits = 4
        dtype = 'complex128'
        rng = np.random.default_rng(123)

        gates = []
        for i in range(num_qubits):
            u = np.linalg.qr(rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))[0].astype(np.complex128)
            gates.append(((i,), u))
        for i in range(num_qubits - 1):
            u = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0].reshape(2, 2, 2, 2).astype(np.complex128)
            gates.append(((i, i + 1), u))

        with NetworkState((2,) * num_qubits, dtype=dtype, config=TNConfig()) as exact_state:
            for modes, gate in gates:
                exact_state.apply_tensor_operator(modes, gate)
            sv_ref = exact_state.compute_state_vector()

        mps_config = MPSConfig(gauge_option=gauge_option)
        with NetworkState((2,) * num_qubits, dtype=dtype, config=mps_config) as mps_state:
            for modes, gate in gates:
                mps_state.apply_tensor_operator(modes, gate)
            mps_tensors = mps_state.compute_output_state()

            # Contract MPS tensors: T0[k,n] T1[p,k,n] ... TN[p,k]
            result = np.asarray(mps_tensors[0])
            for t in mps_tensors[1:]:
                t_np = np.asarray(t)
                result = np.tensordot(result, t_np, axes=([-1], [0]))
            assert result.shape == (2,) * num_qubits

            tol = get_contraction_tolerance(dtype)
            np.testing.assert_allclose(result, np.asarray(sv_ref), **tol)

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
    @pytest.mark.parametrize("with_control", (False, True))
    def test_update_reuse_correctness(self, config, with_control):
        (state_a, op_two_body_x, op_two_body_diagonal_x), (state_b, op_two_body_y, op_two_body_diagonal_y), operator, two_body_op_ids = create_vqc_states(config, "numpy", with_control=with_control)
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
        
        for i in range(len(two_body_op_ids)):
            tensor_id = two_body_op_ids[i]
            if i == 0 or i == 3:
                state_a.update_tensor_operator(tensor_id, op_two_body_diagonal_y, unitary=False)
                state_b.update_tensor_operator(tensor_id, op_two_body_diagonal_x, unitary=False)
            else:
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
        
        for i in range(len(two_body_op_ids)):
            tensor_id = two_body_op_ids[i]
            if i == 0 or i == 3:
                state_a.update_tensor_operator(tensor_id, op_two_body_diagonal_x, unitary=False)
                state_b.update_tensor_operator(tensor_id, op_two_body_diagonal_y, unitary=False)
            else:
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
    

    @pytest.mark.parametrize("remove_identity", (True, False, 'auto'))
    def test_expectation_from_pauli_strings(self, circuit_L0, exact_config, circuit_exact_sv_L0, remove_identity):
        n_qubits = len(CircuitHelper.get_qubits(circuit_L0))
        with NetworkState.from_circuit(circuit_L0, backend="numpy", config=exact_config) as state:
            n_qubits = state.n
            pauli_strings = CircuitHelper.get_random_pauli_strings(n_qubits, 10, np.random.default_rng(4))
            tn_operator = NetworkOperator.from_pauli_strings(
                pauli_strings, backend="numpy", dtype=state.dtype, options=state.options, remove_identity=remove_identity)

            exp = state.compute_expectation(tn_operator)
            exp_ref = PropertyComputeHelper.expectation_from_sv(circuit_exact_sv_L0, pauli_strings)
            assert TensorBackend.verify_close(exp, exp_ref)


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


class TestExpectationGradient:
    """Test compute_expectation_with_gradients against the TorchRef implementation."""

    def test_expectation_gradient_requires_at_least_one_gradient(self):
        """compute_expectation_with_gradients raises if no operator has gradient=True."""
        state_dims = (2, 2)
        dtype = "complex128"
        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        state.apply_tensor_operator((0,), np.eye(2, dtype=dtype), unitary=True)
        state.apply_tensor_operator((1,), np.eye(2, dtype=dtype), unitary=True)
        with state:
            with pytest.raises(ValueError, match=r"at least one tensor operator applied with gradient=True"):
                state.compute_expectation_with_gradients("ZI", 1.0)

    def test_expectation_gradient_requires_all_gates_unitary(self):
        """compute_expectation_with_gradients raises if any gate is non-unitary (C++ allGatesUnitary check)."""
        from cuquantum.bindings import cutensornet as cutn
        state_dims = (2, 2)
        dtype = "complex128"
        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        state.apply_tensor_operator((0,), np.eye(2, dtype=dtype), unitary=True, gradient=True)
        non_unitary = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=dtype)  # not unitary
        state.apply_tensor_operator((1,), non_unitary, unitary=False, gradient=False)
        state.__enter__()
        try:
            with pytest.raises(cutn.cuTensorNetError, match=r"NOT_SUPPORTED"):
                state.compute_expectation_with_gradients("ZI", 1.0)
        finally:
            try:
                state.__exit__(None, None, None)
            except AttributeError:
                pass  # free() may hit workspace_stream is None when prepare failed

    def test_expectation_gradient_state_norm_adjoint_must_be_none(self):
        """compute_expectation_with_gradients raises if state_norm_adjoint is not None (C++ expects null in this release)."""
        state_dims = (2, 2)
        dtype = "complex128"
        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        state.apply_tensor_operator((0,), np.eye(2, dtype=dtype), unitary=True, gradient=True)
        state.apply_tensor_operator((1,), np.eye(2, dtype=dtype), unitary=True)
        with state:
            with pytest.raises(NotImplementedError, match=r"state_norm_adjoint.*not supported|pass None"):
                state.compute_expectation_with_gradients("ZI", 1.0, state_norm_adjoint=1.0)

    def test_expectation_gradient_return_norm_must_be_false(self):
        """compute_expectation_with_gradients raises if return_norm is not False (norm pointer must be null in this release)."""
        state_dims = (2, 2)
        dtype = "complex128"
        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        state.apply_tensor_operator((0,), np.eye(2, dtype=dtype), unitary=True, gradient=True)
        state.apply_tensor_operator((1,), np.eye(2, dtype=dtype), unitary=True)
        with state:
            with pytest.raises(NotImplementedError, match=r"return_norm.*not supported|pass None"):
                state.compute_expectation_with_gradients("ZI", 1.0, return_norm=True)

    @pytest.mark.parametrize(
        "config",
        (
            ExpectationGradientConfig.L0()
            + ExpectationGradientConfig.L1()
            + ExpectationGradientConfig.L2()
        ),
    )
    def test_expectation_gradient_vs_reference(self, config):
        """Build state from config, compare cutn vs TorchRef. Hamiltonian is either Pauli dict or NetworkOperator."""
        if torch is None:
            pytest.skip("torch is required for expectation gradient reference tests")
        factory = config["factory"]
        state_dims = factory.state_dims
        gate_sequence = factory.get_gate_sequence_for_reference()
        expectation_value_adjoint = config.get("expectation_value_adjoint", 1.0)
        hamiltonian = config.get("hamiltonian")
        if hamiltonian is not None:
            if not isinstance(hamiltonian, (dict, NetworkOperatorFactory)):
                raise TypeError(
                    "config['hamiltonian'] must be a Pauli string dict or a NetworkOperatorFactory, "
                    f"got {type(hamiltonian).__name__}"
                )
            if isinstance(hamiltonian, NetworkOperatorFactory):
                hamiltonian = hamiltonian.build()
        dtype = config["dtype"]
        remove_identity = config.get("remove_identity", None)
        if remove_identity is not None and isinstance(hamiltonian, dict):
            hamiltonian = NetworkOperator.from_pauli_strings(
                hamiltonian, dtype=dtype, backend="numpy", remove_identity=remove_identity,
            )

        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        gradient_tensor_ids = []
        for modes, gate_tensor, requires_grad in gate_sequence:
            tensor_id = state.apply_tensor_operator(modes, gate_tensor, unitary=True, gradient=requires_grad)
            if requires_grad:
                gradient_tensor_ids.append(tensor_id)
        with state:
            exp_cutn, _, gradients_cutn = state.compute_expectation_with_gradients(
                hamiltonian, expectation_value_adjoint
            )

        exp_ref, gradients_ref_list = TorchRef().compute_expectation_with_gradients(
            state_dims,
            gate_sequence,
            hamiltonian,
            dtype=dtype,
            expectation_value_adjoint=expectation_value_adjoint,
        )
        
        tol = get_contraction_tolerance(dtype)
        # Looser tolerance when float64 (cutn vs torch ref can differ in float accumulation)
        exp_cutn_real = expectation_as_real(exp_cutn, dtype)
        exp_ref_arr = expectation_as_real(exp_ref, dtype)
        assert np.allclose(exp_cutn_real, exp_ref_arr, **tol), (exp_cutn_real, exp_ref_arr)
        assert_gradients_match(gate_sequence, gradients_cutn, gradients_ref_list, tol, gradient_tensor_ids=gradient_tensor_ids)

    def test_accumulate_and_update_gradient(self):
        """cutn bindings: 3 backward calls. 
        Same buffer twice (accumulate=0 then 1) -> 2*first; 
        update_tensor_operator_gradient to new buffer; third call -> new buffer has first."""
        import cuquantum
        from cuquantum.bindings import cutensornet as cutn
        import cupy as cp
        import cmath
        import math

        dtype = np.complex128
        data_type = cuquantum.cudaDataType.CUDA_C_64F
        num_qubits = 4
        state_dims = (2,) * num_qubits
        theta = math.pi / 4.0
        inv_sqrt2 = 1.0 / (2.0 ** 0.5)
        H_h = np.array([[1, 1], [1, -1]], dtype=dtype) * inv_sqrt2
        cy, sy = math.cos(theta / 2), math.sin(theta / 2)
        Ry_h = np.array([[cy, -sy], [sy, cy]], dtype=dtype)
        c, s = math.cos(theta / 2), -1j * math.sin(theta / 2)
        Rx_h = np.array([[c, s], [s, c]], dtype=dtype)
        pauli_z_h = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
        pauli_y_h = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype)

        handle = cutn.create()
        stream = 0
        try:
            state = cutn.create_state(handle, cutn.StatePurity.PURE, num_qubits, state_dims, data_type)
            d_H = cp.asarray(H_h)
            d_Ry = cp.asarray(Ry_h)
            d_Rx = cp.asarray(Rx_h)
            d_Z = cp.asarray(pauli_z_h)
            d_Y = cp.asarray(pauli_y_h)
            d_grad_ry = cp.zeros(4, dtype=dtype)
            d_grad_rx = cp.zeros(4, dtype=dtype)
            d_grad_ry_2 = cp.zeros(4, dtype=dtype)
            d_grad_rx_2 = cp.zeros(4, dtype=dtype)

            cutn.state_apply_tensor_operator(handle, state, 1, (0,), d_H.data.ptr, 0, immutable=0, adjoint=0, unitary=1)
            ry_id = cutn.state_apply_tensor_operator_with_gradient(
                handle, state, 1, (0,), d_Ry.data.ptr, 0, 0, 0, 1, d_grad_ry.data.ptr, 0
            )
            cutn.state_apply_tensor_operator(handle, state, 1, (1,), d_H.data.ptr, 0, immutable=0, adjoint=0, unitary=1)
            rx_id = cutn.state_apply_tensor_operator_with_gradient(
                handle, state, 1, (1,), d_Rx.data.ptr, 0, 0, 0, 1, d_grad_rx.data.ptr, 0
            )

            hamiltonian = cutn.create_network_operator(handle, num_qubits, state_dims, data_type)
            num_modes_2 = (1, 1)
            state_modes_yy = [(0,), (1,)]
            cutn.network_operator_append_product(
                handle, hamiltonian, np.complex128(2.0), 2, num_modes_2, state_modes_yy, 0, [d_Y.data.ptr, d_Y.data.ptr]) #YYII
            cutn.network_operator_append_product(
                handle, hamiltonian, np.complex128(3.0), 1, (1,), [(1,)], 0, [d_Z.data.ptr]) #IZII
            cutn.network_operator_append_product(
                handle, hamiltonian, np.complex128(5.0), 1, (1,), [(0,)], 0, [d_Z.data.ptr]) #ZIII  

            expectation = cutn.create_expectation(handle, state, hamiltonian)
            num_hyper = np.array(1, dtype=np.int32)
            cutn.expectation_configure(
                handle, expectation,
                cutn.ExpectationAttribute.CONFIG_NUM_HYPER_SAMPLES,
                num_hyper.ctypes.data, num_hyper.nbytes,
            )
            work_desc = cutn.create_workspace_descriptor(handle)
            max_workspace = (1 << 30)
            cutn.expectation_prepare(handle, expectation, max_workspace, work_desc, stream)
            scratch_size = cutn.workspace_get_memory_size(
                handle, work_desc, cutn.WorksizePref.MIN, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH
            )
            cache_size = cutn.workspace_get_memory_size(
                handle, work_desc, cutn.WorksizePref.MIN, cutn.Memspace.DEVICE, cutn.WorkspaceKind.CACHE
            )
            d_scratch = cp.cuda.alloc(int(scratch_size)) if scratch_size > 0 else None
            d_cache = cp.cuda.alloc(int(cache_size)) if cache_size > 0 else None
            if scratch_size > 0:
                cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, d_scratch.ptr, scratch_size)
            if cache_size > 0:
                cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.CACHE, d_cache.ptr, cache_size)

            exp_val = np.zeros(1, dtype=dtype)
            exp_adjoint = np.array(1.0 + 0.0j, dtype=dtype)

            cutn.expectation_compute_with_gradients_backward(
                handle, expectation, 0, exp_adjoint.ctypes.data, 0, work_desc, exp_val.ctypes.data, 0, stream
            )
            cp.cuda.Stream.null.synchronize()
            grad_ry_first = cp.asnumpy(d_grad_ry.copy()).reshape(2, 2)
            grad_rx_first = cp.asnumpy(d_grad_rx.copy()).reshape(2, 2)

            cutn.expectation_compute_with_gradients_backward(
                handle, expectation, 1, exp_adjoint.ctypes.data, 0, work_desc, exp_val.ctypes.data, 0, stream
            )
            cp.cuda.Stream.null.synchronize()
            grad_ry_twice = cp.asnumpy(d_grad_ry.copy()).reshape(2, 2)
            grad_rx_twice = cp.asnumpy(d_grad_rx.copy()).reshape(2, 2)

            cutn.state_update_tensor_operator_gradient(handle, state, ry_id, d_grad_ry_2.data.ptr)
            cutn.state_update_tensor_operator_gradient(handle, state, rx_id, d_grad_rx_2.data.ptr)

            cutn.expectation_compute_with_gradients_backward(
                handle, expectation, 0, exp_adjoint.ctypes.data, 0, work_desc, exp_val.ctypes.data, 0, stream
            )
            cp.cuda.Stream.null.synchronize()
            grad_ry_after_update = cp.asnumpy(d_grad_ry_2.copy()).reshape(2, 2)
            grad_rx_after_update = cp.asnumpy(d_grad_rx_2.copy()).reshape(2, 2)

            cutn.destroy_workspace_descriptor(work_desc)
            cutn.destroy_expectation(expectation)
            cutn.destroy_network_operator(hamiltonian)
            cutn.destroy_state(state)
        finally:
            cutn.destroy(handle)

        tol = get_contraction_tolerance("complex128")
        assert np.allclose(grad_ry_twice, 2.0 * grad_ry_first, **tol), (
            "Ry gradient after 2nd backward (same buffer, accumulate=1): expected 2*first"
        )
        assert np.allclose(grad_rx_twice, 2.0 * grad_rx_first, **tol), (
            "Rx gradient after 2nd backward (same buffer, accumulate=1): expected 2*first"
        )
        assert np.allclose(grad_ry_after_update, grad_ry_first, **tol), (
            "Ry gradient after update + 3rd backward: expected first (update_tensor_operator_gradient redirected output)"
        )
        assert np.allclose(grad_rx_after_update, grad_rx_first, **tol), (
            "Rx gradient after update + 3rd backward: expected first (update_tensor_operator_gradient redirected output)"
        )


class TestAdjointGateCancellation:
    """G followed by G† must act as identity on a pure state."""

    @staticmethod
    def _random_unitary(dim, rng):
        mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        q, _ = np.linalg.qr(mat)
        return q.astype(np.complex128)

    def test_single_qubit_gate_adjoint_cancels(self):
        num_qubits = 3
        dtype = "complex128"
        rng = np.random.default_rng(12345)

        G = self._random_unitary(2, rng)
        assert not np.allclose(G, G.T), "gate must be non-symmetric to distinguish G† from G*"

        with NetworkState((2,) * num_qubits, dtype=dtype) as ref_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                ref_state.apply_tensor_operator((q,), u, unitary=True)
            sv_ref = ref_state.compute_state_vector()

        rng = np.random.default_rng(12345)
        G = self._random_unitary(2, rng)

        with NetworkState((2,) * num_qubits, dtype=dtype) as test_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                test_state.apply_tensor_operator((q,), u, unitary=True)
            test_state.apply_tensor_operator((0,), G, unitary=True)
            test_state.apply_tensor_operator((0,), G, adjoint=True, unitary=True)
            sv_test = test_state.compute_state_vector()

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(sv_test, sv_ref, **tol,
            err_msg="G† G should cancel: statevectors must match")

    def test_two_qubit_gate_adjoint_cancels(self):
        num_qubits = 3
        dtype = "complex128"
        rng = np.random.default_rng(99)

        G = self._random_unitary(4, rng).reshape(2, 2, 2, 2)

        with NetworkState((2,) * num_qubits, dtype=dtype) as ref_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                ref_state.apply_tensor_operator((q,), u, unitary=True)
            sv_ref = ref_state.compute_state_vector()

        rng = np.random.default_rng(99)
        G = self._random_unitary(4, rng).reshape(2, 2, 2, 2)

        with NetworkState((2,) * num_qubits, dtype=dtype) as test_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                test_state.apply_tensor_operator((q,), u, unitary=True)
            test_state.apply_tensor_operator((0, 1), G, unitary=True)
            test_state.apply_tensor_operator((0, 1), G, adjoint=True, unitary=True)
            sv_test = test_state.compute_state_vector()

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(sv_test, sv_ref, **tol,
            err_msg="G† G should cancel: statevectors must match for 2-qubit gate")

    def test_controlled_gate_adjoint_cancels(self):
        num_qubits = 4
        dtype = "complex128"
        rng = np.random.default_rng(777)

        G = self._random_unitary(2, rng)
        assert not np.allclose(G, G.T), "gate must be non-symmetric to distinguish G† from G*"

        with NetworkState((2,) * num_qubits, dtype=dtype) as ref_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                ref_state.apply_tensor_operator((q,), u, unitary=True)
            sv_ref = ref_state.compute_state_vector()

        rng = np.random.default_rng(777)
        G = self._random_unitary(2, rng)

        with NetworkState((2,) * num_qubits, dtype=dtype) as test_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                test_state.apply_tensor_operator((q,), u, unitary=True)
            test_state.apply_tensor_operator((0,), G, control_modes=(1, 2), unitary=True)
            test_state.apply_tensor_operator((0,), G, control_modes=(1, 2), adjoint=True, unitary=True)
            sv_test = test_state.compute_state_vector()

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(sv_test, sv_ref, **tol,
            err_msg="controlled G† G should cancel: statevectors must match")


class TestNetworkOperator:
    """Tests for NetworkOperator with append_product: mode conventions, backend setup, and correctness."""

    @staticmethod
    def _random_unitary(dim, rng):
        mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        q, _ = np.linalg.qr(mat)
        return q.astype(np.complex128)

    def _compare_direct_vs_product(self, n_qubits, gate_modes, gate_tensor, seed):
        """Compare apply_tensor_operator vs append_product for the same gate."""
        dtype = "complex128"
        I2 = np.eye(2, dtype=np.complex128)

        rng = np.random.default_rng(seed)
        with NetworkState((2,) * n_qubits, dtype=dtype) as ref:
            ref.apply_tensor_operator((0,), I2, immutable=True, unitary=True)
            for q in range(n_qubits):
                u = self._random_unitary(2, rng)
                ref.apply_tensor_operator((q,), u, unitary=True)
            ref.apply_tensor_operator(gate_modes, gate_tensor, unitary=True)
            sv_ref = ref.compute_state_vector()

        rng = np.random.default_rng(seed)
        with NetworkState((2,) * n_qubits, dtype=dtype) as test:
            test.apply_tensor_operator((0,), I2, immutable=True, unitary=True)
            for q in range(n_qubits):
                u = self._random_unitary(2, rng)
                test.apply_tensor_operator((q,), u, unitary=True)
            op = NetworkOperator((2,) * n_qubits, dtype=dtype, options=test.options)
            op.append_product(1.0 + 0j, (gate_modes,), [gate_tensor])
            test.apply_network_operator(op, unitary=True)
            sv_test = test.compute_state_vector()

        return sv_ref, sv_test

    def test_single_qubit_product_matches_direct(self):
        """Single-qubit factor: append_product matches apply_tensor_operator."""
        rng = np.random.default_rng(42)
        G = self._random_unitary(2, rng)
        sv_ref, sv_test = self._compare_direct_vs_product(3, (1,), G, seed=100)
        tol = get_contraction_tolerance("complex128")
        np.testing.assert_allclose(sv_test, sv_ref, **tol,
            err_msg="single-qubit: apply_tensor_operator and append_product should agree")

    def test_multi_qubit_product_matches_direct(self):
        """2-qubit factor: append_product matches apply_tensor_operator."""
        rng = np.random.default_rng(42)
        G = self._random_unitary(4, rng).reshape(2, 2, 2, 2)
        assert not np.allclose(G, G.transpose(1, 0, 3, 2)), \
            "gate must not be invariant under qubit swap"
        sv_ref, sv_test = self._compare_direct_vs_product(3, (0, 1), G, seed=100)
        tol = get_contraction_tolerance("complex128")
        np.testing.assert_allclose(sv_test, sv_ref, **tol,
            err_msg="2-qubit: append_product must match apply_tensor_operator")

    def test_apply_without_prior_gate(self):
        """apply_network_operator as the first operation must not crash during backend setup."""
        rng = np.random.default_rng(2)
        op1 = rng.random((2, 2, 2, 2))
        operator = NetworkOperator((2, 2), dtype='float64')
        operator.append_product(1.0, [(0, 1)], [op1])
        with NetworkState((2, 2), dtype='float64') as state:
            state.apply_network_operator(operator)
            sv = state.compute_state_vector()
        assert sv.shape == (2, 2)

    def test_expectation_multi_qubit_product(self):
        """Expectation with multi-qubit tensor product must match numpy reference."""
        rng = np.random.default_rng(2)
        op0 = rng.random((2, 2))
        op1 = rng.random((2, 2, 2, 2))

        operator = NetworkOperator((2, 2), dtype='float64')
        operator.append_product(1.0, [(0, 1)], [op1])

        vac = np.zeros((2, 2), dtype='float64')
        vac[0, 0] = 1
        sv = np.einsum('ij,Ii->Ij', vac, op0)
        expec_ref = np.einsum('ij,IJij,IJ->', sv, op1, sv.conj())

        with NetworkState((2, 2), dtype='float64') as state:
            state.apply_tensor_operator((0,), op0)
            expec_test = state.compute_expectation(operator)

        np.testing.assert_allclose(expec_test, expec_ref,
            atol=1e-12, rtol=1e-12,
            err_msg="expectation with multi-qubit tensor product must match numpy reference")
