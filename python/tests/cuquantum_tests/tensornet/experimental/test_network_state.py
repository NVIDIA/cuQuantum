# Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import pytest
import types

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from nvmath.internal.utils import infer_object_package
from nvmath.internal.tensor_wrapper import wrap_operand

from cuquantum.tensornet import CircuitToEinsum
from cuquantum.tensornet._internal.helpers import _get_backend_asarray_func
from cuquantum.tensornet.experimental import NetworkState, MPSConfig, TNConfig, NetworkOperator
from cuquantum.tensornet.experimental._internal.network_state_utils import (
    EXACT_MPS_EXTENT_LIMIT, get_mps_output_extents)
from cuquantum.bindings import cutensornet as cutn

from ..utils.data import ARRAY_BACKENDS
from ..utils.helpers import (
    TensorBackend,
    TorchRef,
    TorchRefExplicitAdjoints,
    _BaseTester,
    assert_gradients_match,
    assert_torch_gate_grads_match_cutn,
    build_torch_network_state_for_exp_grad,
    build_torch_network_state_pair_for_exp_grad,
    expectation_as_real,
    get_contraction_tolerance,
    get_expectation_gradient_tolerance,
    norm_as_real,
    prepare_expectation_gradient_hamiltonian,
)
from ..utils.circuit_ifc import CircuitHelper, QuantumStateTestHelper, PropertyComputeHelper
from ..utils.circuit_matrix import CircuitMatrix

from ._internal.mps_utils import MPS, trim_mps_config, verify_mps_canonicalization, get_mps_tolerance
from ._internal.state_matrix import (
    CircuitStateMatrix,
    EOverNDenominatorTooSmall,
    ExpectationGradientConfig,
    GenericStateMatrix,
    MixedGenericStateMatrix,
    MPSConfigMatrix,
    NetworkOperatorFactory,
    SimulationConfigMatrix,
    create_state_factory,
    expectation_gradient_L0_torch,
    expectation_gradient_L1_torch,
    expectation_gradient_L2_torch,
    expectation_gradient_loss_factory,
)
from ._internal.state_tester import BaseCircuitStateTester, BaseGenericStateTester, BaseMixedGenericStateTester
from ._internal.state_factory import apply_factory_sequence, create_vqc_states, get_random_network_operator, StateFactory
from ._internal.state_utils import verify_state_sampling

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

# --------------------------------------------------------------------------
# Helpers for the MPS bond-minimality tests in TestNetworkStateBasicFunctionality.
#
# A bond is "overcomplete" when its dimension exceeds min(chi_left * d,
# d * chi_right); the surplus directions are null padding. That is the invariant
# areMPSBondExtentsValid() enforces at the ProjectionMPS entry point and that
# MatrixProductState relies on as a precondition. Sequential value-based
# truncation violates it: a relative cutoff decides per bond against that bond's
# own largest singular value, but the discarded Schmidt component is global, so
# discarding it at an inner bond retroactively lowers the rank across an outer
# bond that was already finalized.
#
# The engineered state's weak branch (_BM_DELTA = 0.085) clears the 10% relative
# threshold at bond 3 (max 0.7045) and misses it at bond 2 (max 0.9964). No RNG.
# --------------------------------------------------------------------------

_BM_NUM_QUBITS = 6

_BM_DELTA = 0.085

def _build_state():
    chi1 = np.zeros((2, 2, 2), dtype=np.complex128)
    chi1[0, 0, 0] = chi1[0, 1, 1] = 0.5
    chi1[1, 0, 1] = chi1[1, 1, 0] = 0.5
    chi2 = np.zeros((2, 2, 2), dtype=np.complex128)
    chi2[0, 0, 0] = 1.0 / np.sqrt(2.0)
    chi2[0, 1, 1] = -1.0 / np.sqrt(2.0)
    psi = np.zeros((2,) * _BM_NUM_QUBITS, dtype=np.complex128)
    psi[0, 0, 0] += np.sqrt(1.0 - _BM_DELTA * _BM_DELTA) * chi1
    psi[0, 0, 1] += _BM_DELTA * chi2
    return psi

def _dense_to_minimal_mps(psi):
    """Exact dense -> MPS conversion; output bond dims equal the Schmidt ranks."""
    tensors = []
    rest = psi.reshape(1, -1)
    chi_l = 1
    for _ in range(_BM_NUM_QUBITS - 1):
        mat = rest.reshape(chi_l * 2, -1)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        rank = int(np.sum(s > 1e-12))
        tensors.append(u[:, :rank].reshape(chi_l, 2, rank))
        rest = s[:rank, None] * vh[:rank, :]
        chi_l = rank
    tensors.append(rest.reshape(chi_l, 2))
    tensors[0] = tensors[0].reshape(2, tensors[0].shape[2])
    return tensors

def _capacity_violations(shapes):
    """Bonds whose dimension exceeds the local capacity min(chi_left*d, d*chi_right)."""
    violations = []
    for i in range(_BM_NUM_QUBITS - 1):
        bond = shapes[i][-1]
        left_capacity = (shapes[i][0] if i > 0 else 1) * 2
        right_capacity = 2 * (shapes[i + 1][-1] if i + 1 < _BM_NUM_QUBITS - 1 else 1)
        if bond > min(left_capacity, right_capacity):
            violations.append((i, bond, min(left_capacity, right_capacity)))
    return violations

def _pad_bond(tensors, bond_index, new_dim):
    """Zero-pad one bond to make the representation overcomplete but exact."""
    left, right = tensors[bond_index], tensors[bond_index + 1]
    old_dim = left.shape[-1]
    assert new_dim > old_dim
    left_padded = np.zeros(left.shape[:-1] + (new_dim,), dtype=left.dtype)
    left_padded[..., :old_dim] = left
    right_padded = np.zeros((new_dim,) + right.shape[1:], dtype=right.dtype)
    right_padded[:old_dim, ...] = right
    tensors = list(tensors)
    tensors[bond_index], tensors[bond_index + 1] = left_padded, right_padded
    return tensors

def _overcomplete_mps():
    """Valid but non-minimal MPS: product state on 0-2, entangled pair on 3-5,
    with bond (3,4) padded from 2 to 3 against a left capacity of 1*2 = 2.
    Shapes: (2,1) (1,2,1) (1,2,1) (1,2,3) (3,2,2) (2,2)."""
    psi = np.zeros((2,) * _BM_NUM_QUBITS, dtype=np.complex128)
    psi[0, 0, 0, 0, 0, 0] = 1.0 / np.sqrt(2.0)
    psi[0, 0, 0, 1, 1, 1] = 1.0 / np.sqrt(2.0)
    return _pad_bond(_dense_to_minimal_mps(psi), 3, 3)

def _dense_to_right_canonical_mps(psi):
    """Exact dense -> MPS with the orthogonality center on the FIRST site."""
    tensors = [None] * _BM_NUM_QUBITS
    rest, chi_r = psi.reshape(-1, 1), 1
    for site in range(_BM_NUM_QUBITS - 1, 0, -1):
        u, s, vh = np.linalg.svd(rest.reshape(-1, 2 * chi_r), full_matrices=False)
        rank = int(np.sum(s > 1e-12))
        u, s, vh = u[:, :rank], s[:rank], vh[:rank, :]
        tensors[site] = (
            vh.reshape(rank, 2, chi_r) if site < _BM_NUM_QUBITS - 1 else vh.reshape(rank, 2)
        )
        rest, chi_r = u * s, rank
    tensors[0] = rest.reshape(2, chi_r)
    return tensors

def _computed_output_shapes(mps_in, **config_kwargs):
    with NetworkState(
        (2,) * _BM_NUM_QUBITS, dtype="complex128", config=MPSConfig(**config_kwargs)
    ) as state:
        state.set_initial_mps([np.array(t, copy=True) for t in mps_in])
        return [tuple(TensorBackend.to_numpy(t).shape) for t in state.compute_output_state()]


def _capture_test_backend(name):
    """Array operations used by the MPS capture tests, on CuPy or PyTorch (CUDA) tensors."""
    if name == "cupy":
        cp = pytest.importorskip("cupy")
        return types.SimpleNamespace(
            asarray=lambda a: cp.asarray(a),
            to_numpy=cp.asnumpy,
            data_ptr=lambda t: t.data.ptr,
            copy=lambda t: t.copy(order="F"))
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch without CUDA")
    return types.SimpleNamespace(
        asarray=lambda a: torch.as_tensor(np.asarray(a), device="cuda"),
        to_numpy=lambda t: t.cpu().numpy(),
        data_ptr=lambda t: t.data_ptr(),
        copy=lambda t: t.clone())


CAPTURE_TEST_BACKENDS = ("cupy", "torch")


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

    def test_from_circuit_mixed(self, circuit_L0, circuit_exact_sv_L0):
        backend = self._get_array_framework(circuit_L0, "from_circuit_mixed")
        with NetworkState.from_circuit(circuit_L0, pure_state=False, backend=backend) as state:
            rdm = state.compute_reduced_density_matrix((0, 1))
            tol = get_contraction_tolerance(state.dtype)
            sv_np = TensorBackend.to_numpy(circuit_exact_sv_L0)
            where_int = [0, 1]
            QuantumStateTestHelper.verify_reduced_density_matrix(sv_np, where_int, rdm, **tol)

    def test_from_converter_mixed(self, circuit_L0, circuit_exact_sv_L0):
        backend = self._get_array_framework(circuit_L0, "from_converter_mixed")
        converter = CircuitToEinsum(circuit_L0, backend=backend)
        with NetworkState.from_converter(converter, pure_state=False) as state:
            rdm = state.compute_reduced_density_matrix((0, 1))
            tol = get_contraction_tolerance(state.dtype)
            sv_np = TensorBackend.to_numpy(circuit_exact_sv_L0)
            where_int = [0, 1]
            QuantumStateTestHelper.verify_reduced_density_matrix(sv_np, where_int, rdm, **tol)

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

    @pytest.mark.parametrize("state_dims", ((2, 2, 2), (2, 3, 2)), ids=lambda d: "x".join(map(str, d)))
    @pytest.mark.parametrize("modes", ((0, 1, 2), (2, 1, 0), (1, 2, 0)), ids=lambda m: "".join(map(str, m)))
    @pytest.mark.parametrize("adjoint", (False, True), ids=("direct", "adjoint"))
    @pytest.mark.parametrize("pure_state", (True, False), ids=("pure", "mixed"))
    def test_three_qudit_diagonal_operator(self, state_dims, modes, adjoint, pure_state):
        """A diagonal operator on modes (q_0, q_1, q_2) is passed as the rank-3 array of its
        diagonal entries D[i_{q_0}, i_{q_1}, i_{q_2}]. Random complex entries make the mode
        order and the adjoint observable; the reference is computed with NumPy."""
        rng = np.random.default_rng(3)
        letters = "abc"
        # Generic product state: one random unitary per qudit on |000>.
        psi = np.zeros(state_dims, dtype=np.complex128)
        psi[0, 0, 0] = 1.0
        unitaries = [np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))[0] for d in state_dims]
        for q, u in enumerate(unitaries):
            psi = np.einsum(f"Jj,{letters[:q]}j{letters[q+1:]}->{letters[:q]}J{letters[q+1:]}", u, psi)
        diag_shape = tuple(state_dims[m] for m in modes)
        diag = rng.normal(size=diag_shape) + 1j * rng.normal(size=diag_shape)
        d_idx = "".join(letters[m] for m in modes)
        psi_ref = np.einsum(f"{d_idx},abc->abc", np.conj(diag) if adjoint else diag, psi)

        with NetworkState(state_dims, dtype="complex128", config=TNConfig(), pure_state=pure_state) as state:
            for q, u in enumerate(unitaries):
                state.apply_tensor_operator((q,), u, unitary=True)
            state.apply_tensor_operator(modes, diag, diagonal=True, adjoint=adjoint, unitary=False)
            tol = get_contraction_tolerance(state.dtype)
            if pure_state:
                np.testing.assert_allclose(state.compute_state_vector(), psi_ref, **tol)
            else:
                rho_ref = np.outer(psi_ref.reshape(-1), psi_ref.reshape(-1).conj()).reshape(state_dims * 2)
                np.testing.assert_allclose(state.compute_density_matrix(), rho_ref, **tol)

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

    def test_sampling_large_mode_count_returns_full_bitstrings(self):
        n_modes = 1001
        op = np.eye(1, dtype=np.complex128)

        with NetworkState((1,) * n_modes, dtype="complex128") as state:
            state.apply_tensor_operator((0,), op, unitary=True)
            samples = state.compute_sampling(2, seed=123)

        assert samples == {"0" * n_modes: 2}
        key = next(iter(samples))
        assert "..." not in key
        assert len(key) == n_modes

    def test_sampling_multi_digit_qudit_no_key_collision(self):
        # A qudit dimension >= 11 produces multi-digit values; preserving the
        # default np.array2string-style spacing keeps them unambiguous without
        # depending on global NumPy print options.
        d = 12
        # Permutation that swaps basis states |0> and |10> on mode 0.
        op = np.eye(d, dtype=np.complex128)
        op[[0, 10]] = op[[10, 0]]

        with NetworkState((d, 2), dtype="complex128") as state:
            state.apply_tensor_operator((0,), op, unitary=True)
            with np.printoptions(formatter={'int': lambda x: f"<{x}>"}):
                samples = state.compute_sampling(4, seed=123)

        # Mode 0 deterministically in |10>, mode 1 in |0>.
        assert samples == {"10 0": 4}

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

        nshots = 2500
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
        
        assert abs(p_zeros - 0.5) < 0.05, \
            f"GHZ |0...0⟩ probability {p_zeros:.3f} deviates >5% from expected 0.5"
        assert abs(p_ones - 0.5) < 0.05, \
            f"GHZ |1...1⟩ probability {p_ones:.3f} deviates >5% from expected 0.5"

    def test_mps_sampling_selected_modes_on_initial_mps(self):
        """Sampling a subset of modes on a NetworkState initialized via set_initial_mps.

        Initializes a NetworkState with an explicit MPS via set_initial_mps and
        applies no gates, then samples a sparse subset of modes (including the
        last site) and checks the returned bitstrings span exactly the requested
        modes.
        """
        backend = self._get_array_framework("test_mps_sampling_selected_modes_on_initial_mps")

        num_sites = 6
        phys_dim = 2
        chi = 4
        rng = np.random.default_rng(0)

        mps = []
        left = 1
        for site in range(num_sites):
            right = chi if 0 < site + 1 < num_sites else 1
            shape = (phys_dim, right) if site == 0 else (
                (left, phys_dim) if site + 1 == num_sites else (left, phys_dim, right)
            )
            a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            mps.append((a / np.linalg.norm(a)).astype(np.complex128))
            left = right

        nshots = 256
        modes = (0, 2, num_sites - 1)
        with NetworkState((phys_dim,) * num_sites, dtype='complex128', config=TNConfig()) as state:
            state.set_initial_mps(mps)
            samples = state.compute_sampling(nshots, modes=modes)

        total = sum(samples.values())
        assert total == nshots, f"Expected {nshots} shots, got {total}"
        for key in samples:
            assert len(key) == len(modes), \
                f"Expected each bitstring to span {len(modes)} selected modes, got len={len(key)} for {key!r}"
            assert all(c in '01' for c in key), f"Unexpected bitstring char in {key!r}"

    def test_mps_sampling_with_locally_overcomplete_bonds(self):
        """Sampling an initialized MPS whose bond extents are locally overcomplete.

        The bond extents below are within the exact half-chain limits for five
        qubits, but the second bond cannot be produced from the left prefix:
        chi_0 * d_1 = 1 * 2 < chi_1 = 4. Sampling such an MPS must still return
        valid bitstrings for the requested modes.
        """
        num_sites = 5
        phys_dim = 2
        bond_extents = (1, 4, 4, 2)
        rng = np.random.default_rng(123)

        mps = []
        left = 1
        for site in range(num_sites):
            right = bond_extents[site] if site + 1 < num_sites else 1
            shape = (phys_dim, right) if site == 0 else (
                (left, phys_dim) if site + 1 == num_sites else (left, phys_dim, right)
            )
            mps.append(rng.standard_normal(shape).astype(np.complex128))
            left = right

        nshots = 32
        with NetworkState((phys_dim,) * num_sites, dtype='complex128', config=TNConfig()) as state:
            state.set_initial_mps(mps)
            samples = state.compute_sampling(nshots, modes=(0, 2, 4), seed=7)

        assert sum(samples.values()) == nshots
        assert all(len(bitstring) == 3 for bitstring in samples)

    def test_mps_sampling_all_modes_on_initial_mps(self):
        """All-modes sampling (``compute_sampling`` with no ``modes``) on an initialized MPS.

        Sampling a freshly-initialized MPS without an explicit mode list is the
        most natural request and must produce correct samples. A
        bond-dimension-2 GHZ MPS is used so the exact distribution is sharp
        (only all-zeros and all-ones, 50/50), giving the test teeth beyond
        shape/shot-count checks.
        """
        num_sites = 6
        phys_dim = 2

        # delta tensors: the only nonzero amplitudes are |0...0> and |1...1>.
        # Bond extents (2,...,2) are within the exact half-chain limits and
        # locally reachable, so the state stays a valid initialized MPS.
        mps = []
        for site in range(num_sites):
            if site == 0:
                t = np.zeros((phys_dim, 2), dtype=np.complex128)
                for s in range(phys_dim):
                    t[s, s] = 1.0
            elif site == num_sites - 1:
                t = np.zeros((2, phys_dim), dtype=np.complex128)
                for s in range(phys_dim):
                    t[s, s] = 1.0
            else:
                t = np.zeros((2, phys_dim, 2), dtype=np.complex128)
                for s in range(phys_dim):
                    t[s, s, s] = 1.0
            mps.append(t)

        nshots = 4000
        with NetworkState((phys_dim,) * num_sites, dtype='complex128', config=TNConfig()) as state:
            state.set_initial_mps(mps)
            samples = state.compute_sampling(nshots, seed=42)

        total = sum(samples.values())
        assert total == nshots, f"Expected {nshots} shots, got {total}"

        all_zeros = '0' * num_sites
        all_ones = '1' * num_sites
        for key in samples:
            assert len(key) == num_sites, \
                f"Expected each bitstring to span all {num_sites} modes, got len={len(key)} for {key!r}"
        assert set(samples.keys()).issubset({all_zeros, all_ones}), \
            f"Unexpected bitstrings in GHZ all-modes sampling: {set(samples.keys()) - {all_zeros, all_ones}}"
        assert all_zeros in samples and all_ones in samples, \
            f"Expected both all-zeros and all-ones in all-modes GHZ sampling, got: {samples}"

        p_zeros = samples.get(all_zeros, 0) / total
        assert abs(p_zeros - 0.5) < 0.05, \
            f"GHZ all-zeros probability {p_zeros:.3f} deviates >5% from expected 0.5"

    def test_mps_sampling_initial_mps_distribution_matches_state_vector(self):
        """Sampling an initialized MPS reproduces the exact distribution.

        The other initial-MPS sampling tests use sharp GHZ/delta states (only
        two nonzero amplitudes) or check shapes only, so they cannot catch a
        distribution that is merely close. Here we sample a *random* (spread-out)
        initialized MPS and require the empirical distribution to overlap >= 0.95
        with the exact distribution obtained by contracting the same MPS to a
        dense state vector, for both all-modes and a selected subset.
        """
        num_sites = 4
        phys_dim = 2
        chi = 2
        rng = np.random.default_rng(2024)

        mps = []
        left = 1
        for site in range(num_sites):
            right = chi if site + 1 < num_sites else 1
            shape = (phys_dim, right) if site == 0 else (
                (left, phys_dim) if site + 1 == num_sites else (left, phys_dim, right)
            )
            mps.append((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128))
            left = right

        # Contract the MPS to a dense state vector of shape (phys_dim,) * num_sites
        # as the exact reference distribution (verify_state_sampling normalizes).
        sv = mps[0]
        for site in range(1, num_sites):
            sv = np.tensordot(sv, mps[site], axes=([sv.ndim - 1], [0]))
        assert sv.shape == (phys_dim,) * num_sites

        all_modes = list(range(num_sites))
        subset_modes = [0, 2]
        with NetworkState((phys_dim,) * num_sites, dtype='complex128', config=TNConfig()) as state:
            state.set_initial_mps(mps)
            verify_state_sampling(state, all_modes, 5000, sv, 3)
            verify_state_sampling(state, subset_modes, 5000, sv, 3)

    def test_mps_sampling_selected_modes_excluding_last_site(self):
        """Sampling a subset of modes whose highest mode is not the last site.

        Every other initial-MPS sampling test requests the final site as its
        highest mode; here the highest requested mode (2) is well below the last
        site (5). A bond-dimension-2 GHZ MPS keeps the exact distribution sharp:
        since every qubit agrees, the (0, 2) outcomes must be exactly ``00`` or
        ``11``, and both must appear.
        """
        num_sites = 6
        phys_dim = 2

        mps = []
        for site in range(num_sites):
            if site == 0:
                t = np.zeros((phys_dim, 2), dtype=np.complex128)
            elif site == num_sites - 1:
                t = np.zeros((2, phys_dim), dtype=np.complex128)
            else:
                t = np.zeros((2, phys_dim, 2), dtype=np.complex128)
            for s in range(phys_dim):
                if site == 0:
                    t[s, s] = 1.0
                elif site == num_sites - 1:
                    t[s, s] = 1.0
                else:
                    t[s, s, s] = 1.0
            mps.append(t)

        nshots = 4000
        modes = (0, 2)
        with NetworkState((phys_dim,) * num_sites, dtype='complex128', config=TNConfig()) as state:
            state.set_initial_mps(mps)
            samples = state.compute_sampling(nshots, modes=modes, seed=42)

        assert sum(samples.values()) == nshots
        for key in samples:
            assert len(key) == len(modes), \
                f"Expected each bitstring to span {len(modes)} modes, got len={len(key)} for {key!r}"
        assert set(samples.keys()).issubset({'00', '11'}), \
            f"Unexpected bitstrings sampling a GHZ prefix subset: {set(samples.keys()) - {'00', '11'}}"
        assert '00' in samples and '11' in samples, \
            f"Expected both '00' and '11' from the GHZ (0, 2) marginal, got: {samples}"

    def test_mps_sampling_rejects_zero_norm_initial_mps(self):
        """Sampling a zero-norm initialized MPS raises instead of emitting garbage.

        A state with no support (here an all-zero initial MPS) must raise
        ``CUTENSORNET_STATUS_INVALID_VALUE`` rather than silently returning a
        degenerate sample.
        """
        num_sites = 3
        phys_dim = 2

        # All-zero, bond-dimension-1 MPS: valid, non-overcomplete bond extents
        # but with zero total mass.
        mps = []
        for site in range(num_sites):
            if site == 0:
                shape = (phys_dim, 1)
            elif site == num_sites - 1:
                shape = (1, phys_dim)
            else:
                shape = (1, phys_dim, 1)
            mps.append(np.zeros(shape, dtype=np.complex128))

        with NetworkState((phys_dim,) * num_sites, dtype='complex128', config=TNConfig()) as state:
            state.set_initial_mps(mps)
            with pytest.raises(cutn.cuTensorNetError) as exc_info:
                state.compute_sampling(16, seed=1)
            assert "INVALID_VALUE" in str(exc_info.value)

    def test_sampling_rejects_zero_norm(self):
        """Sampling a zero-norm contraction-based state raises instead of emitting garbage.

        Companion to ``test_mps_sampling_rejects_zero_norm_initial_mps`` for a
        state built by applying operators: a zero operator drives the state norm
        to zero, and sampling must raise ``CUTENSORNET_STATUS_INVALID_VALUE``
        instead of returning a degenerate sample.
        """
        zero_gate = np.zeros((2, 2), dtype=np.complex128)
        with NetworkState((2, 2), dtype='complex128', config=TNConfig()) as state:
            state.apply_tensor_operator((0,), zero_gate)
            with pytest.raises(cutn.cuTensorNetError) as exc_info:
                state.compute_sampling(16, seed=1)
            assert "INVALID_VALUE" in str(exc_info.value)

    @staticmethod
    def _create_ghz_circuit(n_qubits):
        qiskit = pytest.importorskip("qiskit")

        circuit = qiskit.QuantumCircuit(n_qubits)
        circuit.h(0)
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)
        return circuit

    def _assert_repeated_ghz_single_shots_observe_both_outcomes(self, *, seed):
        n_qubits = 6
        num_trials = 64
        expected_outcomes = {'0' * n_qubits, '1' * n_qubits}
        backend = self._get_array_framework(
            f"test_ghz_single_shot_rng_advances_{'default' if seed is None else 'seeded'}")
        initial_sample_kwargs = {} if seed is None else {'seed': seed}

        circuit = self._create_ghz_circuit(n_qubits)
        observed_outcomes = set()
        with NetworkState.from_circuit(circuit, backend=backend) as state:
            observed_outcomes.update(state.compute_sampling(1, **initial_sample_kwargs))
            for _ in range(num_trials - 1):
                observed_outcomes.update(state.compute_sampling(1))

        assert observed_outcomes == expected_outcomes

    @pytest.mark.parametrize("seed", (None, 42), ids=("default-seed", "configured-seed"))
    def test_ghz_single_shot_sampling_advances_rng(self, seed):
        self._assert_repeated_ghz_single_shots_observe_both_outcomes(seed=seed)

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

    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_mps_release_operators_with_mixed_extents(self, gauge_option):
        # swaps of qudits of unequal extents let bonds grow beyond their targets while a gate is applied, and a
        # compression restores them after the gate; releasing the operators in the middle must still give the result
        # of the uninterrupted run
        dims = (2, 3, 5, 3, 2)
        config = MPSConfig(max_extent=8, rel_cutoff=1e-2, gauge_option=gauge_option)
        rng = np.random.default_rng(5)
        _, initial = _random_mixed_mps(dims, rng, "complex128", 2)
        gates = []
        for modes in ((0, 3), (4, 1), (2, 0), (1, 3), (4, 0), (3, 1)):
            d = dims[modes[0]] * dims[modes[1]]
            u, _ = np.linalg.qr(rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
            gates.append((modes, u.reshape([dims[m] for m in modes] * 2)))

        with NetworkState(dims, dtype="complex128", config=config) as state:
            state.set_initial_mps(initial)
            for modes, gate in gates[:3]:
                state.apply_tensor_operator(modes, gate)
            captured = [o.copy() for o in state.compute_output_state(release_operators=True)]
            for modes, gate in gates[3:]:
                state.apply_tensor_operator(modes, gate)
            sv0 = state.compute_state_vector()
        with NetworkState(dims, dtype="complex128", config=config) as reference:
            reference.set_initial_mps(initial)
            for modes, gate in gates:
                reference.apply_tensor_operator(modes, gate)
            sv1 = reference.compute_state_vector()
        with NetworkState(dims, dtype="complex128", config=config) as restarted:
            restarted.set_initial_mps(captured)
            for modes, gate in gates[3:]:
                restarted.apply_tensor_operator(modes, gate)
            sv2 = restarted.compute_state_vector()

        tol = get_mps_tolerance("complex128")
        QuantumStateTestHelper.verify_state_vector(sv0, sv1, **tol)
        QuantumStateTestHelper.verify_state_vector(sv0, sv2, **tol)

    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    @pytest.mark.parametrize("backend", CAPTURE_TEST_BACKENDS)
    @pytest.mark.parametrize("sequence", ("apply", "update", "channel", "release_each_compute"))
    def test_mps_release_operators_then_compute_again(self, sequence, backend, gauge_option):
        # state computations after release_operators=True start from the captured output state,
        # also when the state is computed more than once before the next release
        xp = _capture_test_backend(backend)
        rng = np.random.default_rng(5)
        gates = [xp.asarray(np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0].reshape(2, 2, 2, 2))
                 for _ in range(3)]
        x = xp.asarray(np.asarray([[0, 1], [1, 0]], dtype="complex128"))
        z = xp.asarray(np.asarray([[1, 0], [0, -1]], dtype="complex128"))
        identity = xp.asarray(np.eye(2, dtype="complex128"))

        def random_state():
            state = NetworkState((2, 2, 2), dtype="complex128", config=MPSConfig(gauge_option=gauge_option))
            for gate, modes in zip(gates, ((0, 1), (1, 2), (0, 1))):
                state.apply_tensor_operator(modes, gate, unitary=True)
            return state

        def reference(operators):
            with random_state() as state:
                for modes, operand in operators:
                    state.apply_tensor_operator(modes, operand, unitary=True)
                return xp.to_numpy(state.compute_state_vector())

        with random_state() as state:
            captured = state.compute_output_state(release_operators=True)
            if sequence == "apply":
                state.apply_tensor_operator((0,), x, unitary=True)
                sv_first = xp.to_numpy(state.compute_state_vector())
                state.apply_tensor_operator((1,), z, unitary=True)
                sv_second = xp.to_numpy(state.compute_state_vector())
                expected = (reference([((0,), x)]), reference([((0,), x), ((1,), z)]))
            elif sequence == "update":
                tensor_id = state.apply_tensor_operator((0,), x, unitary=True, immutable=False)
                sv_first = xp.to_numpy(state.compute_state_vector())
                state.update_tensor_operator(tensor_id, z, unitary=True)
                sv_second = xp.to_numpy(state.compute_state_vector())
                expected = (reference([((0,), x)]), reference([((0,), z)]))
            elif sequence == "channel":
                # every property computation of a state with stochastic channels computes the state again
                state.apply_unitary_tensor_channel((0,), [x, identity], [1.0, 0.0])
                sv_first = xp.to_numpy(state.compute_state_vector())
                sv_second = xp.to_numpy(state.compute_state_vector())
                expected = (reference([((0,), x)]),) * 2
            else:
                state.apply_tensor_operator((0,), x, unitary=True)
                recaptured = state.compute_output_state(release_operators=True)
                # computed in place and captured again: no additional output tensors
                assert [xp.data_ptr(t) for t in recaptured] == [xp.data_ptr(t) for t in captured]
                sv_first = xp.to_numpy(state.compute_state_vector())
                state.apply_tensor_operator((1,), z, unitary=True)
                sv_second = xp.to_numpy(state.compute_state_vector())
                expected = (reference([((0,), x)]), reference([((0,), x), ((1,), z)]))
            np.testing.assert_allclose(sv_first, expected[0], atol=1e-12)
            np.testing.assert_allclose(sv_second, expected[1], atol=1e-12)

    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    @pytest.mark.parametrize("truncation", ({}, {"max_extent": 2}, {"rel_cutoff": 0.1}),
                             ids=("exact", "max_extent", "rel_cutoff"))
    @pytest.mark.parametrize("backend", CAPTURE_TEST_BACKENDS)
    def test_mps_capture_every_computation_in_place(self, truncation, backend, gauge_option):
        # capturing after every computation computes into the tensors holding the captured initial state (an
        # implementation property of NetworkState that the C API does not guarantee); the result matches the same
        # steps computed into separate tensors (a new state initialized with a copy of each output)
        xp = _capture_test_backend(backend)
        rng = np.random.default_rng(11)
        n = 4
        layers = [[(modes, xp.asarray(np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
                                      .reshape(2, 2, 2, 2))) for modes in ((0, 1), (2, 3), (1, 2))]
                  for _ in range(3)]
        config = MPSConfig(gauge_option=gauge_option, **truncation)

        def state_vector(mps):
            mps = [xp.to_numpy(t) for t in mps]
            return np.einsum("ia,ajb,bkc,cl->ijkl", *mps)

        reference = None
        with NetworkState((2,) * n, dtype="complex128", config=config) as state:
            pointers = None
            for layer in layers:
                for modes, gate in layer:
                    state.apply_tensor_operator(modes, gate, unitary=True)
                captured = state.compute_output_state(release_operators=True)
                if pointers is not None:
                    # computed in place: the output tensors are those holding the previous capture
                    assert [xp.data_ptr(t) for t in captured] == pointers
                pointers = [xp.data_ptr(t) for t in captured]
                with NetworkState((2,) * n, dtype="complex128", config=config) as ref_state:
                    if reference is not None:
                        ref_state.set_initial_mps(reference)
                    for modes, gate in layer:
                        ref_state.apply_tensor_operator(modes, gate, unitary=True)
                    reference = [xp.copy(t) for t in ref_state.compute_output_state()]
                np.testing.assert_allclose(state_vector(captured), state_vector(reference), atol=1e-10)

    def test_mps_truncation_with_channel_interleaved_properties(self):
        # With value-based truncation and a stochastic channel whose outcomes give different bond extents, every
        # computation can change the extents and change them back; a cached expectation object must only be reused at
        # the extents it was created for. Every expectation is one of the two per-outcome values.
        cp = pytest.importorskip("cupy")
        n = 4
        config = MPSConfig(max_extent=4, rel_cutoff=0.2)
        operator = NetworkOperator.from_pauli_strings({"ZZZZ": 1.0, "XIII": 0.5}, dtype="complex128")

        def build(channel_branch=None):
            rng = np.random.default_rng(3)
            state = NetworkState((2,) * n, dtype="complex128", config=config)
            for modes in ((0, 1), (1, 2), (2, 3), (0, 1)):
                u = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0].reshape(2, 2, 2, 2)
                state.apply_tensor_operator(modes, cp.asarray(u), unitary=True)
            entangling = cp.asarray(
                np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0].reshape(2, 2, 2, 2))
            identity = cp.eye(4, dtype="complex128").reshape(2, 2, 2, 2)
            if channel_branch is None:
                state.apply_unitary_tensor_channel((1, 2), [entangling, identity], [0.5, 0.5])
            elif channel_branch == "entangling":
                state.apply_tensor_operator((1, 2), entangling, unitary=True)
            return state

        expected = []
        for branch in ("entangling", "identity"):
            with build(branch) as state:
                expected.append(complex(state.compute_expectation(operator)))
        seen = set()
        with build() as state:
            for _ in range(20):
                state.compute_output_state()
                value = complex(state.compute_expectation(operator))
                matches = [i for i, e in enumerate(expected) if abs(value - e) < 1e-10]
                assert matches, f"expectation {value} matches neither channel outcome {expected}"
                seen.update(matches)
        assert seen == {0, 1}, "both channel outcomes should occur in 20 computations"

    @pytest.mark.parametrize("backend", CAPTURE_TEST_BACKENDS)
    def test_mps_own_output_as_initial_state(self, backend):
        # output tensors of a state passed back to set_initial_mps of the same state hold its initial state:
        # later computations must not overwrite them
        xp = _capture_test_backend(backend)
        rng = np.random.default_rng(6)
        shapes = ((2, 2), (2, 2, 2), (2, 2))
        mps = [xp.asarray(rng.normal(size=s) + 1j * rng.normal(size=s)) for s in shapes]
        x = xp.asarray(np.asarray([[0, 1], [1, 0]], dtype="complex128"))
        z = xp.asarray(np.asarray([[1, 0], [0, -1]], dtype="complex128"))

        def run(state, operators):
            results = []
            for modes, operand in operators:
                state.apply_tensor_operator(modes, operand, unitary=True)
                results.append(xp.to_numpy(state.compute_state_vector()))
            return results

        operators = [((0,), x), ((1,), z)]
        with NetworkState((2, 2, 2), dtype="complex128", config=MPSConfig(gauge_option='simple')) as state:
            state.set_initial_mps(mps)
            output = state.compute_output_state()
            initial = [xp.copy(t) for t in output]  # copies, for the reference
            state.set_initial_mps(output)
            actual = run(state, operators)
        with NetworkState((2, 2, 2), dtype="complex128", config=MPSConfig(gauge_option='simple')) as reference:
            reference.set_initial_mps(initial)
            expected = run(reference, operators)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_mps_expectation_after_unsupported_gradients(self):
        # gradients are not supported for MPS simulation; the failed call leaves the object usable
        cp = pytest.importorskip("cupy")
        h = cp.asarray([[1, 1], [1, -1]], dtype="complex128") / 2 ** 0.5
        with NetworkState((2, 2), dtype="complex128", config=MPSConfig()) as state:
            state.apply_tensor_operator((0,), h, unitary=True, gradient=True)
            with pytest.raises(cutn.cuTensorNetError) as e:
                state.compute_expectation_with_gradients("XI", 1.0)
            assert e.value.status == cutn.Status.NOT_SUPPORTED
            np.testing.assert_allclose(complex(state.compute_expectation("XI")), 1.0, atol=1e-12)

    @pytest.mark.parametrize("stream_kind", ("object", "ptr"))
    def test_mps_release_operators_with_explicit_stream(self, stream_kind):
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA required")

        stream = cp.cuda.Stream(non_blocking=True)
        stream_arg = stream.ptr if stream_kind == "ptr" else stream

        with NetworkState((2, 2), dtype="complex128", config=MPSConfig()) as state:
            with stream:
                eye = cp.eye(2, dtype=cp.complex128)
                state.apply_tensor_operator((0,), eye, unitary=True, stream=stream_arg)
                state.apply_tensor_operator((1,), eye, unitary=True, stream=stream_arg)
                mps_tensors = state.compute_output_state(stream=stream_arg, release_operators=True)

            stream.synchronize()

        assert [tensor.shape for tensor in mps_tensors] == [(2, 1), (1, 2)]

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
            assert TensorBackend.verify_close(e1, e2, **tolerance)
        
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
            assert TensorBackend.verify_close(norm, norm0)
            norm = state.compute_amplitude('0'*ndim, return_norm=True)[1]
            assert TensorBackend.verify_close(norm, norm0)
            norm = state.compute_batched_amplitudes({0:'0'}, return_norm=True)[1]
            assert TensorBackend.verify_close(norm, norm0)
            operator = get_random_network_operator(factory.state_dims, rng, backend.name, num_repeats=1, dtype=state.dtype, options=state.options)
            norm = state.compute_expectation(operator, return_norm=True)[1]
            assert TensorBackend.verify_close(norm, norm0)


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

    @pytest.mark.parametrize("max_extent", (None, 2))
    @pytest.mark.parametrize("algorithm", ("gesvd", "gesvdj", "gesvdr", "gesvdp"))
    def test_mps_two_qubit_gate_all_svd_algorithms(self, algorithm, max_extent):
        # Apply a two-qubit gate to an MPS under each SVD algorithm. A bond extent of 2 is exact
        # for the Bell pair, so both settings yield the same state.
        config = MPSConfig(algorithm=algorithm, max_extent=max_extent)
        hadamard = np.array([[1, 1], [1, -1]], dtype='complex128') / np.sqrt(2)
        cnot = np.eye(4, dtype='complex128')[[0, 1, 3, 2]].reshape(2, 2, 2, 2)
        with NetworkState((2, 2, 2), dtype='complex128', config=config) as state:
            state.apply_tensor_operator((0,), hadamard, immutable=True)
            state.apply_tensor_operator((0, 1), cnot, immutable=True)
            mps = state.compute_output_state()
            assert len(mps) == 3
            sv = state.compute_state_vector()
        expected = np.zeros((2, 2, 2), dtype='complex128')
        expected[0, 0, 0] = expected[1, 1, 0] = 1 / np.sqrt(2)
        np.testing.assert_allclose(np.asarray(sv), expected, atol=1e-12)

    @pytest.mark.parametrize("algorithm", ("gesvd", "gesvdj", "gesvdr", "gesvdp"))
    def test_mps_two_qubit_gate_without_host_workspace(self, monkeypatch, algorithm):
        # A caller that provides device scratch workspace only. The documented contract expects the
        # caller to attach the host scratch workspace the prepare call reports; the library nevertheless
        # tolerates its absence for backward compatibility (callers written before cuSOLVER started
        # requesting host workspace for the SVD). This test pins that undocumented tolerance for every
        # SVD algorithm; it is not a guarantee users may rely on.
        set_memory = cutn.workspace_set_memory
        def set_device_memory_only(handle, desc, memspace, kind, ptr, size):
            if memspace == cutn.Memspace.HOST:
                return
            return set_memory(handle, desc, memspace, kind, ptr, size)
        monkeypatch.setattr(cutn, "workspace_set_memory", set_device_memory_only)
        config = MPSConfig(algorithm=algorithm, max_extent=2)
        hadamard = np.array([[1, 1], [1, -1]], dtype='complex128') / np.sqrt(2)
        cnot = np.eye(4, dtype='complex128')[[0, 1, 3, 2]].reshape(2, 2, 2, 2)
        with NetworkState((2, 2, 2), dtype='complex128', config=config) as state:
            state.apply_tensor_operator((0,), hadamard, immutable=True)
            state.apply_tensor_operator((0, 1), cnot, immutable=True)
            sv = state.compute_state_vector()
        expected = np.zeros((2, 2, 2), dtype='complex128')
        expected[0, 0, 0] = expected[1, 1, 0] = 1 / np.sqrt(2)
        np.testing.assert_allclose(np.asarray(sv), expected, atol=1e-12)

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


    @pytest.mark.parametrize("gauge_option", ["free", "simple"])
    def test_canonical_center_sweep_keeps_bonds_minimal(self, gauge_option):
        config = MPSConfig(
            gauge_option=gauge_option,
            max_extent=3,
            canonical_center=2,
            rel_cutoff=0.1,
            normalization="L2",
        )
        mps_in = _dense_to_minimal_mps(_build_state())
        with NetworkState((2,) * _BM_NUM_QUBITS, dtype="complex128", config=config) as state:
            state.set_initial_mps([np.array(t, copy=True) for t in mps_in])
            out = state.compute_output_state()
            shapes = [tuple(TensorBackend.to_numpy(t).shape) for t in out]
            violations = _capacity_violations(shapes)
            assert not violations, (
                f"overcomplete output bonds {violations} in shapes {shapes}"
            )

    @pytest.mark.parametrize("gauge_option", ["free", "simple"])
    def test_sampler_accepts_overcomplete_input_mps(self, gauge_option):
        """NetworkState-level contract: a valid but overcomplete initial MPS samples.

        Overcomplete (dimension > rank) bonds are exact, valid representations:
        here bond (3,4) carries dimension 3 while its left capacity is 1*2 = 2.
        State computation, amplitudes, and expectation values all accept such
        states; sampling must too.

        Deliberately configured with NO truncation settings, so the exit sweeps are
        skipped entirely and the overcomplete state reaches the sampler untouched.
        That isolates the consumer: the sweeping sampler wraps the tensors in a
        MatrixProductState, whose bond-minimality precondition the ProjectionMPS
        entry point validates via areMPSBondExtentsValid() but the sampler path does
        not. Overcomplete states must be routed to the generic sampler instead of
        failing with an opaque internal error.
        """
        config = MPSConfig(gauge_option=gauge_option)
        mps_in = _overcomplete_mps()
        with NetworkState((2,) * _BM_NUM_QUBITS, dtype="complex128", config=config) as state:
            state.set_initial_mps([np.array(t, copy=True) for t in mps_in])
            samples = state.compute_sampling(64)
            assert samples

    @pytest.mark.parametrize(
        "gauge_option",
        [
            pytest.param(
                "free",
                marks=pytest.mark.xfail(
                    reason="Without a requested canonical center the bond-minimality "
                    "repair pass is scoped off -- it would re-gauge the output, and "
                    "truncation is gauge-sensitive -- so an overcomplete bond survives "
                    "into the computed output. The MPS sweeping sampler guards against "
                    "consuming such a state (see test above), but it is still handed to "
                    "the caller, and the ProjectionMPS entry point rejects it.",
                    strict=True,
                ),
            ),
            # 'simple' re-minimizes on ingest via its unconditional gauge-setup sweep.
            pytest.param("simple"),
        ],
    )
    def test_computed_output_is_bond_minimal_without_canonical_center(self, gauge_option):
        """The computed output should satisfy bond minimality with or without a center."""
        shapes = _computed_output_shapes(_overcomplete_mps(), gauge_option=gauge_option)
        violations = _capacity_violations(shapes)
        assert not violations, f"overcomplete output bonds {violations} in shapes {shapes}"

    @pytest.mark.xfail(
        reason="Truncation is only performed where the orthogonality center is: a "
        "sweep starting from the opposite boundary decomposes isometries, whose "
        "singular values are all 1, so it silently truncates nothing at all. "
        "Compression therefore depends on the gauge the input happens to arrive in. "
        "Fixing it means always truncating at the center, a semantics change that "
        "must land together with the NumPy reference in _internal/mps_utils.py, "
        "which mirrors this sweep structure bond for bond.",
        strict=True,
    )
    def test_truncation_does_not_depend_on_input_gauge(self):
        """One state, two exact gauges, one config -> the same compression.

        Both inputs represent the identical state exactly; they differ only in
        which site carries the orthogonality center. With canonical_center=3 the
        left-canonical input is swept from the boundary the center is NOT at, so
        nothing is truncated despite rel_cutoff=0.1; the right-canonical input is
        swept from the center and compresses as requested.
        """
        psi = _build_state()
        config = dict(
            gauge_option="free",
            max_extent=3,
            canonical_center=3,
            rel_cutoff=0.1,
            normalization="L2",
        )
        left = _computed_output_shapes(_dense_to_minimal_mps(psi), **config)
        right = _computed_output_shapes(_dense_to_right_canonical_mps(psi), **config)
        assert left == right, (
            f"compression depends on the input gauge:\n"
            f"  from left-canonical : {left}\n"
            f"  from right-canonical: {right}"
        )


class TestMPSOutputExtentSizing:
    """Output MPS sizing: natural bond extents, the limit without max_extent, and the int64 range of the C API."""

    HADAMARD = (
        np.array([[1, 1], [1, -1]], dtype="complex128") / np.sqrt(2)
    )

    @pytest.mark.parametrize("max_extent, expected", (
        (None, [(2, 2), (2, 3, 6), (6, 5, 7), (7, 7)]),
        (4, [(2, 2), (2, 3, 4), (4, 5, 4), (4, 7)]),
        (10**6, [(2, 2), (2, 3, 6), (6, 5, 7), (7, 7)]),
    ))
    def test_extents_for_uneven_mode_extents(self, max_extent, expected):
        assert get_mps_output_extents((2, 3, 5, 7), max_extent, 16) == expected

    def test_documented_limit_matches(self):
        # the limit is documented by value in MPSConfig; keep the two in sync
        assert f"limited to {EXACT_MPS_EXTENT_LIMIT}" in " ".join(MPSConfig.__doc__.split())

    def test_limit_without_max_extent(self):
        extents = get_mps_output_extents((2,) * 20, None, 16)
        assert max(e[-1] for e in extents[:-1]) == EXACT_MPS_EXTENT_LIMIT
        with pytest.raises(ValueError, match=f"limited to {EXACT_MPS_EXTENT_LIMIT}"):
            get_mps_output_extents((2,) * 22, None, 16)
        with pytest.raises(ValueError, match=f"limited to {EXACT_MPS_EXTENT_LIMIT}"):
            get_mps_output_extents((2,) * 130, None, 16)

    def test_explicit_max_extent_lifts_the_limit(self):
        # a cap at or above the largest exact bond extent gives the exact extents
        extents = get_mps_output_extents((2,) * 22, 2048, 16)
        assert extents == get_mps_output_extents((2,) * 22, 2**40, 16)
        assert max(e[-1] for e in extents[:-1]) == 2048

    @pytest.mark.parametrize("max_extent, itemsize, accepted", (
        (2**28, 16, True),   # central tensor 2**28 * 2 * 2**28 elements of 16 bytes = 2**61 bytes
        (2**29, 16, False),  # 2**63 bytes
        (2**29, 8, True),    # 2**62 bytes
        (np.iinfo(np.int64).max + 1, 16, False),
        (2**70, 16, False),
    ))
    def test_int64_range(self, max_extent, itemsize, accepted):
        if accepted:
            get_mps_output_extents((2,) * 130, max_extent, itemsize)
        else:
            with pytest.raises(ValueError, match="int64"):
                get_mps_output_extents((2,) * 130, max_extent, itemsize)

    @pytest.mark.parametrize("max_extent", (0, -1, 2.5, True, "8"))
    def test_invalid_max_extent_is_rejected(self, max_extent):
        with pytest.raises(ValueError, match="max_extent"):
            MPSConfig(max_extent=max_extent)

    def test_numpy_integer_max_extent_is_accepted(self):
        assert MPSConfig(max_extent=np.int64(4)).max_extent == 4

    @classmethod
    def _hadamard_amplitude(cls, n, max_extent):
        config = MPSConfig(max_extent=max_extent, gauge_option="simple")
        with NetworkState((2,) * n, dtype="complex128", config=config) as state:
            state.apply_tensor_operator((0,), cls.HADAMARD)
            return state.compute_amplitude("0" * n)

    def test_explicit_max_extent_above_limit_is_accepted(self):
        assert np.isclose(self._hadamard_amplitude(22, 1025), 1 / np.sqrt(2))

    def test_extent_sizing_does_not_overflow_beyond_63_qubits(self):
        assert np.isclose(self._hadamard_amplitude(64, 8), 1 / np.sqrt(2))

    def test_uncapped_extent_above_limit_is_rejected(self):
        with pytest.raises(ValueError, match=f"limited to {EXACT_MPS_EXTENT_LIMIT}"):
            self._hadamard_amplitude(22, None)

    def test_explicit_max_extent_above_int64_is_rejected(self):
        with pytest.raises(ValueError, match="int64"):
            self._hadamard_amplitude(128, 2**70)


class TestMPSCanonicalizationTruncation:

    SEED = 2024
    QUDIT_DIMS = (4, 4, 2)
    TRUNCATED_EXTENT = 3
    TOL = 1e-10
    RIGHT_GATE_SITES = ((0, 1), (1, 2), (0, 1))
    LEFT_GATE_SITES = ((2, 1), (1, 0), (2, 1))
    SWEEP_CASES = [
        pytest.param(QUDIT_DIMS, RIGHT_GATE_SITES, 0, 2, id='right'),
        pytest.param(QUDIT_DIMS[::-1], LEFT_GATE_SITES, 1, 0, id='left'),
    ]

    @staticmethod
    def _random_unitary_gate(rng, dim_a, dim_b):
        dim = dim_a * dim_b
        matrix = (
            rng.standard_normal((dim, dim))
            + 1j * rng.standard_normal((dim, dim))
        )
        q, _ = np.linalg.qr(matrix)
        return q.reshape(dim_a, dim_b, dim_a, dim_b)

    @staticmethod
    def _dense_state(mps):
        """Contract the MPS, including its gauges, without mutating it."""
        tensors = list(mps.mps_tensors)
        for site, gauge in mps.gauges.items():
            tensors[site] = np.einsum('i,ijk->ijk', gauge, tensors[site])
        state = tensors[0]
        for tensor in tensors[1:]:
            state = np.tensordot(
                state, tensor, axes=([state.ndim - 1], [0])
            )
        return state.reshape(mps.state_dims)

    @staticmethod
    def _schmidt_values(state, bond):
        left_extent = np.prod(state.shape[:bond + 1])
        return np.linalg.svd(
            state.reshape(left_extent, -1), compute_uv=False
        )

    @staticmethod
    def _fidelity(psi, phi):
        overlap = np.vdot(psi.ravel(), phi.ravel())
        return abs(overlap) ** 2 / (
            np.vdot(psi.ravel(), psi.ravel()).real
            * np.vdot(phi.ravel(), phi.ravel()).real
        )

    @classmethod
    def _build_entangled_mps(cls, gauge_option, qudit_dims, gate_sites):
        rng = np.random.default_rng(cls.SEED)
        mps = MPS(
            range(3),
            'numpy',
            qudit_dims=qudit_dims,
            gauge_option=gauge_option,
        )
        for sites in gate_sites:
            gate = cls._random_unitary_gate(
                rng, qudit_dims[sites[0]], qudit_dims[sites[1]]
            )
            mps.apply_gate(sites, gate)
        return mps

    @classmethod
    def _assert_optimal_truncation(cls, psi, psi_trunc, bond):
        exact_schmidt = cls._schmidt_values(psi, bond)
        assert exact_schmidt[-1] > 1e-3
        achieved = cls._fidelity(psi, psi_trunc)
        optimal = (
            np.sum(exact_schmidt[:cls.TRUNCATED_EXTENT] ** 2)
            / np.sum(exact_schmidt ** 2)
        )
        assert abs(achieved - optimal) < cls.TOL, (
            f"fidelity {achieved!r} vs optimal {optimal!r}"
        )

        truncated_schmidt = cls._schmidt_values(
            psi_trunc, bond
        )[:cls.TRUNCATED_EXTENT]
        expected = exact_schmidt[:cls.TRUNCATED_EXTENT]
        np.testing.assert_allclose(
            truncated_schmidt / np.linalg.norm(truncated_schmidt),
            expected / np.linalg.norm(expected),
            rtol=0,
            atol=cls.TOL,
        )

    @classmethod
    def _check_reference(
            cls, gauge_option, qudit_dims, gate_sites, bond, canonical_center):
        mps = cls._build_entangled_mps(
            gauge_option, qudit_dims, gate_sites
        )
        psi = cls._dense_state(mps)
        assert mps[bond].shape[-1] == qudit_dims[bond]

        mps.max_extents[bond] = cls.TRUNCATED_EXTENT
        mps.canonical_center = canonical_center
        mps.canonicalize()
        assert mps[bond].shape[-1] == cls.TRUNCATED_EXTENT

        cls._assert_optimal_truncation(
            psi, cls._dense_state(mps), bond
        )

    @pytest.mark.parametrize(
        'qudit_dims,gate_sites,bond,canonical_center', SWEEP_CASES
    )
    def test_simple_update_truncates_on_schmidt_values(
            self, qudit_dims, gate_sites, bond, canonical_center):
        """Simple-update canonicalization truncates optimally in both directions."""
        self._check_reference(
            'simple', qudit_dims, gate_sites, bond, canonical_center
        )

    @pytest.mark.xfail(strict=True, reason=(
        "Gauge-free truncating canonicalization has no local information about the far "
        "environment: the sweep truncates a pair that is not the orthogonality center, "
        "so the discarded singular values are not Schmidt values and the result is "
        "suboptimal."))
    def test_gauge_free_canonicalization_is_not_optimal(self):
        self._check_reference(
            'free', self.QUDIT_DIMS, self.RIGHT_GATE_SITES, 0, 1
        )

    @pytest.mark.parametrize(
        'qudit_dims,gate_sites,bond,canonical_center', SWEEP_CASES
    )
    def test_network_state_truncates_on_schmidt_values(
            self, qudit_dims, gate_sites, bond, canonical_center):
        """The native simple-update sweep matches optimal Schmidt truncation."""
        cp = pytest.importorskip('cupy')

        mps = self._build_entangled_mps(
            'simple', qudit_dims, gate_sites
        )
        psi = self._dense_state(mps)
        tensors = mps.absorb_gauges(mps.mps_tensors)
        initial = [
            tensors[0].reshape(tensors[0].shape[1:]),
            tensors[1],
            tensors[2].reshape(tensors[2].shape[:-1]),
        ]
        assert initial[bond].shape[-1] == qudit_dims[bond]

        config = MPSConfig(
            gauge_option='simple',
            max_extent=self.TRUNCATED_EXTENT,
            canonical_center=canonical_center,
        )
        with NetworkState(qudit_dims, dtype='complex128', config=config) as state:
            state.set_initial_mps([cp.asarray(tensor) for tensor in initial])
            psi_trunc = cp.asnumpy(state.compute_state_vector())
            output = state.compute_output_state()
        assert output[bond].shape[-1] == self.TRUNCATED_EXTENT

        self._assert_optimal_truncation(psi, psi_trunc, bond)


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

    def test_marginal_probability(self, circuit_L1, exact_config, circuit_exact_sv_L1):
        super().test_marginal_probability(circuit_L1, exact_config, circuit_exact_sv_L1, NUM_TESTS_PER_CONFIG)

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

    def test_marginal_probability(self, factory_L1, exact_config, sv_factory_L1):
        super().test_marginal_probability(factory_L1, exact_config, sv_factory_L1, NUM_TESTS_PER_CONFIG)

    def test_sampling(self, factory_L1, exact_config, sv_factory_L1):
        super().test_sampling(factory_L1, exact_config, sv_factory_L1, NUM_TESTS_PER_CONFIG)


@pytest.fixture(params=MixedGenericStateMatrix.L1(), scope="class")
def mixed_factory_L1(request):
    return request.param

@pytest.fixture(scope="class")
def sv_mixed_factory_L1(mixed_factory_L1):
    return mixed_factory_L1.compute_state_vector()

@pytest.fixture(params=[TNConfig()], scope="class")
def mixed_exact_config(request):
    return request.param


class TestExactGenericStateMixed(BaseMixedGenericStateTester):
    """Exact generic state tests with pure_state=False (unitary-only, rho=|psi><psi|)."""

    def test_density_matrix(self, mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1):
        super().test_density_matrix(mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1)

    def test_amplitude(self, mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1):
        super().test_amplitude(mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1, NUM_TESTS_PER_CONFIG)

    def test_marginal_probability(self, mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1):
        super().test_marginal_probability(mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1, NUM_TESTS_PER_CONFIG)

    def test_expectation(self, mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1):
        super().test_expectation(mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1, NUM_TESTS_PER_CONFIG)

    def test_reduced_density_matrix(self, mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1):
        super().test_reduced_density_matrix(mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1, NUM_TESTS_PER_CONFIG)

    def test_sampling(self, mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1):
        super().test_sampling(mixed_factory_L1, mixed_exact_config, sv_mixed_factory_L1, NUM_TESTS_PER_CONFIG)


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

    def test_marginal_probability(self, circuit_L2, approx_mps_config, circuit_approx_sv_L2):
        super().test_marginal_probability(circuit_L2, approx_mps_config, circuit_approx_sv_L2, NUM_TESTS_PER_CONFIG)

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

    def test_marginal_probability(self, factory_L2, approx_mps_config, factory_approx_sv_L2):
        super().test_marginal_probability(factory_L2, approx_mps_config, factory_approx_sv_L2, NUM_TESTS_PER_CONFIG)

    def test_sampling(self, factory_L2, approx_mps_config, factory_approx_sv_L2):
        super().test_sampling(factory_L2, approx_mps_config, factory_approx_sv_L2, NUM_TESTS_PER_CONFIG)


class TestPauliExpectationCache:
    """Pauli convenience inputs memoize NetworkOperator instances."""

    def test_repeated_pauli_string_reuses_cached_network_operator(self):
        dtype = "complex128"
        state_dims = (2, 2)
        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        eye = np.eye(2, dtype=dtype)
        state.apply_tensor_operator((0,), eye, unitary=True)
        state.apply_tensor_operator((1,), eye, unitary=True)
        with state:
            assert len(state._pauli_network_operator_cache) == 0
            e0 = state.compute_expectation("ZI")
            assert len(state._pauli_network_operator_cache) == 1
            op_first = next(iter(state._pauli_network_operator_cache.values()))
            e1 = state.compute_expectation("ZI")
            assert len(state._pauli_network_operator_cache) == 1
            assert op_first is next(iter(state._pauli_network_operator_cache.values()))
            assert TensorBackend.verify_close(e0, e1)

    def test_pauli_dict_order_normalized_for_same_cache_entry(self):
        dtype = "complex128"
        state_dims = (2, 2)
        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        eye = np.eye(2, dtype=dtype)
        state.apply_tensor_operator((0,), eye, unitary=True)
        state.apply_tensor_operator((1,), eye, unitary=True)
        with state:
            state.compute_expectation({"ZI": 1j, "IZ": (2 + 1j)})
            assert len(state._pauli_network_operator_cache) == 1
            key_before = next(iter(state._pauli_network_operator_cache.keys()))
            op_before = state._pauli_network_operator_cache[key_before]
            state.compute_expectation({"IZ": (2 + 1j), "ZI": 1j})
            assert len(state._pauli_network_operator_cache) == 1
            assert key_before == next(iter(state._pauli_network_operator_cache.keys()))
            assert op_before is next(iter(state._pauli_network_operator_cache.values()))

    def test_structural_change_clears_pauli_operator_cache(self):
        dtype = "complex128"
        state_dims = (2, 2)
        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        eye = np.eye(2, dtype=dtype)
        state.apply_tensor_operator((0,), eye, unitary=True)
        state.apply_tensor_operator((1,), eye, unitary=True)
        with state:
            state.compute_expectation("ZI")
            assert len(state._pauli_network_operator_cache) == 1
            state.apply_tensor_operator((0,), eye, unitary=True)
            assert len(state._pauli_network_operator_cache) == 0

    def test_repeated_pauli_string_reuses_on_gradients_path(self):
        """Pauli observables memoize on compute_expectation_with_gradients, not only compute_expectation."""
        dtype = "complex128"
        state_dims = (2, 2)
        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        eye = np.eye(2, dtype=dtype)
        state.apply_tensor_operator((0,), eye, unitary=True, gradient=True)
        state.apply_tensor_operator((1,), eye, unitary=True)
        with state:
            assert len(state._pauli_network_operator_cache) == 0
            exp0, g0 = state.compute_expectation_with_gradients("ZI", 1.0)
            assert len(state._pauli_network_operator_cache) == 1
            op_first = next(iter(state._pauli_network_operator_cache.values()))
            exp1, g1 = state.compute_expectation_with_gradients("ZI", 1.0)
            assert len(state._pauli_network_operator_cache) == 1
            assert op_first is next(iter(state._pauli_network_operator_cache.values()))
            assert TensorBackend.verify_close(exp0, exp1)
            assert len(g0) == len(g1) == 1
            tid = next(iter(g0.keys()))
            assert np.allclose(g0[tid], g1[tid])


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="torch with CUDA required",
)
class TestExpectationTorchAutograd:
    """``compute_expectation`` PyTorch autograd integration (``_TorchExpectation``)."""

    def test_returns_tensor_on_graph_and_backward(self):
        """compute_expectation returns a Torch tensor on the graph and backward propagates gradients."""
        dtype = torch.complex128
        device = "cuda"
        theta = torch.tensor(torch.pi / 8, dtype=torch.float64, device=device, requires_grad=True)
        cy = torch.cos(theta / 2)
        sy = torch.sin(theta / 2)
        g0 = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).to(dtype)

        eye1 = torch.eye(2, dtype=dtype, device=device)

        state = NetworkState((2, 2), dtype="complex128", config=TNConfig())
        state.apply_tensor_operator((0,), g0, unitary=True)
        state.apply_tensor_operator((1,), eye1, unitary=True)

        with state:
            e = state.compute_expectation("ZI")
            loss = e.real**2
            loss.backward()

        assert isinstance(e, torch.Tensor)
        assert e.requires_grad
        assert theta.grad is not None
        assert bool(torch.isfinite(theta.grad).item())

    def test_return_norm_tensor_on_graph_and_backward(self):
        """compute_expectation(return_norm=True) returns tensors on the graph; backward uses norm adjoint."""
        dtype = torch.complex128
        device = "cuda"
        theta = torch.tensor(torch.pi / 8, dtype=torch.float64, device=device, requires_grad=True)
        cy = torch.cos(theta / 2)
        sy = torch.sin(theta / 2)
        g0 = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).to(dtype)
        eye1 = torch.eye(2, dtype=dtype, device=device)

        state = NetworkState((2, 2), dtype="complex128", config=TNConfig())
        state.apply_tensor_operator((0,), g0, unitary=True)
        state.apply_tensor_operator((1,), eye1, unitary=True)

        loss_fn = expectation_gradient_loss_factory("e2_plus_3j_n")
        with state:
            e, n = state.compute_expectation("ZI", return_norm=True)
            loss = loss_fn(e, n)
            loss.backward()

        assert isinstance(e, torch.Tensor) and isinstance(n, torch.Tensor)
        assert e.requires_grad and n.requires_grad
        assert theta.grad is not None
        assert bool(torch.isfinite(theta.grad).item())

    @pytest.mark.parametrize(
        "loss_variant",
        ("e_over_n", "e2_plus_3j_n"),
    )
    @pytest.mark.parametrize(
        "config",
        expectation_gradient_L0_torch
        + expectation_gradient_L1_torch
        + expectation_gradient_L2_torch,
    )
    def test_matches_cutn(self, config, loss_variant):
        """Gate grads from autograd match explicit ``compute_expectation_with_gradients``."""
        return_norm = config.get("return_norm", False)
        if not return_norm and loss_variant in ("e_over_n", "e_times_n"):
            pytest.skip("`e_over_n` and `e_times_n` require ``return_norm=True``")

        device = torch.device("cuda")
        dtype = config["dtype"]
        factory = config["factory"]
        state_dims = factory.state_dims
        gate_sequence = factory.get_gate_sequence_for_reference()
        hamiltonian = prepare_expectation_gradient_hamiltonian(config)
        loss_fn = expectation_gradient_loss_factory(loss_variant)
        torch_ref = TorchRef()

        (
            state_autograd,
            state_cutn,
            _,
            _,
            gradient_tensor_ids,
            trainable_gates,
            _,
        ) = build_torch_network_state_pair_for_exp_grad(config, device)

        try:
            with state_autograd:
                if return_norm:
                    e, n = state_autograd.compute_expectation(hamiltonian, return_norm=True)
                    assert isinstance(e, torch.Tensor) and isinstance(n, torch.Tensor)
                    loss = loss_fn(e, n)
                    exp_detached, norm_detached = e.detach(), n.detach()
                else:
                    e = state_autograd.compute_expectation(hamiltonian, return_norm=False)
                    assert isinstance(e, torch.Tensor)
                    norm_detached = None
                    loss = loss_fn(e, None)
                    exp_detached = e.detach()

                _, adj_e_np, adj_n_np = torch_ref.adjoints_from_expectation_and_norm(
                    exp_detached,
                    norm_detached,
                    loss_fn=loss_fn,
                    return_cutn_adjoint_numpy=True,
                    cutn_dtype=dtype,
                )
                loss.backward()

            with state_cutn:
                if return_norm:
                    exp_cutn, norm_cutn, gradients_cutn = state_cutn.compute_expectation_with_gradients(
                        hamiltonian,
                        adj_e_np,
                        return_norm=True,
                        state_norm_adjoint=adj_n_np,
                    )
                else:
                    exp_cutn, gradients_cutn = state_cutn.compute_expectation_with_gradients(
                        hamiltonian,
                        adj_e_np,
                        return_norm=False,
                        state_norm_adjoint=None,
                    )
                    norm_cutn = None
        except EOverNDenominatorTooSmall as exc:
            pytest.skip(str(exc))

        tol = get_contraction_tolerance(dtype)
        grad_tol = get_expectation_gradient_tolerance(dtype, return_norm=return_norm)
        exp_from_autograd = expectation_as_real(exp_detached.cpu().item(), dtype)
        exp_cutn_real = expectation_as_real(exp_cutn, dtype)
        assert np.allclose(exp_from_autograd, exp_cutn_real, **tol), (exp_from_autograd, exp_cutn_real)
        if return_norm:
            assert norm_cutn is not None
            assert np.allclose(
                norm_as_real(norm_detached, dtype), norm_as_real(norm_cutn, dtype), **tol
            ), (norm_detached, norm_cutn)

        assert_torch_gate_grads_match_cutn(trainable_gates, gradients_cutn, grad_tol)

    def test_fallback_no_trainable_gates_returns_torch_scalars_off_graph(self):
        """Without trainable gates, ``compute_expectation`` returns 0-D tensors not on the graph."""
        dtype = torch.complex128
        device = torch.device("cuda")
        eye = torch.eye(2, dtype=dtype, device=device)

        state = NetworkState((2, 2), dtype="complex128", config=TNConfig())
        state.apply_tensor_operator((0,), eye, unitary=True, gradient=False)
        state.apply_tensor_operator((1,), eye, unitary=True, gradient=False)

        with state:
            out = state.compute_expectation("ZI", return_norm=False)
            assert isinstance(out, torch.Tensor)
            assert out.ndim == 0
            assert not out.requires_grad
            out_pair = state.compute_expectation("ZI", return_norm=True)
            assert isinstance(out_pair[0], torch.Tensor) and isinstance(out_pair[1], torch.Tensor)
            assert out_pair[0].ndim == 0 and out_pair[1].ndim == 0
            assert not out_pair[0].requires_grad and not out_pair[1].requires_grad

    def test_disabled_under_no_grad_returns_torch_scalars_off_graph(self):
        """With ``torch.no_grad()``, expectation outputs are not on the autograd graph."""
        config = expectation_gradient_L0_torch[0]
        device = torch.device("cuda")
        state, hamiltonian, _, _, trainable_gates, _ = build_torch_network_state_for_exp_grad(
            config, device
        )
        assert len(trainable_gates) > 0

        with state:
            with torch.no_grad():
                out = state.compute_expectation(hamiltonian, return_norm=False)
            assert isinstance(out, torch.Tensor)
            assert out.ndim == 0
            assert not out.requires_grad

    @pytest.mark.parametrize(
        "apply_kwargs",
        [
            {"diagonal": True},
            {"control_modes": (1,)},
        ],
    )
    def test_requires_grad_unsupported_operator_raises(self, apply_kwargs):
        """Torch ``requires_grad=True`` on controlled/diagonal ops must not be silently ignored."""
        dtype = torch.complex128
        device = torch.device("cuda")
        state = NetworkState((2, 2), dtype="complex128", config=TNConfig())
        if apply_kwargs.get("diagonal"):
            gate = torch.tensor([1.0 + 0j, -1.0 + 0j], dtype=dtype, device=device, requires_grad=True)
        else:
            gate = torch.eye(2, dtype=dtype, device=device, requires_grad=True)
        with pytest.raises(ValueError, match="Gradient registration is only supported for non-controlled, non-diagonal."):
            state.apply_tensor_operator((0,), gate, unitary=True, **apply_kwargs)

    def test_gradient_true_without_requires_grad(self):
        """``gradient=True`` registers CUTN; autograd still requires ``requires_grad=True``."""
        dtype = torch.complex128
        device = torch.device("cuda")
        theta = torch.tensor(torch.pi / 8, dtype=torch.float64, device=device, requires_grad=False)
        cy = torch.cos(theta / 2)
        sy = torch.sin(theta / 2)
        g0 = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).to(dtype)
        eye1 = torch.eye(2, dtype=dtype, device=device)

        state = NetworkState((2, 2), dtype="complex128", config=TNConfig())
        state.apply_tensor_operator((0,), g0, unitary=True, gradient=True)
        state.apply_tensor_operator((1,), eye1, unitary=True)

        with state:
            out = state.compute_expectation("ZI", return_norm=False)
            assert not out.requires_grad
            with pytest.raises(RuntimeError, match="does not require grad"):
                out.backward(torch.ones_like(out))
            assert theta.grad is None

            _, grads = state.compute_expectation_with_gradients(
                "ZI", np.array(1.0 + 0j, dtype=np.complex128)
            )
            assert len(grads) == 1
            
    def test_loss_without_norm_term_matches_cutn(self):
        """``return_norm=True`` but loss depends only on E; zero norm adjoint; grads match CUTN."""
        # Hermitian + unitary gates (no ``return_norm`` in config); API still requests norm forward.
        config = expectation_gradient_L0_torch[0]
        device = torch.device("cuda")
        dtype = config["dtype"]
        loss_fn = expectation_gradient_loss_factory("e2_plus_3j_n")

        state_autograd, state_cutn, hamiltonian, _, _, trainable_gates, _ = (
            build_torch_network_state_pair_for_exp_grad(config, device)
        )

        with state_autograd:
            e, n = state_autograd.compute_expectation(hamiltonian, return_norm=True)
            loss = loss_fn(e, None)
            _, adj_e_np, _ = TorchRef().adjoints_from_expectation_and_norm(
                e.detach(),
                None,
                loss_fn=loss_fn,
                return_cutn_adjoint_numpy=True,
                cutn_dtype=dtype,
            )
            loss.backward()

        grad_tol = get_expectation_gradient_tolerance(dtype, return_norm=True)
        adj_n_np = np.array(0.0, dtype=np.dtype(dtype))
        with state_cutn:
            _, _, gradients_cutn = state_cutn.compute_expectation_with_gradients(
                hamiltonian,
                adj_e_np,
                return_norm=True,
                state_norm_adjoint=adj_n_np,
            )
        assert_torch_gate_grads_match_cutn(trainable_gates, gradients_cutn, grad_tol)


class TestExpectationGradient:
    """Test compute_expectation_with_gradients against the TorchRef implementation."""

    def test_gradient_tensor_ids_sorted_apply_order(self):
        """gradient_tensor_ids returns ascending IDs matching apply order for registered gates."""
        dtype = "complex128"
        state = NetworkState((2, 2, 2), dtype=dtype, config=TNConfig())
        id0 = state.apply_tensor_operator((0,), np.eye(2, dtype=dtype), unitary=True, gradient=False)
        id1 = state.apply_tensor_operator((1,), np.eye(2, dtype=dtype), unitary=True, gradient=True)
        id2 = state.apply_tensor_operator((2,), np.eye(2, dtype=dtype), unitary=True, gradient=True)
        assert state.gradient_tensor_ids() == (id1, id2)
        assert id0 < id1 < id2

        state_empty = NetworkState((2,), dtype=dtype, config=TNConfig())
        state_empty.apply_tensor_operator((0,), np.eye(2, dtype=dtype), unitary=True, gradient=False)
        assert state_empty.gradient_tensor_ids() == ()

    @pytest.mark.parametrize(
        "loss_variant",
        ("e_over_n", "e2_plus_3j_n", "e_times_n", "linear_affine_e_n"),
    )
    @pytest.mark.parametrize(
        "config",
        (
            ExpectationGradientConfig.L0()
            + ExpectationGradientConfig.L1()
            + ExpectationGradientConfig.L2()
        ),
    )
    def test_expectation_gradient_vs_reference(self, config, loss_variant):
        """CUTN vs TorchRef for parametrized real losses ``f(E,N)``.

        ``e_over_n`` / ``e_times_n`` need ``return_norm=True``. ``e2_plus_3j_n`` and ``linear_affine_e_n`` also
        run with ``return_norm=False`` (no ``N`` term in the loss).
        """
        if torch is None:
            pytest.skip("torch is required for expectation gradient reference tests")
        return_norm = config.get("return_norm", False)
        if not return_norm and loss_variant in ("e_over_n", "e_times_n"):
            pytest.skip("`e_over_n` and `e_times_n` require ``N`` (``return_norm=True``)")

        factory = config["factory"]
        state_dims = factory.state_dims
        gate_sequence = factory.get_gate_sequence_for_reference()
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
        for item in gate_sequence:
            modes, gate_tensor, requires_grad = item[0], item[1], item[2]
            is_unitary = item[3] if len(item) > 3 else True
            tensor_id = state.apply_tensor_operator(modes, gate_tensor, unitary=is_unitary, gradient=requires_grad)
            if requires_grad:
                gradient_tensor_ids.append(tensor_id)

        device = torch.device("cpu")
        torch_ref = TorchRef()
        torch_dtype = torch_ref._torch_dtype(dtype)
        torch_asarray = _get_backend_asarray_func(torch)
        loss_fn = expectation_gradient_loss_factory(loss_variant)
        hamiltonian_terms = torch_ref.create_hamiltonian_terms(
            state_dims,
            hamiltonian,
            dtype=dtype,
            device=device,
            torch_dtype=torch_dtype,
            torch_asarray=torch_asarray,
        )
        gate_tensors, _ = torch_ref._as_param_gate_tensors(
            gate_sequence, torch_dtype=torch_dtype, device=device
        )
        any_non_unitary = torch_ref._has_any_non_unitary(gate_sequence)
        E, N = torch_ref.primal_E_N(
            state_dims,
            gate_sequence,
            gate_tensors,
            hamiltonian_terms,
            torch_dtype=torch_dtype,
            device=device,
            torch_asarray=torch_asarray,
            return_norm=return_norm,
            any_non_unitary=any_non_unitary,
        )
        try:
            _, adj_e_np, adj_n_np = torch_ref.adjoints_from_expectation_and_norm(
                E,
                N,
                loss_fn=loss_fn,
                return_cutn_adjoint_numpy=True,
                cutn_dtype=dtype,
            )

            with state:
                if return_norm:
                    exp_cutn, norm_cutn, gradients_cutn = state.compute_expectation_with_gradients(
                        hamiltonian,
                        adj_e_np,
                        return_norm=True,
                        state_norm_adjoint=adj_n_np,
                    )
                else:
                    exp_cutn, gradients_cutn = state.compute_expectation_with_gradients(
                        hamiltonian,
                        adj_e_np,
                        return_norm=False,
                        state_norm_adjoint=None,
                    )
                    norm_cutn = None

            exp_ref, norm_ref, gradients_ref_list = torch_ref.compute_expectation_with_gradients(
                state_dims,
                gate_sequence,
                hamiltonian,
                dtype=dtype,
                return_norm=return_norm,
                loss_fn=loss_fn,
            )
        except EOverNDenominatorTooSmall as exc:
            pytest.skip(str(exc))

        tol = get_contraction_tolerance(dtype)
        exp_cutn_real = expectation_as_real(exp_cutn, dtype)
        exp_ref_arr = expectation_as_real(exp_ref, dtype)
        assert np.allclose(exp_cutn_real, exp_ref_arr, **tol), (exp_cutn_real, exp_ref_arr)
        if return_norm:
            assert norm_cutn is not None, "norm_cutn should not be None when return_norm=True"
            assert norm_ref is not None, "norm_ref should not be None when return_norm=True"
            assert TensorBackend.verify_close(norm_cutn, norm_ref, **tol), (norm_cutn, norm_ref)
        grad_tol = get_expectation_gradient_tolerance(dtype, return_norm=return_norm)
        assert_gradients_match(gate_sequence, gradients_cutn, gradients_ref_list, grad_tol, gradient_tensor_ids=gradient_tensor_ids)

    def test_expectation_gradient_non_unitary_explicit_expectation_adjoint_only(self):
        """Non-unitary marked gates with explicit ``expectation_value_adjoint`` only (no norm path).

        Exercises ``return_norm=False`` / ``state_norm_adjoint=None`` while gates are explicitly
        non-unitary. Reference uses :class:`TorchRefExplicitAdjoints` (per-term ``grad_outputs``).
        """
        if torch is None:
            pytest.skip("torch is required for expectation gradient reference tests")

        dtype = "complex128"
        factory = create_state_factory(
            4,
            dtype,
            "SDSD",
            np.random.default_rng(71),
            backend="numpy",
            mark_gradients=True,
            mark_non_unitary=True,
        )
        state_dims = factory.state_dims
        gate_sequence = factory.get_gate_sequence_for_reference()
        assert any(len(g) > 3 and not g[3] for g in gate_sequence), "expected at least one non-unitary gate"

        hamiltonian = {"ZZII": 2.0, "IXIZ": 1.0 + 0.5j}
        adj_e = np.array(2.25 - 1.375j, dtype=dtype)

        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        gradient_tensor_ids = []
        for item in gate_sequence:
            modes, gate_tensor, requires_grad = item[0], item[1], item[2]
            is_unitary = item[3] if len(item) > 3 else True
            tensor_id = state.apply_tensor_operator(modes, gate_tensor, unitary=is_unitary, gradient=requires_grad)
            if requires_grad:
                gradient_tensor_ids.append(tensor_id)

        torch_ref = TorchRefExplicitAdjoints()
        exp_ref, norm_ref, gradients_ref_list = torch_ref.compute_expectation_with_gradients(
            state_dims,
            gate_sequence,
            hamiltonian,
            dtype=dtype,
            expectation_value_adjoint=adj_e,
            return_norm=False,
            state_norm_adjoint=None,
        )
        assert norm_ref is None

        with state:
            exp_cutn, gradients_cutn = state.compute_expectation_with_gradients(
                hamiltonian,
                adj_e,
                return_norm=False,
                state_norm_adjoint=None,
            )

        tol = get_contraction_tolerance(dtype)
        exp_cutn_real = expectation_as_real(exp_cutn, dtype)
        exp_ref_arr = expectation_as_real(exp_ref, dtype)
        assert np.allclose(exp_cutn_real, exp_ref_arr, **tol), (exp_cutn_real, exp_ref_arr)

        grad_tol = get_expectation_gradient_tolerance(dtype, return_norm=False)
        assert_gradients_match(
            gate_sequence, gradients_cutn, gradients_ref_list, grad_tol, gradient_tensor_ids=gradient_tensor_ids
        )

    def test_expectation_gradient_allows_no_marked_operators(self):
        """compute_expectation_with_gradients matches compute_expectation when no gradient=True gates."""
        state_dims = (2, 2)
        dtype = "complex128"
        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        state.apply_tensor_operator((0,), np.eye(2, dtype=dtype), unitary=True)
        state.apply_tensor_operator((1,), np.eye(2, dtype=dtype), unitary=True)
        with state:
            exp_fwd = state.compute_expectation("ZI", return_norm=False)
            exp_g, grads = state.compute_expectation_with_gradients("ZI", np.array(1.0 + 0j, dtype=dtype))
        assert grads == {}
        tol = {"atol": 1e-9, "rtol": 1e-9}
        assert TensorBackend.verify_close(exp_fwd, exp_g, **tol)

    def test_expectation_gradient_requires_paired_norm_args(self):
        """return_norm and state_norm_adjoint must both be off or both on."""
        dtype = "complex128"
        state = NetworkState((2, 2), dtype=dtype, config=TNConfig())
        state.apply_tensor_operator((0,), np.eye(2, dtype=dtype), unitary=True, gradient=True)
        state.apply_tensor_operator((1,), np.eye(2, dtype=dtype), unitary=True)
        with state:
            with pytest.raises(ValueError, match=r"return_norm and state_norm_adjoint"):
                state.compute_expectation_with_gradients("ZI", 1.0, return_norm=True, state_norm_adjoint=None)
            with pytest.raises(ValueError, match=r"return_norm and state_norm_adjoint"):
                state.compute_expectation_with_gradients("ZI", 1.0, return_norm=False, state_norm_adjoint=1.0)

    @pytest.mark.parametrize("return_norm", (False, True))
    def test_expectation_gradient_simplified_all_marked_gates_zero(self, return_norm):
        """Differentiable gates can drop out of the expectation TN (zero ∂⟨O⟩/∂G); buffers must zero."""
        dtype = "complex128"
        state_dims = (2, 2)
        theta = np.pi / 7
        def ry_mat(theta):
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            return np.array([[c, -s], [s, c]], dtype=dtype)
        adj_e = np.array(2.25 - 1.375j, dtype=dtype)
        pauli_obs = {"IZ": 1.0}
        # |ψ⟩ = Ry(θ)|0⟩ ⊗ |0⟩  ⇒  ⟨I⊗Z⟩ = +1 independent of θ.
        e0 = np.array([1, 0], dtype=dtype)
        psi = np.kron(ry_mat(theta) @ e0, e0)

        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        gid = state.apply_tensor_operator((0,), ry_mat(theta), unitary=True, gradient=True)
        state.apply_tensor_operator((1,), np.eye(2, dtype=dtype), unitary=True)

        ref_expectation = np.array(1.0 + 0.0j, dtype=dtype)

        tol = {"atol": 1e-9, "rtol": 1e-9}
        kw = dict(return_norm=False, state_norm_adjoint=None)
        if return_norm:
            adj_n = np.array(-0.5 + 2.875j, dtype=dtype)
            kw = dict(return_norm=True, state_norm_adjoint=adj_n)
        
        with state:
            if return_norm:
                exp_g, norm_g, grads = state.compute_expectation_with_gradients(pauli_obs, adj_e, **kw)
            else:
                exp_g, grads = state.compute_expectation_with_gradients(pauli_obs, adj_e, **kw)
                norm_g = None

        assert np.allclose(expectation_as_real(exp_g, dtype), expectation_as_real(ref_expectation, dtype), **tol)

        gate_grad = grads[gid]
        if infer_object_package(gate_grad) != "numpy":
            gate_grad = TensorBackend.to_numpy(gate_grad)
        assert gate_grad.dtype == np.dtype(dtype)
        assert np.allclose(gate_grad, 0 + 0j, atol=2e-7, rtol=0), gate_grad

        if return_norm:
            assert norm_g is not None
            psi_norm_sq = float(np.vdot(psi, psi).real)
            assert norm_g == pytest.approx(psi_norm_sq, abs=5e-7, rel=0)

    def test_expectation_gradient_simplified_gate_zero_remainder_gate_nonzero(self):
        """Product state Ry₀|0⟩ ⊗ Ry₁|0⟩ with observable ``IZ``: qubit‑0 Ry simplifies out (∂⟨IZ⟩/∂θ₀ = 0) while qubit‑1 does not.

        Exercises initializer zeroing + backward: buffers for gates absent from the effective gradient subgraph
        must stay identically zero, while another marked gate picks up a non‑trivial ∂⟨IZ⟩.
        """
        dtype = "complex128"

        def ry_mat(theta):
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            return np.array([[c, -s], [s, c]], dtype=dtype)

        theta0 = np.pi / 5
        theta1 = np.pi / 4
        state_dims = (2, 2)
        pauli_obs = {"IZ": 1.0}
        adj_e = np.array(1.0 + 0.0j, dtype=dtype)
        tol_zero = {"atol": 2e-7, "rtol": 0}

        state = NetworkState(state_dims, dtype=dtype, config=TNConfig())
        gid_unc = state.apply_tensor_operator((0,), ry_mat(theta0), unitary=True, gradient=True)
        gid_coupled = state.apply_tensor_operator((1,), ry_mat(theta1), unitary=True, gradient=True)

        ref_expectation = np.cos(theta1)

        with state:
            exp_g, grads = state.compute_expectation_with_gradients(pauli_obs, adj_e)

        exp_out = expectation_as_real(exp_g, dtype)
        assert np.isclose(exp_out, ref_expectation, atol=5e-8, rtol=0), (exp_out, ref_expectation)

        g0 = grads[gid_unc]
        g1 = grads[gid_coupled]
        if infer_object_package(g0) != "numpy":
            g0 = TensorBackend.to_numpy(g0)
        if infer_object_package(g1) != "numpy":
            g1 = TensorBackend.to_numpy(g1)
        assert g0.dtype == np.dtype(dtype) and g1.dtype == np.dtype(dtype)

        assert np.allclose(g0, 0 + 0j, **tol_zero), g0
        assert not np.allclose(g1, 0 + 0j, atol=5e-5, rtol=0), (
            "expected non-zero gradient from coupled Ry on the qubit touched by ``IZ``"
        )

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


class TestAdjointGate:
    """G followed by G† must act as identity on a pure state."""

    CONFIGS = (
        pytest.param(TNConfig(), id="tn"),
        pytest.param(MPSConfig(gauge_option='simple'), id="mps"),
    )

    @staticmethod
    def _random_unitary(dim, rng):
        mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        q, _ = np.linalg.qr(mat)
        return q.astype(np.complex128)

    @staticmethod
    def _random_complex(shape, rng):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)

    @pytest.mark.parametrize("config", CONFIGS)
    def test_single_qubit_gate_adjoint_cancels(self, config):
        num_qubits = 3
        dtype = "complex128"
        rng = np.random.default_rng(12345)

        G = self._random_unitary(2, rng)
        assert not np.allclose(G, G.T), "gate must be non-symmetric to distinguish G† from G*"

        with NetworkState((2,) * num_qubits, dtype=dtype, config=config) as ref_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                ref_state.apply_tensor_operator((q,), u, unitary=True)
            sv_ref = ref_state.compute_state_vector()

        rng = np.random.default_rng(12345)
        G = self._random_unitary(2, rng)

        with NetworkState((2,) * num_qubits, dtype=dtype, config=config) as test_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                test_state.apply_tensor_operator((q,), u, unitary=True)
            test_state.apply_tensor_operator((0,), G, unitary=True)
            test_state.apply_tensor_operator((0,), G, adjoint=True, unitary=True)
            sv_test = test_state.compute_state_vector()

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(sv_test, sv_ref, **tol,
            err_msg="G† G should cancel: statevectors must match")

    @pytest.mark.parametrize("config", CONFIGS)
    def test_two_qubit_gate_adjoint_cancels(self, config):
        num_qubits = 3
        dtype = "complex128"
        rng = np.random.default_rng(99)

        G = self._random_unitary(4, rng).reshape(2, 2, 2, 2)

        with NetworkState((2,) * num_qubits, dtype=dtype, config=config) as ref_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                ref_state.apply_tensor_operator((q,), u, unitary=True)
            sv_ref = ref_state.compute_state_vector()

        rng = np.random.default_rng(99)
        G = self._random_unitary(4, rng).reshape(2, 2, 2, 2)

        with NetworkState((2,) * num_qubits, dtype=dtype, config=config) as test_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                test_state.apply_tensor_operator((q,), u, unitary=True)
            test_state.apply_tensor_operator((0, 1), G, unitary=True)
            test_state.apply_tensor_operator((0, 1), G, adjoint=True, unitary=True)
            sv_test = test_state.compute_state_vector()

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(sv_test, sv_ref, **tol,
            err_msg="G† G should cancel: statevectors must match for 2-qubit gate")

    @pytest.mark.parametrize("config", CONFIGS)
    def test_controlled_gate_adjoint_cancels(self, config):
        num_qubits = 4
        dtype = "complex128"
        rng = np.random.default_rng(777)

        G = self._random_unitary(2, rng)
        assert not np.allclose(G, G.T), "gate must be non-symmetric to distinguish G† from G*"

        with NetworkState((2,) * num_qubits, dtype=dtype, config=config) as ref_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                ref_state.apply_tensor_operator((q,), u, unitary=True)
            sv_ref = ref_state.compute_state_vector()

        rng = np.random.default_rng(777)
        G = self._random_unitary(2, rng)

        with NetworkState((2,) * num_qubits, dtype=dtype, config=config) as test_state:
            for q in range(num_qubits):
                u = self._random_unitary(2, rng)
                test_state.apply_tensor_operator((q,), u, unitary=True)
            test_state.apply_tensor_operator((0,), G, control_modes=(1, 2), unitary=True)
            test_state.apply_tensor_operator((0,), G, control_modes=(1, 2), adjoint=True, unitary=True)
            sv_test = test_state.compute_state_vector()

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(sv_test, sv_ref, **tol,
            err_msg="controlled G† G should cancel: statevectors must match")

    @pytest.mark.parametrize("config", CONFIGS)
    def test_mpo_adjoint_matches_dagger(self, config):
        """apply_mpo(adjoint=True) must equal apply_mpo(adjoint=False) with a manually-daggered MPO.

        The Hermitian conjugate of an MPO with mode order ``pknb`` (previous bond, ket, next
        bond, bra) is obtained per site by swapping the ket and bra axes and complex-conjugating
        the values. For an arbitrary (not necessarily unitary) MPO, this is the precise semantic
        contract of the ``adjoint=True`` flag, so we test it directly rather than via M†M = I.
        """
        num_qubits = 4
        dtype = "complex128"

        # Random non-unitary 2-site MPO acting on qubits (1, 2).
        bond = 3
        mpo_shapes = [(2, bond, 2), (bond, 2, 2)]  # first: (k, n, b); last: (p, k, b)
        mpo_modes = (1, 2)

        def _build():
            rng = np.random.default_rng(2026)
            unitaries = [self._random_unitary(2, rng) for _ in range(num_qubits)]
            tensors = [self._random_complex(s, rng) for s in mpo_shapes]
            return unitaries, tensors

        def _dagger(mpo_tensors):
            """Per-site Hermitian conjugate: swap ket↔bra axes and complex-conjugate values."""
            n = len(mpo_tensors)
            out = []
            for i, t in enumerate(mpo_tensors):
                if i == 0 and n > 1:
                    out.append(np.conj(t).transpose(2, 1, 0))            # (k, n, b) -> (b, n, k)
                elif i == n - 1 and n > 1:
                    out.append(np.conj(t).transpose(0, 2, 1))            # (p, k, b) -> (p, b, k)
                else:
                    out.append(np.conj(t).transpose(0, 3, 2, 1))         # (p, k, n, b) -> (p, b, n, k)
            return out

        # Reference: explicitly apply M† via adjoint=False on the daggered tensors.
        unitaries, mpo_tensors = _build()
        mpo_dagger = _dagger(mpo_tensors)
        with NetworkState((2,) * num_qubits, dtype=dtype, config=config) as ref_state:
            for q, u in enumerate(unitaries):
                ref_state.apply_tensor_operator((q,), u, unitary=True)
            ref_state.apply_mpo(mpo_modes, mpo_dagger)
            sv_ref = ref_state.compute_state_vector()

        # Under test: same MPO with adjoint=True must produce the same state.
        unitaries, mpo_tensors = _build()
        with NetworkState((2,) * num_qubits, dtype=dtype, config=config) as test_state:
            for q, u in enumerate(unitaries):
                test_state.apply_tensor_operator((q,), u, unitary=True)
            test_state.apply_mpo(mpo_modes, mpo_tensors, adjoint=True)
            sv_test = test_state.compute_state_vector()

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(sv_test, sv_ref, **tol,
            err_msg="apply_mpo(adjoint=True) must apply the Hermitian conjugate (M†), "
                    "matching the result of directly applying the conjugate-transposed MPO tensors")

    @pytest.mark.parametrize("config", CONFIGS)
    def test_network_operator_adjoint_matches_dagger(self, config):
        """apply_network_operator(adjoint=True) must equal applying a manually-daggered operator.

        Mirrors test_mpo_adjoint_matches_dagger but exercises the explicit NetworkOperator
        construction path (NetworkOperator.append_mpo + apply_network_operator) instead of
        the apply_mpo shortcut. apply_mpo is implemented in terms of apply_network_operator
        internally, but the explicit-NetworkOperator path is the public surface for users
        who want to compose multiple operators on the same NetworkOperator instance.
        """
        num_qubits = 4
        dtype = "complex128"

        bond = 3
        mpo_shapes = [(2, bond, 2), (bond, 2, 2)]  # first: (k, n, b); last: (p, k, b)
        mpo_modes = (1, 2)

        def _build():
            rng = np.random.default_rng(20260524)
            unitaries = [self._random_unitary(2, rng) for _ in range(num_qubits)]
            tensors = [self._random_complex(s, rng) for s in mpo_shapes]
            return unitaries, tensors

        def _dagger(mpo_tensors):
            """Per-site Hermitian conjugate: swap ket↔bra axes and complex-conjugate values."""
            n = len(mpo_tensors)
            out = []
            for i, t in enumerate(mpo_tensors):
                if i == 0 and n > 1:
                    out.append(np.conj(t).transpose(2, 1, 0))
                elif i == n - 1 and n > 1:
                    out.append(np.conj(t).transpose(0, 2, 1))
                else:
                    out.append(np.conj(t).transpose(0, 3, 2, 1))
            return out

        # Reference: explicit M† via adjoint=False on a daggered NetworkOperator.
        unitaries, mpo_tensors = _build()
        mpo_dagger = _dagger(mpo_tensors)
        with NetworkState((2,) * num_qubits, dtype=dtype, config=config) as ref_state:
            for q, u in enumerate(unitaries):
                ref_state.apply_tensor_operator((q,), u, unitary=True)
            op_dagger = NetworkOperator((2,) * num_qubits, dtype=dtype)
            op_dagger.append_mpo(1.0 + 0j, mpo_modes, mpo_dagger)
            ref_state.apply_network_operator(op_dagger)
            sv_ref = ref_state.compute_state_vector()

        # Under test: adjoint=True must produce the same state.
        unitaries, mpo_tensors = _build()
        with NetworkState((2,) * num_qubits, dtype=dtype, config=config) as test_state:
            for q, u in enumerate(unitaries):
                test_state.apply_tensor_operator((q,), u, unitary=True)
            op = NetworkOperator((2,) * num_qubits, dtype=dtype)
            op.append_mpo(1.0 + 0j, mpo_modes, mpo_tensors)
            test_state.apply_network_operator(op, adjoint=True)
            sv_test = test_state.compute_state_vector()

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(sv_test, sv_ref, **tol,
            err_msg="apply_network_operator(adjoint=True) must apply the Hermitian conjugate (M†), "
                    "matching the result of directly applying the conjugate-transposed NetworkOperator")


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

    def test_mpo_tensor_factor_updates_by_linear_ids(self):
        """Each MPO tensor factor can be updated using base operator id plus factor offset."""
        cp = pytest.importorskip("cupy")
        from cuquantum.bindings import cutensornet as cutn

        dtype = "complex128"
        state_mode_extents = (2, 2, 2)
        factory = StateFactory(
            state_mode_extents,
            dtype,
            "M",
            np.random.default_rng(42),
            backend="cupy",
            mpo_bond_dim=2,
            mpo_num_sites=3,
            mpo_geometry="adjacent-ordered",
        )
        mpo_tensors, modes, _ = factory.sequence[0]

        updated_mpo = []
        for tensor in mpo_tensors:
            updated = cp.random.random(tensor.shape).astype(dtype)
            updated += 1j * cp.random.random(tensor.shape).astype(dtype)
            updated_mpo.append(updated)

        with NetworkState(state_mode_extents, dtype=dtype) as state:
            base_id = apply_factory_sequence(state, factory.sequence)[0]
            initial_sv = state.compute_state_vector()
            for i, tensor in enumerate(updated_mpo):
                cutn.state_update_tensor_operator(
                    state.handle, state.state, base_id + i, tensor.data.ptr, 0)
            updated_sv = state.compute_state_vector()

        with NetworkState(state_mode_extents, dtype=dtype) as ref:
            ref.apply_mpo(modes, updated_mpo)
            ref_sv = ref.compute_state_vector()

        assert not cp.allclose(updated_sv, initial_sv)
        cp.testing.assert_allclose(updated_sv, ref_sv, **get_contraction_tolerance(dtype))

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

        assert TensorBackend.verify_close(
            expec_test, expec_ref,
            atol=1e-12, rtol=1e-12,
        ), "expectation with multi-qubit tensor product must match numpy reference"


# ---------------------------------------------------------------------------------------------
# MPS correctness regressions checked against dense numpy references:
#   * sites no operation touches keep the user's initial MPS tensors; with C-order (row-major)
#     input and the free gauge they were once read back as column-major (silently transposed);
#   * approximate (zip-up) MPO application without truncation must be exact; splits made while
#     an MPO bond was still open were once clamped to the final MPS extents near the chain edge.
# ---------------------------------------------------------------------------------------------

def _random_mps_c_order(n, chi, rng, dtype):
    """Open-boundary qubit MPS: (left, phys, right) tensors and the pkn form with edge bonds dropped."""
    full, dim = [], 1
    for i in range(1, n + 1):
        right = min(2**i, 2**(n - i), chi)
        t = rng.standard_normal((dim, 2, right))
        if dtype.startswith('complex'):
            t = t + 1j * rng.standard_normal((dim, 2, right))
        full.append(t.astype(dtype))
        dim = right
    operands = [full[0][0]] + full[1:-1] + [full[-1][:, :, 0]]
    return full, [np.ascontiguousarray(t) for t in operands]


def _dense_from_mps(full):
    v = full[0]
    for t in full[1:]:
        v = np.tensordot(v, t, axes=(-1, 0))
    return v.reshape(-1)


def _dense_from_mpo(mpo):
    """Dense O[out, in] of an MPO in the cuTensorNet layout (first: k,n,b; middle: p,k,n,b; last: p,k,b),
    where k is contracted with the state and b is the output leg."""
    t = np.tensordot(mpo[0], mpo[1], axes=(1, 0))            # (k0, b0, k1, n1, b1) or (k0, b0, k1, b1)
    for w in mpo[2:]:
        t = np.tensordot(t, w, axes=(-2, 0))
    num_sites = len(mpo)
    t = t.reshape((2, 2) * num_sites)                         # (k0, b0, k1, b1, ...)
    t = t.transpose(list(range(1, 2 * num_sites, 2)) + list(range(0, 2 * num_sites, 2)))
    return t.reshape(2**num_sites, 2**num_sites)


def _apply_dense(op, psi, sites, n):
    """Apply op[out, in] (acting on `sites`) to the dense n-qubit state psi."""
    num_sites = len(sites)
    out = np.tensordot(op.reshape((2,) * 2 * num_sites), psi.reshape((2,) * n),
                       axes=(list(range(num_sites, 2 * num_sites)), sites))
    return np.moveaxis(out, list(range(num_sites)), sites).reshape(-1)


def _random_mpo(num_sites, bond, rng, dtype):
    shapes = [(2, bond, 2)] + [(bond, 2, bond, 2)] * (num_sites - 2) + [(bond, 2, 2)]
    out = []
    for shape in shapes:
        t = rng.standard_normal(shape)
        if dtype.startswith('complex'):
            t = t + 1j * rng.standard_normal(shape)
        out.append(t.astype(dtype))
    return out


def _unitary_mpo(num_sites, rng):
    """Exactly unitary MPO whose individual site tensors are not unitary (SVD split of a random unitary)."""
    dim = 2**num_sites
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)))
    u = q * (np.diag(r) / np.abs(np.diag(r)))
    # (out_0..out_{L-1}, in_0..in_{L-1}) -> per-site (in_i, out_i) pairs, then left-to-right SVD split
    t = u.reshape((2,) * 2 * num_sites)
    t = t.transpose([x for i in range(num_sites) for x in (num_sites + i, i)])
    mpo, left, rest = [], 1, t.reshape(1, -1)
    for i in range(num_sites - 1):
        rest = rest.reshape(left * 4, -1)
        a, s, b = np.linalg.svd(rest, full_matrices=False)
        keep = int((s > 1e-12 * s[0]).sum())
        site = a[:, :keep].reshape(left, 2, 2, keep)          # (p, k, b, n)
        mpo.append(site[0].transpose(0, 2, 1) if i == 0 else site.transpose(0, 1, 3, 2))
        rest, left = s[:keep, None] * b[:keep], keep
    mpo.append(rest.reshape(left, 2, 2))                        # (p, k, b)
    return mpo, u


def _relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


class TestMPSUntouchedInitialSites:
    """set_initial_mps with C-order entangled tensors: sites left untouched must come back unchanged."""

    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    @pytest.mark.parametrize("dtype", ("float64", "complex128"))
    @pytest.mark.parametrize("operation", ("none", "one_site_gate", "two_site_gate", "partial_mpo"))
    def test_state_vector_matches_reference(self, gauge_option, dtype, operation):
        n = 6
        rng = np.random.default_rng(7)
        full, operands = _random_mps_c_order(n, 4, rng, dtype)
        expected = _dense_from_mps(full)
        with NetworkState((2,) * n, dtype=dtype, config=MPSConfig(gauge_option=gauge_option)) as state:
            state.set_initial_mps(operands)
            if operation == "one_site_gate":
                op = rng.standard_normal((2, 2)).astype(dtype)
                state.apply_tensor_operator((2,), op)
                expected = _apply_dense(op, expected, [2], n)
            elif operation == "two_site_gate":
                op = rng.standard_normal((4, 4)).astype(dtype)
                state.apply_tensor_operator((2, 3), op.reshape(2, 2, 2, 2))
                expected = _apply_dense(op, expected, [2, 3], n)
            elif operation == "partial_mpo":
                mpo = _random_mpo(2, 2, rng, dtype)
                state.apply_mpo((2, 3), mpo)
                expected = _apply_dense(_dense_from_mpo(mpo), expected, [2, 3], n)
            actual = state.compute_state_vector().reshape(-1)
        assert _relative_error(actual, expected) < 1e-10


class TestApproxMPOUntruncatedExact:
    """Approximate MPO application without truncation must reproduce the exact result."""

    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    @pytest.mark.parametrize("start", (0, 2, 4))      # left edge, middle, right edge of an 8-site chain
    @pytest.mark.parametrize("mpo_kind", ("random", "unitary"))
    def test_state_vector_matches_dense_reference(self, gauge_option, start, mpo_kind):
        n, num_sites = 8, 4
        rng = np.random.default_rng(11)
        full, operands = _random_mps_c_order(n, 4, rng, "complex128")
        if mpo_kind == "random":
            mpo = _random_mpo(num_sites, 4, rng, "complex128")
            op = _dense_from_mpo(mpo)
        else:
            mpo, op = _unitary_mpo(num_sites, rng)
            assert np.allclose(_dense_from_mpo(mpo), op)
        sites = list(range(start, start + num_sites))
        expected = _apply_dense(op, _dense_from_mps(full), sites, n)
        config = MPSConfig(gauge_option=gauge_option, mpo_application="approximate")
        with NetworkState((2,) * n, dtype="complex128", config=config) as state:
            state.set_initial_mps(operands)
            state.apply_mpo(sites, mpo, unitary=(mpo_kind == "unitary"))
            actual = state.compute_state_vector().reshape(-1)
        assert _relative_error(actual, expected) < 1e-10


def _random_mixed_mps(dims, rng, dtype, chi=None):
    """Open-boundary MPS of qudit extents `dims` with full bonds (or bonds capped at chi), in pkn form with edge
    bonds dropped."""
    full, left = [], 1
    for i, d in enumerate(dims):
        right = min(int(np.prod(dims[:i + 1])), int(np.prod(dims[i + 1:])), chi or np.inf)
        t = rng.standard_normal((left, d, right)) + 1j * rng.standard_normal((left, d, right))
        full.append(t.astype(dtype))
        left = right
    operands = [full[0][0]] + full[1:-1] + [full[-1][:, :, 0]]
    return full, [np.ascontiguousarray(t) for t in operands]


def _random_mixed_mpo(site_dims, bond, rng, dtype):
    """Random MPO in the pknb layout; site j acts on a qudit of extent site_dims[j]."""
    num_sites = len(site_dims)
    out = []
    for j, d in enumerate(site_dims):
        shape = ((bond,) if j > 0 else ()) + (d,) + ((bond,) if j < num_sites - 1 else ()) + (d,)
        out.append((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype))
    return out


def _dense_from_mixed_mpo(mpo):
    """Dense O[out, in] of an MPO in the pknb layout with arbitrary qudit extents."""
    t = mpo[0].transpose(0, 2, 1)                              # (k0, b0, n)
    for w in mpo[1:-1]:
        t = np.moveaxis(np.tensordot(t, w, axes=(-1, 0)), -2, -1)  # (..., k, b, n)
    t = np.tensordot(t, mpo[-1], axes=(-1, 0))                 # (k0, b0, ..., k, b)
    num_sites = len(mpo)
    dim = int(np.prod([w.shape[-1] for w in mpo]))
    t = t.transpose(list(range(1, 2 * num_sites, 2)) + list(range(0, 2 * num_sites, 2)))
    return t.reshape(dim, dim)


def _apply_dense_mixed(op, psi, sites, dims):
    """Apply op[out, in] acting on `sites` (in the given order) to the dense state psi of extents `dims`."""
    site_dims = [dims[s] for s in sites]
    num_sites = len(sites)
    out = np.tensordot(op.reshape(site_dims * 2), psi.reshape(dims),
                       axes=(list(range(num_sites, 2 * num_sites)), sites))
    return np.moveaxis(out, list(range(num_sites)), sites).reshape(-1)


def _emulated_state(full, dims, sites, op, **config):
    """State vector from the test emulator after applying an MPO (list of tensors) or a gate (one tensor) on sites."""
    emulator = MPS(list(range(len(dims))), "numpy", qudit_dims=list(dims), dtype="complex128",
                   mps_tensors=[t.copy() for t in full], **config)
    if isinstance(op, (list, tuple)):
        emulator.apply_mpo(list(sites), op)
    else:
        emulator.apply_gate(list(sites), op)
    return np.asarray(emulator.compute_state_vector()).reshape(-1)


# Non-adjacent sites of mixed-extent states: the swaps that bring them together change which qudits lie on either
# side of a bond, so the bond can need more than the final MPS allows there.
_MIXED_EXTENT_NON_ADJACENT = (
    ((3, 2, 5, 7), (0, 2)),
    ((3, 2, 5, 7), (1, 3)),
    ((3, 2, 5, 7), (0, 3)),
    ((3, 2, 5, 7), (0, 2, 3)),
    ((3, 2, 5, 7), (3, 1)),
    ((2, 3, 5, 7, 2, 3), (1, 4)),
    ((2, 3, 5, 7, 2, 3), (0, 2, 5)),
    ((2, 3, 5, 7, 2, 3), (5, 0)),
    ((2, 3, 5, 7, 2, 3), (4, 0, 2)),
    ((7, 2, 2, 5, 3), (0, 4, 2)),
)


class TestMixedExtentMPO:
    """MPO on a state with mixed qudit extents: exact application, and approximate application whose ranks fit the
    requested extents, must match the dense reference; untruncated approximate application limits intermediate bonds
    to the largest output extent and must match the test emulator, which applies the same limit."""

    @staticmethod
    def _check(dims, sites, gauge_option, mpo_application, *, chi=None, mpo_bond=3, max_extent=None):
        rng = np.random.default_rng(17)
        full, operands = _random_mixed_mps(dims, rng, "complex128", chi)
        mpo = _random_mixed_mpo([dims[s] for s in sites], mpo_bond, rng, "complex128")
        expected = _apply_dense_mixed(_dense_from_mixed_mpo(mpo), _dense_from_mps(full), list(sites), list(dims))
        config = MPSConfig(gauge_option=gauge_option, mpo_application=mpo_application, max_extent=max_extent)
        with NetworkState(dims, dtype="complex128", config=config) as state:
            state.set_initial_mps(operands)
            state.apply_mpo(sites, mpo)
            actual = state.compute_state_vector().reshape(-1)
        if mpo_application == "approximate" and max_extent is None:
            expected = _emulated_state(full, dims, sites, mpo, gauge_option=gauge_option, mpo_application=mpo_application)
            assert _relative_error(actual, expected) < 1e-8
        else:
            assert _relative_error(actual, expected) < 1e-10

    @pytest.mark.parametrize("dims, sites", _MIXED_EXTENT_NON_ADJACENT)
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_non_adjacent_exact_application(self, dims, sites, gauge_option):
        self._check(dims, sites, gauge_option, "exact")

    # Approximate (zip-up) application without truncation on non-adjacent sites of an entangled state.
    @pytest.mark.parametrize("dims, sites", _MIXED_EXTENT_NON_ADJACENT)
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_non_adjacent_approximate_application(self, dims, sites, gauge_option):
        self._check(dims, sites, gauge_option, "approximate")

    # Approximate application compresses the swaps and every split to max_extent. From a product state, a cut
    # through any intermediate crosses at most k-1 bonds of a k-site MPO, so its rank is at most D**(k-1); with
    # D = 2 and k <= 3, max_extent = 4 truncates no bond that matters while staying below the physical bonds.
    @pytest.mark.parametrize("dims, sites", _MIXED_EXTENT_NON_ADJACENT)
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_non_adjacent_approximate_capped_application(self, dims, sites, gauge_option):
        self._check(dims, sites, gauge_option, "approximate", chi=1, mpo_bond=2, max_extent=4)

    # After the swaps of qudits 4 and 0, the split of qudits 0 and 2 has qudits 1 and 2 (3 * 3) on one side of a bond
    # whose final extent is 12 (qudits 0 and 1, 4 * 3), and hands the open MPO bond to that side, so it needs 9 * 2 = 18.
    # The split is bounded by the largest output extent (16), not by its own final extent (12).
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_open_mpo_bond_on_a_permuted_bond(self, gauge_option):
        self._check((4, 3, 3, 2, 2, 2, 2), (4, 0, 2, 1), gauge_option, "approximate", mpo_bond=2)

    # No bound exceeds its final extent during the sweep, so the swaps that restore the qudit order are capped. While
    # the order is still permuted, bond 1 has qudits 1 and 3 (2 * 4 = 8) on one side, more than its final extent of 6,
    # which is also the largest output extent here: the swap is bounded by the order it leaves behind and by that
    # limit, as in the emulator.
    @pytest.mark.parametrize("mpo_bond", (2, 3))
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_order_restoring_swap_on_a_permuted_bond(self, gauge_option, mpo_bond):
        self._check((3, 2, 3, 4), (2, 0, 3, 1), gauge_option, "approximate", mpo_bond=mpo_bond)

    # Random geometries: 4-5 qudits of extents 2-4, MPOs on 2-5 sites in random order, no truncation (library equals
    # the emulator).
    @pytest.mark.parametrize("seed", range(16))
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_random_permuted_application(self, gauge_option, seed):
        rng = np.random.default_rng(seed)
        num_qudits = int(rng.integers(4, 6))
        dims = tuple(int(d) for d in rng.integers(2, 5, num_qudits))
        num_sites = int(rng.integers(2, num_qudits + 1))
        sites = tuple(int(s) for s in rng.permutation(num_qudits)[:num_sites])
        self._check(dims, sites, gauge_option, "approximate", mpo_bond=int(rng.integers(2, 4)))

    # Approximate (zip-up) application without truncation on adjacent sites.
    @pytest.mark.parametrize("dims, sites", (
        ((3, 2, 5, 7), (0, 1)),
        ((3, 2, 5, 7), (1, 2, 3)),
        ((3, 2, 5, 7), (3, 2, 1, 0)),
        ((2, 3, 5, 7, 2, 3), (1, 2, 3, 4)),
        ((2, 3, 5, 7, 2, 3), (5, 4, 3)),
    ))
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_adjacent_approximate_application(self, dims, sites, gauge_option):
        self._check(dims, sites, gauge_option, "approximate")


class TestMixedExtentNonAdjacentGate:
    """Two-qudit gate on non-adjacent qudits of an entangled mixed-extent state: without truncation options the swaps
    are exact and the result matches the dense reference; with a cutoff the swaps are bounded (intermediate bonds at
    most the largest output extent) and the result matches the test emulator."""

    @pytest.mark.parametrize("dims, sites", (
        ((2, 7, 3), (0, 2)),
        ((2, 7, 3), (2, 0)),
        ((3, 2, 5, 7), (0, 3)),
        ((2, 3, 5, 7, 2, 3), (1, 4)),
        ((2, 3, 5, 7, 2, 3), (5, 0)),
        ((7, 2, 2, 5, 3), (0, 4)),
    ))
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    @pytest.mark.parametrize("rel_cutoff", (None, 1e-14), ids=("exact", "rel_cutoff"))
    def test_state_vector(self, dims, sites, gauge_option, rel_cutoff):
        rng = np.random.default_rng(23)
        full, operands = _random_mixed_mps(dims, rng, "complex128")
        site_dims = [dims[s] for s in sites]
        dim = int(np.prod(site_dims))
        op = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        config = MPSConfig(gauge_option=gauge_option, rel_cutoff=rel_cutoff)
        with NetworkState(dims, dtype="complex128", config=config) as state:
            state.set_initial_mps(operands)
            state.apply_tensor_operator(sites, op.reshape(site_dims * 2))
            mps = state.compute_output_state()
            actual = state.compute_state_vector().reshape(-1)
        if rel_cutoff is None:
            expected = _apply_dense_mixed(op, _dense_from_mps(full), list(sites), list(dims))
            assert _relative_error(actual, expected) < 1e-10
            # the bonds enlarged by the swaps are back at their largest exact extents
            exact = [min(math.prod(dims[:i + 1]), math.prod(dims[i + 1:])) for i in range(len(dims) - 1)]
            assert [t.shape[-1] for t in mps[:-1]] == exact
        else:
            expected = _emulated_state(full, dims, sites, op.reshape(site_dims * 2), gauge_option=gauge_option,
                                       rel_cutoff=rel_cutoff)
            assert _relative_error(actual, expected) < 1e-8

    # Multi-controlled operators are applied as MPOs, with swaps that bring the qudits together and restore the order;
    # exact MPO application without truncation options is exact.
    @pytest.mark.parametrize("dims, controls, target", (
        ((3, 2, 3, 4), (2, 0), 3),
        ((3, 2, 3, 4), (3, 0), 1),
        ((2, 3, 5, 7, 2, 3), (5, 0), 3),
        ((2, 3, 5, 7, 2, 3), (1, 4, 0), 5),
    ))
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_multi_controlled_operator(self, dims, controls, target, gauge_option):
        rng = np.random.default_rng(29)
        full, operands = _random_mixed_mps(dims, rng, "complex128")
        d = dims[target]
        block = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        control_dims = [dims[c] for c in controls]
        values = [e - 1 for e in control_dims]
        op = np.eye(math.prod(control_dims) * d, dtype=block.dtype)
        active = int(np.ravel_multi_index(values, control_dims)) * d
        op[active:active + d, active:active + d] = block
        expected = _apply_dense_mixed(op, _dense_from_mps(full), list(controls) + [target], list(dims))
        config = MPSConfig(gauge_option=gauge_option, mpo_application="exact")
        with NetworkState(dims, dtype="complex128", config=config) as state:
            state.set_initial_mps(operands)
            state.apply_tensor_operator((target,), block, control_modes=controls, control_values=values,
                                        unitary=False)
            actual = state.compute_state_vector().reshape(-1)
        assert _relative_error(actual, expected) < 1e-10


class TestApproxMPONonAdjacentFullBond:
    """Untruncated approximate MPO application on non-adjacent sites of a full-bond MPS with equal qudit extents must
    match the test emulator. The splits made while an MPO bond is open leave bonds above their targets; the swaps
    restoring the qudit order were once truncated to the targets next to those bonds, before the compression sweep,
    and lost weight."""

    @pytest.mark.parametrize("dims, sites", (
        ((2,) * 7, (3, 5, 6)),
        ((2,) * 7, (3, 6, 5)),
        ((2,) * 7, (3, 0, 1)),
        ((2,) * 8, (4, 6, 7)),
        ((2,) * 8, (3, 7, 6)),
        ((2,) * 9, (8, 1, 4, 6)),
        ((3,) * 6, (4, 0, 2)),
    ))
    @pytest.mark.parametrize("mpo_bond", (2, 4))
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_state_vector_matches_dense_reference(self, dims, sites, mpo_bond, gauge_option):
        rng = np.random.default_rng(7)
        full, operands = _random_mixed_mps(dims, rng, "complex128")
        mpo = _random_mixed_mpo([dims[s] for s in sites], mpo_bond, rng, "complex128")
        expected = _emulated_state(full, dims, sites, mpo, gauge_option=gauge_option, mpo_application="approximate")
        config = MPSConfig(gauge_option=gauge_option, mpo_application="approximate")
        with NetworkState(dims, dtype="complex128", config=config) as state:
            state.set_initial_mps(operands)
            state.apply_mpo(sites, mpo)
            actual = state.compute_state_vector().reshape(-1)
        assert _relative_error(actual, expected) < 1e-8


def _mpo_geometries(n):
    """MPO site sequences on an n-site chain: contiguous ascending/descending at either edge and in the middle,
    non-contiguous ascending and unordered."""
    return {
        "contiguous-ascending-left": (0, 1, 2, 3),
        "contiguous-ascending-middle": (2, 3, 4),
        "contiguous-ascending-right": tuple(range(n - 4, n)),
        "contiguous-descending-left": (3, 2, 1, 0),
        "contiguous-descending-right": tuple(range(n - 1, n - 5, -1)),
        "non-contiguous-ascending": (0, 2, 5),
        "non-contiguous-ascending-right": (n - 6, n - 3, n - 1),
        "non-contiguous-unordered": (n - 1, 1, 3),
        "non-contiguous-unordered-edges": (2, n - 1, 0),
    }


class TestApproxMPOGeometries:
    """Approximate MPO application across site geometries and qudit extents.

    The library must match the test emulator, which runs the same zip-up algorithm in NumPy (intermediate bounds and
    their limit at the largest output extent, post-sweep compression and simple-update gauges included)."""

    STATES = {"qubit": (2,) * 10, "qutrit": (3,) * 7, "mixed": (2, 3, 5, 7, 2, 3, 2)}

    @pytest.mark.parametrize("state", STATES)
    @pytest.mark.parametrize("geometry", _mpo_geometries(7))
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    @pytest.mark.parametrize("svd_options", ({}, {"rel_cutoff": 1e-2}, {"max_extent": 4}),
                             ids=("untruncated", "rel_cutoff", "max_extent"))
    @pytest.mark.parametrize("seed", (29, 31))
    def test_state_vector(self, state, geometry, gauge_option, svd_options, seed):
        dims = self.STATES[state]
        sites = _mpo_geometries(len(dims))[geometry]
        rng = np.random.default_rng(seed)
        # the initial MPS fits the requested extents, so only the MPO application is compared
        full, operands = _random_mixed_mps(dims, rng, "complex128", svd_options.get("max_extent", 12))
        mpo = _random_mixed_mpo([dims[s] for s in sites], 3, rng, "complex128")
        config = MPSConfig(gauge_option=gauge_option, mpo_application="approximate", **svd_options)
        with NetworkState(dims, dtype="complex128", config=config) as state_:
            state_.set_initial_mps(operands)
            state_.apply_mpo(sites, mpo)
            actual = np.asarray(state_.compute_state_vector()).reshape(-1)

        emulator = MPS(list(range(len(dims))), "numpy", qudit_dims=dims, dtype="complex128",
                       mps_tensors=[t.copy() for t in full], gauge_option=gauge_option,
                       mpo_application="approximate", **trim_mps_config(svd_options))
        emulator.apply_mpo(sites, mpo)
        assert _relative_error(actual, np.asarray(emulator.compute_state_vector()).reshape(-1)) < 1e-8


class TestSmallNormMPS:
    """An MPS whose norm is far below machine epsilon must not lose its tensors: singular values below epsilon are
    negligible only relative to the largest one, not in absolute terms."""

    @pytest.mark.parametrize("operator", ("mpo", "gate"))
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_state_matches_dense_reference(self, operator, gauge_option):
        dims, sites = (2,) * 8, (1, 4, 6)
        rng = np.random.default_rng(43)
        full, operands = _random_mixed_mps(dims, rng, "complex128")
        scale = 1e-3  # 8 tensors: norm ~1e-24 times that of the unscaled state
        full = [t * scale for t in full]
        operands = [t * scale for t in operands]
        if operator == "mpo":
            mpo = _random_mixed_mpo([dims[s] for s in sites], 2, rng, "complex128")
            expected = _apply_dense_mixed(_dense_from_mixed_mpo(mpo), _dense_from_mps(full), list(sites), list(dims))
        else:
            op = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
            expected = _apply_dense_mixed(op, _dense_from_mps(full), [1, 6], list(dims))
        config = MPSConfig(gauge_option=gauge_option, rel_cutoff=1e-14)
        with NetworkState(dims, dtype="complex128", config=config) as state:
            state.set_initial_mps(operands)
            if operator == "mpo":
                state.apply_mpo(sites, mpo)
            else:
                state.apply_tensor_operator((1, 6), op.reshape(2, 2, 2, 2))
            actual = state.compute_state_vector().reshape(-1)
        assert np.linalg.norm(actual) > 0
        assert _relative_error(actual / np.linalg.norm(actual), expected / np.linalg.norm(expected)) < 1e-10

    # Single precision with norms down to just above the smallest normal float: singular values between eps times the
    # largest one and the smallest normal float must not be inverted (their reciprocal overflows and the SVD fails).
    # The free gauge never inverts and stays accurate; simple-gauge gauges hold absolute singular values, which become
    # subnormal at such norms, so only a finite result is required there below norms of about 1e-30.
    @pytest.mark.parametrize("operator", ("mpo", "gate"))
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    @pytest.mark.parametrize("norm", (1e-28, 1e-37))
    def test_single_precision_near_float_minimum(self, operator, gauge_option, norm):
        dims, sites = (2,) * 8, (1, 4, 6)
        rng = np.random.default_rng(47)
        full, _ = _random_mixed_mps(dims, rng, "complex128")
        scale = (norm / np.linalg.norm(_dense_from_mps(full))) ** (1 / len(dims))
        full = [(t * scale).astype("complex64") for t in full]
        operands = [np.ascontiguousarray(t) for t in [full[0][0]] + full[1:-1] + [full[-1][:, :, 0]]]
        full128 = [t.astype("complex128") for t in full]
        if operator == "mpo":
            mpo = [t.astype("complex64") for t in _random_mixed_mpo([dims[s] for s in sites], 2, rng, "complex128")]
            dense_op = _dense_from_mixed_mpo([t.astype("complex128") for t in mpo])
            expected = _apply_dense_mixed(dense_op, _dense_from_mps(full128), list(sites), list(dims))
        else:
            op = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))).astype("complex64")
            expected = _apply_dense_mixed(op.astype("complex128"), _dense_from_mps(full128), [1, 6], list(dims))
        config = MPSConfig(gauge_option=gauge_option, rel_cutoff=1e-6)
        with NetworkState(dims, dtype="complex64", config=config) as state:
            state.set_initial_mps(operands)
            if operator == "mpo":
                state.apply_mpo(sites, mpo)
            else:
                state.apply_tensor_operator((1, 6), op.reshape(2, 2, 2, 2))
            actual = np.asarray(state.compute_state_vector()).reshape(-1).astype("complex128")
        assert np.all(np.isfinite(actual))
        assert np.linalg.norm(actual) > 0
        if gauge_option == "free" or norm >= 1e-30:
            assert _relative_error(actual / np.linalg.norm(actual), expected / np.linalg.norm(expected)) < 1e-3


class TestIntermediateExtentLimit:
    """In a truncated simulation, intermediate bonds are limited to the largest output bond extent; the library must
    match the emulator, which applies the same limit."""

    # Extents 2 and 7: intermediate bonds near the extent-7 block can need more than the largest bond (8); the limit
    # binds and changes the result in all but the MPO on (1, 4, 6).
    @pytest.mark.parametrize("operator, sites", (("gate", (0, 5)), ("gate", (6, 1)), ("mpo", (1, 4, 6)),
                                                 ("mpo", (7, 1, 4))))
    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    def test_matches_emulator(self, operator, sites, gauge_option):
        dims = (2, 2, 2, 7, 7, 2, 2, 2)
        rng = np.random.default_rng(41)
        full, operands = _random_mixed_mps(dims, rng, "complex128", 8)
        site_dims = [dims[s] for s in sites]
        if operator == "gate":
            dim = math.prod(site_dims)
            op = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))).reshape(site_dims * 2)
        else:
            op = _random_mixed_mpo(site_dims, 2, rng, "complex128")
        config = MPSConfig(gauge_option=gauge_option, max_extent=8)
        with NetworkState(dims, dtype="complex128", config=config) as state:
            state.set_initial_mps(operands)
            if operator == "gate":
                state.apply_tensor_operator(sites, op)
            else:
                state.apply_mpo(sites, op)
            actual = np.asarray(state.compute_state_vector()).reshape(-1)
        emulator = MPS(list(range(len(dims))), "numpy", qudit_dims=dims, dtype="complex128",
                       mps_tensors=[t.copy() for t in full], gauge_option=gauge_option, max_extent=8)
        if operator == "gate":
            emulator.apply_gate(list(sites), op)
        else:
            emulator.apply_mpo(list(sites), op)
        assert _relative_error(actual, np.asarray(emulator.compute_state_vector()).reshape(-1)) < 1e-8


def _apply_mpo_exact_mps(full, mpo, first):
    """Exact MPO application on a (left, phys, right) MPS; returns the bond chi*D MPS (mps bond major)."""
    num_sites = len(mpo)
    out = []
    for i, a in enumerate(full):
        j = i - first
        if j < 0 or j >= num_sites:
            out.append(a)
            continue
        w = mpo[j]
        if j == 0:
            w = w[None]                       # (p=1, k, n, b)
        elif j == num_sites - 1:
            w = w[:, :, None, :]              # (p, k, n=1, b)
        t = np.einsum('akc,pknb->apbcn', a, w)
        out.append(t.reshape(a.shape[0] * w.shape[0], 2, a.shape[2] * w.shape[2]))
    return out


def _log_overlap(x, y):
    """log|<x|y>| of two (left, phys, right) MPS via transfer matrices, rescaled per site."""
    env, log_scale = np.ones((1, 1)), 0.0
    for a, b in zip(x, y):
        env = np.einsum('ab,akc,bkd->cd', env, a.conj(), b, optimize=True)
        scale = np.abs(env).max()
        env, log_scale = env / scale, log_scale + np.log(scale)
    return log_scale + np.log(abs(env[0, 0]))


class TestLongMPS:
    """MPS longer than 63 qubits: physical bond products exceed int64 and must not overflow anywhere."""

    @pytest.mark.parametrize("gauge_option", ("free", "simple"))
    @pytest.mark.parametrize("start", (0, 63, 126))   # left edge, middle, right edge of a 130-site chain
    def test_mpo_on_long_chain_matches_exact(self, gauge_option, start):
        n, num_sites = 130, 4
        rng = np.random.default_rng(13)
        full, operands = _random_mps_c_order(n, 4, rng, "complex128")
        mpo = _random_mpo(num_sites, 4, rng, "complex128")
        expected = _apply_mpo_exact_mps(full, mpo, start)
        config = MPSConfig(gauge_option=gauge_option, max_extent=64)
        with NetworkState((2,) * n, dtype="complex128", config=config) as state:
            state.set_initial_mps(operands)
            state.apply_mpo(list(range(start, start + num_sites)), mpo)
            out = [np.asarray(t) for t in state.compute_output_state()]
        actual = [out[0][None]] + out[1:-1] + [out[-1][..., None]]
        infidelity = 1 - np.exp(2 * _log_overlap(actual, expected)
                                - _log_overlap(actual, actual) - _log_overlap(expected, expected))
        assert infidelity < 1e-10
