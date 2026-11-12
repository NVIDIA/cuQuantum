# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Lightweight tests for mixed-state (density matrix) simulation via NetworkState with pure_state=False.

Covers:
  - Unitary-only circuits: mixed state must reproduce pure-state RDM, expectation, and amplitude results.
  - Noisy circuits (general channels): trace preservation, known analytic results.
  - State accessors: compute_state_vector (pure-only) and compute_density_matrix (pure and mixed).
"""

import warnings

import pytest
import numpy as np

from cuquantum.tensornet.experimental import NetworkState, TNConfig, NetworkOperator

from ..utils.circuit_ifc import QuantumStateTestHelper
from ..utils.helpers import _BaseTester, get_contraction_tolerance


def random_unitary(n, rng, dtype='complex128'):
    dim = 2 ** n
    mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(mat)
    return q.reshape((2, 2) * n).astype(dtype)


PAULI_I = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def depolarizing_kraus(rate):
    return [
        np.sqrt(1 - rate) * PAULI_I,
        np.sqrt(rate / 3) * PAULI_X,
        np.sqrt(rate / 3) * PAULI_Y,
        np.sqrt(rate / 3) * PAULI_Z,
    ]


def amplitude_damping_kraus(gamma):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return [K0, K1]


def apply_brick_layer(state, n_qubits, op_one, op_two, *, unitary=True):
    """Apply a single-qubit layer followed by a brick-wall two-qubit layer."""
    for i in range(n_qubits):
        state.apply_tensor_operator((i,), op_one, unitary=unitary)
    for i in range(0, n_qubits - 1, 2):
        state.apply_tensor_operator((i, i + 1), op_two, unitary=unitary)
    for i in range(1, n_qubits - 1, 2):
        state.apply_tensor_operator((i, i + 1), op_two, unitary=unitary)


def controlled_single_target_mpo_tensors(target_gate, dtype='complex128'):
    """Construct a 2-site open-boundary MPO for |0><0|⊗I + |1><1|⊗U.

    Tensor mode order for append_mpo is:
      - first site:  (k, n, b)
      - middle site: (p, k, n, b)
      - last site:   (p, k, b)
    """
    dtype = np.dtype(dtype)
    target_gate = np.asarray(target_gate, dtype=dtype)
    assert target_gate.shape == (2, 2), "target_gate must be shape (2, 2)"
    p0 = np.array([[1, 0], [0, 0]], dtype=dtype)
    p1 = np.array([[0, 0], [0, 1]], dtype=dtype)
    ident = np.eye(2, dtype=dtype)

    w0 = np.zeros((2, 2, 2), dtype=dtype)  # (k, n, b)
    w1 = np.zeros((2, 2, 2), dtype=dtype)  # (p, k, b)
    w0[:, 0, :] = p0
    w0[:, 1, :] = p1
    w1[0, :, :] = ident
    w1[1, :, :] = target_gate
    return [w0, w1]


def cnot_mpo_tensors(dtype='complex128'):
    """Construct a 2-site open-boundary MPO for CNOT (control on mode 0, target on mode 1)."""
    x = np.array([[0, 1], [1, 0]], dtype=np.dtype(dtype))
    return controlled_single_target_mpo_tensors(x, dtype=dtype)


def _make_pauli_dict(n_qubits):
    """Build a Pauli operator dictionary whose strings match n_qubits."""
    base1 = 'ZIZI'
    base2 = 'IXIX'
    s1 = (base1 * ((n_qubits // len(base1)) + 1))[:n_qubits]
    s2 = (base2 * ((n_qubits // len(base2)) + 1))[:n_qubits]
    return {s1: 0.5, s2: 0.5}


def compute_pure_reference(n_qubits, rng, dtype='complex128'):
    """Build a pure state and return (sv, rdm_01, expectation) as references."""
    op1 = random_unitary(1, rng, dtype)
    op2 = random_unitary(2, rng, dtype)
    pauli_dict = _make_pauli_dict(n_qubits)
    with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as state:
        apply_brick_layer(state, n_qubits, op1, op2)
        sv = state.compute_state_vector()
        rdm = state.compute_reduced_density_matrix((0, 1))
        expec = state.compute_expectation(pauli_dict)
    return op1, op2, sv, rdm, expec


def rdm_from_sv(sv, where):
    """Compute RDM from a state vector by tracing out complement modes."""
    n = sv.ndim
    complement = [i for i in range(n) if i not in where]
    rho = np.tensordot(sv, sv.conj(), axes=(complement, complement))
    k = len(where)
    perm = []
    for i in range(k):
        perm.append(i)
        perm.append(i + k)
    # rho currently has modes [where_ket..., where_bra...]
    # we want interleaved [ket0, bra0, ket1, bra1, ...]
    # Actually for verification just reshape to matrix form
    return rho


class TestMixedStateUnitaryOnly(_BaseTester):
    """Mixed state with only unitary gates must reproduce pure-state results."""

    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    @pytest.mark.parametrize("dtype", ["complex64", "complex128"])
    def test_rdm_matches_pure(self, n_qubits, dtype):
        rng = np.random.default_rng(100 + n_qubits)
        op1, op2, sv_pure, rdm_pure, _ = compute_pure_reference(n_qubits, rng, dtype)

        # Reuse same RNG seed for identical operators
        rng2 = np.random.default_rng(100 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        op2b = random_unitary(2, rng2, dtype)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as state:
            apply_brick_layer(state, n_qubits, op1b, op2b)
            rdm_mixed = state.compute_reduced_density_matrix((0, 1))

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_pure, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_expectation_matches_pure(self, n_qubits):
        dtype = 'complex128'
        rng = np.random.default_rng(200 + n_qubits)
        op1, op2, _, _, expec_pure = compute_pure_reference(n_qubits, rng, dtype)

        rng2 = np.random.default_rng(200 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        op2b = random_unitary(2, rng2, dtype)

        pauli_dict = _make_pauli_dict(n_qubits)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as state:
            apply_brick_layer(state, n_qubits, op1b, op2b)
            expec_mixed = state.compute_expectation(pauli_dict)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(expec_mixed, expec_pure, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_full_rdm_is_pure_density_matrix(self, n_qubits):
        """Full RDM over all qubits should equal |psi><psi|."""
        dtype = 'complex128'
        rng = np.random.default_rng(300 + n_qubits)
        op1 = random_unitary(1, rng, dtype)
        op2 = random_unitary(2, rng, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            apply_brick_layer(pure, n_qubits, op1, op2)
            sv = pure.compute_state_vector()

        rng2 = np.random.default_rng(300 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        op2b = random_unitary(2, rng2, dtype)

        all_modes = tuple(range(n_qubits))
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            apply_brick_layer(mixed, n_qubits, op1b, op2b)
            rdm_full = mixed.compute_reduced_density_matrix(all_modes)

        dim = 2 ** n_qubits
        sv_flat = sv.flatten()
        rho_ref = np.outer(sv_flat, sv_flat.conj()).reshape((2,) * (2 * n_qubits))

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_full, rho_ref, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_marginal_probability_returns_diag_of_density_matrix(self, n_qubits):
        """compute_reduced_density_matrix(diagonal=True) on a mixed state must return diag(rho)."""
        dtype = 'complex128'
        rng2 = np.random.default_rng(350 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        op2b = random_unitary(2, rng2, dtype)

        all_modes = tuple(range(n_qubits))
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            apply_brick_layer(mixed, n_qubits, op1b, op2b)
            rho_full = mixed.compute_reduced_density_matrix(all_modes)
            probs = mixed.compute_reduced_density_matrix(all_modes, diagonal=True)

        dim = 2 ** n_qubits
        rho = np.asarray(rho_full).reshape(dim, dim)
        diag_ref = np.diag(rho).real

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(np.asarray(probs).flatten().real, diag_ref, **tol)

    def test_compute_state_vector_raises_for_mixed(self):
        """compute_state_vector is undefined for mixed states and must raise."""
        with NetworkState((2, 2), dtype='complex128', pure_state=False) as state:
            state.apply_tensor_operator((0,), PAULI_X, unitary=True)
            with pytest.raises(TypeError, match="not defined for mixed states"):
                state.compute_state_vector()

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_adjoint_gate_rdm_matches_pure(self, n_qubits):
        """apply_tensor_operator with adjoint=True on mixed must match pure-state RDM."""
        dtype = 'complex128'
        rng = np.random.default_rng(370 + n_qubits)
        op_pre = random_unitary(1, rng, dtype)
        op_adj = random_unitary(1, rng, dtype).reshape(2, 2)
        where = (0, 1)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op_pre, unitary=True)
            pure.apply_tensor_operator((1,), op_adj, unitary=True, adjoint=True)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        rng2 = np.random.default_rng(370 + n_qubits)
        op_pre_b = random_unitary(1, rng2, dtype)
        op_adj_b = random_unitary(1, rng2, dtype).reshape(2, 2)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op_pre_b, unitary=True)
            mixed.apply_tensor_operator((1,), op_adj_b, unitary=True, adjoint=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_pure, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_two_qubit_adjoint_gate_rdm_matches_pure(self, n_qubits):
        """Two-qubit adjoint gate on mixed must match pure-state RDM."""
        dtype = 'complex128'
        rng = np.random.default_rng(380 + n_qubits)
        op_pre = random_unitary(1, rng, dtype)
        op_adj = random_unitary(2, rng, dtype)
        where = (0, 1)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op_pre, unitary=True)
            pure.apply_tensor_operator((0, 1), op_adj, unitary=True, adjoint=True)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        rng2 = np.random.default_rng(380 + n_qubits)
        op_pre_b = random_unitary(1, rng2, dtype)
        op_adj_b = random_unitary(2, rng2, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op_pre_b, unitary=True)
            mixed.apply_tensor_operator((0, 1), op_adj_b, unitary=True, adjoint=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_pure, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_amplitude_matches_pure(self, n_qubits):
        """For a pure-equivalent mixed state, the diagonal element <b|rho|b> equals |<b|psi>|^2.

        Off-diagonal mixed-state amplitudes <b1|rho|b2> must equal psi[b1] * conj(psi[b2]).
        """
        dtype = 'complex128'
        rng = np.random.default_rng(350 + n_qubits)
        op1 = random_unitary(1, rng, dtype)
        op2 = random_unitary(2, rng, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            apply_brick_layer(pure, n_qubits, op1, op2)
            sv = pure.compute_state_vector()

        sv_flat = np.asarray(sv).flatten()
        probs_ref = np.abs(sv_flat) ** 2

        rng2 = np.random.default_rng(350 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        op2b = random_unitary(2, rng2, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            apply_brick_layer(mixed, n_qubits, op1b, op2b)
            for idx in range(2 ** n_qubits):
                bs = tuple(int(b) for b in format(idx, f'0{n_qubits}b'))
                amp_diag = mixed.compute_amplitude((bs, bs))
                np.testing.assert_allclose(amp_diag.real, probs_ref[idx], atol=1e-10,
                    err_msg=f"Diagonal amplitude mismatch for bitstring {bs}")
                np.testing.assert_allclose(amp_diag.imag, 0.0, atol=1e-10,
                    err_msg=f"Imaginary part nonzero for bitstring {bs}")
            # Off-diagonal: <b1|rho|b2> = psi[b1] * conj(psi[b2]) for a pure-equivalent state.
            for b1 in range(2 ** n_qubits):
                for b2 in range(2 ** n_qubits):
                    bs1 = tuple(int(b) for b in format(b1, f'0{n_qubits}b'))
                    bs2 = tuple(int(b) for b in format(b2, f'0{n_qubits}b'))
                    amp = mixed.compute_amplitude((bs1, bs2))
                    np.testing.assert_allclose(
                        amp, sv_flat[b1] * np.conj(sv_flat[b2]), atol=1e-10,
                        err_msg=f"Off-diagonal mismatch for ({bs1}, {bs2})")

    def test_sampling_matches_pure(self):
        """Sampling distribution from mixed state should match pure state."""
        n_qubits = 4
        dtype = 'complex128'
        nshots = 10000

        rng = np.random.default_rng(400)
        op1 = random_unitary(1, rng, dtype)
        op2 = random_unitary(2, rng, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            apply_brick_layer(pure, n_qubits, op1, op2)
            samples_pure = pure.compute_sampling(nshots, seed=42)

        rng2 = np.random.default_rng(400)
        op1b = random_unitary(1, rng2, dtype)
        op2b = random_unitary(2, rng2, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            apply_brick_layer(mixed, n_qubits, op1b, op2b)
            samples_mixed = mixed.compute_sampling(nshots, seed=42)

        all_bitstrings = set(samples_pure.keys()) | set(samples_mixed.keys())
        for bs in all_bitstrings:
            p_pure = samples_pure.get(bs, 0) / nshots
            p_mixed = samples_mixed.get(bs, 0) / nshots
            assert abs(p_pure - p_mixed) < 0.05, \
                f"Sampling mismatch for {bs}: pure={p_pure:.3f} mixed={p_mixed:.3f}"


class TestMixedStateNoisy(_BaseTester):
    """Mixed state with noise channels: verify trace preservation and known analytic results."""

    @pytest.mark.parametrize("depol_rate", [0.01, 0.1, 0.5])
    def test_rdm_trace_preserved(self, depol_rate):
        """Depolarizing channel should preserve the trace of the RDM."""
        n_qubits = 3
        dtype = 'complex128'
        rng = np.random.default_rng(500)
        op1 = random_unitary(1, rng, dtype)
        op2 = random_unitary(2, rng, dtype)
        kraus = depolarizing_kraus(depol_rate)

        where = (0, 1)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            apply_brick_layer(state, n_qubits, op1, op2)
            for i in range(n_qubits):
                state.apply_general_tensor_channel((i,), kraus)
            rdm = state.compute_reduced_density_matrix(where)

        dim = 2 ** len(where)
        rdm_matrix = rdm.reshape(dim, dim)
        trace = np.trace(rdm_matrix).real
        np.testing.assert_allclose(trace, 1.0, atol=1e-10)

    def test_full_depolarization_gives_maximally_mixed(self):
        """Full depolarization on every qubit should yield I/d.

        With the parametrization E(rho)=(1-p)rho + (p/3)(XrhoX+YrhoY+ZrhoZ),
        the fully depolarizing channel (E(rho)=I/2) corresponds to p=3/4.
        """
        n_qubits = 2
        dtype = 'complex128'

        hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        kraus_full = depolarizing_kraus(3.0 / 4.0)

        all_modes = tuple(range(n_qubits))
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            for i in range(n_qubits):
                state.apply_tensor_operator((i,), hadamard, unitary=True)
            for i in range(n_qubits):
                state.apply_general_tensor_channel((i,), kraus_full)
            rdm = state.compute_reduced_density_matrix(all_modes)

        dim = 2 ** n_qubits
        rdm_matrix = rdm.reshape(dim, dim)
        identity_scaled = np.eye(dim, dtype=np.complex128) / dim
        np.testing.assert_allclose(rdm_matrix, identity_scaled, atol=1e-10)

    def test_amplitude_damping_ground_state(self):
        """Full amplitude damping (gamma=1) should drive any state to |0>."""
        n_qubits = 2
        dtype = 'complex128'
        rng = np.random.default_rng(600)
        op1 = random_unitary(1, rng, dtype)

        kraus_ad = amplitude_damping_kraus(1.0)

        all_modes = tuple(range(n_qubits))
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            for i in range(n_qubits):
                state.apply_tensor_operator((i,), op1, unitary=True)
            for i in range(n_qubits):
                state.apply_general_tensor_channel((i,), kraus_ad)
            rdm = state.compute_reduced_density_matrix(all_modes)

        dim = 2 ** n_qubits
        rdm_matrix = rdm.reshape(dim, dim)
        expected = np.zeros((dim, dim), dtype=np.complex128)
        expected[0, 0] = 1.0
        np.testing.assert_allclose(rdm_matrix, expected, atol=1e-10)

    def test_unitary_channel_matches_gate(self):
        """A unitary channel with a single operator (prob=1) should match applying that gate directly."""
        n_qubits = 3
        dtype = 'complex128'
        rng = np.random.default_rng(700)
        op1 = random_unitary(1, rng, dtype)
        extra_gate = random_unitary(1, rng, dtype)

        where = (0, 1)

        # Reference: apply the extra gate as a normal operator
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as ref_state:
            for i in range(n_qubits):
                ref_state.apply_tensor_operator((i,), op1, unitary=True)
            ref_state.apply_tensor_operator((1,), extra_gate, unitary=True)
            rdm_ref = ref_state.compute_reduced_density_matrix(where)

        # Test: apply the same gate via a unitary channel with prob=1
        rng2 = np.random.default_rng(700)
        op1b = random_unitary(1, rng2, dtype)
        extra_gate_b = random_unitary(1, rng2, dtype)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as ch_state:
            for i in range(n_qubits):
                ch_state.apply_tensor_operator((i,), op1b, unitary=True)
            ch_state.apply_unitary_tensor_channel((1,), [extra_gate_b], [1.0])
            rdm_ch = ch_state.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_ch, rdm_ref, **tol)

    @pytest.mark.parametrize("depol_rate", [0.05, 0.2])
    def test_depolarizing_analytical_single_qubit(self, depol_rate):
        """
        For a single-qubit depolarizing channel on |0>:
        rho_out = (1 - p) |0><0| + p/3 (X|0><0|X + Y|0><0|Y + Z|0><0|Z)
                = (1 - 2p/3)|0><0| + (2p/3)|1><1|
        diag = [1 - 2p/3, 2p/3]
        """
        n_qubits = 1
        dtype = 'complex128'
        kraus = depolarizing_kraus(depol_rate)

        with NetworkState((2,), dtype=dtype, pure_state=False, config=TNConfig()) as state:
            state.apply_general_tensor_channel((0,), kraus)
            rdm = state.compute_reduced_density_matrix((0,))

        rdm_matrix = rdm.reshape(2, 2)
        expected = np.diag([1 - 2 * depol_rate / 3, 2 * depol_rate / 3]).astype(np.complex128)
        np.testing.assert_allclose(rdm_matrix, expected, atol=1e-12)

    def test_noise_between_gates(self):
        """Noise inserted between gate layers should produce different results than noiseless."""
        n_qubits = 4
        dtype = 'complex128'
        rng = np.random.default_rng(800)
        op1 = random_unitary(1, rng, dtype)
        op2 = random_unitary(2, rng, dtype)
        kraus = depolarizing_kraus(0.1)

        where = (0, 1)

        # Noiseless
        rng2 = np.random.default_rng(800)
        op1_a = random_unitary(1, rng2, dtype)
        op2_a = random_unitary(2, rng2, dtype)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as clean:
            apply_brick_layer(clean, n_qubits, op1_a, op2_a)
            apply_brick_layer(clean, n_qubits, op1_a, op2_a)
            rdm_clean = clean.compute_reduced_density_matrix(where)

        # Noisy (channels between two layers)
        rng3 = np.random.default_rng(800)
        op1_b = random_unitary(1, rng3, dtype)
        op2_b = random_unitary(2, rng3, dtype)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as noisy:
            apply_brick_layer(noisy, n_qubits, op1_b, op2_b)
            for i in range(n_qubits):
                noisy.apply_general_tensor_channel((i,), kraus)
            apply_brick_layer(noisy, n_qubits, op1_b, op2_b)
            rdm_noisy = noisy.compute_reduced_density_matrix(where)

        assert not np.allclose(rdm_clean, rdm_noisy, atol=1e-6), \
            "Noise should produce a different density matrix"

        # But trace should still be 1
        dim = 2 ** len(where)
        trace = np.trace(rdm_noisy.reshape(dim, dim)).real
        np.testing.assert_allclose(trace, 1.0, atol=1e-10)

    def test_noisy_amplitudes_match_rdm_diagonal(self):
        """Diagonal mixed-state amplitudes <b|rho|b> match the diagonal of the full RDM after noise."""
        n_qubits = 2
        dtype = 'complex128'
        rng = np.random.default_rng(900)
        op1 = random_unitary(1, rng, dtype)
        kraus = depolarizing_kraus(0.2)

        all_modes = tuple(range(n_qubits))
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            for i in range(n_qubits):
                state.apply_tensor_operator((i,), op1, unitary=True)
            for i in range(n_qubits):
                state.apply_general_tensor_channel((i,), kraus)
            rdm = state.compute_reduced_density_matrix(all_modes)

            dim = 2 ** n_qubits
            rdm_diag = np.diag(rdm.reshape(dim, dim)).real

            for idx in range(dim):
                bs = tuple(int(b) for b in format(idx, f'0{n_qubits}b'))
                amp = state.compute_amplitude((bs, bs))
                np.testing.assert_allclose(amp.real, rdm_diag[idx], atol=1e-10,
                    err_msg=f"Amplitude {bs} doesn't match RDM diagonal")

            prob_sum = sum(
                state.compute_amplitude(
                    (tuple(int(b) for b in format(i, f'0{n_qubits}b')),) * 2
                ).real for i in range(dim)
            )
            np.testing.assert_allclose(prob_sum, 1.0, atol=1e-10)


class TestMixedStateDiagonalGate(_BaseTester):
    """Mixed state with diagonal gates must reproduce pure-state results."""

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_diagonal_gate_rdm_matches_pure(self, n_qubits):
        dtype = 'complex128'
        rng = np.random.default_rng(1100 + n_qubits)
        op1 = random_unitary(1, rng, dtype)

        diag_elems = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        diag_elems = diag_elems.astype(np.complex128)
        diag_norm = np.sqrt(np.sum(np.abs(diag_elems) ** 2))
        diag_elems /= diag_norm

        where = (0, 1)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op1, unitary=True)
            pure.apply_tensor_operator((1,), diag_elems, diagonal=True, unitary=False)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        rng2 = np.random.default_rng(1100 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        diag_elems_b = rng2.standard_normal(2) + 1j * rng2.standard_normal(2)
        diag_elems_b = diag_elems_b.astype(np.complex128)
        diag_norm_b = np.sqrt(np.sum(np.abs(diag_elems_b) ** 2))
        diag_elems_b /= diag_norm_b

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op1b, unitary=True)
            mixed.apply_tensor_operator((1,), diag_elems_b, diagonal=True, unitary=False)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_pure, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_two_qubit_diagonal_gate_rdm_matches_pure(self, n_qubits):
        dtype = 'complex128'
        rng = np.random.default_rng(1200 + n_qubits)
        op1 = random_unitary(1, rng, dtype)

        diag_2q = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        diag_2q = diag_2q.astype(np.complex128)

        where = (0, 1)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op1, unitary=True)
            pure.apply_tensor_operator((0, 1), diag_2q, diagonal=True, unitary=False)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        rng2 = np.random.default_rng(1200 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        diag_2q_b = rng2.standard_normal((2, 2)) + 1j * rng2.standard_normal((2, 2))
        diag_2q_b = diag_2q_b.astype(np.complex128)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op1b, unitary=True)
            mixed.apply_tensor_operator((0, 1), diag_2q_b, diagonal=True, unitary=False)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_pure, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_diagonal_gate_adjoint_rdm_matches_pure(self, n_qubits):
        """Diagonal gate with adjoint=True on mixed must match pure-state RDM."""
        dtype = 'complex128'
        rng = np.random.default_rng(1150 + n_qubits)
        op1 = random_unitary(1, rng, dtype)
        diag_elems = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        diag_elems = diag_elems.astype(np.complex128)

        where = (0, 1)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op1, unitary=True)
            pure.apply_tensor_operator((1,), diag_elems, diagonal=True, unitary=False, adjoint=True)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        rng2 = np.random.default_rng(1150 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        diag_elems_b = rng2.standard_normal(2) + 1j * rng2.standard_normal(2)
        diag_elems_b = diag_elems_b.astype(np.complex128)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op1b, unitary=True)
            mixed.apply_tensor_operator((1,), diag_elems_b, diagonal=True, unitary=False, adjoint=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_pure, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_diagonal_gate_expectation_matches_pure(self, n_qubits):
        dtype = 'complex128'
        rng = np.random.default_rng(1300 + n_qubits)
        op1 = random_unitary(1, rng, dtype)

        diag_elems = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        diag_elems = diag_elems.astype(np.complex128)

        pauli_dict = _make_pauli_dict(n_qubits)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op1, unitary=True)
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), diag_elems, diagonal=True, unitary=False)
            expec_pure = pure.compute_expectation(pauli_dict)

        rng2 = np.random.default_rng(1300 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        diag_elems_b = rng2.standard_normal(2) + 1j * rng2.standard_normal(2)
        diag_elems_b = diag_elems_b.astype(np.complex128)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op1b, unitary=True)
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), diag_elems_b, diagonal=True, unitary=False)
            expec_mixed = mixed.compute_expectation(pauli_dict)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(expec_mixed, expec_pure, **tol)


class TestMixedStateControlledGate(_BaseTester):
    """Mixed state with controlled gates must reproduce pure-state results."""

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_controlled_gate_rdm_matches_pure(self, n_qubits):
        dtype = 'complex128'
        rng = np.random.default_rng(1400 + n_qubits)
        op1 = random_unitary(1, rng, dtype)
        target_gate = random_unitary(1, rng, dtype).reshape(2, 2)

        where = (0, 1)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op1, unitary=True)
            pure.apply_tensor_operator(
                (0,), target_gate,
                control_modes=(1,), control_values=(1,),
                unitary=True, immutable=True)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        rng2 = np.random.default_rng(1400 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        target_gate_b = random_unitary(1, rng2, dtype).reshape(2, 2)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op1b, unitary=True)
            mixed.apply_tensor_operator(
                (0,), target_gate_b,
                control_modes=(1,), control_values=(1,),
                unitary=True, immutable=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_pure, **tol)

    @pytest.mark.parametrize("n_qubits", [4, 5])
    def test_multi_controlled_gate_rdm_matches_pure(self, n_qubits):
        dtype = 'complex128'
        rng = np.random.default_rng(1500 + n_qubits)
        op1 = random_unitary(1, rng, dtype)
        target_gate = random_unitary(1, rng, dtype).reshape(2, 2)

        ctrl_modes = tuple(range(n_qubits - 1))
        ctrl_vals = tuple(1 for _ in ctrl_modes)
        target_mode = (n_qubits - 1,)
        where = (0, 1)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op1, unitary=True)
            pure.apply_tensor_operator(
                target_mode, target_gate,
                control_modes=ctrl_modes, control_values=ctrl_vals,
                unitary=True, immutable=True)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        rng2 = np.random.default_rng(1500 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        target_gate_b = random_unitary(1, rng2, dtype).reshape(2, 2)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op1b, unitary=True)
            mixed.apply_tensor_operator(
                target_mode, target_gate_b,
                control_modes=ctrl_modes, control_values=ctrl_vals,
                unitary=True, immutable=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_pure, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_controlled_gate_adjoint_rdm_matches_pure(self, n_qubits):
        """Controlled gate with adjoint=True on mixed must match pure-state RDM."""
        dtype = 'complex128'
        rng = np.random.default_rng(1550 + n_qubits)
        op1 = random_unitary(1, rng, dtype)
        target_gate = random_unitary(1, rng, dtype).reshape(2, 2)

        where = (0, 1)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op1, unitary=True)
            pure.apply_tensor_operator(
                (0,), target_gate,
                control_modes=(1,), control_values=(1,),
                unitary=True, immutable=True, adjoint=True)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        rng2 = np.random.default_rng(1550 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        target_gate_b = random_unitary(1, rng2, dtype).reshape(2, 2)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op1b, unitary=True)
            mixed.apply_tensor_operator(
                (0,), target_gate_b,
                control_modes=(1,), control_values=(1,),
                unitary=True, immutable=True, adjoint=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_pure, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_controlled_gate_expectation_matches_pure(self, n_qubits):
        dtype = 'complex128'
        rng = np.random.default_rng(1600 + n_qubits)
        op1 = random_unitary(1, rng, dtype)
        target_gate = random_unitary(1, rng, dtype).reshape(2, 2)
        pauli_dict = _make_pauli_dict(n_qubits)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op1, unitary=True)
            pure.apply_tensor_operator(
                (0,), target_gate,
                control_modes=(1,), control_values=(1,),
                unitary=True, immutable=True)
            expec_pure = pure.compute_expectation(pauli_dict)

        rng2 = np.random.default_rng(1600 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        target_gate_b = random_unitary(1, rng2, dtype).reshape(2, 2)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op1b, unitary=True)
            mixed.apply_tensor_operator(
                (0,), target_gate_b,
                control_modes=(1,), control_values=(1,),
                unitary=True, immutable=True)
            expec_mixed = mixed.compute_expectation(pauli_dict)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(expec_mixed, expec_pure, **tol)

    def test_controlled_gate_full_rdm_matches_pure(self):
        """Full density matrix with controlled gates should match |psi><psi|."""
        n_qubits = 3
        dtype = 'complex128'
        rng = np.random.default_rng(1700)
        op1 = random_unitary(1, rng, dtype)
        target_gate = random_unitary(1, rng, dtype).reshape(2, 2)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op1, unitary=True)
            pure.apply_tensor_operator(
                (0,), target_gate,
                control_modes=(1,), control_values=(1,),
                unitary=True, immutable=True)
            sv = pure.compute_state_vector()

        rng2 = np.random.default_rng(1700)
        op1b = random_unitary(1, rng2, dtype)
        target_gate_b = random_unitary(1, rng2, dtype).reshape(2, 2)

        all_modes = tuple(range(n_qubits))
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op1b, unitary=True)
            mixed.apply_tensor_operator(
                (0,), target_gate_b,
                control_modes=(1,), control_values=(1,),
                unitary=True, immutable=True)
            rdm_full = mixed.compute_reduced_density_matrix(all_modes)

        dim = 2 ** n_qubits
        sv_flat = sv.flatten()
        rho_ref = np.outer(sv_flat, sv_flat.conj()).reshape((2,) * (2 * n_qubits))

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_full, rho_ref, **tol)


class TestMixedStateNetworkOperator(_BaseTester):
    """Mixed-state correctness checks for apply_network_operator (tensor-product and MPO)."""

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_tensor_product_network_operator_matches_pure(self, n_qubits):
        dtype = 'complex128'
        rng = np.random.default_rng(2100 + n_qubits)
        op_pre = random_unitary(1, rng, dtype).reshape(2, 2)
        op_a = random_unitary(1, rng, dtype).reshape(2, 2)
        op_b = random_unitary(1, rng, dtype).reshape(2, 2)
        modes = ((0,), (n_qubits - 1,))
        where = (0, 1)

        # Pure reference via direct gate application:
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure_ref:
            for i in range(n_qubits):
                pure_ref.apply_tensor_operator((i,), op_pre, unitary=True)
            pure_ref.apply_tensor_operator((0,), op_a, unitary=True, immutable=True)
            pure_ref.apply_tensor_operator((n_qubits - 1,), op_b, unitary=True, immutable=True)
            rdm_ref = pure_ref.compute_reduced_density_matrix(where)

        # Pure via tensor-product NetworkOperator:
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op_pre, unitary=True)
            oper = NetworkOperator((2,) * n_qubits, dtype=dtype, options=pure.options)
            oper.append_product(1.0 + 0.0j, modes, [op_a, op_b])
            pure.apply_network_operator(oper, unitary=True, immutable=True)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        # Mixed via tensor-product NetworkOperator:
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op_pre, unitary=True)
            oper = NetworkOperator((2,) * n_qubits, dtype=dtype, options=mixed.options)
            oper.append_product(1.0 + 0.0j, modes, [op_a, op_b])
            mixed.apply_network_operator(oper, unitary=True, immutable=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_pure, rdm_ref, **tol)
        np.testing.assert_allclose(rdm_mixed, rdm_ref, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_multi_qubit_product_factor_matches_pure(self, n_qubits):
        """A single 2-qubit tensor product factor on mixed must match pure-state RDM."""
        dtype = 'complex128'
        rng = np.random.default_rng(2150 + n_qubits)
        op_pre = random_unitary(1, rng, dtype).reshape(2, 2)
        op_2q = random_unitary(2, rng, dtype)
        where = (0, 1)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure_ref:
            for i in range(n_qubits):
                pure_ref.apply_tensor_operator((i,), op_pre, unitary=True)
            pure_ref.apply_tensor_operator((0, 1), op_2q, unitary=True, immutable=True)
            rdm_ref = pure_ref.compute_reduced_density_matrix(where)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op_pre, unitary=True)
            oper = NetworkOperator((2,) * n_qubits, dtype=dtype, options=mixed.options)
            oper.append_product(1.0 + 0.0j, ((0, 1),), [op_2q])
            mixed.apply_network_operator(oper, unitary=True, immutable=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_ref, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_mpo_network_operator_matches_controlled_gate_reference(self, n_qubits):
        dtype = 'complex128'
        rng = np.random.default_rng(2200 + n_qubits)
        op_pre = random_unitary(1, rng, dtype).reshape(2, 2)
        mpo_tensors = cnot_mpo_tensors(dtype)
        where = (0, 1)

        # Pure reference via explicit controlled gate:
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure_ref:
            for i in range(n_qubits):
                pure_ref.apply_tensor_operator((i,), op_pre, unitary=True)
            pure_ref.apply_tensor_operator(
                (1,), np.array([[0, 1], [1, 0]], dtype=np.complex128),
                control_modes=(0,), control_values=(1,),
                unitary=True, immutable=True
            )
            rdm_ref = pure_ref.compute_reduced_density_matrix(where)

        # Pure via MPO NetworkOperator:
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op_pre, unitary=True)
            oper = NetworkOperator((2,) * n_qubits, dtype=dtype, options=pure.options)
            oper.append_mpo(1.0 + 0.0j, (0, 1), mpo_tensors)
            pure.apply_network_operator(oper, unitary=True, immutable=True)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        # Mixed via MPO NetworkOperator:
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op_pre, unitary=True)
            oper = NetworkOperator((2,) * n_qubits, dtype=dtype, options=mixed.options)
            oper.append_mpo(1.0 + 0.0j, (0, 1), mpo_tensors)
            mixed.apply_network_operator(oper, unitary=True, immutable=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_pure, rdm_ref, **tol)
        np.testing.assert_allclose(rdm_mixed, rdm_ref, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_tensor_product_network_operator_adjoint_matches_pure(self, n_qubits):
        dtype = 'complex128'
        rng = np.random.default_rng(2300 + n_qubits)
        op_pre = random_unitary(1, rng, dtype).reshape(2, 2)
        op_a = random_unitary(1, rng, dtype).reshape(2, 2)
        op_b = random_unitary(1, rng, dtype).reshape(2, 2)
        modes = ((0,), (n_qubits - 1,))
        where = (0, 1)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure_ref:
            for i in range(n_qubits):
                pure_ref.apply_tensor_operator((i,), op_pre, unitary=True)
            pure_ref.apply_tensor_operator((0,), op_a, unitary=True, immutable=True, adjoint=True)
            pure_ref.apply_tensor_operator((n_qubits - 1,), op_b, unitary=True, immutable=True, adjoint=True)
            rdm_ref = pure_ref.compute_reduced_density_matrix(where)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op_pre, unitary=True)
            oper = NetworkOperator((2,) * n_qubits, dtype=dtype, options=pure.options)
            oper.append_product(1.0 + 0.0j, modes, [op_a, op_b])
            pure.apply_network_operator(oper, unitary=True, immutable=True, adjoint=True)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op_pre, unitary=True)
            oper = NetworkOperator((2,) * n_qubits, dtype=dtype, options=mixed.options)
            oper.append_product(1.0 + 0.0j, modes, [op_a, op_b])
            mixed.apply_network_operator(oper, unitary=True, immutable=True, adjoint=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_pure, rdm_ref, **tol)
        np.testing.assert_allclose(rdm_mixed, rdm_ref, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_mpo_network_operator_adjoint_matches_controlled_gate_reference(self, n_qubits):
        dtype = 'complex128'
        rng = np.random.default_rng(2400 + n_qubits)
        op_pre = random_unitary(1, rng, dtype).reshape(2, 2)
        phase_gate = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
        mpo_tensors = controlled_single_target_mpo_tensors(phase_gate, dtype)
        where = (0, 1)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure_ref:
            for i in range(n_qubits):
                pure_ref.apply_tensor_operator((i,), op_pre, unitary=True)
            pure_ref.apply_tensor_operator(
                (1,), phase_gate,
                control_modes=(0,), control_values=(1,),
                unitary=True, immutable=True, adjoint=True
            )
            rdm_ref = pure_ref.compute_reduced_density_matrix(where)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op_pre, unitary=True)
            oper = NetworkOperator((2,) * n_qubits, dtype=dtype, options=pure.options)
            oper.append_mpo(1.0 + 0.0j, (0, 1), mpo_tensors)
            pure.apply_network_operator(oper, unitary=True, immutable=True, adjoint=True)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op_pre, unitary=True)
            oper = NetworkOperator((2,) * n_qubits, dtype=dtype, options=mixed.options)
            oper.append_mpo(1.0 + 0.0j, (0, 1), mpo_tensors)
            mixed.apply_network_operator(oper, unitary=True, immutable=True, adjoint=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_pure, rdm_ref, **tol)
        np.testing.assert_allclose(rdm_mixed, rdm_ref, **tol)


class TestMixedStateGateUpdate(_BaseTester):
    """Mutable gate updates in mixed state must produce correct results."""

    def test_update_changes_result(self):
        """Updating a mutable gate should change the computed RDM."""
        n_qubits = 3
        dtype = 'complex128'
        rng = np.random.default_rng(1800)
        op1 = random_unitary(1, rng, dtype)
        op_a = random_unitary(1, rng, dtype)
        op_b = random_unitary(1, rng, dtype)

        where = (0, 1)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as state:
            for i in range(n_qubits):
                state.apply_tensor_operator((i,), op1, unitary=True, immutable=True)
            tid = state.apply_tensor_operator((1,), op_a, unitary=True, immutable=False)
            rdm_before = state.compute_reduced_density_matrix(where)

            state.update_tensor_operator(tid, op_b, unitary=True)
            rdm_after = state.compute_reduced_density_matrix(where)

        assert not np.allclose(rdm_before, rdm_after, atol=1e-6), \
            "RDM should change after gate update"

    def test_update_matches_fresh_construction(self):
        """After update, the result should match constructing the circuit with the updated gate from scratch."""
        n_qubits = 3
        dtype = 'complex128'
        rng = np.random.default_rng(1900)
        op1 = random_unitary(1, rng, dtype)
        op_initial = random_unitary(1, rng, dtype)
        op_updated = random_unitary(1, rng, dtype)

        where = (0, 1)

        # Build with update
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as state:
            for i in range(n_qubits):
                state.apply_tensor_operator((i,), op1, unitary=True, immutable=True)
            tid = state.apply_tensor_operator((1,), op_initial, unitary=True, immutable=False)
            _ = state.compute_reduced_density_matrix(where)
            state.update_tensor_operator(tid, op_updated, unitary=True)
            rdm_updated = state.compute_reduced_density_matrix(where)

        # Build from scratch with the updated gate
        rng2 = np.random.default_rng(1900)
        op1b = random_unitary(1, rng2, dtype)
        _ = random_unitary(1, rng2, dtype)  # skip op_initial
        op_updated_b = random_unitary(1, rng2, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as fresh:
            for i in range(n_qubits):
                fresh.apply_tensor_operator((i,), op1b, unitary=True, immutable=True)
            fresh.apply_tensor_operator((1,), op_updated_b, unitary=True, immutable=False)
            rdm_fresh = fresh.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_updated, rdm_fresh, **tol)

    def test_update_matches_pure(self):
        """After update, mixed-state RDM should match pure-state RDM with the same updated gate."""
        n_qubits = 4
        dtype = 'complex128'
        rng = np.random.default_rng(2000)
        op1 = random_unitary(1, rng, dtype)
        op_initial = random_unitary(2, rng, dtype)
        op_updated = random_unitary(2, rng, dtype)

        where = (0, 1)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op1, unitary=True, immutable=True)
            tid = pure.apply_tensor_operator((0, 1), op_initial, unitary=True, immutable=False)
            _ = pure.compute_reduced_density_matrix(where)
            pure.update_tensor_operator(tid, op_updated, unitary=True)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        rng2 = np.random.default_rng(2000)
        op1b = random_unitary(1, rng2, dtype)
        op_initial_b = random_unitary(2, rng2, dtype)
        op_updated_b = random_unitary(2, rng2, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op1b, unitary=True, immutable=True)
            tid = mixed.apply_tensor_operator((0, 1), op_initial_b, unitary=True, immutable=False)
            _ = mixed.compute_reduced_density_matrix(where)
            mixed.update_tensor_operator(tid, op_updated_b, unitary=True)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_pure, **tol)

    def test_diagonal_gate_update_matches_pure(self):
        """Updating a mutable diagonal gate should match pure-state result."""  
        n_qubits = 3
        dtype = 'complex128'
        rng = np.random.default_rng(2100)
        op1 = random_unitary(1, rng, dtype)
        diag_initial = (rng.standard_normal(2) + 1j * rng.standard_normal(2)).astype(np.complex128)
        diag_updated = (rng.standard_normal(2) + 1j * rng.standard_normal(2)).astype(np.complex128)

        where = (0, 1)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            for i in range(n_qubits):
                pure.apply_tensor_operator((i,), op1, unitary=True, immutable=True)
            tid = pure.apply_tensor_operator((1,), diag_initial, diagonal=True, immutable=False, unitary=False)
            _ = pure.compute_reduced_density_matrix(where)
            pure.update_tensor_operator(tid, diag_updated, unitary=False)
            rdm_pure = pure.compute_reduced_density_matrix(where)

        rng2 = np.random.default_rng(2100)
        op1b = random_unitary(1, rng2, dtype)
        diag_initial_b = (rng2.standard_normal(2) + 1j * rng2.standard_normal(2)).astype(np.complex128)
        diag_updated_b = (rng2.standard_normal(2) + 1j * rng2.standard_normal(2)).astype(np.complex128)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            for i in range(n_qubits):
                mixed.apply_tensor_operator((i,), op1b, unitary=True, immutable=True)
            tid = mixed.apply_tensor_operator((1,), diag_initial_b, diagonal=True, immutable=False, unitary=False)
            _ = mixed.compute_reduced_density_matrix(where)
            mixed.update_tensor_operator(tid, diag_updated_b, unitary=False)
            rdm_mixed = mixed.compute_reduced_density_matrix(where)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm_mixed, rdm_pure, **tol)


class TestMixedStateExpectationGradient(_BaseTester):
    """Mixed-state expectation gradients must match pure-state results for unitary circuits."""

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_pauli_gradient_matches_pure(self, n_qubits):
        """Expectation gradient w.r.t. a single gate, Pauli-string Hamiltonian."""
        dtype = 'complex128'

        hamiltonian = {"Z" + "I" * (n_qubits - 1): 1.0}

        def build_and_compute(purity):
            rng = np.random.default_rng(5000 + n_qubits)
            ops = [random_unitary(1, rng, dtype) for _ in range(n_qubits)]
            op2 = random_unitary(2, rng, dtype)
            kwargs = dict(dtype=dtype, config=TNConfig())
            if purity == 'mixed':
                kwargs['pure_state'] = False
            with NetworkState((2,) * n_qubits, **kwargs) as state:
                for i, op in enumerate(ops):
                    state.apply_tensor_operator((i,), op, unitary=True)
                state.apply_tensor_operator(
                    (0, 1), op2, unitary=True, gradient=True)
                exp_val, grads = state.compute_expectation_with_gradients(
                    hamiltonian, 1.0)
            return np.asarray(exp_val), {k: np.asarray(v) for k, v in grads.items()}

        exp_pure, grads_pure = build_and_compute('pure')
        exp_mixed, grads_mixed = build_and_compute('mixed')

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(exp_mixed.real, exp_pure.real, **tol)
        assert len(grads_mixed) == len(grads_pure)
        for g_pure, g_mixed in zip(
                (grads_pure[k] for k in sorted(grads_pure)),
                (grads_mixed[k] for k in sorted(grads_mixed))):
            np.testing.assert_allclose(g_mixed, g_pure, **tol)

    @pytest.mark.parametrize("n_qubits", [4])
    def test_multi_gate_gradient_matches_pure(self, n_qubits):
        """Gradient w.r.t. multiple gates with a multi-term Hamiltonian."""
        dtype = 'complex128'

        pauli_strings = {
            "Z" + "I" * (n_qubits - 1): 2.0,
            "I" * (n_qubits - 1) + "Z": -1.0,
            "X" * n_qubits: 0.5,
        }

        def build_and_compute(purity):
            rng = np.random.default_rng(6000)
            ops = [random_unitary(1, rng, dtype) for _ in range(n_qubits)]
            op2 = random_unitary(2, rng, dtype)
            kwargs = dict(dtype=dtype, config=TNConfig())
            if purity == 'mixed':
                kwargs['pure_state'] = False
            with NetworkState((2,) * n_qubits, **kwargs) as state:
                for i, op in enumerate(ops):
                    state.apply_tensor_operator((i,), op, unitary=True, gradient=True)
                state.apply_tensor_operator((0, 1), op2, unitary=True, gradient=True)
                exp_val, grads = state.compute_expectation_with_gradients(
                    pauli_strings, 1.0)
            return np.asarray(exp_val), {k: np.asarray(v) for k, v in grads.items()}

        exp_pure, grads_pure = build_and_compute('pure')
        exp_mixed, grads_mixed = build_and_compute('mixed')

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(exp_mixed.real, exp_pure.real, **tol)
        assert len(grads_mixed) == len(grads_pure)
        for g_pure, g_mixed in zip(
                (grads_pure[k] for k in sorted(grads_pure)),
                (grads_mixed[k] for k in sorted(grads_mixed))):
            np.testing.assert_allclose(g_mixed, g_pure, **tol)

    def test_network_operator_gradient_matches_pure(self):
        """Gradient with a NetworkOperator Hamiltonian (tensor product terms)."""
        n_qubits = 3
        dtype = 'complex128'

        def build_and_compute(purity):
            rng = np.random.default_rng(7000)
            ops = [random_unitary(1, rng, dtype) for _ in range(n_qubits)]
            kwargs = dict(dtype=dtype, config=TNConfig())
            if purity == 'mixed':
                kwargs['pure_state'] = False
            hamiltonian = NetworkOperator((2,) * n_qubits, dtype=dtype)
            hamiltonian.append_product(1.0, [(0,), (1,)], [PAULI_Z, PAULI_Z])
            hamiltonian.append_product(0.5, [(2,)], [PAULI_X])
            with NetworkState((2,) * n_qubits, **kwargs) as state:
                for i, op in enumerate(ops):
                    state.apply_tensor_operator((i,), op, unitary=True, gradient=True)
                exp_val, grads = state.compute_expectation_with_gradients(
                    hamiltonian, 1.0)
            return np.asarray(exp_val), {k: np.asarray(v) for k, v in grads.items()}

        exp_pure, grads_pure = build_and_compute('pure')
        exp_mixed, grads_mixed = build_and_compute('mixed')

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(exp_mixed.real, exp_pure.real, **tol)
        assert len(grads_mixed) == len(grads_pure)
        for g_pure, g_mixed in zip(
                (grads_pure[k] for k in sorted(grads_pure)),
                (grads_mixed[k] for k in sorted(grads_mixed))):
            np.testing.assert_allclose(g_mixed, g_pure, **tol)


def _numpy_depol_rdm(n_qubits, channel_targets, rate, where):
    """Numpy reference: depolarizing channel(s) applied to |0...0>, then partial trace.

    The result is a C-contiguous (2**n_keep, 2**n_keep) matrix whose first axis is
    mode ``where[0]``. That matches ``NetworkState.compute_reduced_density_matrix``,
    which allocates C-order storage and passes those strides through.
    """
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    rho = np.outer(psi, psi.conj())
    ops = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
    weights = [1.0 - rate, rate / 3.0, rate / 3.0, rate / 3.0]
    for (q,) in channel_targets:
        rho_new = np.zeros_like(rho)
        for U, w in zip(ops, weights):
            full_op = np.eye(1, dtype=np.complex128)
            for q2 in range(n_qubits):
                full_op = np.kron(full_op, U if q2 == q else PAULI_I)
            rho_new += w * (full_op @ rho @ full_op.conj().T)
        rho = rho_new
    traced = sorted(set(range(n_qubits)) - set(where))
    rho_t = rho.reshape([2] * n_qubits + [2] * n_qubits)
    n_cur = n_qubits
    for q in sorted(traced, reverse=True):
        rho_t = np.trace(rho_t, axis1=q, axis2=q + n_cur)
        n_cur -= 1
    dim_where = 2 ** len(where)
    return rho_t.reshape(dim_where, dim_where)


class TestMixedStateUnitaryChannel(_BaseTester):
    """Circuits consisting solely of unitary channels (multiple Kraus operators)."""

    RATE = 0.3

    def test_channel_only_full_rdm(self):
        """Full RDM for a circuit with a single depolarizing channel."""
        n_qubits = 3
        dtype = 'complex128'
        ops = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        weights = [1.0 - self.RATE, self.RATE / 3, self.RATE / 3, self.RATE / 3]

        all_modes = tuple(range(n_qubits))
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            state.apply_unitary_tensor_channel((0,), ops, weights)
            rdm = state.compute_reduced_density_matrix(all_modes)

        rho_ref = _numpy_depol_rdm(n_qubits, [(0,)], self.RATE, all_modes)
        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm.reshape(rho_ref.shape), rho_ref, **tol)

    def test_channel_only_partial_trace(self):
        """Partial-trace RDM with a channel on one qubit, tracing out others."""
        n_qubits = 4
        dtype = 'complex128'
        ops = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        weights = [1.0 - self.RATE, self.RATE / 3, self.RATE / 3, self.RATE / 3]

        where = (0, 1)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            state.apply_unitary_tensor_channel((0,), ops, weights)
            rdm = state.compute_reduced_density_matrix(where)

        rho_ref = _numpy_depol_rdm(n_qubits, [(0,)], self.RATE, where)
        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm.reshape(rho_ref.shape), rho_ref, **tol)

    def test_channel_only_expectation(self):
        """Expectation value <Z_0> with only a depolarizing channel."""
        n_qubits = 3
        dtype = 'complex128'
        ops = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        weights = [1.0 - self.RATE, self.RATE / 3, self.RATE / 3, self.RATE / 3]

        all_modes = tuple(range(n_qubits))
        rho_ref = _numpy_depol_rdm(n_qubits, [(0,)], self.RATE, all_modes)
        dim = 2 ** n_qubits
        # rho_ref is C-order: qubit 0 is the most significant bit of the matrix index.
        msb = 1 << (n_qubits - 1)
        Z_full = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            Z_full[i, i] = 1.0 if (i & msb) == 0 else -1.0
        ref_val = np.trace(Z_full @ rho_ref).real

        hamiltonian = {"Z" + "I" * (n_qubits - 1): 1.0}
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            state.apply_unitary_tensor_channel((0,), ops, weights)
            result = state.compute_expectation(hamiltonian)

        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(np.asarray(result).real, ref_val, **tol)

    def test_channel_only_amplitude(self):
        """Diagonal density-matrix elements (amplitudes) with only a channel."""
        n_qubits = 2
        dtype = 'complex128'
        rate = 0.25
        ops = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        weights = [1.0 - rate, rate / 3, rate / 3, rate / 3]

        all_modes = tuple(range(n_qubits))
        rho_ref = _numpy_depol_rdm(n_qubits, [(0,)], rate, all_modes)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            state.apply_unitary_tensor_channel((0,), ops, weights)
            for idx in range(2 ** n_qubits):
                # compute_amplitude takes a per-qubit bitstring; rho_ref is C-order
                # (qubit 0 = most significant bit of the matrix index).
                bs = tuple((idx >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits))
                amp = state.compute_amplitude((bs, bs))
                np.testing.assert_allclose(amp.real, rho_ref[idx, idx].real, atol=1e-10,
                    err_msg=f"Amplitude mismatch for bitstring {bs}")

    def test_channel_only_trace_preserved(self):
        """Tr(rho) == 1 after a single depolarizing channel."""
        n_qubits = 3
        dtype = 'complex128'
        ops = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        weights = [0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3]

        all_modes = tuple(range(n_qubits))
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            state.apply_unitary_tensor_channel((1,), ops, weights)
            rdm = state.compute_reduced_density_matrix(all_modes)

        dim = 2 ** n_qubits
        tr = np.trace(rdm.reshape(dim, dim)).real
        np.testing.assert_allclose(tr, 1.0, atol=1e-10)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_multi_channel_rdm(self, n_qubits):
        """Depolarizing channel on every qubit sequentially."""
        dtype = 'complex128'
        rate = 0.15
        ops = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        weights = [1.0 - rate, rate / 3, rate / 3, rate / 3]

        all_modes = tuple(range(n_qubits))
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            for q in range(n_qubits):
                state.apply_unitary_tensor_channel((q,), ops, weights)
            rdm = state.compute_reduced_density_matrix(all_modes)

        targets = [(q,) for q in range(n_qubits)]
        rho_ref = _numpy_depol_rdm(n_qubits, targets, rate, all_modes)
        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rdm.reshape(rho_ref.shape), rho_ref, **tol)


class TestMixedStateGeneralChannelQudits(_BaseTester):
    """General (Kraus) channels on a (2, 3) qudit density matrix against NumPy.

    rho' = sum_i K_i rho K_i^dagger with non-separable Kraus operators |out><in|. Qudits of
    different dimension make the order in which the operator modes are passed observable.
    """

    DIMS = (2, 3)

    @staticmethod
    def _shift_matrix(dim):
        """Cyclic shift |j> -> |j+1 mod dim>, used to prepare basis states."""
        return np.roll(np.eye(dim, dtype=np.complex128), 1, axis=0)

    @staticmethod
    def _two_qudit_jump(dims, out_index, in_index):
        """|out><in| on the (modes[0], modes[1]) product basis with its trace-preserving complement."""
        volume = dims[0] * dims[1]
        jump = np.zeros((volume, volume), dtype=np.complex128)
        jump[out_index, in_index] = 1.0
        stay = np.eye(volume, dtype=np.complex128)
        stay[in_index, in_index] = 0.0
        return jump, stay

    @classmethod
    def _reference(cls, ops, rho, modes):
        """rho' = sum_i K_i rho K_i^dagger with K_i given on modes (a, b) of the two-qudit state."""
        d0, d1 = cls.DIMS
        rho_t = rho.reshape(d0, d1, d0, d1)  # (ket0, ket1, bra0, bra1)
        out = np.zeros_like(rho_t)
        da, db = (cls.DIMS[m] for m in modes)
        for k in ops:
            k4 = k.reshape(da, db, da, db)  # (out_a, out_b, in_a, in_b)
            if modes == (0, 1):
                k_rho = np.einsum("ABab,abcd->ABcd", k4, rho_t)
                out += np.einsum("ABcd,CDcd->ABCD", k_rho, k4.conj())
            else:
                k_rho = np.einsum("BAba,abcd->ABcd", k4, rho_t)
                out += np.einsum("ABcd,DCdc->ABCD", k_rho, k4.conj())
        return out.reshape(d0 * d1, d0 * d1)

    @pytest.mark.parametrize("dtype", ["complex64", "complex128"])
    @pytest.mark.parametrize("modes", [(0, 1), (1, 0)])
    @pytest.mark.parametrize("basis_state", [(0, 0), (1, 2), (0, 1), (1, 0)])
    def test_nonseparable_kraus_channel(self, dtype, modes, basis_state):
        dims_on_modes = tuple(self.DIMS[m] for m in modes)
        # |out><in| between two product-basis states of the acted-on qudits (modes order).
        jump, stay = self._two_qudit_jump(dims_on_modes, out_index=1,
                                          in_index=dims_on_modes[0] * dims_on_modes[1] - 1)
        ops = [jump.reshape(dims_on_modes + dims_on_modes), stay.reshape(dims_on_modes + dims_on_modes)]

        psi = np.zeros(self.DIMS, dtype=np.complex128)
        psi[basis_state] = 1.0
        rho = np.outer(psi.reshape(-1), psi.reshape(-1).conj())
        expected = self._reference([jump, stay], rho, modes)
        np.testing.assert_allclose(np.trace(expected), 1.0, atol=1e-12)  # trace preserving by construction

        with NetworkState(self.DIMS, dtype=dtype, config=TNConfig(), pure_state=False) as state:
            for q, level in enumerate(basis_state):
                for _ in range(level):
                    state.apply_tensor_operator((q,), self._shift_matrix(self.DIMS[q]).astype(dtype), unitary=True)
            state.apply_general_tensor_channel(modes, [np.ascontiguousarray(op.astype(dtype)) for op in ops])
            rho_out = np.asarray(state.compute_density_matrix()).reshape(np.prod(self.DIMS), -1)
        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(rho_out, expected, **tol)


class TestStateAccessorAPI(_BaseTester):
    """Tests for compute_state_vector (pure-only) and compute_density_matrix (pure and mixed)."""

    def test_compute_state_vector_not_deprecated_pure(self):
        """``compute_state_vector`` must (a) not emit a ``DeprecationWarning`` and (b)
        return the correct rank-N state vector for a pure state, consistent with
        ``compute_density_matrix`` (whose result is the outer product).
        """
        n_qubits = 3
        dtype = 'complex128'
        rng = np.random.default_rng(9000 + n_qubits)
        op1 = random_unitary(1, rng, dtype)
        op2 = random_unitary(2, rng, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as state:
            apply_brick_layer(state, n_qubits, op1, op2)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sv = state.compute_state_vector()
            dm = state.compute_density_matrix()

        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert not deprecation_warnings, \
            "compute_state_vector() should no longer emit a DeprecationWarning"

        sv_flat = np.asarray(sv).flatten()
        rho_ref = np.outer(sv_flat, sv_flat.conj()).reshape((2,) * (2 * n_qubits))
        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(dm, rho_ref, **tol)

    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_density_matrix_mixed(self, n_qubits):
        """compute_density_matrix for mixed state should return full density matrix."""
        dtype = 'complex128'
        rng = np.random.default_rng(9100 + n_qubits)
        op1 = random_unitary(1, rng, dtype)
        op2 = random_unitary(2, rng, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=True) as pure:
            apply_brick_layer(pure, n_qubits, op1, op2)
            sv = pure.compute_state_vector()

        rng2 = np.random.default_rng(9100 + n_qubits)
        op1b = random_unitary(1, rng2, dtype)
        op2b = random_unitary(2, rng2, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as mixed:
            apply_brick_layer(mixed, n_qubits, op1b, op2b)
            dm = mixed.compute_density_matrix()

        dim = 2 ** n_qubits
        sv_flat = sv.flatten()
        rho_ref = np.outer(sv_flat, sv_flat.conj()).reshape((2,) * (2 * n_qubits))
        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(dm, rho_ref, **tol)

    def test_density_matrix_mixed_qudits(self):
        """compute_density_matrix with non-uniform qudit dims and pure_state=False."""
        extents = (2, 3)
        dtype = 'complex128'
        rng = np.random.default_rng(9200)
        dim = 2 * 3
        mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        q, _ = np.linalg.qr(mat)
        op = q.reshape(2, 3, 2, 3).astype(np.complex128)

        with NetworkState(extents, dtype=dtype, pure_state=False) as state:
            state.apply_tensor_operator((0, 1), op, unitary=True)
            dm = state.compute_density_matrix()

        assert dm.shape == (2, 3, 2, 3), f"Expected shape (2,3,2,3), got {dm.shape}"

    def test_mixed_amplitude_consistency(self):
        """For mixed state: full density matrix and unprojected batched_amplitudes must match,
        marginal_probability must equal the diagonal, and per-element amplitudes <b|rho|b>
        must agree with that diagonal."""
        n_qubits = 3
        dtype = 'complex128'
        rng = np.random.default_rng(9300)
        op1 = random_unitary(1, rng, dtype)
        op2 = random_unitary(2, rng, dtype)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as state:
            apply_brick_layer(state, n_qubits, op1, op2)
            dm = state.compute_density_matrix()
            batched_full = state.compute_batched_amplitudes({})
            probs = state.compute_reduced_density_matrix(tuple(range(n_qubits)), diagonal=True)

            dim = 2 ** n_qubits
            dm_np = np.asarray(dm).reshape(dim, dim)
            diag_ref = np.diag(dm_np).real

            tol = get_contraction_tolerance(dtype)
            np.testing.assert_allclose(np.asarray(batched_full), np.asarray(dm), **tol)
            np.testing.assert_allclose(np.asarray(probs).flatten().real, diag_ref, **tol)

            for idx in range(dim):
                bs = tuple(int(b) for b in format(idx, f'0{n_qubits}b'))
                amp = state.compute_amplitude((bs, bs))
                np.testing.assert_allclose(amp.real, diag_ref[idx], atol=1e-10)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_mixed_real_dtype(self, dtype):
        """pure_state=False should work with real dtypes."""
        n_qubits = 3
        rng = np.random.default_rng(9400)
        dim = 2
        mat = rng.standard_normal((dim, dim))
        q, _ = np.linalg.qr(mat)
        op = q.reshape(2, 2).astype(getattr(np, dtype))

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as state:
            for i in range(n_qubits):
                state.apply_tensor_operator((i,), op, unitary=True)
            dm = state.compute_density_matrix()

        assert dm.shape == (2,) * (2 * n_qubits)

    def test_mixed_update_reuse(self):
        """Update a gate in mixed state, recompute, swap back, verify restoration."""
        n_qubits = 4
        dtype = 'complex128'
        rng = np.random.default_rng(9500)
        op1 = random_unitary(1, rng, dtype)
        op_a = random_unitary(2, rng, dtype)
        op_b = random_unitary(2, rng, dtype)
        pauli_dict = _make_pauli_dict(n_qubits)

        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            for i in range(n_qubits):
                state.apply_tensor_operator((i,), op1, unitary=True, immutable=True)
            tid = state.apply_tensor_operator((0, 1), op_a, unitary=True, immutable=False)
            exp_a = state.compute_expectation(pauli_dict)

            state.update_tensor_operator(tid, op_b, unitary=True)
            exp_b = state.compute_expectation(pauli_dict)

            state.update_tensor_operator(tid, op_a, unitary=True)
            exp_a2 = state.compute_expectation(pauli_dict)

        assert not np.allclose(exp_a, exp_b, atol=1e-6), "Expectation should change after update"
        tol = get_contraction_tolerance(dtype)
        np.testing.assert_allclose(np.asarray(exp_a2).real, np.asarray(exp_a).real, **tol)

    def test_mixed_return_norm(self):
        """compute_expectation with return_norm should return Tr(rho) for mixed state."""
        n_qubits = 3
        dtype = 'complex128'
        rng = np.random.default_rng(9600)
        op1 = random_unitary(1, rng, dtype)

        pauli_dict = {"I" * n_qubits: 1.0}
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False, config=TNConfig()) as state:
            for i in range(n_qubits):
                state.apply_tensor_operator((i,), op1, unitary=True)
            exp_val, norm = state.compute_expectation(pauli_dict, return_norm=True)

        np.testing.assert_allclose(np.asarray(norm).real, 1.0, atol=1e-10)

    def test_mixed_batched_amplitudes_single_dict(self):
        """Single-dict mixed compute_batched_amplitudes: scalar (symmetric), (ket, bra)
        off-diagonal, and (val, None) one-sided projections must match slices of the full DM.

        Output modes are ordered (open_ket_asc, open_bra_asc); with 2 qubits and qubit 0 fixed the
        references are direct slices of dm with axes (k0, k1, b0, b1).
        """
        n_qubits = 2
        dtype = 'complex128'
        rng = np.random.default_rng(9700)
        op2 = random_unitary(2, rng, dtype)
        tol = get_contraction_tolerance(dtype)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as state:
            state.apply_tensor_operator((0, 1), op2, unitary=True)
            dm = np.asarray(state.compute_density_matrix())  # (k0, k1, b0, b1)

            # scalar -> symmetric: fix ket0 == bra0 == 0
            sym = np.asarray(state.compute_batched_amplitudes({0: 0}))
            np.testing.assert_allclose(sym, dm[0, :, 0, :], **tol)

            # (ket, bra) -> independent: ket0 = 0, bra0 = 1
            off = np.asarray(state.compute_batched_amplitudes({0: (0, 1)}))
            np.testing.assert_allclose(off, dm[0, :, 1, :], **tol)

            # (val, None) -> one-sided: fix only ket0 = 0, bra fully open
            one = np.asarray(state.compute_batched_amplitudes({0: (0, None)}))
            np.testing.assert_allclose(one, dm[0, :, :, :], **tol)

    def test_mixed_amplitude_single_bitstring(self):
        """Mixed compute_amplitude accepts a single bitstring (symmetric, diagonal element),
        equivalent to passing the (bs, bs) 2-tuple."""
        n_qubits = 3
        dtype = 'complex128'
        rng = np.random.default_rng(9800)
        op1 = random_unitary(1, rng, dtype)
        op2 = random_unitary(2, rng, dtype)
        with NetworkState((2,) * n_qubits, dtype=dtype, pure_state=False) as state:
            apply_brick_layer(state, n_qubits, op1, op2)
            for idx in range(2 ** n_qubits):
                bs = tuple(int(b) for b in format(idx, f'0{n_qubits}b'))
                single = np.asarray(state.compute_amplitude(bs))
                pair = np.asarray(state.compute_amplitude((bs, bs)))
                np.testing.assert_allclose(single, pair, atol=1e-10)

    def test_invalid_pure_state_raises(self):
        """A non-bool pure_state must raise TypeError."""
        with pytest.raises(TypeError, match="pure_state"):
            NetworkState((2, 2), pure_state="mixed")

    def test_mixed_fixed_malformed_tuple_raises(self):
        """A per-key (ket, bra) value of the wrong length must raise ValueError."""
        rng = np.random.default_rng(9900)
        op2 = random_unitary(2, rng, 'complex128')
        with NetworkState((2, 2), dtype='complex128', pure_state=False) as state:
            state.apply_tensor_operator((0, 1), op2, unitary=True)
            with pytest.raises(ValueError, match="length 2"):
                state.compute_batched_amplitudes({0: (0, 1, 0)})

    def test_pure_rejects_tuple_fixed(self):
        """The (fixed_ket, fixed_bra) tuple form is rejected for pure states."""
        rng = np.random.default_rng(9901)
        op2 = random_unitary(2, rng, 'complex128')
        with NetworkState((2, 2), dtype='complex128', pure_state=True) as state:
            state.apply_tensor_operator((0, 1), op2, unitary=True)
            with pytest.raises(TypeError, match="tuple form is no longer used"):
                state.compute_batched_amplitudes(({0: 0}, {0: 0}))
