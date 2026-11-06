# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Trajectories based simulation of noisy quantum channels.

This test uses TrajectorySim API to simulate one-qubit noise channels.
Mixed-state tests use pure_state=False for exact (deterministic) verification
against the same analytical references. The last section checks general channels
with non-symmetric Kraus operators on an MPS against NumPy references.
"""

import numpy as np
import pytest
from cuquantum.tensornet.experimental import NetworkState, MPSConfig
from .quantum_channels import (
    apply_channel_to_mixed_state,
    bitflip_channel,
    depolarizing_channel,
    damping_channel,
    QuantumGates,
)
from .network_state_wrap import network_state_config

# in pytest cases, python objects are not displayed nicely, so let's use string tags
np.random.seed(10)

# -- Tests

# This test file uses `trajectory_sim` fixture defined in conftest.py

# Note: The bitflip_p values are selected to catch errors in conversion of
# unitary to general channel (sqrt(p) vs p).
@pytest.mark.parametrize("bitflip_p", [0.1, 0.75])
@pytest.mark.parametrize("n_qubits", [1])
def test_bitflip_channel(trajectory_sim, bitflip_p, channel_method):
    n_trajectories = 300
    channel = bitflip_channel(bitflip_p)
    if channel_method == "general":
        channel.set_general()

    ensemble_probs = []
    for sim in trajectory_sim.iterate_trajectories(n_trajectories):
        sim.apply_channel((0,), channel)
        prob = sim.probs((0,))
        ensemble_probs.append(prob)

    ensemble_probs = np.stack(ensemble_probs).mean(axis=0)
    print("Bitflip ensemble Probs ", ensemble_probs)
    true_probs = np.array([1 - bitflip_p, bitflip_p])
    print("Bitflip ensemble Reference probs ", true_probs)
    # Use 2.5 sigma, which is about 99% CI
    sigma_ = 1 / np.sqrt(n_trajectories)
    assert (np.abs(ensemble_probs - true_probs) < 2.5 * sigma_).all()


@pytest.mark.parametrize("n_qubits", [1, 2])
def test_depolarizing_channel(trajectory_sim, n_qubits, channel_method):
    """
    For n_qubits>1, pads the channel with identity on the left.
    Checks that we get a I/2 state on the corresponding qubit
    """
    n_trajectories = 300
    error = 1
    channel = depolarizing_channel(error)
    if channel_method == "general":
        channel.set_general()
    if n_qubits > 1:
        for _ in range(n_qubits - 1):
            channel.mul_left(np.eye(2))

    ensemble_probs = []
    qubit_id = n_qubits - 1
    for sim in trajectory_sim.iterate_trajectories(n_trajectories):
        sim.apply_channel(tuple(range(n_qubits)), channel)
        probs = sim.probs((qubit_id,))
        ensemble_probs.append(probs)

    # TODO: check that we don't have a I/2 DM on qubit 0
    ensemble_probs = np.stack(ensemble_probs).mean(axis=0)
    print("Depolarizing ensemble probs ", ensemble_probs)
    true_dm = np.ones(2) / 2
    # Use 2.5 sigma, which is about 99% CI
    sigma_ = 1 / np.sqrt(n_trajectories)
    assert (np.abs(ensemble_probs - true_dm) < 2.5 * sigma_).all()


@pytest.mark.parametrize("damping", [0.08])
@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_damping_channel(trajectory_sim, damping):
    n_trajectories = 50
    channel = damping_channel(damping)
    n_damping_rounds = 5
    ensemble_probs = []
    for sim in trajectory_sim.iterate_trajectories(n_trajectories):
        sim.apply_gate((0,), QuantumGates.X)
        for _ in range(n_damping_rounds):
            sim.apply_channel((0,), channel)
        prob = sim.probs((0,))
        # print("Damping prob ", prob)
        # probabilities are not normalized for non-unitary channels
        ensemble_probs.append(prob/sum(prob))

    ensemble_probs_raw = np.stack(ensemble_probs)
    print("Bitflip ensemble Probs ", ensemble_probs_raw)
    ensemble_probs = ensemble_probs_raw.mean(axis=0)
    print("Bitflip ensemble Probs mean unnormalized", ensemble_probs)
    damped_prob = (1 - damping)**n_damping_rounds
    true_probs = np.array([1-damped_prob, damped_prob])
    print("Bitflip ensemble Reference probs", true_probs)
    # Use 2.5 sigma, which is about 99% CI
    sigma_ = 1 / np.sqrt(n_trajectories)
    assert (np.abs(ensemble_probs - true_probs) < 2.5 * sigma_).all()


# -- Mixed-state exact tests --
# Same analytical references as above, verified deterministically via pure_state=False.

def _mixed_probs(state, qubits):
    """Extract measurement probabilities from a mixed-state RDM."""
    rdm = state.compute_reduced_density_matrix(qubits)
    dim = 2 ** len(qubits)
    return np.diag(np.asarray(rdm).reshape(dim, dim)).real


@pytest.mark.parametrize("bitflip_p", [0.1, 0.75])
@pytest.mark.parametrize("as_general", [False, True], ids=["unitary", "general"])
def test_bitflip_channel_mixed(bitflip_p, as_general):
    """Exact mixed-state bitflip must give probs [1-p, p]."""
    channel = bitflip_channel(bitflip_p)
    with network_state_config(1, 'tn', pure_state=False) as state:
        apply_channel_to_mixed_state(state, (0,), channel, as_general)
        probs = _mixed_probs(state, (0,))
    true_probs = np.array([1 - bitflip_p, bitflip_p])
    np.testing.assert_allclose(probs, true_probs, atol=1e-12)


@pytest.mark.parametrize("n_qubits", [1, 2])
@pytest.mark.parametrize("as_general", [False, True], ids=["unitary", "general"])
def test_depolarizing_channel_mixed(n_qubits, as_general):
    """Full depolarization on the last qubit must give I/2."""
    channel = depolarizing_channel(1)
    if n_qubits > 1:
        for _ in range(n_qubits - 1):
            channel.mul_left(np.eye(2))

    qubit_id = n_qubits - 1
    with network_state_config(n_qubits, 'tn', pure_state=False) as state:
        apply_channel_to_mixed_state(state, tuple(range(n_qubits)), channel, as_general)
        probs = _mixed_probs(state, (qubit_id,))
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-12)


@pytest.mark.parametrize("damping", [0.08, 0.5])
@pytest.mark.parametrize("n_damping_rounds", [1, 5])
def test_damping_channel_mixed(damping, n_damping_rounds):
    """Amplitude damping on |1> must give p(1)=(1-gamma)^n."""
    channel = damping_channel(damping)
    with network_state_config(1, 'tn', pure_state=False) as state:
        state.apply_tensor_operator((0,), QuantumGates.X, unitary=True)
        for _ in range(n_damping_rounds):
            apply_channel_to_mixed_state(state, (0,), channel)
        probs = _mixed_probs(state, (0,))
    damped_prob = (1 - damping) ** n_damping_rounds
    true_probs = np.array([1 - damped_prob, damped_prob])
    np.testing.assert_allclose(probs, true_probs, atol=1e-12)


# -- General channels with non-Hermitian, non-symmetric Kraus operators --
# A trajectory selects one Kraus operator K_i with probability p_i = <psi|K_i^dagger K_i|psi>
# and the state becomes K_i|psi> / sqrt(p_i). The tests below use operators such as |0><1|,
# whose action differs from that of their transpose and adjoint, and check the selected
# operator, the resulting state and its norm against NumPy references, with the operators
# in every position of the Kraus list and the modes passed in both orders. Assertions are
# made on the raw backend result; nothing is renormalized, so an incorrectly weighted
# trajectory is visible as a norm != 1.

# Kraus operators as textbook matrices K[output, input].
JUMP = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # |0><1|
STAY = np.array([[1, 0], [0, 0]], dtype=np.complex128)  # |0><0|
ZERO_OP = np.zeros((2, 2), dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)
ZERO_STATE_2Q = np.array([1, 0, 0, 0], dtype=np.complex128)


def _kraus_tol(dtype):
    return 1e-5 if dtype == "complex64" else 1e-12


# These tests select the trajectory branch from a reduced density matrix computed by SVD on
# the MPS; the result must not depend on the SVD algorithm.
@pytest.fixture(params=["gesvd", "gesvdj"])
def svd_algorithm(request):
    return request.param


def _kraus_mps_state(dtype, svd_algorithm):
    """Two-qubit MPS, free gauge, no truncation."""
    config = MPSConfig(gauge_option="free", algorithm=svd_algorithm, abs_cutoff=0.0, rel_cutoff=0.0)
    return NetworkState((2, 2), dtype=dtype, config=config)


def _channel(ops, dtype):
    return [np.ascontiguousarray(op.astype(dtype)) for op in ops]


def _two_qubit(op):
    """Kron(op, I) as a rank-4 operand with modes (out0, out1, in0, in1)."""
    return np.kron(op, I2).reshape(2, 2, 2, 2)


def _state_vector_and_norm(state):
    sv, norm = state.compute_state_vector(return_norm=True)
    return np.asarray(sv).reshape(-1), float(np.real(np.asarray(norm)))


def _assert_state(sv, norm, expected, dtype):
    tol = _kraus_tol(dtype)
    np.testing.assert_allclose(norm, 1.0, atol=tol)
    np.testing.assert_allclose(sv, expected, atol=tol)


def test_asymmetric_first_kraus_applied_twice(dtype, svd_algorithm):
    """|0> under [|0><1|, |0><0|] twice must stay |0> with unit norm.

    |0><1| annihilates |0> and must never be selected; |0><0| leaves the state unchanged.
    """
    with _kraus_mps_state(dtype, svd_algorithm) as state:
        channel = _channel([JUMP, STAY], dtype)
        state.apply_general_tensor_channel((0,), channel)
        state.apply_general_tensor_channel((0,), channel)
        sv, norm = _state_vector_and_norm(state)
    _assert_state(sv, norm, ZERO_STATE_2Q, dtype)


@pytest.mark.parametrize("order", [(JUMP, STAY), (STAY, JUMP)], ids=["jump_first", "stay_first"])
def test_kraus_order_invariance(dtype, order, svd_algorithm):
    """Reordering the Kraus operators does not change the channel."""
    with _kraus_mps_state(dtype, svd_algorithm) as state:
        channel = _channel(order, dtype)
        state.apply_general_tensor_channel((0,), channel)
        state.apply_general_tensor_channel((0,), channel)
        sv, norm = _state_vector_and_norm(state)
    _assert_state(sv, norm, ZERO_STATE_2Q, dtype)


def test_excited_state_selects_jump(dtype, svd_algorithm):
    """|1> under [|0><1|, |0><0|] must deterministically become |0>.

    |0><1| is the only operator with non-zero weight on |1>; it must be selected and
    applied as given.
    """
    with _kraus_mps_state(dtype, svd_algorithm) as state:
        state.apply_tensor_operator((0,), QuantumGates.X.astype(dtype), unitary=True)
        state.apply_general_tensor_channel((0,), _channel([JUMP, STAY], dtype))
        sv, norm = _state_vector_and_norm(state)
    _assert_state(sv, norm, ZERO_STATE_2Q, dtype)


def test_asymmetric_kraus_in_middle_slot(dtype, svd_algorithm):
    """[0, |0><1|, |0><0|] is a valid channel; |0> has probabilities [0, 0, 1].

    The zero operator moves the non-symmetric operator to the middle of the Kraus list.
    """
    with _kraus_mps_state(dtype, svd_algorithm) as state:
        channel = _channel([ZERO_OP, JUMP, STAY], dtype)
        state.apply_general_tensor_channel((0,), channel)
        state.apply_general_tensor_channel((0,), channel)
        sv, norm = _state_vector_and_norm(state)
    _assert_state(sv, norm, ZERO_STATE_2Q, dtype)


@pytest.mark.parametrize("modes", [(0, 1), (1, 0)])
@pytest.mark.parametrize("excite", [False, True], ids=["from_0", "from_1"])
def test_two_qubit_asymmetric_kraus(dtype, modes, excite, svd_algorithm):
    """[|0><1| x I, |0><0| x I] on two modes maps both |0> and |1> of modes[0] to |0>."""
    with _kraus_mps_state(dtype, svd_algorithm) as state:
        if excite:
            state.apply_tensor_operator((modes[0],), QuantumGates.X.astype(dtype), unitary=True)
        channel = _channel([_two_qubit(JUMP), _two_qubit(STAY)], dtype)
        state.apply_general_tensor_channel(modes, channel)
        state.apply_general_tensor_channel(modes, channel)
        sv, norm = _state_vector_and_norm(state)
    _assert_state(sv, norm, ZERO_STATE_2Q, dtype)


# Non-separable two-qubit Kraus operators |out><in| on the (modes[0], modes[1]) product
# basis with their trace-preserving complements sqrt(I - K^dagger K) = I - |in><in|.
def _two_qubit_jump(out_index, in_index):
    jump = np.zeros((4, 4), dtype=np.complex128)
    jump[out_index, in_index] = 1.0
    stay = np.eye(4, dtype=np.complex128)
    stay[in_index, in_index] = 0.0
    return jump, stay


NONSEPARABLE_CHANNELS = {"01<-10": _two_qubit_jump(1, 2), "00<-10": _two_qubit_jump(0, 2)}


def _apply_two_qubit(op, psi, modes):
    """Reference: apply a (out_a, out_b, in_a, in_b) operator on modes (a, b) to psi[i0, i1]."""
    k = op.reshape(2, 2, 2, 2)
    if modes == (0, 1):
        return np.einsum("ABab,ab->AB", k, psi)
    return np.einsum("ABab,ba->BA", k, psi)


@pytest.mark.parametrize("modes", [(0, 1), (1, 0)])
@pytest.mark.parametrize("excite", [(), (0,), (1,), (0, 1)], ids=["00", "10", "01", "11"])
@pytest.mark.parametrize("channel_name", list(NONSEPARABLE_CHANNELS))
def test_two_qubit_nonseparable_kraus(dtype, modes, excite, channel_name, svd_algorithm):
    """A non-separable |out><in| channel on a basis state is deterministic; check against NumPy."""
    jump, stay = NONSEPARABLE_CHANNELS[channel_name]
    psi = np.zeros((2, 2), dtype=np.complex128)
    psi[tuple(int(q in excite) for q in (0, 1))] = 1.0
    branches = [_apply_two_qubit(k, psi, modes) for k in (jump, stay)]
    probs = [np.linalg.norm(b) ** 2 for b in branches]
    assert sorted(np.round(probs, 12)) == [0.0, 1.0]  # deterministic by construction
    expected = branches[int(np.argmax(probs))].reshape(-1)

    with _kraus_mps_state(dtype, svd_algorithm) as state:
        for q in excite:
            state.apply_tensor_operator((q,), QuantumGates.X.astype(dtype), unitary=True)
        channel = _channel([jump.reshape(2, 2, 2, 2), stay.reshape(2, 2, 2, 2)], dtype)
        state.apply_general_tensor_channel(modes, channel)
        state.apply_general_tensor_channel(modes, channel)
        sv, norm = _state_vector_and_norm(state)
    # The channel is idempotent on its deterministic outputs, so applying it twice gives
    # the single-application result.
    _assert_state(sv, norm, expected, dtype)


# A complex, non-symmetric Kraus operator.
K_COMPLEX = np.array([[0.3, 0.8j], [0.0, 0.2]], dtype=np.complex128)
PSI_PREP = np.array([[1, 1], [1j, -1j]], dtype=np.complex128) / np.sqrt(2)  # |0> -> (|0> + i|1>)/sqrt(2)


def _complement(k):
    """Hermitian PSD square root of I - K^dagger K, completing [K, K1] to a channel."""
    w, v = np.linalg.eigh(I2 - k.conj().T @ k)
    assert (w > -1e-12).all()
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T


def test_complex_asymmetric_kraus_probability(dtype, svd_algorithm):
    """Selection frequency and output norm for a complex non-symmetric operator.

    Every trajectory must have unit norm, and the frequency with which K_0 is selected
    must match p_0 = <psi|K_0^dagger K_0|psi>.
    """
    k0, k1 = K_COMPLEX, _complement(K_COMPLEX)
    psi = PSI_PREP[:, 0]
    branches = [k @ psi for k in (k0, k1)]
    p_ref = np.linalg.norm(branches[0]) ** 2
    # The frequency check is only informative if K_0^T, conj(K_0) and K_0^dagger would
    # be selected with clearly different probabilities.
    for alt in (k0.T, k0.conj(), k0.conj().T):
        assert abs(np.linalg.norm(alt @ psi) ** 2 - p_ref) > 0.07
    channel = _channel([k0, k1], dtype)

    n_trajectories = 300
    tol = 1e-4 if dtype == "complex64" else 1e-10
    n_branch0 = 0
    for _ in range(n_trajectories):
        with _kraus_mps_state(dtype, svd_algorithm) as state:
            state.apply_tensor_operator((0,), PSI_PREP.astype(dtype), unitary=True)
            state.apply_general_tensor_channel((0,), channel)
            sv, norm = _state_vector_and_norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=tol)
        sv_q0 = sv.reshape(2, 2)[:, 0]  # qubit 1 is idle in |0>
        fidelities = [abs(np.vdot(b / np.linalg.norm(b), sv_q0)) ** 2 for b in branches]
        assert max(fidelities) > 1 - 1e-4
        n_branch0 += int(np.argmax(fidelities) == 0)

    freq = n_branch0 / n_trajectories
    sigma = np.sqrt(p_ref * (1 - p_ref) / n_trajectories)
    assert abs(freq - p_ref) < 3.5 * sigma, (freq, p_ref, sigma)
