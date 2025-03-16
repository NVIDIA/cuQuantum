# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Trajectories based simulation of noisy quantum channels.

This test uses TrajectorySim API to simulate one-qubit noise channels
"""

import numpy as np
import pytest
from .quantum_channels import (
    bitflip_channel,
    depolarizing_channel,
    damping_channel,
    QuantumGates,
)

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
