# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Trajectories based simulation of noisy quantum channels.

This test uses TrajectorySim API to simulate changes in expectation of maxcut cost observable under noise
"""

import numpy as np
import networkx as nx
import pytest
from .quantum_channels import (
    bitflip_channel,
    QuantumGates,
)


SEED = 10
np.random.seed(SEED)


@pytest.mark.parametrize("bitflip_p", [0.01, 0.08])
@pytest.mark.parametrize("n_qubits", [10])
def test_bitflip_maxcut_cost(trajectory_sim, bitflip_p, n_qubits):
    """
    Take a unitary operator U=exp(C) and its eigenstate.

    Apply the operator to the state with bitflip noise after each gate.

    Evaluate fidelity and expectation value
    """
    n_trajectories = 30
    channel = bitflip_channel(bitflip_p)
    G = nx.random_regular_graph(3, n_qubits)
    init_cut_value, (init_flips, _) = nx.approximation.one_exchange(G, seed=SEED)
    cost_dict = {}
    for u, v in G.edges:
        pstring = ["I"] * n_qubits
        pstring[u] = "Z"
        pstring[v] = "Z"
        cost_dict["".join(pstring)] = 0.5

    ensemble_dms = []
    ensemble_exps = []
    for sim in trajectory_sim.iterate_trajectories(n_trajectories):
        # -- Prepare init state
        for q in init_flips:
            sim.apply_gate((q,), QuantumGates.X)
        # -- Apply operator
        for u, v in G.edges:
            gate = QuantumGates.eZZ.reshape((2, 2, 2, 2))
            sim.apply_gate((u, v), gate)
            # -- Apply noise on the gate
            for q in (u, v):
                sim.apply_channel((q,), channel)
        # -- Calculate expectation
        exp = sim.expectation(cost_dict)
        ensemble_exps.append(exp)
        # -- Calculate DM
        dm = sim.rdm()
        ensemble_dms.append(dm)

    print(f"{ensemble_exps}")
    avg_cost = G.number_of_edges() / 2 - np.mean(ensemble_exps)
    print(f"{avg_cost=}")
    print(f"{init_cut_value=}")
    ensemble_dm = np.stack(ensemble_dms).mean(axis=0).reshape(2**n_qubits, 2**n_qubits)

    # -- Reference values
    rdm_true = np.ones(2)
    for sim in trajectory_sim.iterate_trajectories(1):
        for q in init_flips:
            sim.apply_gate((q,), QuantumGates.X)
        rdm_true = sim.rdm()
    # --
    # F = <\psi|\rho|\psi>
    fidelity = np.trace(rdm_true.dot(ensemble_dm))
    print(f"{fidelity=}")
    n_noise_gates = 2 * G.number_of_edges()
    expected_fidelity = (1 - bitflip_p) ** (n_noise_gates)
    print(f"{expected_fidelity=}")
    sigma_ = 1 / np.sqrt(n_trajectories)
    print(f"{sigma_=}")
    # - TODO: verify the scaling of fidelity under trajectories
    assert np.abs(fidelity - expected_fidelity) < 2.5 * sigma_
