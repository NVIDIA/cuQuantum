# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Trajectories based simulation of noisy quantum channels.

This test uses TrajectorySim API to simulate quantum volumue circuits
"""

from qiskit.circuit.library import QuantumVolume
from qiskit import transpile

import numpy as np
import pytest
from cuquantum.cutensornet.experimental._internal.network_state_utils import (
    STATE_DEFAULT_DTYPE,
)
from cuquantum.cutensornet._internal import utils
from cuquantum.cutensornet.experimental.configuration import MPSConfig
from cuquantum.cutensornet.configuration import NetworkOptions
from cuquantum.cutensornet.circuit_converter import CircuitToEinsum

from .network_state_wrap import TrajectorySim
from .quantum_channels import (
    depolarizing_channel,
    QuantumChannel,
)


SEED = 10
np.random.seed(SEED)
n_variations = 10
depth = 30

# This test file uses `trajectory_sim` fixture defined in conftest.py


def get_qvolume_circuit(n_qubits, depth, seed=SEED):
    circuit = QuantumVolume(n_qubits, depth, seed=seed)
    circuit.measure_all()
    circuit = transpile(circuit, basis_gates=["u3", "cx"], optimization_level=0)
    return circuit


def apply_circuit_with_noise(
    ns: TrajectorySim,
    circuit,
    channel: QuantumChannel,
    dtype=STATE_DEFAULT_DTYPE,
    backend="numpy",
    options=None,
):
    options = utils.check_or_create_options(NetworkOptions, options, "network options")
    with utils.device_ctx(options.device_id):
        converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend)

    # dtype = getattr(converter.dtype, '__name__', str(converter.dtype).split('.')[-1])
    for gate_operand, gate_qubits in converter.gates:
        # all gate operands are assumed to be unitary
        qubits_indices = [converter.qubits.index(q) for q in gate_qubits]
        ns.apply_gate(qubits_indices, gate_operand)
        for q in qubits_indices:
            ns.apply_channel((q,), channel)


@pytest.mark.parametrize("n_qubits", [5])
@pytest.mark.parametrize("bitflip_p", [0.01, 0.08])
def test_quantum_volume(trajectory_sim, bitflip_p, n_qubits):
    """
    Apply quantum volume circuit M rounds.

    Measure first two qubits at the end of each round.
    Apply projector gate to collapse the state after the measurement.

    Expect the mid-circuit measurements to be uniform

    """
    n_trajectories = 30
    circ_depth_per_round = 5
    channel = depolarizing_channel(bitflip_p)
    measurement_rounds = 2
    measured_qubits = (0, 1)

    ensemble_mid_probs = []
    for sim in trajectory_sim.iterate_trajectories(n_trajectories):
        for m in range(measurement_rounds):
            circuit = get_qvolume_circuit(n_qubits, circ_depth_per_round, seed=SEED)
            apply_circuit_with_noise(sim, circuit, channel)
            probs = sim.probs(measured_qubits)
            # -- Sample result
            if measurement_rounds > 1:
                # projector operators are non-unitary
                probs /= probs.sum()
            if isinstance(sim.ns.config, MPSConfig):
                # MPS is an approximate algorithm
                probs /= probs.sum()

            ensemble_mid_probs.append(probs)
            msmt = np.random.choice(np.arange(len(probs)), p=probs)
            # -- Apply projector based on the measurement
            for i, q in enumerate(measured_qubits):
                projector = np.zeros((2, 2), dtype="complex128")
                # bit = (msmt >> i) % 2
                bit = np.unravel_index(msmt, (2,) * n_qubits)[i]
                projector[bit, bit] = 1
                sim.apply_gate((q,), projector)
            # --

    mean_mid_probs = np.stack(ensemble_mid_probs).mean(axis=0)
    print(f"{mean_mid_probs=}")
    expect_mid_probs = np.ones_like(mean_mid_probs) / mean_mid_probs.size
    sigma_ = 1 / np.sqrt(n_trajectories)
    margin = 2.5 * sigma_
    assert np.allclose(expect_mid_probs, mean_mid_probs, atol=margin)
