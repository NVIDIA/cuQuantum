# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration presets for NetworkState

To add a new NetworkState config preset, edit STATE_CONFIG_MAP,
then use network_state_config to create the NetworkState object in tests.
"""

from abc import ABC
from typing import Iterator
import numpy as np
from cuquantum.tensornet.experimental import NetworkState, TNConfig, MPSConfig
from .quantum_channels import QuantumChannel

# in pytest cases, python objects are not displayed nicely, so let's use string tags
STATE_CONFIG_MAP = {"tn": TNConfig(),
                    "mps": MPSConfig(max_extent=4, rel_cutoff=1e-5, gauge_option='free'),
                    "mps_value": MPSConfig(rel_cutoff=1e-5, gauge_option='free'),
                    "mps_value_su": MPSConfig(rel_cutoff=1e-4, gauge_option='simple')}


def network_state_config(n_qubits, algo: str, dtype="complex128") -> NetworkState:
    """
    Helper function to configure NetworkState to use MPS or TN

    Args:
        - n_qubits: int
        - algo: str
            algorithm config id from network_state_config.STATE_CONFIG_MAP
    """
    # workaround for MPS, one qubit isn't working
    if n_qubits == 1:
        n_qubits = 2
    state_mode_extents = (2,) * n_qubits
    if algo not in STATE_CONFIG_MAP:
        raise ValueError(f"Unknown state config id: {algo}")
    config = STATE_CONFIG_MAP[algo]
    nstate = NetworkState(state_mode_extents, dtype=dtype, config=config)
    return nstate


class TrajectorySim(ABC):
    def apply_channel(self, qubits, channel: QuantumChannel): ...
    def apply_gate(self, qubits, gate): ...
    def rdm(self, qubits) -> np.ndarray: ...
    def probs(self, qubits) -> np.ndarray: ...
    def expectation(self, pauli_dict) -> float: ...
    def iterate_trajectories(self, n_trajectories) -> Iterator["TrajectorySim"]:
        """Use the object only within the iteration."""
        ...


class TrajectoryNaive(TrajectorySim):
    n_qubits: int
    algo: str
    dtype: str

    def __init__(self, n_qubits: int, algo: str, dtype="complex128"):
        self.ns = network_state_config(n_qubits, algo, dtype)
        self.n_qubits = n_qubits
        self.algo = algo
        self.dtype = dtype

    def apply_channel(self, qubits, channel: QuantumChannel):
        gate = channel.choose_op()
        self.apply_gate(qubits, gate)

    def apply_gate(self, qubits, gate, control_modes=None, control_values=None):
        self.ns.apply_tensor_operator(
            qubits,
            gate.astype(self.dtype),
            unitary=False,
            control_modes=control_modes,
            control_values=control_values,
            immutable=True,
        )

    def rdm(self, qubits=None):
        if qubits is None:
            qubits = list(range(self.n_qubits))
        nstates_ = 2 ** len(qubits)
        dm = self.ns.compute_reduced_density_matrix(qubits)
        return dm.reshape(nstates_, nstates_)

    def probs(self, qubits):
        dm = self.rdm(qubits)
        probs = abs(np.diagonal(dm)).astype(float)
        # probs normalization needed for general channel 
        return probs / probs.sum()

    def expectation(self, pauli_dict) -> float:
        return self.ns.compute_expectation(pauli_dict).real

    def iterate_trajectories(self, n_trajectories):
        for _ in range(n_trajectories):
            with network_state_config(self.n_qubits, algo=self.algo, dtype=self.dtype) as ns:
                self.ns = ns
                yield self


class TrajectoryApplyChannel(TrajectoryNaive):
    # Prevents applying a gate to the same object in the trajectory loop
    _constructed: bool
    # Prevents re-using measured MPS between trajectories
    _evolved: bool

    def apply_gate(self, qubits, gate, control_modes=None, control_values=None):
        if self._constructed:
            return
        self.ns.apply_tensor_operator(
            qubits,
            gate.astype(self.dtype),
            unitary=True,
            control_modes=control_modes,
            control_values=control_values,
            immutable=True,
        )

    def apply_channel(self, qubits, channel: QuantumChannel):
        if self._constructed:
            return
        channel.set_dtype(self.dtype)
        if channel.is_general():
            self.ns.apply_general_tensor_channel(qubits, channel.ops)
        else:

            self.ns.apply_unitary_tensor_channel(qubits, channel.ops, channel.probs)

    def evolve(self):
        if self.algo == "mps":
            self.ns.compute_output_state(release_operators=True)
            sv = self.ns.compute_state_vector()
        self._evolved = True

    def rdm(self, qubits=None):
        self.evolve()
        return super().rdm(qubits)

    def iterate_trajectories(self, n_trajectories):
        self._evolved = False
        self._constructed = False
        self.ns = network_state_config(self.n_qubits, algo=self.algo, dtype=self.dtype)
        for i in range(n_trajectories):
            if self._evolved:
                self.ns.free()
                self.ns = network_state_config(self.n_qubits, algo=self.algo, dtype=self.dtype)
                self._constructed = False
            yield self
            self._constructed = True
        self.ns.free()
