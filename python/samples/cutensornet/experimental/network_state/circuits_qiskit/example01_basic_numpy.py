# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quantum circuit simulation based on tensor network contraction. 

NumPy ndarrays are used as inputs and output.
"""
import qiskit

from cuquantum.cutensornet.experimental import NetworkState, TNConfig

# create a QFT circuit
n_qubits = 8
qubits = list(range(n_qubits))
circuit = qiskit.QuantumCircuit(n_qubits)
qft = qiskit.circuit.library.QFT(num_qubits=n_qubits)
circuit.append(qft, qubits)
print(circuit)

# select tensor network contraction as the simulation method
config = TNConfig(num_hyper_samples=4)

# create a NetworkState object
state = NetworkState.from_circuit(circuit, dtype='complex128', config=config, backend='numpy')

# compute the state vector
sv = state.compute_state_vector()
print(f"state vector type: {type(sv)}, shape: {sv.shape}")

# compute the bitstring amplitude
bitstring = '0' * n_qubits
amplitude = state.compute_amplitude(bitstring)
print(f"Bitstring amplitude for {bitstring}: {amplitude}")

# compute batched bitstring amplitude
fixed = {0: 0, 1: 1} # first qubit fixed at state 0 and second qubit at state 1
batched_amplitudes = state.compute_batched_amplitudes(fixed)
print(f"Batched amplitudes shape: {batched_amplitudes.shape}")

# draw samples from the state object
nshots = 100
samples = state.compute_sampling(nshots)
print("Sampling results:")

print(samples)

pauli_string = {'IXIXIXIX': 0.5, 'IYIYIYIY': 0.2, 'IZIZIZIZ': 0.3}
expec = state.compute_expectation(pauli_string)
print(f"{expec=}")
# release resources
state.free()
