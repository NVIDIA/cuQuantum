# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Specify the logging options for quantum circuit simulation. 

PyTorch tensors are used as inputs and output.
"""
import logging

import qiskit

from cuquantum.cutensornet.experimental import NetworkState

# set logging level to INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

# create a QFT circuit
n_qubits = 8
qubits = list(range(n_qubits))
circuit = qiskit.QuantumCircuit(n_qubits)
qft = qiskit.circuit.library.QFT(num_qubits=n_qubits)
circuit.append(qft, qubits)
print(circuit)

# create a NetworkState object with tensor network contraction as simulation method
state = NetworkState.from_circuit(circuit, dtype='complex128', backend='torch', config={'num_hyper_samples':4})

# compute the state vector
sv = state.compute_state_vector()
print(f"state vector type: {type(sv)}, shape: {sv.shape}")

# compute the bitstring amplitude
bitstring = '0' * n_qubits
amplitude = state.compute_amplitude(bitstring)
print(f"Bitstring amplitude for {bitstring}: {amplitude}")

# compute batched bitstring amplitude with first qubit fixed at state 0 and second qubit at state 1
fixed = {qubits[0]: 0, qubits[1]: 1}  # This is equivalent to fixed = {0: 0, 1: 1}
batched_amplitudes = state.compute_batched_amplitudes(fixed)
print(f"Batched amplitudes shape: {batched_amplitudes.shape}")

# compute reduced density matrix for the first two qubits
where = qubits[:2] # This is equivalent to where = (0, 1)
rdm = state.compute_reduced_density_matrix(where)
print(f"RDM shape for {where}: {rdm.shape}")

# draw samples from the state object
nshots = 100
samples = state.compute_sampling(nshots)
print("Sampling results:")

print(samples)

# compute the expectation value for a series of Pauli operators
pauli_string = {'IXIXIXIX': 0.5, 'IYIYIYIY': 0.2, 'IZIZIZIZ': 0.3}
expec = state.compute_expectation(pauli_string)
print(f"{expec=}")
# release resources
state.free()
