# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quantum circuit simulation based on approximate matrix product state (MPS). 

CuPy ndarrays are used as inputs and output.
"""
from qiskit.circuit.random import random_circuit
 
from cuquantum.cutensornet.experimental import NetworkState, MPSConfig

# create a random circuit with random unitary gates
n_qubits = 8
depth = 6
max_operands = 4  # Maximum number of qubit operands for each gate

circuit = random_circuit(n_qubits, depth, max_operands=max_operands, measure=True, seed=10)
print(circuit)

# select MPS as simulation method with truncation parameters
config = MPSConfig(max_extent=4, rel_cutoff=1e-5)

# create a NetworkState object and use it in a context manager
with NetworkState.from_circuit(circuit, dtype='complex128', backend='cupy', config=config) as state:
    # Optional, compute the final mps representation
    mps_tensors = state.compute_output_state()
    for i, o in enumerate(mps_tensors):
        print(f"Site {i}, MPS tensor shape: {o.shape}")

    # Optional, compute the state vector
    sv = state.compute_state_vector()
    print(f"state vector type: {type(sv)}, shape: {sv.shape}")

    # compute the bitstring amplitude
    bitstring = '0' * n_qubits
    amplitude = state.compute_amplitude(bitstring)
    print(f"Bitstring amplitude for {bitstring}: {amplitude}")

    # compute batched bitstring amplitude with first qubit fixed at state 0 and second qubit at state 1
    fixed = {0: 0, 1: 1} # This is equivalent to fixed = {qubits[0]: 0, qubits[1]: 1}
    batched_amplitudes = state.compute_batched_amplitudes(fixed)
    print(f"Batched amplitudes shape: {batched_amplitudes.shape}")

    # compute reduced density matrix for the first two qubits
    where = (0, 1) # This is equivalent to where = qubits[:2]
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
