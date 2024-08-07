# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
An example showing the caching feature of contraction based tensor network simulation. 

Logging is turned on to provide details for performing the same computation consecutively using the same method.
Note that this feature only applies to contraction based simulation and MPS simulation without value based truncations.
"""
import logging

import qiskit
import cupy as cp

from cuquantum.cutensornet.experimental import NetworkState, TNConfig

# set logging level to INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

# create a QFT circuit
n_qubits = 8
qubits = list(range(n_qubits))
circuit = qiskit.QuantumCircuit(n_qubits)
qft = qiskit.circuit.library.QFT(num_qubits=n_qubits)
circuit.append(qft, qubits)
print(circuit)

# select tensor network contraction as the simulation method
config = TNConfig(num_hyper_samples=4)  # also works for MPS simulation with config ={'max_extent': 2}

# create a NetworkState object
state = NetworkState.from_circuit(circuit, dtype='complex128', backend='cupy', config=config)

sv = state.compute_state_vector()
# compute the bitstring amplitude
# Note that for the second amplitude computation call, the preparation step is cached to reduce the overhead
for bit_state in (0, 1):
    bitstring = str(bit_state) * n_qubits
    amplitude = state.compute_amplitude(bitstring)
    amplitude_ref = sv[(bit_state, )* n_qubits]
    print(f"Bitstring amplitude for {bitstring}: {amplitude}")
    assert cp.allclose(amplitude, amplitude_ref)

# compute reduced density matrix for the first qubit with second qubit fixed at |0> or |1> state. 
# Note that for the same where argument, preparation of the reduced density matrix computation 
# is cached to reduce the overhead for the second call
where = (0, )

for second_bit_state in (0, 1):
    fixed = {1 : second_bit_state} # fix second qubit at |0> or |1>
    rdm = state.compute_reduced_density_matrix(where, fixed=fixed)
    rdm_ref = cp.einsum('ijklmno,Ijklmno->iI', sv[:,second_bit_state], sv[:,second_bit_state].conj())
    print(f"RDM shape for {where=}, {fixed=}: {rdm.shape}")
    assert cp.allclose(rdm, rdm_ref)

# draw samples from the state object
# For the same optional modes parameter, preparation of the sampling is cached to reduce the overhead for the second call
for nshots in (100, 1000):
    samples = state.compute_sampling(nshots)
    print(f"Sampling results for {nshots=}:")
    print(samples)

# release resources
state.free()
