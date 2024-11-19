# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Contraction based tensor network simulation on a custom state constructed by iteratively applying tensor operators with the following topology:

Vacuum:         A   B   C   D   E   F
                |   |   |   |   |   |
one body op     O   O   O   O   O   O
                |   |   |   |   |   |
two body op     GGGGG   GGGGG   GGGGG
                |   |   |   |   |   |
two body op     |   GGGGG   GGGGG   |
                |   |   |   |   |   |

The tensor operands for the one body and two body operators are provided as PyTorch gpu tensors.
"""
import torch

from cuquantum.cutensornet.experimental import NetworkState

# specify the dimensions of the tensor network state
n_state_modes = 6
state_mode_extents = (2, ) * n_state_modes
dtype = 'complex128'

# create random operators
torch.manual_seed(0)
op_one_body = torch.rand(2, 2, dtype=getattr(torch, dtype), device='cuda')
op_two_body = torch.rand(2, 2, 2, 2, dtype=getattr(torch, dtype), device='cuda')

# create an emtpy NetworkState object, by default it will tensor network contraction as simulation method
state = NetworkState(state_mode_extents, dtype=dtype)

# apply one body tensor operators to the tensor network state
for i in range(n_state_modes):
    modes_one_body = (i, )
    tensor_id = state.apply_tensor_operator(modes_one_body, op_one_body, unitary=False)
    print(f"Apply one body operator to {modes_one_body}, tensor id {tensor_id}")

# apply two body tensor operators to the tensor network state
for i in range(2):
    for site in range(i, n_state_modes, 2):
        if site + 1 < n_state_modes:
            modes_two_body = (site, site+1)
            tensor_id = state.apply_tensor_operator(modes_two_body, op_two_body, unitary=False)
            print(f"Apply two body operator to {modes_two_body}, tensor id {tensor_id}")

# compute the state vector
sv = state.compute_state_vector()
print(f"state vector type: {type(sv)}, shape: {sv.shape}, device: {sv.device}")

# compute the bitstring amplitude
bitstring = '0' * n_state_modes
amplitude = state.compute_amplitude(bitstring)
print(f"Bitstring amplitude for {bitstring}: {amplitude}")

# compute batched bitstring amplitude with first mode fixed at state 0 and second mode at state 1
fixed = {0: 0, 1: 1}
batched_amplitudes = state.compute_batched_amplitudes(fixed)
print(f"Batched amplitudes shape: {batched_amplitudes.shape}")

# compute reduced density matrix for the first two modes
where = (0, 1)
rdm = state.compute_reduced_density_matrix(where)
print(f"RDM shape for {where}: {rdm.shape}")

# draw samples from the state object
nshots = 100
samples = state.compute_sampling(nshots)
print("Sampling results:")
print(samples)

# compute the normalized expectation value for a series of Pauli operators
pauli_string = {'IXIXIX': 0.5, 'IYIYIY': 0.2, 'IZIZIZ': 0.3}
expec, norm = state.compute_expectation(pauli_string, return_norm=True)
expec = expec.real / norm
print(f"{expec=}")

# release resources
state.free()
