# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Approximate MPS simulation on a custom state constructed by iteratively applying the following operators, including matrix product operators (MPO):

Vacuum:         A   B   C   D   E   F
                |   |   |   |   |   |
one body op     O   O   O   O   O   O
                |   |   |   |   |   |
MPO             G---G---G---G---G---G
                |   |   |   |   |   |
two body op     GGGGG   GGGGG   GGGGG
                |   |   |   |   |   |
two body op     |   GGGGG   GGGGG   |
                |   |   |   |   |   |

The tensor operands for all operators are provided as CuPy ndarrays.
"""
import cupy as cp

from cuquantum.cutensornet.experimental import NetworkState, NetworkOperator

# specify the dimensions of the tensor network state
n_state_modes = 6
state_mode_extents = (2, ) * n_state_modes
dtype = 'complex128'

# create random operators
cp.random.seed(4)
random_complex = lambda *args, **kwargs: cp.random.random(*args, **kwargs) + 1.j * cp.random.random(*args, **kwargs)
op_one_body = random_complex((2, 2,))
op_two_body = random_complex((2, 2, 2, 2))

mpo_bond_dim = 3
mpo_operators = []
for i in range(n_state_modes):
    if i == 0:
        shape  = (2, mpo_bond_dim, 2,)
    elif i == n_state_modes - 1:
        shape = (mpo_bond_dim, 2, 2)
    else:
        shape = (mpo_bond_dim, 2, mpo_bond_dim, 2)
    mpo_operators.append(random_complex(shape))
    
# create an emtpy NetworkState object with MPS as simulation method
state = NetworkState(state_mode_extents, dtype=dtype, config={'max_extent': 4, 'rel_cutoff': 1e-5})

# apply one body tensor operators to the tensor network state
for i in range(n_state_modes):
    modes_one_body = (i, )
    tensor_id = state.apply_tensor_operator(modes_one_body, op_one_body, unitary=False)
    print(f"Apply one body operator to {modes_one_body}, tensor id {tensor_id}")

# apply the MPO
mpo_modes = list(range(n_state_modes))
mpo = NetworkOperator(state_mode_extents, dtype=dtype)
mpo.append_mpo(1, mpo_modes, mpo_operators)
state.apply_network_operator(mpo, unitary=False) # equivalent to state.apply_mpo(mpo_modes, mpo_operators, unitary=False)

# apply two body tensor operators to the tensor network state
for i in range(2):
    for site in range(i, n_state_modes, 2):
        if site + 1 < n_state_modes:
            modes_two_body = (site, site+1)
            tensor_id = state.apply_tensor_operator(modes_two_body, op_two_body, unitary=False)
            print(f"Apply two body operator to {modes_two_body}, tensor id {tensor_id}")

# Optional, compute the final mps representation
mps_tensors = state.compute_output_state()
for i, o in enumerate(mps_tensors):
    print(f"Site {i}, MPS tensor shape: {o.shape}")
    
# compute the state vector
sv = state.compute_state_vector()
print(f"state vector type: {type(sv)}, shape: {sv.shape}, device: {sv.device}")

# compute the bitstring amplitude
bitstring = '0' * n_state_modes
amplitude, norm = state.compute_amplitude(bitstring, return_norm=True)
prob = abs(amplitude) ** 2 / norm
print(f"Bitstring amplitude for {bitstring}: {amplitude}, prob={prob}")

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

# compute the normalized expectation value for the MPO
expec, norm = state.compute_expectation(mpo, return_norm=True)
print(f"normalized expectation value = {expec/norm}")

# release resources
state.free()
