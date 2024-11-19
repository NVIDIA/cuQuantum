# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Contraction based tensor network simulation on a custom state with arbitrary dimensions. 
The generic state is constructed by iteratively applying tensor operators with the following topology:

Vacuum:         A   B   C   D   C   F
                |   |   |   |   |   |
one body op     O   O   O   O   O   O
                |   |   |   |   |   |
two body op     GGGGG   GGGGG   GGGGG
                |   |   |   |   |   |
two body op     |   GGGGG   GGGGG   |
                |   |   |   |   |   |

The tensor operands for the one body and two body operators are provided as Numpy ndarrays.
"""
import numpy as np

from cuquantum.cutensornet.experimental import NetworkState, NetworkOperator

# specify the dimensions of the tensor network state
state_mode_extents = (4, 2, 4, 4, 2, 4)
n_state_modes = len(state_mode_extents)
dtype = 'complex128'
print(f"state dimensions: {state_mode_extents}")

# create random operators
np.random.seed(4)
random_complex = lambda *args, **kwargs: np.random.random(*args, **kwargs) + 1.j * np.random.random(*args, **kwargs)

# create an emtpy NetworkState object, by default it will tensor network contraction as simulation method
state = NetworkState(state_mode_extents, dtype=dtype)

# apply one body tensor operators to the tensor network state
for i, dim in enumerate(state_mode_extents):
    modes_one_body = (i, )
    op_one_body = random_complex((dim, dim))
    tensor_id = state.apply_tensor_operator(modes_one_body, op_one_body, unitary=False)
    print(f"Apply one body operator to {modes_one_body}, tensor id {tensor_id}")

# apply two body tensor operators to the tensor network state
for i in range(2):
    for site in range(i, n_state_modes, 2):
        if site + 1 < n_state_modes:
            modes_two_body = (site, site+1)
            shape = (state_mode_extents[i], state_mode_extents[i+1]) * 2
            op_two_body = random_complex(shape)
            tensor_id = state.apply_tensor_operator(modes_two_body, op_two_body, unitary=False)
            print(f"Apply two body operator to {modes_two_body}, tensor id {tensor_id}")

# compute the state vector
sv = state.compute_state_vector()
print(f"state vector type: {type(sv)}, shape: {sv.shape}")

# compute the un-normalized bitstring amplitude
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

# create a random network operator from a tensor product
expec_prod_operators = []
expec_prod_modes = []
for i, dim in enumerate(state_mode_extents):
    expec_prod_modes.append((i, ))
    expec_prod_operators.append(random_complex((dim, dim)))

expec_operator = NetworkOperator(state_mode_extents, dtype=dtype)
expec_operator.append_product(1, expec_prod_modes, expec_prod_operators)
expec, norm = state.compute_expectation(expec_operator, return_norm=True)
print(f"normalized expectation value = {expec/norm}")

# release resources
state.free()
