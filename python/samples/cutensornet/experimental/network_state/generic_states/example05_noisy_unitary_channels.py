# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Contraction-based tensor network simulation of a noisy quantum state with unitary tensor channels.
The custom state is constructed by iteratively applying tensor operators and unitary bitflip tensor channels with the following topology:

Vacuum:                         A   B   C   D   E   F
                                |   |   |   |   |   |
one body op                     O   O   O   O   O   O
                                |   |   |   |   |   |
bit flip unitary channel        U   U   U   U   U   U
                                |   |   |   |   |   |
two body op                     GGGGG   GGGGG   GGGGG
                                |   |   |   |   |   |
two body op                     |   GGGGG   GGGGG   |
                                |   |   |   |   |   |

The expectation value is statistically computed with a trajectory based simulation.
"""
import cupy as cp

from cuquantum.cutensornet.experimental import NetworkState, NetworkOperator

#
# specify the dimensions of the tensor network state
n_state_modes = 6
state_mode_extents = (2, ) * n_state_modes
dtype = 'complex128'
n_trajectories = 1000

def random_unitary(n):
    """
    Create a random unitary tensor
    """
    mat = cp.random.random((2**n, 2**n)) + 1.j * cp.random.random((2**n, 2**n))
    q, r = cp.linalg.qr(mat)
    unitary = q.reshape((2,2)*n)
    return unitary

# create random operators and random unitary channels
cp.random.seed(1)
op_one_body = random_unitary(1)
op_two_body = random_unitary(2)

bitflip_channel = [
    cp.eye(2, dtype=dtype), # I
    cp.asarray([[0, 1], [1, 0]], dtype=dtype) # X
]
bitflip_probabilities = [0.95, 0.05] # 5% for bitflip

# create an emtpy NetworkState object, by default it will tensor network contraction as simulation method
state = NetworkState(state_mode_extents, dtype=dtype)

# apply one body tensor operators & unitary channels to the tensor network state
for i in range(n_state_modes):
    modes_one_body = (i, )
    tensor_id = state.apply_tensor_operator(modes_one_body, op_one_body, unitary=True, immutable=True)
    channel_id = state.apply_unitary_tensor_channel(modes_one_body, bitflip_channel, bitflip_probabilities)

# apply two body tensor operators & unitary channels to the tensor network state
for i in range(2):
    for site in range(i, n_state_modes, 2):
        if site + 1 < n_state_modes:
            modes_two_body = (site, site+1)
            tensor_id = state.apply_tensor_operator(modes_two_body, op_two_body, unitary=True, immutable=True)

# compute the normalized expectation value for a series of Pauli operators
pauli_string = {'IXIXIX': 0.5, 'IYIYIY': 0.2, 'IZIZIZ': 0.3}

# explicitly construct NetworkOperator to activate caching mechanism
network_operator = NetworkOperator.from_pauli_strings(pauli_string, dtype=dtype)
expec_counter = dict()
for i in range(n_trajectories):
    expec, norm = state.compute_expectation(network_operator, return_norm=True)
    expec = expec.real / norm
    if expec not in expec_counter:
        expec_counter[expec] = 0
    expec_counter[expec] += 1

expec_average = 0
for expec, n_count in sorted(expec_counter.items(), key=lambda item: item[1], reverse=True):
    print(f"{expec=:.6f}, frequency={n_count / n_trajectories}")
    expec_average += expec * n_count / n_trajectories
print(f"Expec average: {expec_average:.6f}")
# release resources
state.free()
