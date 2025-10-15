# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MPS simulation of a noisy quantum state with unitary tensor channels.

The sampling is drawn with a trajectory based simulation.
"""
import numpy as np

from cuquantum.tensornet.experimental import NetworkState, MPSConfig


def random_unitary(n):
    """
    Create a random unitary tensor
    """
    mat = np.random.random((2**n, 2**n)) + 1.j * np.random.random((2**n, 2**n))
    q, r = np.linalg.qr(mat)
    unitary = q.reshape((2, 2) * n)
    return unitary


N = 20

mps_config = MPSConfig(max_extent=256,
                       rel_cutoff=1e-5,
                       abs_cutoff=1e-5,
                       algorithm='gesvdj')
state_mode_extents = (2,) * N
dtype = 'complex128'
state = NetworkState(state_mode_extents, dtype=dtype, config=mps_config)

np.random.seed(1)

h_op = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]).astype("complex128")

for i in range(N):
    modes_one_body = (i,)
    tensor_id = state.apply_tensor_operator(modes_one_body,
                                            h_op,
                                            unitary=True,
                                            immutable=True)

    for j in range(i + 1, N):
        u = random_unitary(2)
        modes_two_body = (i, j)
        tensor_id = state.apply_tensor_operator(modes_two_body,
                                                u,
                                                unitary=True,
                                                immutable=True)
for num_trajectories in range(2):
    trajectory_sample = state.compute_sampling(100)

state.free() 