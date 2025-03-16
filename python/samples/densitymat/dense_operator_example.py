# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dense operator example.
"""

import cupy as cp
import numpy as np
from cuquantum.densitymat import (
    tensor_product,
    full_matrix_product,
    DenseOperator,
    DenseMixedState,
    WorkStream,
    Operator,
    GPUCallback,
    CPUCallback,
)

dev = cp.cuda.Device()  # get current device
props = cp.cuda.runtime.getDeviceProperties(dev.id)
print("===== device info ======")
print("GPU-local-id:", dev.id)
print("GPU-name:", props["name"].decode())
print("GPU-clock:", props["clockRate"])
print("GPU-memoryClock:", props["memoryClockRate"])
print("GPU-nSM:", props["multiProcessorCount"])
print("GPU-major:", props["major"])
print("GPU-minor:", props["minor"])

np.random.seed(42)
cp.random.seed(42)

# Parameters
dtype = "complex128"
hilbert_space_dims = (4, 3, 5, 7)
num_modes = len(hilbert_space_dims)
dissipation_strengths = np.random.random((num_modes,))
modes_string = "abcdefghijkl"[: len(hilbert_space_dims)]
batch_size = 1


def take_complex_conjugate_transpose(arr):
    return arr.transpose(tuple(range(num_modes, 2 * num_modes)) + tuple(range(0, num_modes)) + (2 * num_modes,)).conj()


#############
# Hamiltonian
#############

h01_arr = cp.empty(
    (hilbert_space_dims[0], hilbert_space_dims[1], hilbert_space_dims[0], hilbert_space_dims[1], batch_size), dtype=dtype
)
h12_arr = cp.empty(
    (hilbert_space_dims[1], hilbert_space_dims[2], hilbert_space_dims[1], hilbert_space_dims[2], batch_size), dtype=dtype
)
h23_arr = cp.empty(
    (hilbert_space_dims[2], hilbert_space_dims[3], hilbert_space_dims[2], hilbert_space_dims[3], batch_size), dtype=dtype
)
h23_arr[:] = cp.random.normal(size=h23_arr.shape)


def callback1(t, args, arr):  # inplace callback
    # The library reconstructs args array with batch size as the last dimension
    assert len(args) == batch_size  # length of args should be the same as batch size
    for i in range(batch_size):
        assert len(args[i]) == 1  # only one element per args list in this example

    for i in range(hilbert_space_dims[0]):
        for j in range(hilbert_space_dims[1]):
            for b in range(batch_size):
                omega = args[0][b]
                arr[i, j, i, j, b] = (i + j) * np.sin(omega * t)


def callback2(t, args):  # out-of-place callback
    # The library reconstructs args array with batch size as the last dimension
    assert len(args) == batch_size  # length of args should be the same as batch size
    for i in range(batch_size):
        assert len(args[i]) == 1  # only one element per args list in this example

    arr = cp.empty(
        (hilbert_space_dims[1], hilbert_space_dims[2], hilbert_space_dims[1], hilbert_space_dims[2], batch_size), dtype=dtype
    )
    for i in range(hilbert_space_dims[1]):
        for j in range(hilbert_space_dims[2]):
            for b in range(batch_size):
                omega = args[0][b]
                arr[i, j, i, j, b] = (i + j) * np.cos(omega * t)
    return arr


h01_callback = GPUCallback(callback1, is_inplace=True)
h12_callback = GPUCallback(callback2, is_inplace=False)

h01_op = DenseOperator(h01_arr, h01_callback)
h12_op = DenseOperator(h12_arr, h12_callback)
h23_op = DenseOperator(h23_arr)

H = tensor_product((h01_op, [0, 1])) + tensor_product((h12_op, [1, 2])) + tensor_product((h23_op, [2, 3]))

print("Created an OperatorTerm for the Hamiltonian.")

#############
# Dissipators
#############

l0_arr = cp.empty((hilbert_space_dims[0], hilbert_space_dims[0], batch_size), dtype=dtype)
l1_arr = cp.empty((hilbert_space_dims[1], hilbert_space_dims[1], batch_size), dtype=dtype)
l2_arr = cp.empty((hilbert_space_dims[2], hilbert_space_dims[2], batch_size), dtype=dtype)
l3_arr = cp.empty((hilbert_space_dims[3], hilbert_space_dims[3], batch_size), dtype=dtype)

l0_arr[:] = cp.random.random(l0_arr.shape)
l1_arr[:] = cp.random.random(l1_arr.shape)
l2_arr[:] = cp.random.random(l2_arr.shape)


def callback3(t, args):
    # The library reconstructs args array with batch size as the last dimension
    assert len(args) == batch_size  # length of args should be the same as batch size
    for i in range(batch_size):
        assert len(args[i]) == 1  # only one element per args list in this example

    arr = np.empty((hilbert_space_dims[3], hilbert_space_dims[3], batch_size), dtype=dtype)
    for i in range(hilbert_space_dims[3]):
        for j in range(hilbert_space_dims[3]):
            for b in range(batch_size):
                omega = args[0][b]
                arr[i, j, b] = (i + j) * np.tan(omega * t)
    return arr


l3_callback = CPUCallback(callback3, is_inplace=False)

l0_op = DenseOperator(l0_arr)
l1_op = DenseOperator(l1_arr)
l2_op = DenseOperator(l2_arr)
l3_op = DenseOperator(l3_arr, l3_callback)

Ls = []
for i, l in enumerate([l0_op, l1_op, l2_op, l3_op]):
    l_dag_l = l.dag() @ l
    L = (
        tensor_product((l, [i], [False]), (l.dag(), [i], [True]))
        + tensor_product((-0.5 * l_dag_l, [i], [False]))
        + tensor_product((-0.5 * l_dag_l, [i], [True]))
    )
    Ls.append(dissipation_strengths[i] * L)

print("Created OperatorTerms for the Liouvillian.")

#############
# Liouvillian
#############

liouvillian = Operator(hilbert_space_dims, (H, -1j, False), (H, 1j, True), *[(L,) for L in Ls])

print("Created the Liouvillian operator.")

##########################
# Initial and final states
##########################

ctx = WorkStream()

rho = DenseMixedState(ctx, hilbert_space_dims, batch_size, dtype)
rho.attach_storage(cp.empty(rho.storage_size, dtype=dtype))
rho_arr = rho.view()
rho_arr[:] = cp.random.normal(size=rho_arr.shape)
if "complex" in dtype:
    rho_arr[:] += 1j * cp.random.normal(size=rho_arr.shape)
rho_arr += take_complex_conjugate_transpose(rho_arr)
rho_arr /= rho.trace()
print("Created a Haar random normalized mixed quantum state.")

rho_out = rho.clone(cp.zeros_like(rho.storage, order="F"))
print("Created zero initialized output state.")

#################
# Operator action
#################

liouvillian.prepare_action(ctx, rho)
liouvillian.compute_action(1.0, [3.5], rho, rho_out)

print("Finished computation and exit.")
