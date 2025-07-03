# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

r"""
Gradient computation using tensor callback.

This example is a driven Jaynes-cummings model with the following Hamiltonian:

H = \delta \sigma_z / 2 + cos(\omega * t) (a^\dag \sigma_- + a \sigma_+)

The second term cos(\omega * t) (a^\dag \sigma_- + a \sigma_+) is created as a tensor callback.
"""

import cupy as cp

from cuquantum.densitymat import (
    DenseMixedState,
    DenseOperator,
    Operator,
    tensor_product,
    GPUCallback,
    WorkStream,
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

cp.random.seed(42)

def take_complex_conjugate_transpose(arr):
    return arr.transpose(tuple(range(num_modes, 2 * num_modes)) + tuple(range(0, num_modes)) + (2 * num_modes,)).conj()

############
# Parameters
############

N = 10 # resonator levels
dtype = "complex128"
hilbert_space_dims = (N, 2)
num_modes = 2
batch_size = 1
time = 1.0
delta = 1.2
params = [3.5]

#############
# Hamiltonian
#############

def tensor_callback(t, args, storage):
    """
    Drive callback for the time-dependent Hamiltonian.
    """
    resonator_dim, qubit_dim = storage.shape[:2]
    for i in range(resonator_dim - 1):
        for j in range(qubit_dim - 1):
            storage[i, j + 1, i + 1, j] = cp.sqrt(i + 1) * cp.sqrt(j + 1) * cp.cos(args[0] * t)
            storage[i + 1, j, i, j + 1] = cp.sqrt(i + 1) * cp.sqrt(j + 1) * cp.cos(args[0] * t)


def tensor_grad_callback(t, args, tensor_grad, params_grad):
    """
    Gradient of the drive callback for the time-dependent Hamiltonian.
    """
    resonator_dim, qubit_dim = tensor_grad.shape[:2]
    for i in range(resonator_dim - 1):
        for j in range(qubit_dim - 1):
            params_grad[0] += 2 * (tensor_grad[i, j + 1, i + 1, j] * cp.sqrt(i + 1) * cp.sqrt(j + 1) * (-t) * cp.sin(args[0] * t)).real
            params_grad[0] += 2 * (tensor_grad[i + 1, j, i, j + 1] * cp.sqrt(i + 1) * cp.sqrt(j + 1) * (-t) * cp.sin(args[0] * t)).real


wrapped_tensor_callback = GPUCallback(tensor_callback, is_inplace=True, gradient_callback=tensor_grad_callback)

sigma_z_data = cp.array([[1, 0], [0, -1]], dtype=dtype).astype(dtype, order='F')

sigma_z = DenseOperator(sigma_z_data)
two_body_el_op = DenseOperator(
    cp.zeros((*hilbert_space_dims, *hilbert_space_dims), dtype=dtype), callback=wrapped_tensor_callback)

H = tensor_product((sigma_z_data, [1]), coeff=delta / 2) + tensor_product((two_body_el_op, [0, 1]))

print("Created an OperatorTerm for the Hamiltonian.")

#############
# Liouvillian
#############

liouvillian = Operator(hilbert_space_dims)
liouvillian.append(H, -1j, False)
liouvillian.append(H, 1j, True)

print("Created a Liouvillian operator.")

################
# Density matrix
################

ctx = WorkStream()

state_in = DenseMixedState(ctx, hilbert_space_dims, batch_size, dtype)
state_in.attach_storage(cp.empty(state_in.storage_size, dtype=dtype))
state_in_arr = state_in.view()
state_in_arr[:] = cp.random.normal(size=state_in_arr.shape)
if dtype.startswith("complex"):
    state_in_arr[:] += 1j * cp.random.normal(size=state_in_arr.shape)
state_in_arr += take_complex_conjugate_transpose(state_in_arr)
state_in_arr /= state_in.trace()
print("Created a Haar random normalized mixed quantum state.")

state_out = state_in.clone(cp.zeros_like(state_in.storage, order='F'))
print("Created zero initialized output state.")

#################
# Operator action
#################

liouvillian.prepare_action(ctx, state_in)
liouvillian.compute_action(time, params, state_in, state_out)

print("Finished computing operator action.")

######################
# Gradient computation
######################

state_out_adj = state_out.clone(state_out.storage.conj().copy(order='F'))
state_in_adj = state_in.clone(cp.zeros_like(state_in.storage, order='F'))
liouvillian.prepare_action_gradient_backward(ctx, state_in, state_out_adj)
params_grad = liouvillian.compute_action_gradient_backward(time, params, state_in, state_out_adj, state_in_adj)

print("Finished computing operator action gradient.")

print(f"Gradient with respect to the parameter is {params_grad}.")
