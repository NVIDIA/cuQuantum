# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np

import cuquantum
from cuquantum import cutensornet as cutn


print("cuTensorNet-vers:", cutn.get_version())
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
print("========================")

####################################################
# Expectation computation of a quantum circuit state
####################################################

# Quantum state configuration
num_qubits = 16
dim = 2
qubits_dims = (dim, ) * num_qubits # qubit size
print(f"Quantum circuit with {num_qubits} qubits")

#############
# cuTensorNet
#############

handle = cutn.create()
stream = cp.cuda.Stream()
data_type = cuquantum.cudaDataType.CUDA_C_64F

# Define quantum gate tensors on device
gate_h = 2**-0.5 * cp.asarray([[1,1], [1,-1]], dtype='complex128', order='F')
gate_h_strides = 0

# Pauli X gate
gate_x = cp.asarray([[0, 1], [1, 0]]).T.astype('complex128', order='F')
# Pauli Y gate
gate_y = cp.asarray([[0, -1j], [1j, 0]]).T.astype('complex128', order='F')
# Pauli Z gate
gate_z = cp.asarray([[1, 0], [0, -1]]).T.astype('complex128', order='F')

gate_cx = cp.asarray([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], dtype='complex128').reshape(2,2,2,2, order='F')
gate_cx_strides = 0

free_mem = dev.mem_info[0]
# use half of the totol free size
scratch_size = free_mem // 2
scratch_space = cp.cuda.alloc(scratch_size)
print(f"Allocated {scratch_size} bytes of scratch memory on GPU")

# Create the initial quantum state
quantum_state = cutn.create_state(handle, cutn.StatePurity.PURE, num_qubits, qubits_dims, data_type)
print("Created the initial quantum state")

# Construct the quantum circuit state with gate application
tensor_id = cutn.state_apply_tensor_operator(
        handle, quantum_state, 1, (0, ), 
        gate_h.data.ptr, gate_h_strides, 1, 0, 1)

for i in range(1, num_qubits):
    tensor_id = cutn.state_apply_tensor_operator(
        handle, quantum_state, 2, (i-1, i),  # control on i-1 while target on i
        gate_cx.data.ptr, gate_cx_strides, 1, 0, 1)
print("Quantum gates applied")

# Create an empty tensor network operator
hamiltonian = cutn.create_network_operator(handle, 
    num_qubits, qubits_dims, data_type)
# Append component (0.5 * Z1 * Z2) to the tensor network operator
num_modes = (1, 1) # Z1 acts on 1 mode, Z2 acts on 1 mode
modes_Z1 = (1, ) # state modes Z1 acts on
modes_Z2 = (2, ) # state modes Z2 acts on
state_modes = (modes_Z1, modes_Z2) # state modes (Z1 * Z2) acts on
gate_data = (gate_z.data.ptr, gate_z.data.ptr) # GPU pointers to gate data
operator_id = cutn.network_operator_append_product(handle, hamiltonian, 0.5,
                      2, num_modes, state_modes, 0, gate_data)
# Append component (0.25 * Y3) to the tensor network operator
num_modes = (1, ) # Y3 acts on 1 mode
modes_Y3 = (3, ) # state modes Y3 acts on
state_modes = (modes_Y3, ) # state modes (Y3) acts on
gate_data = (gate_y.data.ptr, ) # GPU pointers to gate data
operator_id = cutn.network_operator_append_product(handle, hamiltonian, 0.25,
                      1, num_modes, state_modes, 0, gate_data)

# Append component (0.13 * Y0 X2 Z3) to the tensor network operator
num_modes = (1, 1, 1) # Y0 acts on 1 mode, X2 acts on 1 mode, Z3 acts on 1 mode
modes_Y0 = (0, ) # state modes Y0 acts on
modes_X2 = (2, ) # state modes X2 acts on
modes_Z3 = (3, ) # state modes Z3 acts on
state_modes = (modes_Y0, modes_X2, modes_Z3) # state modes (Y0 * X2 * Z3) acts on
gate_data = (gate_y.data.ptr, gate_x.data.ptr, gate_z.data.ptr) # GPU pointers to gate data
operator_id = cutn.network_operator_append_product(handle, hamiltonian, 0.13,
                      3, num_modes, state_modes, 0, gate_data)
print("Constructed a tensor network operator: (0.5 * Z1 * Z2) + (0.25 * Y3) + (0.13 * Y0 * X2 * Z3)")

# Specify the quantum circuit expectation value computation
expectation = cutn.create_expectation(handle, quantum_state, hamiltonian)

# Configure the quantum circuit expectation value computation with hyper samples for the contraction optimizer
num_hyper_samples_dtype = cutn.expectation_get_attribute_dtype(cutn.ExpectationAttribute.CONFIG_NUM_HYPER_SAMPLES)
num_hyper_samples = np.asarray(8, dtype=num_hyper_samples_dtype)
cutn.expectation_configure(handle, expectation, 
    cutn.ExpectationAttribute.CONFIG_NUM_HYPER_SAMPLES, 
    num_hyper_samples.ctypes.data, num_hyper_samples.dtype.itemsize)

# Prepare the computation of the specified quantum circuit expectation value
work_desc = cutn.create_workspace_descriptor(handle)
cutn.expectation_prepare(handle, expectation, scratch_size, work_desc, stream.ptr)
print("Prepare the computation of the specified quantum circuit expectation value")

flops_dtype = cutn.expectation_get_attribute_dtype(cutn.ExpectationAttribute.INFO_FLOPS)
flops = np.zeros(1, dtype=flops_dtype)
cutn.expectation_get_info(handle, expectation, cutn.ExpectationAttribute.INFO_FLOPS, flops.ctypes.data, flops.dtype.itemsize)
print(f"Total flop count = {flops.item()/1e9} GFlop")

workspace_size_d = cutn.workspace_get_memory_size(handle, 
    work_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)

if workspace_size_d <= scratch_size:
    cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, scratch_space.ptr, workspace_size_d)
else:
    print("Error:Insufficient workspace size on Device")
    cutn.destroy_workspace_descriptor(work_desc)
    cutn.destroy_expectation(expectation)
    cutn.destroy_network_operator(hamiltonian)
    cutn.destroy_state(quantum_state)
    cutn.destroy(handle)
    del scratch_space
    print("Free resource and exit.")
    exit()
print("Set the workspace buffer")

# Compute the specified quantum circuit expectation value
expectation_value = np.empty(1, dtype='complex128')
state_norm = np.empty(1, dtype='complex128')
cutn.expectation_compute(handle, expectation, 
    work_desc, expectation_value.ctypes.data, state_norm.ctypes.data, stream.ptr)
stream.synchronize()
print("Computed the specified quantum circuit state amplitudes")

print(f"expectation value = {expectation_value.item()}")
print(f"norm of the state = {state_norm.item()}")

cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_expectation(expectation)
cutn.destroy_network_operator(hamiltonian)
cutn.destroy_state(quantum_state)
cutn.destroy(handle)
del scratch_space
print("Free resource and exit.")