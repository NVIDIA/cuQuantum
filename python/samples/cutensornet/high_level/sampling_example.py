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

#####################################
# Sampling of a quantum circuit state
#####################################

# Quantum state configuration
num_samples = 100
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

# Define quantum gate tensors in device memory
gate_h = 2**-0.5 * cp.asarray([[1,1], [1,-1]], dtype='complex128', order='F')
gate_h_strides = 0

gate_cx = cp.asarray([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], dtype='complex128').reshape(2,2,2,2, order='F')
gate_cx_strides = 0

# Allocate device memory for the samples
samples = np.empty((num_qubits, num_samples), dtype='int64', order='F') # samples are stored in F order with shape (num_qubits, num_qubits)

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

# Create the quantum circuit sampler
sampler = cutn.create_sampler(handle, quantum_state, num_qubits, 0)

# Configure the quantum circuit sampler with hyper samples for the contraction optimizer
num_hyper_samples_dtype = cutn.sampler_get_attribute_dtype(cutn.SamplerAttribute.CONFIG_NUM_HYPER_SAMPLES)
num_hyper_samples = np.asarray(8, dtype=num_hyper_samples_dtype)
cutn.sampler_configure(handle, sampler, 
    cutn.SamplerAttribute.CONFIG_NUM_HYPER_SAMPLES, 
    num_hyper_samples.ctypes.data, num_hyper_samples.dtype.itemsize)
rng_dtype = cutn.sampler_get_attribute_dtype(cutn.SamplerAttribute.CONFIG_DETERMINISTIC)
rng = np.asarray(13, dtype=rng_dtype)
cutn.sampler_configure(handle, sampler, 
    cutn.SamplerAttribute.CONFIG_DETERMINISTIC, 
    rng.ctypes.data, rng.dtype.itemsize)

# Prepare the quantum circuit sampler
work_desc = cutn.create_workspace_descriptor(handle)
cutn.sampler_prepare(handle, sampler, scratch_size, work_desc, stream.ptr)
print("Prepared the specified quantum circuit state sampler")

flops_dtype = cutn.sampler_get_attribute_dtype(cutn.SamplerAttribute.INFO_FLOPS)
flops = np.zeros(1, dtype=flops_dtype)
cutn.sampler_get_info(handle, sampler, cutn.SamplerAttribute.INFO_FLOPS, flops.ctypes.data, flops.dtype.itemsize)
print(f"Total flop count for sampling = {flops.item()/1e9} GFlop")

workspace_size_d = cutn.workspace_get_memory_size(handle, 
    work_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)

if workspace_size_d <= scratch_size:
    cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, scratch_space.ptr, workspace_size_d)
else:
    print("Error:Insufficient workspace size on Device")
    cutn.destroy_workspace_descriptor(work_desc)
    cutn.destroy_sampler(sampler)
    cutn.destroy_state(quantum_state)
    cutn.destroy(handle)
    del scratch_space
    print("Free resource and exit.")
    exit()
print("Set the workspace buffer")

# Sample the quantum circuit state
cutn.sampler_sample(handle, sampler, num_samples, work_desc, samples.ctypes.data, stream.ptr)
stream.synchronize()
print("Performed quantum circuit state sampling")
print("Bit-string samples:")
hist = np.unique(samples.T, axis=0, return_counts=True)
for bitstring, count in zip(*hist):
    bitstring = np.array2string(bitstring, separator='')[1:-1]
    print(f"{bitstring}: {count}")

cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_sampler(sampler)
cutn.destroy_state(quantum_state)
cutn.destroy(handle)
del scratch_space
print("Free resource and exit.")
