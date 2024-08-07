# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
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
print("GPU-name:", props["name"].decode())
print("GPU-clock:", props["clockRate"])
print("GPU-memoryClock:", props["memoryClockRate"])
print("GPU-nSM:", props["multiProcessorCount"])
print("GPU-major:", props["major"])
print("GPU-minor:", props["minor"])
print("========================")

##############################################
# Sampling of a quantum circuit state with MPO
##############################################

# Quantum state configuration
num_samples = 128
num_qubits = 6
dim = 2
qudit_dims = (dim, ) * num_qubits # qubit size
mps_bond_dim = 8 # maximal output MPS bond dimension
mpo_bond_dim = 2 # MPO bond dimension
mpo_num_layers = 5 # number of MPO layers
mpo_num_sites = 4 # number of MPO sites
print(f"Quantum circuit with {num_qubits} qubits")

"""
Action of five alternating four-site MPO gates (operators)
on the 6-quqit quantum register (illustration):

  Q----X---------X---------X----
       |         |         |
  Q----X---------X---------X----
       |         |         |
  Q----X----X----X----X----X----
       |    |    |    |    |
  Q----X----X----X----X----X----
            |         |
  Q---------X---------X---------
            |         |
  Q---------X---------X---------

    |layer|
"""


#############
# cuTensorNet
#############

handle = cutn.create()
stream = cp.cuda.Stream()
data_type = cuquantum.cudaDataType.CUDA_C_64F

# Define MPO tensors in device memory  
""" 
MPO tensor mode numeration (open boundary condition):
     2            3                   2        <bra|
     |            |                   |
     X--1------0--X--2---- ... ----0--X
     |            |                   |        |ket>
     0            1                   1
"""
rng = cp.random.default_rng(2024)
mpo_tensors = []
mpo_mode_extents = []
mpo_tensor_strides = []
mpo_tensor_ptrs = []
for i in range(mpo_num_sites):
    if i == 0:
        shape = (dim, mpo_bond_dim, dim)
    elif i == mpo_num_sites - 1:
        shape = (mpo_bond_dim, dim, dim)
    else:
        shape = (mpo_bond_dim, dim, mpo_bond_dim, dim)
    tensor = rng.random(shape, dtype='float64') + 1j * rng.random(shape, dtype='float64')
    mpo_mode_extents.append(shape)
    mpo_tensors.append(tensor)
    mpo_tensor_ptrs.append(tensor.data.ptr)
    mpo_tensor_strides.append([stride_in_bytes // tensor.itemsize for stride_in_bytes in tensor.strides])

print("Allocated and defined MPO tensors in GPU memory")

# Allocate device memory for the final MPS state
mps_bond_dim = 2
mps_tensor_extents = []
mps_tensor_strides = []
mps_tensors = []
mps_tensor_ptrs = []
for i in range(num_qubits):
    if i == 0:
        extents = (2, mps_bond_dim)
    elif i == num_qubits - 1:
        extents = (mps_bond_dim, 2)
    else:
        extents = (mps_bond_dim, 2, mps_bond_dim)
    mps_tensor_extents.append(extents)
    tensor = cp.zeros(extents, dtype='complex128')
    mps_tensors.append(tensor)
    mps_tensor_ptrs.append(tensor.data.ptr)
    mps_tensor_strides.append([stride_in_bytes // tensor.itemsize for stride_in_bytes in tensor.strides])

# Allocate device memory for the samples
samples = np.empty((num_qubits, num_samples), dtype='int64', order='F') # samples are stored in F order with shape (num_qubits, num_qubits)

free_mem = dev.mem_info[0]
# use half of the totol free size
scratch_size = free_mem // 2
scratch_space = cp.cuda.alloc(scratch_size)
print(f"Allocated {scratch_size} bytes of scratch memory on GPU")

# Create the vacuum quantum state
quantum_state = cutn.create_state(handle, cutn.StatePurity.PURE, num_qubits, qudit_dims, data_type)
print("Created the initial quantum state")

# Construct the MPO tensor network operators
tn_operator1 = cutn.create_network_operator(handle, 
    num_qubits, qudit_dims, data_type)
component_id = cutn.network_operator_append_mpo(handle, 
    tn_operator1, 1, mpo_num_sites, (0, 1, 2, 3), mpo_mode_extents, 
    mpo_tensor_strides, mpo_tensor_ptrs, cutn.BoundaryCondition.OPEN)
assert component_id == 0

tn_operator2 = cutn.create_network_operator(handle, 
    num_qubits, qudit_dims, data_type)
component_id = cutn.network_operator_append_mpo(handle, 
    tn_operator2, 1, mpo_num_sites, (2, 3, 4, 5), mpo_mode_extents, 
    mpo_tensor_strides, mpo_tensor_ptrs, cutn.BoundaryCondition.OPEN)
assert component_id == 0
print("Constructed two MPO tensor network operators")

# Apply the MPO tensor network operators to the quantum state
for layer in range(mpo_num_layers):
    if layer % 2 == 0:
        tn_operator = tn_operator1
    else:
        tn_operator = tn_operator2
    operator_id = cutn.state_apply_network_operator(handle, 
        quantum_state, tn_operator, 1, 0, 0)
print(f"Applied {mpo_num_layers} MPO gates to the quantum state")


# Specify the target MPS state
cutn.state_finalize_mps(handle, quantum_state, cutn.BoundaryCondition.OPEN, mps_tensor_extents, mps_tensor_strides)
print("Set the final MPS representation")

# Configure the MPS computation
svd_algorithm_dtype = cutn.state_get_attribute_dtype(cutn.StateAttribute.MPS_SVD_CONFIG_ALGO)
svd_algorithm = np.array(cutn.TensorSVDAlgo.GESVDJ, dtype=svd_algorithm_dtype)
cutn.state_configure(handle, quantum_state, 
    cutn.StateAttribute.MPS_SVD_CONFIG_ALGO, svd_algorithm.ctypes.data, svd_algorithm.dtype.itemsize)
mpo_application_dtype = cutn.state_get_attribute_dtype(cutn.StateAttribute.CONFIG_MPS_MPO_APPLICATION)
mpo_application = np.array(cutn.StateMPOApplication.EXACT, dtype=mpo_application_dtype)
cutn.state_configure(handle, quantum_state, 
    cutn.StateAttribute.CONFIG_MPS_MPO_APPLICATION, mpo_application.ctypes.data, mpo_application.dtype.itemsize)
print("Configured the MPS computation")


# Prepare the specified quantum circuit for MPS computation
work_desc = cutn.create_workspace_descriptor(handle)
cutn.state_prepare(handle, quantum_state, scratch_size, work_desc, stream.ptr)
print("Prepared the specified quantum circuit for MPS computation")

workspace_size_d = cutn.workspace_get_memory_size(handle, 
    work_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)
if workspace_size_d <= scratch_size:
    print(f"{workspace_size_d=}")
    cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, scratch_space.ptr, workspace_size_d)
else:
    print("Error:Insufficient workspace size on Device")
    cutn.destroy_workspace_descriptor(work_desc)
    cutn.destroy_state(quantum_state)
    cutn.destroy_network_operator(tn_operator1)
    cutn.destroy_network_operator(tn_operator2)
    cutn.destroy(handle)
    del scratch_space
    print("Free resource and exit.")
    exit()
print("Set the workspace buffer for MPS computation")

# Compute the final MPS state
extents_out, strides_out = cutn.state_compute(handle, quantum_state, work_desc, mps_tensor_ptrs, stream.ptr)

# If a lower extent is found during runtime, the cupy.ndarray container must be adjusted to reflect the lower extent
for i, (extent_in, extent_out) in enumerate(zip(mps_tensor_extents, extents_out)):
    if extent_in != tuple(extent_out):
        stride_out = [s * mps_tensors[0].itemsize for s in strides_out[i]]
        mps_tensors[i] = cp.ndarray(extent_out, dtype=mps_tensors[i].dtype, memptr=mps_tensors[i].data, strides=stride_out)
print("Computed the final MPS representation")

# Create the quantum circuit sampler
sampler = cutn.create_sampler(handle, quantum_state, num_qubits, 0)

# Configure the quantum circuit sampler with hyper samples for the contraction optimizer
num_hyper_samples_dtype = cutn.sampler_get_attribute_dtype(cutn.SamplerAttribute.CONFIG_NUM_HYPER_SAMPLES)
num_hyper_samples = np.asarray(8, dtype=num_hyper_samples_dtype)
cutn.sampler_configure(handle, sampler, 
    cutn.SamplerAttribute.CONFIG_NUM_HYPER_SAMPLES, 
    num_hyper_samples.ctypes.data, num_hyper_samples.dtype.itemsize)

# Prepare the quantum circuit sampler
cutn.sampler_prepare(handle, sampler, scratch_size, work_desc, stream.ptr)
print("Prepared the specified quantum circuit state sampler")

workspace_size_d = cutn.workspace_get_memory_size(handle, 
    work_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)

flops_dtype = cutn.sampler_get_attribute_dtype(cutn.SamplerAttribute.INFO_FLOPS)
flops = np.zeros(1, dtype=flops_dtype)
cutn.sampler_get_info(handle, sampler, cutn.SamplerAttribute.INFO_FLOPS, flops.ctypes.data, flops.dtype.itemsize)
print(f"Total flop count for sampling = {flops.item()/1e9} GFlop")

if workspace_size_d <= scratch_size:
    cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, scratch_space.ptr, workspace_size_d)
else:
    print("Error:Insufficient workspace size on Device")
    cutn.destroy_workspace_descriptor(work_desc)
    cutn.destroy_sampler(sampler)
    cutn.destroy_state(quantum_state)
    cutn.destroy_network_operator(tn_operator1)
    cutn.destroy_network_operator(tn_operator2)
    cutn.destroy(handle)
    del scratch_space
    print("Free resource and exit.")
    exit()
print("Set the workspace buffer for sampling")

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
cutn.destroy_network_operator(tn_operator1)
cutn.destroy_network_operator(tn_operator2)
cutn.destroy(handle)
del scratch_space
print("Free resource and exit.")
