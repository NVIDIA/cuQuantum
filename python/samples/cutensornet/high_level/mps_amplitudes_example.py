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

#################################################
# Accessor computation of a quantum circuit state
#################################################

# Quantum state configuration
num_qubits = 6
dim = 2
qubits_dims = (dim, ) * num_qubits # qubit size
fixed_modes = (0, 1) # open qubits
num_fixed_modes = len(fixed_modes)
fixed_values = (1, 1) 
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

gate_cx = cp.asarray([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], dtype='complex128').reshape(2,2,2,2, order='F')
gate_cx_strides = 0

# Allocate device memory for the final MPS state
max_extent = 2
mps_tensor_extents = []
mps_tensor_strides = []
mps_tensors = []
mps_tensor_ptrs = []
for i in range(num_qubits):
    if i == 0:
        extents = (2, max_extent)
    elif i == num_qubits - 1:
        extents = (max_extent, 2)
    else:
        extents = (max_extent, 2, max_extent)
    mps_tensor_extents.append(extents)
    tensor = cp.zeros(extents, dtype='complex128')
    mps_tensors.append(tensor)
    mps_tensor_ptrs.append(tensor.data.ptr)
    mps_tensor_strides.append([stride_in_bytes // tensor.itemsize for stride_in_bytes in tensor.strides])

# Allocate device memory for the specified slice of the quantum circuit amplitudes tensor
amplitudes_shape = [qubits_dims[i] for i in range(num_qubits) if i not in fixed_modes]
amplitudes = cp.empty(amplitudes_shape, dtype='complex128')
amplitudes_strides = [stride_in_bytes // amplitudes.itemsize for stride_in_bytes in amplitudes.strides]

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

# Specify the target MPS state
cutn.state_finalize_mps(handle, quantum_state, cutn.BoundaryCondition.OPEN, mps_tensor_extents, mps_tensor_strides)
print("Set the final MPS representation")

# Configure the MPS computation
svd_algorithm_dtype = cutn.state_get_attribute_dtype(cutn.StateAttribute.MPS_SVD_CONFIG_ALGO)
svd_algorithm = np.array(cutn.TensorSVDAlgo.GESVDJ, dtype=svd_algorithm_dtype)
cutn.state_configure(handle, quantum_state, 
    cutn.StateAttribute.MPS_SVD_CONFIG_ALGO, svd_algorithm.ctypes.data, svd_algorithm.dtype.itemsize)

# Prepare the specified quantum circuit for MPS computation
work_desc = cutn.create_workspace_descriptor(handle)
cutn.state_prepare(handle, quantum_state, scratch_size, work_desc, stream.ptr)
print("Prepared the specified quantum circuit for MPS computation")

flops_dtype = cutn.state_get_attribute_dtype(cutn.StateAttribute.INFO_FLOPS)
flops = np.zeros(1, dtype=flops_dtype)
cutn.state_get_info(handle, quantum_state, cutn.StateAttribute.INFO_FLOPS, flops.ctypes.data, flops.dtype.itemsize)
print(f"Total flop count for state computation = {flops.item()/1e9} GFlop")

workspace_size_d = cutn.workspace_get_memory_size(handle, 
    work_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)
if workspace_size_d <= scratch_size:
    cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, scratch_space.ptr, workspace_size_d)
else:
    print("Error:Insufficient workspace size on Device")
    cutn.destroy_workspace_descriptor(work_desc)
    cutn.destroy_state(quantum_state)
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

# Specify the quantum circuit amplitudes accessor
accessor = cutn.create_accessor(handle, 
    quantum_state, num_fixed_modes, fixed_modes, amplitudes_strides)

# Configure the accessor with hyper samples for the contraction optimizer
num_hyper_samples_dtype = cutn.accessor_get_attribute_dtype(cutn.AccessorAttribute.CONFIG_NUM_HYPER_SAMPLES)
num_hyper_samples = np.asarray(8, dtype=num_hyper_samples_dtype)
cutn.accessor_configure(handle, accessor, 
    cutn.AccessorAttribute.CONFIG_NUM_HYPER_SAMPLES, 
    num_hyper_samples.ctypes.data, num_hyper_samples.dtype.itemsize)

# Prepare the computation of the specified slice of the quantum circuit amplitudes tensor
cutn.accessor_prepare(handle, accessor, scratch_size, work_desc, stream.ptr)
print("Prepare the computation of the specified slice of the quantum circuit amplitudes tensor")

flops_dtype = cutn.accessor_get_attribute_dtype(cutn.AccessorAttribute.INFO_FLOPS)
flops = np.zeros(1, dtype=flops_dtype)
cutn.accessor_get_info(handle, accessor, cutn.AccessorAttribute.INFO_FLOPS, flops.ctypes.data, flops.dtype.itemsize)
print(f"Total flop count for amplitude computation = {flops.item()/1e9} GFlop")

workspace_size_d = cutn.workspace_get_memory_size(handle, 
    work_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)

if workspace_size_d <= scratch_size:
    cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, scratch_space.ptr, workspace_size_d)
else:
    print("Error:Insufficient workspace size on Device")
    cutn.destroy_workspace_descriptor(work_desc)
    cutn.destroy_accessor(accessor)
    cutn.destroy_state(quantum_state)
    cutn.destroy(handle)
    del scratch_space
    print("Free resource and exit.")
    exit()
print("Set the workspace buffer")

# Compute the specified slice of the quantum circuit amplitudes tensor
state_norm = np.empty(1, dtype='complex128')
cutn.accessor_compute(handle, accessor, 
    fixed_values, work_desc, amplitudes.data.ptr, state_norm.ctypes.data, stream.ptr)
stream.synchronize()
print("Computed the specified quantum circuit state amplitudes")

print(amplitudes)
print(f"norm of the state = {state_norm.item()}")

cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_accessor(accessor)
cutn.destroy_state(quantum_state)
cutn.destroy(handle)
del scratch_space
print("Free resource and exit.")