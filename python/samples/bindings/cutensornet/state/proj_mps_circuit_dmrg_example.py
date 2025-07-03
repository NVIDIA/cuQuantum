# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np

import cuquantum
from cuquantum.bindings import cutensornet as cutn

from typing import Tuple, List
from numbers import Number

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

#####################################
# Circuit DMRG
#####################################

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
stream.use()
data_type = cuquantum.cudaDataType.CUDA_C_64F

# Define quantum gate tensors in device memory
gate_h = 2**-0.5 * cp.asarray([[1,1], [1,-1]], dtype='complex128', order='F')
gate_h_strides = 0

cr_gates = []
for i in range(num_qubits):
    phi = 1j * np.pi / 2 ** i
    gate_cr = cp.asarray([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, np.exp(phi)]], dtype='complex128').reshape(2,2,2,2, order='F')
    cr_gates.append(gate_cr)

free_mem = dev.mem_info[0]
scratch_size = free_mem // 2
scratch_space = cp.cuda.alloc(scratch_size)
print(f"Allocated {scratch_size} bytes of scratch memory on GPU")

# Create the initial quantum state
quantum_state = cutn.create_state(handle, cutn.StatePurity.PURE, num_qubits, qubits_dims, data_type)
print("Created the initial quantum state")

# Construct the quantum circuit state with gate application
for i in range(num_qubits):
    cutn.state_apply_tensor_operator(
        handle, quantum_state, 1, (i, ), 
        gate_h.data.ptr, gate_h_strides, 1, 0, 1)
    for j in range(i+1, num_qubits):
        cutn.state_apply_tensor_operator(
            handle, quantum_state, 2, (j, i),  
            cr_gates[j-i].data.ptr, 0, 1, 0, 1)
print("Quantum gates applied")

# Prepare input arguments for the projection MPS to compute an MPS approximation to the quantum circuit


states = [quantum_state]
coefficients = np.asarray([1.0], dtype="complex128")

env_specs = [(i-1, i+1) for i in range(num_qubits)]
env_bounds = np.asarray(env_specs, dtype=cutn.mps_env_bounds_dtype)
initial_ortho_spec = np.asarray([(-1, num_qubits)], dtype=cutn.mps_env_bounds_dtype)

# Allocate device memory for the final MPS state
max_extent = 8
proj_mps_tensor_extents = []
proj_mps_tensor_strides = []
proj_mps_tensors = []
proj_mps_tensor_ptrs = []

def get_max_bond_extents(state_mode_extents: Tuple[int, ...], maxdim = None) -> List[int]:
    """
    Helper function to get the maximum possible bond extents given state mode extents.
    
    Args:
        state_mode_extents: Tuple of integers representing the dimensions of each mode
        maxdim: Optional maximum dimension to cap the bond extents
        
    Returns:
        List of maximum bond extents between each pair of modes
    """
    dims = np.array(state_mode_extents)
    mins = np.min([np.cumprod(dims),np.cumprod(state_mode_extents[::-1])[::-1]], axis=0)
    max_bond_extents = np.min([mins[1:], mins[:-1]] , axis = 0)

    if maxdim is not None:
        return [min(int(max_bond_extent), maxdim) for max_bond_extent in max_bond_extents ]
    else:
        return list(int(max_bond_extent) for max_bond_extent in max_bond_extents)

max_bond_extents = get_max_bond_extents(qubits_dims, max_extent)

# allocate the MPS site tensor buffers and initialize them to the vacuum state as initial guess for the optimization
for i in range(num_qubits):
    if i == 0:
        extents = (2, min(max_bond_extents[i], max_extent))
    elif i == num_qubits - 1:
        extents = (min(max_bond_extents[i-1], max_extent), 2)
    else:
        extents = (min(max_bond_extents[i-1], max_extent), 2, min(max_bond_extents[i], max_extent))
    proj_mps_tensor_extents.append(extents)
    tensor = cp.zeros(extents, dtype='complex128')
    tensor.reshape(-1)[0] = 1.0
    proj_mps_tensors.append(tensor)
    proj_mps_tensor_ptrs.append(tensor.data.ptr)
    proj_mps_tensor_strides.append([stride_in_bytes // tensor.itemsize for stride_in_bytes in tensor.strides])

def compute_overlap(mps_tensors_1: List[cp.ndarray], mps_tensors_2: List[cp.ndarray]) -> Number:
    """
    Compute the overlap between two sets of MPS tensors.
    
    Args:
        mps_tensors_1: First set of MPS tensors
        mps_tensors_2: Second set of MPS tensors
        
    Returns:
        Complex number representing the overlap between the two MPS states
    """
    assert len(mps_tensors_1) == len(mps_tensors_2)
    
    tensor_1 = mps_tensors_1[0]
    tensor_2 = mps_tensors_2[0]
    overlap = cp.einsum('xi,xj->ij',tensor_2.conj(), tensor_1, )
    for i in range(1, len(mps_tensors_1)-1):
        tensor_1 = mps_tensors_1[i]
        tensor_2 = mps_tensors_2[i]
        overlap = cp.einsum('axi,ab,bxj->ij', tensor_2.conj(), overlap, tensor_1)
    tensor_1 = mps_tensors_1[-1]
    tensor_2 = mps_tensors_2[-1]
    overlap = cp.einsum('ax,ab,bx->', tensor_2.conj(), overlap, tensor_1)
    return overlap[()]

def compute_fidelity(mps_tensors_1: List[cp.ndarray], mps_tensors_2: List[cp.ndarray]) -> Number:
    """
    Compute the fidelity between two sets of MPS tensors.
    
    Args:
        mps_tensors_1: First set of MPS tensors
        mps_tensors_2: Second set of MPS tensors
        
    Returns:
        Real number between 0 and 1 representing the fidelity F = |<mps_tensors_2 | mps_tensors_1>| between the two MPS states
    """
    assert len(mps_tensors_1) == len(mps_tensors_2)
    overlap = compute_overlap(mps_tensors_1, mps_tensors_2)
    norm1 = compute_overlap(mps_tensors_1, mps_tensors_1)
    norm2 = compute_overlap(mps_tensors_2, mps_tensors_2)
    return cp.abs(overlap / cp.sqrt(norm1 * norm2))

# Create projection MPS state
projection_mps = cutn.create_state_projection_mps(
    handle, 
    len(states),
    states, 
    0,
    False,
    len(env_bounds),
    env_bounds.ctypes.data,
    cutn.BoundaryCondition.OPEN,
    num_qubits,
    0,
    proj_mps_tensor_extents,
    proj_mps_tensor_strides,
    proj_mps_tensor_ptrs,
    initial_ortho_spec.ctypes.data)

# Prepare the projection MPS state
work_desc = cutn.create_workspace_descriptor(handle)

cutn.state_projection_mps_prepare(handle, projection_mps, scratch_size, work_desc, stream.ptr)

workspace_size_d = cutn.workspace_get_memory_size(handle, 
    work_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)
print(f"workspace_size_d = {workspace_size_d}")
print(f"scratch_size = {scratch_size}")
if workspace_size_d <= scratch_size:
    cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, scratch_space.ptr, workspace_size_d)
else:
    print("Error:Insufficient workspace size on Device")
    cutn.destroy_workspace_descriptor(work_desc)
    cutn.destroy_state_projection_mps(projection_mps)
    cutn.destroy_state(quantum_state)
    cutn.destroy(handle)
    del scratch_space
    print("Free resource and exit.")
    exit()
print("Set the workspace buffer for Projection MPS computation")

num_iterations = 5

# Circuit DMRG iterations
for iter_idx in range(num_iterations):    
    # copy previous iterations site tensors to determine convergence
    previous_mps_tensors = [tensor.copy() for tensor in proj_mps_tensors]
    
    # Right sweep over site tensors
    for i in np.arange(num_qubits):
        # Extract the site tensor
        site_tensor = cp.zeros(proj_mps_tensors[i].shape, dtype='complex128')    
        env_bounds = np.asarray([(i-1, i+1)], dtype=cutn.mps_env_bounds_dtype)
        site_tensor_strides = [stride_in_bytes // site_tensor.itemsize for stride_in_bytes in site_tensor.strides]
        
        cutn.state_projection_mps_extract_tensor(handle,
            projection_mps,
            env_bounds.ctypes.data,
            site_tensor_strides,
            site_tensor.data.ptr,
            work_desc,
            stream.ptr
        )

        # Calculate the environment tensor
        env_tensor = cp.zeros(proj_mps_tensors[i].shape, dtype='complex128')
        env_tensor_strides = [stride_in_bytes // env_tensor.itemsize for stride_in_bytes in env_tensor.strides]
        cutn.state_projection_mps_compute_tensor_env(handle, projection_mps, env_bounds.ctypes.data, 0, 0, env_tensor_strides, env_tensor.data.ptr, 0, 0, work_desc, stream.ptr)
        
        # Replace the current site tensor with the environment tensor
        ortho_spec = np.asarray([(i-1, i+1)], dtype=cutn.mps_env_bounds_dtype)
        cutn.state_projection_mps_insert_tensor(handle, projection_mps, env_bounds.ctypes.data, ortho_spec.ctypes.data, env_tensor_strides, env_tensor.data.ptr, work_desc, stream.ptr)

    # Left sweep over site tensors
    for i in np.arange(num_qubits)[::-1]:
        # Extract the site tensor
        site_tensor = cp.zeros(proj_mps_tensors[i].shape, dtype='complex128')    
        env_bounds = np.asarray([(i-1, i+1)], dtype=cutn.mps_env_bounds_dtype)
        site_tensor_strides = [stride_in_bytes // site_tensor.itemsize for stride_in_bytes in site_tensor.strides]
        
        cutn.state_projection_mps_extract_tensor(handle,
            projection_mps,
            env_bounds.ctypes.data,
            site_tensor_strides,
            site_tensor.data.ptr,
            work_desc,
            stream.ptr
        )

        # Calculate the environment tensor
        env_tensor = cp.zeros(proj_mps_tensors[i].shape, dtype='complex128')
        env_tensor_strides = [stride_in_bytes // env_tensor.itemsize for stride_in_bytes in env_tensor.strides]
        cutn.state_projection_mps_compute_tensor_env(handle, projection_mps, env_bounds.ctypes.data, 0, 0, env_tensor_strides, env_tensor.data.ptr, 0, 0, work_desc, stream.ptr)
       
        # Replace the current site tensor with the environment tensor
        ortho_spec = np.asarray([(i-1, i+1)], dtype=cutn.mps_env_bounds_dtype)
        cutn.state_projection_mps_insert_tensor(handle, projection_mps, env_bounds.ctypes.data, ortho_spec.ctypes.data, env_tensor_strides, env_tensor.data.ptr, work_desc, stream.ptr)

    # Compute the estimated fidelity as convergence criterion
    fidelity = compute_fidelity(previous_mps_tensors, proj_mps_tensors)
    print(f"Estimated fidelity: {fidelity} at iteration {iter_idx + 1}")

stream.synchronize()
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_state_projection_mps(projection_mps)
cutn.destroy_state(quantum_state)
cutn.destroy(handle)
del scratch_space
print("Free resource and exit.")
