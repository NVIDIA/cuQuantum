# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import cupy as cp
import numpy as np
from mpi4py import MPI

import cuquantum
from cuquantum.bindings import cutensornet as cutn

ATOL = 1e-8
RTOL = 1e-5
SEED = 1234

root = 0
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
if rank == root:
    print("*** Printing is done only from the root process to prevent jumbled messages ***")
    print(f"The number of processes is {size}")

num_devices = cp.cuda.runtime.getDeviceCount()
device_id = rank % num_devices
dev = cp.cuda.Device(device_id)
dev.use()

props = cp.cuda.runtime.getDeviceProperties(dev.id)
if rank == root:
    print("cuTensorNet-vers:", cutn.get_version())
    print("===== rank 0 device info ======")
    print("GPU-local-id:", dev.id)
    print("GPU-name:", props["name"].decode())
    print("GPU-clock:", props["clockRate"])
    print("GPU-memoryClock:", props["memoryClockRate"])
    print("GPU-nSM:", props["multiProcessorCount"])
    print("GPU-major:", props["major"])
    print("GPU-minor:", props["minor"])
    print("CUDA-available-devices:",num_devices)
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES')
    print("CUDA_VISIBLE_DEVICES:",CUDA_VISIBLE_DEVICES if CUDA_VISIBLE_DEVICES != None else '')
    print("===============================")
else:
    print("===== rank ", rank, " device info ======\nGPU-local-id:", dev.id)

######################################################################################
# Computing: R_{k,l} = A_{a,b,c,d,e,f} B_{b,g,h,e,i,j} C_{m,a,g,f,i,k} D_{l,c,h,d,j,m}
######################################################################################

if rank == root:
    print("Include headers and define data types.")

data_type = cuquantum.cudaDataType.CUDA_R_64F
compute_type = cuquantum.ComputeType.COMPUTE_64F
num_inputs = 4

# Create an array of modes
modes_A = [ord(c) for c in ('a','b','c','d','e','f')]
modes_B = [ord(c) for c in ('b','g','h','e','i','j')]
modes_C = [ord(c) for c in ('m','a','g','f','i','k')]
modes_D = [ord(c) for c in ('l','c','h','d','j','m')]
modes_R = [ord(c) for c in ('k','l')]
tensor_modes = [modes_A, modes_B, modes_C, modes_D, modes_R]
tensor_nmodes = [len(modes) for modes in tensor_modes]

# Create an array of extents (shapes) for each tensor
dim = 8
extent_A = (dim,) * 6
extent_B = (dim,) * 6
extent_C = (dim,) * 6
extent_D = (dim,) * 6
extent_R = (dim,) * 2
tensor_extents = [extent_A, extent_B, extent_C, extent_D, extent_R]
if rank == root:
    print("Define network, modes, and extents.")

#################
# Initialize data
#################

if rank == root:
    np.random.seed(SEED)
    A = np.random.random(np.prod(extent_A)).astype(np.float64)
    B = np.random.random(np.prod(extent_B)).astype(np.float64)
    C = np.random.random(np.prod(extent_C)).astype(np.float64)
    D = np.random.random(np.prod(extent_D)).astype(np.float64)
else:
    A = np.empty(np.prod(extent_A)).astype(np.float64)
    B = np.empty(np.prod(extent_B)).astype(np.float64)
    C = np.empty(np.prod(extent_C)).astype(np.float64)
    D = np.empty(np.prod(extent_D)).astype(np.float64)
R = np.empty(extent_R).astype(np.float64)

comm.Bcast(A, root)
comm.Bcast(B, root)
comm.Bcast(C, root)
comm.Bcast(D, root)

A_d = cp.asarray(A).reshape(extent_A, order='F')
B_d = cp.asarray(B).reshape(extent_B, order='F')
C_d = cp.asarray(C).reshape(extent_C, order='F')
D_d = cp.asarray(D).reshape(extent_D, order='F')
R_d = cp.empty(np.prod(extent_R), dtype=np.float64).reshape(extent_R, order='F')
tensor_data_d = [A_d, B_d, C_d, D_d, R_d]

#############
# cuTensorNet
#############

stream = cp.cuda.Stream()
handle = cutn.create()

################
# Create Network
################

# Create network
net = cutn.create_network(handle)

# Append input tensors
tensor_ids = [0] * num_inputs
for t in range(num_inputs):
    tensor_ids[t] = cutn.network_append_tensor(handle,
                                               net,
                                               tensor_nmodes[t],
                                               tensor_extents[t],
                                               tensor_modes[t],
                                               0,  # qualifiers (NULL)
                                               data_type)

# Set output tensor
cutn.network_set_output_tensor(handle,
                               net,
                               tensor_nmodes[num_inputs],
                               tensor_modes[num_inputs],
                               data_type)

# Set network compute type
cutn.network_set_attribute(handle,
                          net,
                          cutn.NetworkAttribute.COMPUTE_TYPE,
                          np.array([compute_type], dtype=np.int32).ctypes.data,
                          4)

if rank == root:
    print("Initialize the cuTensorNet library and create a network descriptor.")

#####################################################
# Choose workspace limit based on available resources
#####################################################

free_mem, total_mem = dev.mem_info
free_mem = comm.allreduce(free_mem, MPI.MIN)
workspace_limit = int(free_mem * 0.9)

##############################################
# Find "optimal" contraction order and slicing
##############################################

optimizer_config = cutn.create_contraction_optimizer_config(handle)
optimizer_info = cutn.create_contraction_optimizer_info(handle, net)

# Force slicing
min_slices_dtype = cutn.contraction_optimizer_config_get_attribute_dtype(
    cutn.ContractionOptimizerConfigAttribute.SLICER_MIN_SLICES)
min_slices_factor = np.asarray((size,), dtype=min_slices_dtype)
cutn.contraction_optimizer_config_set_attribute(
    handle, optimizer_config, cutn.ContractionOptimizerConfigAttribute.SLICER_MIN_SLICES,
    min_slices_factor.ctypes.data, min_slices_factor.dtype.itemsize)

cutn.contraction_optimize(
    handle, net, optimizer_config, workspace_limit, optimizer_info)

flops_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT)
flops = np.zeros((1,), dtype=flops_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizer_info, cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT,
    flops.ctypes.data, flops.dtype.itemsize)
flops = flops.item()

# Choose the path with the lowest cost.
flops, sender = comm.allreduce(sendobj=(flops, rank), op=MPI.MINLOC)
if rank == root:
    print(f"Process {sender} has the path with the lowest FLOP count {flops}.")

# Get buffer size for optimizer_info and broadcast it.
if rank == sender:
    bufSize = cutn.contraction_optimizer_info_get_packed_size(handle, optimizer_info)
else:
    bufSize = 0  # placeholder
bufSize = comm.bcast(bufSize, sender)

# Allocate buffer.
buf = np.empty((bufSize,), dtype=np.int8)

# Pack optimizer_info on sender and broadcast it.
if rank == sender:
    cutn.contraction_optimizer_info_pack_data(handle, optimizer_info, buf, bufSize)
comm.Bcast(buf, sender)

# Unpack optimizer_info from buffer.
if rank != sender:
    cutn.update_contraction_optimizer_info_from_packed_data(
        handle, buf, bufSize, optimizer_info)

num_slices_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.NUM_SLICES)
num_slices = np.zeros((1,), dtype=num_slices_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizer_info, cutn.ContractionOptimizerInfoAttribute.NUM_SLICES,
    num_slices.ctypes.data, num_slices.dtype.itemsize)
num_slices = num_slices.item()

assert num_slices > 0

# Calculate each process's share of the slices.
proc_chunk = num_slices / size
extra = num_slices % size
proc_slice_begin = rank * proc_chunk + min(rank, extra)
proc_slice_end = num_slices if rank == size - 1 else (rank + 1) * proc_chunk + min(rank + 1, extra)

# Update the network with the modified optimizer info
cutn.network_set_optimizer_info(handle, net, optimizer_info)

# Get the path to be used for reference contraction
path_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(cutn.ContractionOptimizerInfoAttribute.PATH)
_path = np.zeros(1, dtype=path_dtype)
path = np.zeros((2*(num_inputs-1),), dtype=np.int32)
_path["data"] = path.ctypes.data
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizer_info, cutn.ContractionOptimizerInfoAttribute.PATH, _path.ctypes.data, _path.dtype.itemsize)
path = list(zip(*[iter(path)]*2))

if rank == root:
    print("Find an optimized contraction path with cuTensorNet optimizer.")
 
###########################################################
# Initialize all pair-wise contraction plans (for cuTENSOR)
###########################################################

work_desc = cutn.create_workspace_descriptor(handle)
cutn.workspace_compute_contraction_sizes(handle, net, optimizer_info, work_desc)
required_workspace_size = cutn.workspace_get_memory_size(
    handle, work_desc,
    cutn.WorksizePref.MIN,
    cutn.Memspace.DEVICE,
    cutn.WorkspaceKind.SCRATCH)
work = cp.cuda.alloc(required_workspace_size)
cutn.workspace_set_memory(
    handle, work_desc,
    cutn.Memspace.DEVICE,
    cutn.WorkspaceKind.SCRATCH,
    work.ptr, required_workspace_size)

if rank == root:
    print("Allocate workspace.")
    
#########################
# Prepare the contraction
#########################

cutn.network_prepare_contraction(handle, net, work_desc)

# Set tensor's data buffers and strides
for t in range(num_inputs):
    cutn.network_set_input_tensor_memory(handle,
                                         net,
                                         tensor_ids[t],
                                         tensor_data_d[t].data.ptr,
                                         0)  # strides (NULL)

cutn.network_set_output_tensor_memory(handle,
                                      net,
                                      tensor_data_d[num_inputs].data.ptr,
                                      0)  # strides (NULL)

###################################################################################
# Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
###################################################################################

autotune_pref = cutn.create_network_autotune_preference(handle)

num_autotuning_iterations = 5  # may be 0
iterations_dtype = cutn.get_network_autotune_preference_attribute_dtype(
    cutn.NetworkAutotunePreferenceAttribute.NETWORK_AUTOTUNE_MAX_ITERATIONS)
num_autotuning_iterations_array = np.asarray([num_autotuning_iterations], dtype=iterations_dtype)
cutn.network_autotune_preference_set_attribute(handle,
                                              autotune_pref,
                                              cutn.NetworkAutotunePreferenceAttribute.NETWORK_AUTOTUNE_MAX_ITERATIONS,
                                              num_autotuning_iterations_array.ctypes.data,
                                              num_autotuning_iterations_array.dtype.itemsize)

# modify the plan again to find the best pair-wise contractions
cutn.network_autotune_contraction(handle, net, work_desc, autotune_pref, stream.ptr)

cutn.destroy_network_autotune_preference(autotune_pref)

if rank == root: 
    print("Prepared the network contraction for cuTensorNet and optionally auto-tuned it.")
 
###########
# Execution
###########

min_time_cutensornet = 1e100
num_runs = 3  # to get stable perf results
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()
slice_group = cutn.create_slice_group_from_id_range(handle, proc_slice_begin, proc_slice_end, 1)

for i in range(num_runs):
    # Contract over all slices.
    # A user may choose to parallelize over the slices across multiple devices.
    e1.record()
    cutn.network_contract(
        handle, net, False, work_desc, slice_group, stream.ptr)
    e2.record()

    # Synchronize and measure timing
    e2.synchronize()
    time = cp.cuda.get_elapsed_time(e1, e2) / 1000  # ms -> s
    min_time_cutensornet = min_time_cutensornet if min_time_cutensornet < time else time

if rank == root:
    print("Contract the network, each slice uses the same contraction plan.")

# free up the workspace
del work

R[...] = cp.asnumpy(R_d).reshape(extent_R, order='F')
# Reduce on root process.
if rank == root:
    comm.Reduce(MPI.IN_PLACE, R, root=root)
else:
    comm.Reduce(R, R, root=root)

# Compute the reference result.
if rank == root:   
    optimize = ['einsum_path', *path]
    out = cp.einsum("abcdef,bgheij,magfik,lchdjm->kl", A_d, B_d, C_d, D_d, optimize=optimize)
    try:
        cp.testing.assert_allclose(out, R, atol=ATOL, rtol=RTOL)
    except AssertionError as e:
        raise RuntimeError("result is incorrect") from e
    print("Check cuTensorNet result against that of cupy.einsum().")

######################################################

flops_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT)
flops = np.zeros((1,), dtype=flops_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizer_info, cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT,
    flops.ctypes.data, flops.dtype.itemsize)
flops = flops.item()

if rank == root:
    print(f"num_slices: {num_slices}")
    print(f"{min_time_cutensornet * 1000 / num_slices} ms / slice")
    print(f"{flops / 1e9 / min_time_cutensornet} GFLOPS/s")


cutn.destroy_slice_group(slice_group)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_contraction_optimizer_info(optimizer_info)
cutn.destroy_contraction_optimizer_config(optimizer_config)
cutn.destroy_network(net)
cutn.destroy(handle)

if rank == root:
    print("Free resource and exit.")
