# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
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

############################################################################################
# Computing: O_{a,m} = A_{a,b,c,d} B_{b,c,d,e} C_{e,f,g,h} D_{g,h,i,j} E_{i,j,k,l} F_{k,l,m}
############################################################################################

print("Include headers and define data types.")

data_type = cuquantum.cudaDataType.CUDA_R_32F
compute_type = cuquantum.ComputeType.COMPUTE_32F
num_inputs = 6

# Create an array of modes
modes_A = [ord(c) for c in ('a','b','c','d')]
modes_B = [ord(c) for c in ('b','c','d','e')]
modes_C = [ord(c) for c in ('e','f','g','h')]
modes_D = [ord(c) for c in ('g','h','i','j')]
modes_E = [ord(c) for c in ('i','j','k','l')]
modes_F = [ord(c) for c in ('k','l','m')]
modes_O = [ord(c) for c in ('a','m')]

# Create an array of extents (shapes) for each tensor
dim = 8
extent_A = (dim,) * 4 
extent_B = (dim,) * 4 
extent_C = (dim,) * 4 
extent_D = (dim,) * 4 
extent_E = (dim,) * 4 
extent_F = (dim,) * 3 
extent_O = (dim,) * 2

print("Define network, modes, and extents.")

#################
# Initialize data
#################

A_d = cp.random.random((np.prod(extent_A),), dtype=np.float32)
B_d = cp.random.random((np.prod(extent_B),), dtype=np.float32)
C_d = cp.random.random((np.prod(extent_C),), dtype=np.float32)
D_d = cp.random.random((np.prod(extent_D),), dtype=np.float32)
E_d = cp.random.random((np.prod(extent_E),), dtype=np.float32)
F_d = cp.random.random((np.prod(extent_F),), dtype=np.float32)
O_d = cp.zeros((np.prod(extent_O),), dtype=np.float32)
raw_data_in_d = (A_d.data.ptr, B_d.data.ptr, C_d.data.ptr, D_d.data.ptr, E_d.data.ptr, F_d.data.ptr)

#############
# cuTensorNet
#############

stream = cp.cuda.Stream()
handle = cutn.create()

nmode_A = len(modes_A)
nmode_B = len(modes_B)
nmode_C = len(modes_C)
nmode_D = len(modes_D)
nmode_E = len(modes_E)
nmode_F = len(modes_F)
nmode_O = len(modes_O)

###############################
# Create Contraction Descriptor
###############################

modes_in = (modes_A, modes_B, modes_C, modes_D, modes_E, modes_F)
extents_in = (extent_A, extent_B, extent_C, extent_D, extent_E, extent_F)
num_modes_in = (nmode_A, nmode_B, nmode_C, nmode_D, nmode_E, nmode_F)

# Strides are optional; if no stride (0) is provided, then cuTensorNet assumes a generalized column-major data layout
strides_in = (0, 0, 0, 0, 0, 0)

# Set up the tensor qualifiers for all input tensors
qualifiers_in = np.zeros(num_inputs, dtype=cutn.tensor_qualifiers_dtype)
for i in range(5):
    qualifiers_in[i]['is_constant'] = True

# Set up tensor network
desc_net = cutn.create_network_descriptor(handle,
    num_inputs, num_modes_in, extents_in, strides_in, modes_in, qualifiers_in,  # inputs
    nmode_O, extent_O, 0, modes_O,  # output
    data_type, compute_type)

print("Initialize the cuTensorNet library and create a network descriptor.")

#####################################################
# Choose workspace limit based on available resources
#####################################################

free_mem, total_mem = dev.mem_info
workspace_limit = int(free_mem * 0.9)

##############################################
# Set contraction order and slicing
##############################################

optimizer_info = cutn.create_contraction_optimizer_info(handle, desc_net)

path_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(cutn.ContractionOptimizerInfoAttribute.PATH)
path = np.asarray([(0, 1), (0, 4), (0, 3), (0, 2), (0, 1)], dtype=np.int32)
path_obj = np.zeros((1,), dtype=path_dtype)
path_obj["num_contractions"] = num_inputs - 1
path_obj["data"] = path.ctypes.data

cutn.contraction_optimizer_info_set_attribute(
    handle, optimizer_info, cutn.ContractionOptimizerInfoAttribute.PATH, 
    path_obj.ctypes.data, path_obj.dtype.itemsize)

num_slices_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.NUM_SLICES)
num_slices = np.zeros((1,), dtype=num_slices_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizer_info, cutn.ContractionOptimizerInfoAttribute.NUM_SLICES,
    num_slices.ctypes.data, num_slices.dtype.itemsize)
num_slices = int(num_slices)

assert num_slices > 0

print("Set contraction path into cuTensorNet optimizer.")
 
###########################################################
# Initialize all pair-wise contraction plans (for cuTENSOR)
###########################################################

work_desc = cutn.create_workspace_descriptor(handle)
cutn.workspace_compute_contraction_sizes(handle, desc_net, optimizer_info, work_desc)
required_scratch_workspace_size = cutn.workspace_get_memory_size(
    handle, work_desc,
    cutn.WorksizePref.MIN,
    cutn.Memspace.DEVICE,
    cutn.WorkspaceKind.SCRATCH)
work_scratch = cp.cuda.alloc(required_scratch_workspace_size)
cutn.workspace_set_memory(
    handle, work_desc,
    cutn.Memspace.DEVICE,
    cutn.WorkspaceKind.SCRATCH,
    work_scratch.ptr, required_scratch_workspace_size)
required_cache_workspace_size = cutn.workspace_get_memory_size(
    handle, work_desc,
    cutn.WorksizePref.MIN,
    cutn.Memspace.DEVICE,
    cutn.WorkspaceKind.CACHE)
work_cache = cp.cuda.alloc(required_cache_workspace_size)
cutn.workspace_set_memory(
    handle, work_desc,
    cutn.Memspace.DEVICE,
    cutn.WorkspaceKind.CACHE,
    work_cache.ptr, required_cache_workspace_size)
plan = cutn.create_contraction_plan(handle, desc_net, optimizer_info, work_desc)

###################################################################################
# Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
###################################################################################

pref = cutn.create_contraction_autotune_preference(handle)

num_autotuning_iterations = 5 # may be 0
n_iter_dtype = cutn.contraction_autotune_preference_get_attribute_dtype(
    cutn.ContractionAutotunePreferenceAttribute.MAX_ITERATIONS)
num_autotuning_iterations = np.asarray([num_autotuning_iterations], dtype=n_iter_dtype)
cutn.contraction_autotune_preference_set_attribute(
    handle, pref,
    cutn.ContractionAutotunePreferenceAttribute.MAX_ITERATIONS,
    num_autotuning_iterations.ctypes.data, num_autotuning_iterations.dtype.itemsize)

# Modify the plan again to find the best pair-wise contractions
cutn.contraction_autotune(
    handle, plan, raw_data_in_d, O_d.data.ptr,
    work_desc, pref, stream.ptr)

cutn.destroy_contraction_autotune_preference(pref)
 
print("Create a contraction plan for cuTENSOR and optionally auto-tune it.")
 
###########
# Execution
###########

minTimeCUTENSORNET = 1e100
firstTimeCUTENSORNET = 1e100
num_runs = 3  # to get stable perf results
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()
slice_group = cutn.create_slice_group_from_id_range(handle, 0, num_slices, 1)

for i in range(num_runs):
    # Contract over all slices.
    # A user may choose to parallelize over the slices across multiple devices.
    e1.record()
    cutn.contract_slices(
        handle, plan, raw_data_in_d, O_d.data.ptr, False,
        work_desc, slice_group, stream.ptr)
    e2.record()

    # Synchronize and measure timing
    e2.synchronize()
    time = cp.cuda.get_elapsed_time(e1, e2) / 1000  # ms -> s
    minTimeCUTENSORNET = minTimeCUTENSORNET if minTimeCUTENSORNET < time else time
    firstTimeCUTENSORNET = firstTimeCUTENSORNET if i > 0 else time

print("Contract the network, each slice uses the same contraction plan.")

# free up the workspace
del work_scratch
del work_cache

# Recall that we set strides to null (0), so the data are in F-contiguous layout
A_d = A_d.reshape(extent_A, order='F')
B_d = B_d.reshape(extent_B, order='F')
C_d = C_d.reshape(extent_C, order='F')
D_d = D_d.reshape(extent_D, order='F')
E_d = E_d.reshape(extent_E, order='F')
F_d = F_d.reshape(extent_F, order='F')
O_d = O_d.reshape(extent_O, order='F')

# Compute the reference using cupy.einsum with the same path
path = ['einsum_path'] + path.tolist()
out = cp.einsum("abcd,bcde,efgh,ghij,ijkl,klm->am", A_d, B_d, C_d, D_d, E_d, F_d, optimize=path)
if not cp.allclose(out, O_d):
    raise RuntimeError("result is incorrect")
print("Check cuTensorNet result against that of cupy.einsum().")

#######################################################

flops_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT)
flops = np.zeros((1,), dtype=flops_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizer_info, cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT,
    flops.ctypes.data, flops.dtype.itemsize)
flops = float(flops)

print(f"num_slices: {num_slices}")
print(f"First run (intermediate tensors get cached): {firstTimeCUTENSORNET * 1000 / num_slices} ms / slice")
print(f"Subsequent run (cache reused): {minTimeCUTENSORNET * 1000 / num_slices} ms / slice")
print(f"{flops / 1e9 / minTimeCUTENSORNET} GFLOPS/s")

cutn.destroy_slice_group(slice_group)
cutn.destroy_contraction_plan(plan)
cutn.destroy_contraction_optimizer_info(optimizer_info)
cutn.destroy_network_descriptor(desc_net)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy(handle)

print("Free resource and exit.")
