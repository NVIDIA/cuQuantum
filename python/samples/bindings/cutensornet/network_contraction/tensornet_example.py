# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np

import cuquantum
from cuquantum.bindings import cutensornet as cutn

ATOL = 1e-8
RTOL = 1e-5
SEED = 1234

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

######################################################################################
# Computing: R_{k,l} = A_{a,b,c,d,e,f} B_{b,g,h,e,i,j} C_{m,a,g,f,i,k} D_{l,c,h,d,j,m}
######################################################################################

print("Include headers and define data types.")

data_type = cuquantum.cudaDataType.CUDA_R_64F
compute_type = cutn.ComputeType.COMPUTE_64F
num_inputs = 4

# Create vectors of tensor modes
modes_A = [ord(c) for c in ('a','b','c','d','e','f')]
modes_B = [ord(c) for c in ('b','g','h','e','i','j')]
modes_C = [ord(c) for c in ('m','a','g','f','i','k')]
modes_D = [ord(c) for c in ('l','c','h','d','j','m')]
modes_R = [ord(c) for c in ('k','l')]
tensor_modes = [modes_A, modes_B, modes_C, modes_D, modes_R]

tensor_nmodes = [len(modes) for modes in tensor_modes]

# Create a vector of extents for each tensor
dim = 8
extent_A = (dim,) * 6 
extent_B = (dim,) * 6 
extent_C = (dim,) * 6 
extent_D = (dim,) * 6 
extent_R = (dim,) * 2
tensor_extents = [extent_A, extent_B, extent_C, extent_D, extent_R]

print("Define network, modes, and extents.")

#################
# Initialize data
#################

cp.random.seed(SEED)
A_d = cp.random.random((np.prod(extent_A),), dtype=np.float64).reshape(extent_A, order='F')
B_d = cp.random.random((np.prod(extent_B),), dtype=np.float64).reshape(extent_B, order='F')
C_d = cp.random.random((np.prod(extent_C),), dtype=np.float64).reshape(extent_C, order='F')
D_d = cp.random.random((np.prod(extent_D),), dtype=np.float64).reshape(extent_D, order='F')
R_d = cp.zeros((np.prod(extent_R),), dtype=np.float64).reshape(extent_R, order='F')
tensor_data_d = [A_d, B_d, C_d, D_d, R_d]  

#############
# cuTensorNet
#############

stream = cp.cuda.Stream()
handle = cutn.create()

print("Allocated GPU memory for data, initialized data, and created library handle")

################
# Create Network
################

# Set up tensor network
net = cutn.create_network(handle)

tensor_ids = []  # for input tensors
# attach the input tensors to the network
for t in range(num_inputs):
    tensor_id = cutn.network_append_tensor(handle,
                                         net,
                                         tensor_nmodes[t],
                                         tensor_extents[t],
                                         tensor_modes[t],
                                         0,  # qualifiers (NULL)
                                         data_type)
    tensor_ids.append(tensor_id)

# set the output tensor
cutn.network_set_output_tensor(handle,
                              net,
                              tensor_nmodes[num_inputs],
                              tensor_modes[num_inputs],
                              data_type)

# set the network compute type
compute_type_dtype = cutn.get_network_attribute_dtype(cutn.NetworkAttribute.COMPUTE_TYPE)
compute_type_array = np.asarray([compute_type], dtype=compute_type_dtype)
cutn.network_set_attribute(handle,
                          net,
                          cutn.NetworkAttribute.COMPUTE_TYPE,
                          compute_type_array.ctypes.data,
                          compute_type_array.dtype.itemsize)

print("Initialized the cuTensorNet library, created a tensor network descriptor, and appended input tensors.")

######################################################
# Choose workspace limit based on available resources.
######################################################

free_mem, total_mem = dev.mem_info
workspace_limit = int(free_mem * 0.9)
print(f"Workspace limit = {workspace_limit}")

##############################################
# Find "optimal" contraction order and slicing
##############################################

optimizer_config = cutn.create_contraction_optimizer_config(handle)

# Set the desired number of hyper-samples (defaults to 0)
num_hypersamples = 8
hypersamples_dtype = cutn.get_contraction_optimizer_config_attribute_dtype(
    cutn.ContractionOptimizerConfigAttribute.HYPER_NUM_SAMPLES)
num_hypersamples_array = np.asarray([num_hypersamples], dtype=hypersamples_dtype)
cutn.contraction_optimizer_config_set_attribute(handle,
                                               optimizer_config,
                                               cutn.ContractionOptimizerConfigAttribute.HYPER_NUM_SAMPLES,
                                               num_hypersamples_array.ctypes.data,
                                               num_hypersamples_array.dtype.itemsize)

# Create contraction optimizer info and find an optimized contraction path
optimizer_info = cutn.create_contraction_optimizer_info(handle, net)

cutn.contraction_optimize(handle,
                         net,
                         optimizer_config,
                         workspace_limit,
                         optimizer_info)

# Query the number of slices the tensor network execution will be split into
num_slices_dtype = cutn.get_contraction_optimizer_info_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.NUM_SLICES)
num_slices = np.zeros((1,), dtype=num_slices_dtype)
cutn.contraction_optimizer_info_get_attribute(handle,
                                             optimizer_info,
                                             cutn.ContractionOptimizerInfoAttribute.NUM_SLICES,
                                             num_slices.ctypes.data,
                                             num_slices.dtype.itemsize)
num_slices = num_slices.item()

assert num_slices > 0

print("Found an optimized contraction path using cuTensorNet optimizer")

##############################################################
# Create workspace descriptor, allocate workspace, and set it.
##############################################################

work_desc = cutn.create_workspace_descriptor(handle)

cutn.workspace_compute_contraction_sizes(handle,
                                        net,
                                        optimizer_info,
                                        work_desc)

required_workspace_size = cutn.workspace_get_memory_size(handle,
                                                       work_desc,
                                                       cutn.WorksizePref.MIN,
                                                       cutn.Memspace.DEVICE,
                                                       cutn.WorkspaceKind.SCRATCH)

work = cp.cuda.alloc(required_workspace_size)

cutn.workspace_set_memory(handle,
                         work_desc,
                         cutn.Memspace.DEVICE,
                         cutn.WorkspaceKind.SCRATCH,
                         work.ptr,
                         required_workspace_size)

print("Allocated and set up the GPU workspace")

##########################
# Prepare the contraction.
##########################

cutn.network_prepare_contraction(handle,
                                net,
                                work_desc)

# set tensor's data buffers and strides
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

#####################################################################
# Optional: Auto-tune the contraction plan to pick the fastest kernel
#           for each pairwise tensor contraction.
#####################################################################
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

# Modify the network again to find the best pair-wise contractions
cutn.network_autotune_contraction(handle,
                                 net,
                                 work_desc,
                                 autotune_pref,
                                 stream.ptr)

cutn.destroy_network_autotune_preference(autotune_pref)

print("Prepared the network contraction for cuTensorNet and optionally auto-tuned it")

########################################
# Execute the tensor network contraction
########################################

# Create a cutensornetSliceGroup_t object from a range of slice IDs
slice_group = cutn.create_slice_group_from_id_range(handle, 0, num_slices, 1)

min_time_cutensornet = 1e100
num_runs = 3  # number of repeats to get stable performance results
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()

for i in range(num_runs):
    # reset the output tensor data
    tensor_data_d[num_inputs][:] = 0
    
    # Contract all slices of the tensor network
    e1.record()
    
    accumulate_output = 0  # output tensor data will be overwritten
    cutn.network_contract(handle,
                         net,
                         accumulate_output,
                         work_desc,
                         slice_group,  # alternatively, 0 can also be used to contract over all slices
                         stream.ptr)
    
    e2.record()
    
    # Synchronize and measure best timing
    e2.synchronize()
    time = cp.cuda.get_elapsed_time(e1, e2) / 1000  # ms -> s
    min_time_cutensornet = min(min_time_cutensornet, time)

print("Contracted the tensor network, each slice used the same contraction plan")

# Print the 1-norm of the output tensor (verification)
stream.synchronize()
norm1 = abs(tensor_data_d[num_inputs]).sum()
print(f"Computed the 1-norm of the output tensor: {norm1:e}")

# Query the total Flop count for the tensor network contraction
flops_dtype = cutn.get_contraction_optimizer_info_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT)
flops = np.zeros((1,), dtype=flops_dtype)
cutn.contraction_optimizer_info_get_attribute(handle,
                                             optimizer_info,
                                             cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT,
                                             flops.ctypes.data,
                                             flops.dtype.itemsize)
flops = flops.item()

# Verification against cupy.einsum 
path, _ = cuquantum.tensornet.einsum_path("abcdef,bgheij,magfik,lchdjm->kl", A_d, B_d, C_d, D_d)
out = cp.einsum("abcdef,bgheij,magfik,lchdjm->kl", A_d, B_d, C_d, D_d, optimize=path)
try:
    cp.testing.assert_allclose(out, R_d, atol=ATOL, rtol=RTOL)
except AssertionError as e:
    raise RuntimeError("result is incorrect") from e
print("Check cuTensorNet result against that of cupy.einsum().")

print(f"num_slices: {num_slices}")
print(f"{min_time_cutensornet * 1000 / num_slices} ms / slice")
print(f"{flops / 1e9 / min_time_cutensornet} GFLOPS/s")

# Free cuTensorNet resources
cutn.destroy_slice_group(slice_group)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_contraction_optimizer_info(optimizer_info)
cutn.destroy_contraction_optimizer_config(optimizer_config)
cutn.destroy_network(net)
cutn.destroy(handle)

# Free GPU memory resources
del work

print("Free resources and exit.")
