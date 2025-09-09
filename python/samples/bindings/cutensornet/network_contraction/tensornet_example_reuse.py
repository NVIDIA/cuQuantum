# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
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

#################################################################################################
# Computing: O_{a,m} = A_{a,b,c,d} B_{b,c,d,e} C_{e,f,g,h} D_{g,h,i,j} E_{i,j,k,l} F_{k,l,m}
# We will execute the contraction a few times assuming all input tensors being constant except F.
#################################################################################################

print("Include headers and define data types.")

data_type = cuquantum.cudaDataType.CUDA_R_64F
compute_type = cutn.ComputeType.COMPUTE_64F
num_inputs = 6

# Create vectors of tensor modes
modes_A = [ord(c) for c in ('a','b','c','d')]
modes_B = [ord(c) for c in ('b','c','d','e')]
modes_C = [ord(c) for c in ('e','f','g','h')]
modes_D = [ord(c) for c in ('g','h','i','j')]
modes_E = [ord(c) for c in ('i','j','k','l')]
modes_F = [ord(c) for c in ('k','l','m')]
modes_O = [ord(c) for c in ('a','m')]
tensor_modes = [modes_A, modes_B, modes_C, modes_D, modes_E, modes_F, modes_O]
tensor_nmodes = [len(modes) for modes in tensor_modes]

# Set mode extents
dim = 8  
extent_A = (dim,) * 4 
extent_B = (dim,) * 4 
extent_C = (dim,) * 4 
extent_D = (dim,) * 4 
extent_E = (dim,) * 4 
extent_F = (dim,) * 3 
extent_O = (dim,) * 2
tensor_extents = [extent_A, extent_B, extent_C, extent_D, extent_E, extent_F, extent_O]

print("Define network, modes, and extents.")

#################
# Initialize data
#################

cp.random.seed(SEED)
A_d = cp.random.random((np.prod(extent_A),), dtype=np.float64).reshape(extent_A, order='F')
B_d = cp.random.random((np.prod(extent_B),), dtype=np.float64).reshape(extent_B, order='F')
C_d = cp.random.random((np.prod(extent_C),), dtype=np.float64).reshape(extent_C, order='F')
D_d = cp.random.random((np.prod(extent_D),), dtype=np.float64).reshape(extent_D, order='F')
E_d = cp.random.random((np.prod(extent_E),), dtype=np.float64).reshape(extent_E, order='F')
F_d = cp.random.random((np.prod(extent_F),), dtype=np.float64).reshape(extent_F, order='F')
O_d = cp.zeros((np.prod(extent_O),), dtype=np.float64).reshape(extent_O, order='F')
tensor_data_d = [A_d, B_d, C_d, D_d, E_d, F_d, O_d]

#############
# cuTensorNet
#############

stream = cp.cuda.Stream()
handle = cutn.create()

print("Allocated GPU memory for data, initialized data, and created library handle")

############################
# Set constant input tensors
############################

# specify which input tensors are constant - all but the last one (F)
qualifiers_in = np.zeros(num_inputs, dtype=cutn.tensor_qualifiers_dtype)
for i in range(num_inputs-1):
    qualifiers_in[i]['is_constant'] = True
qualifiers_in[num_inputs-1]['is_constant'] = False

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
                                         qualifiers_in[t:t+1].ctypes.data,  # qualifiers for constant tensors
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

print("Initialized the cuTensorNet library and created a tensor network")

######################################################
# Choose workspace limit based on available resources.
######################################################

free_mem, total_mem = dev.mem_info
workspace_limit = int(free_mem * 0.9)
print(f"Workspace limit = {workspace_limit}")

#######################
# Set contraction order
#######################

# Create contraction optimizer info
optimizer_info = cutn.create_contraction_optimizer_info(handle, net)

# set a predetermined contraction path
path_dtype = cutn.get_contraction_optimizer_info_attribute_dtype(cutn.ContractionOptimizerInfoAttribute.PATH)
path = np.asarray([(0, 1), (0, 4), (0, 3), (0, 2), (0, 1)], dtype=np.int32)
path_obj = np.zeros((1,), dtype=path_dtype)
path_obj["num_contractions"] = num_inputs - 1
path_obj["data"] = path.ctypes.data

# provide user-specified contPath
cutn.contraction_optimizer_info_set_attribute(handle,
                                             optimizer_info,
                                             cutn.ContractionOptimizerInfoAttribute.PATH,
                                             path_obj.ctypes.data,
                                             path_obj.dtype.itemsize)

num_slices_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.NUM_SLICES)
num_slices = np.zeros((1,), dtype=num_slices_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizer_info, cutn.ContractionOptimizerInfoAttribute.NUM_SLICES,
    num_slices.ctypes.data, num_slices.dtype.itemsize)
num_slices = num_slices.item()

assert num_slices > 0

print("Set predetermined contraction path into cuTensorNet optimizer")

##############################################################
# Create workspace descriptor, allocate workspace, and set it.
##############################################################

work_desc = cutn.create_workspace_descriptor(handle)

# set SCRATCH workspace, which will be used during each network contraction operation, not needed afterwards
cutn.workspace_compute_contraction_sizes(handle, net, optimizer_info, work_desc)

required_workspace_size_scratch = cutn.workspace_get_memory_size(handle,
                                                               work_desc,
                                                               cutn.WorksizePref.MIN,
                                                               cutn.Memspace.DEVICE,
                                                               cutn.WorkspaceKind.SCRATCH)

work_scratch = cp.cuda.alloc(required_workspace_size_scratch)

cutn.workspace_set_memory(handle,
                         work_desc,
                         cutn.Memspace.DEVICE,
                         cutn.WorkspaceKind.SCRATCH,
                         work_scratch.ptr,
                         required_workspace_size_scratch)

# set CACHE workspace, which will be used across network contraction operations
required_workspace_size_cache = cutn.workspace_get_memory_size(handle,
                                                             work_desc,
                                                             cutn.WorksizePref.MIN,
                                                             cutn.Memspace.DEVICE,
                                                             cutn.WorkspaceKind.CACHE)

work_cache = cp.cuda.alloc(required_workspace_size_cache)

cutn.workspace_set_memory(handle,
                         work_desc,
                         cutn.Memspace.DEVICE,
                         cutn.WorkspaceKind.CACHE,
                         work_cache.ptr,
                         required_workspace_size_cache)

print("Allocated and set up the GPU workspace")

##########################
# Prepare the contraction.
##########################

cutn.network_set_optimizer_info(handle,
                               net,
                               optimizer_info)

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
first_time_cutensornet = 1e100
num_runs = 3  # number of repeats to get stable performance results
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()

for i in range(num_runs):
    # restore the output tensor on GPU
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
    if i == 0:
        first_time_cutensornet = time
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
# Compute the reference using cupy.einsum with the same path
path = ['einsum_path'] + path.tolist()
out = cp.einsum("abcd,bcde,efgh,ghij,ijkl,klm->am", A_d, B_d, C_d, D_d, E_d, F_d, optimize=path)
try:
    cp.testing.assert_allclose(out, O_d, atol=ATOL, rtol=RTOL)
except AssertionError as e:
    raise RuntimeError("result is incorrect") from e
print("Check cuTensorNet result against that of cupy.einsum().")

print(f"num_slices: {num_slices}")
print(f"First run (intermediate tensors get cached): {first_time_cutensornet * 1000 / num_slices} ms / slice")
print(f"Subsequent run (cache reused): {min_time_cutensornet * 1000 / num_slices} ms / slice")
print(f"{flops / 1e9 / min_time_cutensornet} GFLOPS/s")

# Free cuTensorNet resources
cutn.destroy_slice_group(slice_group)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_contraction_optimizer_info(optimizer_info)
cutn.destroy_network(net)
cutn.destroy(handle)

# Free GPU memory resources
del work_scratch
del work_cache

print("Free resources and exit.") 