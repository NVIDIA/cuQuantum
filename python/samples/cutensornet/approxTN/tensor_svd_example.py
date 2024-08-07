# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
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

######################################################
# Tensor SVD: T_{i,j,m,n} -> U_{i,x,m} S_{x} V_{n,x,j}  
######################################################

data_type = cuquantum.cudaDataType.CUDA_R_32F

# Create an array of modes

modes_T = [ord(c) for c in ('i','j','m','n')] # input
modes_U = [ord(c) for c in ('i','x','m')] # SVD output
modes_V = [ord(c) for c in ('n','x','j')]

# Create an array of extent (shapes) for each tensor
extent_T = (16, 16, 16, 16)
shared_extent = 256 // 2 # truncate shared extent from 256 to 128
extent_U = (16, shared_extent, 16)  
extent_V = (16, shared_extent, 16)

############################
# Allocate & initialize data
############################
cp.random.seed(1)
T_d = cp.random.random(extent_T, dtype=np.float32).astype(np.float32, order='F') # we use fortran layout throughout this example
U_d = cp.empty(extent_U, dtype=np.float32, order='F')
S_d = cp.empty(shared_extent, dtype=np.float32)
V_d = cp.empty(extent_V, dtype=np.float32, order='F')

print("Allocate memory for data and initialize data.")

free_mem, total_mem = dev.mem_info
worksize = free_mem *.7

#############
# cuTensorNet
#############

stream = cp.cuda.Stream()
handle = cutn.create()

nmode_T = len(modes_T)
nmode_U = len(modes_U)
nmode_V = len(modes_V)

###############################
# Create tensor descriptor
###############################

# strides are optional; if no stride (0) is provided, then cuTensorNet assumes a generalized column-major data layout
strides = 0
desc_tensor_T = cutn.create_tensor_descriptor(handle, nmode_T, extent_T, strides, modes_T, data_type)
desc_tensor_U = cutn.create_tensor_descriptor(handle, nmode_U, extent_U, strides, modes_U, data_type)
desc_tensor_V = cutn.create_tensor_descriptor(handle, nmode_V, extent_V, strides, modes_V, data_type)

##################################
# Setup SVD truncation parameters
##################################

svd_config = cutn.create_tensor_svd_config(handle)
abs_cutoff_dtype = cutn.tensor_svd_config_get_attribute_dtype(cutn.TensorSVDConfigAttribute.ABS_CUTOFF)
abs_cutoff = np.array(1e-2, dtype=abs_cutoff_dtype)

cutn.tensor_svd_config_set_attribute(handle,
    svd_config, cutn.TensorSVDConfigAttribute.ABS_CUTOFF, abs_cutoff.ctypes.data, abs_cutoff.dtype.itemsize)

rel_cutoff_dtype = cutn.tensor_svd_config_get_attribute_dtype(cutn.TensorSVDConfigAttribute.REL_CUTOFF)
rel_cutoff = np.array(4e-2, dtype=rel_cutoff_dtype)

cutn.tensor_svd_config_set_attribute(handle,
    svd_config, cutn.TensorSVDConfigAttribute.REL_CUTOFF, rel_cutoff.ctypes.data, rel_cutoff.dtype.itemsize)

# optional: choose gesvdj algorithm with customized parameters. Default is gesvd.
algorithm_dtype = cutn.tensor_svd_config_get_attribute_dtype(cutn.TensorSVDConfigAttribute.ALGO)
algorithm = np.array(cutn.TensorSVDAlgo.GESVDJ, dtype=algorithm_dtype)
cutn.tensor_svd_config_set_attribute(handle,
    svd_config, cutn.TensorSVDConfigAttribute.ALGO, algorithm.ctypes.data, algorithm.dtype.itemsize)

algo_params_dtype = cutn.tensor_svd_algo_params_get_dtype(cutn.TensorSVDAlgo.GESVDJ)
algo_params = np.zeros(1, dtype=algo_params_dtype)
algo_params['tol'] = 1e-12
algo_params['max_sweeps'] = 80

cutn.tensor_svd_config_set_attribute(handle,
    svd_config, cutn.TensorSVDConfigAttribute.ALGO_PARAMS, algo_params.ctypes.data, algo_params.dtype.itemsize)

print("Set up SVDConfig to use gesvdj algorithm with truncation")

# create SVDInfo to record truncation information
svd_info = cutn.create_tensor_svd_info(handle)

###############################
# Query Workspace Size
###############################
work_desc = cutn.create_workspace_descriptor(handle)

cutn.workspace_compute_svd_sizes(handle, desc_tensor_T, desc_tensor_U, desc_tensor_V, svd_config, work_desc)
required_workspace_size = cutn.workspace_get_memory_size(handle, 
    work_desc, cutn.WorksizePref.MIN, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)
if worksize < required_workspace_size:
    raise MemoryError("Not enough workspace memory is available.")
work = cp.cuda.alloc(required_workspace_size)
cutn.workspace_set_memory(
    handle, work_desc,
    cutn.Memspace.DEVICE,
    cutn.WorkspaceKind.SCRATCH,
    work.ptr, required_workspace_size)

print("Query and allocate required workspace.")

#####
# Run
#####

min_time_cutensornet = 1e100
num_runs = 3  # to get stable perf results
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()

for i in range(num_runs):
    # restore output
    U_d[:] = 0
    S_d[:] = 0
    V_d[:] = 0
    dev.synchronize()

    # restore output tensor descriptors as `cutensornet.tensor_svd` can potentially update the shared extent in desc_tensor_U/V.
    # therefore we here restore desc_tensor_U/V to the original problem
    cutn.destroy_tensor_descriptor(desc_tensor_U)
    cutn.destroy_tensor_descriptor(desc_tensor_V)
    desc_tensor_U = cutn.create_tensor_descriptor(handle, nmode_U, extent_U, strides, modes_U, data_type)
    desc_tensor_V = cutn.create_tensor_descriptor(handle, nmode_V, extent_V, strides, modes_V, data_type)

    e1.record()
    # execution
    cutn.tensor_svd(handle, desc_tensor_T, T_d.data.ptr,
        desc_tensor_U, U_d.data.ptr,
        S_d.data.ptr, 
        desc_tensor_V, V_d.data.ptr,
        svd_config, svd_info, 
        work_desc, stream.ptr)
    
    e2.record()

    # Synchronize and measure timing
    e2.synchronize()
    time = cp.cuda.get_elapsed_time(e1, e2)  # ms
    min_time_cutensornet = min_time_cutensornet if min_time_cutensornet < time else time

full_extent_dtype = cutn.tensor_svd_info_get_attribute_dtype(cutn.TensorSVDInfoAttribute.FULL_EXTENT)
full_extent = np.empty(1, dtype=full_extent_dtype)
cutn.tensor_svd_info_get_attribute(handle, 
    svd_info, cutn.TensorSVDInfoAttribute.FULL_EXTENT, full_extent.ctypes.data, full_extent.itemsize)
full_extent = int(full_extent)

reduced_extent_dtype = cutn.tensor_svd_info_get_attribute_dtype(cutn.TensorSVDInfoAttribute.REDUCED_EXTENT)
reduced_extent = np.empty(1, dtype=reduced_extent_dtype)
cutn.tensor_svd_info_get_attribute(handle, 
    svd_info, cutn.TensorSVDInfoAttribute.REDUCED_EXTENT, reduced_extent.ctypes.data, reduced_extent.itemsize)
reduced_extent = int(reduced_extent)

discarded_weight_dtype = cutn.tensor_svd_info_get_attribute_dtype(cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT)
discarded_weight = np.empty(1, dtype=discarded_weight_dtype)
cutn.tensor_svd_info_get_attribute(handle, 
    svd_info, cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT, discarded_weight.ctypes.data, discarded_weight.itemsize)
discarded_weight = float(discarded_weight)

algo_status_dtype = cutn.tensor_svd_algo_status_get_dtype(cutn.TensorSVDAlgo.GESVDJ)
algo_status = np.empty(1, dtype=algo_status_dtype)
cutn.tensor_svd_info_get_attribute(handle, 
    svd_info, cutn.TensorSVDInfoAttribute.ALGO_STATUS, algo_status.ctypes.data, algo_status.itemsize)

print(f"Execution time: {min_time_cutensornet} ms")
print("SVD truncation info:")
print(f"GESVDJ residual: {algo_status['residual'].item()}, runtime sweeps = {algo_status['sweeps'].item()}")
print(f"For fixed extent truncation of {shared_extent}, an absolute cutoff value of {float(abs_cutoff)}, and a relative cutoff value of {float(rel_cutoff)}, full extent {full_extent} is reduced to {reduced_extent}")
print(f"Discarded weight: {discarded_weight}")

# Recall that when we do value-based truncation through absolute or relative cutoff, 
# the extent found at runtime maybe lower than we  specified in desc_tensor_.
# Therefore we may need to create new containers to hold the new data which takes on fortran layout corresponding to the new extent
extent_U_out, strides_U_out = cutn.get_tensor_details(handle, desc_tensor_U)[2:]
extent_V_out, strides_V_out = cutn.get_tensor_details(handle, desc_tensor_V)[2:]

if extent_U_out[1] != shared_extent:
    # note strides in cutensornet are in the unit of count and strides in cupy/numpy are in the unit of nbytes
    strides_U_out = [i * U_d.itemsize for i in strides_U_out]
    strides_V_out = [i * V_d.itemsize for i in strides_V_out]
    U_d = cp.ndarray(extent_U_out, dtype=np.float32, memptr=U_d.data, strides=strides_U_out)
    S_d = cp.ndarray(extent_U_out[1], dtype=np.float32, memptr=S_d.data, order='F')
    V_d = cp.ndarray(extent_V_out, dtype=np.float32, memptr=V_d.data, strides=strides_V_out)

out = cp.einsum("ixm,x,nxj->ijmn", U_d, S_d, V_d)

print(f"max diff after truncation {abs(out-T_d).max()}")
print("Check cuTensorNet result.")

#######################################################

cutn.destroy_tensor_descriptor(desc_tensor_T)
cutn.destroy_tensor_descriptor(desc_tensor_U)
cutn.destroy_tensor_descriptor(desc_tensor_V)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_tensor_svd_config(svd_config)
cutn.destroy_tensor_svd_info(svd_info)
cutn.destroy(handle)

print("Free resource and exit.")
