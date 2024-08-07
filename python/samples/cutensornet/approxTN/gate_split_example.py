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

###################################################################################
# Gate Split: A_{i,j,k,l} B_{k,o,p,q} G_{m,n,l,o}-> A'_{i,j,x,m} S_{x} B'_{x,n,p,q} 
###################################################################################

data_type = cuquantum.cudaDataType.CUDA_R_32F
compute_type = cuquantum.ComputeType.COMPUTE_32F

# Create an array of modes

modes_A_in = [ord(c) for c in ('i','j','k','l')] # input
modes_B_in = [ord(c) for c in ('k','o','p','q')]
modes_G_in = [ord(c) for c in ('m','n','l','o')]

modes_A_out = [ord(c) for c in ('i','j','x','m')] # output
modes_B_out = [ord(c) for c in ('x','n','p','q')] 

# Create an array of extent (shapes) for each tensor
extent_A_in = (16, 16, 16, 2)
extent_B_in = (16, 2, 16, 16)
extent_G_in = (2, 2, 2, 2)

shared_extent_out = 16 # truncate shared extent to 16
extent_A_out = (16, 16, shared_extent_out, 2)
extent_B_out = (shared_extent_out, 2, 16, 16)

############################
# Allocate & initialize data
############################
cp.random.seed(1)
A_in_d = cp.random.random(extent_A_in, dtype=np.float32).astype(np.float32, order='F') # we use fortran layout throughout this example
B_in_d = cp.random.random(extent_B_in, dtype=np.float32).astype(np.float32, order='F')
G_in_d = cp.random.random(extent_G_in, dtype=np.float32).astype(np.float32, order='F')

A_out_d = cp.empty(extent_A_out, dtype=np.float32, order='F')
S_out_d = cp.empty(shared_extent_out, dtype=np.float32)
B_out_d = cp.empty(extent_B_out, dtype=np.float32, order='F')

print("Allocate memory for data and initialize data.")

free_mem, total_mem = dev.mem_info
worksize = free_mem *.7

#############
# cuTensorNet
#############

stream = cp.cuda.Stream()
handle = cutn.create()

nmode_A_in = len(modes_A_in)
nmode_B_in = len(modes_B_in)
nmode_G_in = len(modes_G_in)
nmode_A_out = len(modes_A_out)
nmode_B_out = len(modes_B_out)

###############################
# Create tensor descriptors
###############################

# strides are optional; if no stride (0) is provided, then cuTensorNet assumes a generalized column-major data layout
strides = 0
desc_tensor_A_in = cutn.create_tensor_descriptor(handle, nmode_A_in, extent_A_in, strides, modes_A_in, data_type)
desc_tensor_B_in = cutn.create_tensor_descriptor(handle, nmode_B_in, extent_B_in, strides, modes_B_in, data_type)
desc_tensor_G_in = cutn.create_tensor_descriptor(handle, nmode_G_in, extent_G_in, strides, modes_G_in, data_type)

desc_tensor_A_out = cutn.create_tensor_descriptor(handle, nmode_A_out, extent_A_out, strides, modes_A_out, data_type)
desc_tensor_B_out = cutn.create_tensor_descriptor(handle, nmode_B_out, extent_B_out, strides, modes_B_out, data_type)

########################################
# Setup gate split truncation parameters
########################################

svd_config = cutn.create_tensor_svd_config(handle)
absCutoff_dtype = cutn.tensor_svd_config_get_attribute_dtype(cutn.TensorSVDConfigAttribute.ABS_CUTOFF)
absCutoff = np.array(1e-2, dtype=absCutoff_dtype)

cutn.tensor_svd_config_set_attribute(handle,
    svd_config, cutn.TensorSVDConfigAttribute.ABS_CUTOFF, absCutoff.ctypes.data, absCutoff.dtype.itemsize)

relCutoff_dtype = cutn.tensor_svd_config_get_attribute_dtype(cutn.TensorSVDConfigAttribute.REL_CUTOFF)
relCutoff = np.array(1e-2, dtype=relCutoff_dtype)

cutn.tensor_svd_config_set_attribute(handle,
    svd_config, cutn.TensorSVDConfigAttribute.REL_CUTOFF, relCutoff.ctypes.data, relCutoff.dtype.itemsize)

# create SVDInfo to record truncation information
svd_info = cutn.create_tensor_svd_info(handle)

gate_algo = cutn.GateSplitAlgo.REDUCED
print("Setup gate split truncation options.")

###############################
# Query Workspace Size
###############################
work_desc = cutn.create_workspace_descriptor(handle)

cutn.workspace_compute_gate_split_sizes(handle, 
    desc_tensor_A_in, desc_tensor_B_in, desc_tensor_G_in, 
    desc_tensor_A_out, desc_tensor_B_out, 
    gate_algo, svd_config, compute_type, work_desc)
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

###########
# Execution
###########

min_time_cutensornet = 1e100
num_runs = 3  # to get stable perf results
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()

for i in range(num_runs):
    # restore output
    A_out_d[:] = 0
    S_out_d[:] = 0
    B_out_d[:] = 0
    dev.synchronize()
    
    # restore output tensor descriptors as `cutensornet.gate_split` can potentially update the shared extent in desc_tensor_U/V.
    # therefore we here restore desc_tensor_U/V to the original problem
    cutn.destroy_tensor_descriptor(desc_tensor_A_out)
    cutn.destroy_tensor_descriptor(desc_tensor_B_out)
    desc_tensor_A_out = cutn.create_tensor_descriptor(handle, nmode_A_out, extent_A_out, strides, modes_A_out, data_type)
    desc_tensor_B_out = cutn.create_tensor_descriptor(handle, nmode_B_out, extent_B_out, strides, modes_B_out, data_type)

    e1.record()
    # execution
    cutn.gate_split(handle, 
        desc_tensor_A_in, A_in_d.data.ptr,
        desc_tensor_B_in, B_in_d.data.ptr,
        desc_tensor_G_in, G_in_d.data.ptr,
        desc_tensor_A_out, A_out_d.data.ptr,
        S_out_d.data.ptr, 
        desc_tensor_B_out, B_out_d.data.ptr,
        gate_algo, svd_config, compute_type, svd_info, work_desc, stream.ptr)
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

print(f"Execution time: {min_time_cutensornet} ms")
print("SVD truncation info:")
print(f"For fixed extent truncation of {shared_extent_out}, an absolute cutoff value of {float(absCutoff)}, and a relative cutoff value of {float(relCutoff)}, full extent {full_extent} is reduced to {reduced_extent}")
print(f"Discarded weight: {discarded_weight}")

# Recall that when we do value-based truncation through absolute or relative cutoff, 
# the extent found at runtime maybe lower than we  specified in desc_tensor_.
# Therefore we may need to create new containers to hold the new data which takes on fortran layout corresponding to the new extent

if reduced_extent != shared_extent_out:
    extent_A_out_reduced, strides_A_out = cutn.get_tensor_details(handle, desc_tensor_A_out)[2:]
    extent_B_out_reduced, strides_B_out = cutn.get_tensor_details(handle, desc_tensor_B_out)[2:]
    # note strides in cutensornet are in the unit of count and strides in cupy/numpy are in the unit of nbytes
    strides_A_out = [i * A_out_d.itemsize for i in strides_A_out]
    strides_B_out = [i * B_out_d.itemsize for i in strides_B_out]
    A_out_d = cp.ndarray(extent_A_out_reduced, dtype=np.float32, memptr=A_out_d.data, strides=strides_A_out)
    S_out_d = cp.ndarray(reduced_extent, dtype=np.float32, memptr=S_out_d.data, order='F')
    B_out_d = cp.ndarray(extent_B_out_reduced, dtype=np.float32, memptr=B_out_d.data, strides=strides_B_out)

T_d = cp.einsum("ijkl,kopq,mnlo->ijmnpq", A_in_d, B_in_d, G_in_d)
out = cp.einsum("ijxm,x,xnpq->ijmnpq", A_out_d, S_out_d, B_out_d)

print(f"max diff after truncation {abs(out-T_d).max()}")
print("Check cuTensorNet result.")

#######################################################

cutn.destroy_tensor_descriptor(desc_tensor_A_in)
cutn.destroy_tensor_descriptor(desc_tensor_B_in)
cutn.destroy_tensor_descriptor(desc_tensor_G_in)
cutn.destroy_tensor_descriptor(desc_tensor_A_out)
cutn.destroy_tensor_descriptor(desc_tensor_B_out)
cutn.destroy_tensor_svd_config(svd_config)
cutn.destroy_tensor_svd_info(svd_info)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy(handle)

print("Free resource and exit.")
