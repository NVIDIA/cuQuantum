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

###############################################
# Tensor QR: T_{i,j,m,n} -> Q_{i,x,m} R_{n,x,j}  
###############################################

data_type = cuquantum.cudaDataType.CUDA_R_32F

# Create an array of modes

modes_T = [ord(c) for c in ('i','j','m','n')] # input
modes_Q = [ord(c) for c in ('i','x','m')] # QR output
modes_R = [ord(c) for c in ('n','x','j')]

# Create an array of extent (shapes) for each tensor
extent_T = (16, 16, 16, 16)
extent_Q = (16, 256, 16)
extent_R = (16, 256, 16)

############################
# Allocate & initialize data
############################

T_d = cp.random.random(extent_T, dtype=np.float32).astype(np.float32, order='F') # we use fortran layout throughout this example
Q_d = cp.empty(extent_Q, dtype=np.float32, order='F')
R_d = cp.empty(extent_R, dtype=np.float32, order='F')

print("Allocate memory for data and initialize data.")

free_mem, total_mem = dev.mem_info
worksize = free_mem *.7

#############
# cuTensorNet
#############
stream = cp.cuda.Stream()
handle = cutn.create()

nmode_T = len(modes_T)
nmode_Q = len(modes_Q)
nmode_R = len(modes_R)

###############################
# Create tensor descriptors
###############################

# strides are optional; if no stride (0) is provided, then cuTensorNet assumes a generalized column-major data layout
strides = 0
desc_tensor_T = cutn.create_tensor_descriptor(handle, nmode_T, extent_T, strides, modes_T, data_type)
desc_tensor_Q = cutn.create_tensor_descriptor(handle, nmode_Q, extent_Q, strides, modes_Q, data_type)
desc_tensor_R = cutn.create_tensor_descriptor(handle, nmode_R, extent_R, strides, modes_R, data_type)

#######################################
# Query and allocate required workspace
#######################################
work_desc = cutn.create_workspace_descriptor(handle)

cutn.workspace_compute_qr_sizes(handle, desc_tensor_T, desc_tensor_Q, desc_tensor_R, work_desc)
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
    Q_d[:] = 0
    R_d[:] = 0
    dev.synchronize()

    e1.record()
    # execution
    cutn.tensor_qr(handle, desc_tensor_T, T_d.data.ptr,
        desc_tensor_Q, Q_d.data.ptr,
        desc_tensor_R, R_d.data.ptr,
        work_desc, stream.ptr)
    e2.record()

    # Synchronize and measure timing
    e2.synchronize()
    time = cp.cuda.get_elapsed_time(e1, e2)  # ms
    min_time_cutensornet = min_time_cutensornet if min_time_cutensornet < time else time

print(f"Execution time: {min_time_cutensornet} ms")

out = cp.einsum("ixm,nxj->ijmn", Q_d, R_d)

rtol = atol = 1e-5
if not cp.allclose(out, T_d, rtol=rtol, atol=atol):
    raise RuntimeError(f"result is incorrect, max diff {abs(out-T_d).max()}")
print("Check cuTensorNet result.")

################
# Free resources
################

cutn.destroy_tensor_descriptor(desc_tensor_T)
cutn.destroy_tensor_descriptor(desc_tensor_Q)
cutn.destroy_tensor_descriptor(desc_tensor_R)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy(handle)

print("Free resource and exit.")
