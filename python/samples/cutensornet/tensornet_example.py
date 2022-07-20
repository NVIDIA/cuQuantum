# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
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

##########################################################
# Computing: D_{m,x,n,y} = A_{m,h,k,n} B_{u,k,h} C_{x,u,y}
##########################################################

print("Include headers and define data types")

data_type = cuquantum.cudaDataType.CUDA_R_32F
compute_type = cuquantum.ComputeType.COMPUTE_32F
numInputs = 3

# Create an array of modes
modesA = [ord(c) for c in ('m','h','k','n')]
modesB = [ord(c) for c in ('u','k','h')]
modesC = [ord(c) for c in ('x','u','y')]
modesD = [ord(c) for c in ('m','x','n','y')]

# Create an array of extents (shapes) for each tensor
extentA = (96, 64, 64, 96)
extentB = (96, 64, 64)
extentC = (64, 96, 64)
extentD = (96, 64, 96, 64)

print("Define network, modes, and extents")

############################
# Allocate & initialize data
############################

A_d = cp.random.random((np.prod(extentA),), dtype=np.float32)
B_d = cp.random.random((np.prod(extentB),), dtype=np.float32)
C_d = cp.random.random((np.prod(extentC),), dtype=np.float32)
D_d = cp.empty((np.prod(extentD),), dtype=np.float32)
rawDataIn_d = (A_d.data.ptr, B_d.data.ptr, C_d.data.ptr)

A = cp.asnumpy(A_d)
B = cp.asnumpy(B_d)
C = cp.asnumpy(C_d)
D = np.empty(D_d.shape, dtype=np.float32)

####################
# Allocate workspace
####################

# this is one way to proceed: query the currently available memory on the
# device, and allocate a big fraction of it...
#freeMem, totalMem = dev.mem_info
#worksize = int(freeMem * 0.9)
# ...but in this case we can set a much tighter upper bound, since we know
# the rough answer already
worksize = 128*1024**2  # = 128 MB, can be smaller
work = cp.cuda.alloc(worksize)

print("Allocate memory for data and workspace, and initialize data.")

#############
# cuTensorNet
#############

stream = cp.cuda.Stream()
handle = cutn.create()

nmodeA = len(modesA)
nmodeB = len(modesB)
nmodeC = len(modesC)
nmodeD = len(modesD)

###############################
# Create Contraction Descriptor
###############################

# These also work, but require a bit more keystrokes
#modesA = np.asarray(modesA, dtype=np.int32)
#modesB = np.asarray(modesB, dtype=np.int32)
#modesC = np.asarray(modesC, dtype=np.int32)
#modesIn = (modesA.ctypes.data, modesB.ctypes.data, modesC.ctypes.data)
#extentA = np.asarray(extentA, dtype=np.int64)
#extentB = np.asarray(extentB, dtype=np.int64)
#extentC = np.asarray(extentC, dtype=np.int64)
#extentsIn = (extentA.ctypes.data, extentB.ctypes.data, extentC.ctypes.data)

modesIn = (modesA, modesB, modesC)
extentsIn = (extentA, extentB, extentC)
numModesIn = (nmodeA, nmodeB, nmodeC)

# strides are optional; if no stride (0) is provided, then cuTensorNet assumes a generalized column-major data layout
stridesIn = (0, 0, 0)

# compute the alignments
# we hard-code them here because CuPy arrays are at least 256B aligned
alignmentsIn = (256, 256, 256)
alignmentOut = 256

# setup tensor network
descNet = cutn.create_network_descriptor(handle,
    numInputs, numModesIn, extentsIn, stridesIn, modesIn, alignmentsIn,  # inputs
    nmodeD, extentD, 0, modesD, alignmentOut,  # output
    data_type, compute_type)

print("Initialize the cuTensorNet library and create a network descriptor.")

##############################################
# Find "optimal" contraction order and slicing
##############################################

optimizerConfig = cutn.create_contraction_optimizer_config(handle)

# Set the value of the partitioner imbalance factor to 30 (if desired)
imbalance_dtype = cutn.contraction_optimizer_config_get_attribute_dtype(
    cutn.ContractionOptimizerConfigAttribute.GRAPH_IMBALANCE_FACTOR)
imbalance_factor = np.asarray((30,), dtype=imbalance_dtype)
cutn.contraction_optimizer_config_set_attribute(
    handle, optimizerConfig, cutn.ContractionOptimizerConfigAttribute.GRAPH_IMBALANCE_FACTOR,
    imbalance_factor.ctypes.data, imbalance_factor.dtype.itemsize)

optimizerInfo = cutn.create_contraction_optimizer_info(handle, descNet)

cutn.contraction_optimize(
    handle, descNet, optimizerConfig, worksize, optimizerInfo)

numSlices_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.NUM_SLICES)
numSlices = np.zeros((1,), dtype=numSlices_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizerInfo, cutn.ContractionOptimizerInfoAttribute.NUM_SLICES,
    numSlices.ctypes.data, numSlices.dtype.itemsize)
numSlices = int(numSlices)

assert numSlices > 0

print("Find an optimized contraction path with cuTensorNet optimizer.")
 
###########################################################
# Initialize all pair-wise contraction plans (for cuTENSOR)
###########################################################

workDesc = cutn.create_workspace_descriptor(handle)
cutn.workspace_compute_sizes(handle, descNet, optimizerInfo, workDesc)
requiredWorkspaceSize = cutn.workspace_get_size(
    handle, workDesc,
    cutn.WorksizePref.MIN,
    cutn.Memspace.DEVICE)
if worksize < requiredWorkspaceSize:
    raise MemoryError("Not enough workspace memory is available.")
cutn.workspace_set(
    handle, workDesc,
    cutn.Memspace.DEVICE,
    work.ptr, worksize)
plan = cutn.create_contraction_plan(
    handle, descNet, optimizerInfo, workDesc)

###################################################################################
# Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
###################################################################################

pref = cutn.create_contraction_autotune_preference(handle)

numAutotuningIterations = 5  # may be 0
n_iter_dtype = cutn.contraction_autotune_preference_get_attribute_dtype(
    cutn.ContractionAutotunePreferenceAttribute.MAX_ITERATIONS)
numAutotuningIterations = np.asarray([numAutotuningIterations], dtype=n_iter_dtype)
cutn.contraction_autotune_preference_set_attribute(
    handle, pref,
    cutn.ContractionAutotunePreferenceAttribute.MAX_ITERATIONS,
    numAutotuningIterations.ctypes.data, numAutotuningIterations.dtype.itemsize)

# modify the plan again to find the best pair-wise contractions
cutn.contraction_autotune(
    handle, plan, rawDataIn_d, D_d.data.ptr,
    workDesc, pref, stream.ptr)

cutn.destroy_contraction_autotune_preference(pref)
 
print("Create a contraction plan for cuTENSOR and optionally auto-tune it.")
 
#####
# Run
#####

minTimeCUTENSOR = 1e100
numRuns = 3  # to get stable perf results
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()
sliceGroup = cutn.create_slice_group_from_id_range(handle, 0, numSlices, 1)

for i in range(numRuns):
    # restore output
    D_d.data.copy_from(D.ctypes.data, D.size * D.dtype.itemsize)
    dev.synchronize()

    # Contract over all slices.
    # A user may choose to parallelize over the slices across multiple devices.
    e1.record()
    cutn.contract_slices(
        handle, plan, rawDataIn_d, D_d.data.ptr, False,
        workDesc, sliceGroup, stream.ptr)
    e2.record()

    # Synchronize and measure timing
    e2.synchronize()
    time = cp.cuda.get_elapsed_time(e1, e2) / 1000  # ms -> s
    minTimeCUTENSOR = minTimeCUTENSOR if minTimeCUTENSOR < time else time


print("Contract the network, each slice uses the same contraction plan.")

# recall that we set strides to null (0), so the data are in F-contiguous layout
A_d = A_d.reshape(extentA, order='F')
B_d = B_d.reshape(extentB, order='F')
C_d = C_d.reshape(extentC, order='F')
D_d = D_d.reshape(extentD, order='F')
out = cp.einsum("mhkn,ukh,xuy->mxny", A_d, B_d, C_d)
if not cp.allclose(out, D_d):
    raise RuntimeError("result is incorrect")
print("Check cuTensorNet result against that of cupy.einsum().")

#######################################################

flops_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT)
flops = np.zeros((1,), dtype=flops_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizerInfo, cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT,
    flops.ctypes.data, flops.dtype.itemsize)
flops = float(flops)

print(f"numSlices: {numSlices}")
print(f"{minTimeCUTENSOR * 1000 / numSlices} ms / slice")
print(f"{flops/1e9/minTimeCUTENSOR} GFLOPS/s")

cutn.destroy_slice_group(sliceGroup)
cutn.destroy_contraction_plan(plan)
cutn.destroy_contraction_optimizer_info(optimizerInfo)
cutn.destroy_contraction_optimizer_config(optimizerConfig)
cutn.destroy_network_descriptor(descNet)
cutn.destroy_workspace_descriptor(workDesc)
cutn.destroy(handle)

print("Free resource and exit.")
