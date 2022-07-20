# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np
from mpi4py import MPI

import cuquantum
from cuquantum import cutensornet as cutn


root = 0
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
if rank == root:
    print("*** Printing is done only from the root process to prevent jumbled messages ***")
    print(f"The number of processes is {size}")

# Get cuTensorNet version and device properties.
numDevices = cp.cuda.runtime.getDeviceCount()
deviceId = rank % numDevices  # We assume that the processes are mapped to nodes in contiguous chunks.
dev = cp.cuda.Device(deviceId)
dev.use()
props = cp.cuda.runtime.getDeviceProperties(dev.id)
if rank == root:
    print("cuTensorNet-vers:", cutn.get_version())
    print("===== root process device info ======")
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

if rank == root:
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

if rank == root:
    print("Define network, modes, and extents")

############################
# Allocate & initialize data
############################

if rank == root:
    A = np.random.random((np.prod(extentA),)).astype(np.float32)
    B = np.random.random((np.prod(extentB),)).astype(np.float32)
    C = np.random.random((np.prod(extentC),)).astype(np.float32)
else:
    A = np.empty((np.prod(extentA),), dtype=np.float32)
    B = np.empty((np.prod(extentB),), dtype=np.float32)
    C = np.empty((np.prod(extentC),), dtype=np.float32)
D = np.empty(extentD, dtype=np.float32)

# Broadcast data to all ranks.
comm.Bcast(A, root)
comm.Bcast(B, root)
comm.Bcast(C, root)

# Copy data onto the device on all ranks.
A_d = cp.asarray(A)
B_d = cp.asarray(B)
C_d = cp.asarray(C)
D_d = cp.empty((np.prod(extentD),), dtype=np.float32)
rawDataIn_d = (A_d.data.ptr, B_d.data.ptr, C_d.data.ptr)

if rank == root:
    print("Allocate memory for data, calculate workspace limit, and initialize data.")

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

if rank == root:
    print("Initialize the cuTensorNet library and create a network descriptor.")

#####################################################
# Choose workspace limit based on available resources
#####################################################

freeMem, totalMem = dev.mem_info
totalMem = comm.allreduce(totalMem, MPI.MIN)
workspaceLimit = int(totalMem * 0.9)

##############################################
# Find "optimal" contraction order and slicing
##############################################

optimizerConfig = cutn.create_contraction_optimizer_config(handle)
optimizerInfo = cutn.create_contraction_optimizer_info(handle, descNet)

# Compute the path on all ranks so that we can choose the path with the lowest cost. Note that since this is a tiny
# example with 3 operands, all processes will compute the same globally optimal path. This is not the case for large
# tensor networks. For large networks, hyperoptimization is also beneficial and can be enabled by setting the
# optimizer config attribute cutn.ContractionOptimizerConfigAttribute.HYPER_NUM_SAMPLES.

# Force slicing
min_slices_dtype = cutn.contraction_optimizer_config_get_attribute_dtype(
    cutn.ContractionOptimizerConfigAttribute.SLICER_MIN_SLICES)
min_slices_factor = np.asarray((size,), dtype=min_slices_dtype)
cutn.contraction_optimizer_config_set_attribute(
    handle, optimizerConfig, cutn.ContractionOptimizerConfigAttribute.SLICER_MIN_SLICES,
    min_slices_factor.ctypes.data, min_slices_factor.dtype.itemsize)

cutn.contraction_optimize(
    handle, descNet, optimizerConfig, workspaceLimit, optimizerInfo)

flops_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT)
flops = np.zeros((1,), dtype=flops_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizerInfo, cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT,
    flops.ctypes.data, flops.dtype.itemsize)
flops = float(flops)

# Choose the path with the lowest cost.
flops, sender = comm.allreduce(sendobj=(flops, rank), op=MPI.MINLOC)
if rank == root:
    print(f"Process {sender} has the path with the lowest FLOP count {flops}.")

# Get buffer size for optimizerInfo and broadcast it.
if rank == sender:
    bufSize = cutn.contraction_optimizer_info_get_packed_size(handle, optimizerInfo)
else:
    bufSize = 0  # placeholder
bufSize = comm.bcast(bufSize, sender)

# Allocate buffer.
buf = np.empty((bufSize,), dtype=np.int8)

# Pack optimizerInfo on sender and broadcast it.
if rank == sender:
    cutn.contraction_optimizer_info_pack_data(handle, optimizerInfo, buf, bufSize)
comm.Bcast(buf, sender)

# Unpack optimizerInfo from buffer.
if rank != sender:
    cutn.update_contraction_optimizer_info_from_packed_data(
        handle, buf, bufSize, optimizerInfo)

numSlices_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.NUM_SLICES)
numSlices = np.zeros((1,), dtype=numSlices_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizerInfo, cutn.ContractionOptimizerInfoAttribute.NUM_SLICES,
    numSlices.ctypes.data, numSlices.dtype.itemsize)
numSlices = int(numSlices)

assert numSlices > 0

# Calculate each process's share of the slices.
procChunk = numSlices / size
extra = numSlices % size
procSliceBegin = rank * procChunk + min(rank, extra)
procSliceEnd = numSlices if rank == size - 1 else (rank + 1) * procChunk + min(rank + 1, extra)

if rank == root:
    print("Find an optimized contraction path with cuTensorNet optimizer.")
 
#############################################################
# Create workspace descriptor, allocate workspace, and set it
#############################################################

workDesc = cutn.create_workspace_descriptor(handle)
cutn.workspace_compute_sizes(handle, descNet, optimizerInfo, workDesc)
requiredWorkspaceSize = cutn.workspace_get_size(
    handle, workDesc,
    cutn.WorksizePref.MIN,
    cutn.Memspace.DEVICE)
work = cp.cuda.alloc(requiredWorkspaceSize)
cutn.workspace_set(
    handle, workDesc,
    cutn.Memspace.DEVICE,
    work.ptr, requiredWorkspaceSize)

if rank == root:
    print("Allocate workspace.")

###########################################################
# Initialize all pair-wise contraction plans (for cuTENSOR)
###########################################################

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

if rank == root:
    print("Create a contraction plan for cuTENSOR and optionally auto-tune it.")
 
#####
# Run
#####

minTimeCUTENSOR = 1e100
numRuns = 3  # to get stable perf results
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()

# Create a cutensornetSliceGroup_t object from a range of slice IDs.
sliceGroup = cutn.create_slice_group_from_id_range(handle, procSliceBegin, procSliceEnd, 1)

for i in range(numRuns):
    dev.synchronize()

    # Contract over the range of slices this process is responsible for.

    # Don't accumulate into output since we use a one-process-per-gpu model.
    accumulateOutput = False

    e1.record()
    cutn.contract_slices(
        handle, plan, rawDataIn_d, D_d.data.ptr, accumulateOutput,
        workDesc, sliceGroup, stream.ptr)
    e2.record()

    # Synchronize and measure timing
    e2.synchronize()
    time = cp.cuda.get_elapsed_time(e1, e2) / 1000  # ms -> s
    minTimeCUTENSOR = minTimeCUTENSOR if minTimeCUTENSOR < time else time

if rank == root:
    print("Contract the network, all slices within the same rank use the same contraction plan.")
    print(f"numSlices: {numSlices}")
    numSlicesProc = procSliceEnd - procSliceBegin
    print(f"numSlices on root process: {numSlicesProc}")
    if numSlicesProc > 0:
        print(f"{minTimeCUTENSOR * 1000 / numSlicesProc} ms / slice")

cutn.destroy_slice_group(sliceGroup)
D[...] = cp.asnumpy(D_d).reshape(extentD, order='F')
# Reduce on root process.
if rank == root:
    comm.Reduce(MPI.IN_PLACE, D, root=root)
else:
    comm.Reduce(D, D, root=root)

# Compute the reference result.
if rank == root:
    # recall that we set strides to null (0), so the data are in F-contiguous layout
    A_d = A_d.reshape(extentA, order='F')
    B_d = B_d.reshape(extentB, order='F')
    C_d = C_d.reshape(extentC, order='F')
    D_d = D_d.reshape(extentD, order='F')
    out = cp.einsum("mhkn,ukh,xuy->mxny", A_d, B_d, C_d)
    if not cp.allclose(out, D):
        raise RuntimeError("result is incorrect")
    print("Check cuTensorNet result against that of cupy.einsum().")

#######################################################

cutn.destroy_contraction_plan(plan)
cutn.destroy_contraction_optimizer_info(optimizerInfo)
cutn.destroy_contraction_optimizer_config(optimizerConfig)
cutn.destroy_network_descriptor(descNet)
cutn.destroy_workspace_descriptor(workDesc)
cutn.destroy(handle)

if rank == root:
    print("Free resource and exit.")
