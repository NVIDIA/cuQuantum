import numpy as np
import cupy as cp

import cuquantum
from cuquantum import cutensornet as cutn


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

dev = cp.cuda.Device()  # get current device
freeMem, totalMem = dev.mem_info
worksize = int(freeMem * 0.5)
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

plan = cutn.create_contraction_plan(
    handle, descNet, optimizerInfo, worksize)

###################################################################################
# Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
###################################################################################

pref = cutn.create_contraction_autotune_preference(handle)

# may be 0
n_iter_dtype = cutn.contraction_autotune_preference_get_attribute_dtype(
    cutn.ContractionAutotunePreferenceAttribute.MAX_ITERATIONS)
numAutotuningIterations = np.asarray([5], dtype=n_iter_dtype)
cutn.contraction_autotune_preference_set_attribute(
    handle, pref,
    cutn.ContractionAutotunePreferenceAttribute.MAX_ITERATIONS,
    numAutotuningIterations.ctypes.data, numAutotuningIterations.dtype.itemsize)

# modify the plan again to find the best pair-wise contractions
cutn.contraction_autotune(
    handle, plan, rawDataIn_d, D_d.data.ptr,
    work.ptr, worksize, pref, stream.ptr)

cutn.destroy_contraction_autotune_preference(pref)
 
print("Create a contraction plan for cuTENSOR and optionally auto-tune it.")
 
#####
# Run
#####

minTimeCUTENSOR = 1e100
numRuns = 3  # to get stable perf results
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()

for i in range(numRuns):
    # restore output
    D_d.data.copy_from(D.ctypes.data, D.size * D.dtype.itemsize)

    # Contract over all slices.
    # A user may choose to parallelize this loop across multiple devices.
    for sliceId in range(numSlices):
        e1.record()
        cutn.contraction(
            handle, plan, rawDataIn_d, D_d.data.ptr,
            work.ptr, worksize, sliceId, stream.ptr)
        e2.record()

        # Synchronize and measure timing
        e2.synchronize()
        time = cp.cuda.get_elapsed_time(e1, e2) / 1000  # ms -> s
        minTimeCUTENSOR = minTimeCUTENSOR if minTimeCUTENSOR < time else time

print("Contract the network, each slice uses the same contraction plan.")

#######################################################

flops_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT)
flops = np.zeros((1,), dtype=flops_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizerInfo, cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT,
    flops.ctypes.data, flops.dtype.itemsize)
flops = float(flops)

print(f"numSlices: {numSlices}")
print(f"{minTimeCUTENSOR * 1000} ms / slice")
print(f"{flops/1e9/minTimeCUTENSOR} GFLOPS/s")

cutn.destroy_contraction_plan(plan)
cutn.destroy_contraction_optimizer_info(optimizerInfo)
cutn.destroy_contraction_optimizer_config(optimizerConfig)
cutn.destroy_network_descriptor(descNet)
cutn.destroy(handle)

print("Free resource and exit.")
