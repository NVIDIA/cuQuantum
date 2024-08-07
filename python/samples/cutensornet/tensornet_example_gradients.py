# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np
try:
    import torch
except ImportError:
    torch = None

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

##########################################################################################
# Computing: O_{a,m} = A_{a,b,c,d} B_{b,c,d,e} C_{e,g,h} D_{g,h,i,j} E_{i,j,k,l} F_{k,l,m}
# We will execute the contraction and compute the gradients of input tensors A, B, C
##########################################################################################

print("Include headers and define data types.")

data_type = cuquantum.cudaDataType.CUDA_R_32F
compute_type = cuquantum.ComputeType.COMPUTE_32F
num_inputs = 6
grad_input_ids = np.asarray((0, 1, 2), dtype=np.int32)

# Create an array of modes
modes_A = [ord(c) for c in ('a','b','c','d')]
modes_B = [ord(c) for c in ('b','c','d','e')]
modes_C = [ord(c) for c in ('e','g','h')]
modes_D = [ord(c) for c in ('g','h','i','j')]
modes_E = [ord(c) for c in ('i','j','k','l')]
modes_F = [ord(c) for c in ('k','l','m')]
modes_O = [ord(c) for c in ('a','m')]

# Create an array of extents (shapes) for each tensor
dim = 36
extent_A = (dim,) * len(modes_A)
extent_B = (dim,) * len(modes_B)
extent_C = (dim,) * len(modes_C)
extent_D = (dim,) * len(modes_D)
extent_E = (dim,) * len(modes_E)
extent_F = (dim,) * len(modes_F)
extent_O = (dim,) * len(modes_O)

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

# allocate buffers for holding the gradients w.r.t. the first 3 input tensors
grads_d = [cp.empty_like(A_d),
           cp.empty_like(B_d),
           cp.empty_like(C_d),
           None,
           None,
           None]
grads_d_ptr = [grad.data.ptr if grad is not None else 0 for grad in grads_d]

# output gradients (w.r.t itself, so it's all one)
output_grads_d = cp.ones(extent_O, dtype=np.float32, order='F')

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

# Set up tensor network
desc_net = cutn.create_network_descriptor(handle,
    num_inputs, num_modes_in, extents_in, strides_in, modes_in, 0,  # inputs
    nmode_O, extent_O, 0, modes_O,  # output
    data_type, compute_type)

# In this sample we use the new network attributes interface to mark certain
# input tensors as constant, but we can also use the tensor qualifiers as shown
# in other samples (ex: tensornet_example_reuse.py)
net_attr_dtype = cutn.network_get_attribute_dtype(cutn.NetworkAttribute.INPUT_TENSORS_REQUIRE_GRAD)
tensor_ids = np.zeros(1, dtype=net_attr_dtype)
tensor_ids['num_tensors'] = grad_input_ids.size
tensor_ids['data'] = grad_input_ids.ctypes.data
cutn.network_set_attribute(
    handle, desc_net, cutn.NetworkAttribute.INPUT_TENSORS_REQUIRE_GRAD,
    tensor_ids.ctypes.data, tensor_ids.dtype.itemsize)

print("Initialize the cuTensorNet library and create a network descriptor.")

#####################################################
# Choose workspace limit based on available resources
#####################################################

free_mem, total_mem = dev.mem_info
workspace_limit = int(free_mem * 0.9)

#######################
# Set contraction order
#######################

# create contraction optimizer info
optimizer_info = cutn.create_contraction_optimizer_info(handle, desc_net)

# set a predetermined contraction path
path_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(cutn.ContractionOptimizerInfoAttribute.PATH)
path = np.asarray([(0, 1), (0, 4), (0, 3), (0, 2), (0, 1)], dtype=np.int32)
path_obj = np.zeros((1,), dtype=path_dtype)
path_obj["num_contractions"] = num_inputs - 1
path_obj["data"] = path.ctypes.data

# provide user-specified contract path
cutn.contraction_optimizer_info_set_attribute(
    handle, optimizer_info, cutn.ContractionOptimizerInfoAttribute.PATH, 
    path_obj.ctypes.data, path_obj.dtype.itemsize)

num_slices = 1

print("Set predetermined contraction path into cuTensorNet optimizer.")

#############################################################
# Create workspace descriptor, allocate workspace, and set it
#############################################################

work_desc = cutn.create_workspace_descriptor(handle)

# set SCRATCH workspace, which will be used during each network contraction operation, not needed afterwords
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

# set CACHE workspace, which will be used across network contraction operations
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

print("Allocated and set up the GPU workspace")

###########################################################
# Initialize the pair-wise contraction plans (for cuTENSOR)
###########################################################

plan = cutn.create_contraction_plan(handle, desc_net, optimizer_info, work_desc)

###################################################################################
# Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
###################################################################################

pref = cutn.create_contraction_autotune_preference(handle)

num_autotuning_iterations = 5  # may be 0
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

# create a cutensornetSliceGroup_t object from a range of slice IDs
slice_group = cutn.create_slice_group_from_id_range(handle, 0, num_slices, 1)

min_time_cutn = 1e100
num_runs = 3  # to get stable perf results
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()

for i in range(num_runs):
    # Contract over all slices.
    e1.record(stream)
    cutn.contract_slices(
        handle, plan, raw_data_in_d,
        O_d.data.ptr,
        False, work_desc, slice_group, stream.ptr)
    cutn.compute_gradients_backward(
        handle, plan, raw_data_in_d,
        output_grads_d.data.ptr,
        grads_d_ptr,
        False, work_desc, stream.ptr)
    cutn.workspace_purge_cache(handle, work_desc, cutn.Memspace.DEVICE)
    e2.record(stream)

    # Synchronize and measure timing
    e2.synchronize()
    time = cp.cuda.get_elapsed_time(e1, e2) / 1000  # ms -> s
    min_time_cutn = min_time_cutn if min_time_cutn < time else time

print("Contract the network and compute gradients.")

# free up the workspace
del work_scratch
del work_cache

# Recall that we set strides to null (0), so the data are in F-contiguous layout,
# including the gradients (which follow the layout of the input tensors)
A_d = A_d.reshape(extent_A, order='F')
B_d = B_d.reshape(extent_B, order='F')
C_d = C_d.reshape(extent_C, order='F')
D_d = D_d.reshape(extent_D, order='F')
E_d = E_d.reshape(extent_E, order='F')
F_d = F_d.reshape(extent_F, order='F')
O_d = O_d.reshape(extent_O, order='F')
grads_d[0] = grads_d[0].reshape(extent_A, order='F')
grads_d[1] = grads_d[1].reshape(extent_B, order='F')
grads_d[2] = grads_d[2].reshape(extent_C, order='F')

# Compute the contraction reference using cupy.einsum with the same path
path = ['einsum_path'] + path.tolist()
out = cp.einsum("abcd,bcde,egh,ghij,ijkl,klm->am", A_d, B_d, C_d, D_d, E_d, F_d, optimize=path)
if not cp.allclose(out, O_d):
    raise RuntimeError("result is incorrect")
print("Check cuTensorNet contraction result against that of cupy.einsum().")

# Compute the gradient reference using PyTorch
if torch:
    if not torch.cuda.is_available():
        # copy data back to CPU
        dev = "cpu"
        func = cp.asnumpy
        torch_cuda = False
    else:
        # zero-copy from CuPy to PyTorch!
        dev = "cuda"
        func = (lambda x: x)  # no op
        torch_cuda = True

    A = torch.as_tensor(func(A_d), device=dev)
    B = torch.as_tensor(func(B_d), device=dev)
    C = torch.as_tensor(func(C_d), device=dev)
    D = torch.as_tensor(func(D_d), device=dev)
    E = torch.as_tensor(func(E_d), device=dev)
    F = torch.as_tensor(func(F_d), device=dev)
    output_grads = torch.as_tensor(func(output_grads_d), device=dev)

    # do not need gradient for the last 3 tensors
    A.requires_grad_(True)
    B.requires_grad_(True)
    C.requires_grad_(True)
    D.requires_grad_(False)
    E.requires_grad_(False)
    F.requires_grad_(False)

    # We can use either torch.einsum or opt_einsum.contract to establish the
    # computational graph of an einsum op over the PyTorch tensors. Note that
    # torch.einsum does not support passing custom contraction paths.
    out = torch.einsum("abcd,bcde,egh,ghij,ijkl,klm->am", A, B, C, D, E, F)
    out.backward(output_grads)  # backprop to populate the inputs' .grad attributes
    if not cp.allclose(cp.asarray(out.detach()), O_d):
        raise RuntimeError("result is incorrect")

    # If using PyTorch CPU tensors, these move data back to GPU for comparison;
    # otherwise, PyTorch GPU tensors are zero-copied as CuPy arrays.
    assert cp.allclose(cp.asarray(A.grad), grads_d[0])
    assert cp.allclose(cp.asarray(B.grad), grads_d[1])
    assert cp.allclose(cp.asarray(C.grad), grads_d[2])
    # Note: D.grad, E.grad, and F.grad do not exist

    print("Check cuTensorNet gradient results against those from "
          f"PyTorch ({'GPU' if torch_cuda else 'GPU'}).")

#######################################################

print(f"Tensor network contraction and back-propagation time (ms): = {min_time_cutn * 1000}")

cutn.destroy_slice_group(slice_group)
cutn.destroy_contraction_plan(plan)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_contraction_optimizer_info(optimizer_info)
cutn.destroy_network_descriptor(desc_net)
cutn.destroy(handle)

print("Free resource and exit.")
