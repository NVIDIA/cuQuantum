# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np
try:
    import torch
except ImportError:
    torch = None

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

##########################################################################################
# Computing: O_{a,m} = A_{a,b,c,d} B_{b,c,d,e} C_{e,g,h} D_{g,h,i,j} E_{i,j,k,l} F_{k,l,m}
# We will execute the contraction and compute the gradients of input tensors A, B, C
##########################################################################################

print("Include headers and define data types.")

data_type = cuquantum.cudaDataType.CUDA_R_64F
compute_type = cuquantum.ComputeType.COMPUTE_64F
num_inputs = 6
grad_input_ids = np.asarray((0, 1, 2), dtype=np.int32)

# Create vectors of tensor modes
modes_A = [ord(c) for c in ('a','b','c','d')]
modes_B = [ord(c) for c in ('b','c','d','e')]
modes_C = [ord(c) for c in ('e','g','h')]
modes_D = [ord(c) for c in ('g','h','i','j')]
modes_E = [ord(c) for c in ('i','j','k','l')]
modes_F = [ord(c) for c in ('k','l','m')]
modes_O = [ord(c) for c in ('a','m')]
tensor_modes = [modes_A, modes_B, modes_C, modes_D, modes_E, modes_F, modes_O]
tensor_nmodes = [len(modes) for modes in tensor_modes]

# Create an array of extents (shapes) for each tensor
dim = 36
extent_A = (dim,) * len(modes_A)
extent_B = (dim,) * len(modes_B)
extent_C = (dim,) * len(modes_C)
extent_D = (dim,) * len(modes_D)
extent_E = (dim,) * len(modes_E)
extent_F = (dim,) * len(modes_F)
extent_O = (dim,) * len(modes_O)
tensor_extents = [extent_A, extent_B, extent_C, extent_D, extent_E, extent_F, extent_O]

print("Define tensor network, modes, and extents.")

#################
# Initialize data
#################

# Initialize input tensors with random data
cp.random.seed(SEED)
A_d = cp.random.random((np.prod(extent_A),), dtype=np.float64).reshape(extent_A, order='F')
B_d = cp.random.random((np.prod(extent_B),), dtype=np.float64).reshape(extent_B, order='F')
C_d = cp.random.random((np.prod(extent_C),), dtype=np.float64).reshape(extent_C, order='F')
D_d = cp.random.random((np.prod(extent_D),), dtype=np.float64).reshape(extent_D, order='F')
E_d = cp.random.random((np.prod(extent_E),), dtype=np.float64).reshape(extent_E, order='F')
F_d = cp.random.random((np.prod(extent_F),), dtype=np.float64).reshape(extent_F, order='F')
O_d = cp.zeros((np.prod(extent_O),), dtype=np.float64).reshape(extent_O, order='F')
tensor_data_d = [A_d, B_d, C_d, D_d, E_d, F_d, O_d]

# Allocate GPU memory for adjoint tensor (same size as output tensor)
adjoint_d = cp.ones((np.prod(extent_O),), dtype=np.float64).reshape(extent_O, order='F')

# Allocate GPU memory for gradients (only for tensors that need gradients)
# Note that gradient buffers need to be zero initialized.
gradients_d = [cp.zeros_like(A_d).reshape(extent_A, order='F'),
                cp.zeros_like(B_d).reshape(extent_B, order='F'),
                cp.zeros_like(C_d).reshape(extent_C, order='F'),
                None,
                None,
                None]
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
# Attach the input tensors to the network
for t in range(num_inputs):
    # Create qualifiers for this tensor
    qualifiers = np.zeros(1, dtype=cutn.tensor_qualifiers_dtype)
    qualifiers['requires_gradient'] = t in grad_input_ids
    
    tensor_id = cutn.network_append_tensor(handle,
                                          net,
                                          tensor_nmodes[t],
                                          tensor_extents[t],
                                          tensor_modes[t],
                                          qualifiers.ctypes.data,
                                          data_type)
    tensor_ids.append(tensor_id)

# Set output tensor of the network
cutn.network_set_output_tensor(handle,
                               net,
                               tensor_nmodes[num_inputs],
                               tensor_modes[num_inputs],
                               data_type)

# Set the network compute type
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

num_slices = 1

print("Set predetermined contraction path into cuTensorNet optimizer")

##############################################################
# Create workspace descriptor, allocate workspace, and set it.
##############################################################

work_desc = cutn.create_workspace_descriptor(handle)

# Set SCRATCH workspace, which will be used during each network contraction operation
required_workspace_size_scratch = 0
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

# Set CACHE workspace, which will be used across network contraction operations
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

#######################################################
# Prepare the pairwise contraction plan (for cuTENSOR).
#######################################################

cutn.network_set_optimizer_info(handle, net, optimizer_info)

# Set tensor's data buffers and strides
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

cutn.network_prepare_contraction(handle, net, work_desc)

###################################################################################
# Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
#           for each pairwise tensor contraction.
###################################################################################

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

# Modify the plan again to find the best pair-wise contractions
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

# Restore the output tensor on GPU
tensor_data_d[num_inputs][:] = 0

# Contract all slices of the tensor network
accumulate_output = 0  # output tensor data will be overwritten
cutn.network_contract(handle,
                     net,
                     accumulate_output,
                     work_desc,
                     slice_group,  # alternatively, 0 can also be used to contract over all slices
                     stream.ptr)

print("Contracted the tensor network")

##################################################################
# Prepare the tensor network gradient computation and auto-tune it
##################################################################

cutn.network_set_adjoint_tensor_memory(handle, net, adjoint_d.data.ptr, 0)

# Set gradient tensor memory for tensors that require gradients
for i in grad_input_ids:
    if gradients_d[i] is not None:
        cutn.network_set_gradient_tensor_memory(handle,
                                               net,
                                               i,
                                               gradients_d[i].data.ptr,
                                               0)

cutn.network_prepare_gradients_backward(handle, net, work_desc)

#################################################
# Execute the tensor network gradient computation
#################################################

# compute time for compute gradients
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()

e1.record()

cutn.network_compute_gradients_backward(handle,
                                        net,
                                        accumulate_output,
                                        work_desc,
                                        slice_group,  # alternatively, 0 can also be used to contract over all slices
                                        stream.ptr)

e2.record()
e2.synchronize()
time = cp.cuda.get_elapsed_time(e1, e2) / 1000  # ms -> s

print("Contracted the tensor network and computed gradients")

# Verification
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
    output_grads = torch.as_tensor(func(adjoint_d), device=dev)

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
    try:
        cp.testing.assert_allclose(cp.asarray(out.detach()), O_d, atol=ATOL, rtol=RTOL)
    except AssertionError as e:
        raise RuntimeError("result is incorrect") from e

    # If using PyTorch CPU tensors, these move data back to GPU for comparison;
    # otherwise, PyTorch GPU tensors are zero-copied as CuPy arrays.
    try:
        cp.testing.assert_allclose(cp.asarray(A.grad), gradients_d[0], atol=ATOL, rtol=RTOL)
    except AssertionError as e:
        raise RuntimeError("result is incorrect") from e
    try:
        cp.testing.assert_allclose(cp.asarray(B.grad), gradients_d[1], atol=ATOL, rtol=RTOL)
    except AssertionError as e:
        raise RuntimeError("result is incorrect") from e
    # Note: D.grad, E.grad, and F.grad do not exist

    print("Check cuTensorNet gradient results against those from "
          f"PyTorch ({'GPU' if torch_cuda else 'GPU'}).")

print(f"Tensor network contraction and back-propagation time (ms): = {time * 1000:.3f}")


################
# Free resources
################

# Free cuTensorNet resources
cutn.destroy_slice_group(slice_group)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_contraction_optimizer_info(optimizer_info)
cutn.destroy_network(net)
cutn.destroy(handle)

# Free GPU memory resources
del work_scratch
del work_cache

print("Freed resources and exited")
