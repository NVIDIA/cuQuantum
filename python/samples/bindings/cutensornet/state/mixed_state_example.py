# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np

import cuquantum
from cuquantum.bindings import cutensornet as cutn


print("cuTensorNet-vers:", cutn.get_version())
dev = cp.cuda.Device()  # get current device
props = cp.cuda.runtime.getDeviceProperties(dev.id)
print("===== device info ======")
print("GPU-name:", props["name"].decode())
print("GPU-major:", props["major"])
print("GPU-minor:", props["minor"])
print("========================")

##############################################################################
# Mixed-state (density-matrix) simulation of a noisy quantum circuit state.
#
# A mixed state is created with StatePurity.MIXED; a general (non-unitary)
# quantum channel is applied with state_apply_general_channel; and both the full
# reduced density matrix (create_marginal) and its diagonal / marginal
# probabilities (create_marginal_diagonal) are computed.
##############################################################################

# Quantum state configuration
num_qubits = 6
dim = 2
qubits_dims = (dim,) * num_qubits
marginal_modes = (0, 1)  # open qubits defining the marginal
num_marginal_modes = len(marginal_modes)
print(f"Mixed-state simulation of a {num_qubits}-qubit noisy circuit")

#############
# cuTensorNet
#############

handle = cutn.create()
stream = cp.cuda.Stream()
data_type = cuquantum.cudaDataType.CUDA_C_64F

# Quantum gate tensors on device (column-major / 'F' order)
gate_h = 2**-0.5 * cp.asarray([[1, 1], [1, -1]], dtype="complex128", order="F")
gate_cx = cp.asarray([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], dtype="complex128").reshape(2, 2, 2, 2, order="F")
gate_strides = 0

# Amplitude-damping channel Kraus operators (K0^dag K0 + K1^dag K1 = I)
gamma = 0.2
kraus0 = cp.asarray([[1, 0], [0, np.sqrt(1 - gamma)]], dtype="complex128", order="F")
kraus1 = cp.asarray([[0, np.sqrt(gamma)], [0, 0]], dtype="complex128", order="F")

# Allocate device memory for the full reduced density matrix and its diagonal
rdm = cp.empty((dim,) * (2 * num_marginal_modes), dtype="complex128")
rdm_strides = [s // rdm.itemsize for s in rdm.strides]
rdm_diag = cp.empty((dim,) * num_marginal_modes, dtype="complex128")
rdm_diag_strides = [s // rdm_diag.itemsize for s in rdm_diag.strides]

free_mem = dev.mem_info[0]
scratch_size = free_mem // 2
scratch_space = cp.cuda.alloc(scratch_size)
print(f"Allocated {scratch_size} bytes of scratch memory on GPU")

# Create the initial MIXED (density-matrix) quantum state
quantum_state = cutn.create_state(handle, cutn.StatePurity.MIXED, num_qubits, qubits_dims, data_type)
print("Created the initial mixed quantum state")

# Construct the GHZ circuit
cutn.state_apply_tensor_operator(handle, quantum_state, 1, (0,), gate_h.data.ptr, gate_strides, 1, 0, 1)
for i in range(1, num_qubits):
    cutn.state_apply_tensor_operator(handle, quantum_state, 2, (i - 1, i), gate_cx.data.ptr, gate_strides, 1, 0, 1)
print("Applied quantum gates (GHZ circuit)")

# Apply a general (non-unitary) channel; only valid for a mixed state
cutn.state_apply_general_channel(
    handle, quantum_state, 1, (0,), 2, (kraus0.data.ptr, kraus1.data.ptr), 0)
print("Applied an amplitude-damping channel to qubit 0")


def prepare_and_compute(marginal_handle, out_array, label):
    """Configure, prepare, attach workspace, and compute a marginal handle."""
    num_hyper_samples_dtype = cutn.marginal_get_attribute_dtype(cutn.MarginalAttribute.CONFIG_NUM_HYPER_SAMPLES)
    num_hyper_samples = np.asarray(8, dtype=num_hyper_samples_dtype)
    cutn.marginal_configure(handle, marginal_handle,
        cutn.MarginalAttribute.CONFIG_NUM_HYPER_SAMPLES,
        num_hyper_samples.ctypes.data, num_hyper_samples.dtype.itemsize)

    work_desc = cutn.create_workspace_descriptor(handle)
    cutn.marginal_prepare(handle, marginal_handle, scratch_size, work_desc, stream.ptr)
    workspace_size_d = cutn.workspace_get_memory_size(handle, work_desc,
        cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)
    if workspace_size_d > scratch_size:
        raise RuntimeError(f"Insufficient workspace size on Device for {label}")
    cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH,
        scratch_space.ptr, workspace_size_d)

    # Query the host scratch workspace size and attach a host scratch buffer if one is required
    workspace_size_h = cutn.workspace_get_memory_size(handle,
        work_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.HOST, cutn.WorkspaceKind.SCRATCH)
    print(f"Scratch Host workspace size (bytes) = {workspace_size_h}")
    if workspace_size_h > 0:
        scratch_space_h = np.empty(workspace_size_h, dtype=np.int8)  # must stay alive until the computation completes
        cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.HOST, cutn.WorkspaceKind.SCRATCH, scratch_space_h.ctypes.data, workspace_size_h)
    cutn.marginal_compute(handle, marginal_handle, 0, work_desc, out_array.data.ptr, stream.ptr)
    stream.synchronize()
    cutn.destroy_workspace_descriptor(work_desc)


# Full reduced density matrix
marginal = cutn.create_marginal(handle, quantum_state, num_marginal_modes, marginal_modes, 0, 0, rdm_strides)
prepare_and_compute(marginal, rdm, "reduced density matrix")
rdm_mat = rdm.reshape(dim**num_marginal_modes, dim**num_marginal_modes)
print(f"\nReduced density matrix on qubits {marginal_modes}:")
print(rdm_mat)
print(f"Tr(RDM) = {cp.trace(rdm_mat).real.item():.10f} (should be ~1.0)")

# Diagonal of the reduced density matrix (marginal probabilities)
marginal_diag = cutn.create_marginal_diagonal(handle, quantum_state, num_marginal_modes, marginal_modes, 0, 0, rdm_diag_strides)
prepare_and_compute(marginal_diag, rdm_diag, "marginal diagonal")
probs = rdm_diag.reshape(dim**num_marginal_modes).real
print(f"\nMarginal probability distribution over qubits {marginal_modes}:")
print(probs)
print(f"Sum of probabilities = {probs.sum().item():.10f} (should be ~1.0)")

cutn.destroy_marginal(marginal_diag)
cutn.destroy_marginal(marginal)
cutn.destroy_state(quantum_state)
cutn.destroy(handle)
del scratch_space
print("\nFree resource and exit.")
